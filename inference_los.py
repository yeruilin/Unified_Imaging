"""
Phase 1 Inference: Encoder Latent Decoding (LOS Only)

Supports dynamic spatial resolutions (multiples of 128) and D=512.
Patch size: 128x128
"""

import argparse
import logging
import os
import sys
import time
from glob import glob
import numpy as np
import scipy.io as scio

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import LOS components
from encoder.los_encoder import LOSEncoder

# Import Flux VAE
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
from library.flux_models import AutoEncoder
from library.utils import load_safetensors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ae(ckpt_path: str, dtype: torch.dtype, device: str = "cpu"):
    from library.flux_models import configs
    ae_params = configs["dev"].ae_params
    ae = AutoEncoder(ae_params).to(dtype)

    logger.info(f"Loading VAE from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=device, dtype=dtype)
    ae.load_state_dict(sd, strict=False)
    return ae


def inference(args):
    # 动态创建输出文件夹
    out_dir = os.path.join(args.out_dir, "LOS_inference_outputs")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Decoded images will be saved to: {out_dir}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32 if args.mixed_precision == "fp32" else (torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16)
    
    # 1. Load VAE
    logger.info("Loading Flux VAE...")
    ae = load_ae(args.ae, weight_dtype, "cpu")
    ae.requires_grad_(False).eval().to(device, dtype=weight_dtype)
    
    # 2. Initialize & Load Encoder
    logger.info("Initializing LOS Encoder...")
    encoder = LOSEncoder(in_channels=1, out_dim=16)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    encoder.requires_grad_(False).eval().to(device, dtype=weight_dtype)
    
    # 3. Process data
    mat_files = glob(os.path.join(args.data_folder, "*.mat"))
    if not mat_files:
        logger.warning(f"No .mat files found in {args.data_folder}")
        return

    # --- 核心：新的滑动窗口参数 ---
    dim = 128         # 网络要求的输入空间维度 (128x128)
    step = 64         # 每次滑动的步长，确保边缘重叠
    pad = (dim - step) // 2  # 32，为原始图像四周的填充量
    
    for mat_path in mat_files:
        name_test_id = os.path.splitext(os.path.basename(mat_path))[0]
        logger.info(f"Processing data: {name_test_id}...")
        
        # 加载数据
        mat_dict = scio.loadmat(mat_path)
        if "spad_processed_data" in mat_dict:
            raw_data = mat_dict["spad_processed_data"][0, 0].toarray()
            M_mea = raw_data.astype(np.float32).reshape([1, 1, 1536, 256, 256])
            M_mea=M_mea[:,:,50:511+50,:,:]
            M_mea = torch.from_numpy(M_mea).to(device, dtype=weight_dtype) # [B=1, C=1, D=512, H, W]
        elif "spad" in mat_dict:
            M_mea = mat_dict["spad"]
            M_mea = torch.from_numpy(M_mea).view(1, 1, D, H, W).to(device, dtype=weight_dtype) # [B=1, C=1, D=512, H, W]
        else:
            logger.error(f"Cannot find valid keys in {mat_path}. Skipping...")
            continue

        # --- 动态推断形状 ---
        # 假设时间维度固定为 512
        D = 512
        H,W=M_mea.shape[-2],M_mea.shape[-1]

        # 空间维度 Padding: (left, right, top, bottom, front, back)
        M_mea = F.pad(M_mea, (pad, pad, pad, pad, 0, 0)) 

        # 初始化输出画布
        out_rgb = np.zeros((3, H, W), dtype=np.float32)
        out_depth = np.zeros((3, H, W), dtype=np.float32)

        # 动态计算所需的步数
        num_steps_h = H // step
        num_steps_w = W // step

        t_s = time.time()
        
        # 动态滑动窗口遍历
        for i in range(num_steps_h):
            for j in range(num_steps_w):
                # 截取 128x128 的输入特征块 -> [1, 1, 512, 128, 128]
                M_mea_input = M_mea[:, :, :, i*step : i*step+dim, j*step : j*step+dim]

                with torch.no_grad():
                    # 1. 编码提取
                    pred_rgb_latent, pred_depth_latent = encoder(M_mea_input)
                    
                    # 2. VAE 解码 -> [1, 3, 128, 128]
                    pred_rgb_patch = ae.decode(pred_rgb_latent).squeeze(0).cpu().numpy()
                    pred_depth_patch = ae.decode(pred_depth_latent).squeeze(0).cpu().numpy()
                
                # 3. 剥离边缘 32 像素的 Padding，仅取中心 64x64 核心区域
                core_rgb = pred_rgb_patch[:, pad : pad+step, pad : pad+step]
                core_depth = pred_depth_patch[:, pad : pad+step, pad : pad+step]
                
                # 拼接入全景画布
                out_rgb[:, i*step : (i+1)*step, j*step : (j+1)*step] = core_rgb
                out_depth[:, i*step : (i+1)*step, j*step : (j+1)*step] = core_depth

        t_e = time.time()
        logger.info(f"Inference time for {name_test_id} (Size: {H}x{W}): {t_e - t_s:.2f} seconds")

        # 画图保存 (C, H, W) -> (H, W, C)
        out_rgb = out_rgb.transpose(1, 2, 0)
        out_depth = out_depth.transpose(1, 2, 0)

        rgb_save_path = os.path.join(out_dir, f"{name_test_id}_rgb.png")
        depth_save_path = os.path.join(out_dir, f"{name_test_id}_depth.png")

        out_rgb_norm = (out_rgb - out_rgb.min()) / (out_rgb.max() - out_rgb.min() + 1e-8)
        out_depth_norm = (out_depth - out_depth.min()) / (out_depth.max() - out_depth.min() + 1e-8)

        plt.imsave(rgb_save_path, out_rgb_norm)
        plt.imsave(depth_save_path, out_depth_norm)

    logger.info("Inference complete! All images saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 Inference: Decode predicted latents (LOS Only, 128x128x512 adaptive)")
    
    parser.add_argument("--checkpoint", type=str, default="temp/phase1_pretrain_LOS/encoder_step_003500.pth")
    parser.add_argument("--ae", type=str, default="/data/Flux.1/JointDiT-main/models/flux/ae.safetensors")
    parser.add_argument("--data_folder", type=str, default="/data/ImageTask/LOS/lindell2018_data/captured/")
    parser.add_argument("--out_dir", type=str, default="temp/")
    parser.add_argument("--mixed_precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    inference(args)