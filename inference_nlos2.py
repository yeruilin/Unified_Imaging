"""
Phase 1 Inference: Encoder Latent Decoding (Compatible with LOS & NLOS)

This script runs inference using the trained LOSEncoder or NlosEncoder.
The inference process:
1. Load trained Encoder and pre-trained Flux VAE
2. Pass measurement data through Encoder to get predicted latents
3. Decode the predicted latents using Flux VAE back to RGB and Depth images
4. Save the reconstructed images to disk
"""

# 推理单文件示例 (NLOS):
# python inference_nlos2.py --task NLOS --single_mat /data/ImageTask/NLOS/fk_data/fk_teaser180.mat --nlos_wall_size 2.0

# 推理整个文件夹示例 (LOS):
# python inference_nlos2.py --task LOS --data_folder /data/ImageData/LOS/

import argparse
import logging
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import NLOS components
from dataset.nlosdataset import NLOSDataset
from encoder.nlos_encoder import NlosEncoder

# Import LOS components
from dataset.losdataset import LOSDataset
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


def load_ae(ckpt_path: str, dtype: torch.dtype, device: str ="cpu"):
    from library.flux_models import configs
    ae_params = configs["dev"].ae_params
    ae = AutoEncoder(ae_params).to(dtype)

    encoder_params = sum(p.numel() for p in ae.encoder.parameters())
    decoder_params = sum(p.numel() for p in ae.decoder.parameters())
    logger.info(f"Initialized AutoEncoder with {encoder_params} encoder params and {decoder_params} decoder params.")
    
    logger.info(f"Loading VAE from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=device, dtype=dtype)
    info = ae.load_state_dict(sd, strict=False)
    logger.info(f"Loaded VAE: {info}")
    return ae

def load_single_mat(mat_path: str, device: torch.device, dtype: torch.dtype, task: str):
    """Load a single .mat file and convert it to model input format depending on the task."""
    import scipy.io as sio
    logger.info(f"Loading single .mat file: {mat_path} for task: {task}")
    
    mat_dict = sio.loadmat(mat_path)
            
    raw_data = mat_dict.get('data')
        
    logger.info(f"Loaded raw data shape: {raw_data.shape}")
    
    data = torch.from_numpy(raw_data).to(device, dtype=dtype)

    if task == "NLOS":
        # NLOS: Convert [N, N, M] -> [M, N, N] (Time, H, W)
        if data.ndim == 3:
            data = data.permute(2, 1, 0) # [T, H, W]
            data[400:,:,:]=0
            data[:150,:,:]=0
        # Final shape for NLOS model: [Batch=1, Channel=1, T, H, W]
        data = data.unsqueeze(0).unsqueeze(0)
    elif task == "LOS":
        # LOS: Typically 2D [H, W] or 3D [H, W, 1]
        if data.ndim == 3:
            data = data.squeeze(-1) # -> [H, W]
        # Final shape for LOS model: [Batch=1, Channel=1, H, W]
        data = data.unsqueeze(0).unsqueeze(0)

    logger.info(f"Processed data shape for model: {data.shape}")
    return data


def inference(args):
    """Main inference function for decoding predicted latents."""
    
    # 动态创建包含任务名的输出文件夹
    out_dir = args.out_dir+args.task+"inference_outputs/"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Decoded images will be saved to: {out_dir}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Load VAE (AutoEncoder)
    logger.info("Loading VAE for decoding latents")
    ae = load_ae(args.ae, weight_dtype, "cpu")
    ae.requires_grad_(False)
    ae.eval()
    ae.to(device, dtype=weight_dtype)
    
    # --- 根据任务初始化 Encoder ---
    logger.info(f"Initializing Encoder for task: {args.task}")
    if args.task == "LOS":
        encoder = LOSEncoder(in_channels=1, out_dim=16)
    elif args.task == "NLOS":
        encoder = NlosEncoder(
            ch_in=1,
            spatial=args.spatial,
            crop=args.nlos_crop,
            bin_resolution=args.nlos_bin_resolution,
            wall_size=args.nlos_wall_size,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # 加载权重
    logger.info(f"Loading Encoder weights from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        encoder.load_state_dict(checkpoint)
        
    encoder.to(device, dtype=weight_dtype)
    encoder.requires_grad_(False)
    encoder.eval()
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder loaded with {encoder_params} parameters.")
    
    # --- Prepare data loader ---
    if args.single_mat:
        # MODE 1: Single .mat File Inference
        data_tensor = load_single_mat(args.single_mat, device, weight_dtype, args.task)
        test_dataloader = [{"meas": data_tensor, "filename": [os.path.basename(args.single_mat).replace('.mat', '')]}]
        total_samples = 1
        logger.info("Running in single .mat file mode.")
        
    else:
        # MODE 2: Folder Dataset Inference
        if args.task == "LOS":
            test_dataset = LOSDataset(
                args.data_folder,
                split='test', # 假设 LOS 数据集有 test 或 val 划分，若无请改成相应参数
                train_ratio=0.998
            )
        elif args.task == "NLOS":
            test_dataset = NLOSDataset(
                root=[args.data_folder],
                split=False,  # Set to False assuming we are testing/evaluating
                target_size=args.spatial,
                clip=args.nlos_crop,
                background=args.nlos_background,
                target_noise=0,
            )

        n_workers = min(32, os.cpu_count() // 2) if os.cpu_count() else 2
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=n_workers,
            pin_memory=True,
        )
        total_samples = len(test_dataset)
        logger.info(f"Running in dataset mode. Total test samples: {total_samples}")
    
    # Inference loop
    progress_bar = tqdm(test_dataloader, desc="Running Inference")
    sample_idx = 0
    
    for batch in progress_bar:
        input_data = batch["meas"].to(device, dtype=weight_dtype)
        
        # rgb = batch["inten"].to(device, dtype=weight_dtype) # 用于仿真数据真值autoencoder验证
        # dep = batch["dep"].to(device, dtype=weight_dtype)
        # if dep.shape[1] == 1:
        #     dep = dep.repeat(1, 3, 1, 1)
        
        prefixes = batch.get("filename", [f"sample_{sample_idx + b:04d}" for b in range(input_data.shape[0])])
        
        with torch.no_grad():
            # 1. Forward pass through Encoder
            pred_rgb_latent, pred_depth_latent = encoder(input_data)
            # pred_rgb_latent=ae.encode(rgb)
            # pred_depth_latent=ae.encode(dep)
            
            # 2. Decode latents using Flux VAE
            pred_rgb_img = ae.decode(pred_rgb_latent)
            pred_depth_img = ae.decode(pred_depth_latent)
            
        # 3. Save outputs to disk
        for b in range(pred_rgb_img.shape[0]):
            prefix = prefixes[b]
            rgb_save_path = os.path.join(out_dir, f"{prefix}_rgb.png")
            depth_save_path = os.path.join(out_dir, f"{prefix}_depth.png")

            # 保存RGB图片
            temp = pred_rgb_img[b].cpu().numpy().transpose(1, 2, 0)
            temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
            plt.imsave(rgb_save_path, temp)

            # 保存深度图
            temp = pred_depth_img[b].cpu().numpy().transpose(1, 2, 0)
            temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
            plt.imsave(depth_save_path, temp)
            
        sample_idx += input_data.shape[0]
            
    logger.info("Inference complete! All images saved.")


def setup_parser() -> argparse.ArgumentParser:
    """Define command-line arguments for Inference."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Inference: Decode predicted latents back to RGB-D using Flux VAE"
    )
    
    # Task specification
    parser.add_argument(
        "--task",
        type=str,
        default="NLOS",
        choices=["LOS", "NLOS"],
        help="Specify the task: LOS or NLOS"
    )

    # Model checkpoints
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="temp/phase1_pretrain_LOS/encoder_step_003000.pth", #"temp/phase1_nlos_pretrain_0319_snr08/encoder_final.pth",
        # default="temp/phase1_pretrain_NLOS/encoder_step_002000.pth",
        help="Path to the trained Encoder checkpoint (.pth or .safetensors)"
    )
    parser.add_argument(
        "--ae",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/ae.safetensors",
        help="Path to Flux VAE (AutoEncoder) weights"
    )
    
    # Output path
    parser.add_argument(
        "--out_dir",
        type=str,
        default="temp/",
        help="Base output directory to save decoded images"
    )
    
    # --- Input Modes ---
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/data/ImageData/LOS/", #"/data/Flux.1/NLOS_data0310/",
        help="[MODE 1] Folder containing measurement data"
    )
    parser.add_argument(
        "--single_mat",
        type=str,
        default=None, 
        help="[MODE 2] Path to a single .mat file for inference. Overrides folder mode."
    )
    
    # Dataset & Encoder parameters
    parser.add_argument("--spatial", type=int, default=128, help="Spatial resolution")
    parser.add_argument("--nlos_background", type=float, default=0)
    parser.add_argument("--nlos_crop", type=int, default=512)
    parser.add_argument("--nlos_bin_resolution", type=float, default=33e-12)
    parser.add_argument("--nlos_wall_size", type=float, default=1.7)
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"]
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # Start inference
    inference(args)