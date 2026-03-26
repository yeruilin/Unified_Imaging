"""
Inference script for NLOS-based depth estimation and depth-to-image generation.

This script uses NlosEncoder to encode NLOS measurement data into latents,
then uses JointDiT for depth estimation or depth-to-image generation.
"""

# yrl: 完整推理整个flow matching流程，目前还用不上

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, Tuple
from functools import lru_cache

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from accelerate.utils import set_seed

# Add local script path for "sd-scripts" utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
from library import flux_utils
from library.device_utils import clean_memory_on_device

# JointDiT-specific utilities
from jointdit_library.jointdit_utils import load_empty_flux_model, setup_jointdit_model
from jointdit_library.inference_pipeline import conditional_generation

# Import NlosEncoder
from encoder.nlos_encoder import NlosEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_nlos_encoder(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> NlosEncoder:
    """Load trained NlosEncoder from checkpoint."""
    logger.info(f"Loading NlosEncoder from {ckpt_path}")
    
    # Initialize NlosEncoder with default parameters
    # These should match the training configuration
    nlos_encoder = NlosEncoder(
        ch_in=1,
        spatial=128,  # Should match training
        crop=512,     # Should match training
        bin_resolution=33e-12,
        wall_size=1.7,
    )
    
    # Load checkpoint
    if ckpt_path.endswith('.safetensors'):
        state_dict = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    nlos_encoder.load_state_dict(state_dict, strict=False)
    nlos_encoder.to(device, dtype=dtype)
    nlos_encoder.eval()
    nlos_encoder.requires_grad_(False)
    
    logger.info("NlosEncoder loaded successfully")
    return nlos_encoder


def load_nlos_data(data_path: str) -> torch.Tensor:
    """
    Load NLOS measurement data.
    Supports .npy, .pt, .hdr formats.
    """
    logger.info(f"Loading NLOS data from {data_path}")
    
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.npy':
        data = np.load(data_path)
        data = torch.from_numpy(data).float()
    elif ext == '.mat':
        data = torch.load(data_path, map_location='cpu')
    elif ext == '.hdr':
        import cv2
        x = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = x.astype(np.float32)
        width = int(np.sqrt(x.shape[0]))
        x = x.reshape(width, width, x.shape[1], 1)
        x = x.transpose(3, 2, 0, 1)  # (1, T, H, W)
        data = torch.from_numpy(x).float()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Ensure correct shape [1, 1, T, H, W]
    if data.ndim == 3:
        # [T, H, W] -> [1, 1, T, H, W]
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.ndim == 4:
        # [1, T, H, W] -> [1, 1, T, H, W]
        data = data.unsqueeze(0)
    
    # Normalize
    data = data / (data.max() + 1e-8)
    
    logger.info(f"NLOS data shape: {data.shape}")
    return data


def encode_with_nlos_encoder(
    nlos_encoder: NlosEncoder,
    nlos_data: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode NLOS data to RGB and depth latents using NlosEncoder.
    
    Args:
        nlos_encoder: Trained NlosEncoder model
        nlos_data: NLOS measurement data [B, 1, T, H, W]
        device: Device to run on
        dtype: Data type
        
    Returns:
        rgb_latent: RGB latent [B, 16, H//8, W//8]
        depth_latent: Depth latent [B, 16, H//8, W//8]
    """
    with torch.no_grad():
        nlos_data = nlos_data.to(device, dtype=dtype)

        # Input is [B, 1, T, H, W]
        
        # Forward pass
        rgb_latent, depth_latent = nlos_encoder(nlos_data)
        
    return rgb_latent, depth_latent


def prepare_accelerator_simple(mixed_precision: str = "fp32"):
    """
    精简版的 accelerator 初始化，仅用于推理。
    
    Args:
        mixed_precision: 混合精度模式 ("fp32", "fp16", "bf16")
        
    Returns:
        Accelerator 实例
    """
    from accelerate import Accelerator, InitProcessGroupKwargs
    import datetime
    
    kwargs_handlers = []
    
    # 多 GPU 支持
    if torch.cuda.device_count() > 1:
        kwargs_handlers.append(
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                timeout=datetime.timedelta(minutes=30),
            )
        )
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision if mixed_precision != "fp32" else "no",
        kwargs_handlers=kwargs_handlers if kwargs_handlers else None,
    )
    logger.info(f"Accelerator device: {accelerator.device}")
    return accelerator


def inference_nlos(args: argparse.Namespace):
    """
    Main inference routine using NlosEncoder:
      - Load NlosEncoder and JointDiT
      - Encode NLOS data to latents
      - Generate RGB from depth or depth from RGB
    """
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    # Prepare accelerator and dtypes (精简版)
    accelerator = prepare_accelerator_simple(args.mixed_precision)
    
    # 设置数据类型
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # 1) Load NlosEncoder
    nlos_encoder = load_nlos_encoder(args.nlos_encoder_path, device, weight_dtype)
    logger.info("Loading and encoding NLOS data")
    nlos_data = load_nlos_data(args.input_nlos_data)
    
    rgb_latent, depth_latent = encode_with_nlos_encoder(
        nlos_encoder, nlos_data, device, weight_dtype
    )
    
    # 2) Load VAE for decoding (not encoding)
    logger.info("Loading VAE for decoding")
    ae = flux_utils.load_ae(args.ae, weight_dtype, device="cpu")
    ae.eval().requires_grad_(False)
    ae.to(device, dtype=weight_dtype)
    clean_memory_on_device(device)
    
    # 3) Load JointDiT
    logger.info("Loading JointDiT")
    _, jointdit = load_empty_flux_model(
        args.pretrained_model_name_or_path,
        weight_dtype,
        device="cpu",
        disable_mmap=args.disable_mmap_load_safetensors
    )
    
    jointdit = jointdit.to_empty(device=device)
    
    # Load base Flux weights
    base_ckpt = load_file(args.pretrained_model_name_or_path)
    jointdit.load_state_dict(base_ckpt, strict=False)
    
    # Attach JointDiT-specific adapters
    jointdit = setup_jointdit_model(jointdit, lora_rank=64)
    
    # Load saved adapter weights
    addons = load_file(args.jointdit_addons_path)
    jointdit.load_state_dict(addons, strict=False)
    
    jointdit.requires_grad_(False)
    jointdit.to(weight_dtype)
    jointdit.eval()
    
    # block-swap memory optimization
    if args.blocks_to_swap and args.blocks_to_swap > 0:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks")
        jointdit.enable_block_swap(args.blocks_to_swap, accelerator.device)
    
    # Accelerator prepare (handles placement & optional deepspeed)
    jointdit = accelerator.prepare(jointdit, device_placement=[not args.blocks_to_swap])
    if args.blocks_to_swap and args.blocks_to_swap > 0:
        accelerator.unwrap_model(jointdit).move_to_device_except_swap_blocks(accelerator.device)
    
    # 4) Load text encoders
    clip_l = flux_utils.load_clip_l(
        args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    t5xxl = flux_utils.load_t5xxl(
        args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    for m in (clip_l, t5xxl):
        m.eval().requires_grad_(False)
        m.to(device)
    
    # Text encoding
    from library import strategy_flux, strategy_base
    t5xxl_max_len = 16
    flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
    strategy_base.TokenizeStrategy.set_strategy(flux_tok)
    txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(txt_enc)
    
    # 5) Select latent based on generation type
    if args.gen_type == "depth_estimation":
        # Use NlosEncoder's RGB latent as input for depth estimation
        latent = rgb_latent
        logger.info("Using NlosEncoder RGB latent for depth estimation")
    else:  # depth_to_image
        # Use NlosEncoder's depth latent as input for depth-to-image
        latent = depth_latent
        logger.info("Using NlosEncoder depth latent for depth-to-image")

    # Prepare prompts
    if args.gen_type == "depth_to_image":
        text_prompt = [args.text_prompt]
    else:  # depth_estimation
        text_prompt = [" "]
    
    with torch.no_grad():
        tokens_and_masks = flux_tok.tokenize(text_prompt)
        inputs = [t.to(device) for t in tokens_and_masks]
        conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
        l_pooled, t5_out, txt_ids, attn_mask = conds
        
        if not getattr(args, 'is_txt_ids_training', False):
            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=device)
        if not getattr(args, 'is_attnmask_training', False):
            attn_mask = None
    
    
    # Get resolution from latent shape
    resolution = [latent.shape[2] * 8, latent.shape[3] * 8]
    logger.info(f"Resolution: {resolution}")
    
    # 6) Generate
    logger.info(f"Starting {args.gen_type}")
    image, depth_image, depth_raw = conditional_generation(
        accelerator, args, jointdit, ae,
        latent, [l_pooled, t5_out, txt_ids, attn_mask],
        weight_dtype, resolution, args.gen_type
    )
    
    # 7) Save outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"nlos_{args.gen_type}")
    os.makedirs(out_dir, exist_ok=True)
    
    output_prefix = f"{args.output_prefix}_{timestamp}" if args.output_prefix else f"nlos_{timestamp}"
    
    image.save(os.path.join(out_dir, f"image_{output_prefix}.png"))
    depth_image.save(os.path.join(out_dir, f"depth_{output_prefix}.png"))
    
    logger.info(f"Saved outputs to {out_dir}")
    
    return image, depth_image


def setup_parser() -> argparse.ArgumentParser:
    """Define command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NLOS-based inference for JointDiT"
    )

    # Input data
    parser.add_argument(
        "--input_nlos_data",
        type=str,
        default="/data/Flux.1/NLOS_data0122/confocal-6.hdr",
        help="Path to NLOS measurement data (.npy, .pt, or .hdr file)"
    )
    
    # Model paths
    parser.add_argument(
        "--nlos_encoder_path",
        type=str,
        default="temp/phase1_nlos_pretrain/nlos_encoder_step_000800.pth",
        help="Path to trained NlosEncoder checkpoint (.pth or .safetensors)"
    )
    parser.add_argument(
        "--jointdit_addons_path",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/jointdit/jointdit_addons.safetensors",
        help="Path to JointDiT addons weights"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/flux1-dev.safetensors"
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/clip_l.safetensors"
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/t5xxl_fp16.safetensors"
    )
    parser.add_argument(
        "--ae",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/ae.safetensors"
    )
    
    # Generation type
    parser.add_argument(
        "--gen_type",
        type=str,
        choices=["depth_estimation", "depth_to_image"],
        required=True,
        help="Generation mode: 'depth_estimation' or 'depth_to_image'"
    )
    
    # Text prompt (for depth_to_image)
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=" ",
        help="Text prompt for depth_to_image mode"
    )
    
    # Output settings
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for output filenames"
    )
    
    # Model settings
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision mode"
    )
    parser.add_argument(
        "--apply_t5_attn_mask",
        action="store_true",
        help="Apply T5 attention mask"
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="Disable memory mapping for safetensors"
    )
    
    # Generation parameters (inherited from flux_train_utils)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--depth_transform",
        type=str,
        default="none",
        choices=["none", "inverse"],
        help="Depth transform method"
    )
    
    # Accelerate-related parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of blocks to swap for memory optimization"
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # Load flags from JSON if available
    json_path = os.path.splitext(args.jointdit_addons_path)[0] + ".json"
    if os.path.isfile(json_path):
        with open(json_path, "r") as jf:
            flags = json.load(jf)
        args.is_txt_ids_training = flags.get("is_txt_ids_training", False)
        args.is_attnmask_training = flags.get("is_attnmask_training", False)
        args.depth_transform = flags.get("depth_transform", "none")
        logger.info(f"Loaded training flags from {json_path}")
    else:
        args.is_txt_ids_training = False
        args.is_attnmask_training = False
        args.depth_transform = "none"
    
    # Run inference
    inference_nlos(args)
