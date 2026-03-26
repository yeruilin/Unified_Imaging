"""
Phase 1 Training: NLOS Encoder Pre-training

This script trains the NlosEncoder to align NLOS time-of-flight data with RGB-D latents.
The training process:
1. Load RGB-D images from the same scene
2. Encode RGB and depth images using Flux's pre-trained VAE to get ground truth latents
3. Pass NLOS measurement data through NlosEncoder to get predicted latents
4. Minimize the L2 loss between predicted latents and ground truth latents
"""

# python train_phase1.py --data_folder "/data/ImageData/LOS/" --max_train_steps 5000  # LOS训练
# python train_phase1.py --data_folder "/data/Flux.1/NLOS_data0310/" --max_train_steps 1500  --nlos_wall_size 2.0 # NLOS训练

import argparse
import logging
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from safetensors.torch import save_file

# Import LOSDataset, LOSEncoder
from dataset.losdataset import LOSDataset
from encoder.los_encoder import LOSEncoder

# Import NLOSDataset, NlosEncoder
from dataset.nlosdataset import NLOSDataset
from encoder.nlos_encoder import NlosEncoder

task="NLOS"

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
    """Load Flux AutoEncoder."""
    from library.flux_models import configs
    ae_params = configs["dev"].ae_params
    ae = AutoEncoder(ae_params).to(dtype)
    
    logger.info(f"Loading VAE from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=device, dtype=dtype)
    info = ae.load_state_dict(sd, strict=False)
    logger.info(f"Loaded VAE: {info}")
    return ae


def train(args):
    """Main training function for Phase 1: NLOS Encoder pre-training."""
    
    args.exp_name=args.exp_name+f"_{task}"
    out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f"Set random seed: {args.seed}")
    
    # Load VAE (AutoEncoder) for encoding RGB-D to latents
    logger.info("Loading VAE for encoding RGB-D images")
    ae = load_ae(args.ae, weight_dtype, "cpu")
    ae.requires_grad_(False)
    ae.eval()
    ae.to(device, dtype=weight_dtype)
    
    # Initialize NlosEncoder
    logger.info("Initializing NlosEncoder")

    if task == "LOS":
        encoder=LOSEncoder(in_channels=1, out_dim=16)
        # Prepare dataset and dataloader
        train_dataset=LOSDataset(args.data_folder,split='train',train_ratio=0.9)

    elif task == "NLOS":
        encoder = NlosEncoder(
            ch_in=1,
            spatial=args.spatial,
            crop=args.nlos_crop,
            bin_resolution=args.nlos_bin_resolution,
            wall_size=args.nlos_wall_size,
        )
        # Prepare dataset and dataloader
        train_dataset = NLOSDataset(
            root=[args.data_folder],
            split=True,
            target_size=args.spatial,
            clip=args.nlos_crop,
            target_noise=0,  # 训练加噪对结果的影响较小，且加噪会增加训练不稳定性，所以不加噪似乎更好
        )
        # 让物体在不同width条件下训练似乎也不能提升效果

    encoder.to(device, dtype=weight_dtype)
    
    # Setup optimizer
    optimizer = AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters in Encoder: {n_params}")
    
    n_workers = min(32, os.cpu_count() // 2) if os.cpu_count() else 2
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=args.persistent_data_loader_workers,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
        logger.info(f"Override steps. Steps for {args.max_train_epochs} epochs: {args.max_train_steps}")
    
    # Setup learning rate scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_train_steps,
        eta_min=args.lr * 0.1
    )
    
    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from step {global_step}, epoch {start_epoch}")
    
    # Training loop
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info("Starting Phase 1 training: NLOS Encoder pre-training")
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Number of examples: {len(train_dataset)}")
    logger.info(f"Number of batches per epoch: {len(train_dataloader)}")
    logger.info(f"Number of epochs: {math.ceil(args.max_train_steps / num_update_steps_per_epoch)}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Training steps")
    
    # Loss tracking
    running_loss = 0.0
    running_loss_rgb = 0.0
    running_loss_depth = 0.0
    log_interval = 10
    
    encoder.train()
    
    for epoch in range(start_epoch, math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        logger.info(f"\nEpoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            # Get data from NLOSDataset
            nlos_data = batch["meas"].to(device, dtype=weight_dtype)
            images = batch["inten"].to(device, dtype=weight_dtype)
            depths = batch["dep"].to(device, dtype=weight_dtype)
            
            # Encode RGB and depth to latents using VAE (ground truth)
            with torch.no_grad():
                rgb_latents = ae.encode(images)
                # Depth from NLOSDataset should already be multi-channel
                if depths.shape[1] == 1:
                    depths = depths.repeat(1, 3, 1, 1)
                depth_latents = ae.encode(depths)
            
            # Forward pass through NlosEncoder
            pred_rgb_latent, pred_depth_latent = encoder(nlos_data)
            
            # Resize predicted latents to match ground truth if needed
            if pred_rgb_latent.shape != rgb_latents.shape:
                print(f"Resizing predicted latents from {pred_rgb_latent.shape} to {rgb_latents.shape}")
                # pred_rgb_latent = F.interpolate(
                #     pred_rgb_latent, 
                #     size=rgb_latents.shape[2:],
                #     mode='bilinear', 
                #     align_corners=False
                # )
                # pred_depth_latent = F.interpolate(
                #     pred_depth_latent,
                #     size=depth_latents.shape[2:],
                #     mode='bilinear',
                #     align_corners=False
                # )
            
            # Calculate L2 loss (MSE)
            loss_rgb = F.mse_loss(pred_rgb_latent.float(), rgb_latents.float(), reduction="mean")
            loss_depth = F.mse_loss(pred_depth_latent.float(), depth_latents.float(), reduction="mean")
            loss = loss_rgb + loss_depth
            
            # Normalize loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                progress_bar.update(1)
                
                # Logging
                running_loss += loss.item() * args.gradient_accumulation_steps
                running_loss_rgb += loss_rgb.item()
                running_loss_depth += loss_depth.item()
                
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    avg_loss_rgb = running_loss_rgb / log_interval
                    avg_loss_depth = running_loss_depth / log_interval
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "loss_rgb": f"{avg_loss_rgb:.4f}",
                        "loss_depth": f"{avg_loss_depth:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"
                    })
                    
                    running_loss = 0.0
                    running_loss_rgb = 0.0
                    running_loss_depth = 0.0
                
                # Save checkpoint
                if global_step % args.save_every_n_steps == 0 and global_step > 0:
                    save_path = os.path.join(out_dir, f"encoder_step_{global_step:06d}.pth")
                    torch.save({
                        "model_state_dict": encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                
                # Check if training is complete
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    progress_bar.close()
    
    # Training complete - save final model
    logger.info("Training complete!")
    
    # Save final model
    final_save_path = os.path.join(out_dir, "encoder_final.pth")
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "global_step": global_step,
    }, final_save_path)
    logger.info(f"Saved final model to {final_save_path}")
    
    # Also save as safetensors for compatibility
    final_safetensors_path = os.path.join(out_dir, "encoder_final.safetensors")
    save_file(encoder.state_dict(), final_safetensors_path)
    logger.info(f"Saved final model (safetensors) to {final_safetensors_path}")


def setup_parser() -> argparse.ArgumentParser:
    """Define command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Training: Encoder pre-training to align with RGB-D latents"
    )
    
    # Model paths
    parser.add_argument(
        "--ae",
        type=str,
        default="/data/Flux.1/JointDiT-main/models/flux/ae.safetensors",
        help="Path to Flux VAE (AutoEncoder) weights"
    )
    
    # Data paths
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/data/ImageData/NLOS/data0323/",
        help="Folder containing measurement data"
    )
    
    
    # NlosEncoder parameters
    parser.add_argument(
        "--spatial",
        type=int,
        default=128,
        help="Spatial resolution for NLOS data"
    )
    parser.add_argument(
        "--nlos_crop",
        type=int,
        default=512,
        help="Temporal crop size for NLOS data"
    )
    parser.add_argument(
        "--nlos_bin_resolution",
        type=float,
        default=33e-12,
        help="Bin resolution for NLOS data (in seconds)"
    )
    parser.add_argument(
        "--nlos_wall_size",
        type=float,
        default=2.0,
        help="Wall size for NLOS scene"
    )
    
    # Training parameters
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./temp",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="phase1_pretrain",
        help="Experiment name"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5001,
        help="Total number of training steps: 超过数据量的两倍就容易过拟合"
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides max_train_steps if set)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="Keep data loader workers alive between epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # Start training
    train(args)
