import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import yaml

# Import our modules
from core.spectral_gaussian import SpectralGaussian3D
from core.hsi_vae import HyperspectralVAE
from core.spectral_renderer import SpectralGaussianRenderer, SpectralLoss, RenderingConfig
from core.depth_estimator import SpectralDepthEstimator, DepthConsistencyLoss
from data.dataset import HSIDataModule
from data.preprocessing import HSIConfig

class HSIGaussian3DModel(nn.Module):
    """Complete HSI 3D reconstruction model"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.vae = HyperspectralVAE(
            num_channels=config["num_channels"],
            latent_dim=config["latent_dim"],
            num_views=config["num_views"],
            point_cloud_size=config["point_cloud_size"]
        )
        
        self.gaussian_model = SpectralGaussian3D(
            num_points=config["num_gaussians"],
            num_channels=config["num_channels"],
            wavelength_range=tuple(config["wavelength_range"])
        )
        
        self.renderer = SpectralGaussianRenderer(
            RenderingConfig(
                image_height=config["image_height"],
                image_width=config["image_width"],
                background_value=config["background_value"]
            )
        )
        
        self.depth_estimator = SpectralDepthEstimator(
            num_channels=config["num_channels"]
        )
        
        # Loss functions
        self.spectral_loss_fn = SpectralLoss(num_channels=config["num_channels"])
        self.depth_loss_fn = DepthConsistencyLoss()
        
        # Loss weights
        self.loss_weights = config.get("loss_weights", {
            "spectral_mse": 1.0,
            "spectral_sam": 0.1,
            "depth": 0.1,
            "kl": 0.001,
            "smoothness": 0.01
        })
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        # Extract batch data
        hsi_views = batch["hsi"]  # [B, V, C, H, W]
        intrinsics = batch["intrinsics"]  # [B, V, 3, 3]
        extrinsics = batch["extrinsics"]  # [B, V, 4, 4]
        
        B, V, C, H, W = hsi_views.shape
        
        # Encode views to latent
        vae_output = self.vae(hsi_views)
        latent = vae_output["latent"]
        points = vae_output["points"]
        spectral_features = vae_output["spectral_features"]
        
        # Initialize Gaussians from VAE output
        # This is a simplified version - in practice you'd have more sophisticated initialization
        if self.training and hasattr(self, "iteration") and self.iteration == 0:
            with torch.no_grad():
                # Use VAE points to initialize Gaussian positions
                num_points = min(points.shape[1], self.gaussian_model._positions.shape[0])
                self.gaussian_model._positions.data[:num_points] = points[0, :num_points]
                self.gaussian_model._spectral_features_raw.data[:num_points] = spectral_features[0, :num_points]
        
        # Get Gaussian parameters
        gaussian_params = self.gaussian_model({"camera": intrinsics[0, 0]})
        
        # Create SpectralGaussianData object
        from core.spectral_gaussian import SpectralGaussianData
        gaussian_data = SpectralGaussianData(
            positions=gaussian_params['positions'],
            rotations=gaussian_params['rotations'],
            scales=gaussian_params['scales'],
            opacities=gaussian_params['opacities'],
            spectral_features=gaussian_params['spectral_features'],
            wavelengths=gaussian_params['wavelengths']
        )
        
        # Render from multiple views
        rendered_views = []
        target_views = []
        depth_estimates = []
        
        for b in range(B):
            for v in range(V):
                # Camera parameters for this view
                camera_params = {
                    "intrinsics": intrinsics[b, v],
                    "extrinsics": extrinsics[b, v]
                }
                
                # Render view
                rendered = self.renderer(gaussian_data, camera_params)
                rendered_views.append(rendered)
                
                # Target view
                target = {
                    "spectral": hsi_views[b, v],
                    "wavelengths": batch["wavelengths"]
                }
                target_views.append(target)
                
                # Estimate depth
                depth_est = self.depth_estimator(hsi_views[b, v].unsqueeze(0))
                depth_estimates.append(depth_est)
                
        # Combine outputs
        outputs = {
            "rendered_views": rendered_views,
            "target_views": target_views,
            "depth_estimates": depth_estimates,
            "vae_mean": vae_output["mean"],
            "vae_logvar": vae_output["logvar"],
            "gaussian_data": gaussian_data
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Spectral reconstruction loss for each view
        spectral_losses = []
        depth_losses = []
        
        for i, (rendered, target, depth_est) in enumerate(zip(
            outputs["rendered_views"],
            outputs["target_views"],
            outputs["depth_estimates"]
        )):
            # Spectral loss
            view_losses = self.spectral_loss_fn(rendered, target, self.loss_weights)
            for k, v in view_losses.items():
                if k != "total":
                    spectral_losses.append(v)
                    
            # Depth consistency loss
            if "depth" in rendered and "depth" in depth_est:
                depth_loss = self.depth_loss_fn(
                    rendered["depth"].unsqueeze(0),
                    depth_est["depth"],
                    depth_est.get("uncertainty")
                )
                depth_losses.append(depth_loss)
                
        # Average losses across views
        losses["spectral"] = torch.stack(spectral_losses).mean()
        if depth_losses:
            losses["depth"] = torch.stack(depth_losses).mean()
            
        # KL divergence loss for VAE
        mean = outputs["vae_mean"]
        logvar = outputs["vae_logvar"]
        losses["kl"] = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Spectral smoothness regularization
        gaussian_data = outputs["gaussian_data"]
        if hasattr(gaussian_data, 'spectral_features'):
            losses["smoothness"] = gaussian_data.spectral_features.var(dim=1).mean()
        else:
            losses["smoothness"] = gaussian_data["spectral_features"].var(dim=1).mean()
        
        # Total weighted loss
        total_loss = sum(self.loss_weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total_loss
        
        return losses

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Compute losses
        losses = model.compute_loss(outputs)
        
        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update iteration counter
        if hasattr(model, "iteration"):
            model.iteration += 1
        else:
            model.iteration = 1
            
        # Logging
        total_loss += losses["total"].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": losses["total"].item(),
            "spectral": losses.get("spectral", 0).item(),
            "depth": losses.get("depth", 0).item()
        })
        
        # Log to wandb
        if batch_idx % 10 == 0 and wandb.run is not None:
            wandb.log({
                "train/total_loss": losses["total"].item(),
                "train/spectral_loss": losses.get("spectral", 0).item(),
                "train/depth_loss": losses.get("depth", 0).item(),
                "train/kl_loss": losses.get("kl", 0).item(),
                "train/smoothness_loss": losses.get("smoothness", 0).item(),
                "train/iteration": model.iteration
            })
            
    return total_loss / num_batches

def validate(model, dataloader, device):
    """Validation loop"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute losses
            losses = model.compute_loss(outputs)
            
            total_loss += losses["total"].item()
            num_batches += 1
            
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description="Train HSI Gaussian 3D model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_root", type=str, required=True, help="Path to HSI dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=config)
        
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = HSIDataModule(
        data_root=args.data_root,
        batch_size=config["batch_size"],
        num_views=config["num_views"],
        num_workers=config["num_workers"]
    )
    data_module.setup()
    
    # Initialize model
    model = HSIGaussian3DModel(config).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
        
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, config["num_epochs"]):
        # Train
        train_loss = train_epoch(
            model,
            data_module.train_dataloader(),
            optimizer,
            device,
            epoch
        )
        
        # Validate
        val_loss = validate(model, data_module.val_dataloader(), device)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss
            })
            
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "val_loss": val_loss
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            
        # Regular checkpoint
        if epoch % config.get("save_every", 10) == 0:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pth")
            
    print("Training completed!")
    
if __name__ == "__main__":
    main()