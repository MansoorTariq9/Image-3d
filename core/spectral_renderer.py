import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class RenderingConfig:
    """Configuration for spectral rendering"""
    image_height: int = 324
    image_width: int = 332
    near_plane: float = 0.01
    far_plane: float = 100.0
    background_value: float = 1.0  # Following HS-NeRF convention
    max_gaussians_per_pixel: int = 100

class SpectralGaussianRenderer(nn.Module):
    """Differentiable renderer for hyperspectral Gaussian splatting"""
    
    def __init__(self, config: RenderingConfig):
        super().__init__()
        self.config = config
        
        # Learnable tone mapping for each channel
        self.tone_mapping = nn.Parameter(torch.ones(120))
        
    def compute_projection(
        self,
        positions: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D points to 2D screen space
        
        Args:
            positions: [N, 3] 3D positions
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics (world to camera)
            
        Returns:
            screen_pos: [N, 2] screen positions
            depths: [N] depth values
        """
        # Transform to camera space
        positions_homo = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)  # [N, 4]
        cam_positions = (extrinsics @ positions_homo.T).T[:, :3]  # [N, 3]
        
        # Project to screen
        proj_positions = (intrinsics @ cam_positions.T).T  # [N, 3]
        screen_pos = proj_positions[:, :2] / (proj_positions[:, 2:3] + 1e-6)
        depths = cam_positions[:, 2]
        
        return screen_pos, depths
    
    def compute_gaussian_2d(
        self,
        positions_2d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D Gaussian parameters from 3D
        
        Returns:
            cov2d: [N, 2, 2] 2D covariance matrices
            det: [N] determinants
        """
        N = positions_2d.shape[0]
        
        # Simplified 2D covariance computation
        # In practice, this would involve projecting 3D covariance
        scales_2d = scales[:, :2]  # Use only x,y scales
        
        # Create diagonal covariance
        cov2d = torch.zeros(N, 2, 2, device=scales.device)
        cov2d[:, 0, 0] = scales_2d[:, 0] ** 2
        cov2d[:, 1, 1] = scales_2d[:, 1] ** 2
        
        # Compute determinant for normalization
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1]
        
        return cov2d, det
    
    def rasterize_gaussians(
        self,
        positions_2d: torch.Tensor,
        depths: torch.Tensor,
        cov2d: torch.Tensor,
        opacities: torch.Tensor,
        spectral_features: torch.Tensor,
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """Rasterize Gaussians to image
        
        Returns:
            Dictionary with rendered images and depth
        """
        device = positions_2d.device
        C = spectral_features.shape[1]  # 120 channels
        
        # Initialize output buffers
        rendered_spectral = torch.zeros(C, H, W, device=device)
        rendered_depth = torch.zeros(H, W, device=device)
        accumulated_alpha = torch.zeros(H, W, device=device)
        
        # Sort Gaussians by depth (front to back)
        sorted_indices = torch.argsort(depths)
        
        # Create pixel grid
        y_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # [H, W, 2]
        
        # Rasterize each Gaussian
        for idx in sorted_indices:
            if accumulated_alpha.mean() > 0.99:  # Early termination
                break
                
            pos = positions_2d[idx:idx+1]  # [1, 2]
            cov = cov2d[idx]  # [2, 2]
            opacity = opacities[idx]  # [1]
            spectral = spectral_features[idx]  # [C]
            depth = depths[idx]
            
            # Skip if outside viewport
            if (pos[0, 0] < -W or pos[0, 0] > 2*W or 
                pos[0, 1] < -H or pos[0, 1] > 2*H):
                continue
            
            # Compute Gaussian influence
            diff = pixel_coords - pos  # [H, W, 2]
            
            # Inverse covariance
            cov_inv = torch.inverse(cov + 1e-4 * torch.eye(2, device=device))
            
            # Mahalanobis distance
            # diff @ cov_inv @ diff.T for each pixel
            mahal = torch.sum(diff @ cov_inv * diff, dim=-1)  # [H, W]
            
            # Gaussian weight
            weight = torch.exp(-0.5 * mahal)
            
            # Alpha compositing
            alpha = opacity * weight * (1 - accumulated_alpha)
            
            # Accumulate color and depth
            rendered_spectral += spectral.view(-1, 1, 1) * alpha
            rendered_depth += depth * alpha
            accumulated_alpha += alpha
            
        # Add background
        background = self.config.background_value
        rendered_spectral += background * (1 - accumulated_alpha).unsqueeze(0)
        
        return {
            'spectral': rendered_spectral,  # [C, H, W]
            'depth': rendered_depth,  # [H, W]
            'alpha': accumulated_alpha  # [H, W]
        }
    
    def forward(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor],
        wavelength_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Render Gaussian splats
        
        Args:
            gaussian_data: Dictionary with Gaussian parameters
            camera_params: Dictionary with 'intrinsics' and 'extrinsics'
            wavelength_indices: Optional indices of wavelengths to render
            
        Returns:
            Dictionary with rendered outputs
        """
        # Extract Gaussian parameters
        # Handle both dict and SpectralGaussianData dataclass
        if hasattr(gaussian_data, 'positions'):
            positions = gaussian_data.positions
            rotations = gaussian_data.rotations
            scales = gaussian_data.scales
            opacities = gaussian_data.opacities
            spectral_features = gaussian_data.spectral_features
        else:
            positions = gaussian_data['positions']
            rotations = gaussian_data['rotations']
            scales = gaussian_data['scales']
            opacities = gaussian_data['opacities']
            spectral_features = gaussian_data['spectral_features']
        
        # Extract camera parameters
        intrinsics = camera_params['intrinsics']
        extrinsics = camera_params['extrinsics']
        
        # Project to 2D
        positions_2d, depths = self.compute_projection(positions, intrinsics, extrinsics)
        
        # Filter visible Gaussians
        mask = (depths > self.config.near_plane) & (depths < self.config.far_plane)
        mask &= (positions_2d[:, 0] > -100) & (positions_2d[:, 0] < self.config.image_width + 100)
        mask &= (positions_2d[:, 1] > -100) & (positions_2d[:, 1] < self.config.image_height + 100)
        
        # Apply mask
        mask = mask.squeeze(-1)  # Remove extra dimension
        positions_2d = positions_2d[mask]
        depths = depths[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opacities = opacities[mask]
        spectral_features = spectral_features[mask]
        
        # Compute 2D Gaussian parameters
        cov2d, det = self.compute_gaussian_2d(positions_2d, scales, rotations, intrinsics)
        
        # Select wavelengths if specified
        if wavelength_indices is not None:
            spectral_features = spectral_features[:, wavelength_indices]
        
        # Rasterize
        rendered = self.rasterize_gaussians(
            positions_2d, depths, cov2d, opacities, spectral_features,
            self.config.image_height, self.config.image_width
        )
        
        # Apply tone mapping
        if wavelength_indices is None:
            rendered['spectral'] = rendered['spectral'] * self.tone_mapping.view(-1, 1, 1)
        
        return rendered

class SpectralLoss(nn.Module):
    """Loss functions for hyperspectral reconstruction"""
    
    def __init__(self, num_channels: int = 120):
        super().__init__()
        self.num_channels = num_channels
        
    def spectral_angle_mapper(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spectral Angle Mapper (SAM) loss"""
        # Normalize spectra
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Compute angle
        cos_angle = (pred_norm * target_norm).sum(dim=1)
        angle = torch.acos(torch.clamp(cos_angle, -0.999, 0.999))
        
        return angle.mean()
    
    def forward(
        self,
        rendered: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses
        
        Args:
            rendered: Dictionary with rendered outputs
            target: Dictionary with ground truth
            weights: Loss weights
            
        Returns:
            Dictionary with individual losses
        """
        if weights is None:
            weights = {
                'spectral_mse': 1.0,
                'spectral_sam': 0.1,
                'depth': 0.1,
                'smoothness': 0.01
            }
            
        losses = {}
        
        # Spectral reconstruction loss
        pred_spectral = rendered['spectral']
        target_spectral = target['spectral']
        
        # MSE loss
        losses['spectral_mse'] = F.mse_loss(pred_spectral, target_spectral)
        
        # SAM loss
        losses['spectral_sam'] = self.spectral_angle_mapper(pred_spectral, target_spectral)
        
        # Depth loss if available
        if 'depth' in target and 'depth' in rendered:
            valid_mask = target['depth'] > 0
            if valid_mask.any():
                losses['depth'] = F.l1_loss(
                    rendered['depth'][valid_mask],
                    target['depth'][valid_mask]
                )
            else:
                losses['depth'] = torch.tensor(0.0, device=pred_spectral.device)
        
        # Total weighted loss
        total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses