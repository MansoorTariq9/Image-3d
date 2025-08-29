import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class SpectralDepthEstimator(nn.Module):
    """Depth estimation module inspired by HS-NeRF"""
    
    def __init__(self, num_channels: int = 120, feature_dim: int = 256):
        super().__init__()
        self.num_channels = num_channels
        
        # Spectral feature extraction
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU()
        )
        
        # Depth prediction network
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU()
        )
        
        # Depth decoder with skip connections
        self.depth_decoder = nn.ModuleList([
            nn.ConvTranspose2d(feature_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1)
        ])
        
        # Normalization layers for decoder
        self.decoder_norms = nn.ModuleList([
            nn.GroupNorm(16, 128),
            nn.GroupNorm(8, 64),
            nn.GroupNorm(4, 32)
        ])
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Conv2d(32, 1, kernel_size=1)
        
    def extract_spectral_features(self, hsi: torch.Tensor) -> torch.Tensor:
        """Extract depth-relevant features from hyperspectral data"""
        # Reduce spectral dimension
        features = self.spectral_conv(hsi)
        return features
    
    def forward(self, hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate depth from hyperspectral image
        
        Args:
            hsi: [B, C, H, W] hyperspectral image
            
        Returns:
            Dictionary with 'depth' and 'uncertainty'
        """
        B, C, H, W = hsi.shape
        
        # Extract spectral features
        spectral_features = self.extract_spectral_features(hsi)
        
        # Encode
        skip_features = []
        x = spectral_features
        
        # First conv (no downsampling)
        x = self.depth_encoder[0](x)
        x = self.depth_encoder[1](x)
        x = self.depth_encoder[2](x)
        skip_features.append(x)  # 1/1 scale
        
        # Downsample blocks
        x = self.depth_encoder[3](x)
        x = self.depth_encoder[4](x)
        x = self.depth_encoder[5](x)
        skip_features.append(x)  # 1/2 scale
        
        x = self.depth_encoder[6](x)
        x = self.depth_encoder[7](x)
        x = self.depth_encoder[8](x)
        
        # Final encoding
        x = self.depth_encoder[9](x)
        x = self.depth_encoder[10](x)
        x = self.depth_encoder[11](x)
        
        # Decode with skip connections
        # Upsample 1
        x = self.depth_decoder[0](x)
        x = self.decoder_norms[0](x)
        x = F.relu(x)
        
        # Skip connection from 1/2 scale
        if x.shape[2:] == skip_features[1].shape[2:]:
            x = torch.cat([x, skip_features[1]], dim=1)
        
        # Upsample 2
        x = self.depth_decoder[1](x)
        x = self.decoder_norms[1](x)
        x = F.relu(x)
        
        # Skip connection from 1/1 scale
        if x.shape[2:] == skip_features[0].shape[2:]:
            x = torch.cat([x, skip_features[0]], dim=1)
        
        # Final layers
        x = self.depth_decoder[2](x)
        x = self.decoder_norms[2](x)
        features = F.relu(x)
        
        # Predict depth (positive values)
        depth = self.depth_decoder[3](features)
        depth = F.softplus(depth) + 0.01  # Ensure positive depth
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(features)
        uncertainty = torch.sigmoid(uncertainty)
        
        return {
            'depth': depth.squeeze(1),  # [B, H, W]
            'uncertainty': uncertainty.squeeze(1),  # [B, H, W]
            'features': spectral_features
        }

class DepthGuidedSampling(nn.Module):
    """Use depth estimates to guide Gaussian placement"""
    
    def __init__(self, num_samples: int = 10000):
        super().__init__()
        self.num_samples = num_samples
        
    def sample_points_from_depth(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample 3D points from depth map
        
        Args:
            depth: [B, H, W] depth maps
            intrinsics: [B, 3, 3] camera intrinsics
            uncertainty: [B, H, W] uncertainty maps (optional)
            
        Returns:
            points: [B, N, 3] sampled 3D points
        """
        B, H, W = depth.shape
        device = depth.device
        
        # Create pixel grid
        y_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)
        pixel_coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=0)  # [3, H, W]
        
        # Flatten for easier processing
        pixel_coords_flat = pixel_coords.view(3, -1).T  # [H*W, 3]
        
        points_all = []
        
        for b in range(B):
            # Get intrinsics inverse
            K_inv = torch.inverse(intrinsics[b])
            
            # Unproject to 3D
            depth_flat = depth[b].view(-1, 1)  # [H*W, 1]
            cam_coords = (K_inv @ pixel_coords_flat.T).T * depth_flat  # [H*W, 3]
            
            # Sample based on uncertainty if provided
            if uncertainty is not None:
                # Higher uncertainty = higher sampling probability
                sample_probs = uncertainty[b].view(-1)
                sample_probs = sample_probs / (sample_probs.sum() + 1e-6)
                
                # Sample indices
                indices = torch.multinomial(sample_probs, self.num_samples, replacement=True)
                sampled_points = cam_coords[indices]
            else:
                # Uniform sampling
                indices = torch.randperm(cam_coords.shape[0], device=device)[:self.num_samples]
                sampled_points = cam_coords[indices]
                
            points_all.append(sampled_points)
            
        return torch.stack(points_all, dim=0)  # [B, N, 3]

class DepthConsistencyLoss(nn.Module):
    """Enforce depth consistency between rendered and estimated depth"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        rendered_depth: torch.Tensor,
        estimated_depth: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth consistency loss
        
        Args:
            rendered_depth: [B, H, W] depth from Gaussian rendering
            estimated_depth: [B, H, W] depth from depth estimator
            uncertainty: [B, H, W] uncertainty weights (optional)
            valid_mask: [B, H, W] valid pixel mask (optional)
            
        Returns:
            Scalar loss value
        """
        if valid_mask is None:
            valid_mask = (rendered_depth > 0) & (estimated_depth > 0)
            
        if not valid_mask.any():
            return torch.tensor(0.0, device=rendered_depth.device)
            
        # Compute difference
        depth_diff = torch.abs(rendered_depth - estimated_depth)
        
        # Weight by inverse uncertainty if provided
        if uncertainty is not None:
            weights = (1.0 - uncertainty) * valid_mask.float()
        else:
            weights = valid_mask.float()
            
        # Normalize weights
        weights = weights / (weights.sum() + 1e-6)
        
        # Weighted loss
        loss = (depth_diff * weights).sum()
        
        return loss