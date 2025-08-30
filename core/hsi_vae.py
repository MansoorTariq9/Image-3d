import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class SpectralEncoder(nn.Module):
    """Encode hyperspectral channels to lower dimension"""
    
    def __init__(self, in_channels: int = 120, out_channels: int = 16):
        super().__init__()
        
        # Progressive channel reduction
        self.conv1 = nn.Conv1d(120, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] hyperspectral image
        Returns:
            [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to apply 1D conv along spectral dimension
        x_reshaped = x.reshape(B, C, H * W)
        
        # Spectral encoding
        x = F.relu(self.bn1(self.conv1(x_reshaped)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Reshape back to spatial
        return x.reshape(B, -1, H, W)

class HyperspectralVAE(nn.Module):
    """VAE adapted for hyperspectral multi-view input"""
    
    def __init__(
        self,
        num_channels: int = 120,
        latent_dim: int = 512,
        num_views: int = 4,
        point_cloud_size: int = 2048
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_views = num_views
        self.latent_dim = latent_dim
        self.point_cloud_size = point_cloud_size
        
        # Spectral encoder
        self.spectral_encoder = SpectralEncoder(num_channels, 16)
        
        # Spatial encoder (adapted from GaussianAnything)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
        )
        
        # Calculate spatial dimensions after encoding
        self.encoded_spatial_size = self._calculate_encoded_size(332, 324)
        
        # Multi-view cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        
        # Positional encoding for views
        self.view_pos_encoding = nn.Parameter(torch.randn(1, num_views, 512))
        
        # To point cloud latent
        self.to_mean = nn.Linear(512, latent_dim)
        self.to_logvar = nn.Linear(512, latent_dim)
        
        # Point cloud decoder
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, point_cloud_size * 3)  # xyz coordinates
        )
        
        # Feature decoder  
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, 256),  # latent + xyz
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_channels)  # spectral features
        )
        
    def _calculate_encoded_size(self, h: int, w: int) -> int:
        """Calculate size after 4 stride-2 convolutions"""
        for _ in range(4):
            h = (h + 1) // 2
            w = (w + 1) // 2
        return h * w
    
    def encode(self, hsi_views: torch.Tensor, camera_poses: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multi-view HSI to latent distribution
        
        Args:
            hsi_views: [B, V, C, H, W] multi-view hyperspectral images
            camera_poses: [B, V, 4, 4] camera matrices (optional)
            
        Returns:
            mean: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        B, V, C, H, W = hsi_views.shape
        
        # Step 1: Spectral encoding for each view
        spectral_features = []
        for v in range(V):
            view = hsi_views[:, v]  # [B, C, H, W]
            encoded = self.spectral_encoder(view)  # [B, 16, H, W]
            spectral_features.append(encoded)
        
        spectral_features = torch.stack(spectral_features, dim=1)  # [B, V, 16, H, W]
        
        # Step 2: Spatial encoding for each view
        spatial_features = []
        for v in range(V):
            view_feat = spectral_features[:, v]  # [B, 16, H, W]
            spatial = self.spatial_encoder(view_feat)  # [B, 512, H', W']
            B, D, H_enc, W_enc = spatial.shape
            spatial_flat = spatial.view(B, D, -1).mean(dim=2)  # [B, 512]
            spatial_features.append(spatial_flat)
            
        spatial_features = torch.stack(spatial_features, dim=1)  # [B, V, 512]
        
        # Step 3: Add positional encoding
        # Dynamically create positional encoding for the actual number of views
        B, V, D = spatial_features.shape
        if V != self.num_views:
            # Create positional encoding for actual number of views
            pos_encoding = nn.Parameter(torch.randn(1, V, D, device=spatial_features.device))
            spatial_features = spatial_features + pos_encoding
        else:
            spatial_features = spatial_features + self.view_pos_encoding
        
        # Step 4: Cross-attention across views
        attended, _ = self.cross_attention(
            spatial_features, spatial_features, spatial_features
        )  # [B, V, 512]
        
        # Step 5: Aggregate views
        aggregated = attended.mean(dim=1)  # [B, 512]
        
        # Step 6: To latent distribution
        mean = self.to_mean(aggregated)
        logvar = self.to_logvar(aggregated)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent to point cloud with spectral features
        
        Args:
            z: [B, latent_dim] latent code
            
        Returns:
            Dictionary with 'points' and 'spectral_features'
        """
        B = z.shape[0]
        
        # Decode to point cloud positions
        points = self.point_decoder(z)  # [B, point_cloud_size * 3]
        points = points.view(B, self.point_cloud_size, 3)
        
        # Decode spectral features for each point
        # Concatenate latent with point positions
        z_expanded = z.unsqueeze(1).expand(-1, self.point_cloud_size, -1)  # [B, N, latent_dim]
        point_latent = torch.cat([z_expanded, points], dim=-1)  # [B, N, latent_dim + 3]
        
        spectral_features = self.feature_decoder(point_latent.view(-1, self.latent_dim + 3))
        spectral_features = spectral_features.view(B, self.point_cloud_size, self.num_channels)
        spectral_features = F.softplus(spectral_features)  # Ensure positive radiance
        
        return {
            'points': points,
            'spectral_features': spectral_features
        }
    
    def forward(self, hsi_views: torch.Tensor, camera_poses: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Full forward pass"""
        # Encode
        mean, logvar = self.encode(hsi_views, camera_poses)
        
        # Reparameterize
        z = self.reparameterize(mean, logvar)
        
        # Decode
        decoded = self.decode(z)
        
        # Add distribution parameters
        decoded['mean'] = mean
        decoded['logvar'] = logvar
        decoded['latent'] = z
        
        return decoded