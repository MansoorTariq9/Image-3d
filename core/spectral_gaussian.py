import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

@dataclass
class SpectralGaussianData:
    """Container for hyperspectral Gaussian splat data"""
    positions: torch.Tensor          # [N, 3] - 3D positions
    rotations: torch.Tensor          # [N, 4] - quaternions
    scales: torch.Tensor             # [N, 3] - 3D scales
    opacities: torch.Tensor          # [N, 1] - opacity values
    spectral_features: torch.Tensor  # [N, 120] - hyperspectral radiance
    wavelengths: torch.Tensor        # [120] - wavelength values in nm

class SpectralGaussian3D(nn.Module):
    """3D Gaussian Splatting with Hyperspectral Features"""
    
    def __init__(
        self,
        num_points: int = 50000,
        num_channels: int = 120,
        wavelength_range: Tuple[float, float] = (400.0, 1000.0),
        sh_degree: int = 3
    ):
        super().__init__()
        self.num_channels = num_channels
        self.wavelength_range = wavelength_range
        self.sh_degree = sh_degree
        
        # Wavelength array
        self.register_buffer(
            'wavelengths', 
            torch.linspace(wavelength_range[0], wavelength_range[1], num_channels)
        )
        
        # Gaussian parameters
        self._positions = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        self._rotations = nn.Parameter(self._init_rotations(num_points))
        self._scales = nn.Parameter(torch.ones(num_points, 3) * -2.0)  # log scale
        self._opacities = nn.Parameter(torch.zeros(num_points, 1))  # logit opacity
        
        # Spectral features with physical constraints
        self._spectral_features_raw = nn.Parameter(
            self._init_spectral_features(num_points, num_channels)
        )
        
        # Learnable spectral basis (for dimensionality reduction)
        self.spectral_basis = nn.Parameter(
            self._init_spectral_basis(num_channels, min(32, num_channels))
        )
        
    def _init_rotations(self, num_points: int) -> torch.Tensor:
        """Initialize quaternions to identity"""
        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0  # w component
        return quats
    
    def _init_spectral_features(self, num_points: int, num_channels: int) -> torch.Tensor:
        """Initialize with physically plausible spectral curves"""
        features = torch.zeros(num_points, num_channels)
        
        for i in range(num_points):
            # Create diverse spectral signatures
            n_peaks = torch.randint(1, 4, (1,)).item()
            
            for _ in range(n_peaks):
                peak_channel = torch.randint(0, num_channels, (1,)).item()
                width = torch.randint(10, 40, (1,)).item()
                amplitude = torch.rand(1).item() * 0.5
                
                channels = torch.arange(num_channels).float()
                gaussian = amplitude * torch.exp(-0.5 * ((channels - peak_channel) / width) ** 2)
                features[i] += gaussian
        
        return features
    
    def _init_spectral_basis(self, num_channels: int, num_basis: int) -> torch.Tensor:
        """Initialize spectral basis functions"""
        basis = torch.zeros(num_basis, num_channels)
        
        # Create smooth basis functions
        for i in range(num_basis):
            center = i * num_channels / num_basis
            width = num_channels / (2 * num_basis)
            channels = torch.arange(num_channels).float()
            basis[i] = torch.exp(-0.5 * ((channels - center) / width) ** 2)
            
        return basis
    
    @property
    def positions(self) -> torch.Tensor:
        return self._positions
    
    @property
    def rotations(self) -> torch.Tensor:
        """Normalize quaternions"""
        return F.normalize(self._rotations, p=2, dim=1)
    
    @property
    def scales(self) -> torch.Tensor:
        """Apply exponential activation"""
        return torch.exp(self._scales)
    
    @property
    def opacities(self) -> torch.Tensor:
        """Apply sigmoid activation"""
        return torch.sigmoid(self._opacities)
    
    @property
    def spectral_features(self) -> torch.Tensor:
        """Apply softplus for non-negative radiance"""
        return F.softplus(self._spectral_features_raw)
    
    def get_spectral_radiance(
        self, 
        wavelength_nm: Optional[float] = None,
        wavelength_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get radiance at specific wavelengths
        
        Args:
            wavelength_nm: Single wavelength in nm
            wavelength_indices: Tensor of channel indices
            
        Returns:
            Radiance values [N, ?] where ? depends on input
        """
        spectral = self.spectral_features
        
        if wavelength_nm is not None:
            # Interpolate for specific wavelength
            idx = torch.searchsorted(self.wavelengths, wavelength_nm)
            if idx == 0:
                return spectral[:, 0:1]
            elif idx >= self.num_channels:
                return spectral[:, -1:]
            else:
                # Linear interpolation
                w1 = self.wavelengths[idx - 1]
                w2 = self.wavelengths[idx]
                alpha = (wavelength_nm - w1) / (w2 - w1)
                return (1 - alpha) * spectral[:, idx-1:idx] + alpha * spectral[:, idx:idx+1]
                
        elif wavelength_indices is not None:
            return spectral[:, wavelength_indices]
        else:
            return spectral
    
    def spectral_smoothness_loss(self) -> torch.Tensor:
        """Regularization for smooth spectral curves"""
        spectral = self.spectral_features
        
        # First derivative (smoothness)
        diff1 = spectral[:, 1:] - spectral[:, :-1]
        loss_smooth = torch.mean(diff1 ** 2)
        
        # Second derivative (curvature)
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        loss_curve = torch.mean(diff2 ** 2)
        
        return loss_smooth + 0.1 * loss_curve
    
    def forward(self, camera_params: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass returning Gaussian parameters"""
        return {
            'positions': self.positions,
            'rotations': self.rotations,
            'scales': self.scales,
            'opacities': self.opacities,
            'spectral_features': self.spectral_features,
            'wavelengths': self.wavelengths
        }