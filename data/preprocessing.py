import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import cv2

@dataclass
class HSIConfig:
    """Configuration for HSI data processing"""
    num_channels: int = 120
    wavelength_range: Tuple[float, float] = (400.0, 1000.0)
    image_shape: Tuple[int, int] = (332, 324)  # width, height
    background_method: str = "percentile"  # "corners", "percentile", "fixed"
    normalization_method: str = "percentile"  # "percentile", "minmax", "standardize"
    spectral_smoothing: bool = True
    smoothing_kernel_size: int = 3

class HSIPreprocessor:
    """Preprocessing pipeline for hyperspectral images without calibration"""
    
    def __init__(self, config: HSIConfig):
        self.config = config
        self.wavelengths = np.linspace(
            config.wavelength_range[0],
            config.wavelength_range[1],
            config.num_channels
        )
        
    def estimate_background(self, hsi_cube: np.ndarray) -> np.ndarray:
        """Estimate background spectrum
        
        Args:
            hsi_cube: [H, W, C] hyperspectral image
            
        Returns:
            background: [C] estimated background spectrum
        """
        H, W, C = hsi_cube.shape
        
        if self.config.background_method == "corners":
            # Use corners of the image
            corner_size = min(H, W) // 10
            corners = [
                hsi_cube[:corner_size, :corner_size, :],
                hsi_cube[:corner_size, -corner_size:, :],
                hsi_cube[-corner_size:, :corner_size, :],
                hsi_cube[-corner_size:, -corner_size:, :]
            ]
            # Take the brightest corner as background
            corner_means = [np.mean(corner, axis=(0, 1)) for corner in corners]
            background = corner_means[np.argmax([np.mean(cm) for cm in corner_means])]
            
        elif self.config.background_method == "percentile":
            # Use high percentile across the image
            background = np.percentile(hsi_cube, 95, axis=(0, 1))
            
        else:  # "fixed"
            # Use fixed value of 1.0 (HS-NeRF approach)
            background = np.ones(C)
            
        # Ensure no zero values
        background = np.maximum(background, 1e-6)
        return background
    
    def normalize_channels(self, hsi_cube: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize each spectral channel
        
        Returns:
            normalized: Normalized HSI cube
            stats: Normalization statistics for denormalization
        """
        normalized = np.zeros_like(hsi_cube, dtype=np.float32)
        stats = {"method": self.config.normalization_method}
        
        if self.config.normalization_method == "percentile":
            # Percentile normalization per channel
            p_low, p_high = 2, 98
            channel_stats = []
            
            for c in range(self.config.num_channels):
                channel_data = hsi_cube[:, :, c]
                low = np.percentile(channel_data, p_low)
                high = np.percentile(channel_data, p_high)
                
                # Avoid division by zero
                if high - low > 1e-6:
                    normalized[:, :, c] = np.clip((channel_data - low) / (high - low), 0, 1)
                else:
                    normalized[:, :, c] = 0.5
                    
                channel_stats.append({"low": low, "high": high})
                
            stats["channel_stats"] = channel_stats
            
        elif self.config.normalization_method == "minmax":
            # Min-max normalization
            for c in range(self.config.num_channels):
                channel_data = hsi_cube[:, :, c]
                min_val = channel_data.min()
                max_val = channel_data.max()
                
                if max_val - min_val > 1e-6:
                    normalized[:, :, c] = (channel_data - min_val) / (max_val - min_val)
                else:
                    normalized[:, :, c] = 0.5
                    
        elif self.config.normalization_method == "standardize":
            # Z-score normalization
            for c in range(self.config.num_channels):
                channel_data = hsi_cube[:, :, c]
                mean = channel_data.mean()
                std = channel_data.std()
                
                if std > 1e-6:
                    normalized[:, :, c] = (channel_data - mean) / std
                else:
                    normalized[:, :, c] = 0.0
                    
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            # Scale to [0, 1]
            normalized = (normalized + 3) / 6
            
        return normalized, stats
    
    def smooth_spectral_dimension(self, hsi_cube: np.ndarray) -> np.ndarray:
        """Apply smoothing along spectral dimension
        
        Args:
            hsi_cube: [H, W, C] hyperspectral image
            
        Returns:
            smoothed: Smoothed HSI cube
        """
        if not self.config.spectral_smoothing:
            return hsi_cube
            
        kernel_size = self.config.smoothing_kernel_size
        pad = kernel_size // 2
        smoothed = np.zeros_like(hsi_cube)
        
        for c in range(self.config.num_channels):
            start = max(0, c - pad)
            end = min(self.config.num_channels, c + pad + 1)
            smoothed[:, :, c] = np.mean(hsi_cube[:, :, start:end], axis=2)
            
        return smoothed
    
    def preprocess(self, hsi_cube: np.ndarray) -> Dict[str, torch.Tensor]:
        """Complete preprocessing pipeline
        
        Args:
            hsi_cube: [H, W, C] raw hyperspectral image
            
        Returns:
            Dictionary with processed data
        """
        # Ensure float32
        hsi_cube = hsi_cube.astype(np.float32)
        
        # Step 1: Background correction
        background = self.estimate_background(hsi_cube)
        hsi_corrected = hsi_cube / background[np.newaxis, np.newaxis, :]
        
        # Step 2: Spectral smoothing
        hsi_smoothed = self.smooth_spectral_dimension(hsi_corrected)
        
        # Step 3: Normalization
        hsi_normalized, norm_stats = self.normalize_channels(hsi_smoothed)
        
        # Step 4: Convert to torch tensor
        # PyTorch expects [C, H, W] format
        tensor = torch.from_numpy(hsi_normalized).permute(2, 0, 1)
        
        # Create wavelength tensor
        wavelengths = torch.tensor(self.wavelengths, dtype=torch.float32)
        
        return {
            "hsi": tensor,  # [C, H, W]
            "wavelengths": wavelengths,  # [C]
            "background": torch.from_numpy(background),  # [C]
            "norm_stats": norm_stats,
            "original_shape": hsi_cube.shape
        }
    
    def preprocess_batch(self, hsi_batch: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """Process multiple HSI images
        
        Args:
            hsi_batch: List of [H, W, C] HSI cubes
            
        Returns:
            Batched tensors
        """
        processed = [self.preprocess(hsi) for hsi in hsi_batch]
        
        # Stack into batch
        batch_data = {
            "hsi": torch.stack([p["hsi"] for p in processed]),  # [B, C, H, W]
            "wavelengths": processed[0]["wavelengths"],  # [C]
            "background": torch.stack([p["background"] for p in processed]),  # [B, C]
        }
        
        return batch_data

class SpectralAugmentation:
    """Data augmentation for hyperspectral images"""
    
    def __init__(self, 
                 spectral_shift_range: float = 5.0,  # nm
                 intensity_scale_range: Tuple[float, float] = (0.8, 1.2),
                 noise_level: float = 0.01):
        self.spectral_shift_range = spectral_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.noise_level = noise_level
        
    def apply_spectral_shift(self, hsi: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        """Simulate spectral shift"""
        # Random shift in nm
        shift = torch.randint(-int(self.spectral_shift_range), 
                             int(self.spectral_shift_range) + 1, (1,)).item()
        
        if shift == 0:
            return hsi
            
        # Shift channels
        if shift > 0:
            # Shift right (towards longer wavelengths)
            shifted = torch.cat([hsi[:shift], hsi[:-shift]], dim=0)
        else:
            # Shift left (towards shorter wavelengths)
            shifted = torch.cat([hsi[-shift:], hsi[:-shift]], dim=0)
            
        return shifted
    
    def apply_intensity_scaling(self, hsi: torch.Tensor) -> torch.Tensor:
        """Random intensity scaling per channel"""
        C = hsi.shape[0]
        scale = torch.rand(C, 1, 1) * (self.intensity_scale_range[1] - self.intensity_scale_range[0])
        scale += self.intensity_scale_range[0]
        return hsi * scale
    
    def apply_noise(self, hsi: torch.Tensor) -> torch.Tensor:
        """Add random noise"""
        noise = torch.randn_like(hsi) * self.noise_level
        return hsi + noise
    
    def __call__(self, hsi: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        """Apply augmentations"""
        # Apply spectral shift
        hsi = self.apply_spectral_shift(hsi, wavelengths)
        
        # Apply intensity scaling
        hsi = self.apply_intensity_scaling(hsi)
        
        # Add noise
        hsi = self.apply_noise(hsi)
        
        # Ensure valid range
        hsi = torch.clamp(hsi, 0, 1)
        
        return hsi