#!/usr/bin/env python3
"""Test the full pipeline with processed sample data"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from data.dataset import MultiViewHSIDataset, HSIConfig
from core.hsi_vae import HyperspectralVAE
from core.spectral_gaussian import SpectralGaussian3D
from core.spectral_renderer import SpectralGaussianRenderer, RenderingConfig

def test_data_loading():
    """Test loading processed ENVI data"""
    print("Testing data loading...")
    
    # Create dataset
    config = HSIConfig()
    dataset = MultiViewHSIDataset(
        data_root="./processed_data",
        split="train",
        num_views=4,
        config=config,
        augment=False,
        cache_preprocessed=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"HSI shape: {sample['hsi'].shape}")
        print(f"Intrinsics shape: {sample['intrinsics'].shape}")
        print(f"Extrinsics shape: {sample['extrinsics'].shape}")
        print(f"Wavelengths shape: {sample['wavelengths'].shape}")
        
        # Visualize a few channels
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        channels_to_show = [0, 40, 80, 119]  # Show different wavelengths
        
        for i, ch in enumerate(channels_to_show):
            img = sample['hsi'][0, ch].numpy()  # First view, specific channel
            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f"λ={sample['wavelengths'][ch]:.0f}nm")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig('test_data_channels.png')
        print("Saved visualization to test_data_channels.png")
    
    return dataset

def test_vae():
    """Test VAE forward pass"""
    print("\nTesting VAE...")
    
    # Create dummy input
    B, V, C, H, W = 1, 4, 120, 324, 332
    dummy_input = torch.randn(B, V, C, H, W)
    
    # Initialize VAE
    vae = HyperspectralVAE(
        num_channels=120,
        latent_dim=512,
        num_views=4,
        point_cloud_size=2048
    )
    
    # Test encoding
    mean, logvar = vae.encode(dummy_input)
    print(f"VAE encoding - Mean shape: {mean.shape}, Logvar shape: {logvar.shape}")
    
    # Test reparameterization
    z = vae.reparameterize(mean, logvar)
    print(f"Latent shape: {z.shape}")
    
    # Test decoding
    output = vae.decode(z)
    print(f"Decoded points shape: {output['points'].shape}")
    print(f"Decoded spectral features shape: {output['spectral_features'].shape}")
    
    return vae

def test_gaussian_model():
    """Test spectral Gaussian model"""
    print("\nTesting Spectral Gaussian model...")
    
    # Initialize model
    gaussian = SpectralGaussian3D(
        num_points=5000,
        num_channels=120,
        wavelength_range=(400.0, 1000.0)
    )
    
    # Test properties
    print(f"Positions shape: {gaussian.positions.shape}")
    print(f"Rotations shape: {gaussian.rotations.shape}")
    print(f"Scales shape: {gaussian.scales.shape}")
    print(f"Opacities shape: {gaussian.opacities.shape}")
    print(f"Spectral features shape: {gaussian.spectral_features.shape}")
    print(f"Wavelengths shape: {gaussian.wavelengths.shape}")
    
    # Test rendering at specific wavelength
    wavelength = 550.0  # Green light
    colors = gaussian.get_spectral_colors(wavelength)
    print(f"Colors at {wavelength}nm shape: {colors.shape}")
    
    return gaussian

def test_renderer():
    """Test spectral renderer"""
    print("\nTesting Spectral Renderer...")
    
    # Create renderer
    config = RenderingConfig(
        image_height=324,
        image_width=332,
        background_value=1.0
    )
    renderer = SpectralGaussianRenderer(config)
    
    # Create dummy Gaussian data
    from core.spectral_gaussian import SpectralGaussianData
    
    N = 1000
    gaussian_data = SpectralGaussianData(
        positions=torch.randn(N, 3) * 0.5,
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N),  # Identity quaternions
        scales=torch.ones(N, 3) * 0.01,
        opacities=torch.ones(N, 1) * 0.8,
        spectral_features=torch.rand(N, 120),
        wavelengths=torch.linspace(400, 1000, 120)
    )
    
    # Create dummy camera
    intrinsics = torch.tensor([[
        [300.0, 0.0, 166.0],
        [0.0, 300.0, 162.0],
        [0.0, 0.0, 1.0]
    ]])
    
    extrinsics = torch.eye(4).unsqueeze(0)
    
    # Test rendering
    print("Testing single wavelength rendering...")
    rendered = renderer.render_single_wavelength(
        gaussian_data, intrinsics[0], extrinsics[0], wavelength=550.0
    )
    print(f"Rendered image shape: {rendered['image'].shape}")
    print(f"Rendered depth shape: {rendered['depth'].shape}")
    
    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rendered['image'].squeeze().cpu().numpy(), cmap='viridis')
    plt.title('Rendered at 550nm')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(rendered['depth'].squeeze().cpu().numpy(), cmap='plasma')
    plt.title('Depth map')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('test_rendering.png')
    print("Saved rendering to test_rendering.png")
    
    return renderer

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing HSI Gaussian 3D Pipeline")
    print("=" * 50)
    
    # Test each component
    dataset = test_data_loading()
    vae = test_vae()
    gaussian = test_gaussian_model()
    renderer = test_renderer()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
    
    # Summary
    print("\nSummary:")
    print(f"✓ Data loading: {len(dataset)} scenes found")
    print("✓ VAE: Forward pass successful")
    print("✓ Gaussian model: Initialized with spectral features")
    print("✓ Renderer: Successfully rendered test scene")
    
    print("\nNext steps:")
    print("1. Run full training: python train.py --data_root processed_data")
    print("2. Integrate SuperGlue for better pose estimation")
    print("3. Fine-tune hyperparameters for your specific data")

if __name__ == "__main__":
    main()