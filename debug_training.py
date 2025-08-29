#!/usr/bin/env python3
"""Debug training pipeline step by step"""

import torch
import yaml
from pathlib import Path

# Import our modules  
from data.dataset import MultiViewHSIDataset, HSIConfig
from core.hsi_vae import HyperspectralVAE
from core.spectral_gaussian import SpectralGaussian3D, SpectralGaussianData
from core.spectral_renderer import SpectralGaussianRenderer, RenderingConfig
from train import HSIGaussian3DModel

def debug_step_by_step():
    """Debug training step by step"""
    
    print("Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("\n1. Testing dataset loading...")
    hsi_config = HSIConfig()
    dataset = MultiViewHSIDataset(
        data_root="processed_data",
        split="train",
        num_views=config["num_views"],
        config=hsi_config,
        augment=False
    )
    
    if len(dataset) > 0:
        batch = dataset[0]
        print(f"✓ Loaded batch with shape: {batch['hsi'].shape}")
        
        # Add batch dimension
        for key in ['hsi', 'intrinsics', 'extrinsics']:
            batch[key] = batch[key].unsqueeze(0)
        
        print("\n2. Testing model initialization...")
        model = HSIGaussian3DModel(config)
        print(f"✓ Model initialized")
        
        print("\n3. Testing forward pass...")
        try:
            outputs = model(batch)
            print(f"✓ Forward pass successful")
            print(f"  Output keys: {list(outputs.keys())}")
            
            print("\n4. Testing loss computation...")
            losses = model.compute_loss(outputs)
            print(f"✓ Loss computation successful")
            for k, v in losses.items():
                print(f"  {k}: {v.item():.4f}")
                
        except Exception as e:
            print(f"✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("✗ No data found in dataset")

def test_individual_components():
    """Test each component individually"""
    
    print("\n" + "="*50)
    print("Testing Individual Components")
    print("="*50)
    
    # Test VAE
    print("\n1. Testing VAE...")
    vae = HyperspectralVAE(num_channels=120)
    dummy_input = torch.randn(1, 4, 120, 324, 332)
    try:
        vae_output = vae(dummy_input)
        print(f"✓ VAE output keys: {list(vae_output.keys())}")
    except Exception as e:
        print(f"✗ VAE error: {e}")
    
    # Test Gaussian Model
    print("\n2. Testing Gaussian Model...")
    gaussian = SpectralGaussian3D(num_points=1000, num_channels=120)
    try:
        params = gaussian({"camera": torch.eye(3)})
        print(f"✓ Gaussian params keys: {list(params.keys())}")
        
        # Test creating SpectralGaussianData
        data = SpectralGaussianData(**params)
        print(f"✓ SpectralGaussianData created")
    except Exception as e:
        print(f"✗ Gaussian model error: {e}")
    
    # Test Renderer
    print("\n3. Testing Renderer...")
    renderer = SpectralGaussianRenderer(RenderingConfig(
        image_height=324,
        image_width=332,
        background_value=1.0
    ))
    
    try:
        # Create dummy data
        gaussian_data = SpectralGaussianData(
            positions=torch.randn(100, 3) * 0.1,
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 100),
            scales=torch.ones(100, 3) * 0.01,
            opacities=torch.ones(100, 1) * 0.8,
            spectral_features=torch.rand(100, 120),
            wavelengths=torch.linspace(400, 1000, 120)
        )
        
        camera_params = {
            'intrinsics': torch.tensor([[
                [300.0, 0.0, 166.0],
                [0.0, 300.0, 162.0],
                [0.0, 0.0, 1.0]
            ]]),
            'extrinsics': torch.eye(4).unsqueeze(0)
        }
        
        output = renderer(gaussian_data, camera_params)
        print(f"✓ Renderer output shape: {output['image'].shape}")
    except Exception as e:
        print(f"✗ Renderer error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run debug tests"""
    test_individual_components()
    print("\n" + "="*50)
    print("Testing Full Pipeline")
    print("="*50)
    debug_step_by_step()

if __name__ == "__main__":
    main()