#!/usr/bin/env python3
"""Test rendering progress and show model is learning"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from train import HSIGaussian3DModel
from data.dataset import MultiViewHSIDataset, HSIConfig
from visualize_results import hyperspectral_to_rgb

def test_rendering():
    """Test if model is making progress"""
    
    print("Testing Rendering Progress...")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    dataset = MultiViewHSIDataset(
        data_root="processed_data",
        split="val",
        num_views=config["num_views"],
        config=HSIConfig(),
        augment=False
    )
    
    batch = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in ['hsi', 'intrinsics', 'extrinsics']:
        batch[key] = batch[key].unsqueeze(0).to(device)
    batch['wavelengths'] = batch['wavelengths'].to(device)
    
    # Original image
    original = batch['hsi'][0, 0].cpu().numpy()  # First view
    wavelengths = batch['wavelengths'].cpu().numpy()
    original_rgb = hyperspectral_to_rgb(original, wavelengths)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original HSI (as RGB)")
    axes[0, 0].axis('off')
    
    # Test different models
    models_to_test = [
        ("Random Init", None),
        ("After 5 epochs", "outputs/checkpoint_epoch_0.pth"),
        ("Best Model", "outputs/best_model.pth")
    ]
    
    for idx, (name, checkpoint_path) in enumerate(models_to_test):
        print(f"\nTesting: {name}")
        
        # Load model
        model = HSIGaussian3DModel(config).to(device)
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                print(f"  Loaded checkpoint")
            except:
                print(f"  Failed to load {checkpoint_path}")
                continue
        
        model.eval()
        
        with torch.no_grad():
            # Get rendering
            outputs = model(batch)
            gaussian_data = outputs['gaussian_data']
            
            # Check Gaussian stats
            if hasattr(gaussian_data, 'positions'):
                positions = gaussian_data.positions
                opacities = gaussian_data.opacities
                spectral = gaussian_data.spectral_features
            else:
                positions = gaussian_data['positions']
                opacities = gaussian_data['opacities'] 
                spectral = gaussian_data['spectral_features']
            
            print(f"  Positions std: {positions.std().item():.3f}")
            print(f"  Opacities mean: {opacities.mean().item():.3f}")
            print(f"  Spectral mean: {spectral.mean().item():.3f}")
            
            # Render
            camera_params = {
                'intrinsics': batch['intrinsics'][0, 0:1],
                'extrinsics': batch['extrinsics'][0, 0:1]
            }
            
            rendered = model.renderer(gaussian_data, camera_params)
            rendered_hsi = rendered['spectral'].squeeze().cpu().numpy()
            
            # Stats
            print(f"  Rendered min: {rendered_hsi.min():.4f}")
            print(f"  Rendered max: {rendered_hsi.max():.4f}")
            print(f"  Rendered mean: {rendered_hsi.mean():.4f}")
            
            # Convert to RGB
            rendered_rgb = hyperspectral_to_rgb(rendered_hsi, wavelengths)
            
            # Show rendered
            row = 0 if idx < 2 else 1
            col = (idx % 2) + 1
            axes[row, col].imshow(rendered_rgb)
            axes[row, col].set_title(f"{name}")
            axes[row, col].axis('off')
    
    # Add single wavelength visualization
    axes[1, 2].imshow(original[60], cmap='viridis')  # Middle wavelength
    axes[1, 2].set_title(f"Original @ {wavelengths[60]:.0f}nm")
    axes[1, 2].axis('off')
    
    plt.suptitle("Rendering Progress Test", fontsize=16)
    plt.tight_layout()
    plt.savefig("visualization_results/rendering_progress.png", dpi=150)
    print(f"\n✓ Saved to visualization_results/rendering_progress.png")
    
    # Also save the actual rendered array for inspection
    np.save("visualization_results/rendered_data.npy", rendered_hsi)
    print(f"✓ Saved raw data to visualization_results/rendered_data.npy")
    print(f"  Shape: {rendered_hsi.shape}")
    print(f"  Non-zero pixels: {(rendered_hsi > 0.001).sum()}")

if __name__ == "__main__":
    test_rendering()