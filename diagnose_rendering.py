#!/usr/bin/env python3
"""Diagnose why rendering is blank"""

import torch
import numpy as np
import yaml
from pathlib import Path

from train import HSIGaussian3DModel
from data.dataset import MultiViewHSIDataset, HSIConfig

def diagnose_rendering():
    """Diagnose rendering issues"""
    
    print("="*50)
    print("Diagnosing Rendering Issues")
    print("="*50)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSIGaussian3DModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load("outputs/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load data
    dataset = MultiViewHSIDataset(
        data_root="processed_data",
        split="val",
        num_views=config["num_views"],
        config=HSIConfig(),
        augment=False
    )
    
    batch = dataset[0]
    for key in ['hsi', 'intrinsics', 'extrinsics']:
        batch[key] = batch[key].unsqueeze(0).to(device)
    batch['wavelengths'] = batch['wavelengths'].to(device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(batch)
        gaussian_data = outputs['gaussian_data']
        
        # Check Gaussian parameters
        print("\n1. Gaussian Parameters:")
        if hasattr(gaussian_data, 'positions'):
            positions = gaussian_data.positions
            scales = gaussian_data.scales
            opacities = gaussian_data.opacities
            spectral_features = gaussian_data.spectral_features
        else:
            positions = gaussian_data['positions']
            scales = gaussian_data['scales']
            opacities = gaussian_data['opacities']
            spectral_features = gaussian_data['spectral_features']
        
        print(f"   Positions range: [{positions.min().item():.3f}, {positions.max().item():.3f}]")
        print(f"   Scales range: [{scales.min().item():.3f}, {scales.max().item():.3f}]")
        print(f"   Opacities range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]")
        print(f"   Spectral features range: [{spectral_features.min().item():.3f}, {spectral_features.max().item():.3f}]")
        
        # Check how many Gaussians are visible
        opacity_threshold = 0.01
        visible_gaussians = (opacities > opacity_threshold).sum().item()
        print(f"   Visible Gaussians (opacity > {opacity_threshold}): {visible_gaussians}/{opacities.shape[0]}")
        
        # Render and check output
        print("\n2. Rendering Output:")
        camera_params = {
            'intrinsics': batch['intrinsics'][0, 0:1],
            'extrinsics': batch['extrinsics'][0, 0:1]
        }
        
        rendered = model.renderer(gaussian_data, camera_params)
        rendered_spectral = rendered['spectral'].detach().cpu().numpy()
        
        print(f"   Rendered shape: {rendered_spectral.shape}")
        print(f"   Rendered range: [{rendered_spectral.min():.6f}, {rendered_spectral.max():.6f}]")
        print(f"   Rendered mean: {rendered_spectral.mean():.6f}")
        print(f"   Non-zero pixels: {(rendered_spectral > 0.001).sum()}")
        
        # Check if it's the background value
        background_value = config.get('background_value', 1.0)
        print(f"   Background value: {background_value}")
        all_background = np.allclose(rendered_spectral, background_value, atol=0.001)
        print(f"   All pixels are background: {all_background}")
        
        # Check loss values
        print("\n3. Training Loss Analysis:")
        print(f"   Final validation loss: {checkpoint['val_loss']:.4f}")
        
        # Suggestions
        print("\n4. Diagnosis:")
        if visible_gaussians == 0:
            print("   ❌ No visible Gaussians - opacities are too low")
            print("   → The model hasn't learned to make Gaussians visible yet")
        elif all_background:
            print("   ❌ All pixels show background - Gaussians not covering the image")
            print("   → Gaussians might be outside the view frustum or too small")
        else:
            print("   ⚠️  Rendering is producing very low values")
        
        print("\n5. Solutions:")
        print("   1. Train for more epochs (current: 5, recommended: 100+)")
        print("   2. Increase learning rate to 0.001 or 0.01")
        print("   3. Check if Gaussians are initialized properly")
        print("   4. Adjust opacity regularization in loss function")
        print("   5. Use a lower background value (e.g., 0.0 instead of 1.0)")

def quick_fix_attempt():
    """Attempt a quick fix by adjusting parameters"""
    
    print("\n" + "="*50)
    print("Attempting Quick Fix")
    print("="*50)
    
    # Load everything
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSIGaussian3DModel(config).to(device)
    
    checkpoint = torch.load("outputs/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Manually boost some parameters for visualization
    print("\nBoosting Gaussian parameters for visualization...")
    with torch.no_grad():
        # Increase opacities
        model.gaussian_model._opacities.data += 2.0  # Increase opacity
        
        # Ensure Gaussians are in view
        model.gaussian_model._positions.data *= 0.1  # Bring closer to origin
        
        # Increase scales slightly
        model.gaussian_model._scales.data += 0.5
    
    # Save modified model
    torch.save({
        'epoch': checkpoint['epoch'],
        'model': model.state_dict(),
        'optimizer': checkpoint['optimizer'],
        'config': checkpoint['config'],
        'val_loss': checkpoint['val_loss']
    }, 'outputs/best_model_boosted.pth')
    
    print("✓ Saved boosted model to outputs/best_model_boosted.pth")
    print("\nTry running visualization again with:")
    print("python visualize_results.py")
    print("(after changing checkpoint_path to 'outputs/best_model_boosted.pth')")

if __name__ == "__main__":
    diagnose_rendering()
    quick_fix_attempt()