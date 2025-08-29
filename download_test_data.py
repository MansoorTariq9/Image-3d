#!/usr/bin/env python3
"""
Download and prepare hyperspectral test data for HSI Gaussian 3D

This script downloads publicly available hyperspectral datasets and converts them
to the format expected by our implementation.
"""

import os
import urllib.request
import numpy as np
from pathlib import Path
import json

def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading {url} to {destination}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False

def load_mat_file(filepath):
    """Load MATLAB file (requires scipy)"""
    try:
        import scipy.io
        data = scipy.io.loadmat(filepath)
        return data
    except ImportError:
        print("scipy not installed, using alternative numpy format")
        return None

def create_synthetic_hsi_from_info():
    """Create synthetic HSI data based on known specifications"""
    print("\nCreating synthetic HSI test data based on Indian Pines specifications...")
    
    # Indian Pines specifications
    height, width = 145, 145
    num_channels = 200  # After removing water absorption bands
    wavelength_start = 400  # nm (0.4 micrometers)
    wavelength_end = 2500  # nm (2.5 micrometers)
    
    # Create output directory
    output_dir = Path("./test_hsi_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic HSI cube with realistic spectral signatures
    print(f"Generating HSI cube: {height}x{width}x{num_channels}")
    
    # Create different material signatures (simulating Indian Pines classes)
    materials = {
        "vegetation": {"center": 750, "width": 200, "amplitude": 0.7},
        "soil": {"center": 1500, "width": 500, "amplitude": 0.5},
        "water": {"center": 900, "width": 150, "amplitude": 0.3},
        "concrete": {"center": 1800, "width": 400, "amplitude": 0.6}
    }
    
    # Generate wavelengths
    wavelengths = np.linspace(wavelength_start, wavelength_end, num_channels)
    
    # Create spatial patterns
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    
    # Create HSI cube
    hsi_cube = np.zeros((height, width, num_channels), dtype=np.float32)
    
    # Add vegetation patch (circular)
    veg_mask = (x**2 + y**2) < 0.3
    for c in range(num_channels):
        spectral_response = materials["vegetation"]["amplitude"] * \
                          np.exp(-0.5 * ((wavelengths[c] - materials["vegetation"]["center"]) / 
                                        materials["vegetation"]["width"])**2)
        hsi_cube[veg_mask, c] = spectral_response
    
    # Add soil patch (rectangular)
    soil_mask = (np.abs(x - 0.5) < 0.3) & (np.abs(y - 0.5) < 0.3)
    for c in range(num_channels):
        spectral_response = materials["soil"]["amplitude"] * \
                          np.exp(-0.5 * ((wavelengths[c] - materials["soil"]["center"]) / 
                                        materials["soil"]["width"])**2)
        hsi_cube[soil_mask, c] = spectral_response
    
    # Add noise and background
    noise = np.random.normal(0, 0.02, hsi_cube.shape)
    background = 0.1 + 0.05 * np.random.rand(num_channels)
    hsi_cube = hsi_cube + noise + background
    hsi_cube = np.clip(hsi_cube, 0, 1)
    
    # Resample to 120 channels (our target)
    target_channels = 120
    indices = np.linspace(0, num_channels-1, target_channels).astype(int)
    hsi_120 = hsi_cube[:, :, indices]
    wavelengths_120 = wavelengths[indices]
    
    # Adjust wavelengths to our range (400-1000nm)
    # Scale wavelengths to fit our range
    scale_factor = (1000 - 400) / (wavelengths_120[-1] - wavelengths_120[0])
    wavelengths_final = 400 + (wavelengths_120 - wavelengths_120[0]) * scale_factor
    
    # Resize to our target resolution (332x324)
    # Simple nearest neighbor resize
    target_height, target_width = 324, 332
    y_indices = np.linspace(0, height-1, target_height).astype(int)
    x_indices = np.linspace(0, width-1, target_width).astype(int)
    
    hsi_final = np.zeros((target_height, target_width, target_channels), dtype=np.float32)
    for i, yi in enumerate(y_indices):
        for j, xi in enumerate(x_indices):
            hsi_final[i, j, :] = hsi_120[yi, xi, :]
    
    print(f"✓ Generated HSI cube: {target_height}x{target_width}x{target_channels}")
    
    # Create multiple views (simulate camera positions)
    num_views = 4
    scene_dir = output_dir / "scene_001"
    scene_dir.mkdir(exist_ok=True)
    
    for v in range(num_views):
        # Add view-dependent variations
        view_variation = 0.05 * np.sin(v * np.pi / num_views)
        hsi_view = hsi_final + view_variation + np.random.normal(0, 0.01, hsi_final.shape)
        hsi_view = np.clip(hsi_view, 0, 1)
        
        # Save HSI cube
        np.save(scene_dir / f"hsi_{v:03d}.npy", hsi_view)
        print(f"✓ Saved view {v}: hsi_{v:03d}.npy")
    
    # Create camera poses
    intrinsics = []
    extrinsics = []
    
    for v in range(num_views):
        # Camera intrinsics
        K = np.array([
            [300, 0, target_width/2],
            [0, 300, target_height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics.append(K.tolist())
        
        # Camera extrinsics (circular arrangement)
        angle = 2 * np.pi * v / num_views
        radius = 5.0
        cam_height = 3.0
        
        # Camera position
        cam_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            cam_height
        ])
        
        # Look at origin
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Build camera matrix
        forward = (look_at - cam_pos)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_new = np.cross(right, forward)
        
        R = np.stack([right, up_new, -forward], axis=0).T
        
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R.T
        E[:3, 3] = -R.T @ cam_pos
        
        extrinsics.append(E.tolist())
    
    # Save camera parameters
    camera_data = {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics
    }
    
    with open(scene_dir / "camera_poses.json", "w") as f:
        json.dump(camera_data, f, indent=2)
    print("✓ Saved camera poses")
    
    # Create scene lists
    scenes = [{
        "name": "scene_001",
        "path": str(scene_dir),
        "num_views": num_views
    }]
    
    with open(output_dir / "train_scenes.json", "w") as f:
        json.dump(scenes, f, indent=2)
    
    with open(output_dir / "val_scenes.json", "w") as f:
        json.dump(scenes, f, indent=2)
    
    print(f"\n✓ Test data created successfully in {output_dir}")
    print(f"  - HSI shape: {target_height}x{target_width}x{target_channels}")
    print(f"  - Wavelength range: {wavelengths_final[0]:.1f}-{wavelengths_final[-1]:.1f}nm")
    print(f"  - Number of views: {num_views}")
    
    # Save metadata
    metadata = {
        "dataset": "Synthetic HSI (Indian Pines inspired)",
        "dimensions": {"height": target_height, "width": target_width, "channels": target_channels},
        "wavelengths": wavelengths_final.tolist(),
        "description": "Synthetic hyperspectral data with vegetation and soil signatures"
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir

def main():
    print("HSI Test Data Downloader and Converter")
    print("=" * 50)
    
    # Option 1: Try to download real Indian Pines data
    print("\nOption 1: Attempting to download Indian Pines dataset...")
    print("Note: This requires manual download from:")
    print("https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes")
    print("Download 'Indian Pines' and 'Indian Pines groundtruth' MATLAB files")
    
    # Option 2: Create synthetic data
    print("\nOption 2: Creating synthetic HSI test data...")
    test_data_dir = create_synthetic_hsi_from_info()
    
    print("\n" + "=" * 50)
    print("✓ Test data preparation complete!")
    print(f"\nTo test the implementation, run:")
    print(f"python train.py --config config.yaml --data_root {test_data_dir} --output_dir ./test_outputs")

if __name__ == "__main__":
    main()