#!/usr/bin/env python3
"""Verify the HSI Gaussian 3D implementation structure"""

import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def verify_implementation():
    """Verify all components of the implementation"""
    print("HSI Gaussian 3D Implementation Verification")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    all_good = True
    
    # Check core modules
    print("\n1. Core Modules:")
    core_files = {
        "core/spectral_gaussian.py": "Spectral Gaussian Model",
        "core/hsi_vae.py": "Hyperspectral VAE",
        "core/spectral_renderer.py": "Spectral Renderer",
        "core/depth_estimator.py": "Depth Estimator",
        "core/__init__.py": "Core package init"
    }
    
    for filepath, desc in core_files.items():
        if not check_file_exists(base_dir / filepath, desc):
            all_good = False
    
    # Check data modules
    print("\n2. Data Modules:")
    data_files = {
        "data/preprocessing.py": "HSI Preprocessing",
        "data/dataset.py": "Dataset Loader",
        "data/__init__.py": "Data package init"
    }
    
    for filepath, desc in data_files.items():
        if not check_file_exists(base_dir / filepath, desc):
            all_good = False
    
    # Check training/inference scripts
    print("\n3. Main Scripts:")
    main_files = {
        "train.py": "Training Script",
        "inference.py": "Inference Script",
        "generate_sample_data.py": "Sample Data Generator"
    }
    
    for filepath, desc in main_files.items():
        if not check_file_exists(base_dir / filepath, desc):
            all_good = False
    
    # Check configuration files
    print("\n4. Configuration Files:")
    config_files = {
        "config.yaml": "Model Configuration",
        "requirements.txt": "Dependencies",
        "README.md": "Documentation",
        "__init__.py": "Package init"
    }
    
    for filepath, desc in config_files.items():
        if not check_file_exists(base_dir / filepath, desc):
            all_good = False
    
    # Verify key implementations
    print("\n5. Key Implementation Details:")
    
    # Check spectral gaussian implementation
    spectral_gaussian_file = base_dir / "core/spectral_gaussian.py"
    if spectral_gaussian_file.exists():
        with open(spectral_gaussian_file, 'r') as f:
            content = f.read()
            checks = [
                ("120 spectral channels", "num_channels: int = 120" in content),
                ("Wavelength range 400-1000nm", "wavelength_range: Tuple[float, float] = (400.0, 1000.0)" in content),
                ("Spectral smoothness loss", "spectral_smoothness_loss" in content),
                ("Wavelength interpolation", "get_spectral_radiance" in content)
            ]
            
            for desc, found in checks:
                status = "✓" if found else "✗"
                print(f"  {status} {desc}")
                if not found:
                    all_good = False
    
    # Check VAE adaptation
    vae_file = base_dir / "core/hsi_vae.py"
    if vae_file.exists():
        with open(vae_file, 'r') as f:
            content = f.read()
            checks = [
                ("120 channel input", "num_channels: int = 120" in content),
                ("Spectral encoder", "SpectralEncoder" in content),
                ("Multi-view cross-attention", "cross_attention" in content),
                ("Point cloud output", "point_cloud_size" in content)
            ]
            
            for desc, found in checks:
                status = "✓" if found else "✗"
                print(f"  {status} {desc}")
                if not found:
                    all_good = False
    
    # Check preprocessing
    preprocess_file = base_dir / "data/preprocessing.py"
    if preprocess_file.exists():
        with open(preprocess_file, 'r') as f:
            content = f.read()
            checks = [
                ("No calibration handling", "estimate_background" in content),
                ("Percentile normalization", "normalize_percentile" in content),
                ("Spectral smoothing", "smooth_spectral_dimension" in content),
                ("332x324 resolution", "image_shape: Tuple[int, int] = (332, 324)" in content)
            ]
            
            for desc, found in checks:
                status = "✓" if found else "✗"
                print(f"  {status} {desc}")
                if not found:
                    all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✓ All components verified successfully!")
    else:
        print("✗ Some components are missing or incomplete")
    
    print("\nImplementation Summary:")
    print("- Adapts GaussianAnything for 120-channel HSI data")
    print("- Handles uncalibrated data (no white/dark references)")
    print("- Wavelength range: 400-1000nm")
    print("- Image resolution: 332x324")
    print("- Includes depth estimation from HSI")
    print("- Supports multi-view 3D reconstruction")

if __name__ == "__main__":
    verify_implementation()