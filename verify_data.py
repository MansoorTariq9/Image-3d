#!/usr/bin/env python3
"""Verify ENVI data conversion without full dependencies"""

import json
import numpy as np
from pathlib import Path
from data.envi_loader import ENVIReader

def verify_conversion():
    """Verify that the ENVI data was converted correctly"""
    
    print("Verifying data conversion...")
    
    # Check processed data directory
    processed_dir = Path("./processed_data")
    if not processed_dir.exists():
        print("❌ Processed data directory not found!")
        return False
    
    # Find scene directory
    scene_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
    if not scene_dirs:
        print("❌ No scene directories found!")
        return False
    
    scene_dir = scene_dirs[0]
    print(f"✓ Found scene directory: {scene_dir.name}")
    
    # Check for converted HSI files
    hsi_files = list(scene_dir.glob("hsi_*.npy"))
    print(f"✓ Found {len(hsi_files)} HSI view files")
    
    # Check camera poses
    camera_file = scene_dir / "camera_poses.json"
    if camera_file.exists():
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        print(f"✓ Camera poses loaded: {len(camera_data['angles'])} views")
        print(f"  Angles: {camera_data['angles']}")
    else:
        print("❌ Camera poses file not found!")
        return False
    
    # Verify HSI data format
    if hsi_files:
        # Load first view
        hsi_data = np.load(hsi_files[0])
        print(f"✓ HSI data shape: {hsi_data.shape}")
        print(f"  Data type: {hsi_data.dtype}")
        print(f"  Value range: [{hsi_data.min():.3f}, {hsi_data.max():.3f}]")
    
    # Check scene lists
    for split in ['train', 'val']:
        scene_list_file = processed_dir / f"{split}_scenes.json"
        if scene_list_file.exists():
            with open(scene_list_file, 'r') as f:
                scenes = json.load(f)
            print(f"✓ {split} scenes: {len(scenes)}")
        else:
            print(f"❌ {split}_scenes.json not found!")
    
    return True

def test_envi_reader():
    """Test ENVI reader with original data"""
    
    print("\nTesting ENVI reader with original data...")
    
    reader = ENVIReader()
    test_file = "./sample_data/0degree_001/0degree_raw"
    
    if Path(test_file).exists():
        data, header = reader.read_envi(test_file)
        print(f"✓ Successfully loaded ENVI data")
        print(f"  Shape: {data.shape}")
        print(f"  Wavelengths: {header['wavelength'][0]:.1f} - {header['wavelength'][-1]:.1f} nm")
        print(f"  Data type: {header['data type']}")
        print(f"  Interleave: {header['interleave']}")
    else:
        print("❌ Sample data file not found!")

def main():
    """Run verification"""
    print("=" * 50)
    print("HSI Gaussian 3D Data Verification")
    print("=" * 50)
    
    # Test ENVI reader
    test_envi_reader()
    
    # Verify conversion
    print()
    success = verify_conversion()
    
    if success:
        print("\n✅ Data conversion successful!")
        print("\nNext steps:")
        print("1. Install PyTorch and dependencies:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("2. Run the training pipeline:")
        print("   python train.py --data_root processed_data")
    else:
        print("\n❌ Data conversion failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()