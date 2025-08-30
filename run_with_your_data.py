#!/usr/bin/env python3
"""
Example script showing how to run the complete pipeline with your ENVI HSI data
"""

import os
import sys
from pathlib import Path

def main():
    print("=== HSI Gaussian 3D Pipeline ===\n")
    
    # Check if sample data exists
    if not Path("sample_data").exists():
        print("❌ Error: sample_data directory not found!")
        print("Please ensure your ENVI data is in the sample_data directory")
        sys.exit(1)
    
    print("📁 Found sample_data directory")
    
    # List available angles
    angles = []
    for item in Path("sample_data").iterdir():
        if item.is_dir() and "degree" in item.name:
            angles.append(item.name)
    
    print(f"📐 Found {len(angles)} viewing angles: {angles[:5]}...")
    
    # Step 1: View one sample
    print("\n1️⃣ Viewing sample HSI data...")
    os.system(f"python view_hsi_data.py sample_data/{angles[0]}/{angles[0].replace('_001', '_raw')}")
    
    # Step 2: Convert to training format
    print("\n2️⃣ Converting ENVI to training format...")
    os.system("python convert_envi_to_numpy.py --input_dir sample_data --output_dir processed_data")
    
    # Step 3: Quick training test (just 2 epochs)
    print("\n3️⃣ Running quick training test (2 epochs)...")
    os.system("""python train.py \
        --config config.yaml \
        --data_root processed_data \
        --output_dir quick_test \
        --num_epochs 2""")
    
    # Step 4: Run inference if training succeeded
    checkpoint = Path("quick_test/checkpoint_epoch_1.pth")
    if checkpoint.exists():
        print("\n4️⃣ Running inference and 3D reconstruction...")
        os.system(f"""python inference.py \
            --checkpoint {checkpoint} \
            --hsi_paths processed_data/sample_data/hsi_00*.npy \
            --camera_params processed_data/sample_data/camera_poses.json \
            --output_dir inference_output \
            --export_ply \
            --wavelengths 450 550 650""")
        
        print("\n✅ Pipeline complete! Check:")
        print("  - hsi_visualizations/ for input data visualization")
        print("  - quick_test/ for training outputs")
        print("  - inference_output/ for 3D reconstruction")
        print("    - pointcloud.ply (view in MeshLab)")
        print("    - numpy_data/ (spectral features)")
    else:
        print("\n⚠️ Training did not complete. Check error messages above.")
    

if __name__ == "__main__":
    main()