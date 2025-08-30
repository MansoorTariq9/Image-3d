#!/usr/bin/env python3
"""
Convert ENVI format HSI data to numpy arrays for training
"""

import argparse
from pathlib import Path
from data.envi_loader import convert_envi_to_standard_format, ENVIMultiAngleLoader
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description="Convert ENVI HSI data to training format")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing ENVI files (e.g., sample_data)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="Fraction of data for validation (default: 0.2)")
    args = parser.parse_args()
    
    print(f"Converting ENVI data from {args.input_dir} to {args.output_dir}")
    
    # Convert ENVI to standard format
    try:
        output_path = convert_envi_to_standard_format(
            args.input_dir,
            args.output_dir
        )
        print(f"\n✅ Successfully converted data to: {output_path}")
        
        # Show what was created
        output_path = Path(args.output_dir)
        
        # List scenes
        train_scenes = json.load(open(output_path / "train_scenes.json"))
        val_scenes = json.load(open(output_path / "val_scenes.json"))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  - Training scenes: {len(train_scenes)}")
        print(f"  - Validation scenes: {len(val_scenes)}")
        
        if train_scenes:
            scene_path = Path(train_scenes[0]["path"])
            if scene_path.exists():
                hsi_files = list(scene_path.glob("hsi_*.npy"))
                print(f"  - Views per scene: {len(hsi_files)}")
                
                # Check data shape
                if hsi_files:
                    sample = np.load(hsi_files[0])
                    print(f"  - HSI data shape: {sample.shape}")
                    print(f"  - Data type: {sample.dtype}")
        
        print("\n🎯 Next steps:")
        print(f"1. Train model: python train.py --config config.yaml --data_root {args.output_dir}")
        print(f"2. View data: python view_hsi_data.py {args.input_dir}/[angle]_001/[angle]_raw")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your ENVI files have .hdr headers")
        print("2. Ensure directory structure: angle_folders/data_files")
        print("3. Try viewing a single file first with view_hsi_data.py")
        

if __name__ == "__main__":
    main()