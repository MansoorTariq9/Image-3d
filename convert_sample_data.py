#!/usr/bin/env python3
"""Convert ENVI sample data to standard format for training"""

import argparse
from pathlib import Path
from data.envi_loader import convert_envi_to_standard_format

def main():
    parser = argparse.ArgumentParser(description='Convert ENVI HSI data to standard format')
    parser.add_argument('--input_dir', type=str, default='./sample_data',
                      help='Path to ENVI data directory')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                      help='Output directory for processed data')
    
    args = parser.parse_args()
    
    print(f"Converting ENVI data from {args.input_dir} to {args.output_dir}")
    
    # Convert the data
    output_path = convert_envi_to_standard_format(args.input_dir, args.output_dir)
    
    print(f"\nConversion complete!")
    print(f"Processed data saved to: {output_path}")
    print("\nYou can now train the model using:")
    print(f"python train.py --data_root {output_path}")

if __name__ == "__main__":
    main()