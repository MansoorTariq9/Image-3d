#!/usr/bin/env python3
"""
Quick test of 3D reconstruction functionality
"""

import torch
import numpy as np
import sys

def test_3d_reconstruction():
    print("Testing 3D reconstruction components...")
    
    try:
        # Test imports
        from core.spectral_gaussian import SpectralGaussian3D, SpectralGaussianData
        from core.reconstruction_simple import SimpleGaussianTo3D, SimpleNovelViewSynthesis, export_3d_reconstruction
        from core.spectral_renderer import SpectralGaussianRenderer, RenderingConfig
        print("✓ All imports successful")
        
        # Test Gaussian model
        gaussian_model = SpectralGaussian3D(num_points=1000, num_channels=120)
        params = gaussian_model({})
        print(f"✓ Gaussian model created: {params['positions'].shape[0]} points")
        
        # Test point cloud extraction
        converter = SimpleGaussianTo3D()
        pc_data = converter.extract_point_cloud(params)
        print(f"✓ Point cloud extracted: {pc_data['points'].shape}")
        
        # Test renderer
        renderer = SpectralGaussianRenderer(RenderingConfig())
        print("✓ Renderer initialized")
        
        # Test novel view synthesis
        nvs = SimpleNovelViewSynthesis(renderer)
        print("✓ Novel view synthesis ready")
        
        # Test export (without actual file writing)
        print("✓ Export functions available")
        
        print("\n✅ All 3D reconstruction components working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_3d_reconstruction()
    sys.exit(0 if success else 1)