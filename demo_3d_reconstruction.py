#!/usr/bin/env python3
"""
Demo script for HSI 3D reconstruction pipeline
Shows complete workflow from HSI data to 3D models
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core.spectral_gaussian import SpectralGaussian3D, SpectralGaussianData
from core.reconstruction import GaussianTo3D, NovelViewSynthesis
from core.spectral_renderer import SpectralGaussianRenderer, RenderingConfig
from data.envi_loader import ENVIMultiAngleLoader


def demo_reconstruction_pipeline():
    """Demonstrate the complete 3D reconstruction pipeline"""
    
    print("=== HSI 3D Reconstruction Demo ===\n")
    
    # 1. Load multi-angle HSI data
    print("1. Loading multi-angle HSI data...")
    loader = ENVIMultiAngleLoader("./sample_data")
    angles = loader.scan_directory()
    print(f"   Found {len(angles)} viewing angles: {angles[:5]}...")
    
    # Load a few angles for demo
    demo_angles = angles[:4]  # Use 4 views
    hsi_data = {}
    for angle in demo_angles:
        hsi_data[angle] = loader.load_angle_data(angle)
    print(f"   Loaded HSI data shape: {hsi_data[demo_angles[0]].shape}")
    
    # 2. Initialize 3D Gaussian model
    print("\n2. Initializing 3D Gaussian model...")
    gaussian_model = SpectralGaussian3D(
        num_points=10000,  # Smaller for demo
        num_channels=120,
        wavelength_range=(400.0, 1000.0)
    )
    
    # Get Gaussian parameters
    gaussian_params = gaussian_model({})
    print(f"   Initialized {gaussian_params['positions'].shape[0]} Gaussians")
    print(f"   Each with {gaussian_params['spectral_features'].shape[1]} spectral channels")
    
    # 3. Create renderer
    print("\n3. Setting up renderer...")
    renderer = SpectralGaussianRenderer(
        RenderingConfig(
            image_height=324,
            image_width=332,
            background_value=1.0
        )
    )
    
    # 4. Demonstrate rendering at different wavelengths
    print("\n4. Rendering at different wavelengths...")
    wavelengths_to_render = [450, 550, 650, 800]  # Blue, Green, Red, NIR
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    # Generate camera parameters for demo view
    camera_params = {
        'intrinsics': torch.tensor([
            [300, 0, 166],
            [0, 300, 162],
            [0, 0, 1]
        ], dtype=torch.float32),
        'extrinsics': torch.eye(4, dtype=torch.float32)
    }
    
    gaussian_data = SpectralGaussianData(
        positions=gaussian_params['positions'],
        rotations=gaussian_params['rotations'],
        scales=gaussian_params['scales'],
        opacities=gaussian_params['opacities'],
        spectral_features=gaussian_params['spectral_features'],
        wavelengths=gaussian_params['wavelengths']
    )
    
    for i, wl in enumerate(wavelengths_to_render):
        # Find wavelength index
        wl_tensor = torch.tensor(wl, dtype=torch.float32)
        idx = torch.argmin(torch.abs(gaussian_params['wavelengths'] - wl_tensor))
        
        # Render
        rendered = renderer(gaussian_data, camera_params, wavelength_indices=idx.unsqueeze(0))
        
        # Display
        img = rendered['spectral'][0].detach().cpu().numpy()
        axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f'{wl}nm')
        axes[i].axis('off')
    
    plt.suptitle('Rendered at Different Wavelengths')
    plt.tight_layout()
    plt.savefig('demo_wavelength_rendering.png')
    plt.close()
    print("   Saved wavelength rendering demo")
    
    # 5. Extract and visualize point cloud
    print("\n5. Extracting 3D point cloud...")
    converter = GaussianTo3D()
    pc_data = converter.extract_point_cloud(
        gaussian_params,
        color_wavelengths=[650, 550, 450]  # RGB
    )
    
    # Visualize point cloud
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    points = pc_data['points']
    colors = pc_data['colors']
    
    # Subsample for visualization
    subsample = np.random.choice(points.shape[0], min(5000, points.shape[0]), replace=False)
    ax1.scatter(
        points[subsample, 0],
        points[subsample, 1],
        points[subsample, 2],
        c=colors[subsample],
        s=1
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud')
    
    # Spectral signature plot
    ax2 = fig.add_subplot(122)
    # Plot spectral signatures of random points
    spectral = pc_data['spectral_features']
    wavelengths = gaussian_params['wavelengths'].numpy()
    
    for i in range(10):
        idx = np.random.randint(spectral.shape[0])
        ax2.plot(wavelengths, spectral[idx], alpha=0.5)
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Radiance')
    ax2.set_title('Sample Spectral Signatures')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_3d_pointcloud.png')
    plt.close()
    print("   Saved point cloud visualization")
    
    # 6. Novel view synthesis
    print("\n6. Generating novel views...")
    nvs = NovelViewSynthesis(renderer)
    novel_data = nvs.generate_novel_views(
        gaussian_params,
        num_views=8,
        radius=3.0,
        height=1.5
    )
    
    # Create a grid of novel views
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, img in enumerate(novel_data['images'][:8]):
        # Convert to RGB for display
        rgb = torch.stack([
            img[40],  # ~650nm (R)
            img[60],  # ~550nm (G) 
            img[20]   # ~450nm (B)
        ])
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        axes[i].imshow(rgb.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f'Novel View {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Novel View Synthesis')
    plt.tight_layout()
    plt.savefig('demo_novel_views.png')
    plt.close()
    print("   Saved novel view synthesis")
    
    # 7. Export 3D models
    print("\n7. Exporting 3D models...")
    output_dir = Path("demo_3d_output")
    output_dir.mkdir(exist_ok=True)
    
    # Export point cloud as PLY
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data['points'])
        pcd.colors = o3d.utility.Vector3dVector(pc_data['colors'])
        pcd.normals = o3d.utility.Vector3dVector(pc_data['normals'])
        
        o3d.io.write_point_cloud(str(output_dir / "demo_pointcloud.ply"), pcd)
        print("   Exported point cloud to demo_pointcloud.ply")
    except ImportError:
        print("   (Open3D not available, skipping PLY export)")
    
    # Save spectral features
    np.save(output_dir / "spectral_features.npy", pc_data['spectral_features'])
    print("   Saved spectral features")
    
    # Summary statistics
    print("\n=== Reconstruction Summary ===")
    print(f"Points extracted: {pc_data['points'].shape[0]}")
    print(f"Spectral channels: {pc_data['spectral_features'].shape[1]}")
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"Novel views generated: {len(novel_data['images'])}")
    print(f"Output saved to: {output_dir}")
    
    return gaussian_params, pc_data, novel_data


def visualize_reconstruction_quality(gt_views, rendered_views):
    """Visualize and compare reconstruction quality"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(min(4, len(gt_views))):
        # Ground truth
        gt = gt_views[i]
        axes[0, i].imshow(gt[60], cmap='viridis')  # Show one wavelength
        axes[0, i].set_title(f'GT View {i+1}')
        axes[0, i].axis('off')
        
        # Rendered
        rendered = rendered_views[i]
        axes[1, i].imshow(rendered[60], cmap='viridis')
        axes[1, i].set_title(f'Rendered View {i+1}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(gt[60] - rendered[60])
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Difference')
        axes[2, i].axis('off')
    
    plt.suptitle('Reconstruction Quality Comparison')
    plt.tight_layout()
    plt.savefig('demo_quality_comparison.png')
    plt.close()


if __name__ == "__main__":
    # Run the demo
    gaussian_params, pc_data, novel_data = demo_reconstruction_pipeline()
    
    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("  - demo_wavelength_rendering.png")
    print("  - demo_3d_pointcloud.png") 
    print("  - demo_novel_views.png")
    print("  - demo_3d_output/")
    print("    - demo_pointcloud.ply")
    print("    - spectral_features.npy")