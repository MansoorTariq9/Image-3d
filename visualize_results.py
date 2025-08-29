#!/usr/bin/env python3
"""Visualize trained HSI Gaussian 3D model results"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm

# Import our modules
from train import HSIGaussian3DModel
from data.dataset import MultiViewHSIDataset, HSIConfig
from core.spectral_gaussian import SpectralGaussianData

def hyperspectral_to_rgb(hsi_image, wavelengths):
    """Convert hyperspectral image to RGB visualization
    
    Args:
        hsi_image: [C, H, W] or [H, W, C] hyperspectral image
        wavelengths: array of wavelengths in nm
        
    Returns:
        rgb_image: [H, W, 3] RGB image
    """
    # Ensure HWC format
    if hsi_image.shape[0] == len(wavelengths):
        hsi_image = hsi_image.transpose(1, 2, 0)
    
    H, W, C = hsi_image.shape
    
    # Define RGB wavelength ranges (approximate)
    # R: 620-750nm, G: 495-570nm, B: 450-495nm
    r_mask = (wavelengths >= 620) & (wavelengths <= 750)
    g_mask = (wavelengths >= 495) & (wavelengths <= 570)
    b_mask = (wavelengths >= 450) & (wavelengths <= 495)
    
    # Average channels in each range
    r_channel = hsi_image[:, :, r_mask].mean(axis=2) if r_mask.any() else np.zeros((H, W))
    g_channel = hsi_image[:, :, g_mask].mean(axis=2) if g_mask.any() else np.zeros((H, W))
    b_channel = hsi_image[:, :, b_mask].mean(axis=2) if b_mask.any() else np.zeros((H, W))
    
    # Stack and normalize
    rgb = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    # Normalize to 0-1 range
    for i in range(3):
        channel = rgb[:, :, i]
        p2, p98 = np.percentile(channel, [2, 98])
        if p98 > p2:
            rgb[:, :, i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
    
    return rgb

def visualize_spectral_signatures(model, gaussian_data, num_samples=10):
    """Visualize spectral signatures of random Gaussians"""
    
    # Get spectral features
    if hasattr(gaussian_data, 'spectral_features'):
        spectral_features = gaussian_data.spectral_features.detach().cpu().numpy()
        wavelengths = gaussian_data.wavelengths.cpu().numpy()
    else:
        spectral_features = gaussian_data['spectral_features'].detach().cpu().numpy()
        wavelengths = gaussian_data['wavelengths'].cpu().numpy()
    
    # Randomly sample Gaussians
    num_gaussians = spectral_features.shape[0]
    indices = np.random.choice(num_gaussians, min(num_samples, num_gaussians), replace=False)
    
    # Plot spectral signatures
    plt.figure(figsize=(12, 6))
    for idx in indices:
        plt.plot(wavelengths, spectral_features[idx], alpha=0.7, linewidth=2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spectral Radiance')
    plt.title(f'Spectral Signatures of {num_samples} Random Gaussians')
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 1000)
    plt.tight_layout()
    
    return plt.gcf()

def render_wavelength_series(model, batch, wavelengths_to_render=[450, 550, 650, 750, 850, 950]):
    """Render the scene at different wavelengths"""
    
    model.eval()
    with torch.no_grad():
        # Get model output
        outputs = model(batch)
        gaussian_data = outputs['gaussian_data']
        
        # Camera parameters from first view
        intrinsics = batch['intrinsics'][0, 0]
        extrinsics = batch['extrinsics'][0, 0]
        camera_params = {
            'intrinsics': intrinsics.unsqueeze(0),
            'extrinsics': extrinsics.unsqueeze(0)
        }
        
        # Render at different wavelengths
        rendered_images = []
        wavelengths_tensor = batch['wavelengths']
        
        for target_wavelength in wavelengths_to_render:
            # Find closest wavelength index
            idx = torch.argmin(torch.abs(wavelengths_tensor - target_wavelength)).item()
            
            # Render at this wavelength
            rendered = model.renderer(gaussian_data, camera_params, wavelength_indices=[idx])
            rendered_img = rendered['spectral'][0].squeeze().cpu().numpy()  # Only one channel returned
            rendered_images.append((target_wavelength, rendered_img))
    
    return rendered_images

def create_comparison_figure(original, rendered, wavelengths, title="Original vs Rendered"):
    """Create side-by-side comparison of original and rendered images"""
    
    # Convert to RGB
    original_rgb = hyperspectral_to_rgb(original, wavelengths)
    rendered_rgb = hyperspectral_to_rgb(rendered, wavelengths)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(rendered_rgb)
    axes[1].set_title("Rendered")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def visualize_depth_map(model, batch):
    """Visualize estimated depth maps"""
    
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        
        # Get depth from first view
        if outputs['depth_estimates']:
            depth = outputs['depth_estimates'][0]['depth'].squeeze().cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            im = plt.imshow(depth, cmap='viridis')
            plt.colorbar(im)
            plt.title('Estimated Depth Map')
            plt.axis('off')
            
            return plt.gcf()
    
    return None

def main():
    """Main visualization pipeline"""
    
    print("=" * 50)
    print("HSI Gaussian 3D Visualization")
    print("=" * 50)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\nLoading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSIGaussian3DModel(config).to(device)
    
    # Load checkpoint
    checkpoint_path = "outputs/best_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("⚠️  No checkpoint found, using random weights")
    
    # Load data
    print("\nLoading data...")
    dataset = MultiViewHSIDataset(
        data_root="processed_data",
        split="val",
        num_views=config["num_views"],
        config=HSIConfig(),
        augment=False
    )
    
    # Get a sample
    batch = dataset[0]
    for key in ['hsi', 'intrinsics', 'extrinsics']:
        batch[key] = batch[key].unsqueeze(0).to(device)
    batch['wavelengths'] = batch['wavelengths'].to(device)
    
    print(f"✓ Loaded scene: {batch['scene_name']}")
    
    # Create output directory
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n1. Visualizing spectral signatures...")
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        gaussian_data = outputs['gaussian_data']
    
    fig = visualize_spectral_signatures(model, gaussian_data, num_samples=20)
    fig.savefig(output_dir / "spectral_signatures.png", dpi=150)
    print(f"✓ Saved to {output_dir}/spectral_signatures.png")
    
    print("\n2. Rendering at different wavelengths...")
    wavelength_renders = render_wavelength_series(model, batch)
    
    # Create wavelength montage
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (wavelength, img) in enumerate(wavelength_renders):
        axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f"{wavelength}nm")
        axes[i].axis('off')
    
    plt.suptitle("Rendered at Different Wavelengths")
    plt.tight_layout()
    fig.savefig(output_dir / "wavelength_series.png", dpi=150)
    print(f"✓ Saved to {output_dir}/wavelength_series.png")
    
    print("\n3. Creating RGB visualization...")
    # Get original and rendered full spectrum
    original_hsi = batch['hsi'][0, 0].cpu().numpy()  # First view
    wavelengths = batch['wavelengths'].cpu().numpy()
    
    # Render full spectrum
    with torch.no_grad():
        rendered = model.renderer(gaussian_data, {
            'intrinsics': batch['intrinsics'][0, 0:1],
            'extrinsics': batch['extrinsics'][0, 0:1]
        })
        rendered_hsi = rendered['spectral'].squeeze().cpu().numpy()
    
    fig = create_comparison_figure(original_hsi, rendered_hsi, wavelengths)
    fig.savefig(output_dir / "rgb_comparison.png", dpi=150)
    print(f"✓ Saved to {output_dir}/rgb_comparison.png")
    
    print("\n4. Visualizing depth map...")
    depth_fig = visualize_depth_map(model, batch)
    if depth_fig:
        depth_fig.savefig(output_dir / "depth_map.png", dpi=150)
        print(f"✓ Saved to {output_dir}/depth_map.png")
    
    print("\n5. Creating 3D Gaussian visualization...")
    # Visualize Gaussian positions
    if hasattr(gaussian_data, 'positions'):
        positions = gaussian_data.positions.detach().cpu().numpy()
    else:
        positions = gaussian_data['positions'].detach().cpu().numpy()
    
    # Sample for visibility
    num_vis = min(5000, positions.shape[0])
    indices = np.random.choice(positions.shape[0], num_vis, replace=False)
    pos_vis = positions[indices]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by z-coordinate
    scatter = ax.scatter(pos_vis[:, 0], pos_vis[:, 1], pos_vis[:, 2], 
                        c=pos_vis[:, 2], cmap='viridis', s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Gaussian Positions ({num_vis} points shown)')
    plt.colorbar(scatter, ax=ax, label='Z coordinate')
    
    fig.savefig(output_dir / "gaussian_positions_3d.png", dpi=150)
    print(f"✓ Saved to {output_dir}/gaussian_positions_3d.png")
    
    print("\n" + "=" * 50)
    print("✅ Visualization complete!")
    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()