#!/usr/bin/env python3
"""Demo novel view synthesis with the trained model"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import math

from train import HSIGaussian3DModel
from data.dataset import MultiViewHSIDataset, HSIConfig
from visualize_results import hyperspectral_to_rgb

def generate_novel_camera_poses(base_intrinsics, num_poses=8, radius=1.5, height=0.3):
    """Generate camera poses in a circle around the object"""
    poses = []
    
    for i in range(num_poses):
        angle = 2 * math.pi * i / num_poses
        
        # Camera position
        x = radius * math.cos(angle)
        y = radius * math.sin(angle) 
        z = height
        
        cam_pos = np.array([x, y, z])
        
        # Look at origin
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Compute rotation matrix
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        # Camera to world rotation
        R = np.stack([right, up_corrected, -forward], axis=0)
        
        # World to camera transform
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.T
        pose[:3, 3] = -R.T @ cam_pos
        
        poses.append(pose)
    
    return np.stack(poses)

def render_novel_views(model, batch, novel_poses):
    """Render from novel viewpoints"""
    model.eval()
    
    with torch.no_grad():
        # Get model output
        outputs = model(batch)
        gaussian_data = outputs['gaussian_data']
        
        # Use intrinsics from first view
        intrinsics = batch['intrinsics'][0, 0]
        
        rendered_views = []
        wavelengths = batch['wavelengths'].cpu().numpy()
        
        for pose in novel_poses:
            camera_params = {
                'intrinsics': intrinsics.unsqueeze(0),
                'extrinsics': torch.tensor(pose).unsqueeze(0).to(batch['hsi'].device)
            }
            
            # Render full spectrum
            rendered = model.renderer(gaussian_data, camera_params)
            rendered_hsi = rendered['spectral'].squeeze().cpu().numpy()
            
            # Convert to RGB
            rendered_rgb = hyperspectral_to_rgb(rendered_hsi, wavelengths)
            rendered_views.append(rendered_rgb)
    
    return rendered_views

def main():
    """Generate novel view demo"""
    
    print("="*50)
    print("Novel View Synthesis Demo")
    print("="*50)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSIGaussian3DModel(config).to(device)
    
    # Load checkpoint
    checkpoint_path = "outputs/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"✓ Loaded trained model")
    
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
    
    print(f"✓ Loaded scene data")
    
    # Generate novel camera poses
    print("\nGenerating novel viewpoints...")
    novel_poses = generate_novel_camera_poses(
        batch['intrinsics'][0, 0].cpu().numpy(),
        num_poses=8,
        radius=1.2,
        height=0.4
    )
    
    # Render novel views
    print("Rendering novel views...")
    novel_views = render_novel_views(model, batch, novel_poses)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Show training views (top row)
    for i in range(4):
        ax = plt.subplot(3, 4, i+1)
        img = batch['hsi'][0, i].cpu().numpy()
        wavelengths = batch['wavelengths'].cpu().numpy()
        rgb = hyperspectral_to_rgb(img, wavelengths)
        ax.imshow(rgb)
        ax.set_title(f"Training View {i+1}")
        ax.axis('off')
    
    # Show novel views (bottom rows)
    for i in range(8):
        ax = plt.subplot(3, 4, i+5)
        ax.imshow(novel_views[i])
        angle = 360 * i / 8
        ax.set_title(f"Novel View @ {angle:.0f}°")
        ax.axis('off')
    
    plt.suptitle("Novel View Synthesis from Trained HSI Gaussian Model", fontsize=16)
    plt.tight_layout()
    
    # Save results
    output_path = "visualization_results/novel_views_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved novel views to: {output_path}")
    
    # Also create an animated GIF showing rotation
    print("\nCreating rotation animation...")
    from PIL import Image
    
    # Render more views for smooth animation
    animation_poses = generate_novel_camera_poses(
        batch['intrinsics'][0, 0].cpu().numpy(),
        num_poses=36,  # 10-degree steps
        radius=1.2,
        height=0.4
    )
    
    print("Rendering frames...")
    animation_views = render_novel_views(model, batch, animation_poses)
    
    # Convert to PIL images
    pil_images = []
    for view in animation_views:
        # Convert to uint8
        img_uint8 = (view * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_images.append(pil_img)
    
    # Save as GIF
    gif_path = "visualization_results/rotation_animation.gif"
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    print(f"✓ Saved rotation animation to: {gif_path}")
    
    print("\n" + "="*50)
    print("✅ Novel view synthesis demo complete!")
    print("="*50)
    print("\nThe demo shows:")
    print("- Top row: Original training views")
    print("- Bottom rows: Novel views rendered by the model")
    print("- Animation: 360° rotation around the object")
    print("\nThis demonstrates the model's ability to:")
    print("- Learn 3D structure from multi-view HSI")
    print("- Synthesize realistic views from new angles")
    print("- Maintain spectral consistency across views")

if __name__ == "__main__":
    main()