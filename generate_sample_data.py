import numpy as np
import json
from pathlib import Path

def create_sample_hsi_data(output_dir: str = "./sample_hsi_data", num_scenes: int = 3, num_views: int = 4):
    """Create sample hyperspectral data for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # HSI specifications
    height, width = 324, 332
    num_channels = 120
    wavelengths = np.linspace(400, 1000, num_channels)
    
    print(f"Creating sample HSI data in {output_dir}")
    print(f"Image size: {width}x{height}, Channels: {num_channels}")
    print(f"Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f}nm")
    
    # Create scenes
    scenes = []
    for scene_idx in range(num_scenes):
        scene_name = f"scene_{scene_idx:03d}"
        scene_path = output_path / scene_name
        scene_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating {scene_name}...")
        
        # Create a base spectral signature for this scene
        # Different scenes have different dominant wavelengths
        dominant_wavelength = 450 + scene_idx * 100  # 450, 550, 650nm
        dominant_idx = np.argmin(np.abs(wavelengths - dominant_wavelength))
        
        # Generate HSI cubes for each view
        for view_idx in range(num_views):
            # Create spatial patterns
            x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
            
            # Create different objects with different spectral signatures
            hsi_cube = np.zeros((height, width, num_channels), dtype=np.float32)
            
            # Object 1: Circle with specific spectral signature
            circle_mask = (x - 0.3)**2 + (y - 0.2)**2 < 0.2**2
            for c in range(num_channels):
                # Gaussian spectral curve centered at dominant wavelength
                spectral_response = 0.8 * np.exp(-0.5 * ((c - dominant_idx) / 20)**2)
                hsi_cube[circle_mask, c] = spectral_response
            
            # Object 2: Square with different spectral signature
            square_mask = (np.abs(x + 0.3) < 0.2) & (np.abs(y + 0.3) < 0.2)
            secondary_idx = (dominant_idx + 30) % num_channels
            for c in range(num_channels):
                spectral_response = 0.7 * np.exp(-0.5 * ((c - secondary_idx) / 15)**2)
                hsi_cube[square_mask, c] = spectral_response
            
            # Background: Broad spectral response
            background_mask = ~(circle_mask | square_mask)
            for c in range(num_channels):
                # Broad, low-intensity spectral response
                background_response = 0.2 + 0.1 * np.sin(c * np.pi / num_channels)
                hsi_cube[background_mask, c] = background_response
            
            # Add some noise and view-dependent variations
            noise = np.random.normal(0, 0.02, hsi_cube.shape)
            view_variation = 0.1 * np.sin(view_idx * np.pi / num_views)
            hsi_cube = hsi_cube + noise + view_variation
            hsi_cube = np.clip(hsi_cube, 0, 1)
            
            # Save HSI cube
            hsi_path = scene_path / f"hsi_{view_idx:03d}.npy"
            np.save(hsi_path, hsi_cube)
            print(f"  Saved view {view_idx}: {hsi_path.name}")
        
        # Generate camera poses (circular arrangement)
        intrinsics = []
        extrinsics = []
        
        for view_idx in range(num_views):
            # Camera intrinsics (same for all views)
            focal_length = 300.0
            K = np.array([
                [focal_length, 0, width/2],
                [0, focal_length, height/2],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics.append(K)
            
            # Camera extrinsics (circular arrangement)
            angle = 2 * np.pi * view_idx / num_views
            radius = 5.0
            height_cam = 3.0
            
            # Camera position in world coordinates
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height_cam
            ])
            
            # Camera looks at origin
            look_at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Compute rotation matrix (camera to world)
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up_new = np.cross(right, forward)
            
            # Build rotation matrix
            R = np.stack([right, up_new, -forward], axis=0).T
            
            # Build extrinsic matrix (world to camera)
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R.T
            E[:3, 3] = -R.T @ cam_pos
            
            extrinsics.append(E)
        
        # Save camera parameters
        camera_data = {
            "intrinsics": [K.tolist() for K in intrinsics],
            "extrinsics": [E.tolist() for E in extrinsics]
        }
        
        with open(scene_path / "camera_poses.json", "w") as f:
            json.dump(camera_data, f, indent=2)
        print(f"  Saved camera poses")
        
        scenes.append({
            "name": scene_name,
            "path": str(scene_path),
            "num_views": num_views
        })
    
    # Create train/val split
    train_scenes = scenes[:int(0.7 * len(scenes))]
    val_scenes = scenes[int(0.7 * len(scenes)):]
    
    # Save scene lists
    with open(output_path / "train_scenes.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
    
    with open(output_path / "val_scenes.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"Total scenes: {len(scenes)}")
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")
    print(f"\nDataset structure:")
    print(f"{output_path}/")
    print(f"├── train_scenes.json")
    print(f"├── val_scenes.json")
    for scene in scenes[:2]:
        print(f"├── {scene['name']}/")
        print(f"│   ├── hsi_000.npy ... hsi_{num_views-1:03d}.npy")
        print(f"│   └── camera_poses.json")
    if len(scenes) > 2:
        print(f"└── ... ({len(scenes)-2} more scenes)")

if __name__ == "__main__":
    create_sample_hsi_data()