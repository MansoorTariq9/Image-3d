import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json
from .preprocessing import HSIPreprocessor, HSIConfig, SpectralAugmentation

class MultiViewHSIDataset(Dataset):
    """Dataset for multi-view hyperspectral images"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_views: int = 4,
        config: Optional[HSIConfig] = None,
        augment: bool = True,
        cache_preprocessed: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_views = num_views
        self.config = config or HSIConfig()
        self.augment = augment and (split == "train")
        self.cache_preprocessed = cache_preprocessed
        
        # Initialize preprocessor
        self.preprocessor = HSIPreprocessor(self.config)
        
        # Initialize augmentation
        self.augmentor = SpectralAugmentation() if self.augment else None
        
        # Load scene list
        self.scenes = self._load_scene_list()
        
        # Cache for preprocessed data
        self.cache = {}
        
    def _load_scene_list(self) -> List[Dict]:
        """Load list of scenes and their metadata"""
        scene_file = self.data_root / f"{self.split}_scenes.json"
        
        if scene_file.exists():
            with open(scene_file, "r") as f:
                scenes = json.load(f)
        else:
            # If no scene file, scan directory
            scenes = []
            scene_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
            
            for scene_dir in scene_dirs:
                # Check if scene has required data
                hsi_files = list(scene_dir.glob("hsi_*.npy"))
                pose_file = scene_dir / "camera_poses.json"
                
                if len(hsi_files) >= self.num_views and pose_file.exists():
                    scenes.append({
                        "name": scene_dir.name,
                        "path": str(scene_dir),
                        "num_views": len(hsi_files)
                    })
                    
        return scenes
    
    def _load_hsi_cube(self, path: str) -> np.ndarray:
        """Load hyperspectral image cube"""
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".mat"):
            import scipy.io
            mat = scipy.io.loadmat(path)
            # Assume HSI is stored under 'data' key
            return mat.get("data", mat[list(mat.keys())[-1]])
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def _load_camera_params(self, scene_path: Path) -> Dict[str, torch.Tensor]:
        """Load camera parameters for a scene"""
        pose_file = scene_path / "camera_poses.json"
        
        with open(pose_file, "r") as f:
            poses = json.load(f)
            
        # Convert to tensors
        intrinsics = torch.tensor(poses["intrinsics"], dtype=torch.float32)
        extrinsics = torch.tensor(poses["extrinsics"], dtype=torch.float32)
        
        return {
            "intrinsics": intrinsics,  # [V, 3, 3]
            "extrinsics": extrinsics   # [V, 4, 4]
        }
    
    def _get_cache_key(self, scene_idx: int, view_indices: List[int]) -> str:
        """Generate cache key"""
        return f"{scene_idx}_{'_'.join(map(str, view_indices))}"
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-view HSI data for a scene"""
        scene = self.scenes[idx]
        scene_path = Path(scene["path"])
        
        # Select views
        available_views = min(scene["num_views"], 20)  # Limit max views
        if self.split == "train":
            # Random view selection for training
            view_indices = np.random.choice(available_views, self.num_views, replace=False)
        else:
            # Fixed views for validation/test
            view_indices = np.linspace(0, available_views - 1, self.num_views, dtype=int)
            
        # Check cache
        cache_key = self._get_cache_key(idx, view_indices.tolist())
        if self.cache_preprocessed and cache_key in self.cache:
            return self.cache[cache_key]
            
        # Load HSI cubes
        hsi_views = []
        for v_idx in view_indices:
            hsi_path = scene_path / f"hsi_{v_idx:03d}.npy"
            hsi_cube = self._load_hsi_cube(str(hsi_path))
            
            # Ensure correct shape [H, W, C]
            if hsi_cube.shape[2] != self.config.num_channels:
                if hsi_cube.shape[0] == self.config.num_channels:
                    hsi_cube = hsi_cube.transpose(1, 2, 0)
                else:
                    raise ValueError(f"Unexpected HSI shape: {hsi_cube.shape}")
                    
            hsi_views.append(hsi_cube)
            
        # Preprocess all views
        preprocessed_views = []
        wavelengths = None
        for hsi in hsi_views:
            processed = self.preprocessor.preprocess(hsi)
            
            # Store wavelengths from first view
            if wavelengths is None:
                wavelengths = processed["wavelengths"]
            
            # Apply augmentation if enabled
            if self.augmentor:
                processed["hsi"] = self.augmentor(processed["hsi"], processed["wavelengths"])
                
            preprocessed_views.append(processed["hsi"])
            
        # Stack views
        hsi_tensor = torch.stack(preprocessed_views)  # [V, C, H, W]
        
        # Load camera parameters
        camera_params = self._load_camera_params(scene_path)
        
        # Select corresponding camera parameters
        intrinsics = camera_params["intrinsics"][view_indices]  # [V, 3, 3]
        extrinsics = camera_params["extrinsics"][view_indices]  # [V, 4, 4]
        
        # Prepare output
        output = {
            "hsi": hsi_tensor,  # [V, C, H, W]
            "intrinsics": intrinsics,  # [V, 3, 3]
            "extrinsics": extrinsics,  # [V, 4, 4]
            "wavelengths": wavelengths,  # [C]
            "scene_name": scene["name"],
            "view_indices": torch.tensor(view_indices)
        }
        
        # Cache if enabled
        if self.cache_preprocessed:
            self.cache[cache_key] = output
            
        return output

class HSIDataModule:
    """Data module for managing train/val/test splits"""
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 1,
        num_views: int = 4,
        num_workers: int = 4,
        config: Optional[HSIConfig] = None
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_workers = num_workers
        self.config = config or HSIConfig()
        
    def setup(self):
        """Initialize datasets"""
        self.train_dataset = MultiViewHSIDataset(
            self.data_root,
            split="train",
            num_views=self.num_views,
            config=self.config,
            augment=True
        )
        
        self.val_dataset = MultiViewHSIDataset(
            self.data_root,
            split="val",
            num_views=self.num_views,
            config=self.config,
            augment=False
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def create_dummy_data(output_dir: str, num_scenes: int = 5, num_views: int = 8):
    """Create dummy HSI data for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create scenes
    scenes = []
    for i in range(num_scenes):
        scene_name = f"scene_{i:03d}"
        scene_path = output_path / scene_name
        scene_path.mkdir(exist_ok=True)
        
        # Generate HSI cubes
        for v in range(num_views):
            hsi = np.random.rand(324, 332, 120).astype(np.float32)
            # Add some structure
            x, y = np.meshgrid(np.linspace(0, 1, 332), np.linspace(0, 1, 324))
            for c in range(120):
                hsi[:, :, c] *= np.sin(x * 10 + c * 0.1) * np.cos(y * 10 - c * 0.1)
                
            np.save(scene_path / f"hsi_{v:03d}.npy", hsi)
            
        # Generate camera poses
        intrinsics = []
        extrinsics = []
        
        for v in range(num_views):
            # Simple intrinsics
            K = np.array([
                [300, 0, 166],
                [0, 300, 162],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics.append(K)
            
            # Circular camera arrangement
            angle = 2 * np.pi * v / num_views
            radius = 5.0
            
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.0
            ])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross([0, 0, 1], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Rotation matrix (camera to world)
            R = np.stack([right, up, -forward], axis=0)
            
            # Extrinsic matrix (world to camera)
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R.T
            E[:3, 3] = -R.T @ cam_pos
            
            extrinsics.append(E)
            
        # Save camera parameters
        camera_data = {
            "intrinsics": np.stack(intrinsics).tolist(),
            "extrinsics": np.stack(extrinsics).tolist()
        }
        
        with open(scene_path / "camera_poses.json", "w") as f:
            json.dump(camera_data, f, indent=2)
            
        scenes.append({
            "name": scene_name,
            "path": str(scene_path),
            "num_views": num_views
        })
        
    # Save scene lists
    train_scenes = scenes[:int(0.8 * num_scenes)]
    val_scenes = scenes[int(0.8 * num_scenes):]
    
    with open(output_path / "train_scenes.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
        
    with open(output_path / "val_scenes.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
        
    print(f"Created dummy HSI dataset at {output_path}")
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")

if __name__ == "__main__":
    # Test dataset creation
    create_dummy_data("./dummy_hsi_data")