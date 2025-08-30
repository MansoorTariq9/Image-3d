"""
Simplified 3D Reconstruction utilities without external dependencies
Core functionality for HSI Gaussian Splatting
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class SimpleGaussianTo3D:
    """Convert Gaussian splats to 3D representations using only numpy"""
    
    def __init__(self, opacity_threshold: float = 0.1):
        self.opacity_threshold = opacity_threshold
        
    def extract_point_cloud(
        self, 
        gaussian_data,
        wavelength_nm: Optional[float] = None,
        color_wavelengths: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract point cloud from Gaussian representation"""
        # Handle both dict and dataclass formats
        if hasattr(gaussian_data, 'positions'):
            # It's a dataclass (SpectralGaussianData)
            positions = gaussian_data.positions.detach().cpu().numpy()
            opacities = gaussian_data.opacities.detach().cpu().numpy().squeeze()
            scales = gaussian_data.scales.detach().cpu().numpy()
            spectral_features = gaussian_data.spectral_features.detach().cpu().numpy()
            wavelengths = gaussian_data.wavelengths.detach().cpu().numpy()
        else:
            # It's a dictionary
            positions = gaussian_data['positions'].detach().cpu().numpy()
            opacities = gaussian_data['opacities'].detach().cpu().numpy().squeeze()
            scales = gaussian_data['scales'].detach().cpu().numpy()
            spectral_features = gaussian_data['spectral_features'].detach().cpu().numpy()
            wavelengths = gaussian_data['wavelengths'].detach().cpu().numpy()
        
        # Filter by opacity
        valid_mask = opacities > self.opacity_threshold
        positions = positions[valid_mask]
        scales = scales[valid_mask]
        spectral_features = spectral_features[valid_mask]
        opacities = opacities[valid_mask]
        
        # Generate color visualization
        if color_wavelengths is None:
            color_wavelengths = [650.0, 550.0, 450.0]  # R, G, B
            
        colors = np.zeros((positions.shape[0], 3))
        for i, wl in enumerate(color_wavelengths):
            idx = np.argmin(np.abs(wavelengths - wl))
            colors[:, i] = spectral_features[:, idx]
            
        # Normalize colors
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        
        return {
            'points': positions,
            'colors': colors,
            'scales': scales,
            'opacities': opacities,
            'spectral_features': spectral_features,
            'wavelengths': wavelengths
        }
    
    def save_as_ply(self, pc_data: Dict[str, np.ndarray], filepath: str):
        """Save point cloud as simple PLY file"""
        points = pc_data['points']
        colors = pc_data['colors']
        
        # PLY header
        header = f"""ply
format ascii 1.0
element vertex {points.shape[0]}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        
        # Write PLY file
        with open(filepath, 'w') as f:
            f.write(header)
            
            for i in range(points.shape[0]):
                f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} ")
                f.write(f"{int(colors[i,0]*255)} {int(colors[i,1]*255)} {int(colors[i,2]*255)}\n")
        
        print(f"Saved {points.shape[0]} points to {filepath}")
    
    def export_to_numpy(self, pc_data: Dict[str, np.ndarray], output_dir: str):
        """Export point cloud data as numpy arrays"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        np.save(os.path.join(output_dir, 'points.npy'), pc_data['points'])
        np.save(os.path.join(output_dir, 'colors.npy'), pc_data['colors'])
        np.save(os.path.join(output_dir, 'spectral_features.npy'), pc_data['spectral_features'])
        np.save(os.path.join(output_dir, 'scales.npy'), pc_data['scales'])
        
        # Save metadata
        metadata = {
            'num_points': pc_data['points'].shape[0],
            'num_channels': pc_data['spectral_features'].shape[1],
            'wavelengths': pc_data['wavelengths'].tolist()
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Exported point cloud data to {output_dir}")


class SimpleNovelViewSynthesis:
    """Generate novel views without external dependencies"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        
    def generate_camera_trajectory(
        self,
        num_views: int = 20,
        radius: float = 5.0,
        height_range: Tuple[float, float] = (1.0, 3.0),
        look_at: np.ndarray = np.array([0, 0, 0])
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate camera parameters for circular trajectory"""
        camera_params_list = []
        
        for i in range(num_views):
            # Vary both angle and height
            angle = 2 * np.pi * i / num_views
            height = height_range[0] + (height_range[1] - height_range[0]) * (i / num_views)
            
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Create camera parameters
            camera_params = self._create_camera_params(cam_pos, look_at)
            camera_params_list.append(camera_params)
            
        return camera_params_list
    
    def _create_camera_params(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray = np.array([0, 0, 1]),
        fov: float = 60.0,
        width: int = 332,
        height: int = 324
    ) -> Dict[str, torch.Tensor]:
        """Create camera parameters"""
        # View matrix computation
        forward = look_at - position
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-6)
        
        up = np.cross(right, forward)
        
        # Camera to world
        R = np.stack([right, up, -forward], axis=0)
        
        # World to camera
        R_inv = R.T
        t = -R_inv @ position
        
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R_inv
        extrinsics[:3, 3] = t
        
        # Intrinsics
        focal = width / (2 * np.tan(np.radians(fov) / 2))
        intrinsics = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return {
            'intrinsics': torch.tensor(intrinsics),
            'extrinsics': torch.tensor(extrinsics)
        }
    
    def render_trajectory(
        self,
        gaussian_data,
        camera_trajectory: List[Dict[str, torch.Tensor]],
        wavelength_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, List[torch.Tensor]]:
        """Render views along camera trajectory"""
        rendered_views = []
        depth_maps = []
        
        for camera_params in camera_trajectory:
            rendered = self.renderer(gaussian_data, camera_params, wavelength_indices)
            rendered_views.append(rendered['spectral'])
            depth_maps.append(rendered['depth'])
            
        return {
            'images': rendered_views,
            'depths': depth_maps
        }


def compute_novel_view_metrics(
    rendered: torch.Tensor,
    ground_truth: torch.Tensor
) -> Dict[str, float]:
    """Compute metrics between rendered and ground truth views"""
    # MSE
    mse = torch.mean((rendered - ground_truth) ** 2).item()
    
    # PSNR
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-6))
    
    # Spectral Angle Mapper (SAM)
    rendered_norm = rendered / (torch.norm(rendered, dim=0, keepdim=True) + 1e-6)
    gt_norm = ground_truth / (torch.norm(ground_truth, dim=0, keepdim=True) + 1e-6)
    
    cos_angle = (rendered_norm * gt_norm).sum(dim=0)
    angle = torch.acos(torch.clamp(cos_angle, -0.999, 0.999))
    sam = torch.mean(angle).item() * 180 / np.pi  # Convert to degrees
    
    return {
        'mse': mse,
        'psnr': psnr,
        'sam_degrees': sam
    }


def export_3d_reconstruction(
    gaussian_data,
    output_dir: str,
    renderer = None,
    num_novel_views: int = 20
) -> Dict[str, str]:
    """Export 3D reconstruction in simple formats"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    # Extract point cloud
    converter = SimpleGaussianTo3D()
    pc_data = converter.extract_point_cloud(gaussian_data)
    
    # Save as PLY
    ply_path = os.path.join(output_dir, "pointcloud.ply")
    converter.save_as_ply(pc_data, ply_path)
    exported_files['pointcloud'] = ply_path
    
    # Save numpy arrays
    np_dir = os.path.join(output_dir, "numpy_data")
    converter.export_to_numpy(pc_data, np_dir)
    exported_files['numpy_data'] = np_dir
    
    # Generate and save novel views if renderer available
    if renderer is not None:
        nvs = SimpleNovelViewSynthesis(renderer)
        trajectory = nvs.generate_camera_trajectory(num_views=num_novel_views)
        novel_data = nvs.render_trajectory(gaussian_data, trajectory)
        
        # Save novel views
        views_dir = os.path.join(output_dir, "novel_views")
        os.makedirs(views_dir, exist_ok=True)
        
        for i, (img, depth) in enumerate(zip(novel_data['images'], novel_data['depths'])):
            # Convert spectral to RGB
            rgb = torch.stack([
                img[40],  # ~650nm
                img[60],  # ~550nm
                img[20]   # ~450nm
            ])
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            
            # Save as numpy
            np.save(os.path.join(views_dir, f"view_{i:03d}.npy"), 
                   rgb.permute(1, 2, 0).cpu().numpy())
            np.save(os.path.join(views_dir, f"depth_{i:03d}.npy"), 
                   depth.cpu().numpy())
            
        exported_files['novel_views'] = views_dir
    
    # Save summary
    summary = {
        'num_points': pc_data['points'].shape[0],
        'num_channels': pc_data['spectral_features'].shape[1],
        'wavelength_range': [float(pc_data['wavelengths'][0]), 
                            float(pc_data['wavelengths'][-1])],
        'exported_files': exported_files
    }
    
    import json
    with open(os.path.join(output_dir, 'reconstruction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n3D Reconstruction exported to: {output_dir}")
    print(f"  Points: {pc_data['points'].shape[0]}")
    print(f"  Spectral channels: {pc_data['spectral_features'].shape[1]}")
    
    return exported_files