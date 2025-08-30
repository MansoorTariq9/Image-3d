"""
3D Reconstruction utilities for HSI Gaussian Splatting
Includes point cloud export, mesh extraction, and novel view synthesis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

# Optional imports with fallbacks
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class GaussianTo3D:
    """Convert Gaussian splats to explicit 3D representations"""
    
    def __init__(self, opacity_threshold: float = 0.1, voxel_size: float = 0.01):
        self.opacity_threshold = opacity_threshold
        self.voxel_size = voxel_size
        
    def extract_point_cloud(
        self, 
        gaussian_data: Dict[str, torch.Tensor],
        wavelength_nm: Optional[float] = None,
        color_wavelengths: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract point cloud from Gaussian representation
        
        Args:
            gaussian_data: Dictionary with Gaussian parameters
            wavelength_nm: Single wavelength for intensity (optional)
            color_wavelengths: RGB wavelengths for color visualization
            
        Returns:
            Dictionary with 'points', 'colors', 'normals', 'spectral_features'
        """
        # Extract data
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
        
        # Generate color visualization
        if color_wavelengths is None:
            # Default RGB wavelengths
            color_wavelengths = [650.0, 550.0, 450.0]  # R, G, B
            
        colors = np.zeros((positions.shape[0], 3))
        for i, wl in enumerate(color_wavelengths):
            # Find closest wavelength
            idx = np.argmin(np.abs(wavelengths - wl))
            colors[:, i] = spectral_features[:, idx]
            
        # Normalize colors to 0-1
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        
        # Estimate normals from local neighborhood
        normals = self._estimate_normals(positions)
        
        # Single wavelength intensity if requested
        intensity = None
        if wavelength_nm is not None:
            idx = np.argmin(np.abs(wavelengths - wavelength_nm))
            intensity = spectral_features[:, idx]
            
        return {
            'points': positions,
            'colors': colors,
            'normals': normals,
            'scales': scales,
            'spectral_features': spectral_features,
            'intensity': intensity
        }
    
    def _estimate_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """Estimate normals using PCA on local neighborhoods"""
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        _, indices = nbrs.kneighbors(points)
        
        normals = np.zeros_like(points)
        
        for i, neighbors in enumerate(indices):
            # Get neighborhood points
            neighborhood = points[neighbors]
            
            # Center points
            centroid = np.mean(neighborhood, axis=0)
            centered = neighborhood - centroid
            
            # PCA to find normal
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Normal is eigenvector with smallest eigenvalue
            normal = eigvecs[:, 0]
            
            # Ensure consistent orientation
            if normal[2] < 0:
                normal = -normal
                
            normals[i] = normal
            
        return normals
    
    def extract_mesh(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        method: str = "poisson",
        depth: int = 9
    ):
        """Extract mesh from Gaussian representation
        
        Args:
            gaussian_data: Gaussian parameters
            method: 'poisson' or 'marching_cubes'
            depth: Octree depth for Poisson reconstruction
            
        Returns:
            Trimesh object or dict with mesh data
        """
        # First extract point cloud
        pc_data = self.extract_point_cloud(gaussian_data)
        points = pc_data['points']
        normals = pc_data['normals']
        colors = pc_data['colors']
        
        if method == "poisson" and OPEN3D_AVAILABLE:
            # Use Open3D for Poisson reconstruction
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            
            # Convert to trimesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # Transfer colors
            vertex_colors = self._transfer_colors_to_vertices(
                vertices, points, colors
            )
            
            return trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
            
        elif method == "marching_cubes":
            # Voxelize and use marching cubes
            voxel_grid, voxel_colors = self._voxelize_gaussians(
                gaussian_data, resolution=128
            )
            
            # Apply marching cubes
            from skimage import measure
            verts, faces, normals, _ = measure.marching_cubes(
                voxel_grid, level=0.5
            )
            
            # Scale vertices back to world coordinates
            scale = (points.max(axis=0) - points.min(axis=0)) / 128
            offset = points.min(axis=0)
            verts = verts * scale + offset
            
            # Get vertex colors
            vertex_colors = self._sample_voxel_colors(
                verts, voxel_colors, scale, offset
            )
            
            return trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=normals,
                vertex_colors=vertex_colors
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _voxelize_gaussians(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        resolution: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Gaussians to voxel grid"""
        positions = gaussian_data['positions'].detach().cpu().numpy()
        opacities = gaussian_data['opacities'].detach().cpu().numpy().squeeze()
        scales = gaussian_data['scales'].detach().cpu().numpy()
        
        # Filter by opacity
        valid_mask = opacities > self.opacity_threshold
        positions = positions[valid_mask]
        scales = scales[valid_mask]
        opacities = opacities[valid_mask]
        
        # Compute bounds
        min_bound = positions.min(axis=0) - scales.max()
        max_bound = positions.max(axis=0) + scales.max()
        
        # Create voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution))
        voxel_colors = np.zeros((resolution, resolution, resolution, 3))
        
        # Voxel coordinates
        x = np.linspace(min_bound[0], max_bound[0], resolution)
        y = np.linspace(min_bound[1], max_bound[1], resolution)
        z = np.linspace(min_bound[2], max_bound[2], resolution)
        
        # Fill voxels
        for i, pos in enumerate(positions):
            # Find affected voxels
            scale = scales[i]
            opacity = opacities[i]
            
            x_idx = np.where(np.abs(x - pos[0]) < scale[0] * 3)[0]
            y_idx = np.where(np.abs(y - pos[1]) < scale[1] * 3)[0]
            z_idx = np.where(np.abs(z - pos[2]) < scale[2] * 3)[0]
            
            for xi in x_idx:
                for yi in y_idx:
                    for zi in z_idx:
                        # Gaussian weight
                        dist = np.array([
                            (x[xi] - pos[0]) / scale[0],
                            (y[yi] - pos[1]) / scale[1],
                            (z[zi] - pos[2]) / scale[2]
                        ])
                        weight = opacity * np.exp(-0.5 * np.sum(dist**2))
                        
                        voxel_grid[xi, yi, zi] += weight
                        
        return voxel_grid, voxel_colors
    
    def _transfer_colors_to_vertices(
        self,
        vertices: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray
    ) -> np.ndarray:
        """Transfer colors from points to mesh vertices"""
        nbrs = NearestNeighbors(n_neighbors=1).fit(points)
        _, indices = nbrs.kneighbors(vertices)
        return colors[indices.squeeze()]
    
    def _sample_voxel_colors(
        self,
        vertices: np.ndarray,
        voxel_colors: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray
    ) -> np.ndarray:
        """Sample colors from voxel grid"""
        # Convert vertices to voxel coordinates
        voxel_coords = (vertices - offset) / scale
        
        # Trilinear interpolation would go here
        # For now, nearest neighbor
        voxel_coords = np.clip(voxel_coords.astype(int), 0, voxel_colors.shape[0]-1)
        
        vertex_colors = voxel_colors[
            voxel_coords[:, 0],
            voxel_coords[:, 1],
            voxel_coords[:, 2]
        ]
        
        return vertex_colors


class NovelViewSynthesis:
    """Generate and validate novel views"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        
    def generate_novel_views(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        num_views: int = 20,
        radius: float = 5.0,
        height: float = 2.0,
        look_at: np.ndarray = np.array([0, 0, 0])
    ) -> Dict[str, List[torch.Tensor]]:
        """Generate novel views in a circular pattern
        
        Returns:
            Dictionary with 'images', 'depths', 'camera_params'
        """
        novel_views = []
        depth_maps = []
        camera_params_list = []
        
        for i in range(num_views):
            # Camera position
            angle = 2 * np.pi * i / num_views
            cam_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Camera matrix
            camera_params = self._create_camera_params(cam_pos, look_at)
            camera_params_list.append(camera_params)
            
            # Render view
            rendered = self.renderer(gaussian_data, camera_params)
            
            novel_views.append(rendered['spectral'])
            depth_maps.append(rendered['depth'])
            
        return {
            'images': novel_views,
            'depths': depth_maps,
            'camera_params': camera_params_list
        }
    
    def _create_camera_params(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray = np.array([0, 0, 1]),
        fov: float = 60.0,
        width: int = 332,
        height: int = 324
    ) -> Dict[str, torch.Tensor]:
        """Create camera parameters from position and look-at point"""
        # View matrix
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Camera to world
        R = np.stack([right, up, -forward], axis=0)
        
        # World to camera
        R_inv = R.T
        t = -R_inv @ position
        
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_inv
        extrinsics[:3, 3] = t
        
        # Intrinsics
        focal = width / (2 * np.tan(np.radians(fov) / 2))
        intrinsics = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ])
        
        return {
            'intrinsics': torch.tensor(intrinsics, dtype=torch.float32),
            'extrinsics': torch.tensor(extrinsics, dtype=torch.float32)
        }
    
    def validate_novel_views(
        self,
        rendered_views: List[torch.Tensor],
        ground_truth_views: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for novel view synthesis"""
        metrics = {
            'psnr': [],
            'ssim': [],
            'sam': []  # Spectral Angle Mapper
        }
        
        for rendered, gt in zip(rendered_views, ground_truth_views):
            # PSNR
            mse = torch.mean((rendered - gt) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'].append(psnr.item())
            
            # SSIM (simplified)
            ssim = self._compute_ssim(rendered, gt)
            metrics['ssim'].append(ssim)
            
            # SAM
            sam = self._compute_sam(rendered, gt)
            metrics['sam'].append(sam)
            
        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Simplified SSIM computation"""
        # This is a placeholder - use proper SSIM implementation
        return torch.nn.functional.cosine_similarity(
            img1.flatten(), img2.flatten(), dim=0
        ).item()
    
    def _compute_sam(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Spectral Angle Mapper"""
        # Normalize
        img1_norm = img1 / (torch.norm(img1, dim=0, keepdim=True) + 1e-6)
        img2_norm = img2 / (torch.norm(img2, dim=0, keepdim=True) + 1e-6)
        
        # Compute angle
        cos_angle = (img1_norm * img2_norm).sum(dim=0)
        angle = torch.acos(torch.clamp(cos_angle, -0.999, 0.999))
        
        return angle.mean().item()


def export_reconstruction(
    gaussian_data: Dict[str, torch.Tensor],
    output_path: str,
    export_mesh: bool = True,
    export_pointcloud: bool = True,
    export_novel_views: bool = True,
    renderer = None
) -> Dict[str, str]:
    """Export 3D reconstruction in various formats
    
    Returns:
        Dictionary with paths to exported files
    """
    from pathlib import Path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    # Initialize converter
    converter = GaussianTo3D()
    
    # Export point cloud
    if export_pointcloud:
        pc_data = converter.extract_point_cloud(gaussian_data)
        
        # Save as PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data['points'])
        pcd.colors = o3d.utility.Vector3dVector(pc_data['colors'])
        pcd.normals = o3d.utility.Vector3dVector(pc_data['normals'])
        
        ply_path = output_dir / "pointcloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        exported_files['pointcloud'] = str(ply_path)
        
        # Save spectral features as NPY
        npy_path = output_dir / "spectral_features.npy"
        np.save(npy_path, pc_data['spectral_features'])
        exported_files['spectral_features'] = str(npy_path)
    
    # Export mesh
    if export_mesh:
        try:
            mesh = converter.extract_mesh(gaussian_data, method='poisson')
            mesh_path = output_dir / "mesh.obj"
            mesh.export(str(mesh_path))
            exported_files['mesh'] = str(mesh_path)
        except Exception as e:
            print(f"Mesh extraction failed: {e}")
    
    # Export novel views
    if export_novel_views and renderer is not None:
        nvs = NovelViewSynthesis(renderer)
        novel_data = nvs.generate_novel_views(gaussian_data)
        
        # Save as video or image sequence
        views_dir = output_dir / "novel_views"
        views_dir.mkdir(exist_ok=True)
        
        for i, (img, depth) in enumerate(zip(novel_data['images'], novel_data['depths'])):
            # Convert to RGB for visualization
            rgb = img[[40, 60, 20], :, :]  # Sample bands for RGB
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            
            # Save image
            import matplotlib.pyplot as plt
            plt.imsave(views_dir / f"view_{i:03d}.png", rgb.permute(1, 2, 0).cpu().numpy())
            
            # Save depth
            np.save(views_dir / f"depth_{i:03d}.npy", depth.cpu().numpy())
            
        exported_files['novel_views'] = str(views_dir)
    
    print(f"Exported reconstruction to: {output_dir}")
    for key, path in exported_files.items():
        print(f"  {key}: {path}")
        
    return exported_files