import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.spectral_gaussian import SpectralGaussian3D
from core.hsi_vae import HyperspectralVAE
from core.spectral_renderer import SpectralGaussianRenderer, RenderingConfig
from core.depth_estimator import SpectralDepthEstimator
from core.reconstruction_simple import SimpleGaussianTo3D, SimpleNovelViewSynthesis, export_3d_reconstruction
from data.preprocessing import HSIPreprocessor, HSIConfig
from train import HSIGaussian3DModel

class HSIInference:
    """Inference pipeline for HSI 3D reconstruction"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        
        # Initialize model
        self.model = HSIGaussian3DModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        # Initialize preprocessor
        hsi_config = HSIConfig(
            num_channels=self.config["num_channels"],
            wavelength_range=tuple(self.config["wavelength_range"]),
            image_shape=(self.config["image_width"], self.config["image_height"])
        )
        self.preprocessor = HSIPreprocessor(hsi_config)
        
    def preprocess_input(self, hsi_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess input HSI images"""
        preprocessed_views = []
        
        for path in hsi_paths:
            # Load HSI cube
            if path.endswith(".npy"):
                hsi_cube = np.load(path)
            else:
                raise ValueError(f"Unsupported format: {path}")
                
            # Preprocess
            processed = self.preprocessor.preprocess(hsi_cube)
            preprocessed_views.append(processed["hsi"])
            
        # Stack views
        hsi_tensor = torch.stack(preprocessed_views).unsqueeze(0)  # [1, V, C, H, W]
        
        return {
            "hsi": hsi_tensor.to(self.device),
            "wavelengths": processed["wavelengths"].to(self.device)
        }
    
    def reconstruct(
        self,
        hsi_paths: List[str],
        camera_params: Dict[str, np.ndarray],
        output_wavelengths: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform 3D reconstruction from multi-view HSI
        
        Args:
            hsi_paths: List of paths to HSI files
            camera_params: Dict with 'intrinsics' and 'extrinsics' arrays
            output_wavelengths: Optional list of wavelengths to render (nm)
            
        Returns:
            Dictionary with reconstruction results
        """
        # Preprocess input
        batch = self.preprocess_input(hsi_paths)
        
        # Add camera parameters
        batch["intrinsics"] = torch.tensor(camera_params["intrinsics"]).unsqueeze(0).to(self.device)
        batch["extrinsics"] = torch.tensor(camera_params["extrinsics"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(batch)
            
            # Extract Gaussian parameters
            gaussian_data = outputs["gaussian_data"]
            
            # Render at specific wavelengths if requested
            if output_wavelengths:
                rendered_wavelengths = {}
                
                for wavelength in output_wavelengths:
                    # Find closest wavelength index
                    wavelength_tensor = torch.tensor(wavelength).to(self.device)
                    idx = torch.argmin(torch.abs(batch["wavelengths"] - wavelength_tensor))
                    
                    # Render at this wavelength
                    rendered = self.model.renderer(
                        gaussian_data,
                        {"intrinsics": batch["intrinsics"][0, 0],
                         "extrinsics": batch["extrinsics"][0, 0]},
                        wavelength_indices=idx.unsqueeze(0)
                    )
                    
                    rendered_wavelengths[f"{wavelength:.0f}nm"] = rendered["spectral"].squeeze(0)
                    
                outputs["rendered_wavelengths"] = rendered_wavelengths
                
        return outputs
    
    def render_novel_view(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        wavelength_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Render from a novel viewpoint
        
        Args:
            gaussian_data: Gaussian parameters from reconstruction
            camera_intrinsics: [3, 3] intrinsic matrix
            camera_extrinsics: [4, 4] extrinsic matrix
            wavelength_indices: Optional wavelength indices to render
            
        Returns:
            Rendered images
        """
        camera_params = {
            "intrinsics": torch.tensor(camera_intrinsics).to(self.device),
            "extrinsics": torch.tensor(camera_extrinsics).to(self.device)
        }
        
        with torch.no_grad():
            rendered = self.model.renderer(
                gaussian_data,
                camera_params,
                wavelength_indices
            )
            
        return rendered
    
    def extract_depth_map(self, hsi_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract depth map from single HSI image"""
        # Preprocess
        processed = self.preprocessor.preprocess(hsi_image)
        hsi_tensor = processed["hsi"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth_output = self.model.depth_estimator(hsi_tensor)
            
        return {
            "depth": depth_output["depth"].cpu().numpy()[0],
            "uncertainty": depth_output["uncertainty"].cpu().numpy()[0]
        }
    
    def export_point_cloud(
        self,
        gaussian_data: Dict[str, torch.Tensor],
        output_path: str,
        num_points: int = 100000
    ):
        """Export reconstructed point cloud"""
        positions = gaussian_data["positions"].cpu().numpy()
        spectral_features = gaussian_data["spectral_features"].cpu().numpy()
        opacities = gaussian_data["opacities"].cpu().numpy()
        
        # Filter by opacity
        opacity_threshold = 0.1
        mask = opacities.squeeze() > opacity_threshold
        positions = positions[mask]
        spectral_features = spectral_features[mask]
        
        # Sample if too many points
        if positions.shape[0] > num_points:
            indices = np.random.choice(positions.shape[0], num_points, replace=False)
            positions = positions[indices]
            spectral_features = spectral_features[indices]
            
        # Convert spectral to RGB for visualization (using specific wavelengths)
        # R: 650nm, G: 550nm, B: 450nm
        wavelengths = gaussian_data["wavelengths"].cpu().numpy()
        r_idx = np.argmin(np.abs(wavelengths - 650))
        g_idx = np.argmin(np.abs(wavelengths - 550))
        b_idx = np.argmin(np.abs(wavelengths - 450))
        
        colors = spectral_features[:, [r_idx, g_idx, b_idx]]
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        
        # Save as PLY
        from plyfile import PlyData, PlyElement
        
        vertex_data = np.zeros(
            positions.shape[0],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        
        vertex_data['x'] = positions[:, 0]
        vertex_data['y'] = positions[:, 1]
        vertex_data['z'] = positions[:, 2]
        vertex_data['red'] = (colors[:, 0] * 255).astype(np.uint8)
        vertex_data['green'] = (colors[:, 1] * 255).astype(np.uint8)
        vertex_data['blue'] = (colors[:, 2] * 255).astype(np.uint8)
        
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(output_path)
        
        print(f"Exported {positions.shape[0]} points to {output_path}")

def visualize_results(outputs: Dict, save_path: Optional[str] = None):
    """Visualize reconstruction results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Rendered views at different wavelengths
    if "rendered_wavelengths" in outputs:
        wavelengths = list(outputs["rendered_wavelengths"].keys())[:3]
        for i, wl in enumerate(wavelengths):
            img = outputs["rendered_wavelengths"][wl].cpu().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Rendered at {wl}")
            axes[0, i].axis('off')
            
    # Depth maps
    if "depth_estimates" in outputs:
        for i in range(min(3, len(outputs["depth_estimates"]))):
            depth = outputs["depth_estimates"][i]["depth"].cpu().numpy()[0]
            axes[1, i].imshow(depth, cmap='viridis')
            axes[1, i].set_title(f"Depth View {i+1}")
            axes[1, i].axis('off')
            
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="HSI 3D reconstruction inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--hsi_paths", nargs="+", required=True, help="Paths to input HSI files")
    parser.add_argument("--camera_params", type=str, required=True, help="Path to camera parameters JSON")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Output directory")
    parser.add_argument("--wavelengths", nargs="+", type=float, help="Wavelengths to render (nm)")
    parser.add_argument("--export_ply", action="store_true", help="Export point cloud as PLY")
    parser.add_argument("--export_mesh", action="store_true", help="Export mesh reconstruction")
    parser.add_argument("--export_novel_views", action="store_true", help="Export novel view synthesis")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera parameters
    import json
    with open(args.camera_params, "r") as f:
        camera_params = json.load(f)
        
    # Initialize inference pipeline
    pipeline = HSIInference(args.checkpoint)
    
    # Perform reconstruction
    print("Performing 3D reconstruction...")
    outputs = pipeline.reconstruct(
        args.hsi_paths,
        camera_params,
        args.wavelengths
    )
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(outputs, output_dir / "reconstruction_results.png")
    
    # Export 3D reconstruction in multiple formats
    if args.export_ply or args.export_mesh:
        print("Exporting 3D reconstruction...")
        export_3d_reconstruction(
            outputs["gaussian_data"],
            str(output_dir),
            renderer=pipeline.model.renderer if args.export_novel_views else None,
            num_novel_views=20
        )
        
    # Extract depth maps
    print("Extracting depth maps...")
    for i, hsi_path in enumerate(args.hsi_paths[:3]):
        hsi_cube = np.load(hsi_path)
        depth_data = pipeline.extract_depth_map(hsi_cube)
        
        # Save depth map
        np.save(output_dir / f"depth_map_{i}.npy", depth_data["depth"])
        
        # Visualize depth
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(depth_data["depth"], cmap='viridis')
        plt.colorbar()
        plt.title("Depth Map")
        
        plt.subplot(1, 2, 2)
        plt.imshow(depth_data["uncertainty"], cmap='hot')
        plt.colorbar()
        plt.title("Uncertainty")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"depth_visualization_{i}.png")
        plt.close()
        
    print(f"Results saved to {output_dir}")
    
if __name__ == "__main__":
    main()