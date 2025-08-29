import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class ENVIReader:
    """Reader for ENVI format hyperspectral images"""
    
    def __init__(self):
        self.header_info = {}
        self.wavelengths = None
        
    def parse_header(self, header_path: str) -> Dict:
        """Parse ENVI header file (.hdr)"""
        header_info = {}
        
        with open(header_path, 'r') as f:
            lines = f.readlines()
            
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and ENVI keyword
            if not line or line == 'ENVI':
                continue
                
            # Check if this is a new key-value pair
            if '=' in line and not line.startswith(' '):
                # Save previous key-value if exists
                if current_key:
                    value_str = ' '.join(current_value)
                    header_info[current_key] = self._parse_value(value_str)
                    
                # Parse new key-value
                parts = line.split('=', 1)
                current_key = parts[0].strip()
                current_value = [parts[1].strip()]
                
            else:
                # Continuation of previous value
                current_value.append(line)
                
        # Don't forget the last key-value pair
        if current_key:
            value_str = ' '.join(current_value)
            header_info[current_key] = self._parse_value(value_str)
            
        return header_info
    
    def _parse_value(self, value_str: str):
        """Parse ENVI header value string"""
        value_str = value_str.strip()
        
        # Handle bracketed lists
        if value_str.startswith('{') and value_str.endswith('}'):
            # Remove brackets and split by comma
            value_str = value_str[1:-1]
            values = [v.strip() for v in value_str.split(',')]
            
            # Try to convert to numbers
            try:
                return [float(v) for v in values]
            except ValueError:
                return values
                
        # Handle single values
        try:
            # Try integer first
            if '.' not in value_str:
                return int(value_str)
            # Then float
            return float(value_str)
        except ValueError:
            # Return as string
            return value_str
            
    def read_envi(self, data_path: str, header_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """Read ENVI format data
        
        Args:
            data_path: Path to the binary data file
            header_path: Path to the header file (if None, assumes .hdr extension)
            
        Returns:
            data: Numpy array with shape based on interleave format
            header: Parsed header dictionary
        """
        # Determine header path
        if header_path is None:
            base_path = os.path.splitext(data_path)[0]
            header_path = base_path + '.hdr'
            
        # Parse header
        header = self.parse_header(header_path)
        self.header_info = header
        
        # Extract dimensions
        samples = int(header['samples'])  # Width
        lines = int(header['lines'])      # Height  
        bands = int(header['bands'])      # Spectral channels
        
        # Data type mapping
        dtype_map = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            12: np.uint16,
            13: np.uint32,
            14: np.int64,
            15: np.uint64
        }
        
        data_type = int(header.get('data type', 4))
        dtype = dtype_map.get(data_type, np.float32)
        
        # Byte order
        byte_order = int(header.get('byte order', 0))
        if byte_order == 0:
            dtype = np.dtype(dtype).newbyteorder('<')  # Little endian
        else:
            dtype = np.dtype(dtype).newbyteorder('>')  # Big endian
            
        # Header offset
        offset = int(header.get('header offset', 0))
        
        # Read binary data
        with open(data_path, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype)
            
        # Reshape based on interleave format
        interleave = header.get('interleave', 'bsq').lower()
        
        if interleave == 'bsq':  # Band sequential (band, row, column)
            data = data.reshape((bands, lines, samples))
            # Convert to (height, width, bands) for easier use
            data = np.transpose(data, (1, 2, 0))
            
        elif interleave == 'bil':  # Band interleaved by line (row, band, column)
            data = data.reshape((lines, bands, samples))
            # Convert to (height, width, bands)
            data = np.transpose(data, (0, 2, 1))
            
        elif interleave == 'bip':  # Band interleaved by pixel (row, column, band)
            data = data.reshape((lines, samples, bands))
            # Already in (height, width, bands) format
            
        else:
            raise ValueError(f"Unknown interleave format: {interleave}")
            
        # Store wavelengths if available
        if 'wavelength' in header:
            self.wavelengths = np.array(header['wavelength'])
            
        return data, header
    
    def read_envi_subset(self, data_path: str, 
                        bands: Optional[List[int]] = None,
                        spatial_subset: Optional[Tuple[int, int, int, int]] = None,
                        header_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """Read subset of ENVI data
        
        Args:
            data_path: Path to the binary data file
            bands: List of band indices to read (None = all bands)
            spatial_subset: (x_start, y_start, x_end, y_end) spatial subset
            header_path: Path to the header file
            
        Returns:
            data: Subset numpy array
            header: Parsed header dictionary
        """
        # First read full data (can be optimized for large files)
        data, header = self.read_envi(data_path, header_path)
        
        # Apply spatial subset
        if spatial_subset:
            x1, y1, x2, y2 = spatial_subset
            data = data[y1:y2, x1:x2, :]
            
        # Apply band subset
        if bands:
            data = data[:, :, bands]
            if self.wavelengths is not None:
                self.wavelengths = self.wavelengths[bands]
                
        return data, header


class ENVIMultiAngleLoader:
    """Loader for multi-angle ENVI hyperspectral data"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.reader = ENVIReader()
        self.angles = []
        self.data_dict = {}
        
    def scan_directory(self) -> List[str]:
        """Scan directory for angle-based ENVI files"""
        angle_pattern = re.compile(r'(\d+)degree')
        found_angles = []
        
        for item in self.base_path.iterdir():
            if item.is_dir():
                match = angle_pattern.search(item.name)
                if match:
                    angle = int(match.group(1))
                    found_angles.append((angle, item))
                    
        # Sort by angle
        found_angles.sort(key=lambda x: x[0])
        self.angles = [angle for angle, _ in found_angles]
        self.angle_paths = {angle: path for angle, path in found_angles}
        
        return self.angles
    
    def load_angle_data(self, angle: int) -> np.ndarray:
        """Load data for a specific angle"""
        if angle not in self.angle_paths:
            raise ValueError(f"Angle {angle} not found in dataset")
            
        angle_dir = self.angle_paths[angle]
        
        # Find ENVI files in the directory
        data_files = list(angle_dir.glob("*_raw"))
        if not data_files:
            # Try without _raw suffix
            data_files = [f for f in angle_dir.iterdir() 
                         if f.is_file() and not f.suffix in ['.hdr', '.txt', '.json']]
                         
        if not data_files:
            raise FileNotFoundError(f"No ENVI data file found in {angle_dir}")
            
        data_file = data_files[0]
        
        # Read ENVI data
        data, header = self.reader.read_envi(str(data_file))
        
        return data
    
    def load_all_angles(self, angles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Load data for multiple angles
        
        Args:
            angles: List of angles to load (None = all available)
            
        Returns:
            Dictionary mapping angle to HSI data
        """
        if angles is None:
            angles = self.angles
            
        data_dict = {}
        for angle in angles:
            print(f"Loading data for {angle}° view...")
            data_dict[angle] = self.load_angle_data(angle)
            
        self.data_dict = data_dict
        return data_dict
    
    def get_camera_poses(self, angles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Generate camera poses for each angle
        
        Assumes circular camera arrangement around the object
        """
        if angles is None:
            angles = self.angles
            
        poses = {}
        for angle in angles:
            # Convert angle to radians
            theta = np.radians(angle)
            
            # Assume camera at fixed radius, varying azimuth
            radius = 1.0  # Can be adjusted based on actual setup
            height = 0.5  # Camera height
            
            # Camera position
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = height
            
            camera_pos = np.array([x, y, z])
            
            # Look at origin
            look_at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Compute camera matrix
            forward = look_at - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            # Camera to world rotation
            R = np.stack([right, up, -forward], axis=0)
            
            # World to camera transform
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ camera_pos
            
            poses[angle] = pose
            
        return poses
    
    def get_wavelengths(self) -> Optional[np.ndarray]:
        """Get wavelength array from the loaded data"""
        return self.reader.wavelengths


def convert_envi_to_standard_format(envi_path: str, output_dir: str):
    """Convert ENVI multi-angle data to standard format expected by the dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize loader
    loader = ENVIMultiAngleLoader(envi_path)
    angles = loader.scan_directory()
    
    print(f"Found {len(angles)} angle views: {angles}")
    
    # Load all angle data
    data_dict = loader.load_all_angles()
    
    # Get camera poses
    poses = loader.get_camera_poses()
    
    # Create scene directory
    scene_name = Path(envi_path).name
    scene_dir = output_path / scene_name
    scene_dir.mkdir(exist_ok=True)
    
    # Save HSI data as numpy arrays
    view_idx = 0
    intrinsics_list = []
    extrinsics_list = []
    
    for angle in sorted(angles):
        hsi_data = data_dict[angle]
        
        # Save HSI cube
        np.save(scene_dir / f"hsi_{view_idx:03d}.npy", hsi_data)
        
        # Generate simple intrinsics (can be refined with actual calibration)
        height, width, _ = hsi_data.shape
        focal_length = max(width, height) * 0.8
        K = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        intrinsics_list.append(K)
        extrinsics_list.append(poses[angle])
        
        view_idx += 1
        
    # Save camera parameters
    import json
    camera_data = {
        "intrinsics": np.stack(intrinsics_list).tolist(),
        "extrinsics": np.stack(extrinsics_list).tolist(),
        "angles": angles,
        "wavelengths": loader.get_wavelengths().tolist() if loader.get_wavelengths() is not None else None
    }
    
    with open(scene_dir / "camera_poses.json", "w") as f:
        json.dump(camera_data, f, indent=2)
        
    # Create scene list
    scene_info = {
        "name": scene_name,
        "path": str(scene_dir),
        "num_views": len(angles)
    }
    
    # Save as both train and val for now (can be split later)
    for split in ["train", "val"]:
        with open(output_path / f"{split}_scenes.json", "w") as f:
            json.dump([scene_info], f, indent=2)
            
    print(f"Converted ENVI data saved to {output_path}")
    print(f"Scene: {scene_name}")
    print(f"Views: {len(angles)}")
    
    return output_path


if __name__ == "__main__":
    # Test ENVI reader
    test_path = "/home/dell/upwork/hsi_gaussian_3d/sample_data/0degree_001/0degree_raw"
    
    reader = ENVIReader()
    data, header = reader.read_envi(test_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Wavelengths: {len(reader.wavelengths)} bands")
    print(f"Wavelength range: {reader.wavelengths[0]:.2f} - {reader.wavelengths[-1]:.2f} nm")