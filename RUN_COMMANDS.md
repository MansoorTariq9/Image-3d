# HSI Gaussian 3D - Complete Run Guide

## 🚀 Quick Start Commands

### 1. **Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib opencv-python tqdm pyyaml
```

### 2. **Prepare Your ENVI Data**
```bash
# Convert ENVI files to numpy format
python convert_envi_to_numpy.py \
  --input_dir ./sample_data \
  --output_dir ./processed_data
```

### 3. **View Your HSI Data**
```bash
# Visualize single ENVI file
python view_hsi_data.py ./sample_data/0degree_001/0degree_raw

# Visualize all angles
python view_hsi_data.py --all
```

### 4. **Train the Model**
```bash
# Basic training
python train.py \
  --config config.yaml \
  --data_root ./processed_data \
  --output_dir ./outputs

# With experiment tracking
python train.py \
  --config config.yaml \
  --data_root ./processed_data \
  --output_dir ./outputs \
  --wandb_project hsi_gaussian_3d
```

### 5. **Run Inference & 3D Reconstruction**
```bash
# Basic inference with point cloud export
python inference.py \
  --checkpoint ./outputs/best_model.pth \
  --hsi_paths ./processed_data/scene_001/hsi_*.npy \
  --camera_params ./processed_data/scene_001/camera_poses.json \
  --output_dir ./inference_output \
  --export_ply

# Full 3D reconstruction with novel views
python inference.py \
  --checkpoint ./outputs/best_model.pth \
  --hsi_paths ./processed_data/scene_001/hsi_*.npy \
  --camera_params ./processed_data/scene_001/camera_poses.json \
  --output_dir ./inference_output \
  --export_ply \
  --export_novel_views \
  --wavelengths 450 550 650 800
```

### 6. **Demo Scripts**
```bash
# Run 3D reconstruction demo
python demo_3d_reconstruction.py

# Test components
python test_3d_reconstruction.py
```

## 📁 Data Preparation Scripts

### Convert ENVI to Training Format
```python
# create_training_data.py
from data.envi_loader import convert_envi_to_standard_format

# Convert your ENVI data
convert_envi_to_standard_format(
    envi_path="./sample_data",
    output_dir="./processed_data"
)
```

### Create Camera Parameters (if not using SuperGlue)
```python
# generate_cameras.py
import numpy as np
import json

# Generate circular camera arrangement
num_views = 14  # Number of angles in your data
cameras = {
    "intrinsics": [],
    "extrinsics": []
}

for i in range(num_views):
    # Camera intrinsics (adjust focal length as needed)
    K = np.array([
        [300, 0, 166],    # fx, 0, cx
        [0, 300, 162],    # 0, fy, cy
        [0, 0, 1]
    ])
    cameras["intrinsics"].append(K.tolist())
    
    # Camera extrinsics (circular arrangement)
    angle = 2 * np.pi * i / num_views
    radius = 5.0
    height = 2.0
    
    # Camera position
    pos = np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        height
    ])
    
    # Look at origin
    forward = -pos / np.linalg.norm(pos)
    right = np.cross([0, 0, 1], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    
    # World to camera transform
    R = np.stack([right, up, -forward], axis=0).T
    t = -R @ pos
    
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    
    cameras["extrinsics"].append(E.tolist())

# Save camera parameters
with open("camera_poses.json", "w") as f:
    json.dump(cameras, f, indent=2)
```

## 🎯 Step-by-Step Workflow

### Step 1: Prepare ENVI Data
```bash
# Your ENVI data structure should look like:
sample_data/
├── 0degree_001/
│   ├── 0degree_raw      # Binary data
│   └── 0degree_raw.hdr  # Header file
├── 8degree_001/
│   ├── 8degree_raw
│   └── 8degree_raw.hdr
└── ... (other angles)
```

### Step 2: Convert to NumPy
```bash
python -c "
from data.envi_loader import ENVIMultiAngleLoader, convert_envi_to_standard_format
convert_envi_to_standard_format('./sample_data', './processed_data')
"
```

### Step 3: Train Model
```bash
# Small test run
python train.py \
  --config config.yaml \
  --data_root ./processed_data \
  --output_dir ./test_run \
  --num_epochs 5

# Full training
python train.py \
  --config config.yaml \
  --data_root ./processed_data \
  --output_dir ./full_training \
  --num_epochs 100
```

### Step 4: Extract 3D Model
```bash
# After training, extract 3D reconstruction
python inference.py \
  --checkpoint ./full_training/best_model.pth \
  --hsi_paths ./sample_data/*/[!zero]*_raw \
  --camera_params ./processed_data/sample_data/camera_poses.json \
  --output_dir ./3d_output \
  --export_ply \
  --export_novel_views
```

## 📊 Expected Outputs

```
3d_output/
├── pointcloud.ply              # View in MeshLab/CloudCompare
├── numpy_data/
│   ├── points.npy             # 3D positions
│   ├── spectral_features.npy  # 120 channels per point
│   └── metadata.json
├── novel_views/
│   ├── view_000.npy           # Rendered views
│   └── depth_000.npy          # Depth maps
├── reconstruction_results.png  # Visualization
└── reconstruction_summary.json # Statistics
```

## 🛠️ Troubleshooting

### If ENVI loading fails:
```bash
# Test ENVI loader directly
python -c "
from data.envi_loader import ENVIReader
reader = ENVIReader()
data, header = reader.read_envi('./sample_data/0degree_001/0degree_raw')
print(f'Loaded data shape: {data.shape}')
"
```

### If training crashes:
```bash
# Reduce batch size in config.yaml
batch_size: 1  # Instead of 4

# Reduce number of Gaussians
num_gaussians: 10000  # Instead of 50000
```

### View results:
```bash
# Install viewer (optional)
pip install open3d

# View point cloud
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('./3d_output/pointcloud.ply')
o3d.visualization.draw_geometries([pcd])
"
```

## 🎨 Visualization Commands

### Plot spectral signatures:
```bash
python -c "
import numpy as np
import matplotlib.pyplot as plt

# Load spectral features
features = np.load('./3d_output/numpy_data/spectral_features.npy')
wavelengths = np.linspace(400, 1000, 120)

# Plot random spectra
for i in range(10):
    idx = np.random.randint(features.shape[0])
    plt.plot(wavelengths, features[idx], alpha=0.5)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance')
plt.title('Sample Spectral Signatures')
plt.savefig('spectral_signatures.png')
"
```

## 💡 Tips

1. **Memory Issues**: Reduce `num_gaussians` and `point_cloud_size` in config.yaml
2. **Faster Training**: Use `--num_views 2` instead of 4 for testing
3. **Better Results**: Ensure good camera calibration or use SuperGlue
4. **Wavelength Selection**: Choose wavelengths with high contrast for visualization