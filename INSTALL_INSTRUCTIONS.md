# Installation Instructions for HSI Gaussian 3D

## Required Python Packages

### Core Dependencies:
```bash
# Essential packages
numpy>=1.21.0          # Array operations
scipy>=1.7.0           # Loading MATLAB files, scientific computing
matplotlib>=3.5.0      # Visualization
pillow>=9.0.0         # Image processing
pyyaml>=6.0           # Configuration files
tqdm>=4.65.0          # Progress bars

# Deep Learning Framework
torch>=2.0.0          # PyTorch for neural networks
torchvision>=0.15.0   # Vision utilities

# Optional but recommended
opencv-python>=4.7.0  # Image resizing and processing
h5py>=3.8.0          # HDF5 file support
plyfile>=0.7.4       # PLY point cloud export
wandb>=0.13.0        # Experiment tracking (optional)
```

## Installation Steps:

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. **Install all packages at once:**
```bash
pip install numpy scipy matplotlib pillow pyyaml tqdm torch torchvision opencv-python h5py plyfile
```

3. **Or install from requirements.txt:**
```bash
pip install -r requirements.txt
```

## Minimal Installation (for testing without GPU):

If you just want to test the data generation and preprocessing:
```bash
pip install numpy scipy matplotlib pyyaml
```

## GPU Support (Optional):

For GPU acceleration with CUDA:
```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## After Installation:

1. **Generate test data:**
```bash
python generate_sample_data.py
```

2. **Run tests:**
```bash
python test_full_pipeline.py
```

3. **Start training (requires all packages):**
```bash
python train.py --config config.yaml --data_root ./sample_hsi_data --output_dir ./test_outputs
```

## Package Versions Known to Work:

```
numpy==1.24.3
scipy==1.11.0
matplotlib==3.7.1
pillow==10.0.0
pyyaml==6.0.1
tqdm==4.65.0
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0.74
h5py==3.9.0
plyfile==0.9
```

## Troubleshooting:

1. **If pip is slow**, use a faster mirror:
```bash
pip install -i https://pypi.org/simple/ numpy scipy matplotlib
```

2. **For M1/M2 Macs**, you might need:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Memory issues during installation**:
```bash
pip install --no-cache-dir torch
```