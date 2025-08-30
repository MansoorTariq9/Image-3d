#!/bin/bash
# Quick start script for HSI Gaussian 3D reconstruction

echo "🚀 HSI Gaussian 3D - Quick Start"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic dependencies if needed
echo "Checking dependencies..."
python -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

python -c "import numpy" 2>/dev/null || pip install numpy matplotlib opencv-python tqdm pyyaml

# Step 1: View sample data
echo -e "\n📊 Step 1: Viewing sample HSI data..."
if [ -f "sample_data/0degree_001/0degree_raw" ]; then
    python view_hsi_data.py sample_data/0degree_001/0degree_raw
    echo "✅ Visualization saved to hsi_visualizations/"
else
    echo "⚠️  No sample data found at sample_data/0degree_001/"
fi

# Step 2: Convert ENVI to numpy
echo -e "\n🔄 Step 2: Converting ENVI data to training format..."
if [ -d "sample_data" ]; then
    python convert_envi_to_numpy.py --input_dir sample_data --output_dir processed_data
else
    echo "⚠️  sample_data directory not found"
fi

# Step 3: Test 3D reconstruction components
echo -e "\n🧪 Step 3: Testing 3D reconstruction components..."
python test_3d_reconstruction.py

# Step 4: Run demo
echo -e "\n🎨 Step 4: Running 3D reconstruction demo..."
python demo_3d_reconstruction.py

echo -e "\n✅ Quick start complete!"
echo -e "\n📝 Next steps:"
echo "1. Train model: python train.py --config config.yaml --data_root processed_data"
echo "2. Run inference: python inference.py --checkpoint outputs/best_model.pth --hsi_paths processed_data/*/hsi_*.npy --camera_params processed_data/*/camera_poses.json --export_ply"
echo "3. View results: Check inference_output/ directory"