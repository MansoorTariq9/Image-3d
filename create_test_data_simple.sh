#!/bin/bash

# Create test HSI data directory structure
echo "Creating test HSI data structure..."

# Create directories
mkdir -p test_hsi_data/scene_001

# Create metadata files
cat > test_hsi_data/train_scenes.json << EOF
[{
    "name": "scene_001",
    "path": "test_hsi_data/scene_001",
    "num_views": 4
}]
EOF

cp test_hsi_data/train_scenes.json test_hsi_data/val_scenes.json

# Create camera poses
cat > test_hsi_data/scene_001/camera_poses.json << EOF
{
    "intrinsics": [
        [[300, 0, 166], [0, 300, 162], [0, 0, 1]],
        [[300, 0, 166], [0, 300, 162], [0, 0, 1]],
        [[300, 0, 166], [0, 300, 162], [0, 0, 1]],
        [[300, 0, 166], [0, 300, 162], [0, 0, 1]]
    ],
    "extrinsics": [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5], [0, 0, 0, 1]],
        [[0, 0, 1, -5], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 5], [0, 0, 0, 1]],
        [[0, 0, -1, 5], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    ]
}
EOF

echo "✓ Created test data structure at test_hsi_data/"
echo ""
echo "Note: The actual HSI numpy arrays (hsi_000.npy, etc.) need to be created"
echo "using Python with numpy installed. The expected shape is:"
echo "  - 324 x 332 x 120 (height x width x channels)"
echo "  - Float32 values in range [0, 1]"
echo ""
echo "Directory structure created:"
find test_hsi_data -type f | sed 's/^/  /'