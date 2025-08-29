# HSI Gaussian 3D System Overview

## Architecture Flow

```
Input: Multi-view HSI Images (120 channels, 400-1000nm)
    ↓
[Preprocessing]
- Background estimation (no calibration refs)
- Percentile normalization per channel  
- Spectral smoothing
    ↓
[Hyperspectral VAE Encoder]
- Spectral reduction: 120 → 16 channels
- Multi-view cross-attention
- Point cloud latent generation
    ↓
[Spectral Gaussian Model]
- 3D Gaussians with 120 spectral coefficients
- Physically-plausible spectral curves
- Spectral smoothness regularization
    ↓
[Spectral Renderer]
- Wavelength-dependent rendering
- Gaussian splatting for efficiency
- Render at any wavelength 400-1000nm
    ↓
[Depth Estimator]
- Extract depth from spectral signatures
- Uncertainty estimation
- Depth-guided optimization
    ↓
Output: 3D Reconstruction + Spectral Information
```

## Key Innovations

1. **Spectral Gaussian Representation**
   - Traditional: 3D Gaussian stores RGB (3 values)
   - Our approach: 3D Gaussian stores full spectrum (120 values)
   - Enables wavelength-specific analysis and rendering

2. **HSI-Adapted VAE**
   - Challenge: 120 channels is 40x more data than RGB
   - Solution: Spectral encoder compresses 120→16 before spatial processing
   - Maintains spectral fidelity while being computationally efficient

3. **No Calibration Required**
   - Problem: Your data lacks white/dark reference images
   - Solution: Percentile-based normalization + background estimation
   - Robust to illumination variations

4. **Depth from Spectral Signatures**
   - Insight: Different materials have unique spectral responses
   - Implementation: CNN extracts depth cues from spectral patterns
   - Benefit: Better 3D structure than RGB-only methods

## Usage Example

```python
# Training
python train.py \
  --config config.yaml \
  --data_root /path/to/hsi_data \
  --output_dir ./outputs

# Inference - Reconstruct and render at specific wavelengths
python inference.py \
  --checkpoint outputs/best_model.pth \
  --hsi_paths scene_001/*.npy \
  --camera_params scene_001/camera_poses.json \
  --wavelengths 450 550 650 850 \
  --export_ply
```

## Expected Results

1. **3D Point Cloud**: With full spectral information at each point
2. **Depth Maps**: Extracted from spectral signatures
3. **Novel Views**: Render from any viewpoint at any wavelength
4. **Spectral Analysis**: Query spectral response at any 3D location

## Comparison with Original Approaches

| Feature | HS-NeRF | GaussianAnything | Our HSI-Gaussian3D |
|---------|---------|------------------|-------------------|
| Spectral Channels | 128 | 3 (RGB) | 120 |
| Representation | Neural fields | Gaussian splats | Spectral Gaussians |
| Speed | Slow (NeRF) | Fast | Fast |
| Calibration | Required | N/A | Not required |
| Depth Estimation | Yes | No | Yes |
| Memory Usage | High | Low | Medium |

## Data Requirements

- **Input**: HSI cubes [324, 332, 120] in .npy format
- **Views**: 4+ views recommended for good reconstruction
- **Camera**: Intrinsic and extrinsic matrices in JSON
- **No calibration**: System handles uncalibrated data automatically