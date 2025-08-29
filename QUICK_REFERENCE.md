# Quick Reference for HSI Gaussian 3D Project

## Client's Data
- **Format**: ENVI (BSQ interleave, float32)
- **Dimensions**: 332×324×120 (W×H×Bands)
- **Wavelengths**: 400-1000nm (5.04nm spacing)
- **No calibration refs** - use percentile normalization

## Key Innovation
**Each 3D Gaussian stores 120 spectral values** (not RGB like standard Gaussian splatting)

## Architecture Summary
```
Multi-view HSI (120ch) 
    ↓
SuperGlue poses + Preprocessing (no cal)
    ↓
Modified VAE (120→16ch compression)
    ↓
Spectral Gaussians (50K points)
    ↓
Render at any wavelength + Depth estimation
```

## Critical Implementation Points
1. **VAE Input**: Modified from RGB-D-Normal (6ch) to HSI (120ch)
2. **Background = 1.0** (client specified)
3. **Use SuperGlue**, not COLMAP (client emphasized)
4. **Losses**: MSE + SAM + depth + KL + smoothness

## File Locations
- Implementation: `/home/dell/upwork/hsi_gaussian_3d/`
- Core modules: `core/` directory
- Training script: `train.py`
- Config: `config.yaml`

## Wednesday Deliverables
1. ENVI data loader
2. Basic 3D reconstruction
3. Visualizations at key wavelengths
4. Initial depth maps

## Commands to Remember
```bash
# Activate environment
source venv/bin/activate

# Train
python train.py --config config.yaml --data_root <path>

# Test
python inference.py --checkpoint <model> --hsi_paths <data>
```