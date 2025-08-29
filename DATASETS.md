# Available Hyperspectral Datasets for Testing

## 1. **Indian Pines** (Most Popular)
- **Download**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- **Size**: 145×145 pixels, 224 bands (200 after removing water absorption bands)
- **Wavelength**: 400-2500 nm
- **Format**: MATLAB (.mat) files
- **Classes**: 16 (vegetation, soil, etc.)

### How to use:
```python
import scipy.io
import numpy as np

# Load the data
mat = scipy.io.loadmat('Indian_pines.mat')
hsi_data = mat['indian_pines']  # Shape: (145, 145, 200)

# Resize to our format (324, 332, 120)
# 1. Select 120 channels evenly
indices = np.linspace(0, 199, 120, dtype=int)
hsi_120 = hsi_data[:, :, indices]

# 2. Resize spatial dimensions
# Use cv2.resize or similar for each channel
```

## 2. **Salinas**
- **Download**: Same website as Indian Pines
- **Size**: 512×217 pixels, 224 bands (204 after correction)
- **Wavelength**: 400-2500 nm
- **Classes**: 16 (vegetables, soil, vineyard)

## 3. **Pavia University**
- **Download**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- **Size**: 610×340 pixels, 103 bands
- **Wavelength**: 430-860 nm (closer to our 400-1000nm range!)
- **Format**: MATLAB files
- **Classes**: 9 (urban materials)

## 4. **HeiPorSPECTRAL** (Medical HSI)
- **Website**: https://www.heiporspectral.org/
- **Size**: 480×640 pixels, 100 bands
- **Wavelength**: 500-1000 nm (perfect for our range!)
- **Format**: NumPy arrays
- **Content**: Organ tissues (different from landscape data)

## 5. **AVIRIS Free Data**
- **Website**: https://aviris.jpl.nasa.gov/data/free_data.html
- **Format**: ENVI format
- **Note**: Requires ENVI reader or conversion tools

## Converting to Our Format

### From MATLAB files:
```python
import scipy.io
import numpy as np

def convert_mat_to_our_format(mat_file, output_dir):
    # Load MATLAB file
    data = scipy.io.loadmat(mat_file)
    
    # Get HSI cube (find the main data key)
    keys = [k for k in data.keys() if not k.startswith('__')]
    hsi_data = data[keys[0]]
    
    # Adjust channels to 120
    if hsi_data.shape[2] > 120:
        # Select 120 channels evenly
        indices = np.linspace(0, hsi_data.shape[2]-1, 120, dtype=int)
        hsi_data = hsi_data[:, :, indices]
    
    # Resize to 324x332
    # (Implementation depends on available libraries)
    
    # Normalize to [0, 1]
    hsi_data = (hsi_data - hsi_data.min()) / (hsi_data.max() - hsi_data.min())
    
    # Save as numpy
    np.save(f"{output_dir}/hsi_000.npy", hsi_data.astype(np.float32))
```

## Quick Test Data

For immediate testing without downloads, use the synthetic data generator:
```bash
python3 generate_sample_data.py
```

This creates realistic HSI data with:
- Vegetation signatures (peak ~750nm)
- Soil signatures (peak ~1500nm)
- Multiple views with consistent spectral properties
- Proper camera calibration

## Notes

1. Most datasets have more than 120 channels - you'll need to downsample
2. Wavelength ranges vary - Pavia University is closest to our 400-1000nm
3. All datasets can be loaded with scipy.io.loadmat()
4. Remember to normalize to [0, 1] range for our implementation