#!/usr/bin/env python3
"""
Simple viewer for ENVI hyperspectral data files
"""

import numpy as np
import matplotlib.pyplot as plt
from data.envi_loader import ENVIReader
import os
import argparse

def view_hsi_data(data_path, output_dir="hsi_visualizations"):
    """View hyperspectral data with various visualizations"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading HSI data from: {data_path}")
    reader = ENVIReader()
    hsi_data, metadata = reader.read_envi(data_path)
    
    print(f"Data shape: {hsi_data.shape}")
    print(f"Wavelengths: {len(metadata['wavelength'])} bands from {metadata['wavelength'][0]:.1f}nm to {metadata['wavelength'][-1]:.1f}nm")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. RGB composite (approximate RGB bands)
    ax1 = plt.subplot(2, 3, 1)
    # Find bands closest to R(650nm), G(550nm), B(450nm)
    wavelengths = np.array(metadata['wavelength'])
    r_idx = np.argmin(np.abs(wavelengths - 650))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 450))
    
    rgb_image = np.stack([
        hsi_data[:, :, r_idx],
        hsi_data[:, :, g_idx],
        hsi_data[:, :, b_idx]
    ], axis=-1)
    
    # Normalize to 0-1 range
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    
    ax1.imshow(rgb_image)
    ax1.set_title(f'RGB Composite\nR:{wavelengths[r_idx]:.0f}nm, G:{wavelengths[g_idx]:.0f}nm, B:{wavelengths[b_idx]:.0f}nm')
    ax1.axis('off')
    
    # 2. Single band visualization (middle wavelength)
    ax2 = plt.subplot(2, 3, 2)
    mid_band = hsi_data.shape[2] // 2
    ax2.imshow(hsi_data[:, :, mid_band], cmap='viridis')
    ax2.set_title(f'Single Band: {wavelengths[mid_band]:.1f}nm')
    ax2.axis('off')
    
    # 3. Mean spectrum across all pixels
    ax3 = plt.subplot(2, 3, 3)
    mean_spectrum = np.mean(hsi_data.reshape(-1, hsi_data.shape[2]), axis=0)
    ax3.plot(wavelengths, mean_spectrum)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Mean Spectrum (All Pixels)')
    ax3.grid(True, alpha=0.3)
    
    # 4. False color composite (NIR-R-G)
    ax4 = plt.subplot(2, 3, 4)
    nir_idx = np.argmin(np.abs(wavelengths - 800))  # Near-infrared
    false_color = np.stack([
        hsi_data[:, :, nir_idx],  # NIR as red
        hsi_data[:, :, r_idx],    # Red as green
        hsi_data[:, :, g_idx]     # Green as blue
    ], axis=-1)
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
    ax4.imshow(false_color)
    ax4.set_title(f'False Color (NIR-R-G)\n{wavelengths[nir_idx]:.0f}-{wavelengths[r_idx]:.0f}-{wavelengths[g_idx]:.0f}nm')
    ax4.axis('off')
    
    # 5. Spectral signatures from different regions
    ax5 = plt.subplot(2, 3, 5)
    # Sample 5 random points
    h, w = hsi_data.shape[:2]
    for i in range(5):
        y, x = np.random.randint(h//4, 3*h//4), np.random.randint(w//4, 3*w//4)
        spectrum = hsi_data[y, x, :]
        ax5.plot(wavelengths, spectrum, alpha=0.7, label=f'Pixel ({x},{y})')
    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Sample Pixel Spectra')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Band correlation matrix (subset)
    ax6 = plt.subplot(2, 3, 6)
    # Use every 10th band to make it visible
    subset_indices = range(0, hsi_data.shape[2], 10)
    subset_data = hsi_data[:, :, subset_indices].reshape(-1, len(subset_indices))
    corr_matrix = np.corrcoef(subset_data.T)
    im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_title('Band Correlation Matrix (Every 10th band)')
    ax6.set_xlabel('Band Index')
    ax6.set_ylabel('Band Index')
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    
    # Save the figure
    angle = os.path.basename(os.path.dirname(data_path))
    output_path = os.path.join(output_dir, f'{angle}_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Also save individual bands as a montage
    fig2, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    
    # Show 20 bands evenly spaced across the spectrum
    band_indices = np.linspace(0, hsi_data.shape[2]-1, 20, dtype=int)
    
    for i, band_idx in enumerate(band_indices):
        axes[i].imshow(hsi_data[:, :, band_idx], cmap='viridis')
        axes[i].set_title(f'{wavelengths[band_idx]:.0f}nm', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Spectral Bands Montage - {angle}', fontsize=16)
    plt.tight_layout()
    
    montage_path = os.path.join(output_dir, f'{angle}_bands_montage.png')
    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
    print(f"Saved bands montage to: {montage_path}")
    
    # Close figures to prevent memory issues
    plt.close('all')
    
    return hsi_data, metadata

def main():
    parser = argparse.ArgumentParser(description='View ENVI hyperspectral data')
    parser.add_argument('data_path', nargs='?', 
                       default='/home/dell/upwork/hsi_gaussian_3d/sample_data/0degree_001/0degree_raw',
                       help='Path to ENVI raw file (without .hdr extension)')
    parser.add_argument('--output-dir', default='hsi_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--all', action='store_true',
                       help='Process all angles in sample_data')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all angles
        sample_dir = '/home/dell/upwork/hsi_gaussian_3d/sample_data'
        for angle_dir in sorted(os.listdir(sample_dir)):
            if 'degree' in angle_dir:
                raw_file = os.path.join(sample_dir, angle_dir, angle_dir.replace('_001', '_raw'))
                if os.path.exists(raw_file):
                    print(f"\nProcessing {angle_dir}...")
                    view_hsi_data(raw_file, args.output_dir)
    else:
        # Process single file
        view_hsi_data(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()