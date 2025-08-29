import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(5, 13.5, 'HSI Gaussian 3D Architecture', fontsize=24, weight='bold', ha='center')
ax.text(5, 13, 'Combining HS-NeRF Spectral Expertise with GaussianAnything Efficiency', 
        fontsize=14, ha='center', style='italic')

# Color scheme
color_input = '#E8F4FF'
color_preprocess = '#FFE8E8'
color_vae = '#E8FFE8'
color_gaussian = '#FFF0E8'
color_render = '#F0E8FF'
color_output = '#E8FFF0'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, weight='bold')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, text='', curved=False):
    if curved:
        style = "arc3,rad=0.3"
    else:
        style = "arc3,rad=0"
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=style,
                                    arrowstyle='->,head_width=0.4,head_length=0.8',
                                    lw=2, color='black')
    ax.add_patch(arrow)
    
    if text:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, text, fontsize=9, ha='center')

# 1. Input Layer
create_box(ax, 1, 11, 3.5, 1, 'Multi-view HSI Images\n120ch, 400-1000nm\n332×324', color_input)
create_box(ax, 5.5, 11, 3, 1, 'Camera Parameters\nIntrinsics/Extrinsics', color_input)

# 2. Preprocessing Layer
create_box(ax, 0.5, 9, 2, 0.8, 'Background\nEstimation', color_preprocess, 9)
create_box(ax, 2.7, 9, 2, 0.8, 'Percentile\nNormalization', color_preprocess, 9)
create_box(ax, 4.9, 9, 2, 0.8, 'Spectral\nSmoothing', color_preprocess, 9)
create_box(ax, 7.1, 9, 2.4, 0.8, 'No Calibration\nRequired!', color_preprocess, 9)

# 3. VAE Encoding Layer
ax.text(5, 7.8, 'Hyperspectral VAE (Adapted from GaussianAnything)', 
        fontsize=12, ha='center', weight='bold', style='italic')
create_box(ax, 1, 6.5, 3, 1, 'Spectral Encoder\n120→64→32→16ch', color_vae)
create_box(ax, 4.5, 6.5, 2, 1, 'Spatial Encoder\n4× Downsample', color_vae)
create_box(ax, 7, 6.5, 2, 1, 'Cross-Attention\nMulti-view Fusion', color_vae)

# 4. Latent Space
create_box(ax, 3.5, 5, 3, 0.8, 'Point Cloud Latent Space\n[512D latent + 2048 points]', '#FFE8FF')

# 5. Gaussian Model Layer
ax.text(5, 4.2, 'Spectral Gaussian 3D (Novel Contribution)', 
        fontsize=12, ha='center', weight='bold', style='italic')
create_box(ax, 0.5, 2.8, 2.2, 1, 'Position\n[50K × 3]', color_gaussian, 9)
create_box(ax, 2.9, 2.8, 2, 1, 'Rotation/Scale\n[50K × 4/3]', color_gaussian, 9)
create_box(ax, 5.1, 2.8, 2.2, 1, 'Opacity\n[50K × 1]', color_gaussian, 9)
create_box(ax, 7.4, 2.8, 2.2, 1, 'Spectral Features\n[50K × 120]', color_gaussian, 9)

# 6. Rendering Layer
ax.text(2.5, 2, 'Spectral Renderer (HS-NeRF inspired)', 
        fontsize=11, ha='center', style='italic')
ax.text(7.5, 2, 'Depth Estimator (HS-NeRF inspired)', 
        fontsize=11, ha='center', style='italic')

create_box(ax, 1, 0.8, 3, 0.8, 'Wavelength-dependent\nGaussian Splatting', color_render)
create_box(ax, 6, 0.8, 3, 0.8, 'Spectral→Depth\nCNN Network', color_render)

# 7. Output Layer
create_box(ax, 0.5, -0.5, 2, 0.8, '3D Point Cloud\nwith Spectra', color_output, 9)
create_box(ax, 2.7, -0.5, 2, 0.8, 'Novel Views\nAny λ', color_output, 9)
create_box(ax, 4.9, -0.5, 2, 0.8, 'Depth Maps', color_output, 9)
create_box(ax, 7.1, -0.5, 2.3, 0.8, 'Spectral Analysis', color_output, 9)

# Add arrows
# Input to preprocessing
create_arrow(ax, 2.75, 11, 1.5, 9.8)
create_arrow(ax, 2.75, 11, 3.7, 9.8)
create_arrow(ax, 2.75, 11, 5.9, 9.8)
create_arrow(ax, 7, 11, 8.3, 9.8)

# Preprocessing to VAE
create_arrow(ax, 3.7, 9, 2.5, 7.5)
create_arrow(ax, 5.9, 9, 5.5, 7.5)
create_arrow(ax, 8.3, 9, 8, 7.5)

# VAE internal connections
create_arrow(ax, 2.5, 6.5, 5.5, 6.5)
create_arrow(ax, 5.5, 6.5, 8, 6.5)

# VAE to latent
create_arrow(ax, 5, 6.5, 5, 5.8)

# Latent to Gaussians
create_arrow(ax, 5, 5, 1.6, 3.8)
create_arrow(ax, 5, 5, 3.9, 3.8)
create_arrow(ax, 5, 5, 6.2, 3.8)
create_arrow(ax, 5, 5, 8.5, 3.8)

# Gaussians to rendering
create_arrow(ax, 1.6, 2.8, 2.5, 1.6)
create_arrow(ax, 8.5, 2.8, 7.5, 1.6)

# Rendering to output
create_arrow(ax, 2.5, 0.8, 1.5, -0.5)
create_arrow(ax, 2.5, 0.8, 3.7, -0.5)
create_arrow(ax, 7.5, 0.8, 5.9, -0.5)
create_arrow(ax, 2.5, 0.8, 8.2, -0.5, curved=True)

# Add key innovations boxes
innovation_box = Rectangle((0.2, 10.3), 4, 0.5, fill=True, 
                          facecolor='#FFFACD', edgecolor='red', linewidth=2)
ax.add_patch(innovation_box)
ax.text(2.2, 10.55, 'KEY: No calibration required!', fontsize=10, 
        ha='center', weight='bold', color='red')

innovation_box2 = Rectangle((5.8, 10.3), 4, 0.5, fill=True, 
                           facecolor='#FFFACD', edgecolor='red', linewidth=2)
ax.add_patch(innovation_box2)
ax.text(7.8, 10.55, 'KEY: 120 spectral channels per Gaussian', fontsize=10, 
         ha='center', weight='bold', color='red')

# Add loss functions box
loss_box = Rectangle((0.2, 4.4), 2.5, 0.5, fill=True, 
                    facecolor='#FFE4E1', edgecolor='blue', linewidth=2)
ax.add_patch(loss_box)
ax.text(1.45, 4.65, 'Losses: MSE + SAM + Depth + KL', fontsize=9, 
        ha='center', color='blue')

# Add specifications
ax.text(9.5, 6, 'Specs Met:\n✓ 120 channels\n✓ 400-1000nm\n✓ 332×324\n✓ No calibration', 
        fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('/home/dell/upwork/hsi_gaussian_3d/architecture_flowchart.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/dell/upwork/hsi_gaussian_3d/architecture_flowchart.pdf', bbox_inches='tight')
plt.close()

# Create a simpler flowchart for the processing pipeline
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 6)
ax2.axis('off')

ax2.text(7, 5.5, 'HSI Processing Pipeline', fontsize=20, weight='bold', ha='center')

# Processing steps
steps = [
    (1, 4, 'Input HSI\n120ch'),
    (3, 4, 'Preprocess\nNo Cal Needed'),
    (5, 4, 'VAE Encode\n120→16ch'),
    (7, 4, 'Gaussian\nGeneration'),
    (9, 4, 'Spectral\nRendering'),
    (11, 4, 'Depth\nEstimation'),
    (13, 4, 'Output\n3D + Spectra')
]

for i, (x, y, text) in enumerate(steps):
    if i == 0 or i == len(steps)-1:
        color = '#FFE4B5'
    else:
        color = '#E6E6FA'
    
    circle = Circle((x, y), 0.8, facecolor=color, edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    ax2.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
    
    if i < len(steps)-1:
        create_arrow(ax2, x+0.8, y, steps[i+1][0]-0.8, steps[i+1][1])

# Add details below
details = [
    (1, 2.5, '• 332×324×120\n• 400-1000nm\n• Multi-view'),
    (3, 2.5, '• Background est.\n• Percentile norm\n• Spectral smooth'),
    (5, 2.5, '• Cross-attention\n• Point cloud\n• Latent: 512D'),
    (7, 2.5, '• 50K Gaussians\n• 120 spectra each\n• Smoothness reg'),
    (9, 2.5, '• Any wavelength\n• Real-time\n• Alpha blending'),
    (11, 2.5, '• From spectra\n• Uncertainty\n• Guides 3D'),
    (13, 2.5, '• 3D model\n• Novel views\n• Spectral data')
]

for x, y, text in details:
    ax2.text(x, y, text, ha='center', va='top', fontsize=8)

plt.tight_layout()
plt.savefig('/home/dell/upwork/hsi_gaussian_3d/pipeline_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Flowcharts created successfully!")
print("1. architecture_flowchart.png - Detailed architecture")
print("2. pipeline_flowchart.png - Simplified pipeline")