#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'HSI Gaussian 3D: Combining HS-NeRF + GaussianAnything', 
        fontsize=20, weight='bold', ha='center')

# Colors
colors = {
    'input': '#E8F4FD',
    'preprocess': '#FFE4E1',
    'vae': '#E8FFE8',
    'gaussian': '#FFF4E1',
    'render': '#F4E8FF',
    'output': '#E1FFE8'
}

# Helper functions
def draw_box(x, y, w, h, text, color, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', 
            fontsize=fontsize, weight='bold')

def draw_arrow(x1, y1, x2, y2, style='->', lw=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           connectionstyle="arc3,rad=0", 
                           arrowstyle=style, lw=lw, color='black')
    ax.add_patch(arrow)

# 1. Input Layer
draw_box(1, 7.5, 3, 1.2, 'Multi-view HSI\n120ch @ 332×324\n400-1000nm', colors['input'])
draw_box(5, 7.5, 2.5, 1.2, 'Camera\nPoses', colors['input'])

# 2. Preprocessing
draw_box(2, 5.8, 4, 0.8, 'Preprocessing (No Calibration!)', colors['preprocess'])
ax.text(4, 5.3, '• Background Estimation • Percentile Norm • Spectral Smooth', 
        ha='center', fontsize=9)

# 3. VAE
draw_box(1, 4, 2.8, 1, 'Spectral Encoder\n120→16ch', colors['vae'])
draw_box(4.2, 4, 2.8, 1, 'Spatial Encoder\n+ Downsample', colors['vae'])
draw_box(7.4, 4, 2.8, 1, 'Cross-Attention\nMulti-view', colors['vae'])

# 4. Gaussian Model
draw_box(2.5, 2.2, 4, 1, 'Spectral Gaussians (50K points)\nEach stores 120 spectral values!', colors['gaussian'])
draw_box(7, 2.2, 3, 1, 'Depth Estimator\nFrom Spectra', colors['gaussian'])

# 5. Output
draw_box(0.5, 0.5, 2.5, 0.8, '3D Point Cloud', colors['output'], 10)
draw_box(3.5, 0.5, 2.5, 0.8, 'Novel Views\n(Any λ)', colors['output'], 10)
draw_box(6.5, 0.5, 2.5, 0.8, 'Depth Maps', colors['output'], 10)
draw_box(9.5, 0.5, 2.5, 0.8, 'Spectral Data', colors['output'], 10)

# Arrows
# Input to preprocessing
draw_arrow(2.5, 7.5, 3, 6.6)
draw_arrow(6.25, 7.5, 5, 6.6)

# Preprocessing to VAE
draw_arrow(4, 5.8, 2.4, 5)
draw_arrow(4, 5.8, 5.6, 5)
draw_arrow(4, 5.8, 8.6, 5)

# VAE connections
draw_arrow(2.4, 4, 5.6, 4.5)
draw_arrow(5.6, 4.5, 8.6, 4)

# VAE to Gaussians
draw_arrow(5.6, 4, 4.5, 3.2)
draw_arrow(8.6, 4, 8.5, 3.2)

# Gaussians to output
draw_arrow(4.5, 2.2, 1.75, 1.3)
draw_arrow(4.5, 2.2, 4.75, 1.3)
draw_arrow(8.5, 2.2, 7.75, 1.3)
draw_arrow(4.5, 2.2, 10.75, 1.3, style='->')

# Add key features
ax.text(11, 8.5, 'Key Features:', fontsize=12, weight='bold')
ax.text(11, 8, '✓ No calibration', fontsize=10)
ax.text(11, 7.6, '✓ 120 channels', fontsize=10)
ax.text(11, 7.2, '✓ Fast (Gaussian)', fontsize=10)
ax.text(11, 6.8, '✓ Depth from HSI', fontsize=10)

# Add paper references
ax.text(0.5, 3.5, 'GaussianAnything:\nVAE + Gaussians', 
        fontsize=9, style='italic', color='blue')
ax.text(10.5, 3.5, 'HS-NeRF:\nSpectral + Depth', 
        fontsize=9, style='italic', color='blue')

plt.tight_layout()
plt.savefig('hsi_architecture_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: hsi_architecture_flowchart.png")

# Create a simpler pipeline chart
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 3)
ax2.axis('off')

# Pipeline boxes
boxes = [
    (0.5, 1, 'Input HSI\n120ch'),
    (2.5, 1, 'Preprocess\nNo Cal'),
    (4.5, 1, 'VAE\n120→16'),
    (6.5, 1, 'Gaussians\n+ Spectra'),
    (8.5, 1, 'Render\n+ Depth'),
    (10.5, 1, 'Output\n3D+HSI')
]

for i, (x, y, text) in enumerate(boxes):
    color = ['#E8F4FD', '#FFE4E1', '#E8FFE8', '#FFF4E1', '#F4E8FF', '#E1FFE8'][i]
    draw_box(x, y, 1.5, 1, text, color, 10)
    if i < len(boxes)-1:
        draw_arrow(x+1.5, y+0.5, x+2, y+0.5)

plt.tight_layout()
plt.savefig('hsi_pipeline_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: hsi_pipeline_simple.png")

plt.close('all')