#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Figure 1: Corrected Architecture showing paper details vs our approach
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 12))
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 12)
ax1.axis('off')

ax1.text(8, 11.5, 'HSI-Gaussian 3D: Architecture Overview', fontsize=22, weight='bold', ha='center')
ax1.text(8, 11, 'Adapting HS-NeRF (128ch) + GaussianAnything (RGB-D-N) → Your HSI (120ch)', 
         fontsize=14, ha='center', style='italic', color='darkblue')

# Helper functions
def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', 
            fontsize=fontsize, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2, style='->', lw=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           connectionstyle="arc3,rad=0", 
                           arrowstyle=style, lw=lw, color='black')
    ax.add_patch(arrow)

# Paper comparison boxes
ax1.text(2, 10, 'HS-NeRF', fontsize=14, weight='bold', color='red')
draw_box(ax1, 0.5, 8.5, 3.5, 1.2, '• 128 spectral channels\n• NeRF representation\n• Implicit depth from NeRF\n• Wavelength interpolation', '#FFE4E4', 9)

ax1.text(6.25, 10, 'GaussianAnything', fontsize=14, weight='bold', color='blue')
draw_box(ax1, 4.5, 8.5, 3.5, 1.2, '• RGB-D-Normal (6ch)\n• Gaussian Splatting\n• VAE + DiT\n• Two-stage generation', '#E4E4FF', 9)

ax1.text(10.25, 10, 'Our Approach (Your Data)', fontsize=14, weight='bold', color='green')
draw_box(ax1, 8.5, 8.5, 3.5, 1.2, '• 120 spectral channels\n• Spectral Gaussians\n• Modified VAE for HSI\n• Explicit depth + spectra', '#E4FFE4', 9)

ax1.text(14, 10, 'Key Difference', fontsize=12, weight='bold', color='darkred')
draw_box(ax1, 12.5, 8.5, 3, 1.2, 'Each Gaussian stores\n120 spectral values\n(not just RGB!)', '#FFFACD', 9)

# Main pipeline
ax1.text(8, 7.5, 'Implementation Pipeline', fontsize=16, weight='bold', ha='center')

# 1. Input
draw_box(ax1, 1, 6, 3, 1, 'Multi-view HSI\n120ch (not 128!)\n332×324', '#FFE4E1', 10)
draw_box(ax1, 4.5, 6, 2.5, 1, 'SuperGlue\nPoses', '#E8F4FD', 10)

# 2. Preprocessing  
draw_box(ax1, 2, 4.5, 4, 0.8, 'Preprocessing (No Calibration)\nBackground=1.0 • Percentile Norm', '#FFE4E1', 10)

# 3. Modified VAE
ax1.text(1, 3.8, 'Modified from GaussianAnything:', fontsize=10, style='italic')
draw_box(ax1, 0.5, 2.5, 2.8, 1, 'Spectral Encoder\n120→16ch\n(not RGB-D-N!)', '#E8FFE8', 9)
draw_box(ax1, 3.5, 2.5, 2.8, 1, 'Spatial Encoder\nConv + Down', '#E8FFE8', 9)
draw_box(ax1, 6.5, 2.5, 2.8, 1, 'Cross-Attention\n(Same as GA)', '#E8FFE8', 9)

# 4. Gaussian Generation
ax1.text(10.5, 3.8, 'Novel contribution:', fontsize=10, style='italic')
draw_box(ax1, 10, 2.5, 5, 1, 'Spectral Gaussians: 50K points\nEach point: [x,y,z,rot,scale,α,spectrum[120]]', '#FFF4E1', 10)

# 5. Rendering
draw_box(ax1, 1, 1, 3, 0.8, 'Spectral Renderer\nAny λ ∈ [400,1000]', '#F4E8FF', 10)
draw_box(ax1, 4.5, 1, 3, 0.8, 'Depth Network\n(Our addition)', '#F4E8FF', 10)

# Arrows
draw_arrow(ax1, 2.5, 6, 4, 5.3)
draw_arrow(ax1, 5.75, 6, 4, 5.3)
draw_arrow(ax1, 4, 4.5, 1.9, 3.5)
draw_arrow(ax1, 4, 4.5, 4.9, 3.5)
draw_arrow(ax1, 4, 4.5, 7.9, 3.5)
draw_arrow(ax1, 7.9, 2.5, 10, 3)
draw_arrow(ax1, 12.5, 2.5, 2.5, 1.8)
draw_arrow(ax1, 12.5, 2.5, 6, 1.8)

plt.tight_layout()
plt.savefig('corrected_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')

# Figure 2: Loss function comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis('off')

ax2.text(7, 7.5, 'Loss Functions: Papers vs Our Implementation', fontsize=20, weight='bold', ha='center')

# HS-NeRF losses
draw_box(ax2, 0.5, 5.5, 4, 1.5, 'HS-NeRF Loss\n(Not specified in detail)\nLikely: Photometric + Regularization', '#FFE4E4', 10)

# GaussianAnything losses  
draw_box(ax2, 5, 5.5, 4, 1.5, 'GaussianAnything Loss\nDiffusion Loss\nVAE Reconstruction\nKL Divergence', '#E4E4FF', 10)

# Our losses
draw_box(ax2, 9.5, 5.5, 4, 1.5, 'Our HSI Losses\nL_MSE + L_SAM (spectral)\nL_depth + L_smooth\nL_KL + L_sparse', '#E4FFE4', 10)

# Details
ax2.text(7, 4.5, 'Our Loss Design Rationale:', fontsize=14, weight='bold', ha='center')
losses_detail = [
    (1, 3, 2.5, 0.8, 'MSE: Standard\nreconstruction', '#FFE4E1'),
    (4, 3, 2.5, 0.8, 'SAM: Illumination\ninvariant (HSI)', '#FFE4E1'),
    (7, 3, 2.5, 0.8, 'Depth: Consistency\nwith estimates', '#FFE4E1'),
    (10, 3, 2.5, 0.8, 'Smooth: Physical\nspectra', '#FFE4E1'),
]

for x, y, w, h, text, color in losses_detail:
    draw_box(ax2, x, y, w, h, text, color, 9)

ax2.text(7, 1.5, 'L_total = λ₁L_MSE + λ₂L_SAM + λ₃L_depth + λ₄L_KL + λ₅L_smooth', 
         fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))

plt.tight_layout()
plt.savefig('corrected_loss_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')

# Figure 3: Data flow with correct dimensions
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 12))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 12)
ax3.axis('off')

ax3.text(5, 11.5, 'Corrected Data Flow', fontsize=20, weight='bold', ha='center')

# Data flow
flow_items = [
    (2, 10, 6, 0.8, 'Your Input: Multi-view HSI\n[B, V=4, C=120, H=324, W=332]', '#FFE4E1'),
    (2, 9, 6, 0.8, '(HS-NeRF uses 128ch, GA uses 6ch RGB-D-N)', '#FFFACD'),
    (2, 7.5, 6, 0.8, 'After Channel Selection for SuperGlue\n[B, V=4, C=3-5 selected, H=324, W=332]', '#E8F4FD'),
    (2, 6, 6, 0.8, 'Spectral Encoding (Our design)\n[B, V=4, C=16, H=324, W=332]', '#E8FFE8'),
    (2, 4.5, 6, 0.8, 'Spatial Encoding (From GA)\n[B, V=4, F=512, H\'=81, W\'=83]', '#E8FFE8'),
    (2, 3, 6, 0.8, 'Cross-Attention Output\n[B, Latent=512]', '#E8FFE8'),
    (2, 1.5, 6, 0.8, 'Spectral Gaussians (Novel)\n[N=50K, xyz=3, rot=4, s=3, α=1, λ=120]', '#FFF4E1'),
]

for x, y, w, h, text, color in flow_items:
    draw_box(ax3, x, y, w, h, text, color, 9)
    if y > 1.5 and y != 9:
        draw_arrow(ax3, 5, y, 5, y-0.7)

plt.tight_layout()
plt.savefig('corrected_data_flow.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.close('all')
print("Created corrected diagrams:")
print("1. corrected_architecture.png - Shows paper specs vs our adaptations")
print("2. corrected_loss_comparison.png - Clarifies loss function origins")
print("3. corrected_data_flow.png - Correct dimensions and comparisons")