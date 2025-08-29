#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Figure 1: Training Pipeline
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(8, 9.5, 'HSI-Gaussian 3D: Training Pipeline', fontsize=22, weight='bold', ha='center')

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

# Training flow
# 1. Data preparation
draw_box(ax1, 0.5, 8, 3, 0.8, 'HSI Multi-view Dataset\n(120ch, 332×324)', '#FFE4E1', 10)
draw_box(ax1, 4, 8, 2.5, 0.8, 'SuperGlue\nPose Estimation', '#E8F4FD', 10)

# 2. Preprocessing
draw_box(ax1, 1, 6.8, 2, 0.8, 'Channel Selection\n(Best 3-5)', '#FFE4E1', 9)
draw_box(ax1, 3.5, 6.8, 2, 0.8, 'Preprocessing\n(No Cal)', '#FFE4E1', 9)

# 3. Training components
draw_box(ax1, 0.5, 5, 3, 1, 'Hyperspectral VAE\n120→64→32→16', '#E8FFE8', 10)
draw_box(ax1, 4, 5, 3, 1, 'Spectral Gaussians\n50K points × 120ch', '#FFF4E1', 10)
draw_box(ax1, 7.5, 5, 3, 1, 'Depth Network\nSpectral→Depth', '#F4E8FF', 10)

# 4. Rendering
draw_box(ax1, 2, 3.2, 3, 1, 'Differentiable Renderer\n(Gaussian Splatting)', '#E1F4FF', 10)
draw_box(ax1, 6, 3.2, 3, 1, 'Multi-view Rendering\n@ Selected λ', '#E1F4FF', 10)

# 5. Loss computation
draw_box(ax1, 11, 7, 4.5, 2.5, 'Loss Functions\n\nL_total = λ₁L_MSE + λ₂L_SAM\n+ λ₃L_depth + λ₄L_KL\n+ λ₅L_smooth', '#FFE8E8', 10)

# 6. Optimization
draw_box(ax1, 4, 1.5, 3, 0.8, 'Adam Optimizer\nlr=1e-4', '#E8E8E8', 10)
draw_box(ax1, 4, 0.3, 3, 0.8, 'Backpropagation\nUpdate all params', '#E8E8E8', 10)

# Draw arrows
draw_arrow(ax1, 2, 8, 2, 7.6)
draw_arrow(ax1, 5.25, 8, 5.25, 7.6)
draw_arrow(ax1, 2, 6.8, 2, 6)
draw_arrow(ax1, 4.5, 6.8, 5.5, 6)
draw_arrow(ax1, 2, 5, 3.5, 4.2)
draw_arrow(ax1, 5.5, 5, 7.5, 4.2)
draw_arrow(ax1, 9, 5, 9, 4.2)
draw_arrow(ax1, 7.5, 3.2, 11, 7.5)
draw_arrow(ax1, 13.25, 7, 5.5, 2.3)
draw_arrow(ax1, 5.5, 1.5, 5.5, 1.1)

# Add iteration loop
ax1.annotate('', xy=(0.5, 0.5), xytext=(9.5, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
ax1.text(5, 0.1, 'Training Loop (100 epochs)', ha='center', color='red', fontsize=10)

plt.tight_layout()
plt.savefig('training_pipeline_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')

# Figure 2: Testing/Inference Pipeline
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis('off')

ax2.text(7, 7.5, 'HSI-Gaussian 3D: Inference Pipeline', fontsize=22, weight='bold', ha='center')

# Testing flow
draw_box(ax2, 0.5, 6, 3, 0.8, 'New HSI Views\n(120ch, 332×324)', '#FFE4E1', 10)
draw_box(ax2, 4, 6, 2.5, 0.8, 'Camera Poses\n(Known/Est.)', '#E8F4FD', 10)

draw_box(ax2, 1.5, 4.5, 4, 0.8, 'Preprocessing\n(Same as training)', '#FFE4E1', 10)

draw_box(ax2, 0.5, 3, 3, 1, 'Load Trained Model\nGaussians + VAE', '#E8FFE8', 10)
draw_box(ax2, 4, 3, 3, 1, 'Feature Extraction\n(Optional refinement)', '#FFF4E1', 10)

draw_box(ax2, 1, 1.3, 2.5, 0.8, 'Novel View\nSynthesis', '#E1F4FF', 9)
draw_box(ax2, 4, 1.3, 2.5, 0.8, 'Wavelength\nRendering', '#E1F4FF', 9)
draw_box(ax2, 7, 1.3, 2.5, 0.8, 'Depth Map\nExtraction', '#E1F4FF', 9)
draw_box(ax2, 10, 1.3, 2.5, 0.8, '3D Export\n(.ply + HSI)', '#E1F4FF', 9)

# Wavelength selection
draw_box(ax2, 8.5, 4, 4.5, 2, 'Wavelength Selection\n\n• 450nm (Blue)\n• 550nm (Green)\n• 650nm (Red)\n• Any λ ∈ [400,1000]', '#FFF4E1', 9)

# Draw arrows
draw_arrow(ax2, 2, 6, 3.5, 5.3)
draw_arrow(ax2, 5.25, 6, 3.5, 5.3)
draw_arrow(ax2, 3.5, 4.5, 2, 4)
draw_arrow(ax2, 3.5, 4.5, 5.5, 4)
draw_arrow(ax2, 2, 3, 2.25, 2.1)
draw_arrow(ax2, 5.5, 3, 5.25, 2.1)
draw_arrow(ax2, 5.5, 3, 8.25, 2.1)
draw_arrow(ax2, 5.5, 3, 11.25, 2.1)
draw_arrow(ax2, 10.75, 4, 8.25, 2.1)

plt.tight_layout()
plt.savefig('inference_pipeline_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')

# Figure 3: Loss Functions Visualization
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 10)
ax3.axis('off')

ax3.text(6, 9.5, 'Loss Functions in Detail', fontsize=22, weight='bold', ha='center')

# Loss components
losses = [
    (1, 7.5, 2.5, 1.2, 'L_MSE\n\n||I_rendered - I_target||²', '#FFE4E1'),
    (4.5, 7.5, 2.5, 1.2, 'L_SAM\n\ncos⁻¹(S_r·S_t/|S_r||S_t|)', '#FFE4E1'),
    (8, 7.5, 2.5, 1.2, 'L_depth\n\n||D_rendered - D_estimated||', '#FFE4E1'),
    (1, 5, 2.5, 1.2, 'L_KL\n\n-0.5(1+log(σ²)-μ²-σ²)', '#E8FFE8'),
    (4.5, 5, 2.5, 1.2, 'L_smooth\n\n||∂S/∂λ||² + ||∂²S/∂λ²||²', '#E8FFE8'),
    (8, 5, 2.5, 1.2, 'L_sparse\n\n||opacity||₁', '#E8FFE8'),
]

for x, y, w, h, text, color in losses:
    draw_box(ax3, x, y, w, h, text, color, 9)

# Total loss
draw_box(ax3, 2.5, 2.5, 7, 1.5, 'L_total = λ₁L_MSE + λ₂L_SAM + λ₃L_depth + λ₄L_KL + λ₅L_smooth + λ₆L_sparse\n\nDefault weights: λ₁=1.0, λ₂=0.1, λ₃=0.1, λ₄=0.001, λ₅=0.01, λ₆=0.001', '#E1E1E1', 10)

# Arrows to total
for x in [2.25, 5.75, 9.25]:
    draw_arrow(ax3, x, 7.5, 6, 4)
    draw_arrow(ax3, x, 5, 6, 4)

# Add descriptions
descriptions = [
    (1, 0.8, "• MSE: Pixel-wise reconstruction\n• SAM: Spectral angle (illumination invariant)\n• Depth: Consistency with estimated depth"),
    (5, 0.8, "• KL: VAE regularization\n• Smooth: Physical spectral curves\n• Sparse: Reduce floaters"),
]

for x, y, text in descriptions:
    ax3.text(x, y, text, fontsize=9, va='top')

plt.tight_layout()
plt.savefig('loss_functions_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')

# Figure 4: Data Flow
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 12))
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 12)
ax4.axis('off')

ax4.text(5, 11.5, 'Data Flow & Dimensions', fontsize=20, weight='bold', ha='center')

# Data flow with dimensions
flow_items = [
    (2.5, 10, 5, 1, 'Input: Multi-view HSI\n[B, V=4, C=120, H=324, W=332]', '#FFE4E1'),
    (2.5, 8.5, 5, 1, 'After Preprocessing\n[B, V=4, C=120, H=324, W=332]', '#FFE4E1'),
    (2.5, 7, 5, 1, 'Spectral Encoding\n[B, V=4, C=16, H=324, W=332]', '#E8FFE8'),
    (2.5, 5.5, 5, 1, 'Spatial Encoding\n[B, V=4, F=512, H\'=81, W\'=83]', '#E8FFE8'),
    (2.5, 4, 5, 1, 'Cross-Attention Output\n[B, Latent=512]', '#E8FFE8'),
    (2.5, 2.5, 5, 1, 'Gaussian Parameters\n[N=50K, (xyz=3, rot=4, s=3, α=1, λ=120)]', '#FFF4E1'),
    (2.5, 1, 5, 1, 'Rendered Output\n[B, V=4, C=120, H=324, W=332]', '#E1F4FF'),
]

for x, y, w, h, text, color in flow_items:
    draw_box(ax4, x, y, w, h, text, color, 9)
    if y > 1:
        draw_arrow(ax4, 5, y, 5, y-0.5)

plt.tight_layout()
plt.savefig('data_flow_dimensions.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.close('all')
print("Created detailed diagrams:")
print("1. training_pipeline_detailed.png")
print("2. inference_pipeline_detailed.png") 
print("3. loss_functions_detailed.png")
print("4. data_flow_dimensions.png")