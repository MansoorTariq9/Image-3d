#!/usr/bin/env python3
"""
Quick 3D viewer - simplest possible way to view the point cloud
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the 3D data
print("Loading 3D point cloud...")
points = np.load('inference_output/numpy_data/points.npy')
colors = np.load('inference_output/numpy_data/colors.npy')

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Subsample for faster rendering
n_points = min(5000, len(points))
idx = np.random.choice(len(points), n_points, replace=False)

# Plot
scatter = ax.scatter(
    points[idx, 0], 
    points[idx, 1], 
    points[idx, 2],
    c=colors[idx],
    s=1,
    alpha=0.8
)

# Labels
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title(f'HSI 3D Reconstruction\n{n_points} points out of {len(points)} total', fontsize=14)

# Add some stats
x_range = points[:, 0].max() - points[:, 0].min()
y_range = points[:, 1].max() - points[:, 1].min()
z_range = points[:, 2].max() - points[:, 2].min()

ax.text2D(0.02, 0.98, f'Dimensions:\nX: {x_range:.2f}\nY: {y_range:.2f}\nZ: {z_range:.2f}', 
          transform=ax.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save high-res image
plt.savefig('3d_quick_view.png', dpi=150, bbox_inches='tight')
print("Saved to: 3d_quick_view.png")

# Show the plot
plt.show()

# Also create a top-down view
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.scatter(points[idx, 0], points[idx, 1], c=colors[idx], s=1, alpha=0.6)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Top-Down View')
ax2.set_aspect('equal')
plt.savefig('3d_top_view.png', dpi=150, bbox_inches='tight')
print("Saved to: 3d_top_view.png")
plt.show()