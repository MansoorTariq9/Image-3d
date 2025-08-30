#!/usr/bin/env python3
"""
View 3D reconstruction without external software
Multiple visualization options using matplotlib and plotly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def view_3d_matplotlib(points, colors, output_path=None, max_points=10000):
    """Create interactive 3D view using matplotlib"""
    print(f"Creating matplotlib 3D visualization with {min(max_points, len(points))} points...")
    
    # Subsample if too many points
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 15))
    
    # Main 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud - Interactive View')
    
    # Top view (X-Y)
    ax2 = fig.add_subplot(222)
    ax2.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (X-Y)')
    ax2.set_aspect('equal')
    
    # Front view (X-Z)
    ax3 = fig.add_subplot(223)
    ax3.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Front View (X-Z)')
    ax3.set_aspect('equal')
    
    # Side view (Y-Z)
    ax4 = fig.add_subplot(224)
    ax4.scatter(points[:, 1], points[:, 2], c=colors, s=1, alpha=0.6)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Side View (Y-Z)')
    ax4.set_aspect('equal')
    
    plt.suptitle('HSI 3D Reconstruction - Multiple Views', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()
    return fig


def view_3d_plotly(points, colors, spectral_features=None, output_html='3d_view.html', max_points=20000):
    """Create interactive 3D view using plotly (viewable in browser)"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Installing plotly...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    
    print(f"Creating interactive HTML visualization with {min(max_points, len(points))} points...")
    
    # Subsample if too many points
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        if spectral_features is not None:
            spectral_features = spectral_features[idx]
    
    # Convert colors to hex
    colors_hex = ['#%02x%02x%02x' % tuple((c * 255).astype(int)) for c in colors]
    
    # Create 3D scatter plot
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d', 'rowspan': 2}, {'type': 'scatter'}],
               [None, {'type': 'scatter'}]],
        subplot_titles=('3D Point Cloud', 'Spectral Signature (Random Points)', 'Point Distribution')
    )
    
    # 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors_hex,
                opacity=0.8
            ),
            text=[f'Point {i}<br>X: {p[0]:.3f}<br>Y: {p[1]:.3f}<br>Z: {p[2]:.3f}' 
                  for i, p in enumerate(points[:1000])],  # Limit hover text
            hoverinfo='text',
            name='3D Points'
        ),
        row=1, col=1
    )
    
    # Add spectral signatures if available
    if spectral_features is not None:
        wavelengths = np.linspace(400, 1000, spectral_features.shape[1])
        # Plot 10 random spectral signatures
        for i in range(min(10, len(spectral_features))):
            idx = np.random.randint(len(spectral_features))
            fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=spectral_features[idx],
                    mode='lines',
                    line=dict(width=2),
                    name=f'Point {idx}',
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Add histogram of Z values (height distribution)
    fig.add_trace(
        go.Histogram(
            x=points[:, 2],
            nbinsx=50,
            name='Z Distribution',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='HSI 3D Reconstruction - Interactive Viewer',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
    fig.update_yaxes(title_text="Radiance", row=1, col=2)
    fig.update_xaxes(title_text="Z Value", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Save to HTML
    fig.write_html(output_html)
    print(f"\n✅ Interactive 3D view saved to: {output_html}")
    print(f"   Open this file in your web browser to interact with the 3D model!")
    
    # Also create a simpler version for better performance
    simple_fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Color by height
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Height (Z)")
        )
    )])
    
    simple_fig.update_layout(
        title='HSI 3D Point Cloud - Simple View',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    simple_fig.write_html(output_html.replace('.html', '_simple.html'))
    
    return fig


def create_animated_rotation(points, colors, output_gif='3d_rotation.gif', max_points=5000):
    """Create animated GIF of rotating 3D view"""
    import matplotlib.animation as animation
    
    print("Creating animated rotation...")
    
    # Subsample
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=1, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('HSI 3D Reconstruction - Rotating View')
    
    # Set initial view
    ax.view_init(elev=20, azim=0)
    
    # Animation function
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 4)
        return [scatter]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=90, interval=50, blit=True)
    
    # Save as GIF
    try:
        anim.save(output_gif, writer='pillow', fps=20)
        print(f"Saved animated GIF to: {output_gif}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        print("Showing animation instead...")
        plt.show()
    
    return anim


def view_spectral_cube(spectral_features, output_path='spectral_analysis.png'):
    """Visualize spectral characteristics of the point cloud"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    wavelengths = np.linspace(400, 1000, spectral_features.shape[1])
    
    # 1. Mean spectrum
    mean_spectrum = np.mean(spectral_features, axis=0)
    std_spectrum = np.std(spectral_features, axis=0)
    axes[0, 0].plot(wavelengths, mean_spectrum, 'b-', linewidth=2)
    axes[0, 0].fill_between(wavelengths, 
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.3)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Mean Radiance')
    axes[0, 0].set_title('Average Spectral Signature ± Std')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spectral clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(spectral_features)
    
    for i in range(5):
        cluster_spectra = spectral_features[clusters == i]
        axes[0, 1].plot(wavelengths, np.mean(cluster_spectra, axis=0), 
                       linewidth=2, label=f'Cluster {i+1}')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Radiance')
    axes[0, 1].set_title('Spectral Clusters')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Spectral variance
    variance = np.var(spectral_features, axis=0)
    axes[0, 2].plot(wavelengths, variance, 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Wavelength (nm)')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title('Spectral Variance by Wavelength')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. RGB composite histogram
    r_idx = np.argmin(np.abs(wavelengths - 650))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 450))
    
    axes[1, 0].hist(spectral_features[:, r_idx], bins=50, alpha=0.5, color='red', label='650nm')
    axes[1, 0].hist(spectral_features[:, g_idx], bins=50, alpha=0.5, color='green', label='550nm')
    axes[1, 0].hist(spectral_features[:, b_idx], bins=50, alpha=0.5, color='blue', label='450nm')
    axes[1, 0].set_xlabel('Radiance')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('RGB Channel Distributions')
    axes[1, 0].legend()
    
    # 5. Correlation matrix
    # Sample subset for correlation
    sample_idx = np.random.choice(spectral_features.shape[0], 
                                 min(1000, spectral_features.shape[0]), 
                                 replace=False)
    corr_matrix = np.corrcoef(spectral_features[sample_idx].T)
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('Wavelength Band')
    axes[1, 1].set_ylabel('Wavelength Band')
    axes[1, 1].set_title('Spectral Band Correlation')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(spectral_features)
    
    axes[1, 2].scatter(pca_features[:, 0], pca_features[:, 1], 
                      c=pca_features[:, 2], s=1, alpha=0.5, cmap='viridis')
    axes[1, 2].set_xlabel('PC1')
    axes[1, 2].set_ylabel('PC2')
    axes[1, 2].set_title('PCA of Spectral Features (colored by PC3)')
    
    plt.suptitle('Spectral Analysis of 3D Point Cloud', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectral analysis to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View 3D reconstruction without external software")
    parser.add_argument("--input_dir", type=str, default="inference_output",
                       help="Directory containing reconstruction data")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum points to display (for performance)")
    parser.add_argument("--output_dir", type=str, default="3d_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--create_html", action="store_true",
                       help="Create interactive HTML visualization")
    parser.add_argument("--create_gif", action="store_true",
                       help="Create rotating GIF animation")
    parser.add_argument("--analyze_spectra", action="store_true",
                       help="Create spectral analysis plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading 3D reconstruction from {args.input_dir}...")
    numpy_dir = Path(args.input_dir) / "numpy_data"
    
    points = np.load(numpy_dir / "points.npy")
    colors = np.load(numpy_dir / "colors.npy")
    spectral_features = np.load(numpy_dir / "spectral_features.npy")
    
    print(f"Loaded {len(points)} points with {spectral_features.shape[1]} spectral channels")
    
    # Create matplotlib visualization
    print("\n1. Creating matplotlib visualization...")
    view_3d_matplotlib(points, colors, 
                      output_path=output_dir / "3d_matplotlib_views.png",
                      max_points=args.max_points)
    
    # Create interactive HTML
    if args.create_html:
        print("\n2. Creating interactive HTML visualization...")
        view_3d_plotly(points, colors, spectral_features,
                      output_html=str(output_dir / "3d_interactive.html"),
                      max_points=args.max_points * 2)
    
    # Create animated GIF
    if args.create_gif:
        print("\n3. Creating animated rotation...")
        create_animated_rotation(points, colors,
                               output_gif=str(output_dir / "3d_rotation.gif"),
                               max_points=5000)
    
    # Analyze spectra
    if args.analyze_spectra:
        print("\n4. Analyzing spectral characteristics...")
        view_spectral_cube(spectral_features,
                          output_path=str(output_dir / "spectral_analysis.png"))
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("\nTo view:")
    print(f"  - Static images: Open {output_dir}/*.png")
    print(f"  - Interactive 3D: Open {output_dir}/3d_interactive.html in browser")
    print(f"  - Animation: Open {output_dir}/3d_rotation.gif")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Run with defaults if no arguments
        sys.argv.extend(["--create_html", "--analyze_spectra"])
    main()