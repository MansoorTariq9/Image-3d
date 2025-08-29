#!/usr/bin/env python3
"""
Create ASCII flowchart diagrams for the client proposal
"""

def create_architecture_flowchart():
    """Create detailed architecture flowchart"""
    
    flowchart = """
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                         HSI GAUSSIAN 3D ARCHITECTURE                                  ║
║          Combining HS-NeRF Spectral Expertise with GaussianAnything Efficiency        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────┐     ┌─────────────────────────┐
│  Multi-view HSI Images  │     │   Camera Parameters     │
│  • 120 channels         │     │  • Intrinsics (K)       │
│  • 400-1000nm range     │     │  • Extrinsics (R,t)     │  
│  • 332×324 resolution   │     │  • Multi-view poses     │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             └─────────────┬─────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              PREPROCESSING (No Calibration!)                          │
├─────────────────┬──────────────────┬──────────────────┬─────────────────────────────┤
│  Background     │   Percentile      │    Spectral      │   Key Innovation:         │
│  Estimation     │   Normalization   │    Smoothing     │   No white/dark refs!     │
│  • 95th %ile    │   • 2-98% range   │    • 3-ch avg    │   • Robust to lighting    │
└─────────────────┴──────────────────┴──────────────────┴─────────────────────────────┘
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                    HYPERSPECTRAL VAE (Adapted from GaussianAnything)                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐           ║
║  │ Spectral Encoder│ --> │  Spatial Encoder │ --> │  Cross-Attention    │           ║
║  │ 120→64→32→16 ch │     │  Conv + Downsamp │     │  Multi-view Fusion  │           ║
║  └─────────────────┘     └──────────────────┘     └─────────────────────┘           ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │   Point Cloud Latent Space   │
                          │   • 512D latent code         │
                          │   • 2048 3D points           │
                          │   • Spectral features        │
                          └──────────────┬───────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                        SPECTRAL GAUSSIAN 3D MODEL                                     ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐   ║
║  │  Position   │  │   Rotation   │  │   Opacity   │  │   Spectral Features     │   ║
║  │  [50K × 3]  │  │   [50K × 4]  │  │  [50K × 1]  │  │   [50K × 120] ← KEY!    │   ║
║  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
                │                                                │
                ▼                                                ▼
┌───────────────────────────────┐                ┌──────────────────────────────┐
│    SPECTRAL RENDERER          │                │    DEPTH ESTIMATOR           │
│  • Wavelength-dependent       │                │  • Spectral → Depth CNN      │
│  • Gaussian splatting         │                │  • Uncertainty estimation    │
│  • Any λ ∈ [400,1000]nm      │                │  • HS-NeRF inspired          │
└───────────────┬───────────────┘                └──────────────┬───────────────┘
                │                                                │
                └─────────────────┬──────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                    OUTPUTS                                            │
├──────────────────┬────────────────────┬─────────────────┬────────────────────────────┤
│  3D Point Cloud  │   Novel Views      │   Depth Maps    │   Spectral Analysis        │
│  • With spectra  │   • Any wavelength │   • Per view    │   • Material signatures   │
└──────────────────┴────────────────────┴─────────────────┴────────────────────────────┘

LOSS FUNCTIONS: Spectral MSE + SAM + Depth Consistency + KL Divergence + Smoothness
"""
    return flowchart

def create_pipeline_flowchart():
    """Create simplified pipeline flowchart"""
    
    flowchart = """
                        HSI 3D RECONSTRUCTION PIPELINE
    ═══════════════════════════════════════════════════════════════════════

    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  Input  │ --> │Preproc. │ --> │   VAE   │ --> │Gaussian │ --> │ Render  │
    │   HSI   │     │ No Cal! │     │ Encode  │     │  Model  │     │ & Depth │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
         │               │               │               │               │
    120 channels    Background     120→16 ch      50K Gaussians    Any wavelength
    332×324         Normalize      Cross-attn     120 spectra/pt   Depth from HSI
    Multi-view      Smooth         Point cloud    Smooth spectra   Real-time

                                        ▼
                          ┌─────────────────────────────┐
                          │         OUTPUTS             │
                          │  • 3D reconstruction        │
                          │  • Spectral at each point   │
                          │  • Depth maps              │
                          │  • Novel view synthesis    │
                          └─────────────────────────────┘

    KEY ADVANTAGES:
    ✓ No calibration references needed (unlike standard HSI processing)
    ✓ 10-100x faster than HS-NeRF (Gaussian splatting vs implicit fields)
    ✓ Full spectral information preserved (120 channels per 3D point)
    ✓ Wavelength-specific rendering (any λ between 400-1000nm)
"""
    return flowchart

def create_comparison_table():
    """Create comparison table"""
    
    table = """
    APPROACH COMPARISON
    ═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┬──────────────┬─────────────────────┬──────────────────────┐
    │     Feature     │   HS-NeRF    │  GaussianAnything   │   Our HSI-Gaussian   │
    ├─────────────────┼──────────────┼─────────────────────┼──────────────────────┤
    │ Spectral Bands  │     128      │      3 (RGB)        │        120          │
    │ Representation  │ Neural Field │  Gaussian Splats    │  Spectral Gaussians  │
    │ Speed           │     Slow     │      Fast           │        Fast          │
    │ Calibration     │   Required   │       N/A           │    Not Required      │
    │ Depth Estimation│     Yes      │       No            │        Yes           │
    │ Memory Usage    │     High     │      Low            │       Medium         │
    │ Novel Views     │     Yes      │      Yes            │   Yes + Any λ        │
    │ 3D Output       │   Implicit   │   Point Cloud       │  Point Cloud + HSI   │
    └─────────────────┴──────────────┴─────────────────────┴──────────────────────┘

    TECHNICAL INNOVATIONS:
    • Each Gaussian stores 120-dimensional spectral signature
    • VAE compresses spectral dimension for efficiency (120→16)
    • Depth estimated directly from spectral patterns
    • No calibration required through robust normalization
"""
    return table

def main():
    print("Creating flowchart diagrams for client proposal...\n")
    
    # Save architecture flowchart
    arch_chart = create_architecture_flowchart()
    with open('architecture_flowchart.txt', 'w', encoding='utf-8') as f:
        f.write(arch_chart)
    print("✓ Created architecture_flowchart.txt")
    
    # Save pipeline flowchart
    pipeline_chart = create_pipeline_flowchart()
    with open('pipeline_flowchart.txt', 'w', encoding='utf-8') as f:
        f.write(pipeline_chart)
    print("✓ Created pipeline_flowchart.txt")
    
    # Save comparison table
    comparison = create_comparison_table()
    with open('approach_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(comparison)
    print("✓ Created approach_comparison.txt")
    
    # Display all
    print("\n" + "="*80)
    print(arch_chart)
    print("\n" + "="*80)
    print(pipeline_chart)
    print("\n" + "="*80)
    print(comparison)

if __name__ == "__main__":
    main()