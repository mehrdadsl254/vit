"""
Visualization for patch similarity experiment.

Creates multi-panel figures with:
- Rows = different layers
- Columns = different selected patches
- Each cell shows similarity heatmap overlaid on image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Tuple, Optional
from PIL import Image
import os

from config import ExperimentConfig, PatchPosition
from similarity import reshape_similarities_to_grid


def create_patch_overlay(
    image: Image.Image,
    similarities: np.ndarray,
    selected_patch_idx: int,
    num_patches_per_side: int,
    ax: plt.Axes,
    show_values: bool = True,
    cmap: str = 'RdYlGn',
    alpha: float = 0.6,
    fontsize: int = 6,
    decimal_places: int = 2
):
    """
    Create a single panel showing patches with similarity overlay.
    
    Args:
        image: Background image
        similarities: 2D array [H, W] of similarity scores
        selected_patch_idx: Index of selected patch (for highlighting)
        num_patches_per_side: Number of patches per side
        ax: Matplotlib axes
        show_values: Whether to show similarity values as text
        cmap: Colormap for similarity intensity
        alpha: Transparency of overlay
        fontsize: Font size for similarity values
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    patch_h = h // num_patches_per_side
    patch_w = w // num_patches_per_side
    
    # Show base image
    ax.imshow(img_array)
    
    # Create colormap normalization
    norm = Normalize(vmin=0, vmax=1)
    cmap_func = plt.cm.get_cmap(cmap)
    
    # Calculate selected patch row/col
    selected_row = selected_patch_idx // num_patches_per_side
    selected_col = selected_patch_idx % num_patches_per_side
    
    # Overlay patches with similarity colors
    for row in range(num_patches_per_side):
        for col in range(num_patches_per_side):
            x = col * patch_w
            y = row * patch_h
            
            sim_value = similarities[row, col]
            color = cmap_func(norm(sim_value))
            
            # Create rectangle for patch
            is_selected = (row == selected_row and col == selected_col)
            
            if is_selected:
                # Selected patch: thick red border, no fill
                rect = mpatches.Rectangle(
                    (x, y), patch_w, patch_h,
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none',
                    zorder=10
                )
            else:
                # Other patches: colored background
                rect = mpatches.Rectangle(
                    (x, y), patch_w, patch_h,
                    linewidth=0.5,
                    edgecolor='white',
                    facecolor=color,
                    alpha=alpha
                )
            ax.add_patch(rect)
            
            # Add similarity value text
            if show_values and not is_selected:
                text_color = 'white' if sim_value < 0.5 else 'black'
                fmt = f'{{:.{decimal_places}f}}'
                ax.text(
                    x + patch_w / 2, y + patch_h / 2,
                    fmt.format(sim_value),
                    ha='center', va='center',
                    fontsize=fontsize,
                    color=text_color,
                    fontweight='bold',
                    zorder=11
                )
    
    ax.axis('off')


def create_similarity_figure(
    image: Image.Image,
    all_similarities: Dict[str, Dict[str, np.ndarray]],
    layers: List[str],
    selected_patches: List[PatchPosition],
    config: ExperimentConfig,
    title: str = "Patch Similarity Analysis",
    figsize_per_cell: Tuple[float, float] = (2.5, 2.5),
    show_values: bool = True,
) -> plt.Figure:
    """
    Create the full multi-panel figure.
    
    Args:
        image: Input image
        all_similarities: Nested dict layer_name -> patch_name -> similarity grid
        layers: List of layer names (for rows)
        selected_patches: List of selected patch positions (for columns)
        config: Experiment configuration
        title: Figure title
        figsize_per_cell: Size of each subplot
        show_values: Whether to show similarity values
        
    Returns:
        Matplotlib figure
    """
    n_rows = len(layers)
    n_cols = len(selected_patches)
    
    fig_w = figsize_per_cell[0] * n_cols + 1  # Extra for colorbar
    fig_h = figsize_per_cell[1] * n_rows + 1  # Extra for title
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    
    # Handle single row/col case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Determine font size based on grid size
    fontsize = max(4, 10 - config.num_patches_per_side // 4)
    
    for row_idx, layer_name in enumerate(layers):
        for col_idx, patch_pos in enumerate(selected_patches):
            ax = axes[row_idx, col_idx]
            
            # Get similarity data
            if layer_name in all_similarities and patch_pos.name in all_similarities[layer_name]:
                sim_grid = all_similarities[layer_name][patch_pos.name]
                
                # Ensure it's reshaped to grid
                if sim_grid.ndim == 1:
                    sim_grid = reshape_similarities_to_grid(
                        sim_grid if not hasattr(sim_grid, 'numpy') else sim_grid,
                        config.num_patches_per_side
                    )
                
                patch_idx = config.get_patch_index(patch_pos)
                
                create_patch_overlay(
                    image, sim_grid, patch_idx,
                    config.num_patches_per_side, ax,
                    show_values=show_values,
                    fontsize=fontsize
                )
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
            
            # Add column header (patch name) for first row
            if row_idx == 0:
                ax.set_title(patch_pos.name, fontsize=10, fontweight='bold')
            
            # Add row label (layer) for first column
            if col_idx == 0:
                # Extract layer number from name
                layer_label = layer_name.replace('encoder_layer_', 'E').replace('decoder_layer_', 'D')
                ax.set_ylabel(layer_label, fontsize=10, rotation=0, labelpad=30, va='center')
    
    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Cosine Similarity', fontsize=10)
    
    # Add title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    return fig


def save_figure(fig: plt.Figure, output_path: str, dpi: int = 150):
    """Save figure to file"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def create_single_layer_figure(
    image: Image.Image,
    similarities: Dict[str, np.ndarray],  # patch_name -> similarity grid
    layer_name: str,
    selected_patches: List[PatchPosition],
    config: ExperimentConfig,
    title: str = None,
    figsize: Tuple[float, float] = (20, 4),
    decimal_places: int = 4,
) -> plt.Figure:
    """
    Create a single-layer figure with one row of selected patches.
    Much larger and clearer than the multi-layer version.
    
    Args:
        image: Input image
        similarities: patch_name -> similarity grid for this layer
        layer_name: Name of the layer
        selected_patches: List of selected patch positions
        config: Experiment configuration
        title: Figure title (auto-generated if None)
        figsize: Figure size (width, height)
        decimal_places: Number of decimal places to show
    
    Returns:
        Matplotlib figure
    """
    n_cols = len(selected_patches)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    if n_cols == 1:
        axes = [axes]
    
    # Larger font for bigger images
    fontsize = max(6, 14 - config.num_patches_per_side // 3)
    
    for col_idx, patch_pos in enumerate(selected_patches):
        ax = axes[col_idx]
        
        if patch_pos.name in similarities:
            sim_grid = similarities[patch_pos.name]
            
            # Ensure it's reshaped to grid
            if sim_grid.ndim == 1:
                sim_grid = reshape_similarities_to_grid(
                    sim_grid if not hasattr(sim_grid, 'numpy') else sim_grid,
                    config.num_patches_per_side
                )
            
            patch_idx = config.get_patch_index(patch_pos)
            
            create_patch_overlay(
                image, sim_grid, patch_idx,
                config.num_patches_per_side, ax,
                show_values=True,
                fontsize=fontsize,
                decimal_places=decimal_places
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
        
        ax.set_title(patch_pos.name, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # Add title
    if title is None:
        layer_num = layer_name.replace('encoder_layer_', 'E').replace('decoder_layer_', 'D')
        title = f"Layer {layer_num} Patch Similarity"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    return fig


def create_comparison_figure(
    encoder_fig_path: str,
    decoder_fig_path: str,
    output_path: str
):
    """
    Create a side-by-side comparison of encoder and decoder results.
    (Load saved figures and combine them)
    """
    from PIL import Image
    
    enc_img = Image.open(encoder_fig_path)
    dec_img = Image.open(decoder_fig_path)
    
    # Create combined figure
    total_width = enc_img.width + dec_img.width
    max_height = max(enc_img.height, dec_img.height)
    
    combined = Image.new('RGB', (total_width, max_height), 'white')
    combined.paste(enc_img, (0, 0))
    combined.paste(dec_img, (enc_img.width, 0))
    
    combined.save(output_path)
    print(f"Combined figure saved to: {output_path}")


# Utility function to visualize from saved numpy file
def visualize_from_saved_features(
    features_path: str,
    image_path: str,
    output_dir: str,
    component: str = 'encoder'  # 'encoder' or 'decoder'
):
    """
    Load saved features and create visualization.
    
    This can be run locally after extracting features on a remote server.
    """
    from feature_extractor import load_features
    from similarity import SimilarityAnalyzer
    
    config = ExperimentConfig()
    
    # Load features
    features = load_features(features_path)
    
    # Load image
    image = Image.open(image_path)
    
    # Compute similarities
    analyzer = SimilarityAnalyzer(config)
    
    if component == 'encoder':
        similarities = analyzer.analyze_encoder(features['encoder'])
        layers = [f"encoder_layer_{i}" for i in config.encoder_layers if f"encoder_layer_{i}" in features['encoder']]
        title = "Encoder (ViT) Patch Similarity"
    else:
        vision_indices = features.get('vision_token_indices')
        similarities = analyzer.analyze_decoder(features['decoder'], vision_indices)
        layers = [f"decoder_layer_{i}" for i in config.decoder_layers if f"decoder_layer_{i}" in features['decoder']]
        title = "Decoder (LLM) Patch Similarity"
    
    # Convert torch tensors to numpy grids
    sim_grids = {}
    for layer_name, patch_sims in similarities.items():
        sim_grids[layer_name] = {}
        for patch_name, sims in patch_sims.items():
            sim_grids[layer_name][patch_name] = reshape_similarities_to_grid(
                sims, config.num_patches_per_side
            )
    
    # Create figure
    fig = create_similarity_figure(
        image, sim_grids, layers, config.selected_patches, config,
        title=title
    )
    
    # Save
    output_path = os.path.join(output_dir, f"{component}_similarity.png")
    save_figure(fig, output_path)
    
    return output_path


if __name__ == "__main__":
    # Test visualization with synthetic data
    from PIL import Image
    import torch
    
    config = ExperimentConfig()
    
    print("Testing visualization...")
    
    # Create test image
    test_image = Image.new('RGB', config.image_size, (0, 255, 0))
    
    # Create synthetic similarity data
    # For a solid color image, we expect high similarity everywhere
    test_similarities = {}
    
    for layer_idx in [1, 2, 3]:
        layer_name = f"encoder_layer_{layer_idx}"
        test_similarities[layer_name] = {}
        
        for patch_pos in config.selected_patches:
            # Random similarities for testing
            sims = np.random.rand(config.num_patches_per_side, config.num_patches_per_side)
            sims = sims * 0.5 + 0.5  # Scale to 0.5-1.0 range
            
            # Set self-similarity to 1.0
            patch_idx = config.get_patch_index(patch_pos)
            row = patch_idx // config.num_patches_per_side
            col = patch_idx % config.num_patches_per_side
            sims[row, col] = 1.0
            
            test_similarities[layer_name][patch_pos.name] = sims
    
    # Create figure
    layers = list(test_similarities.keys())
    fig = create_similarity_figure(
        test_image, test_similarities, layers, config.selected_patches[:3], config,
        title="Test Visualization (Synthetic Data)"
    )
    
    # Save
    os.makedirs("outputs/figures", exist_ok=True)
    save_figure(fig, "outputs/figures/test_visualization.png")
    print("Test visualization complete!")
