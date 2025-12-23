"""
Multi-Image Dataset Similarity Experiment

Generates a dataset of images with random geometric objects,
computes z-scored patch embeddings, and averages similarity across images.

Usage:
    python run_dataset_experiment.py --n-images 64 --n-objects 20
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from tqdm import tqdm

from config import ExperimentConfig, ModelType, get_model_type, PatchPosition
from dataset_generator import generate_dataset
from feature_extractor import get_feature_extractor
from similarity import reshape_similarities_to_grid
from visualize import save_figure, create_single_layer_figure
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Selected patch positions (same as encoder experiment)
SELECTED_PATCHES = [
    PatchPosition("Center", 0.5, 0.5),
    PatchPosition("North", 0.5, 0.15),
    PatchPosition("South", 0.5, 0.85),
    PatchPosition("East", 0.85, 0.5),
    PatchPosition("West", 0.15, 0.5),
]


def compute_zscore_similarity(
    all_features: List[torch.Tensor],
    selected_patch_idx: int
) -> List[torch.Tensor]:
    """
    Compute z-scored cosine similarity across multiple images.
    """
    # Stack features: [n_images, n_patches, hidden_dim]
    stacked = torch.stack(all_features, dim=0).float()
    
    # Compute mean and std across all images and patches
    mean = stacked.mean(dim=(0, 1), keepdim=True)
    std = stacked.std(dim=(0, 1), keepdim=True) + 1e-8
    
    # Z-normalize
    z_features = (stacked - mean) / std
    
    # L2 normalize
    z_features_normalized = F.normalize(z_features, p=2, dim=-1)
    
    # Compute similarity for each image
    similarities = []
    for i in range(z_features_normalized.shape[0]):
        features = z_features_normalized[i]
        selected = features[selected_patch_idx]
        sim = torch.mv(features, selected)
        similarities.append(sim)
    
    return similarities


def get_patch_index(pos: PatchPosition, n_patches: int) -> int:
    """Convert relative position to patch index"""
    col = int(pos.rel_x * (n_patches - 1))
    row = int(pos.rel_y * (n_patches - 1))
    return row * n_patches + col


def run_dataset_experiment(
    model_type: ModelType,
    n_images: int = 64,
    n_objects: int = 20,
    background_color: tuple = (0, 0, 255),
    output_dir: str = None,
    encoder_layers: List[int] = None,
):
    """Run the dataset similarity experiment"""
    
    config = ExperimentConfig(model_type=model_type)
    
    if output_dir is None:
        output_dir = f"outputs/dataset_{model_type.value}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    print("=" * 60)
    print("DATASET SIMILARITY EXPERIMENT")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Number of images: {n_images}")
    print(f"Objects per image: {n_objects}")
    print(f"Background color: {background_color}")
    print(f"Selected patches: {[p.name for p in SELECTED_PATCHES]}")
    
    # Generate dataset
    print("\n1. Generating dataset...")
    images = generate_dataset(
        n_images=n_images,
        image_size=config.image_size,
        n_objects=n_objects,
        background_color=background_color,
        output_dir=os.path.join(output_dir, "images"),
    )
    
    # Use specific layers or defaults
    if encoder_layers is None:
        encoder_layers = [1, 5, 15, 31]
    
    # Get feature extractor
    print("\n2. Loading model and extracting features...")
    extractor = get_feature_extractor(model_type)
    
    # Extract features for all images
    all_features = {f"encoder_layer_{l}": [] for l in encoder_layers}
    
    for i, image in enumerate(tqdm(images, desc="Extracting features")):
        features = extractor.extract_features(image, encoder_layers=encoder_layers, decoder_layers=None)
        
        for layer_name in all_features.keys():
            if layer_name in features['encoder']:
                layer_feat = features['encoder'][layer_name]
                if layer_feat.dim() == 3:
                    layer_feat = layer_feat[0]
                all_features[layer_name].append(layer_feat)
    
    # Compute z-scored average similarities for all 5 positions
    print("\n3. Computing z-scored similarities for all positions...")
    
    n_patches = config.num_patches_per_side
    
    # Results structure: layer -> patch_name -> {'mean': grid, 'std': grid}
    results = {}
    
    for layer_name, features_list in all_features.items():
        if not features_list:
            continue
        
        print(f"  Processing {layer_name}...")
        results[layer_name] = {}
        
        for patch_pos in SELECTED_PATCHES:
            patch_idx = get_patch_index(patch_pos, n_patches)
            
            # Compute similarities for each image
            similarities = compute_zscore_similarity(features_list, patch_idx)
            
            # Convert to grids
            grids = []
            for sim in similarities:
                if sim.shape[0] >= n_patches * n_patches:
                    grid = reshape_similarities_to_grid(sim[:n_patches*n_patches], n_patches)
                    grids.append(grid)
            
            if grids:
                avg_grid = np.mean(grids, axis=0)
                std_grid = np.std(grids, axis=0)
                results[layer_name][patch_pos.name] = {'mean': avg_grid, 'std': std_grid}
    
    # Visualize results - one figure per layer showing all 5 positions
    print("\n4. Creating visualizations...")
    
    # Use a sample image for reference
    sample_image = images[0] if images else None
    
    for layer_name in results.keys():
        layer_num = layer_name.replace('encoder_layer_', '')
        
        # Create figure with 5 subplots (one per position)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        for i, patch_pos in enumerate(SELECTED_PATCHES):
            if patch_pos.name in results[layer_name]:
                data = results[layer_name][patch_pos.name]
                
                im = axes[i].imshow(data['mean'], cmap='RdYlGn', vmin=-1, vmax=1)
                axes[i].set_title(f"{patch_pos.name}")
                
                # Mark selected patch
                patch_idx = get_patch_index(patch_pos, n_patches)
                row = patch_idx // n_patches
                col = patch_idx % n_patches
                axes[i].plot(col, row, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=2)
                
                # Add min/max annotation
                axes[i].text(0.02, 0.98, f"min: {data['mean'].min():.3f}\nmax: {data['mean'].max():.3f}",
                            transform=axes[i].transAxes, verticalalignment='top',
                            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        plt.colorbar(im, ax=axes.ravel().tolist(), label="Avg Cosine Similarity", shrink=0.8)
        
        fig.suptitle(f"Layer {layer_num} - Average Similarity (n={n_images} images)", fontsize=14)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, "figures", f"avg_similarity_L{layer_num}.png")
        save_figure(fig, fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
    
    # Print row vs column analysis for center position
    print("\n" + "=" * 60)
    print("ROW vs COLUMN SIMILARITY ANALYSIS (Center patch)")
    print("=" * 60)
    
    center_pos = SELECTED_PATCHES[0]  # Center
    center_row = int(center_pos.rel_y * (n_patches - 1))
    center_col = int(center_pos.rel_x * (n_patches - 1))
    
    for layer_name in results.keys():
        layer_num = layer_name.replace('encoder_layer_', '')
        
        if center_pos.name not in results[layer_name]:
            continue
            
        grid = results[layer_name][center_pos.name]['mean']
        
        row_sims = []
        col_sims = []
        
        for offset in range(1, min(5, n_patches // 2)):
            if center_col + offset < n_patches:
                row_sims.append(grid[center_row, center_col + offset])
            if center_col - offset >= 0:
                row_sims.append(grid[center_row, center_col - offset])
            if center_row + offset < n_patches:
                col_sims.append(grid[center_row + offset, center_col])
            if center_row - offset >= 0:
                col_sims.append(grid[center_row - offset, center_col])
        
        avg_row = np.mean(row_sims) if row_sims else 0
        avg_col = np.mean(col_sims) if col_sims else 0
        
        print(f"\nLayer {layer_num}:")
        print(f"  Row neighbors avg:    {avg_row:.6f}")
        print(f"  Column neighbors avg: {avg_col:.6f}")
        if avg_row > avg_col:
            print(f"  → Row neighbors MORE similar (diff: {avg_row - avg_col:.6f})")
        else:
            print(f"  → Column neighbors MORE similar (diff: {avg_col - avg_row:.6f})")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run dataset similarity experiment")
    parser.add_argument("--model", type=str, default="qwen", help="Model: 'qwen' or 'llava'")
    parser.add_argument("--n-images", type=int, default=64, help="Number of images")
    parser.add_argument("--n-objects", type=int, default=20, help="Objects per image")
    parser.add_argument("--background", type=str, default="blue", 
                        help="Background color: blue, green, red, etc.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--layers", type=str, default="1,5,15,31",
                        help="Comma-separated encoder layers to analyze")
    
    args = parser.parse_args()
    
    # Parse background color
    bg_colors = {
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
    }
    background = bg_colors.get(args.background.lower(), (0, 0, 255))
    
    # Parse layers
    encoder_layers = [int(l) for l in args.layers.split(',')]
    
    model_type = get_model_type(args.model)
    
    run_dataset_experiment(
        model_type=model_type,
        n_images=args.n_images,
        n_objects=args.n_objects,
        background_color=background,
        output_dir=args.output,
        encoder_layers=encoder_layers,
    )


if __name__ == "__main__":
    main()
