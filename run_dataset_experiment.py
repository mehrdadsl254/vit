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

from config import ExperimentConfig, ModelType, get_model_type
from dataset_generator import generate_dataset
from feature_extractor import get_feature_extractor
from similarity import reshape_similarities_to_grid
from visualize import save_figure
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_zscore_similarity(
    all_features: List[torch.Tensor],
    selected_patch_idx: int
) -> List[torch.Tensor]:
    """
    Compute z-scored cosine similarity across multiple images.
    
    Steps:
    1. Stack all features
    2. Compute mean and std across all images
    3. Z-normalize each patch embedding
    4. Compute cosine similarity for each image
    
    Args:
        all_features: List of feature tensors, each [num_patches, hidden_dim]
        selected_patch_idx: Index of the reference patch
        
    Returns:
        List of similarity tensors, one per image
    """
    # Stack features: [n_images, n_patches, hidden_dim]
    stacked = torch.stack(all_features, dim=0).float()
    
    # Compute mean and std across all images and patches
    # Mean/std over dim 0 (images) and dim 1 (patches)
    mean = stacked.mean(dim=(0, 1), keepdim=True)  # [1, 1, hidden_dim]
    std = stacked.std(dim=(0, 1), keepdim=True) + 1e-8  # [1, 1, hidden_dim]
    
    # Z-normalize
    z_features = (stacked - mean) / std  # [n_images, n_patches, hidden_dim]
    
    # L2 normalize
    z_features_normalized = F.normalize(z_features, p=2, dim=-1)
    
    # Compute similarity for each image
    similarities = []
    for i in range(z_features_normalized.shape[0]):
        features = z_features_normalized[i]  # [n_patches, hidden_dim]
        selected = features[selected_patch_idx]  # [hidden_dim]
        sim = torch.mv(features, selected)  # [n_patches]
        similarities.append(sim)
    
    return similarities


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
    
    # Compute z-scored average similarities
    print("\n3. Computing z-scored similarities...")
    
    n_patches = config.num_patches_per_side
    center_idx = n_patches // 2 * n_patches + n_patches // 2
    
    results = {}
    
    for layer_name, features_list in all_features.items():
        if not features_list:
            continue
        
        print(f"  Processing {layer_name}...")
        
        # Compute similarities for each image
        similarities = compute_zscore_similarity(features_list, center_idx)
        
        # Convert to grids
        grids = []
        for sim in similarities:
            if sim.shape[0] >= n_patches * n_patches:
                grid = reshape_similarities_to_grid(sim[:n_patches*n_patches], n_patches)
                grids.append(grid)
        
        if grids:
            # Average across images
            avg_grid = np.mean(grids, axis=0)
            std_grid = np.std(grids, axis=0)
            results[layer_name] = {'mean': avg_grid, 'std': std_grid}
    
    # Visualize results
    print("\n4. Creating visualizations...")
    
    for layer_name, data in results.items():
        layer_num = layer_name.replace('encoder_layer_', '')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean heatmap
        im1 = axes[0].imshow(data['mean'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title(f"Layer {layer_num} - Average Similarity (n={n_images})")
        axes[0].set_xlabel("Patch Column")
        axes[0].set_ylabel("Patch Row")
        plt.colorbar(im1, ax=axes[0], label="Cosine Similarity")
        
        # Mark center
        center_row = n_patches // 2
        center_col = n_patches // 2
        axes[0].plot(center_col, center_row, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=2)
        
        # Std heatmap
        im2 = axes[1].imshow(data['std'], cmap='viridis', vmin=0)
        axes[1].set_title(f"Layer {layer_num} - Std Deviation")
        axes[1].set_xlabel("Patch Column")
        axes[1].set_ylabel("Patch Row")
        plt.colorbar(im2, ax=axes[1], label="Standard Deviation")
        axes[1].plot(center_col, center_row, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=2)
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, "figures", f"avg_similarity_L{layer_num}.png")
        save_figure(fig, fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
    
    # Print row vs column analysis
    print("\n" + "=" * 60)
    print("ROW vs COLUMN SIMILARITY ANALYSIS")
    print("=" * 60)
    
    for layer_name, data in results.items():
        layer_num = layer_name.replace('encoder_layer_', '')
        grid = data['mean']
        
        # Get row and column neighbors
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
