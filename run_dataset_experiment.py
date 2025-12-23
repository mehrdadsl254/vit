"""
Multi-Image Dataset Similarity Experiment

Generates/loads a dataset of images, computes z-scored patch embeddings,
and averages similarity across images.

Usage:
    # Synthetic geometric shapes
    python run_dataset_experiment.py --n-images 64 --n-objects 20 --encoder
    
    # COCO dataset
    python run_dataset_experiment.py --coco --coco-path /path/to/coco/val2017 --n-images 1000 --encoder
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from tqdm import tqdm
import math
import glob
import random

from config import ExperimentConfig, ModelType, get_model_type, PatchPosition
from dataset_generator import generate_dataset
from feature_extractor import get_feature_extractor
from similarity import reshape_similarities_to_grid
from visualize import save_figure
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Selected patch positions - spread across image
SELECTED_PATCHES = [
    PatchPosition("Center", 0.5, 0.5),
    PatchPosition("TopLeft", 0.1, 0.1),
    PatchPosition("TopRight", 0.9, 0.1),
    PatchPosition("BottomLeft", 0.1, 0.9),
    PatchPosition("BottomRight", 0.9, 0.9),
]


def load_coco_images(
    coco_path: str,
    n_images: int,
    target_size: tuple = (448, 448),
    seed: int = 42,
) -> List[Image.Image]:
    """
    Load images from COCO dataset directory.
    
    Args:
        coco_path: Path to COCO images directory (e.g., val2017)
        n_images: Number of images to load
        target_size: Resize all images to this size for consistent patches
        seed: Random seed for reproducible selection
        
    Returns:
        List of PIL Images, all resized to target_size
    """
    print(f"Loading COCO images from: {coco_path}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(coco_path, ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {coco_path}")
    
    print(f"Found {len(image_files)} images")
    
    # Random sample
    random.seed(seed)
    if len(image_files) > n_images:
        image_files = random.sample(image_files, n_images)
    else:
        print(f"Warning: Only {len(image_files)} images available, using all")
    
    # Load and resize images
    images = []
    for img_path in tqdm(image_files, desc="Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            # Resize to target size (maintains consistent patch count)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    print(f"Loaded {len(images)} images, resized to {target_size}")
    return images


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


def run_encoder_dataset_experiment(
    model_type: ModelType,
    n_images: int,
    output_dir: str,
    layers: List[int],
    images: List[Image.Image],
    extractor,
    config: ExperimentConfig,
):
    """Run encoder-specific dataset experiment"""
    print("\n--- ENCODER ANALYSIS ---")
    
    n_patches = config.num_patches_per_side
    
    # Extract features for all images
    all_features = {f"encoder_layer_{l}": [] for l in layers}
    
    for image in tqdm(images, desc="Extracting encoder features"):
        features = extractor.extract_features(image, encoder_layers=layers, decoder_layers=None)
        
        for layer_name in all_features.keys():
            if layer_name in features['encoder']:
                layer_feat = features['encoder'][layer_name]
                if layer_feat.dim() == 3:
                    layer_feat = layer_feat[0]
                all_features[layer_name].append(layer_feat)
    
    # Compute and visualize
    results = compute_and_visualize(
        all_features, n_patches, n_images, output_dir, "encoder", config
    )
    
    return results


def run_decoder_dataset_experiment(
    model_type: ModelType,
    n_images: int,
    output_dir: str,
    layers: List[int],
    images: List[Image.Image],
    extractor,
    config: ExperimentConfig,
):
    """Run decoder-specific dataset experiment"""
    print("\n--- DECODER ANALYSIS ---")
    
    # Extract features for all images
    all_features = {f"decoder_layer_{l}": [] for l in layers}
    vision_token_counts = []
    
    for image in tqdm(images, desc="Extracting decoder features"):
        features = extractor.extract_features(image, encoder_layers=None, decoder_layers=layers)
        
        vision_indices = features.get('vision_token_indices')
        
        for layer_name in all_features.keys():
            if layer_name in features['decoder']:
                layer_feat = features['decoder'][layer_name]
                if layer_feat.dim() == 3:
                    layer_feat = layer_feat[0]
                
                # Extract vision tokens only
                if vision_indices is not None:
                    layer_feat = layer_feat[vision_indices]
                    vision_token_counts.append(len(vision_indices))
                
                all_features[layer_name].append(layer_feat)
    
    # Determine decoder grid size
    if vision_token_counts:
        n_vision_tokens = vision_token_counts[0]
        n_patches = int(math.sqrt(n_vision_tokens))
        print(f"Decoder vision tokens: {n_vision_tokens} ({n_patches}x{n_patches} grid)")
    else:
        n_patches = config.num_patches_per_side
        print(f"Using encoder grid: {n_patches}")
    
    # Compute and visualize
    results = compute_and_visualize(
        all_features, n_patches, n_images, output_dir, "decoder", config
    )
    
    return results


def compute_and_visualize(
    all_features: Dict[str, List[torch.Tensor]],
    n_patches: int,
    n_images: int,
    output_dir: str,
    part_name: str,
    config: ExperimentConfig,
):
    """Compute similarities and create visualizations"""
    
    print(f"\nComputing z-scored similarities for {part_name}...")
    
    results = {}
    
    for layer_name, features_list in all_features.items():
        if not features_list:
            continue
        
        print(f"  Processing {layer_name}...")
        results[layer_name] = {}
        
        for patch_pos in SELECTED_PATCHES:
            patch_idx = get_patch_index(patch_pos, n_patches)
            
            # Check if patch index is valid
            if features_list[0].shape[0] <= patch_idx:
                print(f"    Skipping {patch_pos.name}: idx {patch_idx} > {features_list[0].shape[0]}")
                continue
            
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
    
    # Visualize results - using 0 to 1 range
    print(f"\nCreating {part_name} visualizations...")
    
    for layer_name in results.keys():
        layer_num = layer_name.replace(f'{part_name}_layer_', '')
        
        # Create figure with 5 subplots (one per position)
        n_positions = len([p for p in SELECTED_PATCHES if p.name in results[layer_name]])
        if n_positions == 0:
            continue
            
        fig, axes = plt.subplots(1, n_positions, figsize=(5*n_positions, 5))
        if n_positions == 1:
            axes = [axes]
        
        ax_idx = 0
        for patch_pos in SELECTED_PATCHES:
            if patch_pos.name not in results[layer_name]:
                continue
                
            data = results[layer_name][patch_pos.name]
            
            # Use 0 to 1 range (cosine similarity is already in [-1, 1], shift to [0, 1])
            # For z-scored features, values can be outside [-1, 1], so we clip
            display_grid = (data['mean'] + 1) / 2  # Map [-1, 1] to [0, 1]
            display_grid = np.clip(display_grid, 0, 1)
            
            im = axes[ax_idx].imshow(display_grid, cmap='RdYlGn', vmin=0, vmax=1)
            axes[ax_idx].set_title(f"{patch_pos.name}")
            
            # Mark selected patch
            patch_idx = get_patch_index(patch_pos, n_patches)
            row = patch_idx // n_patches
            col = patch_idx % n_patches
            axes[ax_idx].plot(col, row, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            # Add min/max annotation (original values)
            axes[ax_idx].text(0.02, 0.98, f"min: {data['mean'].min():.3f}\nmax: {data['mean'].max():.3f}",
                        transform=axes[ax_idx].transAxes, verticalalignment='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_idx += 1
        
        # Add colorbar with 0-1 range
        plt.colorbar(im, ax=axes, label="Similarity (0-1)", shrink=0.8)
        
        fig.suptitle(f"{part_name.upper()} Layer {layer_num} - Average Similarity (n={n_images})", fontsize=14)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, "figures", f"{part_name}_avg_similarity_L{layer_num}.png")
        save_figure(fig, fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
    
    # Row vs column analysis
    print_row_vs_column_analysis(results, n_patches, part_name)
    
    return results


def print_row_vs_column_analysis(results: dict, n_patches: int, part_name: str):
    """Print row vs column similarity analysis"""
    print(f"\n{'='*60}")
    print(f"ROW vs COLUMN ANALYSIS ({part_name.upper()} - Center patch)")
    print("=" * 60)
    
    center_pos = SELECTED_PATCHES[0]  # Center
    center_row = int(center_pos.rel_y * (n_patches - 1))
    center_col = int(center_pos.rel_x * (n_patches - 1))
    
    for layer_name in results.keys():
        layer_num = layer_name.replace(f'{part_name}_layer_', '')
        
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


def run_dataset_experiment(
    model_type: ModelType,
    n_images: int = 64,
    n_objects: int = 20,
    background_color: tuple = (0, 0, 255),
    output_dir: str = None,
    encoder_layers: List[int] = None,
    decoder_layers: List[int] = None,
    run_encoder: bool = True,
    run_decoder: bool = False,
    use_coco: bool = False,
    coco_path: str = None,
):
    """Run the dataset similarity experiment"""
    
    config = ExperimentConfig(model_type=model_type)
    
    if output_dir is None:
        dataset_type = "coco" if use_coco else "synthetic"
        output_dir = f"outputs/dataset_{dataset_type}_{model_type.value}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    print("=" * 60)
    print("DATASET SIMILARITY EXPERIMENT")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {'COCO' if use_coco else 'Synthetic geometric'}")
    print(f"Number of images: {n_images}")
    print(f"Run encoder: {run_encoder}")
    print(f"Run decoder: {run_decoder}")
    print(f"Selected patches: {[p.name for p in SELECTED_PATCHES]}")
    
    # Load/generate dataset
    print("\n1. Loading dataset...")
    
    if use_coco:
        if not coco_path or not os.path.exists(coco_path):
            raise ValueError(f"COCO path not found: {coco_path}")
        images = load_coco_images(coco_path, n_images, target_size=config.image_size)
    else:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        images = generate_dataset(
            n_images=n_images,
            image_size=config.image_size,
            n_objects=n_objects,
            background_color=background_color,
            output_dir=os.path.join(output_dir, "images"),
        )
    
    # Get feature extractor
    print("\n2. Loading model...")
    extractor = get_feature_extractor(model_type)
    
    results = {}
    
    # Run encoder experiment
    if run_encoder:
        if encoder_layers is None:
            encoder_layers = [1, 5, 15, 31]
        results['encoder'] = run_encoder_dataset_experiment(
            model_type, n_images, output_dir, encoder_layers, images, extractor, config
        )
    
    # Run decoder experiment
    if run_decoder:
        if decoder_layers is None:
            decoder_layers = [1, 5, 10, 15, 20, 27]
        results['decoder'] = run_decoder_dataset_experiment(
            model_type, n_images, output_dir, decoder_layers, images, extractor, config
        )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run dataset similarity experiment")
    parser.add_argument("--model", type=str, default="qwen", help="Model: 'qwen' or 'llava'")
    parser.add_argument("--n-images", type=int, default=64, help="Number of images")
    parser.add_argument("--n-objects", type=int, default=20, help="Objects per image (synthetic only)")
    parser.add_argument("--background", type=str, default="blue", 
                        help="Background color (synthetic only)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--encoder-layers", type=str, default="1,5,15,31",
                        help="Comma-separated encoder layers")
    parser.add_argument("--decoder-layers", type=str, default="1,5,10,15,20,27",
                        help="Comma-separated decoder layers")
    parser.add_argument("--encoder", action="store_true", help="Run encoder analysis")
    parser.add_argument("--decoder", action="store_true", help="Run decoder analysis")
    
    # COCO dataset options
    parser.add_argument("--coco", action="store_true", help="Use COCO dataset instead of synthetic")
    parser.add_argument("--coco-path", type=str, default=None,
                        help="Path to COCO images directory (e.g., /path/to/val2017)")
    
    args = parser.parse_args()
    
    # If neither specified, run encoder by default
    run_encoder = args.encoder
    run_decoder = args.decoder
    if not run_encoder and not run_decoder:
        run_encoder = True
    
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
    encoder_layers = [int(l) for l in args.encoder_layers.split(',')]
    decoder_layers = [int(l) for l in args.decoder_layers.split(',')]
    
    model_type = get_model_type(args.model)
    
    run_dataset_experiment(
        model_type=model_type,
        n_images=args.n_images,
        n_objects=args.n_objects,
        background_color=background,
        output_dir=args.output,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        run_encoder=run_encoder,
        run_decoder=run_decoder,
        use_coco=args.coco,
        coco_path=args.coco_path,
    )


if __name__ == "__main__":
    main()
