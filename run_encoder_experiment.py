"""
Encoder (ViT) Patch Similarity Experiment

Analyzes cosine similarity between patches at different ViT encoder layers.
Run this on a server with GPU and the Qwen 2.5 VL model.

Usage:
    python run_encoder_experiment.py --image green_surface.png --output outputs/encoder/
    python run_encoder_experiment.py --color green --output outputs/encoder/
"""

import argparse
import os
import torch
from PIL import Image

from config import ExperimentConfig
from generate_image import create_solid_color_image
from feature_extractor import QwenVLFeatureExtractor, save_features
from similarity import SimilarityAnalyzer, reshape_similarities_to_grid
from visualize import create_similarity_figure, save_figure


def run_encoder_experiment(
    image_path: str = None,
    color: str = None,
    output_dir: str = "outputs/encoder",
    save_features_only: bool = False,
    use_flash_attention: bool = False,  # Disabled by default
):
    """
    Run the encoder experiment.
    
    Args:
        image_path: Path to input image (optional)
        color: Color name for generating solid color image (optional)
        output_dir: Output directory
        save_features_only: If True, only save features without visualization
        use_flash_attention: Whether to use flash attention
    """
    config = ExperimentConfig()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Prepare image
    if image_path:
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
    elif color:
        colors = {
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
        }
        rgb = colors.get(color.lower(), (0, 255, 0))
        image = create_solid_color_image(rgb, config.image_size)
        image_name = f"{color}_surface"
    else:
        # Default: green surface
        image = create_solid_color_image((0, 255, 0), config.image_size)
        image_name = "green_surface"
    
    # Save input image
    image_save_path = os.path.join(output_dir, f"{image_name}.png")
    image.save(image_save_path)
    print(f"Input image saved: {image_save_path}")
    
    # Initialize feature extractor
    print("\n" + "=" * 50)
    print("ENCODER EXPERIMENT")
    print("=" * 50)
    
    extractor = QwenVLFeatureExtractor(
        model_name=config.model_name,
        use_flash_attention=use_flash_attention
    )
    
    # Extract features
    print(f"\nExtracting features from encoder layers: {config.encoder_layers}")
    features = extractor.extract_features(
        image,
        encoder_layers=config.encoder_layers,
        decoder_layers=None  # Only encoder for this experiment
    )
    
    # Save features
    features_path = os.path.join(output_dir, "features", f"{image_name}_encoder_features.npz")
    save_features(features, features_path)
    
    if save_features_only:
        print("\nFeatures saved. Run visualization separately.")
        return features_path
    
    # Compute similarities
    print("\nComputing similarities...")
    analyzer = SimilarityAnalyzer(config)
    similarities = analyzer.analyze_encoder(features['encoder'])
    
    # Print statistics
    stats = analyzer.get_statistics(similarities)
    print("\nSimilarity Statistics:")
    for layer_name, patch_stats in stats.items():
        print(f"\n  {layer_name}:")
        for patch_name, s in patch_stats.items():
            print(f"    {patch_name}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    
    # Convert to grid format for visualization
    sim_grids = {}
    for layer_name, patch_sims in similarities.items():
        sim_grids[layer_name] = {}
        for patch_name, sims in patch_sims.items():
            sim_grids[layer_name][patch_name] = reshape_similarities_to_grid(
                sims, config.num_patches_per_side
            )
    
    # Get layer names that we actually extracted
    layers = [f"encoder_layer_{i}" for i in config.encoder_layers 
              if f"encoder_layer_{i}" in features['encoder']]
    
    # Create visualizations - split into groups of 3 layers for clarity
    print("\nCreating visualizations...")
    layers_per_figure = 3
    fig_paths = []
    
    for fig_idx, start_idx in enumerate(range(0, len(layers), layers_per_figure)):
        layer_group = layers[start_idx:start_idx + layers_per_figure]
        
        # Get layer numbers for title
        layer_nums = [l.replace('encoder_layer_', '') for l in layer_group]
        group_label = f"L{layer_nums[0]}-L{layer_nums[-1]}"
        
        fig = create_similarity_figure(
            image, sim_grids, layer_group, config.selected_patches, config,
            title=f"Encoder Layers {group_label} - {image_name}",
            figsize_per_cell=(3.0, 3.0)  # Larger cells for clarity
        )
        
        # Save figure
        fig_path = os.path.join(output_dir, "figures", f"{image_name}_encoder_{group_label}.png")
        save_figure(fig, fig_path, dpi=150)
        fig_paths.append(fig_path)
        print(f"  Saved: {fig_path}")
    
    print("\n" + "=" * 50)
    print("ENCODER EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Features: {features_path}")
    print(f"Figures ({len(fig_paths)} files):")
    for p in fig_paths:
        print(f"  - {p}")
    
    return fig_paths


def main():
    parser = argparse.ArgumentParser(description="Run encoder patch similarity experiment")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--color", type=str, default="green",
                        help="Color for solid image (green, red, blue, etc.)")
    parser.add_argument("--output", type=str, default="outputs/encoder",
                        help="Output directory")
    parser.add_argument("--features-only", action="store_true",
                        help="Only save features, skip visualization")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash attention")
    
    args = parser.parse_args()
    
    run_encoder_experiment(
        image_path=args.image,
        color=args.color,
        output_dir=args.output,
        save_features_only=args.features_only,
        use_flash_attention=not args.no_flash_attn,
    )


if __name__ == "__main__":
    main()
