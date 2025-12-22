"""
Offline visualization from saved features.

Use this to visualize results locally after running extraction on a remote server.

Usage:
    python visualize_offline.py --features outputs/encoder/features/green_surface_encoder_features.npz \
                                --image outputs/encoder/green_surface.png \
                                --component encoder \
                                --output outputs/local/
"""

import argparse
import os
from PIL import Image

from config import ExperimentConfig
from feature_extractor import load_features
from similarity import SimilarityAnalyzer, reshape_similarities_to_grid
from visualize import create_similarity_figure, save_figure


def visualize_offline(
    features_path: str,
    image_path: str,
    component: str,  # 'encoder' or 'decoder'
    output_dir: str = "outputs/local",
):
    """
    Create visualization from saved features.
    
    Args:
        features_path: Path to saved .npz features file
        image_path: Path to input image
        component: 'encoder' or 'decoder'
        output_dir: Output directory
    """
    config = ExperimentConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features and image
    print(f"Loading features from: {features_path}")
    features = load_features(features_path)
    
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Print available layers
    if component == 'encoder':
        available = list(features.get('encoder', {}).keys())
        layers_config = config.encoder_layers
    else:
        available = list(features.get('decoder', {}).keys())
        layers_config = config.decoder_layers
    
    print(f"Available {component} layers: {available}")
    
    # Compute similarities
    print("\nComputing similarities...")
    analyzer = SimilarityAnalyzer(config)
    
    if component == 'encoder':
        similarities = analyzer.analyze_encoder(features['encoder'])
        layer_prefix = "encoder_layer_"
        title = f"Encoder (ViT) Patch Similarity - {image_name}"
    else:
        vision_indices = features.get('vision_token_indices')
        similarities = analyzer.analyze_decoder(features['decoder'], vision_indices)
        layer_prefix = "decoder_layer_"
        title = f"Decoder (LLM) Patch Similarity - {image_name}"
    
    # Get layer names
    layers = [f"{layer_prefix}{i}" for i in layers_config 
              if f"{layer_prefix}{i}" in features[component]]
    
    if not layers:
        print(f"Error: No matching layers found for {component}")
        return None
    
    # Convert to grid format
    sim_grids = {}
    for layer_name, patch_sims in similarities.items():
        sim_grids[layer_name] = {}
        for patch_name, sims in patch_sims.items():
            sim_grids[layer_name][patch_name] = reshape_similarities_to_grid(
                sims, config.num_patches_per_side
            )
    
    # Create and save visualization
    print("\nCreating visualization...")
    fig = create_similarity_figure(
        image, sim_grids, layers, config.selected_patches, config,
        title=title
    )
    
    fig_path = os.path.join(output_dir, f"{image_name}_{component}_similarity.png")
    save_figure(fig, fig_path, dpi=150)
    
    print(f"\nVisualization saved to: {fig_path}")
    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Visualize from saved features")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to saved features .npz file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--component", type=str, choices=['encoder', 'decoder'],
                        required=True, help="Component to visualize")
    parser.add_argument("--output", type=str, default="outputs/local",
                        help="Output directory")
    
    args = parser.parse_args()
    
    visualize_offline(
        features_path=args.features,
        image_path=args.image,
        component=args.component,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
