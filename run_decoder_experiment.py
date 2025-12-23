"""
Decoder (LLM) Patch Similarity Experiment

Analyzes cosine similarity between vision token patches at different LLM decoder layers.
Supports multiple VLM models.

Usage:
    python run_decoder_experiment.py --model qwen --color green
    python run_decoder_experiment.py --model llava --image test_images/test_scene.png
"""

import argparse
import os
import torch
from PIL import Image

from config import ExperimentConfig, ModelType, get_model_type
from generate_image import create_solid_color_image
from feature_extractor import get_feature_extractor, save_features
from similarity import SimilarityAnalyzer, reshape_similarities_to_grid
from visualize import create_single_layer_figure, save_figure


def run_decoder_experiment(
    model_type: ModelType,
    image_path: str = None,
    color: str = None,
    output_dir: str = None,
    save_features_only: bool = False,
):
    """Run the decoder experiment."""
    config = ExperimentConfig(model_type=model_type)
    
    if output_dir is None:
        output_dir = f"outputs/decoder_{model_type.value}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Prepare image
    if image_path:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(config.image_size)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
    elif color:
        colors = {
            'green': (0, 255, 0), 'red': (255, 0, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
            'white': (255, 255, 255), 'gray': (128, 128, 128),
        }
        rgb = colors.get(color.lower(), (0, 255, 0))
        image = create_solid_color_image(rgb, config.image_size)
        image_name = f"{color}_surface"
    else:
        image = create_solid_color_image((0, 255, 0), config.image_size)
        image_name = "green_surface"
    
    # Save input image
    image_save_path = os.path.join(output_dir, f"{image_name}.png")
    image.save(image_save_path)
    print(f"Input image saved: {image_save_path}")
    
    print("\n" + "=" * 50)
    print(f"DECODER EXPERIMENT - {model_type.value.upper()}")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Positional Encoding: {config.positional_encoding}")
    
    # Get feature extractor
    extractor = get_feature_extractor(model_type)
    
    # Extract features
    print(f"\nExtracting features from decoder layers: {config.decoder_layers}")
    features = extractor.extract_features(
        image,
        encoder_layers=None,
        decoder_layers=config.decoder_layers
    )
    
    # Save features
    features_path = os.path.join(output_dir, "features", f"{image_name}_decoder_features.npz")
    save_features(features, features_path)
    
    if save_features_only:
        print("\nFeatures saved. Run visualization separately.")
        return features_path
    
    # Compute similarities
    print("\nComputing similarities...")
    analyzer = SimilarityAnalyzer(config)
    
    vision_indices = features.get('vision_token_indices')
    similarities = analyzer.analyze_decoder(features['decoder'], vision_indices)
    
    # Convert to grid format
    sim_grids = {}
    for layer_name, patch_sims in similarities.items():
        sim_grids[layer_name] = {}
        for patch_name, sims in patch_sims.items():
            sim_grids[layer_name][patch_name] = reshape_similarities_to_grid(
                sims, config.num_patches_per_side
            )
    
    # Get layer names
    layers = [f"decoder_layer_{i}" for i in config.decoder_layers 
              if f"decoder_layer_{i}" in features['decoder']]
    
    # Create ONE figure per layer
    print("\nCreating visualizations (one per layer)...")
    fig_paths = []
    
    for layer_name in layers:
        layer_num = layer_name.replace('decoder_layer_', '')
        
        fig = create_single_layer_figure(
            image, 
            sim_grids[layer_name], 
            layer_name,
            config.selected_patches, 
            config,
            title=f"Decoder Layer {layer_num} ({model_type.value}) - {image_name}",
            figsize=(24, 5),
            decimal_places=4
        )
        
        fig_path = os.path.join(output_dir, "figures", f"{image_name}_decoder_L{layer_num}.png")
        save_figure(fig, fig_path, dpi=150)
        fig_paths.append(fig_path)
        print(f"  Saved: {fig_path}")
    
    print("\n" + "=" * 50)
    print("DECODER EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Features: {features_path}")
    print(f"Figures ({len(fig_paths)} files)")
    
    return fig_paths


def main():
    parser = argparse.ArgumentParser(description="Run decoder patch similarity experiment")
    parser.add_argument("--model", type=str, default="qwen",
                        help="Model to use: 'qwen' or 'llava'")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--color", type=str, default="green",
                        help="Color for solid image (green, red, blue, etc.)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--features-only", action="store_true",
                        help="Only save features, skip visualization")
    
    args = parser.parse_args()
    
    model_type = get_model_type(args.model)
    
    run_decoder_experiment(
        model_type=model_type,
        image_path=args.image,
        color=args.color,
        output_dir=args.output,
        save_features_only=args.features_only,
    )


if __name__ == "__main__":
    main()
