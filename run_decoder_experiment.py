"""
Decoder (LLM) Patch Similarity Experiment

Analyzes cosine similarity between vision token patches at different LLM decoder layers.

Usage:
    python run_decoder_experiment.py --model qwen --noisy --noise-level 0.001
    python run_decoder_experiment.py --model qwen --noisy --centered
"""

import argparse
import os
import torch
from PIL import Image

from config import ExperimentConfig, ModelType, get_model_type
from generate_image import create_solid_color_image, create_noisy_uniform_image
from feature_extractor import get_feature_extractor, save_features
from similarity import SimilarityAnalyzer, reshape_similarities_to_grid
from visualize import create_single_layer_figure, save_figure


def run_decoder_experiment(
    model_type: ModelType,
    image_path: str = None,
    color: str = None,
    output_dir: str = None,
    save_features_only: bool = False,
    use_noisy: bool = False,
    noise_level: float = 0.001,
    use_centering: bool = False,
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
        
        if use_noisy:
            image = create_noisy_uniform_image(rgb, config.image_size, config.patch_size, noise_level)
            image_name = f"{color}_noisy_{noise_level}"
        else:
            image = create_solid_color_image(rgb, config.image_size)
            image_name = f"{color}_surface"
    else:
        image = create_solid_color_image((0, 255, 0), config.image_size)
        image_name = "green_surface"
    
    # Add centering info to name
    if use_centering:
        image_name += "_centered"
    
    # Save input image
    image_save_path = os.path.join(output_dir, f"{image_name}.png")
    image.save(image_save_path)
    print(f"Input image saved: {image_save_path}")
    
    print("\n" + "=" * 50)
    print(f"DECODER EXPERIMENT - {model_type.value.upper()}")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Use noisy image: {use_noisy} (noise_level={noise_level})")
    print(f"Use mean-centering: {use_centering}")
    
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
    
    # Determine decoder vision grid size
    vision_indices = features.get('vision_token_indices')
    if vision_indices is not None:
        num_vision_tokens = len(vision_indices)
        import math
        decoder_grid_size = int(math.sqrt(num_vision_tokens))
        print(f"Decoder vision tokens: {num_vision_tokens} ({decoder_grid_size}x{decoder_grid_size} grid)")
    else:
        decoder_grid_size = config.num_patches_per_side
        print(f"No vision indices found, using encoder grid: {decoder_grid_size}")
    
    # Compute similarities
    print("\nComputing similarities...")
    analyzer = SimilarityAnalyzer(config, use_centering=use_centering)
    similarities = analyzer.analyze_decoder(features['decoder'], vision_indices)
    
    # Convert to grid format using decoder grid size
    sim_grids = {}
    for layer_name, patch_sims in similarities.items():
        sim_grids[layer_name] = {}
        for patch_name, sims in patch_sims.items():
            sim_grids[layer_name][patch_name] = reshape_similarities_to_grid(
                sims, decoder_grid_size
            )
    
    # Get layer names
    layers = [f"decoder_layer_{i}" for i in config.decoder_layers 
              if f"decoder_layer_{i}" in features['decoder']]
    
    # Create a modified config for decoder grid size
    from dataclasses import replace
    # Can't use replace on our config, so just override manually in visualization
    
    # Create ONE figure per layer
    print("\nCreating visualizations (one per layer)...")
    fig_paths = []
    
    for layer_name in layers:
        layer_num = layer_name.replace('decoder_layer_', '')
        
        title_suffix = "centered" if use_centering else "raw"
        
        # For decoder, we need to adjust selected patches to decoder grid
        adjusted_patches = []
        for patch_pos in config.selected_patches:
            # Create adjusted patches with decoder grid index calculation
            class AdjustedPatch:
                def __init__(self, name, rel_x, rel_y, grid_size):
                    self.name = name
                    self.rel_x = rel_x
                    self.rel_y = rel_y
                    self._grid_size = grid_size
            adjusted_patches.append(AdjustedPatch(
                patch_pos.name, patch_pos.rel_x, patch_pos.rel_y, decoder_grid_size
            ))
        
        # Override config for decoder visualization
        class DecoderConfig:
            def __init__(self, original_config, decoder_grid):
                self.num_patches_per_side = decoder_grid
                self.selected_patches = original_config.selected_patches
                self.patch_size = original_config.patch_size
                self.image_size = original_config.image_size
            def get_patch_index(self, pos):
                n = self.num_patches_per_side
                col = int(pos.rel_x * (n - 1))
                row = int(pos.rel_y * (n - 1))
                return row * n + col
        
        decoder_config = DecoderConfig(config, decoder_grid_size)
        
        fig = create_single_layer_figure(
            image, 
            sim_grids[layer_name], 
            layer_name,
            config.selected_patches, 
            decoder_config,
            title=f"Decoder Layer {layer_num} ({title_suffix}) - {image_name}",
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
                        help="Color for solid image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--features-only", action="store_true",
                        help="Only save features, skip visualization")
    parser.add_argument("--noisy", action="store_true",
                        help="Add subtle per-patch noise")
    parser.add_argument("--noise-level", type=float, default=0.001,
                        help="Noise intensity (default 0.001 = 0.1%)")
    parser.add_argument("--centered", action="store_true",
                        help="Subtract mean before computing similarity")
    
    args = parser.parse_args()
    
    model_type = get_model_type(args.model)
    
    run_decoder_experiment(
        model_type=model_type,
        image_path=args.image,
        color=args.color,
        output_dir=args.output,
        save_features_only=args.features_only,
        use_noisy=args.noisy,
        noise_level=args.noise_level,
        use_centering=args.centered,
    )


if __name__ == "__main__":
    main()
