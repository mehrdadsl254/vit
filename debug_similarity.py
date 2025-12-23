"""
Debug script to compare positional encoding behavior between models.
Tests Qwen VL (RoPE) vs LLaVA (additive positional embeddings).
"""

import argparse
import torch
import numpy as np
from PIL import Image
from feature_extractor import get_feature_extractor
from similarity import reshape_similarities_to_grid
from config import ExperimentConfig, ModelType, get_model_type
from generate_image import create_solid_color_image
import torch.nn.functional as F


def compute_similarity(features, selected_patch_idx):
    """Compute cosine similarity without centering"""
    features = features.float()
    features_normalized = F.normalize(features, p=2, dim=-1)
    selected_features = features_normalized[selected_patch_idx]
    similarities = torch.mv(features_normalized, selected_features)
    return similarities


def analyze_positional_encoding(model_type: ModelType):
    config = ExperimentConfig(model_type=model_type)
    
    print("=" * 60)
    print(f"POSITIONAL ENCODING ANALYSIS: {model_type.value.upper()}")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Positional Encoding Type: {config.positional_encoding}")
    print(f"Image Size: {config.image_size}")
    print(f"Expected Patches: {config.num_patches_per_side}x{config.num_patches_per_side} = {config.total_patches}")
    
    # Create uniform test image
    image = create_solid_color_image((0, 255, 0), config.image_size)
    
    # Get extractor
    extractor = get_feature_extractor(model_type)
    
    # Test a few layers
    if model_type == ModelType.QWEN_VL:
        test_layers = [1, 5, 15, 31]
    else:  # LLaVA
        test_layers = [1, 5, 12, 23]
    
    features = extractor.extract_features(image, encoder_layers=test_layers, decoder_layers=None)
    
    # Get reference patch indices
    center_idx = config.get_patch_index(config.selected_patches[0])
    neighbor_idx = center_idx + 1
    far_idx = 0
    
    print(f"\nReference patches: Center={center_idx}, Neighbor={neighbor_idx}, Far={far_idx}")
    
    all_identical = True
    
    for layer_idx in test_layers:
        layer_name = f'encoder_layer_{layer_idx}'
        if layer_name not in features['encoder']:
            print(f"\nLayer {layer_idx} not found, skipping...")
            continue
        
        layer_features = features['encoder'][layer_name]
        if layer_features.dim() == 3:
            layer_features = layer_features[0]
        
        # Handle case where we might have fewer tokens than expected
        actual_tokens = layer_features.shape[0]
        if actual_tokens != config.total_patches:
            print(f"\n⚠️ Layer {layer_idx}: Expected {config.total_patches} patches, got {actual_tokens}")
        
        # Adjust indices if needed
        if center_idx >= actual_tokens:
            center_idx = actual_tokens // 2
            neighbor_idx = center_idx + 1 if center_idx + 1 < actual_tokens else center_idx - 1
        
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        print(f"Feature shape: {layer_features.shape}")
        
        # Check embedding differences
        emb_center = layer_features[center_idx].float()
        emb_neighbor = layer_features[neighbor_idx].float()
        emb_far = layer_features[far_idx].float()
        
        diff_neighbor = (emb_center - emb_neighbor).abs().max().item()
        diff_far = (emb_center - emb_far).abs().max().item()
        
        print(f"Max diff Center-Neighbor: {diff_neighbor:.8f}")
        print(f"Max diff Center-Far: {diff_far:.8f}")
        
        if diff_neighbor > 0.0001:
            all_identical = False
            sim = compute_similarity(layer_features, center_idx)
            print(f"✓ DIFFERENT embeddings! Similarity range: [{sim.min():.6f}, {sim.max():.6f}]")
            
            # Show spatial pattern for first layer with differences
            grid_size = int(np.sqrt(actual_tokens))
            if grid_size * grid_size == actual_tokens:
                grid = reshape_similarities_to_grid(sim, grid_size)
                center_row = center_idx // grid_size
                center_col = center_idx % grid_size
                print(f"\n3x3 around center patch:")
                for r in range(max(0, center_row-1), min(grid_size, center_row+2)):
                    row_vals = []
                    for c in range(max(0, center_col-1), min(grid_size, center_col+2)):
                        row_vals.append(f"{grid[r,c]:.4f}")
                    print(f"  {row_vals}")
        else:
            print(f"⚠️ ALL EMBEDDINGS IDENTICAL at this layer")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if all_identical:
        print(f"All patches have IDENTICAL embeddings for uniform image.")
        print(f"This confirms {model_type.value} uses RoPE (position in attention only).")
    else:
        print(f"Patches have DIFFERENT embeddings even for uniform image!")
        print(f"This confirms {model_type.value} uses ADDITIVE positional embeddings.")


def main():
    parser = argparse.ArgumentParser(description="Compare positional encoding behavior")
    parser.add_argument("--model", type=str, default="qwen",
                        help="Model to test: 'qwen' or 'llava'")
    args = parser.parse_args()
    
    model_type = get_model_type(args.model)
    analyze_positional_encoding(model_type)


if __name__ == "__main__":
    main()
