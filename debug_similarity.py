"""
Debug script to analyze actual similarity patterns.
Check which layers show positional encoding effects.
"""

import torch
import numpy as np
from PIL import Image
from feature_extractor import QwenVLFeatureExtractor
from similarity import reshape_similarities_to_grid
from config import ExperimentConfig
from generate_image import create_solid_color_image
import torch.nn.functional as F


def compute_similarity_no_centering(features, selected_patch_idx):
    """Compute cosine similarity WITHOUT mean-centering"""
    features = features.float()
    features_normalized = F.normalize(features, p=2, dim=-1)
    selected_features = features_normalized[selected_patch_idx]
    similarities = torch.mv(features_normalized, selected_features)
    return similarities


def analyze_similarity_patterns():
    config = ExperimentConfig()
    
    print("=" * 60)
    print("DEBUG: Similarity Pattern Analysis - Multi-Layer")
    print("=" * 60)
    
    # Create test image
    image = create_solid_color_image((0, 255, 0), config.image_size)
    
    # Initialize extractor
    extractor = QwenVLFeatureExtractor(
        model_name=config.model_name,
        use_flash_attention=False
    )
    
    # Extract features from multiple layers (early, middle, late)
    test_layers = [1, 5, 15, 25, 31]  # Early to late
    features = extractor.extract_features(image, encoder_layers=test_layers, decoder_layers=None)
    
    center_idx = config.get_patch_index(config.selected_patches[0])  # Center
    neighbor_idx = center_idx + 1  # Right neighbor
    far_idx = 0  # Top-left corner
    
    print(f"\nCenter patch idx: {center_idx}, Neighbor: {neighbor_idx}, Far: {far_idx}")
    
    for layer_idx in test_layers:
        layer_name = f'encoder_layer_{layer_idx}'
        if layer_name not in features['encoder']:
            print(f"\nLayer {layer_idx} not found, skipping...")
            continue
            
        layer_features = features['encoder'][layer_name]
        if layer_features.dim() == 3:
            layer_features = layer_features[0]
        
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Check if embeddings differ
        emb_center = layer_features[center_idx].float()
        emb_neighbor = layer_features[neighbor_idx].float()
        emb_far = layer_features[far_idx].float()
        
        diff_neighbor = (emb_center - emb_neighbor).abs().max().item()
        diff_far = (emb_center - emb_far).abs().max().item()
        
        print(f"Max embedding diff Center-Neighbor: {diff_neighbor:.8f}")
        print(f"Max embedding diff Center-Far: {diff_far:.8f}")
        
        if diff_neighbor > 0.0001:
            # Compute similarities
            sim = compute_similarity_no_centering(layer_features, center_idx)
            print(f"Similarity range: [{sim.min():.6f}, {sim.max():.6f}]")
            print(f"Similarity std: {sim.std():.6f}")
            
            # Check spatial pattern
            grid = reshape_similarities_to_grid(sim, config.num_patches_per_side)
            center_row = center_idx // config.num_patches_per_side
            center_col = center_idx % config.num_patches_per_side
            
            # 3x3 around center
            print(f"\n3x3 region around center:")
            for r in range(center_row-1, center_row+2):
                row_vals = []
                for c in range(center_col-1, center_col+2):
                    if 0 <= r < config.num_patches_per_side and 0 <= c < config.num_patches_per_side:
                        row_vals.append(f"{grid[r,c]:.4f}")
                print(f"  {row_vals}")
        else:
            print("⚠️  ALL EMBEDDINGS IDENTICAL - no positional info encoded!")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If all layers show identical embeddings, Qwen VL uses RoPE")
    print("which only affects attention computation, not output embeddings.")
    print("For a uniform image, patches remain identical regardless of position.")
    
    # Demonstrate the numerical artifact issue
    print("\n" + "=" * 60)
    print("NUMERICAL ARTIFACT DEMONSTRATION")
    print("=" * 60)
    
    # Use last layer features
    layer_features = features['encoder']['encoder_layer_31']
    if layer_features.dim() == 3:
        layer_features = layer_features[0]
    
    # Show what happens with mean-centering on identical vectors
    mean = layer_features.float().mean(dim=0, keepdim=True)
    centered = layer_features.float() - mean
    
    print(f"Original embedding norm (patch 0): {layer_features[0].float().norm():.6f}")
    print(f"Mean embedding norm: {mean.norm():.6f}")
    print(f"Centered embedding norm (patch 0): {centered[0].norm():.10f}")
    print(f"Centered embedding norm (patch 500): {centered[500].norm():.10f}")
    
    if centered[0].norm() < 1e-6:
        print("\n⚠️ PROBLEM: Centered embeddings are near-zero!")
        print("When normalizing near-zero vectors, we get NUMERICAL NOISE!")
        
        # Normalize and show the noise
        normalized = F.normalize(centered, p=2, dim=-1)
        sim = torch.mv(normalized, normalized[center_idx])
        print(f"\nAfter normalization, similarity range: [{sim.min():.6f}, {sim.max():.6f}]")
        print("These 'variations' are numerical artifacts, NOT positional information!")


if __name__ == "__main__":
    analyze_similarity_patterns()
