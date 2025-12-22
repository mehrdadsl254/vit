"""
Debug script to analyze actual similarity patterns.
Compare with and without mean-centering.
"""

import torch
import numpy as np
from PIL import Image
from feature_extractor import QwenVLFeatureExtractor, load_features, save_features
from similarity import compute_patch_similarity, reshape_similarities_to_grid
from config import ExperimentConfig
from generate_image import create_solid_color_image
import torch.nn.functional as F


def compute_similarity_no_centering(features, selected_patch_idx):
    """Compute cosine similarity WITHOUT mean-centering (original method)"""
    features = features.float()
    features_normalized = F.normalize(features, p=2, dim=-1)
    selected_features = features_normalized[selected_patch_idx]
    similarities = torch.mv(features_normalized, selected_features)
    return similarities


def compute_similarity_with_centering(features, selected_patch_idx):
    """Compute cosine similarity WITH mean-centering"""
    features = features.float()
    mean_embedding = features.mean(dim=0, keepdim=True)
    centered_features = features - mean_embedding
    features_normalized = F.normalize(centered_features, p=2, dim=-1)
    selected_features = features_normalized[selected_patch_idx]
    similarities = torch.mv(features_normalized, selected_features)
    return similarities


def analyze_similarity_patterns():
    config = ExperimentConfig()
    
    print("=" * 60)
    print("DEBUG: Similarity Pattern Analysis")
    print("=" * 60)
    
    # Create test image
    image = create_solid_color_image((0, 255, 0), config.image_size)
    
    # Initialize extractor
    extractor = QwenVLFeatureExtractor(
        model_name=config.model_name,
        use_flash_attention=False
    )
    
    # Extract features from first layer only
    features = extractor.extract_features(image, encoder_layers=[1], decoder_layers=None)
    
    layer_features = features['encoder']['encoder_layer_1']
    if layer_features.dim() == 3:
        layer_features = layer_features[0]
    
    print(f"\nFeature shape: {layer_features.shape}")
    
    # Select center patch
    center_idx = config.get_patch_index(config.selected_patches[0])  # Center
    print(f"Center patch index: {center_idx}")
    
    # Compute similarity WITHOUT centering
    sim_no_center = compute_similarity_no_centering(layer_features, center_idx)
    grid_no_center = reshape_similarities_to_grid(sim_no_center, config.num_patches_per_side)
    
    # Compute similarity WITH centering
    sim_with_center = compute_similarity_with_centering(layer_features, center_idx)
    grid_with_center = reshape_similarities_to_grid(sim_with_center, config.num_patches_per_side)
    
    print("\n" + "=" * 60)
    print("WITHOUT Mean-Centering:")
    print("=" * 60)
    print(f"Min: {sim_no_center.min():.6f}")
    print(f"Max: {sim_no_center.max():.6f}")
    print(f"Mean: {sim_no_center.mean():.6f}")
    print(f"Std: {sim_no_center.std():.6f}")
    
    # Show a 5x5 region around center
    center_row = center_idx // config.num_patches_per_side
    center_col = center_idx % config.num_patches_per_side
    print(f"\n5x5 region around center ({center_row}, {center_col}):")
    r1, r2 = max(0, center_row-2), min(config.num_patches_per_side, center_row+3)
    c1, c2 = max(0, center_col-2), min(config.num_patches_per_side, center_col+3)
    print(grid_no_center[r1:r2, c1:c2])
    
    print("\n" + "=" * 60)
    print("WITH Mean-Centering:")
    print("=" * 60)
    print(f"Min: {sim_with_center.min():.6f}")
    print(f"Max: {sim_with_center.max():.6f}")
    print(f"Mean: {sim_with_center.mean():.6f}")
    print(f"Std: {sim_with_center.std():.6f}")
    
    print(f"\n5x5 region around center ({center_row}, {center_col}):")
    print(grid_with_center[r1:r2, c1:c2])
    
    # Check if nearby patches are more similar
    print("\n" + "=" * 60)
    print("Spatial Proximity Analysis (WITH centering):")
    print("=" * 60)
    
    # Compute average similarity by Manhattan distance from center
    distances = {}
    for row in range(config.num_patches_per_side):
        for col in range(config.num_patches_per_side):
            dist = abs(row - center_row) + abs(col - center_col)
            if dist not in distances:
                distances[dist] = []
            distances[dist].append(grid_with_center[row, col])
    
    print("Average similarity by Manhattan distance from center:")
    for dist in sorted(distances.keys())[:10]:
        avg = np.mean(distances[dist])
        print(f"  Distance {dist}: avg similarity = {avg:.6f} (n={len(distances[dist])})")
    
    # Check embeddings directly
    print("\n" + "=" * 60)
    print("Embedding Analysis (first 5 dims of selected patches):")
    print("=" * 60)
    
    # Compare center with neighbor and far patch
    neighbor_idx = center_idx + 1  # Right neighbor
    far_idx = 0  # Top-left corner
    
    print(f"Center [{center_idx}] first 5 dims: {layer_features[center_idx, :5]}")
    print(f"Neighbor [{neighbor_idx}] first 5 dims: {layer_features[neighbor_idx, :5]}")
    print(f"Far [{far_idx}] first 5 dims: {layer_features[far_idx, :5]}")
    
    # Direct cosine similarity
    def raw_cosine(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    
    print(f"\nRaw cosine similarity (no centering):")
    print(f"  Center-Neighbor: {raw_cosine(layer_features[center_idx], layer_features[neighbor_idx]):.6f}")
    print(f"  Center-Far: {raw_cosine(layer_features[center_idx], layer_features[far_idx]):.6f}")


if __name__ == "__main__":
    analyze_similarity_patterns()
