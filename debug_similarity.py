"""
Debug script to test positional encoding effects using noisy uniform images.
Analyzes if row neighbors are more similar than column neighbors.
"""

import argparse
import torch
import numpy as np
from PIL import Image
from feature_extractor import get_feature_extractor
from similarity import reshape_similarities_to_grid
from config import ExperimentConfig, ModelType, get_model_type
from generate_image import create_noisy_uniform_image, create_solid_color_image
import torch.nn.functional as F


def compute_similarity(features, selected_patch_idx):
    """Compute cosine similarity without centering"""
    features = features.float()
    features_normalized = F.normalize(features, p=2, dim=-1)
    selected_features = features_normalized[selected_patch_idx]
    similarities = torch.mv(features_normalized, selected_features)
    return similarities


def analyze_row_vs_column_similarity(model_type: ModelType, use_noise: bool = True, noise_level: float = 0.01):
    """
    Test if row neighbors (adjacent in sequence) are more similar than column neighbors.
    
    Hypothesis: With 1D positional encoding applied row-by-row:
    - Patches in same row should be more similar (adjacent positions)
    - Patches in same column should be less similar (far apart in sequence)
    """
    config = ExperimentConfig(model_type=model_type)
    
    print("=" * 60)
    print(f"ROW vs COLUMN SIMILARITY ANALYSIS")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Positional Encoding: {config.positional_encoding}")
    print(f"Using noisy image: {use_noise} (noise_level={noise_level})")
    
    # Create test image
    if use_noise:
        image = create_noisy_uniform_image(
            base_color=(0, 255, 0),
            size=config.image_size,
            patch_size=config.patch_size,
            noise_level=noise_level
        )
        print(f"Created noisy uniform image with per-patch variation")
    else:
        image = create_solid_color_image((0, 255, 0), config.image_size)
        print(f"Created solid uniform image")
    
    # Get extractor
    extractor = get_feature_extractor(model_type)
    
    # Test multiple layers
    test_layers = [1, 5, 15, 31] if model_type == ModelType.QWEN_VL else [1, 5, 12, 23]
    features = extractor.extract_features(image, encoder_layers=test_layers, decoder_layers=None)
    
    n = config.num_patches_per_side
    center_row, center_col = n // 2, n // 2
    center_idx = center_row * n + center_col
    
    print(f"\nGrid: {n}x{n} patches")
    print(f"Center patch: ({center_row}, {center_col}) = idx {center_idx}")
    
    for layer_idx in test_layers:
        layer_name = f'encoder_layer_{layer_idx}'
        if layer_name not in features['encoder']:
            continue
        
        layer_features = features['encoder'][layer_name]
        if layer_features.dim() == 3:
            layer_features = layer_features[0]
        
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Check if embeddings differ
        emb_check = layer_features[:10].float()
        max_diff = (emb_check[0] - emb_check[1]).abs().max().item()
        
        if max_diff < 0.0001:
            print("⚠️ All embeddings still identical! Increase noise level.")
            continue
        
        print(f"✓ Embeddings differ (max_diff: {max_diff:.6f})")
        
        # Compute similarities from center patch
        similarities = compute_similarity(layer_features, center_idx)
        grid = reshape_similarities_to_grid(similarities, n)
        
        # Analyze row neighbors vs column neighbors
        row_sims = []  # Same row, different columns
        col_sims = []  # Same column, different rows
        
        for offset in range(1, min(5, n // 2)):  # Check up to 4 neighbors
            # Row neighbors (same row, offset columns)
            if center_col + offset < n:
                row_sims.append(grid[center_row, center_col + offset])
            if center_col - offset >= 0:
                row_sims.append(grid[center_row, center_col - offset])
            
            # Column neighbors (same column, offset rows)
            if center_row + offset < n:
                col_sims.append(grid[center_row + offset, center_col])
            if center_row - offset >= 0:
                col_sims.append(grid[center_row - offset, center_col])
        
        avg_row = np.mean(row_sims) if row_sims else 0
        avg_col = np.mean(col_sims) if col_sims else 0
        
        print(f"\n  Avg similarity to ROW neighbors:    {avg_row:.6f}")
        print(f"  Avg similarity to COLUMN neighbors: {avg_col:.6f}")
        
        if avg_row > avg_col:
            print(f"  ✓ ROW neighbors MORE similar (diff: {avg_row - avg_col:.6f})")
            print(f"    This supports 1D row-wise positional encoding!")
        elif avg_col > avg_row:
            print(f"  ✗ COLUMN neighbors more similar (diff: {avg_col - avg_row:.6f})")
            print(f"    Suggests 2D or no positional effect")
        else:
            print(f"  = No difference detected")
        
        # Show local similarity pattern
        print(f"\n  5x5 region around center:")
        r1, r2 = max(0, center_row-2), min(n, center_row+3)
        c1, c2 = max(0, center_col-2), min(n, center_col+3)
        for r in range(r1, r2):
            row_str = "  "
            for c in range(c1, c2):
                if r == center_row and c == center_col:
                    row_str += " [SEL] "
                else:
                    row_str += f" {grid[r, c]:.3f} "
            print(row_str)
    
    # Conclusion
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("If ROW neighbors are consistently more similar than COLUMN neighbors,")
    print("this indicates 1D positional encoding is applied row-by-row.")
    print("(Patches are numbered 0,1,2...n-1 in first row, then n,n+1...2n-1 in second row, etc)")


def main():
    parser = argparse.ArgumentParser(description="Test row vs column similarity")
    parser.add_argument("--model", type=str, default="qwen", help="Model: 'qwen' or 'llava'")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise level (0-1)")
    parser.add_argument("--no-noise", action="store_true", help="Use uniform image (no noise)")
    args = parser.parse_args()
    
    model_type = get_model_type(args.model)
    analyze_row_vs_column_similarity(
        model_type=model_type,
        use_noise=not args.no_noise,
        noise_level=args.noise
    )


if __name__ == "__main__":
    main()
