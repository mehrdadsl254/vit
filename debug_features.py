"""
Debug script to verify feature extraction from Qwen 2.5 VL.

Run this to check:
1. Actual number of patches extracted
2. Shape of features at each layer
3. Verify the patch grid assumption
"""

import torch
from PIL import Image
import numpy as np
from feature_extractor import QwenVLFeatureExtractor
from config import ExperimentConfig
from generate_image import create_solid_color_image


def debug_feature_extraction():
    config = ExperimentConfig()
    
    print("=" * 60)
    print("DEBUG: Feature Extraction Verification")
    print("=" * 60)
    
    # Create test image
    print(f"\nCreating test image: {config.image_size}")
    image = create_solid_color_image((0, 255, 0), config.image_size)
    
    print(f"Expected patches per side: {config.num_patches_per_side}")
    print(f"Expected total patches: {config.total_patches}")
    
    # Initialize extractor
    extractor = QwenVLFeatureExtractor(
        model_name=config.model_name,
        use_flash_attention=False
    )
    
    # Load model
    extractor.load_model()
    
    # Prepare inputs and check what we get
    inputs = extractor.prepare_inputs(image)
    print(f"\n--- Input Shapes ---")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
    
    # Look at the pixel_values more closely
    if 'pixel_values' in inputs:
        pv = inputs['pixel_values']
        print(f"\n--- Pixel Values Analysis ---")
        print(f"  Shape: {pv.shape}")
        print(f"  Dtype: {pv.dtype}")
        # For Qwen VL, pixel_values might have shape [batch, seq, channels, h, w]
        # where seq is the number of visual tokens
    
    if 'image_grid_thw' in inputs:
        grid_thw = inputs['image_grid_thw']
        print(f"\n--- Image Grid THW ---")
        print(f"  {grid_thw}")
        print(f"  This tells us: (temporal, height, width) in patches")
    
    # Extract features
    print(f"\n--- Extracting Features ---")
    features = extractor.extract_features(
        image,
        encoder_layers=[0, 1, 2],  # Just first few layers for debug
        decoder_layers=None
    )
    
    print(f"\n--- Encoder Features ---")
    for layer_name, feat in features['encoder'].items():
        print(f"  {layer_name}: shape = {feat.shape}, dtype = {feat.dtype}")
        
        # Check if shape matches our expectation
        num_tokens = feat.shape[-2] if feat.dim() >= 2 else feat.shape[0]
        hidden_dim = feat.shape[-1]
        
        print(f"    num_tokens = {num_tokens}, hidden_dim = {hidden_dim}")
        
        # Try to infer grid size
        sqrt_tokens = int(np.sqrt(num_tokens))
        if sqrt_tokens * sqrt_tokens == num_tokens:
            print(f"    Grid: {sqrt_tokens} x {sqrt_tokens} (perfect square)")
        else:
            print(f"    NOT a perfect square! sqrt({num_tokens}) = {np.sqrt(num_tokens):.2f}")
    
    # Vision token indices
    if features.get('vision_token_indices') is not None:
        print(f"\n--- Vision Token Indices ---")
        print(f"  {features['vision_token_indices']}")
    else:
        print(f"\n--- Vision Token Indices: NOT FOUND ---")


if __name__ == "__main__":
    debug_feature_extraction()
