"""
Configuration for Qwen 2.5 VL Patch Similarity Experiment

Two experiments:
1. Encoder (ViT) - Vision encoder layers
2. Decoder (LLM) - Language model layers with vision tokens
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Image configuration
DEFAULT_IMAGE_SIZE = (448, 448)  # Width x Height
DEFAULT_COLOR = (0, 255, 0)  # Green RGB

# Patch configuration (Qwen VL uses 14x14 patches)
PATCH_SIZE = 14

# Layer selection pattern:
# First layers: 1,2,3,4,5 (increment by 1)
# Then: 7,9 (increment by 2)  
# Then: 12,15,18,... (increment by 3)
def generate_layer_indices(max_layers: int) -> List[int]:
    """
    Generate layer indices with pattern:
    - 1,2,3,4,5 (by 1)
    - 7,9,11 (by 2)
    - 14,17,20,... (by 3)
    """
    layers = []
    
    # First 5 layers (by 1)
    for i in range(1, 6):
        if i <= max_layers:
            layers.append(i)
    
    # Next layers (by 2)
    current = 7
    while current <= max_layers and current <= 11:
        layers.append(current)
        current += 2
    
    # Remaining layers (by 3)
    current = 14
    while current <= max_layers:
        layers.append(current)
        current += 3
    
    return layers


# Qwen 2.5 VL 7B architecture:
# - ViT encoder: ~32 blocks (approximate, need to verify)
# - LLM decoder: 28 layers
ENCODER_MAX_LAYERS = 32
DECODER_MAX_LAYERS = 28

ENCODER_LAYERS = generate_layer_indices(ENCODER_MAX_LAYERS)
DECODER_LAYERS = generate_layer_indices(DECODER_MAX_LAYERS)


@dataclass
class PatchPosition:
    """Named patch position with relative coordinates (0-1 range)"""
    name: str
    rel_x: float  # 0 = left, 1 = right
    rel_y: float  # 0 = top, 1 = bottom


# 6 selected patch positions
SELECTED_PATCHES: List[PatchPosition] = [
    PatchPosition("Center", 0.5, 0.5),
    PatchPosition("North", 0.5, 0.15),
    PatchPosition("South", 0.5, 0.85),
    PatchPosition("East", 0.85, 0.5),
    PatchPosition("West", 0.15, 0.5),
    PatchPosition("NW", 0.15, 0.15),
]


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model_name: str = MODEL_NAME
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    image_color: Tuple[int, int, int] = DEFAULT_COLOR
    patch_size: int = PATCH_SIZE
    encoder_layers: List[int] = field(default_factory=lambda: ENCODER_LAYERS)
    decoder_layers: List[int] = field(default_factory=lambda: DECODER_LAYERS)
    selected_patches: List[PatchPosition] = field(default_factory=lambda: SELECTED_PATCHES)
    output_dir: str = "outputs"
    
    @property
    def num_patches_per_side(self) -> int:
        return self.image_size[0] // self.patch_size
    
    @property
    def total_patches(self) -> int:
        return self.num_patches_per_side ** 2
    
    def get_patch_index(self, position: PatchPosition) -> int:
        """Convert relative position to patch index"""
        n = self.num_patches_per_side
        col = int(position.rel_x * (n - 1))
        row = int(position.rel_y * (n - 1))
        return row * n + col


def print_config():
    """Print current configuration"""
    config = ExperimentConfig()
    print("=" * 50)
    print("Qwen 2.5 VL Patch Similarity Experiment Config")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Image size: {config.image_size}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Patches per side: {config.num_patches_per_side}")
    print(f"Total patches: {config.total_patches}")
    print(f"\nEncoder layers ({len(config.encoder_layers)}): {config.encoder_layers}")
    print(f"Decoder layers ({len(config.decoder_layers)}): {config.decoder_layers}")
    print(f"\nSelected patches:")
    for pos in config.selected_patches:
        idx = config.get_patch_index(pos)
        print(f"  {pos.name}: ({pos.rel_x}, {pos.rel_y}) -> index {idx}")


if __name__ == "__main__":
    print_config()
