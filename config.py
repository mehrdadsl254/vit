"""
Configuration for VLM Patch Similarity Experiment

Supports multiple models:
- Qwen 2.5 VL 7B (RoPE positional encoding)
- LLaVA 1.5 7B (Additive positional encoding)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import os


class ModelType(Enum):
    """Supported model types"""
    QWEN_VL = "qwen"
    LLAVA = "llava"


# Model configurations
MODEL_CONFIGS = {
    ModelType.QWEN_VL: {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "image_size": (448, 448),  # Can be dynamic
        "patch_size": 14,
        "encoder_max_layers": 32,
        "decoder_max_layers": 28,
        "positional_encoding": "rope",  # Rotary Position Embedding
    },
    ModelType.LLAVA: {
        "name": "llava-hf/llava-1.5-7b-hf",
        "image_size": (336, 336),  # Fixed for CLIP ViT
        "patch_size": 14,
        "encoder_max_layers": 24,  # CLIP ViT-L has 24 layers
        "decoder_max_layers": 32,  # Vicuna/LLaMA has 32 layers
        "positional_encoding": "additive",  # Learned additive embeddings
    },
}


# Layer selection pattern
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
    model_type: ModelType = ModelType.QWEN_VL
    image_color: Tuple[int, int, int] = (0, 255, 0)  # Green RGB
    selected_patches: List[PatchPosition] = field(default_factory=lambda: SELECTED_PATCHES)
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Set model-specific configurations"""
        config = MODEL_CONFIGS[self.model_type]
        self._model_name = config["name"]
        self._image_size = config["image_size"]
        self._patch_size = config["patch_size"]
        self._encoder_max_layers = config["encoder_max_layers"]
        self._decoder_max_layers = config["decoder_max_layers"]
        self._positional_encoding = config["positional_encoding"]
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size
    
    @property
    def patch_size(self) -> int:
        return self._patch_size
    
    @property
    def positional_encoding(self) -> str:
        return self._positional_encoding
    
    @property
    def encoder_layers(self) -> List[int]:
        return generate_layer_indices(self._encoder_max_layers)
    
    @property
    def decoder_layers(self) -> List[int]:
        return generate_layer_indices(self._decoder_max_layers)
    
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


def print_config(model_type: ModelType = ModelType.QWEN_VL):
    """Print current configuration"""
    config = ExperimentConfig(model_type=model_type)
    print("=" * 50)
    print(f"VLM Patch Similarity Experiment Config")
    print("=" * 50)
    print(f"Model Type: {model_type.value}")
    print(f"Model: {config.model_name}")
    print(f"Positional Encoding: {config.positional_encoding}")
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


def get_model_type(model_str: str) -> ModelType:
    """Convert string to ModelType enum"""
    model_str = model_str.lower()
    if model_str in ("qwen", "qwen_vl", "qwenvl"):
        return ModelType.QWEN_VL
    elif model_str in ("llava", "llava1.5", "llava-1.5"):
        return ModelType.LLAVA
    else:
        raise ValueError(f"Unknown model: {model_str}. Use 'qwen' or 'llava'")


if __name__ == "__main__":
    print("\n=== QWEN VL Configuration ===")
    print_config(ModelType.QWEN_VL)
    print("\n\n=== LLAVA Configuration ===")
    print_config(ModelType.LLAVA)
