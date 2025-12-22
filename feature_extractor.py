"""
Feature extraction from Qwen 2.5 VL with layer hooks

Supports extracting features from:
1. Vision Encoder (ViT) layers
2. LLM Decoder layers (for vision tokens)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from PIL import Image

# Transformers imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Install with: pip install qwen-vl-utils")
    process_vision_info = None


@dataclass
class ExtractedFeatures:
    """Container for extracted features"""
    layer_name: str
    layer_idx: int
    features: torch.Tensor  # Shape: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
    component: str  # 'encoder' or 'decoder'


class HookManager:
    """Manages forward hooks for feature extraction"""
    
    def __init__(self):
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def create_hook(self, name: str) -> Callable:
        """Create a hook function that stores output"""
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                # Most transformer layers return (hidden_states, ...)
                self.features[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                self.features[name] = output.detach().cpu()
            else:
                # Try to get hidden states from output object
                if hasattr(output, 'last_hidden_state'):
                    self.features[name] = output.last_hidden_state.detach().cpu()
                elif hasattr(output, 'hidden_states'):
                    self.features[name] = output.hidden_states[-1].detach().cpu()
        return hook_fn
    
    def register_hook(self, module: nn.Module, name: str):
        """Register a forward hook on a module"""
        handle = module.register_forward_hook(self.create_hook(name))
        self.hooks.append(handle)
    
    def remove_all_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def clear_features(self):
        """Clear stored features"""
        self.features.clear()


class QwenVLFeatureExtractor:
    """
    Feature extractor for Qwen 2.5 VL model.
    
    Hooks into specified layers of:
    - Vision Encoder (ViT blocks)
    - LLM Decoder (transformer layers)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Initialize hook managers
        self.encoder_hooks = HookManager()
        self.decoder_hooks = HookManager()
        
        # Model will be loaded lazily
        self.model = None
        self.processor = None
        self.use_flash_attention = use_flash_attention
        
        # Cache for vision token indices
        self._vision_token_indices: Optional[torch.Tensor] = None
        
    def load_model(self):
        """Load the model and processor"""
        print(f"Loading model: {self.model_name}")
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device,
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print("Model loaded successfully")
        self._print_model_structure()
        
    def _print_model_structure(self):
        """Print model structure for debugging"""
        print("\n--- Model Structure ---")
        
        # Vision encoder
        if hasattr(self.model, 'visual'):
            visual = self.model.visual
            print(f"Vision Encoder type: {type(visual).__name__}")
            if hasattr(visual, 'blocks'):
                print(f"  Number of ViT blocks: {len(visual.blocks)}")
            elif hasattr(visual, 'encoder') and hasattr(visual.encoder, 'layers'):
                print(f"  Number of encoder layers: {len(visual.encoder.layers)}")
        
        # LLM
        if hasattr(self.model, 'model'):
            llm = self.model.model
            if hasattr(llm, 'layers'):
                print(f"LLM Decoder layers: {len(llm.layers)}")
        
        print("-" * 25)
    
    def get_encoder_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get all encoder (ViT) layers"""
        layers = []
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Navigate to vision encoder
        visual = self.model.visual
        
        # Try different possible structures
        if hasattr(visual, 'blocks'):
            # Structure: visual.blocks[i]
            for i, block in enumerate(visual.blocks):
                layers.append((i, block))
        elif hasattr(visual, 'encoder'):
            if hasattr(visual.encoder, 'layers'):
                for i, layer in enumerate(visual.encoder.layers):
                    layers.append((i, layer))
            elif hasattr(visual.encoder, 'blocks'):
                for i, block in enumerate(visual.encoder.blocks):
                    layers.append((i, block))
        
        return layers
    
    def get_decoder_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get all decoder (LLM) layers"""
        layers = []
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Navigate to LLM layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                layers.append((i, layer))
        
        return layers
    
    def register_encoder_hooks(self, layer_indices: List[int]):
        """Register hooks on specified encoder layers"""
        self.encoder_hooks.clear_features()
        self.encoder_hooks.remove_all_hooks()
        
        encoder_layers = self.get_encoder_layers()
        
        for idx in layer_indices:
            if idx < len(encoder_layers):
                layer_idx, layer_module = encoder_layers[idx]
                name = f"encoder_layer_{idx}"
                self.encoder_hooks.register_hook(layer_module, name)
                print(f"Registered hook on encoder layer {idx}")
            else:
                print(f"Warning: Encoder layer {idx} does not exist (max: {len(encoder_layers)-1})")
    
    def register_decoder_hooks(self, layer_indices: List[int]):
        """Register hooks on specified decoder layers"""
        self.decoder_hooks.clear_features()
        self.decoder_hooks.remove_all_hooks()
        
        decoder_layers = self.get_decoder_layers()
        
        for idx in layer_indices:
            if idx < len(decoder_layers):
                layer_idx, layer_module = decoder_layers[idx]
                name = f"decoder_layer_{idx}"
                self.decoder_hooks.register_hook(layer_module, name)
                print(f"Registered hook on decoder layer {idx}")
            else:
                print(f"Warning: Decoder layer {idx} does not exist (max: {len(decoder_layers)-1})")
    
    def prepare_inputs(self, image: Image.Image, prompt: str = "Describe this image."):
        """Prepare inputs for the model"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        if process_vision_info:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None
        
        # Process through processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs
    
    def find_vision_token_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find indices of vision tokens in the sequence"""
        # Vision tokens are typically marked with special token IDs
        # This is model-specific; adjust based on actual tokenizer
        
        # For Qwen VL, vision tokens are typically at the start after special tokens
        # We'll need to check the actual token IDs
        # This is a placeholder - actual implementation may vary
        
        vision_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        # Find range between vision start and end
        input_ids_list = input_ids[0].tolist()
        
        try:
            start_idx = input_ids_list.index(vision_token_id) + 1
            end_idx = input_ids_list.index(vision_end_id)
            return torch.arange(start_idx, end_idx)
        except ValueError:
            # Fallback: assume first N tokens are vision tokens
            # N = number of patches from image
            print("Warning: Could not find vision token markers, using heuristic")
            return None
    
    @torch.no_grad()
    def extract_features(
        self,
        image: Image.Image,
        encoder_layers: Optional[List[int]] = None,
        decoder_layers: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract features from specified layers.
        
        Args:
            image: Input PIL Image
            encoder_layers: List of encoder layer indices to extract from
            decoder_layers: List of decoder layer indices to extract from
            
        Returns:
            Dictionary with 'encoder' and 'decoder' keys, each containing
            layer_name -> features mapping
        """
        if not self.model:
            self.load_model()
        
        # Register hooks
        if encoder_layers:
            self.register_encoder_hooks(encoder_layers)
        if decoder_layers:
            self.register_decoder_hooks(decoder_layers)
        
        # Prepare inputs
        inputs = self.prepare_inputs(image)
        inputs = inputs.to(self.model.device)
        
        # Forward pass (don't generate, just encode)
        # Use the model's forward method directly
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Find vision token indices
        self._vision_token_indices = self.find_vision_token_indices(inputs.input_ids)
        
        # Collect results
        results = {
            'encoder': dict(self.encoder_hooks.features),
            'decoder': dict(self.decoder_hooks.features),
            'vision_token_indices': self._vision_token_indices,
        }
        
        # Clean up hooks
        self.encoder_hooks.remove_all_hooks()
        self.decoder_hooks.remove_all_hooks()
        
        return results
    
    def get_patch_features(
        self,
        features: torch.Tensor,
        num_patches_per_side: int,
        vision_token_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract and reshape features for image patches.
        
        Args:
            features: Raw features [batch, seq_len, hidden_dim]
            num_patches_per_side: Number of patches per image side
            vision_token_indices: Indices of vision tokens (for decoder features)
            
        Returns:
            Patch features [num_patches, hidden_dim] or [H, W, hidden_dim]
        """
        # Remove batch dimension if present
        if features.dim() == 3:
            features = features[0]
        
        # For encoder features, all tokens are typically patches
        # For decoder features, we need to select vision tokens
        if vision_token_indices is not None:
            features = features[vision_token_indices]
        
        num_patches = num_patches_per_side ** 2
        
        # If we have more tokens than patches, might include CLS token etc.
        if features.shape[0] > num_patches:
            # Skip any extra tokens (like CLS)
            features = features[:num_patches]
        elif features.shape[0] < num_patches:
            print(f"Warning: Expected {num_patches} patches, got {features.shape[0]}")
        
        return features


def save_features(features: Dict, output_path: str):
    """Save extracted features to file"""
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Convert to numpy for saving
    save_dict = {}
    for component in ['encoder', 'decoder']:
        if component in features:
            for layer_name, tensor in features[component].items():
                save_dict[f"{component}/{layer_name}"] = tensor.numpy()
    
    if 'vision_token_indices' in features and features['vision_token_indices'] is not None:
        save_dict['vision_token_indices'] = features['vision_token_indices'].numpy()
    
    np.savez_compressed(output_path, **save_dict)
    print(f"Features saved to: {output_path}")


def load_features(input_path: str) -> Dict:
    """Load features from file"""
    data = np.load(input_path)
    
    features = {'encoder': {}, 'decoder': {}}
    
    for key in data.files:
        if key.startswith('encoder/'):
            layer_name = key.split('/', 1)[1]
            features['encoder'][layer_name] = torch.from_numpy(data[key])
        elif key.startswith('decoder/'):
            layer_name = key.split('/', 1)[1]
            features['decoder'][layer_name] = torch.from_numpy(data[key])
        elif key == 'vision_token_indices':
            features['vision_token_indices'] = torch.from_numpy(data[key])
    
    return features


if __name__ == "__main__":
    # Test feature extraction structure (without actually loading model)
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    print("Feature Extractor Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Encoder layers to extract: {config.encoder_layers}")
    print(f"  Decoder layers to extract: {config.decoder_layers}")
    print("\nTo run extraction, use:")
    print("  extractor = QwenVLFeatureExtractor()")
    print("  extractor.load_model()")
    print("  features = extractor.extract_features(image, encoder_layers, decoder_layers)")
