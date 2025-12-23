"""
Multi-model feature extraction for VLM patch similarity experiments.

Supports:
1. Qwen 2.5 VL (RoPE positional encoding)
2. LLaVA 1.5 (Additive positional encoding)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod

from config import ModelType, ExperimentConfig

# Transformers imports
from transformers import AutoProcessor


class HookManager:
    """Manages forward hooks for feature extraction"""
    
    def __init__(self):
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def create_hook(self, name: str) -> Callable:
        """Create a hook function that stores output"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                self.features[name] = output.detach().cpu()
            else:
                if hasattr(output, 'last_hidden_state'):
                    self.features[name] = output.last_hidden_state.detach().cpu()
                elif hasattr(output, 'hidden_states'):
                    self.features[name] = output.hidden_states[-1].detach().cpu()
        return hook_fn
    
    def register_hook(self, module: nn.Module, name: str):
        handle = module.register_forward_hook(self.create_hook(name))
        self.hooks.append(handle)
    
    def remove_all_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def clear_features(self):
        self.features.clear()


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self, model_name: str, device: str = "auto", dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.encoder_hooks = HookManager()
        self.decoder_hooks = HookManager()
        self.model = None
        self.processor = None
        self._vision_token_indices = None
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def get_encoder_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get encoder layers"""
        pass
    
    @abstractmethod
    def get_decoder_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get decoder layers"""
        pass
    
    @abstractmethod
    def prepare_inputs(self, image: Image.Image, prompt: str = "Describe this image."):
        """Prepare model inputs"""
        pass
    
    def register_encoder_hooks(self, layer_indices: List[int]):
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
    
    @torch.no_grad()
    def extract_features(
        self,
        image: Image.Image,
        encoder_layers: Optional[List[int]] = None,
        decoder_layers: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract features from specified layers"""
        if not self.model:
            self.load_model()
        
        if encoder_layers:
            self.register_encoder_hooks(encoder_layers)
        if decoder_layers:
            self.register_decoder_hooks(decoder_layers)
        
        inputs = self.prepare_inputs(image)
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        results = {
            'encoder': dict(self.encoder_hooks.features),
            'decoder': dict(self.decoder_hooks.features),
            'vision_token_indices': self._vision_token_indices,
        }
        
        self.encoder_hooks.remove_all_hooks()
        self.decoder_hooks.remove_all_hooks()
        
        return results


class QwenVLFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Qwen 2.5 VL (uses RoPE)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 device: str = "auto", dtype: torch.dtype = torch.bfloat16):
        super().__init__(model_name, device, dtype)
        
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            print("Warning: qwen_vl_utils not found. Install with: pip install qwen-vl-utils")
            self.process_vision_info = None
    
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device,
            "attn_implementation": "sdpa",
        }
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
        except ImportError:
            model_kwargs["attn_implementation"] = "eager"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully")
        self._print_structure()
    
    def _print_structure(self):
        print("\n--- Model Structure ---")
        if hasattr(self.model, 'visual'):
            visual = self.model.visual
            print(f"Vision Encoder type: {type(visual).__name__}")
            if hasattr(visual, 'blocks'):
                print(f"  Number of ViT blocks: {len(visual.blocks)}")
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            print(f"LLM Decoder layers: {len(self.model.model.layers)}")
        print("-" * 25)
    
    def get_encoder_layers(self) -> List[Tuple[int, nn.Module]]:
        layers = []
        visual = self.model.visual
        if hasattr(visual, 'blocks'):
            for i, block in enumerate(visual.blocks):
                layers.append((i, block))
        return layers
    
    def get_decoder_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get decoder (LLM) layers from the model"""
        layers = []
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Qwen 2.5 VL structure: model.model.layers
        if hasattr(self.model, 'model'):
            llm = self.model.model
            if hasattr(llm, 'layers'):
                for i, layer in enumerate(llm.layers):
                    layers.append((i, layer))
                print(f"Found {len(layers)} decoder layers in model.model.layers")
                return layers
        
        # Try alternative: model.language_model.model.layers
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                for i, layer in enumerate(lm.model.layers):
                    layers.append((i, layer))
                print(f"Found {len(layers)} decoder layers in language_model.model.layers")
                return layers
        
        # Debug: print model structure if no layers found
        if not layers:
            print("WARNING: No decoder layers found. Model structure:")
            for name, _ in self.model.named_children():
                print(f"  - {name}")
        
        return layers
    
    def prepare_inputs(self, image: Image.Image, prompt: str = "Describe this image."):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if self.process_vision_info:
            image_inputs, video_inputs = self.process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, 
                                padding=True, return_tensors="pt")
        return inputs


class LLaVAFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for LLaVA 1.5 (uses additive positional encoding)"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "auto", dtype: torch.dtype = torch.float16):
        super().__init__(model_name, device, dtype)
    
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        from transformers import LlavaForConditionalGeneration
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully")
        self._print_structure()
    
    def _print_structure(self):
        print("\n--- Model Structure ---")
        if hasattr(self.model, 'vision_tower'):
            vt = self.model.vision_tower
            print(f"Vision Tower type: {type(vt).__name__}")
            if hasattr(vt, 'vision_model') and hasattr(vt.vision_model, 'encoder'):
                layers = vt.vision_model.encoder.layers
                print(f"  Number of ViT layers: {len(layers)}")
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            llm = self.model.language_model.model
            if hasattr(llm, 'layers'):
                print(f"LLM Decoder layers: {len(llm.layers)}")
        print("-" * 25)
    
    def get_encoder_layers(self) -> List[Tuple[int, nn.Module]]:
        layers = []
        if hasattr(self.model, 'vision_tower'):
            vt = self.model.vision_tower
            if hasattr(vt, 'vision_model') and hasattr(vt.vision_model, 'encoder'):
                for i, layer in enumerate(vt.vision_model.encoder.layers):
                    layers.append((i, layer))
        return layers
    
    def get_decoder_layers(self) -> List[Tuple[int, nn.Module]]:
        layers = []
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            llm = self.model.language_model.model
            if hasattr(llm, 'layers'):
                for i, layer in enumerate(llm.layers):
                    layers.append((i, layer))
        return layers
    
    def prepare_inputs(self, image: Image.Image, prompt: str = "Describe this image."):
        # LLaVA uses a specific format
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        return inputs


def get_feature_extractor(model_type: ModelType, **kwargs) -> BaseFeatureExtractor:
    """Factory function to get the appropriate feature extractor"""
    config = ExperimentConfig(model_type=model_type)
    
    if model_type == ModelType.QWEN_VL:
        return QwenVLFeatureExtractor(model_name=config.model_name, **kwargs)
    elif model_type == ModelType.LLAVA:
        return LLaVAFeatureExtractor(model_name=config.model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_features(features: Dict, output_path: str):
    """Save extracted features to file"""
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    save_dict = {}
    for component in ['encoder', 'decoder']:
        if component in features:
            for layer_name, tensor in features[component].items():
                save_dict[f"{component}/{layer_name}"] = tensor.float().numpy()
    
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
