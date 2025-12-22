"""
Cosine similarity computation between patches
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import ExperimentConfig, PatchPosition


def cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for all patches.
    
    Args:
        features: Patch features [num_patches, hidden_dim]
        
    Returns:
        Similarity matrix [num_patches, num_patches]
    """
    # Normalize features
    features_normalized = F.normalize(features, p=2, dim=-1)
    
    # Compute pairwise cosine similarity
    similarity = torch.mm(features_normalized, features_normalized.t())
    
    return similarity


def compute_patch_similarity(
    features: torch.Tensor,
    selected_patch_idx: int
) -> torch.Tensor:
    """
    Compute cosine similarity between a selected patch and all other patches.
    
    Steps:
    1. Compute mean embedding across all patches
    2. Subtract mean from each patch (centering)
    3. L2 normalize the centered embeddings
    4. Compute cosine similarity
    
    Args:
        features: Patch features [num_patches, hidden_dim]
        selected_patch_idx: Index of the selected patch
        
    Returns:
        Similarity scores [num_patches]
    """
    features = features.float()
    
    # Step 1: Compute mean embedding across all patches
    mean_embedding = features.mean(dim=0, keepdim=True)
    
    # Step 2: Subtract mean from each patch (centering)
    centered_features = features - mean_embedding
    
    # Step 3: L2 normalize the centered embeddings
    features_normalized = F.normalize(centered_features, p=2, dim=-1)
    
    # Step 4: Get selected patch features (now centered and normalized)
    selected_features = features_normalized[selected_patch_idx]
    
    # Step 5: Compute cosine similarity with all patches
    similarities = torch.mv(features_normalized, selected_features)
    
    return similarities


def compute_all_similarities(
    layer_features: Dict[str, torch.Tensor],
    selected_patches: List[PatchPosition],
    config: ExperimentConfig,
    vision_token_indices: Optional[torch.Tensor] = None,
    is_decoder: bool = False
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute similarities for all (layer, selected_patch) combinations.
    
    Args:
        layer_features: Dictionary of layer_name -> features
        selected_patches: List of selected patch positions
        config: Experiment configuration
        vision_token_indices: Indices of vision tokens (for decoder)
        is_decoder: Whether these are decoder features
        
    Returns:
        Nested dict: layer_name -> patch_name -> similarity_scores
    """
    results = {}
    
    for layer_name, features in layer_features.items():
        results[layer_name] = {}
        
        # Handle batch dimension
        if features.dim() == 3:
            features = features[0]  # Remove batch dim
        
        # For decoder features, extract vision tokens
        if is_decoder and vision_token_indices is not None:
            features = features[vision_token_indices]
        
        # Get patch features
        num_patches = config.total_patches
        if features.shape[0] > num_patches:
            features = features[:num_patches]
        
        # Compute similarity for each selected patch
        for patch_pos in selected_patches:
            patch_idx = config.get_patch_index(patch_pos)
            
            if patch_idx < features.shape[0]:
                similarities = compute_patch_similarity(features, patch_idx)
                results[layer_name][patch_pos.name] = similarities
            else:
                print(f"Warning: Patch index {patch_idx} out of range for layer {layer_name}")
    
    return results


def reshape_similarities_to_grid(
    similarities: torch.Tensor,
    num_patches_per_side: int
) -> np.ndarray:
    """
    Reshape flat similarity scores to 2D grid.
    
    Args:
        similarities: Flat similarity scores [num_patches]
        num_patches_per_side: Number of patches per side
        
    Returns:
        2D array [H, W] of similarity scores
    """
    return similarities.numpy().reshape(num_patches_per_side, num_patches_per_side)


class SimilarityAnalyzer:
    """
    Analyzer for patch similarity across layers.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def analyze_encoder(
        self,
        encoder_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Analyze encoder layer features"""
        return compute_all_similarities(
            encoder_features,
            self.config.selected_patches,
            self.config,
            is_decoder=False
        )
    
    def analyze_decoder(
        self,
        decoder_features: Dict[str, torch.Tensor],
        vision_token_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Analyze decoder layer features"""
        return compute_all_similarities(
            decoder_features,
            self.config.selected_patches,
            self.config,
            vision_token_indices=vision_token_indices,
            is_decoder=True
        )
    
    def get_statistics(
        self,
        similarities: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, dict]]:
        """
        Compute statistics for similarity scores.
        
        Returns:
            Nested dict with mean, std, min, max for each (layer, patch)
        """
        stats = {}
        
        for layer_name, patch_sims in similarities.items():
            stats[layer_name] = {}
            for patch_name, sims in patch_sims.items():
                stats[layer_name][patch_name] = {
                    'mean': sims.mean().item(),
                    'std': sims.std().item(),
                    'min': sims.min().item(),
                    'max': sims.max().item(),
                }
        
        return stats


if __name__ == "__main__":
    # Test with random features
    config = ExperimentConfig()
    
    print("Testing similarity computation...")
    print(f"Total patches: {config.total_patches}")
    print(f"Patches per side: {config.num_patches_per_side}")
    
    # Create random features
    hidden_dim = 1024
    features = torch.randn(config.total_patches, hidden_dim)
    
    # Test cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(features)
    print(f"\nSimilarity matrix shape: {sim_matrix.shape}")
    print(f"Diagonal values (should be 1.0): {sim_matrix.diag()[:5]}")
    
    # Test single patch similarity
    patch_idx = config.get_patch_index(config.selected_patches[0])
    similarities = compute_patch_similarity(features, patch_idx)
    print(f"\nSimilarity to patch '{config.selected_patches[0].name}' (idx {patch_idx}):")
    print(f"  Shape: {similarities.shape}")
    print(f"  Self-similarity: {similarities[patch_idx]:.4f}")
    print(f"  Mean: {similarities.mean():.4f}, Std: {similarities.std():.4f}")
    
    # Test grid reshape
    grid = reshape_similarities_to_grid(similarities, config.num_patches_per_side)
    print(f"\nReshaped to grid: {grid.shape}")
