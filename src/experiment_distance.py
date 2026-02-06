"""
Distance-Probability Experiment

Analyzes the relationship between spatial distance of patch pairs and:
1. IsSameObject probability from trained probe
2. Cosine similarity of patch embeddings

Generates 4 scatter plots with regression lines.
"""

import os
import sys

# Add libs to path for dinov2 and ADE20K modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(SCRIPT_DIR, 'libs')
sys.path.insert(0, LIBS_DIR)
# Also add parent directory for imports
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from utils.models import get_model
from utils.utils import set_random_seed

# Global activations buffer
activations = {}


def get_activations(name):
    """Hook function to capture layer activations."""
    def hook(model, input, output):
        if activations.get(name) is None:
            activations[name] = []
        activations[name].append(output.detach())
    return hook


class DistanceExperiment:
    """
    Experiment to analyze relationship between patch spatial distance
    and IsSameObject probability / cosine similarity.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        
        # Get experiment config with defaults
        exp_cfg = cfg.get('experiment_distance', {})
        self.num_images = exp_cfg.get('num_images', 200)
        self.num_pairs_per_image = exp_cfg.get('num_pairs_per_image', 1000)
        self.checkpoint_path = exp_cfg.get('checkpoint_path', 
            '/home/mmd/data/outputs2/quadratic_/layer_18_probe_quadratic/best_checkpoint.pth')
        self.layer = exp_cfg.get('layer', 18)
        self.output_dir = exp_cfg.get('output_dir', 
            '/home/mmd/VIT/vit-object-binding2/results/distance_experiment/')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load backbone
        self._load_backbone()
        
        # Load probe
        self._load_probe()
        
        # Load dataset
        self._load_dataset()
        
    def _load_backbone(self):
        """Load DINOv2 backbone with hook on specified layer."""
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[self.cfg.model.backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"
        
        self.backbone_model = torch.hub.load(
            repo_or_dir=self.cfg.model.model_dir, 
            model=backbone_name, 
            source='local'
        )
        self.backbone_model.eval()
        self.backbone_model.to(self.cfg.device)
        
        # Add hook to capture activations from specified layer
        for num_layer, child in self.backbone_model.blocks.named_children():
            if int(num_layer) == self.layer:
                child.register_forward_hook(get_activations(str(self.layer)))
                print(f"Added hook to layer {self.layer}")
        
        self.patch_size = self.backbone_model.patch_size
        print(f"Loaded DINOv2 backbone: {backbone_name}")
        
    def _load_probe(self):
        """Load trained probe from checkpoint."""
        self.probe = get_model(self.cfg)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.cfg.device, weights_only=False)
        self.probe.load_state_dict(checkpoint['model_state_dict'])
        self.probe.eval()
        self.probe.to(self.cfg.device)
        print(f"Loaded probe from: {self.checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        
    def _load_dataset(self):
        """Load ADE20K validation dataset."""
        from utils.dataset import ADE20KSegmentation
        self.dataset = ADE20KSegmentation(
            root=self.cfg.dataset.data_dir, 
            image_set="val"
        )
        print(f"Loaded dataset with {len(self.dataset)} images")
        
    def _get_patch_embeddings(self, image):
        """
        Get patch embeddings from layer 18 for an image.
        
        Args:
            image: numpy array [H, W, 3] in RGB format
            
        Returns:
            embeddings: tensor [num_patches, C]
            patch_grid_size: int (e.g., 37 for 518x518 input)
        """
        # Clear activation buffer
        activations[str(self.layer)] = []
        
        # Prepare image for DINOv2
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),  # DINOv2 expected size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.cfg.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.backbone_model(img_tensor)
        
        # Get activation from hook
        activation = activations[str(self.layer)][0]  # [1, num_patches+1, C]
        
        # Remove CLS token
        embeddings = activation[0, 1:]  # [num_patches, C]
        
        patch_grid_size = int(math.sqrt(embeddings.shape[0]))
        
        return embeddings, patch_grid_size
    
    def _get_patch_labels(self, instance_mask, patch_grid_size):
        """
        Get instance label for each patch by majority voting.
        
        Args:
            instance_mask: numpy array [H, W] with instance IDs
            patch_grid_size: int
            
        Returns:
            patch_labels: tensor [num_patches] with instance IDs
        """
        import cv2
        
        # Resize instance mask to match patch grid
        target_size = patch_grid_size * self.patch_size
        resized_mask = cv2.resize(
            instance_mask.astype(np.float32), 
            (target_size, target_size), 
            interpolation=cv2.INTER_NEAREST
        ).astype(int)
        
        # Compute majority label for each patch
        patch_labels = np.zeros(patch_grid_size * patch_grid_size, dtype=int)
        for py in range(patch_grid_size):
            for px in range(patch_grid_size):
                y0 = py * self.patch_size
                y1 = (py + 1) * self.patch_size
                x0 = px * self.patch_size
                x1 = (px + 1) * self.patch_size
                patch_region = resized_mask[y0:y1, x0:x1]
                # Majority vote
                values, counts = np.unique(patch_region, return_counts=True)
                patch_labels[py * patch_grid_size + px] = values[np.argmax(counts)]
        
        return torch.tensor(patch_labels)
    
    def _compute_distances(self, idx1, idx2, patch_grid_size):
        """
        Compute Euclidean and Manhattan distances between patches.
        
        Args:
            idx1, idx2: patch indices (flattened)
            patch_grid_size: int
            
        Returns:
            euclidean_dist, manhattan_dist: floats
        """
        y1, x1 = idx1 // patch_grid_size, idx1 % patch_grid_size
        y2, x2 = idx2 // patch_grid_size, idx2 % patch_grid_size
        
        euclidean_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        manhattan_dist = abs(x1 - x2) + abs(y1 - y2)
        
        return euclidean_dist, manhattan_dist
    
    def _compute_probe_probability(self, emb1, emb2):
        """
        Compute IsSameObject probability using the trained probe.
        
        Args:
            emb1, emb2: embeddings [C]
            
        Returns:
            probability: float in [0, 1]
        """
        with torch.no_grad():
            logit = self.probe.forward(emb1.unsqueeze(0), emb2.unsqueeze(0))
            prob = torch.sigmoid(logit).item()
        return prob
    
    def _compute_cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between embeddings.
        
        Args:
            emb1, emb2: embeddings [C]
            
        Returns:
            similarity: float in [-1, 1]
        """
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def run(self):
        """Run the experiment and generate plots."""
        print(f"\n{'='*60}")
        print("Distance-Probability Experiment")
        print(f"{'='*60}")
        print(f"Number of images: {self.num_images}")
        print(f"Pairs per image: {self.num_pairs_per_image}")
        print(f"Layer: {self.layer}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Data collection
        data = {
            'euclidean_dist': [],
            'manhattan_dist': [],
            'probe_prob': [],
            'cosine_sim': [],
            'same_object': []
        }
        
        # Randomly sample image indices
        num_available = len(self.dataset)
        if self.num_images > num_available:
            print(f"Warning: Requested {self.num_images} images but only {num_available} available")
            self.num_images = num_available
        
        image_indices = random.sample(range(num_available), self.num_images)
        
        for img_idx in tqdm(image_indices, desc="Processing images"):
            try:
                # Load image and instance mask
                image, seg, instance_mask = self.dataset[img_idx]
                
                # Get embeddings
                embeddings, patch_grid_size = self._get_patch_embeddings(image)
                num_patches = embeddings.shape[0]
                
                # Get patch labels
                patch_labels = self._get_patch_labels(instance_mask, patch_grid_size)
                
                # Sample random pairs
                if self.num_pairs_per_image == -1:
                    # All pairs
                    pairs = [(i, j) for i in range(num_patches) for j in range(i+1, num_patches)]
                else:
                    # Random sample
                    pairs = []
                    for _ in range(self.num_pairs_per_image):
                        i = random.randint(0, num_patches - 1)
                        j = random.randint(0, num_patches - 1)
                        if i != j:
                            pairs.append((min(i, j), max(i, j)))
                    pairs = list(set(pairs))  # Remove duplicates
                
                # Process pairs
                for idx1, idx2 in pairs:
                    # Distances
                    euc_dist, man_dist = self._compute_distances(idx1, idx2, patch_grid_size)
                    
                    # Probe probability
                    prob = self._compute_probe_probability(embeddings[idx1], embeddings[idx2])
                    
                    # Cosine similarity
                    cos_sim = self._compute_cosine_similarity(embeddings[idx1], embeddings[idx2])
                    
                    # Same object label (both non-background and same instance)
                    label1 = patch_labels[idx1].item()
                    label2 = patch_labels[idx2].item()
                    same_obj = (label1 == label2) and (label1 != 0)
                    
                    # Store
                    data['euclidean_dist'].append(euc_dist)
                    data['manhattan_dist'].append(man_dist)
                    data['probe_prob'].append(prob)
                    data['cosine_sim'].append(cos_sim)
                    data['same_object'].append(same_obj)
                    
            except Exception as e:
                print(f"Error processing image {img_idx}: {e}")
                continue
        
        # Convert to numpy
        for key in data:
            data[key] = np.array(data[key])
        
        print(f"\nCollected {len(data['euclidean_dist'])} data points")
        print(f"  Same object pairs: {data['same_object'].sum()}")
        print(f"  Different object pairs: {(~data['same_object']).sum()}")
        
        # Generate plots
        self._generate_plots(data)
        
        print(f"\nExperiment complete! Results saved to: {self.output_dir}")
    
    def _generate_plots(self, data):
        """Generate 4 scatter plots with regression lines."""
        
        # Separate data by label
        same_mask = data['same_object']
        diff_mask = ~data['same_object']
        
        plots = [
            ('euclidean_dist', 'probe_prob', 'Euclidean Distance', 'Probe Probability', 
             'distance_vs_prob_euclidean.png'),
            ('manhattan_dist', 'probe_prob', 'Manhattan Distance', 'Probe Probability',
             'distance_vs_prob_manhattan.png'),
            ('euclidean_dist', 'cosine_sim', 'Euclidean Distance', 'Cosine Similarity',
             'distance_vs_cosine_euclidean.png'),
            ('manhattan_dist', 'cosine_sim', 'Manhattan Distance', 'Cosine Similarity',
             'distance_vs_cosine_manhattan.png'),
        ]
        
        for x_key, y_key, x_label, y_label, filename in plots:
            self._create_scatter_plot(
                data[x_key], data[y_key], same_mask, diff_mask,
                x_label, y_label, filename
            )
    
    def _create_scatter_plot(self, x_data, y_data, same_mask, diff_mask, 
                             x_label, y_label, filename):
        """Create a single scatter plot with regression lines."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot different object pairs (red) - plot first so same object is on top
        if diff_mask.sum() > 0:
            ax.scatter(x_data[diff_mask], y_data[diff_mask], 
                      c='red', alpha=0.3, s=10, label='Different Object')
            
            # Regression line for different objects
            if diff_mask.sum() > 1:
                slope, intercept, r_value, _, _ = stats.linregress(
                    x_data[diff_mask], y_data[diff_mask]
                )
                x_line = np.linspace(x_data[diff_mask].min(), x_data[diff_mask].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=2.5, 
                       label=f'Different Reg (R²={r_value**2:.3f})')
        
        # Plot same object pairs (blue)
        if same_mask.sum() > 0:
            ax.scatter(x_data[same_mask], y_data[same_mask], 
                      c='blue', alpha=0.3, s=10, label='Same Object')
            
            # Regression line for same objects
            if same_mask.sum() > 1:
                slope, intercept, r_value, _, _ = stats.linregress(
                    x_data[same_mask], y_data[same_mask]
                )
                x_line = np.linspace(x_data[same_mask].min(), x_data[same_mask].max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='green', linestyle='--', linewidth=2.5,
                       label=f'Same Reg (R²={r_value**2:.3f})')
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label} vs {x_label}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
