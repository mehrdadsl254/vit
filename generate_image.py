"""
Generate solid color test images for the experiment
"""

from PIL import Image
import os
from typing import Tuple, Optional


# Default settings
DEFAULT_IMAGE_SIZE = (448, 448)
DEFAULT_COLOR = (0, 255, 0)  # Green


def create_solid_color_image(
    color: Tuple[int, int, int] = DEFAULT_COLOR,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Create a solid color image.
    
    Args:
        color: RGB tuple (0-255 for each channel)
        size: (width, height) in pixels
        output_path: Optional path to save the image
        
    Returns:
        PIL Image object
    """
    img = Image.new('RGB', size, color)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        print(f"Saved image to: {output_path}")
    
    return img


def create_test_images(output_dir: str = "outputs", size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> dict:
    """
    Create a set of test images with different colors.
    
    Returns:
        Dictionary of color_name -> image_path
    """
    colors = {
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    for name, color in colors.items():
        path = os.path.join(output_dir, f"{name}_surface.png")
        create_solid_color_image(color, size, path)
        paths[name] = path
    
    return paths


if __name__ == "__main__":
    print("Creating test images...")
    paths = create_test_images()
    print("\nCreated images:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def create_noisy_uniform_image(
    base_color: Tuple[int, int, int] = DEFAULT_COLOR,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    patch_size: int = 14,
    noise_level: float = 0.01,  # Small noise (1% of 255)
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Create an image with uniform base color but subtle noise per patch.
    
    Each patch gets a tiny random offset so they're similar but not identical.
    This breaks the "all identical embeddings" problem while keeping patches
    close enough that positional encoding effects should dominate.
    
    Args:
        base_color: Base RGB color
        size: Image (width, height)
        patch_size: Size of each patch
        noise_level: Noise intensity (0.01 = 1% variation)
        output_path: Optional path to save
        
    Returns:
        PIL Image with subtle per-patch noise
    """
    import numpy as np
    
    width, height = size
    n_patches_w = width // patch_size
    n_patches_h = height // patch_size
    
    # Create base image array
    img_array = np.zeros((height, width, 3), dtype=np.float32)
    
    # Add base color
    for c in range(3):
        img_array[:, :, c] = base_color[c]
    
    # Add unique noise to each patch
    np.random.seed(42)  # For reproducibility
    for py in range(n_patches_h):
        for px in range(n_patches_w):
            # Random offset for this patch (very small)
            noise = np.random.randn(3) * noise_level * 255
            
            # Apply to this patch
            y1, y2 = py * patch_size, (py + 1) * patch_size
            x1, x2 = px * patch_size, (px + 1) * patch_size
            
            for c in range(3):
                img_array[y1:y2, x1:x2, c] += noise[c]
    
    # Clip to valid range and convert to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        print(f"Saved noisy image to: {output_path}")
    
    return img


def create_gradient_image(
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    direction: str = "horizontal",  # "horizontal", "vertical", or "diagonal"
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Create a gradient image where position determines color.
    This ensures each patch has different content based on position.
    
    Args:
        size: Image (width, height)
        direction: Gradient direction
        output_path: Optional path to save
        
    Returns:
        PIL Image with gradient
    """
    import numpy as np
    
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            if direction == "horizontal":
                val = int(255 * x / width)
                img_array[y, x] = [val, 128, 255 - val]
            elif direction == "vertical":
                val = int(255 * y / height)
                img_array[y, x] = [128, val, 255 - val]
            else:  # diagonal
                val = int(255 * (x + y) / (width + height))
                img_array[y, x] = [val, 255 - val, 128]
    
    img = Image.fromarray(img_array, mode='RGB')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        print(f"Saved gradient image to: {output_path}")
    
    return img
