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
