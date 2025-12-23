"""
Generate datasets of images with random geometric objects.
Used for multi-image similarity experiments.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Default colors (RGB)
DEFAULT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
]

# Shape types
SHAPE_TYPES = ['circle', 'square', 'rectangle', 'triangle', 'ellipse']


@dataclass
class GeometricObject:
    """Represents a geometric object in an image"""
    shape: str
    color: Tuple[int, int, int]
    x: int  # Top-left x
    y: int  # Top-left y
    width: int
    height: int
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def overlaps(self, other: 'GeometricObject', margin: int = 5) -> bool:
        """Check if this object overlaps with another (with margin)"""
        x1, y1, x2, y2 = self.bbox
        ox1, oy1, ox2, oy2 = other.bbox
        
        # Add margin
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        
        return not (x2 < ox1 or ox2 < x1 or y2 < oy1 or oy2 < y1)


def draw_shape(draw: ImageDraw.Draw, obj: GeometricObject):
    """Draw a geometric shape on the image"""
    x1, y1 = obj.x, obj.y
    x2, y2 = obj.x + obj.width, obj.y + obj.height
    
    if obj.shape == 'circle':
        # For circle, use min dimension
        size = min(obj.width, obj.height)
        cx, cy = x1 + obj.width // 2, y1 + obj.height // 2
        draw.ellipse([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=obj.color)
    
    elif obj.shape == 'square':
        size = min(obj.width, obj.height)
        draw.rectangle([x1, y1, x1 + size, y1 + size], fill=obj.color)
    
    elif obj.shape == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=obj.color)
    
    elif obj.shape == 'triangle':
        # Draw equilateral-ish triangle
        points = [
            (x1 + obj.width // 2, y1),  # Top
            (x1, y2),                    # Bottom left
            (x2, y2),                    # Bottom right
        ]
        draw.polygon(points, fill=obj.color)
    
    elif obj.shape == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], fill=obj.color)


def generate_random_object(
    image_size: Tuple[int, int],
    existing_objects: List[GeometricObject],
    shapes: List[str] = SHAPE_TYPES,
    colors: List[Tuple[int, int, int]] = DEFAULT_COLORS,
    min_size: int = 20,
    max_size: int = 60,
    max_attempts: int = 100
) -> Optional[GeometricObject]:
    """Generate a random object that doesn't overlap with existing ones"""
    
    width, height = image_size
    
    for _ in range(max_attempts):
        # Random shape and color
        shape = random.choice(shapes)
        color = random.choice(colors)
        
        # Random size
        obj_width = random.randint(min_size, max_size)
        obj_height = random.randint(min_size, max_size)
        
        # For squares, make dimensions equal
        if shape == 'square':
            obj_height = obj_width
        
        # Random position (ensure fully inside image)
        max_x = width - obj_width - 5
        max_y = height - obj_height - 5
        
        if max_x < 5 or max_y < 5:
            continue
        
        x = random.randint(5, max_x)
        y = random.randint(5, max_y)
        
        # Create candidate object
        candidate = GeometricObject(shape, color, x, y, obj_width, obj_height)
        
        # Check for overlaps
        overlaps = False
        for existing in existing_objects:
            if candidate.overlaps(existing):
                overlaps = True
                break
        
        if not overlaps:
            return candidate
    
    return None  # Failed to place object after max_attempts


def generate_single_image(
    image_size: Tuple[int, int] = (448, 448),
    n_objects: int = 20,
    background_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
    shapes: List[str] = SHAPE_TYPES,
    colors: List[Tuple[int, int, int]] = DEFAULT_COLORS,
    min_size: int = 20,
    max_size: int = 60,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, List[GeometricObject]]:
    """Generate a single image with random geometric objects"""
    
    if seed is not None:
        random.seed(seed)
    
    # Create image with background
    img = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(img)
    
    objects = []
    
    for i in range(n_objects):
        obj = generate_random_object(
            image_size, objects, shapes, colors, min_size, max_size
        )
        if obj is not None:
            objects.append(obj)
            draw_shape(draw, obj)
    
    return img, objects


def generate_dataset(
    n_images: int = 64,
    image_size: Tuple[int, int] = (448, 448),
    n_objects: int = 20,
    background_color: Tuple[int, int, int] = (0, 0, 255),
    shapes: List[str] = SHAPE_TYPES,
    colors: List[Tuple[int, int, int]] = DEFAULT_COLORS,
    min_size: int = 20,
    max_size: int = 60,
    output_dir: Optional[str] = None,
    base_seed: int = 42,
) -> List[Image.Image]:
    """
    Generate a dataset of images with random geometric objects.
    
    Args:
        n_images: Number of images to generate
        image_size: Size of each image (width, height)
        n_objects: Number of objects per image
        background_color: RGB background color
        shapes: List of shape types to use
        colors: List of colors to use
        min_size: Minimum object size
        max_size: Maximum object size
        output_dir: If provided, save images to this directory
        base_seed: Base random seed for reproducibility
        
    Returns:
        List of generated images
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    images = []
    
    for i in range(n_images):
        img, objects = generate_single_image(
            image_size=image_size,
            n_objects=n_objects,
            background_color=background_color,
            shapes=shapes,
            colors=colors,
            min_size=min_size,
            max_size=max_size,
            seed=base_seed + i,
        )
        
        images.append(img)
        
        if output_dir:
            path = os.path.join(output_dir, f"image_{i:04d}.png")
            img.save(path)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_images} images")
    
    print(f"Dataset generation complete: {len(images)} images")
    
    return images


if __name__ == "__main__":
    # Test generation
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate geometric shapes dataset")
    parser.add_argument("--n-images", type=int, default=10, help="Number of images")
    parser.add_argument("--n-objects", type=int, default=20, help="Objects per image")
    parser.add_argument("--output", type=str, default="datasets/geometric", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show first image")
    
    args = parser.parse_args()
    
    images = generate_dataset(
        n_images=args.n_images,
        n_objects=args.n_objects,
        output_dir=args.output,
    )
    
    print(f"Saved {len(images)} images to {args.output}")
    
    if args.show and images:
        images[0].show()
