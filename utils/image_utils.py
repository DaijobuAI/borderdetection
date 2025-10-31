"""
Image loading, saving, and preprocessing utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array (BGR format) or None if failed
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file path.

    Args:
        image: Image as numpy array
        output_path: Path to save the image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f" Saved: {output_path}")


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified dimensions.

    Args:
        image: Input image
        size: Target size as (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, size)


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.

    Args:
        image: Input image

    Returns:
        Dictionary with image information
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    return {
        'height': h,
        'width': w,
        'channels': channels,
        'shape': image.shape,
        'dtype': str(image.dtype)
    }
