"""
Visualization utilities for document corner detection.
"""

import cv2
import numpy as np
from typing import Optional


def draw_corners(image: np.ndarray, 
                corners: np.ndarray,
                color: tuple = (0, 255, 0),
                thickness: int = 2,
                radius: int = 5) -> np.ndarray:
    """
    Draw detected corners on image.
    
    Args:
        image: Input image
        corners: Corner points array, shape (4, 2)
        color: BGR color tuple for drawing
        thickness: Line thickness
        radius: Corner point radius
        
    Returns:
        Image with corners drawn
    """
    vis_image = image.copy()
    
    # Draw corner points
    for corner in corners:
        x, y = corner.astype(int)
        cv2.circle(vis_image, (x, y), radius, color, -1)
    
    # Draw connecting lines
    corners_int = corners.astype(int)
    for i in range(4):
        pt1 = tuple(corners_int[i])
        pt2 = tuple(corners_int[(i + 1) % 4])
        cv2.line(vis_image, pt1, pt2, color, thickness)
    
    return vis_image


def create_visualization(original_image: np.ndarray,
                        corners: np.ndarray,
                        cropped_image: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None) -> np.ndarray:
    """
    Create comprehensive visualization showing original, corners, and cropped result.
    
    Args:
        original_image: Original input image
        corners: Detected corner points
        cropped_image: Optional cropped result
        save_path: Optional path to save visualization
        
    Returns:
        Combined visualization image
    """
    # Draw corners on original
    vis_original = draw_corners(original_image, corners)
    
    if cropped_image is not None:
        # Create side-by-side visualization
        h1, w1 = vis_original.shape[:2]
        h2, w2 = cropped_image.shape[:2]
        
        # Resize cropped to match height
        if h1 != h2:
            scale = h1 / h2
            new_w = int(w2 * scale)
            cropped_resized = cv2.resize(cropped_image, (new_w, h1))
        else:
            cropped_resized = cropped_image
        
        # Combine images horizontally
        combined = np.hstack([vis_original, cropped_resized])
        
        if save_path:
            cv2.imwrite(save_path, combined)
            print(f"✅ Visualization saved: {save_path}")
        
        return combined
    else:
        if save_path:
            cv2.imwrite(save_path, vis_original)
            print(f"✅ Visualization saved: {save_path}")
        
        return vis_original


def add_text_overlay(image: np.ndarray, 
                    text: str,
                    position: tuple = (10, 30),
                    color: tuple = (255, 255, 255),
                    font_scale: float = 0.7,
                    thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position for text
        color: BGR color tuple
        font_scale: Font scale factor
        thickness: Text thickness
        
    Returns:
        Image with text overlay
    """
    vis_image = image.copy()
    cv2.putText(vis_image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    return vis_image
