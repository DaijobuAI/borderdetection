"""
Perspective transformation utilities for document cropping.
"""

import cv2
import numpy as np
from typing import Tuple, List


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        corners: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        Ordered corners array
    """
    if corners.shape != (4, 2):
        raise ValueError("Expected 4 corners with shape (4, 2)")
    
    # Calculate center point
    center = np.mean(corners, axis=0)
    
    # Sort by angle from center
    angles = []
    for corner in corners:
        angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
        angles.append(angle)
    
    # Sort corners by angle (clockwise)
    sorted_indices = np.argsort(angles)
    ordered_corners = corners[sorted_indices]
    
    return ordered_corners


def apply_perspective_transform(image: np.ndarray, 
                              corners: np.ndarray,
                              output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Apply perspective transformation to crop document.
    
    Args:
        image: Input image
        corners: 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        output_size: Desired output size (width, height). If None, auto-calculated.
        
    Returns:
        Cropped and straightened document image
    """
    if corners.shape != (4, 2):
        raise ValueError("Expected 4 corners with shape (4, 2)")
    
    # Order corners consistently
    src_points = order_corners(corners).astype(np.float32)
    
    # Define destination points (rectangle)
    if output_size is None:
        # Calculate output size based on corner distances
        width = int(max(
            np.linalg.norm(src_points[0] - src_points[1]),  # top width
            np.linalg.norm(src_points[3] - src_points[2])   # bottom width
        ))
        height = int(max(
            np.linalg.norm(src_points[0] - src_points[3]),  # left height
            np.linalg.norm(src_points[1] - src_points[2])   # right height
        ))
    else:
        width, height = output_size
    
    dst_points = np.array([
        [0, 0],           # top-left
        [width-1, 0],     # top-right
        [width-1, height-1], # bottom-right
        [0, height-1]     # bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    warped = cv2.warpPerspective(image, matrix, (width, height), 
                                 flags=cv2.INTER_LINEAR)
    
    return warped


def calculate_document_dimensions(corners: np.ndarray) -> Tuple[int, int]:
    """
    Calculate optimal document dimensions from corners.
    
    Args:
        corners: 4 corner points
        
    Returns:
        Tuple of (width, height)
    """
    ordered = order_corners(corners)
    
    # Calculate distances
    width = int(max(
        np.linalg.norm(ordered[0] - ordered[1]),  # top
        np.linalg.norm(ordered[3] - ordered[2])   # bottom
    ))
    
    height = int(max(
        np.linalg.norm(ordered[0] - ordered[3]),  # left
        np.linalg.norm(ordered[1] - ordered[2])   # right
    ))
    
    return width, height
