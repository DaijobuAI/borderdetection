"""
Document Scanner - Inference Script using DocAligner PyPI Package

This script performs document corner detection and cropping using:
- Pre-trained DocAligner model from PyPI (default)
- OR your custom fine-tuned model weights

Features:
- Automatically saves both scanned document and visualization for each image
- Organizes outputs in separate folders: outputs/{image_name}/

Usage:
    # Use pre-trained model (automatic download)
    # Creates: outputs/document/scanned_document.jpg and outputs/document/visualization_document.jpg
    python inference_pypi.py --image document.jpg

    # Use your custom trained model
    python inference_pypi.py --image document.jpg --weights weights/finetuned_model.pth

    # Process multiple images (each gets its own folder)
    python inference_pypi.py --image receipt.jpg     # outputs/receipt/
    python inference_pypi.py --image invoice.jpg     # outputs/invoice/
"""

import argparse
import sys
import os
from pathlib import Path

# Enable UTF-8 output for emojis on Windows
if sys.platform == 'win32':
    os.system('')  # Enable ANSI escape sequences
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

try:
    import cv2
    import numpy as np
except ImportError:
    print("[X] Error: Required packages not installed")
    print("Run: pip install opencv-python numpy")
    sys.exit(1)

try:
    from docaligner import DocAligner
except ImportError as e:
    print("[X] Error: docaligner package not installed")
    print("Run: pip install docaligner-docsaid")
    print(f"\nDetails: {e}")
    sys.exit(1)

from utils import (
    load_image,
    save_image,
    apply_perspective_transform,
    order_corners,
    draw_corners,
    create_visualization
)


def detect_document(image_path: str, model, visualize: bool = False):
    """
    Detect document corners and crop document.

    Args:
        image_path: Path to input image
        model: DocAligner model instance
        visualize: If True, create visualization with corners drawn

    Returns:
        tuple: (cropped_image, corners, original_image) or None if detection fails
    """
    # Load image
    print(f" Loading image: {image_path}")
    image = load_image(image_path)
    if image is None:
        print(f" Failed to load image: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"   Image size: {w}x{h}")

    # Detect corners using DocAligner
    print(" Detecting document corners...")
    try:
        # DocAligner expects BGR image and returns numpy array of corners
        corners = model(image)

        # Convert to numpy array if needed
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners, dtype=np.float32)

        # Check if we got valid corners
        if corners is None or len(corners) == 0:
            print(" No document detected in image")
            return None

    except Exception as e:
        print(f" Error during detection: {e}")
        return None

    # Ensure we have 4 corners
    if corners.shape[0] != 4:
        print(f" Expected 4 corners, got {corners.shape[0]}")
        return None

    print(f"✓ Detected 4 corners:")
    for i, (x, y) in enumerate(corners):
        print(f"   Corner {i+1}: ({int(x)}, {int(y)})")

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)

    # Apply perspective transform to crop document
    print(" Applying perspective transform...")
    cropped = apply_perspective_transform(image, corners)

    if cropped is None:
        print(" Failed to crop document")
        return None

    crop_h, crop_w = cropped.shape[:2]
    print(f" Cropped document size: {crop_w}x{crop_h}")

    # Create visualization if requested
    visualization = None
    if visualize:
        print(" Creating visualization...")
        visualization = create_visualization(image, corners, cropped)

    return cropped, corners, image, visualization


def main():
    parser = argparse.ArgumentParser(
        description="Document Scanner - Detect and crop documents from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (pre-trained model)
  # Creates: outputs/document/scanned_document.jpg and outputs/document/visualization_document.jpg
  python inference_pypi.py --image document.jpg

  # Use custom trained model
  python inference_pypi.py --image document.jpg --weights weights/my_model.pth

  # Process multiple images (each gets its own folder)
  python inference_pypi.py --image receipt.jpg     # outputs/receipt/
  python inference_pypi.py --image invoice.jpg     # outputs/invoice/

  # Custom output paths
  python inference_pypi.py --image invoice.jpg --output my_scan.jpg --vis-output my_vis.jpg
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input document image'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save cropped document (default: outputs/{image_name}/scanned_{image_name}.jpg)'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to custom trained model weights (optional). If not provided, uses pre-trained PyPI model'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Create visualization showing detected corners (default: True)'
    )

    parser.add_argument(
        '--vis-output',
        type=str,
        default=None,
        help='Path to save visualization (default: outputs/{image_name}/visualization_{image_name}.jpg)'
    )

    args = parser.parse_args()

    # Validate input image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)

    # Auto-generate output filenames if not provided
    input_path = Path(args.image)
    input_stem = input_path.stem  # filename without extension

    # Create output directory for this image
    output_dir = Path("outputs") / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        args.output = str(output_dir / f"scanned_{input_stem}.jpg")

    if args.vis_output is None:
        args.vis_output = str(output_dir / f"visualization_{input_stem}.jpg")

    # Initialize DocAligner model
    print(" Initializing DocAligner model...")
    try:
        if args.weights:
            # Load custom trained weights
            if not Path(args.weights).exists():
                print(f"❌ Error: Weights file not found: {args.weights}")
                sys.exit(1)
            print(f"   Loading custom weights: {args.weights}")
            model = DocAligner(weights_path=args.weights)
        else:
            # Use pre-trained PyPI model (automatic download)
            print("   Using pre-trained model from PyPI")
            model = DocAligner()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Process document
    print(f"\n{'='*60}")
    result = detect_document(args.image, model, visualize=args.visualize)

    if result is None:
        print(f"{'='*60}")
        print("❌ Document detection failed")
        sys.exit(1)

    cropped, corners, original, visualization = result

    # Save cropped document
    print(f"\n Saving results...")
    save_image(cropped, args.output)
    print(f"✓ Cropped document saved: {args.output}")

    # Save visualization if created
    if visualization is not None:
        save_image(visualization, args.vis_output)
        print(f"✓ Visualization saved: {args.vis_output}")

    print(f"{'='*60}")
    print(" Document scanning complete!")
    print(f"\nOutputs:")
    print(f"   Scanned document: {args.output}")
    if visualization is not None:
        print(f"   Visualization: {args.vis_output}")
    print()


if __name__ == "__main__":
    main()
