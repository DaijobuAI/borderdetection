"""
Enhanced Document Scanner with Multiple Detection Strategies
Tries different approaches to detect difficult documents
"""

import argparse
import sys
import os
from pathlib import Path

# Enable UTF-8 output for emojis on Windows
if sys.platform == 'win32':
    os.system('')
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
    from docaligner.aligner import ModelType
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


def preprocess_image(image, method='clahe'):
    """
    Preprocess image to enhance document edges

    Args:
        image: Input BGR image
        method: Preprocessing method ('clahe', 'sharpen', 'edge_enhance', 'bilateral')

    Returns:
        Preprocessed image
    """
    if method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    elif method == 'sharpen':
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    elif method == 'edge_enhance':
        # Edge enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    elif method == 'bilateral':
        # Bilateral filter (smooths while preserving edges)
        return cv2.bilateralFilter(image, 9, 75, 75)

    elif method == 'brightness':
        # Increase brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        enhanced = cv2.merge([h, s, v])
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)

    return image


def detect_document_enhanced(image_path: str, visualize: bool = False):
    """
    Try multiple detection strategies to find document corners.

    Strategies:
    1. Default heatmap model, no preprocessing
    2. Point model, no preprocessing
    3. Heatmap model with center crop
    4. CLAHE preprocessing + heatmap
    5. Sharpen preprocessing + heatmap
    6. Brightness enhancement + heatmap
    7. Point model with center crop

    Returns:
        tuple: (cropped_image, corners, original_image, visualization, strategy_used) or None
    """
    # Load image once
    print(f" Loading image: {image_path}")
    image = load_image(image_path)
    if image is None:
        print(f" Failed to load image: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"   Image size: {w}x{h}")

    # Define strategies to try
    strategies = [
        ("Heatmap model (default)", ModelType.heatmap, False, None),
        ("Point model", ModelType.point, False, None),
        ("Heatmap + center crop", ModelType.heatmap, True, None),
        ("Heatmap + CLAHE", ModelType.heatmap, False, 'clahe'),
        ("Heatmap + sharpen", ModelType.heatmap, False, 'sharpen'),
        ("Heatmap + brightness", ModelType.heatmap, False, 'brightness'),
        ("Point + center crop", ModelType.point, True, None),
        ("Point + CLAHE", ModelType.point, False, 'clahe'),
    ]

    print(f"\n Trying {len(strategies)} detection strategies...")

    for i, (name, model_type, center_crop, preprocess) in enumerate(strategies, 1):
        print(f"\n Strategy {i}/{len(strategies)}: {name}")

        try:
            # Initialize model with specified type
            model = DocAligner(model_type=model_type)

            # Preprocess if needed
            test_image = image.copy()
            if preprocess:
                print(f"   Applying {preprocess} preprocessing...")
                test_image = preprocess_image(test_image, preprocess)

            # Detect corners
            print("   Detecting corners...")
            corners = model(test_image, do_center_crop=center_crop)

            # Convert to numpy array if needed
            if not isinstance(corners, np.ndarray):
                corners = np.array(corners, dtype=np.float32)

            # Check if we got valid corners
            if corners is None or len(corners) == 0:
                print("   ✗ No document detected")
                continue

            # Ensure we have 4 corners
            if corners.shape[0] != 4:
                print(f"   ✗ Got {corners.shape[0]} corners, need 4")
                continue

            # Success!
            print(f"   ✓ Successfully detected 4 corners!")
            for j, (x, y) in enumerate(corners):
                print(f"      Corner {j+1}: ({int(x)}, {int(y)})")

            # Order corners
            corners = order_corners(corners)

            # Apply perspective transform (use original image, not preprocessed)
            print("   Applying perspective transform...")
            cropped = apply_perspective_transform(image, corners)

            if cropped is None:
                print("   ✗ Failed to crop document")
                continue

            crop_h, crop_w = cropped.shape[:2]
            print(f"   Cropped document size: {crop_w}x{crop_h}")

            # Create visualization
            visualization = None
            if visualize:
                print("   Creating visualization...")
                visualization = create_visualization(image, corners, cropped)

            return cropped, corners, image, visualization, name

        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue

    print("\n ✗ All strategies failed")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Document Scanner - Try multiple detection strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--image', type=str, required=True, help='Path to input document image')
    parser.add_argument('--output', type=str, default=None, help='Path to save cropped document')
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualization')
    parser.add_argument('--vis-output', type=str, default=None, help='Path to save visualization')

    args = parser.parse_args()

    # Validate input
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)

    # Auto-generate output filenames
    input_path = Path(args.image)
    input_stem = input_path.stem

    output_dir = Path("outputs") / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        args.output = str(output_dir / f"scanned_{input_stem}.jpg")
    if args.vis_output is None:
        args.vis_output = str(output_dir / f"visualization_{input_stem}.jpg")

    # Process document with enhanced detection
    print(f"\n{'='*60}")
    result = detect_document_enhanced(args.image, visualize=args.visualize)

    if result is None:
        print(f"{'='*60}")
        print("❌ Document detection failed with all strategies")
        sys.exit(1)

    cropped, corners, original, visualization, strategy = result

    # Save results
    print(f"\n Saving results...")
    save_image(cropped, args.output)
    print(f"✓ Cropped document saved: {args.output}")

    if visualization is not None:
        save_image(visualization, args.vis_output)
        print(f"✓ Visualization saved: {args.vis_output}")

    print(f"{'='*60}")
    print(f" Document scanning complete!")
    print(f" Strategy used: {strategy}")
    print(f"\nOutputs:")
    print(f"   Scanned document: {args.output}")
    if visualization is not None:
        print(f"   Visualization: {args.vis_output}")
    print()


if __name__ == "__main__":
    main()
