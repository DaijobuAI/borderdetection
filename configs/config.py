"""
Configuration settings for document detection and training.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

# Model settings
MODEL_CONFIG = {
    "input_size": 512,  # Input image size for model
    "num_classes": 8,   # 4 corners × 2 coordinates
    "model_name": "docaligner_v1",
    "weights_file": "docaligner_v1.pth"
}

# Training settings
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "epochs": 100,
    "image_size": 256,  # Training image size
    "validation_split": 0.2,
    "save_interval": 10,  # Save model every N epochs
    "patience": 20  # Early stopping patience
}

# Inference settings
INFERENCE_CONFIG = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "max_detections": 1,  # Only one document per image
    "device": "auto"  # 'auto', 'cpu', or 'cuda'
}

# Output settings
OUTPUT_CONFIG = {
    "save_visualization": True,
    "visualization_suffix": "_visualization.jpg",
    "cropped_suffix": "_cropped.jpg",
    "default_output_dir": OUTPUTS_DIR
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    "rotation_range": 10,  # degrees
    "scale_range": (0.8, 1.2),
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "noise_std": 0.01
}
