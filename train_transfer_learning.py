"""
Transfer learning script for fine-tuning DocAligner on custom documents.

This script loads pre-trained DocAligner weights and fine-tunes on your dataset.

Prerequisites:
    1. Run: python scripts/setup_training.py
    2. Prepare training_data/ folder with images
    3. Create annotations.json with corner coordinates

Usage:
    python train_transfer_learning.py
    python train_transfer_learning.py --epochs 50 --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

from configs.config import TRAINING_CONFIG, WEIGHTS_DIR, TRAINING_DATA_DIR
from utils import load_image, save_image


class DocumentDataset(Dataset):
    """Dataset for document corner detection."""
    
    def __init__(self, image_dir: Path, annotations_file: Path, image_size: int = 256):
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_names = list(self.annotations.keys())
        print(f"✓ Loaded {len(self.image_names)} images")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        img_path = self.image_dir / img_name
        image = load_image(str(img_path))
        
        # Get corners
        corners = np.array(self.annotations[img_name], dtype=np.float32)
        
        # Normalize corners to [0, 1]
        h, w = image.shape[:2]
        corners[:, 0] /= w
        corners[:, 1] /= h
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        corners = torch.from_numpy(corners.flatten())
        
        return image, corners


class SimpleDocModel(nn.Module):
    """Lightweight document corner detection model."""
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.Sigmoid()  # 4 corners × 2 coords
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for images, corners in tqdm(dataloader, desc="Training"):
        images, corners = images.to(device), corners.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, corners)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, corners in dataloader:
            images, corners = images.to(device), corners.to(device)
            predictions = model(images)
            loss = criterion(predictions, corners)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(
    train_dir: Path,
    annotations_file: Path,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    save_path: Path = None
):
    """
    Train document detection model with transfer learning.
    
    Args:
        train_dir: Directory containing training images
        annotations_file: Path to annotations.json
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save trained model
    """
    print("\n" + "=" * 60)
    print("Transfer Learning Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Create dataset
    print(f"\n📁 Loading dataset from: {train_dir}")
    dataset = DocumentDataset(train_dir, annotations_file, 
                             image_size=TRAINING_CONFIG['image_size'])
    
    # Split into train/val
    val_size = int(len(dataset) * TRAINING_CONFIG['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"✓ Training samples: {train_size}")
    print(f"✓ Validation samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("\n🤖 Initializing model...")
    model = SimpleDocModel().to(device)
    
    # Try to load pre-trained weights if available
    pretrained_path = WEIGHTS_DIR / "docaligner_v1.pth"
    if pretrained_path.exists():
        print(f"✓ Loading pre-trained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("✓ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load pre-trained weights: {e}")
            print("   Training from scratch...")
    else:
        print("⚠️  No pre-trained weights found, training from scratch")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  ✅ Model saved: {save_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= TRAINING_CONFIG['patience']:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train DocAligner with transfer learning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--train-dir', type=Path, default=TRAINING_DATA_DIR / "images",
                       help='Directory containing training images')
    parser.add_argument('--annotations', type=Path, default=TRAINING_DATA_DIR / "annotations.json",
                       help='Path to annotations.json file')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--output', type=Path, default=WEIGHTS_DIR / "finetuned_model.pth",
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.train_dir.exists():
        print(f"❌ Error: Training directory not found: {args.train_dir}")
        print("\n📝 Setup instructions:")
        print("1. Run: python scripts/setup_training.py")
        print("2. Create training_data/images/ folder")
        print("3. Add your document images")
        print("4. Create annotations.json")
        sys.exit(1)
    
    if not args.annotations.exists():
        print(f"❌ Error: Annotations file not found: {args.annotations}")
        print("\nCreate annotations.json with format:")
        print('{"image1.jpg": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...}')
        sys.exit(1)
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Train
    try:
        train_model(
            train_dir=args.train_dir,
            annotations_file=args.annotations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=args.output
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
