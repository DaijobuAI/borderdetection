# Document Scanner

Automatic document detection and cropping using DocAligner AI model.

## What It Does

- Detects 4 corners of documents in photos
- Applies perspective correction to straighten documents
- Outputs clean, cropped images
- Works with receipts, invoices, forms, ID cards, etc.

---

## Quick Start

### Installation

```bash
pip install docaligner-docsaid opencv-python numpy
```

### Basic Usage

```bash
python inference_pypi.py --image document.jpg
```

This creates:
- `outputs/document/scanned_document.jpg` - Cropped result
- `outputs/document/visualization_document.jpg` - Shows detected corners

---

## Command Line Examples

### Single Image
```bash
python inference_pypi.py --image receipt.jpg
```

### Process Multiple Images
```bash
python inference_pypi.py --image test_images/test1.jpg
python inference_pypi.py --image test_images/test2.jpg
python inference_pypi.py --image test_images/test3.jpg
```

### Batch Processing (Windows)
```bash
for %f in (test_images\*.jpg) do python inference_pypi.py --image %f
```

### Batch Processing (Linux/Mac)
```bash
for img in test_images/*.jpg; do python inference_pypi.py --image "$img"; done
```

### Custom Output Paths
```bash
python inference_pypi.py --image doc.jpg --output my_scan.jpg --vis-output my_vis.jpg
```

---

## Output Structure

Each processed image gets its own folder:

```
outputs/
├── receipt/
│   ├── scanned_receipt.jpg
│   └── visualization_receipt.jpg
├── invoice/
│   ├── scanned_invoice.jpg
│   └── visualization_invoice.jpg
└── test1/
    ├── scanned_test1.jpg
    └── visualization_test1.jpg
```

---

## How It Works

1. Load image
2. AI model detects 4 document corners
3. Calculate perspective transform matrix
4. Warp image to straighten document
5. Save cropped result and visualization

---

## Project Structure

```
borderdetection/
├── inference_pypi.py          Main inference script
├── requirements.txt           Dependencies
├── README.md                  This file
│
├── utils/                     Helper functions
│   ├── image_utils.py        Image loading/saving
│   ├── transform_utils.py    Perspective transform
│   └── visualization_utils.py Draw corners
│
├── configs/                   Configuration files
│   └── config.py             Settings (for training)
│
├── scripts/                   Setup scripts
│   └── setup_training.py     Training environment setup
│
├── test_images/               Sample images
├── outputs/                   Generated results
├── weights/                   Model weights (optional)
└── training_data/             Training data (optional)
```

---

## Advanced: Training Custom Model

Only needed if the default model doesn't work well for your documents.

### Step 1: Setup Training Environment
```bash
python scripts/setup_training.py
```

### Step 2: Prepare Training Data

Create folder structure:
```
training_data/
├── images/
│   ├── doc1.jpg
│   ├── doc2.jpg
│   └── ...
└── annotations.json
```

### Step 3: Create Annotations

Edit `training_data/annotations.json`:
```json
{
  "doc1.jpg": [[100, 100], [500, 100], [500, 700], [100, 700]],
  "doc2.jpg": [[50, 80], [450, 90], [440, 680], [60, 670]]
}
```

Corner order: top-left, top-right, bottom-right, bottom-left

### Step 4: Train Model
```bash
python train_transfer_learning.py --epochs 50 --batch-size 8
```

### Step 5: Use Custom Model
```bash
python inference_pypi.py --image test.jpg --weights weights/finetuned_model.pth
```

---

## Requirements

### For Inference (Recommended)
```
docaligner-docsaid
opencv-python
numpy
```

### For Training (Optional)
```
torch>=2.0.0
torchvision>=0.15.0
tqdm
tensorboard
```

---


## Model Details

- Based on DocAligner
- Backbone: MobileNetV2 (lightweight and fast)
- Input: RGB images (automatically resized)
- Output: 4 corner coordinates (x,y)
- Inference speed: ~50-200ms per image on CPU

---

## Summary

For most users:
1. Install dependencies
2. Run `python inference_pypi.py --image your_image.jpg`
3. Get scanned document in `outputs/` folder

