"""
Setup script to clone DocAligner repository and prepare for training.

This script:
1. Clones the official DocAligner repository
2. Installs the package from source
3. Downloads pre-trained weights for transfer learning
4. Sets up the training environment

Usage:
    python scripts/setup_training.py
"""

import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, cwd=None):
    """Run shell command and print output."""
    print(f"\n🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    return True


def setup_training_environment():
    """Setup complete training environment."""
    
    print("=" * 60)
    print("DocAligner Training Environment Setup")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    docaligner_dir = project_root / "DocAligner"
    weights_dir = project_root / "weights"
    
    # Step 1: Clone repository
    print("\n📦 Step 1: Cloning DocAligner repository...")
    if docaligner_dir.exists():
        print(f"⚠️  DocAligner directory already exists: {docaligner_dir}")
        response = input("Delete and re-clone? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(docaligner_dir)
        else:
            print("Skipping clone step...")
            return
    
    clone_cmd = "git clone https://github.com/DocsaidLab/DocAligner.git"
    if not run_command(clone_cmd, cwd=project_root):
        print("❌ Failed to clone repository")
        return
    
    print("✅ Repository cloned successfully")
    
    # Step 2: Install wheel package
    print("\n📦 Step 2: Installing wheel package...")
    if not run_command("pip install wheel"):
        print("❌ Failed to install wheel")
        return
    
    # Step 3: Build wheel
    print("\n🔨 Step 3: Building wheel package...")
    if not run_command("python setup.py bdist_wheel", cwd=docaligner_dir):
        print("❌ Failed to build wheel")
        return
    
    print("✅ Wheel built successfully")
    
    # Step 4: Install wheel
    print("\n📦 Step 4: Installing DocAligner from wheel...")
    dist_dir = docaligner_dir / "dist"
    wheel_files = list(dist_dir.glob("docaligner_docsaid-*.whl"))
    
    if not wheel_files:
        print("❌ No wheel file found in dist/")
        return
    
    wheel_file = wheel_files[0]
    if not run_command(f"pip install {wheel_file}"):
        print("❌ Failed to install wheel")
        return
    
    print("✅ DocAligner installed successfully")
    
    # Step 5: Copy weights
    print("\n📁 Step 5: Setting up weights directory...")
    weights_dir.mkdir(exist_ok=True)
    
    source_weights = docaligner_dir / "weights"
    if source_weights.exists():
        for weight_file in source_weights.glob("*.pth"):
            dest = weights_dir / weight_file.name
            shutil.copy(weight_file, dest)
            print(f"✅ Copied: {weight_file.name}")
    else:
        print("⚠️  No weights found in cloned repository")
        print("   You may need to download them separately")
    
    # Step 6: Install training dependencies
    print("\n📦 Step 6: Installing training dependencies...")
    training_deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorboard",
        "tqdm"
    ]
    
    for dep in training_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️  Failed to install {dep}")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\n📚 Next steps:")
    print("1. Prepare your training data in training_data/")
    print("2. Create annotations.json with corner coordinates")
    print("3. Run: python train_transfer_learning.py")
    print("\n💡 For inference only (no training):")
    print("   python inference_pypi.py --image your_image.jpg")


if __name__ == '__main__':
    try:
        setup_training_environment()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
