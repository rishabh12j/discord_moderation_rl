"""
Verification script to check all dependencies and setup.
Run this after installation to ensure everything is working.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: FAILED - {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False

def check_directories():
    """Check project directory structure."""
    import os
    dirs = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'src/env',
        'src/agent',
        'src/utils',
        'models/checkpoints',
        'logs',
        'configs'
    ]
    
    all_exist = True
    for d in dirs:
        if os.path.exists(d):
            print(f"✓ Directory exists: {d}")
        else:
            print(f"✗ Directory missing: {d}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("Discord Moderation RL - Environment Setup Verification")
    print("=" * 60)
    
    print("\n1. Checking Core Dependencies...")
    print("-" * 60)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('gymnasium', 'Gymnasium'),
        ('stable_baselines3', 'Stable-Baselines3'),
        ('sb3_contrib', 'SB3-Contrib'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence-Transformers'),
        ('datasets', 'Datasets'),
        ('wandb', 'Weights & Biases'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]
    
    all_ok = all(check_import(mod, name) for mod, name in dependencies)
    
    print("\n2. Checking CUDA...")
    print("-" * 60)
    check_cuda()
    
    print("\n3. Checking Directory Structure...")
    print("-" * 60)
    dirs_ok = check_directories()
    
    print("\n" + "=" * 60)
    if all_ok and dirs_ok:
        print("✓ Setup complete! Ready to proceed to Day 2.")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
