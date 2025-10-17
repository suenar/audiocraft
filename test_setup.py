#!/usr/bin/env python3
"""
Test AudioCraft Setup

This script verifies that your environment is properly configured for MusicGen finetuning.
"""

import sys

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("✓ PyTorch installed")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision torchaudio")
        return False

def check_audiocraft():
    """Check AudioCraft installation."""
    try:
        import audiocraft
        print(f"AudioCraft installed")
        print("✓ AudioCraft OK")
        return True
    except ImportError:
        print("❌ AudioCraft not installed")
        print("   Install with: pip install -e .")
        return False

def check_dependencies():
    """Check other dependencies."""
    missing = []
    deps = [
        'transformers',
        'encodec',
        'hydra',
        'omegaconf',
        'flashy',
        'torchaudio',
        'einops',
        'librosa',
    ]
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ {dep} missing")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_dora():
    """Check Dora installation."""
    try:
        import dora
        print("✓ Dora installed")
        return True
    except ImportError:
        print("⚠ Dora not installed (optional but recommended)")
        print("   Install with: pip install -U git+https://github.com/facebookresearch/dora")
        return True  # Not critical

def check_config():
    """Check configuration files."""
    from pathlib import Path
    
    files = [
        'config/solver/musicgen/finetune_32khz.yaml',
        'config/dset/audio/custom_music.yaml',
        'scripts/finetune_musicgen.py',
        'scripts/prepare_music_dataset.py',
    ]
    
    for file in files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} missing")
            return False
    return True

def main():
    print("=" * 80)
    print("AudioCraft MusicGen Finetuning - Setup Check")
    print("=" * 80)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("AudioCraft", check_audiocraft),
        ("Dependencies", check_dependencies),
        ("Dora", check_dora),
        ("Configuration Files", check_config),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())
    
    print("\n" + "=" * 80)
    if all(results[:-1]):  # Exclude Dora from critical checks
        print("✓ Setup complete! You're ready to finetune MusicGen.")
        print("\nNext steps:")
        print("1. Prepare your dataset: python scripts/prepare_music_dataset.py /path/to/music")
        print("2. Start finetuning: python scripts/finetune_musicgen.py --model medium")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print("❌ Setup incomplete. Please install missing dependencies.")
        print("\nQuick fix:")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        return 1

if __name__ == '__main__':
    sys.exit(main())
