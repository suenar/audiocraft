#!/bin/bash
# AudioCraft MusicGen Finetuning - Dependency Installation Script

set -e  # Exit on error

echo "=========================================="
echo "AudioCraft MusicGen - Dependency Installer"
echo "=========================================="
echo

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if Python version is >= 3.8
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python version OK"
echo

# Detect CUDA version
echo "Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "Found CUDA $CUDA_VERSION"
    
    # Determine PyTorch CUDA version
    if [[ $CUDA_VERSION == 11.* ]]; then
        TORCH_CUDA="cu118"
    elif [[ $CUDA_VERSION == 12.* ]]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu118"  # Default
    fi
else
    echo "Warning: CUDA not found. Installing CPU-only version."
    TORCH_CUDA="cpu"
fi
echo

# Install PyTorch
echo "Installing PyTorch with CUDA support ($TORCH_CUDA)..."
if [ "$TORCH_CUDA" == "cpu" ]; then
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CUDA
fi
echo "✓ PyTorch installed"
echo

# Install AudioCraft dependencies
echo "Installing AudioCraft dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo

# Install AudioCraft in development mode
echo "Installing AudioCraft..."
pip install -e .
echo "✓ AudioCraft installed"
echo

# Install Dora (experiment manager)
echo "Installing Dora..."
pip install -U git+https://github.com/facebookresearch/dora
echo "✓ Dora installed"
echo

# Verify installation
echo "Verifying installation..."
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import audiocraft; print('AudioCraft: OK')"
python3 -c "import transformers; print('Transformers: OK')"
python3 -c "import encodec; print('EnCodec: OK')"
echo

echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Set environment variables:"
echo "   export AUDIOCRAFT_TEAM=default"
echo "   export AUDIOCRAFT_DORA_DIR=./experiments"
echo
echo "2. Edit config/teams/default.yaml and set dora_dir to a permanent location"
echo
echo "3. Prepare your dataset:"
echo "   python3 scripts/prepare_music_dataset.py /path/to/music --output-dir egs/my_music"
echo
echo "4. Start finetuning:"
echo "   python3 scripts/finetune_musicgen.py --model medium --dataset custom_music"
echo
echo "See QUICKSTART.md and FINETUNING_GUIDE.md for more details."
