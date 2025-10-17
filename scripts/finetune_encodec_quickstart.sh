#!/bin/bash
# Quick start script for EnCodec finetuning
# This script helps you get started with finetuning an EnCodec model

set -e  # Exit on error

echo "======================================"
echo "EnCodec Finetuning Quick Start Script"
echo "======================================"
echo ""

# Check if music directory is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_music_directory> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --split <ratio>     Train/valid split ratio (default: 0.9)"
    echo "  --gpus <n>          Number of GPUs to use (default: 1)"
    echo "  --batch-size <n>    Batch size (default: 32)"
    echo "  --epochs <n>        Number of epochs (default: 100)"
    echo "  --lr <value>        Learning rate (default: 1e-4)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/music --split 0.9 --gpus 2 --batch-size 64"
    exit 1
fi

MUSIC_DIR="$1"
shift

# Default parameters
SPLIT_RATIO=0.9
GPUS=1
BATCH_SIZE=32
EPOCHS=100
LR=1e-4

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT_RATIO="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if music directory exists
if [ ! -d "$MUSIC_DIR" ]; then
    echo "Error: Music directory not found: $MUSIC_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Music directory: $MUSIC_DIR"
echo "  Train/valid split: $SPLIT_RATIO"
echo "  GPUs: $GPUS"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

# Step 1: Prepare dataset
echo "Step 1: Preparing dataset..."
python3 scripts/prepare_music_dataset.py "$MUSIC_DIR" egs/music_finetune \
    --split $SPLIT_RATIO \
    --recursive

if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed"
    exit 1
fi

echo ""
echo "Dataset prepared successfully!"
echo ""

# Step 2: Launch training
echo "Step 2: Launching training..."
echo ""

if [ $GPUS -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    dora run solver=compression/encodec_musicgen_32khz \
        dset=audio/music_finetune \
        continue_from=//pretrained/facebook/encodec_32khz \
        dataset.batch_size=$BATCH_SIZE \
        optim.lr=$LR \
        optim.epochs=$EPOCHS
else
    # Multi-GPU training
    echo "Starting multi-GPU training with $GPUS GPUs..."
    dora run -d solver=compression/encodec_musicgen_32khz \
        dset=audio/music_finetune \
        continue_from=//pretrained/facebook/encodec_32khz \
        dataset.batch_size=$BATCH_SIZE \
        optim.lr=$LR \
        optim.epochs=$EPOCHS
fi

echo ""
echo "======================================"
echo "Training launched successfully!"
echo "======================================"
echo ""
echo "To monitor progress:"
echo "  1. Find your experiment signature in the output above (e.g., a1b2c3d4)"
echo "  2. Run: dora info -f <SIGNATURE> -t"
echo ""
echo "To view tensorboard:"
echo "  tensorboard --logdir /tmp/audiocraft_\$USER/outputs/<SIGNATURE>"
echo ""
