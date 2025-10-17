# EnCodec Finetuning Guide

This guide will help you finetune an EnCodec model on your own music dataset using AudioCraft.

## Overview

EnCodec is a neural audio codec that compresses audio into discrete tokens. Finetuning EnCodec on your music data can improve reconstruction quality for your specific domain (e.g., specific genres, instruments, or recording styles).

## Prerequisites

1. AudioCraft installed and working
2. A dataset of music audio files (recommended: at least 10-100 hours for finetuning, 1000+ hours for training from scratch)
3. GPU(s) for training (recommended: 1-8 GPUs with at least 16GB VRAM each)

## Quick Start

### Step 1: Prepare Your Dataset

First, organize your music files in a directory. The structure can be flat or nested:

```
/path/to/music/
├── song1.mp3
├── song2.wav
├── genre1/
│   ├── track1.flac
│   └── track2.mp3
└── genre2/
    └── track3.wav
```

### Step 2: Create Manifest Files

Use the provided script to create manifest files:

```bash
# Basic usage - creates train and valid manifests
python scripts/prepare_music_dataset.py /path/to/music egs/music_finetune --split 0.9 --recursive

# This will create:
# - egs/music_finetune/train/data.jsonl (90% of your data)
# - egs/music_finetune/valid/data.jsonl (10% of your data)
```

Options:
- `--split 0.9`: Use 90% for training, 10% for validation
- `--recursive`: Scan subdirectories recursively
- `--extensions .mp3 .wav .flac`: Specify which audio formats to include
- `--relative-paths`: Use relative paths in manifest (useful if moving data)

### Step 3: Configure the Dataset

The dataset configuration is already created at `config/dset/audio/music_finetune.yaml`. Verify it points to your manifest files:

```yaml
datasource:
  max_sample_rate: 32000
  max_channels: 1
  train: egs/music_finetune/train
  valid: egs/music_finetune/valid
  evaluate: egs/music_finetune/valid
  generate: egs/music_finetune/valid
```

**Important Configuration Notes:**
- `max_sample_rate`: Set to 32000 for 32kHz EnCodec (default for MusicGen)
  - Can also use 24000 for 24kHz models or 48000 for higher quality
- `max_channels`: Set to 1 for mono, 2 for stereo
  - Note: The default MusicGen EnCodec is mono (1 channel)

### Step 4: Launch Training

#### Option A: Finetune from Pretrained Model (Recommended)

This initializes the model with pretrained weights and continues training:

```bash
# Single GPU training
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz

# Multi-GPU training (e.g., 4 GPUs)
dora run -d solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz

# Or use the grid (edit the grid file first to uncomment the option you want)
dora grid compression.encodec_finetune_music
```

#### Option B: Train from Scratch

Only recommended if you have a very large dataset (100k+ hours):

```bash
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune
```

### Step 5: Monitor Training

Check training progress:

```bash
# Get the experiment signature from the dora output (e.g., a1b2c3d4)
dora info -f <SIGNATURE>

# Tail the logs
dora info -f <SIGNATURE> -t

# Check the tensorboard logs
tensorboard --logdir /tmp/audiocraft_<user>/outputs/<SIGNATURE>
```

### Step 6: Evaluate and Export

Once training is complete, you can:

1. **Load the model in Python:**
```python
from audiocraft.solvers import CompressionSolver

# Load from experiment signature
model = CompressionSolver.model_from_checkpoint('//sig/<SIGNATURE>')

# Or from checkpoint path
model = CompressionSolver.model_from_checkpoint('/path/to/checkpoint.th')

# Test the model
import torchaudio
wav, sr = torchaudio.load('test.mp3')
wav = wav.unsqueeze(0)  # Add batch dimension: [B, C, T]

# Encode and decode
with torch.no_grad():
    encoded = model.encode(wav)
    decoded = model.decode(encoded[0])
```

2. **Export for use with MusicGen:**
```python
from audiocraft.utils import export
from audiocraft import train

xp = train.main.get_xp_from_sig('<SIGNATURE>')
export.export_encodec(
    xp.folder / 'checkpoint.th',
    '/path/to/output/compression_state_dict.bin'
)
```

## Configuration Options

### Model Architecture

The default configuration uses `encodec_large_nq4_s640`:
- Stride: 640 (frame rate: 50 Hz at 32kHz)
- Codebooks: 4
- Codebook size: 2048

To use a different architecture, edit the training grid or command:

```bash
# Use smaller model (faster training, less quality)
dora run solver=compression/encodec_base_24khz dset=audio/music_finetune \
    sample_rate=24000 continue_from=//pretrained/facebook/encodec_24khz
```

### Hyperparameters

Key hyperparameters in `config/solver/compression/default.yaml`:

```yaml
# Training
optim:
  epochs: 200              # Total epochs
  updates_per_epoch: 2000  # Steps per epoch
  lr: 3e-4                 # Learning rate (use 1e-4 for finetuning)
  
# Dataset
dataset:
  batch_size: 64           # Batch size (reduce for limited GPU memory)
  segment_duration: 1      # Audio segment length in seconds
  
# Losses
losses:
  adv: 4.0    # Adversarial loss weight
  feat: 4.0   # Feature matching loss weight
  l1: 0.1     # L1 reconstruction loss weight
  msspec: 2.0 # Multi-scale spectrogram loss weight
```

**Recommended settings for finetuning:**
```bash
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 \
    dataset.batch_size=32 \
    optim.epochs=100
```

## Advanced Options

### Using Different Sample Rates

For 24kHz audio:
```bash
dora run solver=compression/encodec_base_24khz dset=audio/music_finetune \
    sample_rate=24000 channels=1 \
    continue_from=//pretrained/facebook/encodec_24khz
```

### Stereo Training

For stereo audio, you need to use the stereo model configuration:
```bash
dora run solver=compression/encodec_base_24khz dset=audio/music_finetune \
    sample_rate=24000 channels=2
```

### Adjusting for Limited GPU Memory

If you run out of GPU memory:
```bash
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    dataset.batch_size=16 \
    dataset.segment_duration=0.5 \
    continue_from=//pretrained/facebook/encodec_32khz
```

### Custom Loss Weights

To emphasize certain losses:
```bash
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    losses.msspec=4.0 \
    losses.l1=0.5 \
    continue_from=//pretrained/facebook/encodec_32khz
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `dataset.batch_size` (e.g., from 64 to 32 or 16)
- Reduce `dataset.segment_duration` (e.g., from 1 to 0.5)
- Use gradient accumulation: `optim.accumulate_grad_batches=2`

### Poor Reconstruction Quality
- Train for more epochs
- Increase `losses.msspec` for better spectral reconstruction
- Increase `losses.l1` for better time-domain reconstruction
- Ensure your dataset has sufficient diversity and quality

### Training is Too Slow
- Use multiple GPUs: `dora run -d ...`
- Increase `dataset.batch_size` if you have GPU memory
- Reduce `optim.updates_per_epoch` for faster epochs

### Model Not Learning
- Check that `continue_from` is loading correctly (check logs)
- Ensure learning rate isn't too low (use 1e-4 to 3e-4 for finetuning)
- Verify your data is being loaded correctly (check first few batches)

## Dataset Requirements

### Minimum Dataset Size
- **Finetuning:** 10-100 hours (recommended 50+ hours)
- **Training from scratch:** 1000+ hours (recommended 10k+ hours)

### Audio Quality
- Sample rate: 24kHz-48kHz (will be resampled to match model)
- Bit depth: 16-bit or 24-bit
- Format: MP3, WAV, FLAC, M4A, OGG
- Avoid: Heavily compressed or low-bitrate audio

### Data Diversity
For best results, include:
- Various instruments/sounds
- Different recording qualities
- Different genres/styles (if applicable)
- Various audio characteristics

## Using the Finetuned Model

### With MusicGen

After exporting your model, you can use it as a custom compression model with MusicGen:

```python
from audiocraft.models import MusicGen
from audiocraft.models import CompressionModel

# Load your custom compression model
compression_model = CompressionModel.get_pretrained('/path/to/compression_state_dict.bin')

# Load MusicGen and replace its compression model
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.compression_model = compression_model

# Generate music with your custom codec
model.set_generation_params(duration=8)
wav = model.generate(['happy rock music'])
```

### Standalone Usage

```python
from audiocraft.solvers import CompressionSolver
import torchaudio

# Load model
model = CompressionSolver.model_from_checkpoint('//sig/<SIGNATURE>')
model.eval()

# Load audio
wav, sr = torchaudio.load('input.mp3')
if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

# Ensure correct shape: [batch, channels, time]
if wav.dim() == 2:
    wav = wav.unsqueeze(0)

# Encode and decode
import torch
with torch.no_grad():
    codes, scale = model.encode(wav)
    reconstructed = model.decode(codes, scale)

# Save reconstructed audio
torchaudio.save('output.wav', reconstructed[0], model.sample_rate)
```

## Best Practices

1. **Start with finetuning:** Always finetune from a pretrained model unless you have a huge dataset
2. **Use validation data:** Always split your data to monitor overfitting
3. **Monitor metrics:** Watch SI-SNR and other reconstruction metrics during training
4. **Save checkpoints:** Keep multiple checkpoints to compare quality
5. **Test thoroughly:** Evaluate on held-out data before deploying
6. **Document changes:** Keep track of hyperparameters and dataset versions

## References

- [EnCodec Paper](https://arxiv.org/abs/2210.13438)
- [AudioCraft Documentation](./docs/TRAINING.md)
- [Dora Documentation](https://github.com/facebookresearch/dora)

## Support

For issues and questions:
1. Check the AudioCraft documentation
2. Review the example configurations in `config/`
3. Check existing issues on GitHub
4. Open a new issue with detailed logs and configuration
