# EnCodec Finetuning Setup Complete ✓

All necessary files have been created for finetuning an EnCodec model for music generation in AudioCraft.

## What Was Created

### 📋 Configuration Files
- ✅ `config/dset/audio/music_finetune.yaml` - Dataset configuration

### 🔧 Scripts and Tools
- ✅ `scripts/prepare_music_dataset.py` - Dataset preparation script
- ✅ `scripts/finetune_encodec_quickstart.sh` - One-command training launcher
- ✅ `examples/encodec_finetune_example.py` - Example usage code

### 🎯 Training Configuration
- ✅ `audiocraft/grids/compression/encodec_finetune_music.py` - Dora training grid

### 📚 Documentation
- ✅ `ENCODEC_FINETUNING.md` - Complete detailed guide (10KB)
- ✅ `ENCODEC_FINETUNE_SUMMARY.md` - Quick reference guide (6KB)

## Quick Start Guide

### Method 1: Quick Start Script (Easiest)

```bash
# One command to prepare data and start training
./scripts/finetune_encodec_quickstart.sh /path/to/your/music

# With options
./scripts/finetune_encodec_quickstart.sh /path/to/music \
    --split 0.9 \
    --gpus 2 \
    --batch-size 64 \
    --epochs 100
```

### Method 2: Manual Steps

```bash
# Step 1: Prepare dataset
python3 scripts/prepare_music_dataset.py \
    /path/to/music \
    egs/music_finetune \
    --split 0.9 \
    --recursive

# Step 2: Start training
dora run solver=compression/encodec_musicgen_32khz \
    dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 \
    dataset.batch_size=32
```

### Method 3: Using the Training Grid

```bash
# Edit the grid file if needed, then run:
dora grid compression.encodec_finetune_music
```

## File Descriptions

### `scripts/prepare_music_dataset.py`
Scans a directory for audio files and creates manifest files (.jsonl) for AudioCraft training.

**Features:**
- Automatic train/valid split
- Recursive directory scanning
- Multiple audio format support (MP3, WAV, FLAC, M4A, OGG)
- Audio metadata extraction (duration, sample rate)

**Usage:**
```bash
python3 scripts/prepare_music_dataset.py [INPUT_DIR] [OUTPUT_DIR] [OPTIONS]

Options:
  --split RATIO          Train/valid split (e.g., 0.9 = 90% train, 10% valid)
  --recursive            Scan subdirectories recursively
  --extensions EXT...    File extensions to include (default: .mp3 .wav .flac)
  --relative-paths       Use relative paths in manifest
  --seed SEED           Random seed for splitting
```

### `scripts/finetune_encodec_quickstart.sh`
One-command script that prepares data and launches training.

**Features:**
- Automated dataset preparation
- Configurable training parameters
- Single or multi-GPU support
- Helpful output and progress messages

**Usage:**
```bash
./scripts/finetune_encodec_quickstart.sh [MUSIC_DIR] [OPTIONS]

Options:
  --split RATIO          Train/valid split (default: 0.9)
  --gpus N              Number of GPUs (default: 1)
  --batch-size N        Batch size (default: 32)
  --epochs N            Training epochs (default: 100)
  --lr VALUE            Learning rate (default: 1e-4)
```

### `examples/encodec_finetune_example.py`
Example code demonstrating how to:
- Load pretrained models
- Encode and decode audio
- Evaluate reconstruction quality
- Use finetuned models

**Usage:**
```bash
python3 examples/encodec_finetune_example.py
```

### `audiocraft/grids/compression/encodec_finetune_music.py`
Dora grid configuration for training experiments.

**Features:**
- Multiple training configurations
- Finetune from pretrained vs. train from scratch
- Different hyperparameter combinations
- Easy to customize for your needs

**Usage:**
```bash
dora grid compression.encodec_finetune_music
```

### `config/dset/audio/music_finetune.yaml`
Dataset configuration file.

**Configuration:**
```yaml
datasource:
  max_sample_rate: 32000  # Audio sample rate
  max_channels: 1         # Mono (1) or stereo (2)
  train: egs/music_finetune/train       # Training data manifest
  valid: egs/music_finetune/valid       # Validation data manifest
  evaluate: egs/music_finetune/valid    # Evaluation data manifest
  generate: egs/music_finetune/valid    # Generation data manifest
```

## Training Workflow

```
1. Prepare Dataset
   ↓
   python3 scripts/prepare_music_dataset.py /path/to/music egs/music_finetune --split 0.9
   Creates: egs/music_finetune/train/data.jsonl
           egs/music_finetune/valid/data.jsonl

2. Configure Training (optional)
   ↓
   Edit config/dset/audio/music_finetune.yaml if needed

3. Launch Training
   ↓
   dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
       continue_from=//pretrained/facebook/encodec_32khz

4. Monitor Progress
   ↓
   dora info -f <SIGNATURE> -t

5. Use Finetuned Model
   ↓
   from audiocraft.solvers import CompressionSolver
   model = CompressionSolver.model_from_checkpoint('//sig/<SIGNATURE>')
```

## Common Training Scenarios

### Scenario 1: Small Dataset (<50 hours)
```bash
# Use smaller batch size, lower learning rate, and finetune from pretrained
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    dataset.batch_size=16 \
    optim.lr=5e-5 \
    optim.epochs=50
```

### Scenario 2: Large Dataset (>500 hours)
```bash
# Use larger batch size and train longer
dora run -d solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    dataset.batch_size=64 \
    optim.epochs=200
```

### Scenario 3: Limited GPU Memory
```bash
# Reduce batch size and segment duration
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    dataset.batch_size=8 \
    dataset.segment_duration=0.5
```

### Scenario 4: Specific Genre (e.g., Classical)
```bash
# Prepare genre-specific data and finetune with emphasis on spectral loss
python3 scripts/prepare_music_dataset.py /path/to/classical egs/classical_finetune --split 0.9

dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    losses.msspec=4.0 \
    optim.epochs=150
```

## Key Configuration Parameters

| Parameter | Description | Default | Recommended for Finetuning |
|-----------|-------------|---------|---------------------------|
| `sample_rate` | Audio sample rate | 32000 | 32000 (music) |
| `channels` | Mono (1) or stereo (2) | 1 | 1 (matches MusicGen) |
| `dataset.batch_size` | Samples per batch | 64 | 32 (for finetuning) |
| `dataset.segment_duration` | Audio segment length (sec) | 1 | 1 |
| `optim.lr` | Learning rate | 3e-4 | 1e-4 (for finetuning) |
| `optim.epochs` | Training epochs | 200 | 100 (for finetuning) |
| `losses.adv` | Adversarial loss weight | 4.0 | 4.0 |
| `losses.msspec` | Spectral loss weight | 2.0 | 2.0 |

## Next Steps

1. **Read the Documentation**
   - `ENCODEC_FINETUNING.md` - Complete guide with all details
   - `ENCODEC_FINETUNE_SUMMARY.md` - Quick reference

2. **Prepare Your Dataset**
   - Organize music files in a directory
   - Run `scripts/prepare_music_dataset.py`

3. **Start Training**
   - Use quick start script for easy setup
   - Or use dora commands for more control

4. **Monitor Training**
   - Use `dora info -f <SIG> -t` to watch logs
   - Check tensorboard for metrics

5. **Evaluate Results**
   - Load the model and test reconstruction quality
   - Compare with pretrained model

6. **Use the Model**
   - Export for use with MusicGen
   - Or use standalone for audio compression

## Helpful Commands

```bash
# Get help for data preparation
python3 scripts/prepare_music_dataset.py --help

# Get help for quick start script
./scripts/finetune_encodec_quickstart.sh

# List all dora experiments
dora info

# Get specific experiment info
dora info -f <SIGNATURE>

# Cancel and restart experiment
dora run -f <SIGNATURE> --clear

# Monitor training logs
dora info -f <SIGNATURE> -t

# Launch tensorboard
tensorboard --logdir /tmp/audiocraft_$USER/outputs/<SIGNATURE>
```

## Documentation Reference

- **Full Guide**: `ENCODEC_FINETUNING.md` (10KB, comprehensive)
- **Quick Reference**: `ENCODEC_FINETUNE_SUMMARY.md` (6KB, essential info)
- **AudioCraft Training**: `docs/TRAINING.md`
- **AudioCraft Datasets**: `docs/DATASETS.md`
- **EnCodec Overview**: `docs/ENCODEC.md`

## Example Workflow

```bash
# 1. Prepare data
python3 scripts/prepare_music_dataset.py \
    /path/to/music \
    egs/music_finetune \
    --split 0.9 \
    --recursive

# 2. Start training
dora run solver=compression/encodec_musicgen_32khz \
    dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 \
    dataset.batch_size=32 \
    optim.epochs=100

# 3. Monitor (get signature from step 2 output)
dora info -f <SIGNATURE> -t

# 4. Load and test model
python3 -c "
from audiocraft.solvers import CompressionSolver
model = CompressionSolver.model_from_checkpoint('//sig/<SIGNATURE>')
print('Model loaded successfully!')
print(f'Sample rate: {model.sample_rate}')
print(f'Codebooks: {model.num_codebooks}')
"
```

## Support and Resources

- **EnCodec Paper**: https://arxiv.org/abs/2210.13438
- **AudioCraft GitHub**: https://github.com/facebookresearch/audiocraft
- **Dora**: https://github.com/facebookresearch/dora
- **Flashy**: https://github.com/facebookresearch/flashy

---

**Everything is ready!** Start with:
```bash
./scripts/finetune_encodec_quickstart.sh /path/to/your/music
```

For detailed information, see `ENCODEC_FINETUNING.md`.
