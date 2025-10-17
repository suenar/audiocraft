# 🎵 EnCodec Finetuning for Music Generation

A complete setup for finetuning EnCodec models on custom music datasets in AudioCraft.

## 📁 What's Included

All necessary files have been created for you:

### 🔧 Ready-to-Use Scripts
- **`scripts/prepare_music_dataset.py`** - Automated dataset preparation
- **`scripts/finetune_encodec_quickstart.sh`** - One-command training launcher
- **`examples/encodec_finetune_example.py`** - Usage examples and test code

### ⚙️ Configuration Files
- **`config/dset/audio/music_finetune.yaml`** - Dataset configuration
- **`audiocraft/grids/compression/encodec_finetune_music.py`** - Training grid

### 📚 Documentation
- **`ENCODEC_FINETUNING.md`** - Complete detailed guide (11KB)
- **`ENCODEC_FINETUNE_SUMMARY.md`** - Quick reference (6KB)
- **`SETUP_COMPLETE.md`** - Setup verification and workflow guide (9KB)

## 🚀 Quick Start (3 Minutes)

### Option 1: Automated Setup (Easiest)

```bash
# One command does everything!
./scripts/finetune_encodec_quickstart.sh /path/to/your/music
```

### Option 2: Manual Setup (More Control)

```bash
# Step 1: Prepare your dataset (creates train/valid splits)
python3 scripts/prepare_music_dataset.py \
    /path/to/music \
    egs/music_finetune \
    --split 0.9 \
    --recursive

# Step 2: Start finetuning
dora run solver=compression/encodec_musicgen_32khz \
    dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 \
    dataset.batch_size=32
```

## 📖 Documentation

Choose the right guide for your needs:

| Document | Best For | Size |
|----------|----------|------|
| **Quick Start** (this file) | Getting started immediately | 2 min read |
| **ENCODEC_FINETUNE_SUMMARY.md** | Quick reference and common scenarios | 5 min read |
| **ENCODEC_FINETUNING.md** | Complete guide with all details | 15 min read |
| **SETUP_COMPLETE.md** | Detailed workflow and troubleshooting | 10 min read |

## 🎯 What Can You Do?

### 1. Finetune for Specific Genres
```bash
# Jazz music
python3 scripts/prepare_music_dataset.py /data/jazz egs/jazz_finetune --split 0.9
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz
```

### 2. Improve Quality for Your Instruments
```bash
# Piano recordings
python3 scripts/prepare_music_dataset.py /data/piano egs/piano_finetune --split 0.9
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    losses.msspec=4.0  # Emphasize spectral quality
```

### 3. Adapt to Recording Style
```bash
# Lo-fi or specific recording quality
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.epochs=150
```

## 💡 Common Scenarios

### Small Dataset (<50 hours)
```bash
./scripts/finetune_encodec_quickstart.sh /path/to/music \
    --batch-size 16 \
    --epochs 50 \
    --lr 5e-5
```

### Large Dataset (>500 hours)
```bash
./scripts/finetune_encodec_quickstart.sh /path/to/music \
    --gpus 4 \
    --batch-size 64 \
    --epochs 200
```

### Limited GPU Memory
```bash
./scripts/finetune_encodec_quickstart.sh /path/to/music \
    --batch-size 8
```

## 📊 Monitor Training

```bash
# Get your experiment signature from the training output (e.g., a1b2c3d4)
# Then monitor with:

dora info -f <SIGNATURE> -t            # Watch logs in real-time
tensorboard --logdir /tmp/audiocraft_$USER/outputs/<SIGNATURE>  # View metrics
```

## 🔍 Using Your Finetuned Model

### Standalone Usage
```python
from audiocraft.solvers import CompressionSolver
import torchaudio

# Load your finetuned model
model = CompressionSolver.model_from_checkpoint('//sig/<SIGNATURE>')

# Load audio
wav, sr = torchaudio.load('input.mp3')
wav = wav.unsqueeze(0)

# Encode and decode
codes, scale = model.encode(wav)
reconstructed = model.decode(codes, scale)

# Save
torchaudio.save('output.wav', reconstructed[0], model.sample_rate)
```

### With MusicGen
```python
from audiocraft.models import MusicGen, CompressionModel

# Load your custom codec
compression_model = CompressionModel.get_pretrained('/path/to/model.bin')

# Use with MusicGen
musicgen = MusicGen.get_pretrained('facebook/musicgen-small')
musicgen.compression_model = compression_model

# Generate music with your custom codec
wav = musicgen.generate(['happy rock music'])
```

## ⚡ Key Features

- ✅ **Easy Setup**: One-command training with sensible defaults
- ✅ **Flexible**: Extensive configuration options for advanced users
- ✅ **Well-Documented**: Comprehensive guides for all skill levels
- ✅ **Production-Ready**: Based on Meta's AudioCraft framework
- ✅ **GPU Efficient**: Works on single or multi-GPU setups
- ✅ **Pretrained Start**: Finetune from Meta's pretrained models

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- AudioCraft installed
- GPU with 16GB+ VRAM (recommended)
- Music dataset (10+ hours recommended for finetuning)

## 🛠️ File Structure

```
/workspace/
├── config/dset/audio/
│   └── music_finetune.yaml           # Dataset configuration
├── audiocraft/grids/compression/
│   └── encodec_finetune_music.py     # Training grid
├── scripts/
│   ├── prepare_music_dataset.py      # Data preparation
│   └── finetune_encodec_quickstart.sh # Quick start launcher
├── examples/
│   └── encodec_finetune_example.py   # Usage examples
├── egs/music_finetune/                # Your data manifests (created by scripts)
│   ├── train/data.jsonl
│   └── valid/data.jsonl
└── Documentation:
    ├── ENCODEC_FINETUNING.md          # Complete guide
    ├── ENCODEC_FINETUNE_SUMMARY.md    # Quick reference
    └── SETUP_COMPLETE.md              # Setup details
```

## 🎓 Learning Path

1. **Beginner**: Start with this README and the quick start script
2. **Intermediate**: Read `ENCODEC_FINETUNE_SUMMARY.md` for common scenarios
3. **Advanced**: Study `ENCODEC_FINETUNING.md` for deep customization
4. **Expert**: Explore `SETUP_COMPLETE.md` and modify training grids

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` to 8 or 16 |
| Training too slow | Use `--gpus 2` or more |
| Poor quality | Train longer with `--epochs 200` |
| Can't find music | Check path and use `--recursive` |

See `ENCODEC_FINETUNING.md` for detailed troubleshooting.

## 📞 Getting Help

1. Check the documentation files (especially `ENCODEC_FINETUNING.md`)
2. Run example script: `python3 examples/encodec_finetune_example.py`
3. Review AudioCraft docs in `docs/`
4. Check the [AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)

## 🎯 Next Steps

1. **Read Quick Reference**: `ENCODEC_FINETUNE_SUMMARY.md` (5 min)
2. **Prepare Data**: Run `python3 scripts/prepare_music_dataset.py --help`
3. **Start Training**: Use the quick start script
4. **Monitor Progress**: Watch logs and metrics
5. **Test Model**: Load and evaluate your finetuned model

## 📚 Additional Resources

- [EnCodec Paper](https://arxiv.org/abs/2210.13438) - Original research
- [AudioCraft Docs](docs/TRAINING.md) - Training guide
- [Dora](https://github.com/facebookresearch/dora) - Experiment manager
- [Example Models](https://huggingface.co/facebook) - Pretrained models

---

## 🎉 Ready to Start?

```bash
# Quick start with your music:
./scripts/finetune_encodec_quickstart.sh /path/to/your/music

# Or step by step:
# 1. Prepare data
python3 scripts/prepare_music_dataset.py /path/to/music egs/music_finetune --split 0.9 --recursive

# 2. Start training
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz

# 3. Monitor
dora info -f <SIGNATURE> -t
```

**Good luck with your finetuning!** 🚀

---

*Created for the `cursor/finetune-encodec-for-music-generation-fe78` branch*
