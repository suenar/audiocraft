# EnCodec Finetuning - Quick Reference

This is a quick reference for finetuning an EnCodec model for music generation in AudioCraft.

## Files Created

### Configuration
- **`config/dset/audio/music_finetune.yaml`**: Dataset configuration for your music data

### Scripts
- **`scripts/prepare_music_dataset.py`**: Automated dataset preparation script
- **`scripts/finetune_encodec_quickstart.sh`**: One-command finetuning script
- **`examples/encodec_finetune_example.py`**: Example code for loading and using models

### Training Grid
- **`audiocraft/grids/compression/encodec_finetune_music.py`**: Dora grid for training experiments

### Documentation
- **`ENCODEC_FINETUNING.md`**: Complete finetuning guide with all details

## Quick Start (3 Steps)

### 1. Prepare Your Data
```bash
# Organize your music files in a directory
# Then run:
python scripts/prepare_music_dataset.py /path/to/music egs/music_finetune --split 0.9 --recursive
```

### 2. Launch Training
```bash
# Quick start with one command:
./scripts/finetune_encodec_quickstart.sh /path/to/music

# Or manually with dora:
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 dataset.batch_size=32
```

### 3. Monitor and Use
```bash
# Monitor training (replace <SIG> with your experiment signature)
dora info -f <SIG> -t

# Load the finetuned model in Python
python -c "
from audiocraft.solvers import CompressionSolver
model = CompressionSolver.model_from_checkpoint('//sig/<SIG>')
"
```

## Common Use Cases

### Finetuning for Specific Genre
```bash
# Prepare genre-specific dataset
python scripts/prepare_music_dataset.py /path/to/jazz_music egs/jazz_finetune --split 0.9

# Finetune with genre data
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    optim.lr=1e-4 optim.epochs=100
```

### Limited GPU Memory
```bash
# Use smaller batch size and segments
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz \
    dataset.batch_size=16 dataset.segment_duration=0.5
```

### Multi-GPU Training
```bash
# Use -d flag for distributed training
dora run -d solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    continue_from=//pretrained/facebook/encodec_32khz
```

### Higher Quality (24kHz → 32kHz)
```bash
# Finetune a 32kHz model for better quality
dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \
    sample_rate=32000 channels=1 \
    continue_from=//pretrained/facebook/encodec_32khz
```

## Key Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 32000 | Audio sample rate (24000, 32000, or 48000) |
| `channels` | 1 | Number of audio channels (1=mono, 2=stereo) |
| `dataset.batch_size` | 64 | Batch size per GPU |
| `dataset.segment_duration` | 1 | Audio segment length in seconds |
| `optim.lr` | 3e-4 | Learning rate (use 1e-4 for finetuning) |
| `optim.epochs` | 200 | Total training epochs |
| `losses.adv` | 4.0 | Adversarial loss weight |
| `losses.msspec` | 2.0 | Multi-scale spectrogram loss weight |

## Common Commands

```bash
# Get experiment info
dora info -f <SIG>

# Tail training logs
dora info -f <SIG> -t

# List all experiments
dora grid compression.encodec_finetune_music --dry_run --init

# Cancel and restart training
dora run -f <SIG> --clear
```

## Using the Finetuned Model

### Standalone Usage
```python
from audiocraft.solvers import CompressionSolver
import torchaudio

# Load model
model = CompressionSolver.model_from_checkpoint('//sig/<SIG>')

# Load and process audio
wav, sr = torchaudio.load('input.mp3')
wav = wav.unsqueeze(0)  # Add batch dimension

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
compression_model = CompressionModel.get_pretrained('/path/to/exported_model.bin')

# Load MusicGen and replace codec
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.compression_model = compression_model

# Generate music
wav = model.generate(['happy rock music'])
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` to 16 or `segment_duration` to 0.5 |
| Training too slow | Use multi-GPU with `-d` flag |
| Poor quality | Train longer or increase spectral loss weight |
| Model not learning | Check learning rate (1e-4 to 3e-4) and data loading |

## Project Structure

```
audiocraft/
├── config/
│   └── dset/
│       └── audio/
│           └── music_finetune.yaml          # Dataset config
├── audiocraft/
│   └── grids/
│       └── compression/
│           └── encodec_finetune_music.py    # Training grid
├── scripts/
│   ├── prepare_music_dataset.py             # Data preparation
│   └── finetune_encodec_quickstart.sh       # Quick start
├── examples/
│   └── encodec_finetune_example.py          # Usage examples
├── egs/
│   └── music_finetune/                      # Your manifest files
│       ├── train/data.jsonl
│       └── valid/data.jsonl
└── ENCODEC_FINETUNING.md                    # Full documentation
```

## Resources

- **Full Documentation**: `ENCODEC_FINETUNING.md`
- **Example Code**: `examples/encodec_finetune_example.py`
- **Dataset Guide**: `docs/DATASETS.md`
- **Training Guide**: `docs/TRAINING.md`
- **EnCodec Paper**: https://arxiv.org/abs/2210.13438

## Getting Help

1. Read `ENCODEC_FINETUNING.md` for detailed instructions
2. Check example code in `examples/encodec_finetune_example.py`
3. Review configuration files in `config/`
4. Check AudioCraft documentation in `docs/`

---

**Ready to start?** Run:
```bash
./scripts/finetune_encodec_quickstart.sh /path/to/your/music
```
