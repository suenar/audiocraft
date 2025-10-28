# MusicGen Finetuning Guide

This guide will help you finetune a MusicGen model for music generation on your custom dataset.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Dataset Preparation](#dataset-preparation)
6. [Configuration](#configuration)
7. [Training](#training)
8. [Evaluation and Generation](#evaluation-and-generation)
9. [Export and Use](#export-and-use)
10. [Troubleshooting](#troubleshooting)

## Overview

MusicGen is a single-stage autoregressive Transformer model for music generation. It uses:
- **EnCodec**: A neural audio codec that encodes audio into discrete tokens
- **Transformer LM**: A language model that learns to generate sequences of these tokens
- **Text Conditioning**: Optional text descriptions to guide generation

This guide focuses on finetuning a pretrained MusicGen model on your custom music dataset.

## Prerequisites

### Hardware Requirements
- **GPU**: At least 16GB VRAM recommended (e.g., NVIDIA V100, A100, RTX 3090, RTX 4090)
  - Small model: ~8GB VRAM
  - Medium model: ~16GB VRAM
  - Large model: ~32GB VRAM
- **Storage**: Enough space for your dataset and checkpoints
- **RAM**: At least 32GB system RAM

### Software Requirements
- Python 3.8+
- CUDA-capable GPU with drivers installed
- PyTorch with CUDA support

### Installation

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/facebookresearch/audiocraft
cd audiocraft
```

2. **Install dependencies**:
```bash
# Install PyTorch with CUDA support first (adjust based on your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AudioCraft dependencies
pip install -r requirements.txt

# Install AudioCraft in development mode
pip install -e .
```

3. **Verify installation**:
```bash
python -c "import audiocraft; print('AudioCraft version:', audiocraft.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Quick Start

Here's the fastest way to start finetuning:

### 1. Prepare Your Dataset

```bash
# Option A: Create manifest from a directory of audio files
python scripts/prepare_music_dataset.py /path/to/your/music --output-dir egs/my_music

# Option B: With metadata CSV file
python scripts/prepare_music_dataset.py /path/to/your/music \
    --metadata /path/to/metadata.csv \
    --output-dir egs/my_music

# Option C: With train/validation split (90% train, 10% validation)
python scripts/prepare_music_dataset.py /path/to/your/music \
    --split 0.9 \
    --output-dir egs/my_music
```

### 2. Update Dataset Configuration

Edit `config/dset/audio/custom_music.yaml` to point to your data:
```yaml
datasource:
  train: egs/my_music/train
  valid: egs/my_music/valid
  evaluate: egs/my_music/valid
  generate: egs/my_music/valid
```

### 3. Start Finetuning

```bash
# Finetune the medium model (recommended)
python scripts/finetune_musicgen.py --model medium --dataset custom_music

# Or use dora directly
dora run solver=musicgen/finetune_32khz \
    model/lm/model_scale=medium \
    dset=audio/custom_music \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=text2music
```

That's it! Training will start and you can monitor progress in the terminal.

## Detailed Setup

### Environment Configuration

AudioCraft uses environment variables for configuration. Set them in your shell:

```bash
# Set the team configuration (use 'default' for single-user setup)
export AUDIOCRAFT_TEAM=default

# Optional: Set custom output directory for experiments
export AUDIOCRAFT_DORA_DIR=/path/to/your/experiments

# Optional: Set reference directory for pretrained models
export AUDIOCRAFT_REFERENCE_DIR=/path/to/reference/models
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export AUDIOCRAFT_TEAM=default' >> ~/.bashrc
source ~/.bashrc
```

### Edit Team Configuration

Edit `config/teams/default.yaml` to set your paths:

```yaml
default:
  dora_dir: /tmp/audiocraft_outputs  # Change this to a permanent location!
  reference_dir: /tmp/audiocraft_reference
  partitions:
    global: []
    team: []
```

**Important**: Change `dora_dir` from `/tmp/` to a permanent location to avoid losing your trained models!

## Dataset Preparation

### Dataset Structure

Your dataset should consist of:
1. **Audio files**: MP3, WAV, FLAC, OGG, or M4A format
2. **Metadata files** (optional): JSON files with the same name as audio files

### Audio File Requirements

- **Sample rate**: Any (will be resampled to 32kHz)
- **Channels**: Mono or stereo
- **Duration**: At least 1 second (recommended: 10-30 seconds minimum)
- **Format**: MP3, WAV, FLAC, OGG, M4A
- **Quality**: Higher quality audio produces better results

### Metadata Format

For each audio file `song.mp3`, create a `song.json` file with metadata:

```json
{
  "description": "A happy upbeat electronic song with synthesizers",
  "genre": "electronic",
  "bpm": "128",
  "mood": "happy",
  "keywords": "upbeat, energetic, dance",
  "artist": "Artist Name",
  "title": "Song Title",
  "instrument": "synthesizer",
  "key": "C major"
}
```

**Important fields**:
- `description`: Text description used for conditioning (most important!)
- `genre`: Music genre
- `mood`: Emotional mood
- `keywords`: Additional descriptive keywords

### Creating Metadata from CSV

Create a CSV file with your metadata:

```csv
filename,description,genre,bpm,mood,artist,title
song1.mp3,"Happy upbeat electronic music with synthesizers",electronic,128,happy,Artist1,Song Title 1
song2.mp3,"Calm acoustic guitar melody",acoustic,90,relaxing,Artist2,Song Title 2
song3.mp3,"Energetic rock song with electric guitars",rock,140,energetic,Artist3,Song Title 3
```

Then use the preparation script:
```bash
python scripts/prepare_music_dataset.py /path/to/music \
    --metadata metadata.csv \
    --output-dir egs/my_music \
    --split 0.9
```

### Manual Dataset Preparation

If you prefer manual setup:

1. **Create manifest directory**:
```bash
mkdir -p egs/my_music/train egs/my_music/valid
```

2. **Create JSONL manifest**:
```bash
# For training data
python -m audiocraft.data.audio_dataset \
    /path/to/training/music \
    egs/my_music/train/data.jsonl

# For validation data
python -m audiocraft.data.audio_dataset \
    /path/to/validation/music \
    egs/my_music/valid/data.jsonl
```

3. **Create metadata JSON files** (optional but recommended):
Place `.json` files next to each audio file with the same name.

## Configuration

### Available Model Scales

- **small**: ~300M parameters (8GB VRAM)
- **medium**: ~1.5B parameters (16GB VRAM) - **Recommended**
- **large**: ~3.3B parameters (32GB VRAM)

### Key Configuration Parameters

Edit `config/solver/musicgen/finetune_32khz.yaml` to customize:

#### Dataset Settings
```yaml
dataset:
  batch_size: 8  # Reduce if out of memory
  segment_duration: 30  # Audio segment length in seconds
  num_workers: 10  # Data loading workers
```

#### Training Settings
```yaml
optim:
  epochs: 100  # Total training epochs
  updates_per_epoch: 100  # Gradient updates per epoch
  lr: 1e-4  # Learning rate (lower = safer for finetuning)
```

#### Generation Settings
```yaml
generate:
  every: 5  # Generate samples every N epochs
  lm:
    use_sampling: true
    top_k: 250
    temperature: 1.0
    cfg_coef: 3.0  # Classifier-free guidance
```

### Command-Line Overrides

You can override any configuration from the command line:

```bash
dora run solver=musicgen/finetune_32khz \
    dataset.batch_size=4 \
    optim.lr=5e-5 \
    optim.epochs=200 \
    dataset.segment_duration=20
```

## Training

### Basic Training

```bash
# Using the convenience script
python scripts/finetune_musicgen.py --model medium --dataset custom_music

# Using dora directly
dora run solver=musicgen/finetune_32khz \
    model/lm/model_scale=medium \
    dset=audio/custom_music \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=text2music
```

### Advanced Training Options

#### Multi-GPU Training
```bash
# Distributed training on all available GPUs
python scripts/finetune_musicgen.py --model medium --distributed

# Or with dora
dora run -d solver=musicgen/finetune_32khz \
    model/lm/model_scale=medium \
    dset=audio/custom_music \
    continue_from=//pretrained/facebook/musicgen-medium
```

#### Custom Batch Size and Learning Rate
```bash
python scripts/finetune_musicgen.py \
    --model medium \
    --batch-size 4 \
    --lr 5e-5 \
    --epochs 200
```

#### Stereo Training
```bash
python scripts/finetune_musicgen.py \
    --model medium \
    --stereo
```

#### Resume from Checkpoint
```bash
# Resume using experiment signature
dora run -f abc123def  # where abc123def is your experiment signature

# Or continue from a specific checkpoint
python scripts/finetune_musicgen.py \
    --continue-from //sig/abc123def
```

### Monitoring Training

#### View Logs
```bash
# Get experiment signature and view logs
dora info -f SIGNATURE -t  # tail logs

# Or find your experiment in the output directory
ls $AUDIOCRAFT_DORA_DIR/experiments/audiocraft/outputs/
```

#### TensorBoard
```bash
# Training logs are saved with TensorBoard format
tensorboard --logdir $AUDIOCRAFT_DORA_DIR/experiments/audiocraft/outputs/SIGNATURE

# Then open http://localhost:6006 in your browser
```

### Monitoring Progress

During training, you'll see:
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better
- **Generated samples**: Listen to them to assess quality

## Evaluation and Generation

### Automatic Evaluation

Evaluation happens automatically during training based on the configuration:

```yaml
evaluate:
  every: 5  # Evaluate every 5 epochs
```

Metrics computed:
- Cross-entropy loss
- Perplexity
- Optional: FAD, KLD, text consistency

### Manual Generation

```bash
# Generate samples from a trained model
python -c "
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load your finetuned model
model = MusicGen.get_pretrained('/path/to/checkpoint/folder')
model.set_generation_params(duration=10)

# Generate with text prompts
descriptions = [
    'A happy electronic dance track',
    'A calm acoustic guitar melody',
    'An energetic rock song with drums'
]
wav = model.generate(descriptions)

# Save generated audio
for idx, one_wav in enumerate(wav):
    audio_write(f'generated_{idx}', one_wav.cpu(), model.sample_rate, 
                strategy='loudness', loudness_compressor=True)
"
```

### Using the MOS Tool

Listen to and compare generated samples:

```bash
# Install Flask if not already installed
pip install Flask gunicorn

# Start the MOS (Mean Opinion Score) server
gunicorn -w 4 -b 127.0.0.1:8895 -t 120 'scripts.mos:app' --access-logfile -

# Open http://127.0.0.1:8895 in your browser
```

## Export and Use

### Export Model

After training, export your model for use:

```python
from audiocraft.utils import export
from audiocraft import train

# Get your experiment
xp = train.main.get_xp_from_sig('YOUR_SIGNATURE')

# Export the language model
export.export_lm(
    xp.folder / 'checkpoint.th',
    '/my/models/my_musicgen/state_dict.bin'
)

# Export or reference the compression model (EnCodec)
# If you used a pretrained EnCodec:
export.export_pretrained_compression_model(
    'facebook/encodec_32khz',
    '/my/models/my_musicgen/compression_state_dict.bin'
)

# If you trained your own EnCodec:
xp_encodec = train.main.get_xp_from_sig('ENCODEC_SIGNATURE')
export.export_encodec(
    xp_encodec.folder / 'checkpoint.th',
    '/my/models/my_musicgen/compression_state_dict.bin'
)
```

### Load and Use Model

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

# Load your exported model
model = MusicGen.get_pretrained('/my/models/my_musicgen')

# Set generation parameters
model.set_generation_params(
    duration=10,        # Duration in seconds
    temperature=1.0,    # Sampling temperature
    top_k=250,          # Top-k sampling
    top_p=0.0,          # Top-p (nucleus) sampling
    cfg_coef=3.0        # Classifier-free guidance coefficient
)

# Generate music
descriptions = ['A beautiful piano melody']
with torch.no_grad():
    wav = model.generate(descriptions)

# Save the result
for idx, one_wav in enumerate(wav):
    audio_write(
        f'output_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy='loudness',
        loudness_compressor=True
    )
```

### Share on Hugging Face

You can upload your model to Hugging Face for sharing:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/my/models/my_musicgen",
    repo_id="your-username/my-musicgen-model",
    repo_type="model",
)
```

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size or use gradient accumulation:
```bash
python scripts/finetune_musicgen.py --batch-size 2
```

Or edit the config:
```yaml
dataset:
  batch_size: 2  # Reduce this
```

### Slow Training

**Solutions**:
1. Use a smaller model scale: `--model small`
2. Reduce segment duration: `--segment-duration 20`
3. Use mixed precision (already enabled by default)
4. Use multiple GPUs: `--distributed`

### Poor Generation Quality

**Solutions**:
1. **More training data**: At least 10-100 hours recommended
2. **Better metadata**: Add detailed text descriptions
3. **Longer training**: Increase epochs
4. **Lower learning rate**: Try `--lr 5e-5` or `1e-5`
5. **Use better base model**: Start from `medium` or `large`

### Loss Not Decreasing

**Solutions**:
1. **Check data**: Ensure audio files are valid and diverse
2. **Lower learning rate**: Try `--lr 1e-5`
3. **Check for data leakage**: Ensure validation set is separate
4. **Increase batch size**: If possible

### Checkpoint Errors

**Solution**: Clear and restart:
```bash
python scripts/finetune_musicgen.py --clear --model medium
```

### Import Errors

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Advanced Topics

### Custom Audio Tokenizer

To use a different audio tokenizer (e.g., DAC):

```bash
dora run solver=musicgen/finetune_32khz \
    compression_model_checkpoint=//pretrained/dac_44khz \
    transformer_lm.n_q=9 \
    transformer_lm.card=1024 \
    'codebooks_pattern.delay.delays=[0,1,2,3,4,5,6,7,8]'
```

### Fine-tuning from Your Own Checkpoint

```bash
dora run solver=musicgen/finetune_32khz \
    continue_from=//sig/YOUR_PREVIOUS_SIGNATURE \
    model/lm/model_scale=medium
```

### Conditional Generation Modes

AudioCraft supports different conditioning:
- `text2music`: Text-to-music generation
- `music_style`: Style transfer
- `melody`: Melody-guided generation

Change conditioner in config or command line:
```bash
dora run solver=musicgen/finetune_32khz conditioner=text2music
```

## Best Practices

1. **Start Small**: Begin with the debug config to verify everything works
2. **Use Pretrained Models**: Always finetune from a pretrained model
3. **Quality Data**: Better quality training data = better results
4. **Diverse Data**: Include diverse music styles for better generalization
5. **Text Descriptions**: Write detailed, accurate descriptions
6. **Monitor Training**: Regularly listen to generated samples
7. **Save Checkpoints**: Keep multiple checkpoints at different epochs
8. **Experiment**: Try different hyperparameters and model scales

## Resources

- **Paper**: [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
- **Documentation**: [AudioCraft Docs](https://github.com/facebookresearch/audiocraft)
- **Samples**: [Sample Page](https://ai.honu.io/papers/musicgen/)
- **Issues**: [GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Search [GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)
3. Ask in the [Discussions](https://github.com/facebookresearch/audiocraft/discussions)

## Citation

If you use AudioCraft in your research, please cite:

```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

---

Happy finetuning! 🎵
