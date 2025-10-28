# MusicGen Finetuning - Quick Start

Get started with MusicGen finetuning in 5 steps!

## Step 1: Install Dependencies

```bash
# Make sure you have PyTorch with CUDA installed
pip install torch==2.1.0 torchvision==0.16.0 torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AudioCraft
pip install -r requirements.txt
pip install -e .
```

## Step 2: Prepare Your Music Dataset

Place your music files in a folder, then run:

```bash
# Basic preparation
python scripts/prepare_music_dataset.py /path/to/your/music --output-dir egs/my_music --split 0.9
```

**With metadata** (recommended for better results):

1. Create a CSV file with your metadata (use `metadata_template.csv` as reference)
2. Run:
```bash
python scripts/prepare_music_dataset.py /path/to/your/music \
    --metadata metadata.csv \
    --output-dir egs/my_music \
    --split 0.9
```

## Step 3: Configure Dataset Path

The dataset config is already created at `config/dset/audio/custom_music.yaml`.
If you used a different output directory, edit it:

```yaml
datasource:
  train: egs/my_music/train
  valid: egs/my_music/valid
```

## Step 4: Set Environment Variables

```bash
# Quick setup
export AUDIOCRAFT_TEAM=default
export AUDIOCRAFT_DORA_DIR=./experiments  # Your experiments will be saved here
```

**Important**: Edit `config/teams/default.yaml` and change `dora_dir` from `/tmp/` to a permanent location!

## Step 5: Start Finetuning

```bash
# Finetune the medium model (recommended)
python scripts/finetune_musicgen.py --model medium --dataset custom_music

# Or use small model if you have limited GPU memory
python scripts/finetune_musicgen.py --model small --dataset custom_music --batch-size 4
```

## Monitor Progress

```bash
# Find your experiment signature in the logs, then:
dora info -f SIGNATURE -t  # View training logs

# Or use TensorBoard
tensorboard --logdir ./experiments/audiocraft/outputs/SIGNATURE
```

## Generate Music

After training (or even during training), test your model:

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load from checkpoint directory (find it in your experiments folder)
model = MusicGen.get_pretrained('./experiments/audiocraft/outputs/SIGNATURE')
model.set_generation_params(duration=10)

# Generate!
descriptions = ['A happy upbeat electronic song']
wav = model.generate(descriptions)

# Save
for idx, one_wav in enumerate(wav):
    audio_write(f'generated_{idx}', one_wav.cpu(), model.sample_rate, 
                strategy='loudness', loudness_compressor=True)
```

## Common Issues

### Out of Memory?
```bash
# Use smaller batch size
python scripts/finetune_musicgen.py --model small --batch-size 2
```

### Want to resume training?
```bash
# Training automatically resumes from the last checkpoint
# Just run the same command again
python scripts/finetune_musicgen.py --model medium --dataset custom_music
```

### Start fresh?
```bash
# Clear previous checkpoint and start over
python scripts/finetune_musicgen.py --model medium --dataset custom_music --clear
```

## Tips for Better Results

1. **More data is better**: Aim for at least 10-100 hours of music
2. **Good metadata**: Write detailed, accurate descriptions for your music
3. **Diverse data**: Include different styles and genres
4. **Quality audio**: Use high-quality audio files (WAV or lossless formats preferred)
5. **Longer training**: Let it train for at least 50-100 epochs
6. **Monitor samples**: Listen to generated samples regularly to assess quality

## Next Steps

- Read the full [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) for advanced options
- Check out the [official documentation](docs/MUSICGEN.md)
- Experiment with different hyperparameters

---

Need help? Check [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) or [open an issue](https://github.com/facebookresearch/audiocraft/issues).
