# 🎵 MusicGen Finetuning Setup - Complete Guide

This repository is now fully configured for finetuning MusicGen models for music generation!

## ✅ What's Been Set Up

A complete finetuning pipeline has been created with the following components:

### 📄 Documentation (Start Here!)
- **`QUICKSTART.md`** - Get started in 5 simple steps ⚡
- **`FINETUNING_GUIDE.md`** - Comprehensive 20-page guide 📚
- **`FINETUNING_SUMMARY.md`** - Overview of all created files 📋
- **`README_FINETUNING.md`** - This file 📖

### 🛠️ Scripts & Tools
- **`install_dependencies.sh`** - One-click dependency installation
- **`scripts/finetune_musicgen.py`** - Easy-to-use finetuning launcher
- **`scripts/prepare_music_dataset.py`** - Dataset preparation tool
- **`test_setup.py`** - Environment verification script
- **`example_usage.py`** - Code examples for using trained models

### ⚙️ Configuration Files
- **`config/solver/musicgen/finetune_32khz.yaml`** - Training configuration
- **`config/dset/audio/custom_music.yaml`** - Dataset configuration
- **`metadata_template.csv`** - Template for music metadata

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
bash install_dependencies.sh

# 2. Prepare your dataset
python scripts/prepare_music_dataset.py /path/to/your/music --output-dir egs/my_music --split 0.9

# 3. Start finetuning
python scripts/finetune_musicgen.py --model medium --dataset custom_music
```

That's it! Your model will start training.

## 📊 What You'll Get

After finetuning, you'll have:
- ✅ A custom music generation model trained on your data
- ✅ Ability to generate music from text descriptions
- ✅ Control over genre, mood, instruments, and style
- ✅ High-quality audio output (32kHz, stereo optional)
- ✅ Integration with the MusicGen API

## 🎯 Typical Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PREPARE DATA                                             │
│    • Collect music files (MP3, WAV, FLAC, etc.)            │
│    • Create metadata (descriptions, genres, moods)          │
│    • Run prepare_music_dataset.py                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CONFIGURE                                                │
│    • Set environment variables                              │
│    • Choose model size (small/medium/large)                 │
│    • Adjust batch size and learning rate                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAIN                                                    │
│    • Run finetune_musicgen.py                              │
│    • Monitor with TensorBoard                               │
│    • Listen to generated samples                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EVALUATE                                                 │
│    • Check loss/perplexity metrics                         │
│    • Generate test samples                                  │
│    • Compare with validation data                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DEPLOY                                                   │
│    • Export model checkpoint                                │
│    • Use MusicGen API for generation                        │
│    • Share on Hugging Face (optional)                       │
└─────────────────────────────────────────────────────────────┘
```

## 💻 System Requirements

### Minimum
- **GPU**: 8GB VRAM (for small model)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space
- **OS**: Linux, macOS, or Windows with WSL

### Recommended
- **GPU**: 16GB+ VRAM (RTX 3090, A100, V100)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space
- **CPU**: 8+ cores

### Software
- Python 3.8+
- CUDA 11.8+ or 12.1+
- PyTorch 2.1.0
- Internet connection (for downloading pretrained models)

## 📚 Documentation Guide

### For Beginners
1. Start with **QUICKSTART.md** (5-minute read)
2. Run the setup and training commands
3. Refer to **FINETUNING_GUIDE.md** as needed

### For Advanced Users
1. Read **FINETUNING_SUMMARY.md** for file overview
2. Dive into **FINETUNING_GUIDE.md** for advanced topics
3. Customize configuration files directly

### For Troubleshooting
1. Check **FINETUNING_GUIDE.md** → Troubleshooting section
2. Run `python test_setup.py` to verify environment
3. Check GitHub issues or discussions

## 🎓 Key Concepts

### What is MusicGen?
MusicGen is a state-of-the-art music generation model that:
- Uses an **EnCodec** neural audio codec to tokenize audio
- Employs a **Transformer language model** to generate token sequences
- Supports **text conditioning** for controllable generation
- Generates **high-quality audio** at 32kHz

### What is Finetuning?
Finetuning adapts a pretrained model to your specific data:
- **Faster** than training from scratch
- **Better quality** with less data
- **Preserves** general music knowledge
- **Specializes** in your music style

### What You Need
1. **Music Files**: 10-100 hours recommended (minimum: 1 hour)
2. **Metadata**: Text descriptions of your music (highly recommended)
3. **GPU**: CUDA-capable NVIDIA GPU
4. **Time**: Several hours to days of training

## 📋 File Structure Reference

```
audiocraft/
├── 📄 QUICKSTART.md                          ← Start here!
├── 📄 FINETUNING_GUIDE.md                   ← Complete reference
├── 📄 FINETUNING_SUMMARY.md                 ← File overview
├── 📄 metadata_template.csv                  ← Metadata template
│
├── 🔧 install_dependencies.sh                ← Install everything
├── 🔧 test_setup.py                         ← Verify setup
├── 🔧 example_usage.py                      ← Usage examples
│
├── scripts/
│   ├── finetune_musicgen.py                 ← Main training script
│   └── prepare_music_dataset.py             ← Dataset preparation
│
├── config/
│   ├── solver/musicgen/
│   │   └── finetune_32khz.yaml             ← Training config
│   └── dset/audio/
│       └── custom_music.yaml                ← Dataset config
│
├── egs/                                     ← Dataset manifests
│   └── my_music/                            (you'll create this)
│       ├── train/data.jsonl
│       └── valid/data.jsonl
│
└── experiments/                             ← Training outputs
    └── audiocraft/outputs/                  (created during training)
        └── SIGNATURE/
            ├── checkpoint.th                ← Model checkpoint
            ├── samples/                     ← Generated audio
            └── logs/                        ← Training logs
```

## 🎨 Usage Examples

### Example 1: Basic Finetuning
```bash
# Prepare dataset
python scripts/prepare_music_dataset.py ~/Music/my_collection \
    --output-dir egs/my_music \
    --split 0.9

# Start training
python scripts/finetune_musicgen.py \
    --model medium \
    --dataset custom_music
```

### Example 2: Advanced Configuration
```bash
python scripts/finetune_musicgen.py \
    --model large \
    --dataset custom_music \
    --batch-size 4 \
    --lr 5e-5 \
    --epochs 200 \
    --distributed
```

### Example 3: Generate Music
```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load your model
model = MusicGen.get_pretrained('./experiments/audiocraft/outputs/abc123')
model.set_generation_params(duration=30)

# Generate
wav = model.generate(['Your custom prompt here'])

# Save
audio_write('output', wav[0].cpu(), model.sample_rate)
```

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./experiments/audiocraft/outputs/
# Open http://localhost:6006
```

### Command Line
```bash
# View logs
dora info -f SIGNATURE -t

# List experiments
ls ./experiments/audiocraft/outputs/
```

### Listen to Samples
Generated samples are saved during training:
```bash
./experiments/audiocraft/outputs/SIGNATURE/samples/
```

## 🆘 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of Memory | `--batch-size 2` or `--model small` |
| Slow Training | Use smaller model or `--distributed` |
| Poor Quality | More data, better metadata, longer training |
| Import Errors | Run `bash install_dependencies.sh` |
| No GPU | Training will be very slow (not recommended) |

## 🎯 Best Practices

1. ✅ **Quality over Quantity**: 10 hours of high-quality, well-described music > 100 hours of poor quality
2. ✅ **Diverse Data**: Include various styles for better generalization
3. ✅ **Good Descriptions**: Detailed text descriptions = better control
4. ✅ **Monitor Regularly**: Listen to generated samples during training
5. ✅ **Start Small**: Use debug config first to verify everything works
6. ✅ **Save Checkpoints**: Keep multiple checkpoints at different epochs
7. ✅ **Use Pretrained**: Always finetune from a pretrained model

## 📖 Additional Resources

### Official Documentation
- [AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [Sample Page](https://ai.honu.io/papers/musicgen/)
- [Hugging Face Demo](https://huggingface.co/spaces/facebook/MusicGen)

### Community
- [GitHub Discussions](https://github.com/facebookresearch/audiocraft/discussions)
- [GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)

### Tutorials
- [Training Documentation](docs/TRAINING.md)
- [MusicGen Documentation](docs/MUSICGEN.md)
- [Dataset Documentation](docs/DATASETS.md)

## 🎊 You're All Set!

Everything is configured and ready to go. Here's your action plan:

### Step 1: Install (5 minutes)
```bash
bash install_dependencies.sh
```

### Step 2: Verify (1 minute)
```bash
python test_setup.py
```

### Step 3: Configure (2 minutes)
```bash
export AUDIOCRAFT_TEAM=default
export AUDIOCRAFT_DORA_DIR=./experiments
```

### Step 4: Prepare Data (varies)
```bash
python scripts/prepare_music_dataset.py /path/to/music --output-dir egs/my_music
```

### Step 5: Train! (hours to days)
```bash
python scripts/finetune_musicgen.py --model medium --dataset custom_music
```

## 💡 Tips for Success

- **Start with the example dataset** to verify everything works
- **Use the medium model** for best quality/speed tradeoff
- **Monitor GPU usage** with `nvidia-smi`
- **Listen to samples early** to catch issues
- **Be patient** - good results take time!

## 🤝 Support

Need help? Check:
1. **FINETUNING_GUIDE.md** - Comprehensive troubleshooting
2. **test_setup.py** - Verify your environment
3. **GitHub Issues** - Search existing issues
4. **GitHub Discussions** - Ask questions

## 📝 Citation

If you use this setup or AudioCraft in your work, please cite:

```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

---

## 🎵 Happy Finetuning!

You now have a complete, production-ready setup for finetuning MusicGen. 

**Questions?** → Read FINETUNING_GUIDE.md  
**Ready to start?** → Follow QUICKSTART.md  
**Need examples?** → Run example_usage.py  

Good luck creating amazing music! 🎶

---

*Setup created: 2025-10-17*  
*AudioCraft Repository: https://github.com/facebookresearch/audiocraft*
