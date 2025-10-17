# MusicGen Finetuning - Setup Summary

This document summarizes the MusicGen finetuning setup that has been created for you.

## 📁 Files Created

### Configuration Files
1. **`config/solver/musicgen/finetune_32khz.yaml`**
   - Main finetuning configuration
   - Optimized for finetuning pretrained MusicGen models
   - Configurable batch size, learning rate, and training parameters

2. **`config/dset/audio/custom_music.yaml`**
   - Dataset configuration
   - Points to your custom music data manifests
   - Edit this to match your dataset location

### Scripts
3. **`scripts/finetune_musicgen.py`**
   - Convenient wrapper for launching finetuning jobs
   - Handles model selection, dataset configuration, and hyperparameters
   - Usage: `python scripts/finetune_musicgen.py --model medium --dataset custom_music`

4. **`scripts/prepare_music_dataset.py`**
   - Prepares your music dataset for training
   - Creates JSONL manifest files required by AudioCraft
   - Supports metadata CSV files and train/validation splitting
   - Usage: `python scripts/prepare_music_dataset.py /path/to/music --output-dir egs/my_music`

5. **`test_setup.py`**
   - Verifies your environment is properly configured
   - Checks for required dependencies
   - Usage: `python test_setup.py`

6. **`install_dependencies.sh`**
   - Automated installation script
   - Installs PyTorch, AudioCraft, and all dependencies
   - Usage: `bash install_dependencies.sh`

### Documentation
7. **`FINETUNING_GUIDE.md`**
   - Comprehensive guide covering all aspects of finetuning
   - Includes troubleshooting, best practices, and advanced topics
   - ~500 lines of detailed documentation

8. **`QUICKSTART.md`**
   - Quick reference for getting started
   - 5-step process to begin finetuning
   - Common issues and solutions

9. **`metadata_template.csv`**
   - Template CSV file for your music metadata
   - Shows the expected format for descriptions, genres, moods, etc.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
bash install_dependencies.sh
```

### 2. Set Environment
```bash
export AUDIOCRAFT_TEAM=default
export AUDIOCRAFT_DORA_DIR=./experiments
```

### 3. Prepare Dataset
```bash
python scripts/prepare_music_dataset.py /path/to/your/music \
    --output-dir egs/my_music \
    --split 0.9
```

### 4. Start Finetuning
```bash
python scripts/finetune_musicgen.py --model medium --dataset custom_music
```

## 📋 What This Setup Provides

### Training Configuration
- **Model Scales**: Small (300M), Medium (1.5B), Large (3.3B)
- **Optimizer**: AdamW with cosine learning rate schedule
- **Mixed Precision**: Automatic mixed precision (FP16) enabled
- **Batch Size**: Configurable (default: 8)
- **Learning Rate**: 1e-4 (optimized for finetuning)
- **Epochs**: 100 (customizable)

### Features
- ✅ Pretrained model loading (MusicGen small/medium/large)
- ✅ Text-to-music conditioning
- ✅ Automatic data loading from JSONL manifests
- ✅ Train/validation split support
- ✅ Automatic checkpointing
- ✅ TensorBoard logging
- ✅ Sample generation during training
- ✅ Resume from checkpoint
- ✅ Multi-GPU (distributed) training
- ✅ Stereo audio support
- ✅ Custom audio tokenizer support

### Dataset Support
- **Formats**: MP3, WAV, FLAC, OGG, M4A
- **Metadata**: JSON files with descriptions, genres, moods, etc.
- **CSV Import**: Convert CSV metadata to JSON automatically
- **Automatic Splitting**: Train/validation split with configurable ratio

## 🎯 Finetuning Workflow

```
1. Prepare Dataset
   ├─ Collect music files
   ├─ Create metadata (optional but recommended)
   └─ Run prepare_music_dataset.py
   
2. Configure
   ├─ Set environment variables
   ├─ Edit config/dset/audio/custom_music.yaml
   └─ Choose model scale (small/medium/large)
   
3. Train
   ├─ Run finetune_musicgen.py
   ├─ Monitor with TensorBoard
   └─ Listen to generated samples
   
4. Evaluate
   ├─ Check loss and perplexity
   ├─ Generate test samples
   └─ Compare with validation data
   
5. Export & Use
   ├─ Export model checkpoint
   ├─ Load with MusicGen API
   └─ Generate new music!
```

## 🔧 Configuration Options

### Model Selection
```bash
--model small   # 300M params, ~8GB VRAM
--model medium  # 1.5B params, ~16GB VRAM (recommended)
--model large   # 3.3B params, ~32GB VRAM
```

### Training Parameters
```bash
--batch-size 8           # Adjust based on GPU memory
--lr 1e-4                # Learning rate
--epochs 100             # Total training epochs
--segment-duration 30    # Audio segment length in seconds
```

### Advanced Options
```bash
--distributed            # Multi-GPU training
--stereo                 # Stereo audio training
--continue-from PATH     # Resume from checkpoint
--clear                  # Clear previous checkpoint
```

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./experiments/audiocraft/outputs/SIGNATURE
# Open http://localhost:6006
```

### Logs
```bash
# View training logs
dora info -f SIGNATURE -t
```

### Generated Samples
Samples are automatically generated every N epochs in:
```
./experiments/audiocraft/outputs/SIGNATURE/samples/
```

## 💡 Best Practices

1. **Start with the medium model** - Best quality/compute tradeoff
2. **Use high-quality audio** - Better input = better output
3. **Provide detailed descriptions** - Text conditioning improves results
4. **Diverse training data** - Include various styles and genres
5. **Monitor samples regularly** - Listen to generated music during training
6. **Sufficient training time** - At least 50-100 epochs recommended
7. **Validate properly** - Use separate validation set

## 🎵 Expected Results

With proper dataset (10-100 hours of music):
- **After 10 epochs**: Basic melody structure appears
- **After 30 epochs**: Recognizable musical patterns
- **After 50 epochs**: Good quality generation
- **After 100+ epochs**: High-quality, coherent music

## 📚 Documentation Overview

### QUICKSTART.md
- **Purpose**: Get started in 5 steps
- **Length**: ~2 pages
- **Target**: Beginners wanting to start quickly

### FINETUNING_GUIDE.md
- **Purpose**: Comprehensive reference
- **Length**: ~20 pages
- **Coverage**: 
  - Detailed setup instructions
  - Dataset preparation
  - Configuration options
  - Training strategies
  - Troubleshooting
  - Advanced topics
  - Export and deployment

## 🔍 File Locations Reference

```
audiocraft/
├── config/
│   ├── solver/musicgen/
│   │   └── finetune_32khz.yaml      # Main config
│   └── dset/audio/
│       └── custom_music.yaml         # Dataset config
├── scripts/
│   ├── finetune_musicgen.py          # Finetuning script
│   └── prepare_music_dataset.py      # Dataset preparation
├── egs/
│   └── my_music/                     # Your dataset manifests
│       ├── train/data.jsonl
│       └── valid/data.jsonl
├── experiments/                       # Training outputs
│   └── audiocraft/outputs/SIGNATURE/
│       ├── checkpoint.th             # Model checkpoint
│       ├── samples/                  # Generated samples
│       └── logs/                     # Training logs
├── QUICKSTART.md                     # Quick start guide
├── FINETUNING_GUIDE.md              # Comprehensive guide
├── FINETUNING_SUMMARY.md            # This file
├── metadata_template.csv             # Metadata template
├── test_setup.py                     # Setup verification
└── install_dependencies.sh           # Installation script
```

## 🆘 Getting Help

### Troubleshooting
1. Check **FINETUNING_GUIDE.md** - Troubleshooting section
2. Run `python test_setup.py` - Verify setup
3. Check GPU memory: `nvidia-smi`
4. View logs: `dora info -f SIGNATURE -t`

### Common Issues
- **Out of Memory**: Reduce batch size (`--batch-size 2`)
- **Slow Training**: Use smaller model or GPU
- **Poor Quality**: More data, better metadata, longer training
- **Import Errors**: Run `pip install -r requirements.txt`

### Resources
- [AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [Sample Page](https://ai.honu.io/papers/musicgen/)

## 📝 Next Steps

1. ✅ **Install dependencies**: `bash install_dependencies.sh`
2. ✅ **Verify setup**: `python test_setup.py`
3. ✅ **Prepare dataset**: Use `prepare_music_dataset.py`
4. ✅ **Start training**: Use `finetune_musicgen.py`
5. ✅ **Monitor progress**: TensorBoard and logs
6. ✅ **Generate music**: Test your finetuned model

## 🎉 You're Ready!

Everything is set up for you to start finetuning MusicGen. Follow the QUICKSTART.md for the fastest path, or dive into FINETUNING_GUIDE.md for comprehensive documentation.

Happy finetuning! 🎵

---

**Created**: $(date)
**AudioCraft Version**: Latest from repository
**Python**: 3.8+
**PyTorch**: 2.1.0
**CUDA**: 11.8+ or 12.1+
