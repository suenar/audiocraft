# Testing Your Fine-Tuned 32kHz Encodec Model

## Overview

I've created a complete toolkit for testing your fine-tuned Encodec compression model. The toolkit includes scripts for compression testing, quality analysis, and visualization.

## What I Created

### 1. **test_encodec_compression.py**
Full-featured command-line tool with:
- Model loading and validation
- Audio compression/decompression
- Comprehensive metrics (compression ratio, bandwidth, SNR, PSNR, MSE)
- Automatic audio resampling and channel conversion
- Saves reconstructed audio and discrete codes

### 2. **simple_compression_test.py**
Minimal standalone script:
- Easy to edit and customize
- All paths configured at the top
- Step-by-step output
- Perfect for quick tests

### 3. **visualize_compression.py**
Creates comprehensive visualizations:
- Original vs reconstructed waveforms
- Original vs reconstructed spectrograms
- Difference signal analysis
- Discrete codes heatmap
- Quality metrics overlay

### 4. **run_full_test.sh**
Automated bash script that runs the complete workflow:
- Runs compression test
- Creates visualizations
- Organizes output files
- Prints summary

### 5. **README_COMPRESSION_TEST.md**
Complete documentation including:
- Quick start guides
- Detailed usage instructions
- Troubleshooting tips
- Integration examples

## Quick Start Guide

### Easiest Method (Automated):

1. Edit `run_full_test.sh` to set your paths:
   ```bash
   CHECKPOINT="/path/to/compression_32khz_violin.bin"
   INPUT_AUDIO="/path/to/prompt_A.wav"
   ```

2. Run:
   ```bash
   ./run_full_test.sh
   ```

### Alternative (Step-by-step):

```bash
# 1. Test compression
python test_encodec_compression.py \
    --checkpoint /path/to/compression_32khz_violin.bin \
    --input /path/to/prompt_A.wav \
    --output reconstructed.wav

# 2. Visualize results
python visualize_compression.py \
    --original /path/to/prompt_A.wav \
    --reconstructed reconstructed.wav \
    --codes reconstructed.codes.pt \
    --output comparison.png
```

### Simplest (Python only):

1. Edit paths in `simple_compression_test.py`
2. Run: `python simple_compression_test.py`

## What You'll Get

### Metrics:
- **Compression Ratio**: How much smaller the compressed version is (e.g., 10.5x)
- **Bandwidth**: Bitrate in kbps (e.g., 6.4 kbps)
- **SNR**: Signal-to-Noise Ratio in dB (higher is better, >20 dB is good)
- **Codebook Statistics**: Number of codebooks used, frame rate, etc.

### Output Files:
- **reconstructed.wav**: The compressed and decompressed audio
- **reconstructed.codes.pt**: Discrete codes (can be used for other tasks)
- **comparison.png**: Visual comparison (waveforms, spectrograms, etc.)

## Important Notes

### Path Issues
The paths you provided don't exist on this system:
- `/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin`
- `/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav`

**You'll need to update the paths in the scripts to match your actual file locations.**

### Model Loading
The scripts try two methods to load your checkpoint:
1. `CompressionSolver.model_from_checkpoint()` - for full training checkpoints
2. `CompressionModel.get_pretrained()` - for exported models

If both fail, the checkpoint might be in a different format.

### Dependencies
Make sure you have these installed:
```bash
pip install torch torchaudio audiocraft
pip install matplotlib librosa numpy
```

## Use Cases

### 1. Quality Assessment
Test how well your model reconstructs violin audio:
```bash
python test_encodec_compression.py \
    --checkpoint model.bin \
    --input violin.wav \
    --output test_output.wav
```

### 2. Bandwidth Testing
Try different compression levels:
```bash
# Low bandwidth (2 codebooks)
python test_encodec_compression.py --checkpoint model.bin --input audio.wav --num-codebooks 2

# High bandwidth (8 codebooks)
python test_encodec_compression.py --checkpoint model.bin --input audio.wav --num-codebooks 8
```

### 3. Batch Testing
Test multiple files:
```bash
for file in violin_samples/*.wav; do
    python test_encodec_compression.py \
        --checkpoint model.bin \
        --input "$file" \
        --output "outputs/$(basename $file)"
done
```

### 4. Extract Discrete Codes
Get discrete codes for language modeling:
```python
import torch
from audiocraft.solvers import CompressionSolver

model = CompressionSolver.model_from_checkpoint('model.bin', device='cuda')
# ... load and encode audio ...
# codes shape: [batch, num_codebooks, num_frames]
# Each code is an integer from 0 to cardinality-1
```

## Troubleshooting

### "FileNotFoundError"
→ Update the paths in the scripts to match your file locations

### "CUDA out of memory"
→ Use `--device cpu` or reduce the audio length

### "Model loading failed"
→ The checkpoint might be in a different format; check the error message for details

### "Audio quality is poor"
→ Try increasing `--num-codebooks` for higher quality (but larger file size)

## Next Steps

1. **Test your model**: Run the scripts with your actual file paths
2. **Evaluate quality**: Listen to reconstructed audio and check metrics
3. **Tune parameters**: Adjust num_codebooks for your quality/size needs
4. **Use the codes**: Feed discrete codes to downstream models (VALL-E, etc.)
5. **Test on diverse data**: Try different types of violin audio

## Need Help?

- Check `README_COMPRESSION_TEST.md` for detailed documentation
- Read the inline comments in the scripts
- Check error messages - they often suggest solutions

---

**All scripts are ready to use! Just update the file paths and run them.**
