# Encodec Compression Testing

This guide explains how to test your fine-tuned 32kHz Encodec model for compression.

## Files Created

1. **`test_encodec_compression.py`** - Full-featured script with comprehensive metrics and command-line interface
2. **`simple_compression_test.py`** - Minimal script that you can easily modify
3. **`visualize_compression.py`** - Create visual comparisons of original vs reconstructed audio
4. **`run_full_test.sh`** - Automated script to run the complete workflow
5. **`README_COMPRESSION_TEST.md`** - This documentation file

## Quick Start

### Option 1: Automated Script (Recommended)

The fastest way to run a complete test:

```bash
# Edit paths in run_full_test.sh, then run:
./run_full_test.sh
```

This automatically runs compression testing and creates visualizations.

### Option 2: Simple Python Script

Edit `simple_compression_test.py` to set your paths, then:

```bash
python simple_compression_test.py
```

This will:
- Load your fine-tuned model
- Compress and decompress your audio
- Print compression statistics (ratio, bandwidth, SNR)
- Save the reconstructed audio to `reconstructed_audio.wav`
- Save the discrete codes to `audio_codes.pt`

## Full-Featured Script

For more control and better metrics:

```bash
python test_encodec_compression.py \
    --checkpoint /home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin \
    --input /home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav \
    --output reconstructed.wav \
    --device cuda
```

### Options:

- `--checkpoint`: Path to your model checkpoint (.bin file)
- `--input`: Path to input audio file
- `--output`: Where to save reconstructed audio (optional)
- `--device`: `cuda` or `cpu` (default: auto-detect)
- `--num-codebooks`: Number of codebooks to use (optional, default: use all)

### Example with different bandwidths:

Test different compression levels by varying the number of codebooks:

```bash
# Low bandwidth (fewer codebooks)
python test_encodec_compression.py \
    --checkpoint /path/to/checkpoint.bin \
    --input input.wav \
    --output output_low.wav \
    --num-codebooks 2

# High bandwidth (more codebooks)
python test_encodec_compression.py \
    --checkpoint /path/to/checkpoint.bin \
    --input input.wav \
    --output output_high.wav \
    --num-codebooks 8
```

## Understanding the Output

### Compression Metrics:
- **Compression ratio**: How much the audio is compressed (e.g., 10x means the compressed version is 10 times smaller)
- **Bandwidth**: Bitrate in kbps (lower = more compression)
- **Number of codebooks**: More codebooks = higher quality but less compression

### Quality Metrics:
- **SNR (Signal-to-Noise Ratio)**: Higher is better. Good values: > 20 dB
- **PSNR (Peak SNR)**: Similar to SNR but normalized by peak values
- **MSE (Mean Squared Error)**: Lower is better

## Troubleshooting

### If the model path doesn't work:

The checkpoint file might be in a different format. Try these alternatives:

```python
# Option 1: Load as CompressionSolver checkpoint
from audiocraft.solvers import CompressionSolver
model = CompressionSolver.model_from_checkpoint('/path/to/checkpoint.bin', device='cuda')

# Option 2: Load as direct model
from audiocraft.models import CompressionModel
checkpoint = torch.load('/path/to/checkpoint.bin', map_location='cpu')
# Then manually build and load the model
```

### If audio loading fails:

Make sure:
- The audio file exists and is readable
- It's in a supported format (wav, mp3, flac, etc.)
- The sample rate matches (32kHz for your model)

### Channel mismatch:

If your model is mono but audio is stereo (or vice versa):
- The script automatically converts
- Stereo → Mono: averages the channels
- Mono → Stereo: duplicates the channel

## Using the Compressed Codes

The discrete codes are saved to `.codes.pt` files. You can load and use them:

```python
import torch

# Load codes
data = torch.load('audio_codes.pt')
codes = data['codes']  # Shape: [1, num_codebooks, num_frames]
scale = data['scale']  # Optional scale factor

# Use codes for other tasks (e.g., language modeling, analysis)
print(f"Code vocabulary size: {codes.max().item() + 1}")
print(f"Code statistics: min={codes.min()}, max={codes.max()}, unique={len(codes.unique())}")

# Reconstruct from codes
from audiocraft.solvers import CompressionSolver
model = CompressionSolver.model_from_checkpoint('/path/to/checkpoint.bin')
reconstructed = model.decode(codes, scale)
```

## Visualizing Results

Use the visualization script to create plots comparing original and reconstructed audio:

```bash
python visualize_compression.py \
    --original /path/to/original.wav \
    --reconstructed reconstructed.wav \
    --codes audio_codes.pt \
    --output compression_comparison.png
```

This creates a comprehensive visualization showing:
- Original and reconstructed waveforms
- Original and reconstructed spectrograms
- Difference signal and spectrogram
- Discrete codes heatmap (if codes file provided)
- Quality metrics (SNR, RMSE)

## Complete Workflow Example

Here's a complete example workflow:

```bash
# Step 1: Test compression
python test_encodec_compression.py \
    --checkpoint /home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin \
    --input /home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav \
    --output reconstructed.wav \
    --device cuda

# Step 2: Visualize results
python visualize_compression.py \
    --original /home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav \
    --reconstructed reconstructed.wav \
    --codes reconstructed.codes.pt \
    --output comparison.png

# Step 3: Listen to the results and check the visualization
# - Original: prompt_A.wav
# - Reconstructed: reconstructed.wav
# - Visualization: comparison.png
```

## Next Steps

After testing compression:

1. **Compare quality**: Listen to original vs reconstructed audio
2. **Check visualizations**: Review the spectrogram and waveform plots
3. **Test on multiple files**: Try different types of violin audio
4. **Tune bandwidth**: Adjust num_codebooks for your quality/size tradeoff
5. **Use for downstream tasks**: Use the codes for music generation, etc.

## Integration with Other Models

The compressed codes can be used as input to:
- Language models (like MusicGen/AudioGen)
- VALL-E style models for speech/music synthesis
- Music understanding/analysis models
- Any model that works with discrete audio tokens
