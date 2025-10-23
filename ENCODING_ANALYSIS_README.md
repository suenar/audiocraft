# Audio Encoding Analysis Scripts

This repository contains two Python scripts for analyzing audio encodings using compression models (like EnCodec).

## Scripts Overview

### 1. `batch_encode.py` - Batch Encoding with Visualization

This script processes multiple .wav files and provides comprehensive statistics and visualizations of the encoding results.

**Features:**
- Batch processing of all .wav files in a directory
- Statistical analysis of code distributions across all codebooks
- No storage of reconstructed audio (analysis only)
- Comprehensive visualizations

**Outputs:**
- `encoding_statistics.txt` - Text file with detailed statistics
- `code_distribution_histograms.png` - Distribution histograms for each codebook
- `code_frequency_bars.png` - Top 20 most frequent codes per codebook
- `codebook_summary_statistics.png` - Comparison across codebooks (mean, std, min/max, utilization)

**Usage:**
```python
# Edit these variables in the script:
input_dir = "/path/to/your/wav/files"
output_dir = "./encoding_analysis"
device = 'cuda:0'  # or 'cpu'

# Run:
python batch_encode.py
```

**Statistics Provided:**
- Per-codebook: min, max, mean, std, unique values
- Overall: combined statistics across all codebooks
- Value distribution: top 10 most frequent codes with percentages
- Codebook utilization: how many unique codes are used

---

### 2. `analyze_intermediate_quantization.py` - Intermediate Quantization Analysis

This script extracts and analyzes codes from **each individual quantization level** in the residual vector quantizer (RVQ), allowing you to see how codes evolve through the quantization hierarchy.

**Features:**
- Extracts codes at each quantization level separately
- Analyzes the distribution of codes at each level
- Shows how different levels utilize the codebook
- Entropy analysis to measure information content at each level

**Key Insight:**
In Residual Vector Quantization (RVQ), the model uses multiple quantizers in sequence:
- Level 0: First quantizer (captures main features)
- Level 1: Second quantizer (captures residual from level 0)
- Level 2: Third quantizer (captures residual from levels 0-1)
- ... and so on

This script lets you see the statistics for EACH level independently.

**Outputs:**
- `intermediate_quantization_statistics.txt` - Detailed statistics per level
- `intermediate_distribution_histograms.png` - Distribution at each level
- `intermediate_frequency_bars.png` - Top 20 codes per level
- `quantization_level_comparison.png` - Comparison of statistics across levels
- `code_usage_heatmap.png` - Heatmap showing which codes are used at each level
- `entropy_analysis.png` - Information content (entropy) at each level

**Usage:**
```python
# Edit these variables in the script:
input_dir = "/path/to/your/wav/files"
output_dir = "./intermediate_quantization_analysis"
device = 'cuda:0'  # or 'cpu'

# Run:
python analyze_intermediate_quantization.py
```

**Statistics Provided Per Level:**
- Min, max, mean, std of code values
- Number of unique codes used
- Codebook utilization percentage
- Top 5 most frequent codes
- Entropy (information content)

---

## Key Differences Between Scripts

| Feature | batch_encode.py | analyze_intermediate_quantization.py |
|---------|----------------|-------------------------------------|
| View | All codebooks combined | Each quantization level separately |
| Analysis | Final encoded result | Step-by-step quantization process |
| Use Case | Overall encoding statistics | Understanding RVQ behavior |
| Outputs | 4 visualizations | 5 visualizations + entropy |
| Reconstruction | Not saved | Not saved |

---

## Understanding the Outputs

### Codebook vs Quantization Level

- **Codebook** (in `batch_encode.py`): Refers to the K different codebooks in the final result [B, K, T]
- **Quantization Level** (in `analyze_intermediate_quantization.py`): Refers to each step in the residual quantization process

### Key Metrics

1. **Mean/Std**: Average code value and variation
2. **Unique Values**: How many different codes are actually used
3. **Utilization**: Percentage of the codebook that is actively used
4. **Entropy**: Information content in bits (higher = more diverse codes)

### Interpreting Results

- **High utilization**: Model is using most of the codebook → good capacity usage
- **Low utilization**: Only a few codes are used → might indicate:
  - Limited data diversity
  - Codebook is too large
  - Dead codes (unused codebook entries)

- **Decreasing entropy across levels**: Normal for RVQ, as later levels capture finer details
- **Very low entropy**: Codes are very predictable → low information content

---

## Requirements

```bash
pip install torch torchaudio matplotlib numpy audiocraft
```

## Model Path

Both scripts use a pretrained model. Update this line with your model path:

```python
model = CompressionModel.get_pretrained('/path/to/your/model.bin').to(device)
```

---

## Tips

1. **Memory**: If processing many/large files, consider processing in smaller batches
2. **Device**: Use 'cuda:0' for GPU acceleration, 'cpu' for CPU-only systems
3. **Visualization**: All plots are saved as high-resolution PNG files (300 DPI)
4. **Custom analysis**: The scripts store all raw codes, so you can add custom analysis as needed

---

## Example Workflow

1. **First**, run `batch_encode.py` to get overall encoding statistics
2. **Then**, run `analyze_intermediate_quantization.py` to understand how each quantization level behaves
3. **Compare** the results to understand:
   - Which codebooks/levels are most active
   - How information is distributed across levels
   - Whether the model is efficiently using its capacity

---

## Questions?

These scripts provide deep insights into how your compression model encodes audio. The intermediate quantization analysis is particularly useful for:
- Debugging RVQ models
- Understanding codebook utilization
- Optimizing the number of quantization levels
- Analyzing model capacity
