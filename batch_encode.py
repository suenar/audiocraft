import torchaudio
import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
input_dir = "/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin"  # Change this to your directory
output_dir = "./encoding_analysis"  # Directory to save analysis results
device = 'cuda:0'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Instantiate a pretrained EnCodec model
from audiocraft.models import CompressionModel
model = CompressionModel.get_pretrained('/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin').to(device)

# The number of codebooks used will be determined by the bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
#model.set_target_bandwidth(12.0)

# Find all .wav files in the directory
wav_files = list(Path(input_dir).glob("*.wav"))
print(f"Found {len(wav_files)} .wav files in {input_dir}")

if len(wav_files) == 0:
    print("No .wav files found. Please check the directory path.")
    exit(1)

# Storage for statistics
all_codes = []
all_scales = []
file_info = []

model.eval()

# Process each file
for idx, wav_path in enumerate(wav_files):
    print(f"\n[{idx+1}/{len(wav_files)}] Processing: {wav_path.name}")
    
    try:
        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(str(wav_path))
        
        if sr != model.sample_rate:
            print(f"  Resampling from {sr} Hz to {model.sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
            wav = resampler(wav)
            sr = model.sample_rate
        
        if wav.shape[0] > 1 and model.channels == 1:
            print("  Converting to mono")
            wav = wav.mean(dim=0, keepdim=True)
        
        wav = wav.unsqueeze(0).to(device)
        
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            codes, scale = model.encode(wav)
        
        print(f"  Codes shape: {codes[0].shape}")  # codes is a list of tensors
        
        # Store for statistics
        all_codes.append(codes[0].cpu())  # codes[0] is the tensor [B, n_q, T]
        all_scales.append(scale)
        file_info.append({
            'filename': wav_path.name,
            'shape': codes[0].shape,
            'original_sr': sr
        })
        
    except Exception as e:
        print(f"  Error processing {wav_path.name}: {e}")
        continue

# Calculate and display statistics
print("\n" + "="*80)
print("ENCODING STATISTICS")
print("="*80)

if len(all_codes) > 0:
    # Concatenate all codes for statistics
    # Note: codes are [B, n_q, T], we'll compute stats across all dimensions
    
    print(f"\nTotal files processed: {len(all_codes)}")
    print(f"Files failed: {len(wav_files) - len(all_codes)}")
    
    # Per-file statistics
    print("\n--- Per-file Information ---")
    for info in file_info:
        print(f"  {info['filename']}: shape={info['shape']}, sr={info['original_sr']} Hz")
    
    # Overall statistics
    print("\n--- Overall Encoding Statistics ---")
    
    # Stack all codes for batch statistics
    # Remove batch dimension and concatenate along time dimension
    codes_for_stats = [c.squeeze(0) for c in all_codes]  # Each is now [n_q, T]
    
    # Calculate per-codebook statistics
    n_codebooks = codes_for_stats[0].shape[0]
    print(f"\nNumber of codebooks: {n_codebooks}")
    
    for q_idx in range(n_codebooks):
        codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats])
        print(f"\nCodebook {q_idx}:")
        print(f"  Shape (total frames): {codebook_values.shape}")
        print(f"  Min value: {codebook_values.min().item()}")
        print(f"  Max value: {codebook_values.max().item()}")
        print(f"  Mean value: {codebook_values.float().mean().item():.2f}")
        print(f"  Std value: {codebook_values.float().std().item():.2f}")
        print(f"  Unique values: {len(torch.unique(codebook_values))}")
    
    # All codebooks combined
    all_values = torch.cat([c.flatten() for c in codes_for_stats])
    print(f"\n--- Combined Statistics (All Codebooks) ---")
    print(f"  Total frames: {all_values.shape[0]}")
    print(f"  Min value: {all_values.min().item()}")
    print(f"  Max value: {all_values.max().item()}")
    print(f"  Mean value: {all_values.float().mean().item():.2f}")
    print(f"  Std value: {all_values.float().std().item():.2f}")
    print(f"  Unique values: {len(torch.unique(all_values))}")
    
    # Value distribution
    print(f"\n--- Value Distribution ---")
    unique_vals, counts = torch.unique(all_values, return_counts=True)
    top_k = 10
    top_indices = torch.argsort(counts, descending=True)[:top_k]
    print(f"  Top {top_k} most frequent values:")
    for i, idx in enumerate(top_indices):
        val = unique_vals[idx].item()
        count = counts[idx].item()
        percentage = 100.0 * count / all_values.shape[0]
        print(f"    {i+1}. Value {val}: {count} occurrences ({percentage:.2f}%)")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, "encoding_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("ENCODING STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total files processed: {len(all_codes)}\n")
        f.write(f"Number of codebooks: {n_codebooks}\n\n")
        
        for q_idx in range(n_codebooks):
            codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats])
            f.write(f"Codebook {q_idx}:\n")
            f.write(f"  Min: {codebook_values.min().item()}\n")
            f.write(f"  Max: {codebook_values.max().item()}\n")
            f.write(f"  Mean: {codebook_values.float().mean().item():.2f}\n")
            f.write(f"  Std: {codebook_values.float().std().item():.2f}\n")
            f.write(f"  Unique values: {len(torch.unique(codebook_values))}\n\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    # 1. Distribution histogram for each codebook
    fig, axes = plt.subplots(n_codebooks, 1, figsize=(12, 3 * n_codebooks))
    if n_codebooks == 1:
        axes = [axes]
    
    for q_idx in range(n_codebooks):
        codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats]).numpy()
        axes[q_idx].hist(codebook_values, bins=50, edgecolor='black', alpha=0.7)
        axes[q_idx].set_xlabel('Code Value')
        axes[q_idx].set_ylabel('Frequency')
        axes[q_idx].set_title(f'Codebook {q_idx} - Distribution of Code Values')
        axes[q_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_file = os.path.join(output_dir, "code_distribution_histograms.png")
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved histograms to: {hist_file}")
    
    # 2. Value frequency bar plot for each codebook (top 20 values)
    fig, axes = plt.subplots(n_codebooks, 1, figsize=(14, 4 * n_codebooks))
    if n_codebooks == 1:
        axes = [axes]
    
    for q_idx in range(n_codebooks):
        codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats])
        unique_vals, counts = torch.unique(codebook_values, return_counts=True)
        top_indices = torch.argsort(counts, descending=True)[:20]
        top_vals = unique_vals[top_indices].numpy()
        top_counts = counts[top_indices].numpy()
        
        axes[q_idx].bar(range(len(top_vals)), top_counts, alpha=0.7)
        axes[q_idx].set_xticks(range(len(top_vals)))
        axes[q_idx].set_xticklabels([str(int(v)) for v in top_vals], rotation=45)
        axes[q_idx].set_xlabel('Code Value')
        axes[q_idx].set_ylabel('Frequency')
        axes[q_idx].set_title(f'Codebook {q_idx} - Top 20 Most Frequent Code Values')
        axes[q_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    freq_file = os.path.join(output_dir, "code_frequency_bars.png")
    plt.savefig(freq_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved frequency bars to: {freq_file}")
    
    # 3. Overall statistics comparison across codebooks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    codebook_means = []
    codebook_stds = []
    codebook_mins = []
    codebook_maxs = []
    
    for q_idx in range(n_codebooks):
        codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats])
        codebook_means.append(codebook_values.float().mean().item())
        codebook_stds.append(codebook_values.float().std().item())
        codebook_mins.append(codebook_values.min().item())
        codebook_maxs.append(codebook_values.max().item())
    
    codebook_indices = list(range(n_codebooks))
    
    axes[0, 0].plot(codebook_indices, codebook_means, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Codebook Index')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].set_title('Mean Code Value per Codebook')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(codebook_indices, codebook_stds, marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Codebook Index')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].set_title('Standard Deviation per Codebook')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(codebook_indices, codebook_mins, marker='o', linewidth=2, color='green', label='Min')
    axes[1, 0].plot(codebook_indices, codebook_maxs, marker='o', linewidth=2, color='red', label='Max')
    axes[1, 0].set_xlabel('Codebook Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Min/Max Values per Codebook')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Unique values per codebook
    unique_counts = []
    for q_idx in range(n_codebooks):
        codebook_values = torch.cat([c[q_idx].flatten() for c in codes_for_stats])
        unique_counts.append(len(torch.unique(codebook_values)))
    
    axes[1, 1].bar(codebook_indices, unique_counts, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Codebook Index')
    axes[1, 1].set_ylabel('Number of Unique Values')
    axes[1, 1].set_title('Codebook Utilization (Unique Values)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_file = os.path.join(output_dir, "codebook_summary_statistics.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary statistics to: {summary_file}")

else:
    print("No files were successfully processed.")

print("\n" + "="*80)
print("Processing complete!")
print("="*80)
