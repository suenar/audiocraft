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
output_dir = "./intermediate_quantization_analysis"  # Directory to save analysis results
device = 'cuda:0'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Instantiate a pretrained EnCodec model
from audiocraft.models import CompressionModel
model = CompressionModel.get_pretrained('/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin').to(device)

print(f"Model info:")
print(f"  Total codebooks: {model.total_codebooks}")
print(f"  Active codebooks: {model.num_codebooks}")
print(f"  Cardinality (codebook size): {model.cardinality}")
print(f"  Sample rate: {model.sample_rate}")
print(f"  Frame rate: {model.frame_rate}")

# Find all .wav files in the directory
wav_files = list(Path(input_dir).glob("*.wav"))
print(f"\nFound {len(wav_files)} .wav files in {input_dir}")

if len(wav_files) == 0:
    print("No .wav files found. Please check the directory path.")
    exit(1)

# Custom function to extract intermediate quantization results
def extract_intermediate_codes(model, x):
    """
    Extract codes at each quantization level.
    
    Returns:
        intermediate_codes: List of tensors, each containing codes up to that level
        intermediate_quantized: List of quantized representations at each level
    """
    # Preprocess
    x_preprocessed, scale = model.preprocess(x)
    
    # Encode to get embeddings
    emb = model.encoder(x_preprocessed)
    
    # Get quantizer
    quantizer = model.quantizer
    n_q = quantizer.n_q
    
    # Manually perform residual quantization to capture intermediate results
    residual = emb
    intermediate_codes = []
    intermediate_quantized = []
    cumulative_quantized = torch.zeros_like(emb)
    
    for i, layer in enumerate(quantizer.vq.layers[:n_q]):
        # Encode this level
        indices = layer.encode(residual)
        # Decode to get quantized version
        quantized = layer.decode(indices)
        # Update residual
        residual = residual - quantized
        # Store results
        intermediate_codes.append(indices.cpu())
        cumulative_quantized = cumulative_quantized + quantized
        intermediate_quantized.append(cumulative_quantized.clone().cpu())
    
    # Stack codes [K, B, T] -> [B, K, T]
    all_codes = torch.stack([c for c in intermediate_codes], dim=1)
    
    return all_codes, intermediate_codes, intermediate_quantized, scale

# Storage for statistics
all_files_codes = []
all_files_intermediate_codes = []
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
        
        # Extract intermediate codes
        with torch.no_grad():
            all_codes, intermediate_codes, intermediate_quantized, scale = extract_intermediate_codes(model, wav)
        
        print(f"  Full codes shape: {all_codes.shape}")
        print(f"  Number of quantization levels: {len(intermediate_codes)}")
        
        # Store for statistics
        all_files_codes.append(all_codes.cpu())
        all_files_intermediate_codes.append(intermediate_codes)
        file_info.append({
            'filename': wav_path.name,
            'shape': all_codes.shape,
            'n_levels': len(intermediate_codes)
        })
        
    except Exception as e:
        print(f"  Error processing {wav_path.name}: {e}")
        continue

# Calculate and display statistics
print("\n" + "="*80)
print("INTERMEDIATE QUANTIZATION ANALYSIS")
print("="*80)

if len(all_files_codes) > 0:
    n_levels = file_info[0]['n_levels']
    
    print(f"\nTotal files processed: {len(all_files_codes)}")
    print(f"Number of quantization levels: {n_levels}")
    
    # Analyze each quantization level
    print("\n--- Statistics Per Quantization Level ---")
    
    level_stats = []
    
    for level_idx in range(n_levels):
        print(f"\nQuantization Level {level_idx}:")
        
        # Collect codes at this level from all files
        level_codes = []
        for file_intermediate_codes in all_files_intermediate_codes:
            level_codes.append(file_intermediate_codes[level_idx].flatten())
        
        level_codes_tensor = torch.cat(level_codes)
        
        min_val = level_codes_tensor.min().item()
        max_val = level_codes_tensor.max().item()
        mean_val = level_codes_tensor.float().mean().item()
        std_val = level_codes_tensor.float().std().item()
        unique_vals = len(torch.unique(level_codes_tensor))
        
        print(f"  Total frames: {level_codes_tensor.shape[0]}")
        print(f"  Min value: {min_val}")
        print(f"  Max value: {max_val}")
        print(f"  Mean value: {mean_val:.2f}")
        print(f"  Std value: {std_val:.2f}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Utilization: {100.0 * unique_vals / model.cardinality:.2f}% of codebook")
        
        # Get top 5 most frequent values
        unique, counts = torch.unique(level_codes_tensor, return_counts=True)
        top_indices = torch.argsort(counts, descending=True)[:5]
        print(f"  Top 5 most frequent values:")
        for i, idx in enumerate(top_indices):
            val = unique[idx].item()
            count = counts[idx].item()
            percentage = 100.0 * count / level_codes_tensor.shape[0]
            print(f"    {i+1}. Value {val}: {count} occurrences ({percentage:.2f}%)")
        
        level_stats.append({
            'level': level_idx,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'unique': unique_vals,
            'codes': level_codes_tensor
        })
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, "intermediate_quantization_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("INTERMEDIATE QUANTIZATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total files processed: {len(all_files_codes)}\n")
        f.write(f"Number of quantization levels: {n_levels}\n")
        f.write(f"Codebook cardinality: {model.cardinality}\n\n")
        
        for stats in level_stats:
            f.write(f"Quantization Level {stats['level']}:\n")
            f.write(f"  Min: {stats['min']}\n")
            f.write(f"  Max: {stats['max']}\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n")
            f.write(f"  Unique values: {stats['unique']}\n")
            f.write(f"  Utilization: {100.0 * stats['unique'] / model.cardinality:.2f}%\n\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    # 1. Distribution histogram for each quantization level
    fig, axes = plt.subplots(n_levels, 1, figsize=(14, 3 * n_levels))
    if n_levels == 1:
        axes = [axes]
    
    for level_idx in range(n_levels):
        codes_data = level_stats[level_idx]['codes'].numpy()
        axes[level_idx].hist(codes_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[level_idx].set_xlabel('Code Value')
        axes[level_idx].set_ylabel('Frequency')
        axes[level_idx].set_title(f'Level {level_idx} - Code Distribution (Utilization: {100.0 * level_stats[level_idx]["unique"] / model.cardinality:.1f}%)')
        axes[level_idx].grid(True, alpha=0.3)
        axes[level_idx].axvline(level_stats[level_idx]['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {level_stats[level_idx]['mean']:.1f}")
        axes[level_idx].legend()
    
    plt.tight_layout()
    hist_file = os.path.join(output_dir, "intermediate_distribution_histograms.png")
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved histograms to: {hist_file}")
    
    # 2. Top 20 most frequent codes per level
    fig, axes = plt.subplots(n_levels, 1, figsize=(14, 4 * n_levels))
    if n_levels == 1:
        axes = [axes]
    
    for level_idx in range(n_levels):
        codes_data = level_stats[level_idx]['codes']
        unique_vals, counts = torch.unique(codes_data, return_counts=True)
        top_indices = torch.argsort(counts, descending=True)[:20]
        top_vals = unique_vals[top_indices].numpy()
        top_counts = counts[top_indices].numpy()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_vals)))
        axes[level_idx].bar(range(len(top_vals)), top_counts, alpha=0.8, color=colors)
        axes[level_idx].set_xticks(range(len(top_vals)))
        axes[level_idx].set_xticklabels([str(int(v)) for v in top_vals], rotation=45)
        axes[level_idx].set_xlabel('Code Value')
        axes[level_idx].set_ylabel('Frequency')
        axes[level_idx].set_title(f'Level {level_idx} - Top 20 Most Frequent Codes')
        axes[level_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    freq_file = os.path.join(output_dir, "intermediate_frequency_bars.png")
    plt.savefig(freq_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved frequency bars to: {freq_file}")
    
    # 3. Comparison across quantization levels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    levels = [s['level'] for s in level_stats]
    means = [s['mean'] for s in level_stats]
    stds = [s['std'] for s in level_stats]
    uniques = [s['unique'] for s in level_stats]
    utilizations = [100.0 * s['unique'] / model.cardinality for s in level_stats]
    
    axes[0, 0].plot(levels, means, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].set_xlabel('Quantization Level')
    axes[0, 0].set_ylabel('Mean Code Value')
    axes[0, 0].set_title('Mean Code Value Across Quantization Levels')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(levels)
    
    axes[0, 1].plot(levels, stds, marker='s', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Quantization Level')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].set_title('Standard Deviation Across Quantization Levels')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(levels)
    
    axes[1, 0].plot(levels, uniques, marker='^', linewidth=2, markersize=8, color='green')
    axes[1, 0].axhline(model.cardinality, color='red', linestyle='--', linewidth=2, label=f'Max ({model.cardinality})')
    axes[1, 0].set_xlabel('Quantization Level')
    axes[1, 0].set_ylabel('Number of Unique Codes')
    axes[1, 0].set_title('Codebook Utilization (Absolute)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(levels)
    
    axes[1, 1].plot(levels, utilizations, marker='D', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Quantization Level')
    axes[1, 1].set_ylabel('Utilization (%)')
    axes[1, 1].set_title('Codebook Utilization (Percentage)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(levels)
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, "quantization_level_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved level comparison to: {comparison_file}")
    
    # 4. Heatmap of code usage across levels
    fig, ax = plt.subplots(figsize=(16, max(8, n_levels * 0.8)))
    
    # Create a matrix showing which codes are used at each level
    max_codes_to_show = 100  # Show top 100 most used codes
    
    # Get all unique codes across all levels
    all_unique_codes = set()
    for stats in level_stats:
        unique_codes = torch.unique(stats['codes'])
        all_unique_codes.update(unique_codes.tolist())
    
    # Create frequency matrix
    code_list = sorted(list(all_unique_codes))[:max_codes_to_show]
    freq_matrix = np.zeros((n_levels, len(code_list)))
    
    for level_idx, stats in enumerate(level_stats):
        codes_data = stats['codes']
        for code_idx, code_val in enumerate(code_list):
            count = (codes_data == code_val).sum().item()
            freq_matrix[level_idx, code_idx] = count
    
    # Normalize by row (level)
    freq_matrix_norm = freq_matrix / (freq_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    im = ax.imshow(freq_matrix_norm, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Code Value (Top 100 Most Used)')
    ax.set_ylabel('Quantization Level')
    ax.set_title('Code Usage Heatmap Across Quantization Levels (Normalized)')
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f'Level {i}' for i in range(n_levels)])
    
    # Only show some x-axis labels to avoid clutter
    num_xticks = min(20, len(code_list))
    xtick_positions = np.linspace(0, len(code_list) - 1, num_xticks, dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(code_list[i]) for i in xtick_positions], rotation=45)
    
    plt.colorbar(im, ax=ax, label='Normalized Frequency')
    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, "code_usage_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap to: {heatmap_file}")
    
    # 5. Entropy analysis across levels
    print("\n--- Entropy Analysis ---")
    entropies = []
    for level_idx, stats in enumerate(level_stats):
        codes_data = stats['codes']
        unique_vals, counts = torch.unique(codes_data, return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
        max_entropy = np.log2(model.cardinality)
        relative_entropy = entropy / max_entropy
        entropies.append(entropy)
        print(f"  Level {level_idx}: Entropy = {entropy:.2f} bits, Relative entropy = {relative_entropy:.2%}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, entropies, marker='o', linewidth=2, markersize=10, color='teal')
    ax.axhline(np.log2(model.cardinality), color='red', linestyle='--', linewidth=2, label=f'Max entropy ({np.log2(model.cardinality):.1f} bits)')
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Information Content Across Quantization Levels')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(levels)
    plt.tight_layout()
    entropy_file = os.path.join(output_dir, "entropy_analysis.png")
    plt.savefig(entropy_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved entropy analysis to: {entropy_file}")

else:
    print("No files were successfully processed.")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
