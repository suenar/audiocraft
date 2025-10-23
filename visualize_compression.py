#!/usr/bin/env python3
"""
Visualize compression effects by comparing original and reconstructed audio.
Creates plots showing waveforms, spectrograms, and compression metrics.
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from pathlib import Path
import librosa
import librosa.display


def load_audio(path: str, sr: int = 32000):
    """Load audio file."""
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        wav = resampler(wav)
    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze().numpy(), sr


def plot_waveform(ax, audio, sr, title):
    """Plot waveform."""
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_spectrogram(ax, audio, sr, title):
    """Plot spectrogram."""
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title)
    return img


def plot_difference(ax, original, reconstructed, sr, title):
    """Plot difference between original and reconstructed."""
    diff = original - reconstructed
    time = np.arange(len(diff)) / sr
    ax.plot(time, diff, linewidth=0.5, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude Difference')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_codes_heatmap(ax, codes, title):
    """Plot discrete codes as a heatmap."""
    # codes shape: [num_codebooks, num_frames]
    im = ax.imshow(codes, aspect='auto', cmap='tab20', interpolation='nearest')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Codebook')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Code Index')


def visualize_compression(
    original_path: str,
    reconstructed_path: str,
    codes_path: str = None,
    output_path: str = "compression_visualization.png",
    sample_rate: int = 32000
):
    """Create comprehensive visualization of compression effects."""
    
    # Load audio files
    print(f"Loading original audio from: {original_path}")
    original, sr = load_audio(original_path, sample_rate)
    
    print(f"Loading reconstructed audio from: {reconstructed_path}")
    reconstructed, _ = load_audio(reconstructed_path, sample_rate)
    
    # Make same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Load codes if available
    codes = None
    if codes_path and Path(codes_path).exists():
        print(f"Loading codes from: {codes_path}")
        data = torch.load(codes_path)
        codes = data['codes'].squeeze(0).numpy()  # Remove batch dimension
    
    # Calculate metrics
    noise = original - reconstructed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    
    # Create figure
    if codes is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Row 1: Waveforms
    ax1 = fig.add_subplot(gs[0, 0])
    plot_waveform(ax1, original, sr, 'Original Waveform')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_waveform(ax2, reconstructed, sr, 'Reconstructed Waveform')
    
    # Row 2: Spectrograms
    ax3 = fig.add_subplot(gs[1, 0])
    plot_spectrogram(ax3, original, sr, 'Original Spectrogram')
    
    ax4 = fig.add_subplot(gs[1, 1])
    img = plot_spectrogram(ax4, reconstructed, sr, 'Reconstructed Spectrogram')
    
    # Add colorbar for spectrograms
    cbar = plt.colorbar(img, ax=[ax3, ax4], location='right', pad=0.01)
    cbar.set_label('Magnitude (dB)')
    
    # Row 3: Difference and metrics
    ax5 = fig.add_subplot(gs[2, 0])
    plot_difference(ax5, original, reconstructed, sr, f'Difference (SNR: {snr_db:.2f} dB)')
    
    ax6 = fig.add_subplot(gs[2, 1])
    # Plot difference spectrogram
    diff_spec = librosa.stft(noise)
    diff_db = librosa.amplitude_to_db(np.abs(diff_spec), ref=np.max)
    librosa.display.specshow(diff_db, sr=sr, x_axis='time', y_axis='hz', ax=ax6, cmap='coolwarm')
    ax6.set_title('Difference Spectrogram')
    
    # Row 4: Codes heatmap (if available)
    if codes is not None:
        ax7 = fig.add_subplot(gs[3, :])
        plot_codes_heatmap(ax7, codes, f'Discrete Codes (Shape: {codes.shape})')
    
    # Add overall title with metrics
    duration = len(original) / sr
    fig.suptitle(
        f'Compression Analysis\n'
        f'Duration: {duration:.2f}s | SNR: {snr_db:.2f} dB | RMSE: {rmse:.6f}',
        fontsize=14, fontweight='bold'
    )
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving visualization to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("✓ Visualization saved!")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
    print(f"Root Mean Square Error: {rmse:.6f}")
    print(f"Mean Squared Error: {mse:.8f}")
    if codes is not None:
        print(f"\nCodes shape: {codes.shape}")
        print(f"  - Number of codebooks: {codes.shape[0]}")
        print(f"  - Number of frames: {codes.shape[1]}")
        print(f"  - Unique codes per codebook: {[len(np.unique(codes[i])) for i in range(codes.shape[0])]}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize compression effects',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original audio file'
    )
    parser.add_argument(
        '--reconstructed',
        type=str,
        required=True,
        help='Path to reconstructed audio file'
    )
    parser.add_argument(
        '--codes',
        type=str,
        default=None,
        help='Path to codes file (.pt) - optional'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='compression_visualization.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=32000,
        help='Sample rate to use'
    )
    
    args = parser.parse_args()
    
    visualize_compression(
        original_path=args.original,
        reconstructed_path=args.reconstructed,
        codes_path=args.codes,
        output_path=args.output,
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()
