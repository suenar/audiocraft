#!/usr/bin/env python3
"""
Simple script to test Encodec compression.
This is a minimal example that you can easily modify.
"""

import torch
import torchaudio
from audiocraft.solvers import CompressionSolver
from pathlib import Path
import sys

# Configuration
CHECKPOINT_PATH = "/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin"
INPUT_AUDIO = "/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav"
OUTPUT_AUDIO = "reconstructed_audio.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Check if files exist
if not Path(CHECKPOINT_PATH).exists():
    print(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
    print(f"Please edit this script and set the correct path to your model checkpoint.")
    sys.exit(1)

if not Path(INPUT_AUDIO).exists():
    print(f"ERROR: Input audio file not found: {INPUT_AUDIO}")
    print(f"Please edit this script and set the correct path to your input audio.")
    sys.exit(1)

# Load the model
print(f"Loading model from: {CHECKPOINT_PATH}")
try:
    model = CompressionSolver.model_from_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    model.eval()
except Exception as e:
    print(f"ERROR loading model: {e}")
    print(f"\nTrying alternative loading method...")
    try:
        from audiocraft.models import CompressionModel
        model = CompressionModel.get_pretrained(CHECKPOINT_PATH, device=DEVICE)
        model.eval()
        print("Model loaded successfully with alternative method!")
    except Exception as e2:
        print(f"ERROR: Could not load model: {e2}")
        sys.exit(1)

print(f"Model info:")
print(f"  Sample rate: {model.sample_rate} Hz")
print(f"  Channels: {model.channels}")
print(f"  Frame rate: {model.frame_rate} Hz")
print(f"  Total codebooks: {model.total_codebooks}")
print(f"  Active codebooks: {model.num_codebooks}")
print(f"  Codebook size: {model.cardinality}")

# Load audio
print(f"\nLoading audio from: {INPUT_AUDIO}")
wav, sr = torchaudio.load(INPUT_AUDIO)

# Resample if needed
if sr != model.sample_rate:
    print(f"Resampling from {sr} Hz to {model.sample_rate} Hz")
    resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
    wav = resampler(wav)
    sr = model.sample_rate

# Convert to mono if needed (model expects mono for 32khz violin model)
if wav.shape[0] > 1 and model.channels == 1:
    print("Converting to mono")
    wav = wav.mean(dim=0, keepdim=True)

# Add batch dimension: [C, T] -> [B, C, T]
wav = wav.unsqueeze(0).to(DEVICE)
print(f"Input audio shape: {wav.shape}")
print(f"Duration: {wav.shape[-1] / sr:.2f} seconds")

# Encode
print("\nEncoding...")
with torch.no_grad():
    codes, scale = model.encode(wav)

print(f"Codes shape: {codes.shape}")  # [B, K, T] where K is number of codebooks
print(f"Number of codebooks used: {codes.shape[1]}")
print(f"Number of frames: {codes.shape[2]}")

# Calculate compression stats
original_size_bits = wav.shape[-1] * wav.shape[1] * 16  # 16-bit audio
compressed_size_bits = codes.shape[1] * codes.shape[2] * torch.log2(torch.tensor(model.cardinality)).item()
compression_ratio = original_size_bits / compressed_size_bits
bandwidth_kbps = compressed_size_bits / (wav.shape[-1] / sr) / 1000

print(f"\nCompression stats:")
print(f"  Compression ratio: {compression_ratio:.2f}x")
print(f"  Bandwidth: {bandwidth_kbps:.2f} kbps")

# Decode
print("\nDecoding...")
with torch.no_grad():
    reconstructed = model.decode(codes, scale)

print(f"Reconstructed audio shape: {reconstructed.shape}")

# Calculate reconstruction quality (SNR)
min_len = min(wav.shape[-1], reconstructed.shape[-1])
original_trim = wav[..., :min_len]
reconstructed_trim = reconstructed[..., :min_len]

noise = original_trim - reconstructed_trim
signal_power = torch.mean(original_trim ** 2)
noise_power = torch.mean(noise ** 2)
snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))

print(f"\nReconstruction quality:")
print(f"  SNR: {snr_db.item():.2f} dB")

# Save reconstructed audio
print(f"\nSaving reconstructed audio to: {OUTPUT_AUDIO}")
reconstructed_cpu = reconstructed.squeeze(0).cpu()
torchaudio.save(OUTPUT_AUDIO, reconstructed_cpu, sr)

# Save codes
codes_file = "audio_codes.pt"
print(f"Saving codes to: {codes_file}")
torch.save({
    'codes': codes.cpu(),
    'scale': scale.cpu() if scale is not None else None,
    'shape': wav.shape,
    'sample_rate': sr
}, codes_file)

print("\n✓ Done!")
print(f"\nYou can now:")
print(f"  1. Listen to the original: {INPUT_AUDIO}")
print(f"  2. Listen to the reconstructed: {OUTPUT_AUDIO}")
print(f"  3. Compare the quality and compression ratio")
