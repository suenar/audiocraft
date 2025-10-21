#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Script to test the compression effect of a fine-tuned Encodec model.
This script loads an Encodec checkpoint, compresses and decompresses an audio file,
and provides metrics about the compression quality and efficiency.
"""

import argparse
import logging
import sys
from pathlib import Path
import typing as tp

import torch
import torchaudio

from audiocraft.models import CompressionModel
from audiocraft.solvers import CompressionSolver


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio(audio_path: Path, target_sr: int = 32000) -> tp.Tuple[torch.Tensor, int]:
    """Load audio file and resample if necessary.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default: 32000)
    
    Returns:
        Tuple of (audio tensor [1, C, T], sample rate)
    """
    logger.info(f"Loading audio from: {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        logger.info(f"Resampling from {sr} Hz to {target_sr} Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
    
    # Add batch dimension
    wav = wav.unsqueeze(0)  # [1, C, T]
    
    logger.info(f"Audio shape: {wav.shape}, Sample rate: {sr} Hz, Duration: {wav.shape[-1] / sr:.2f}s")
    return wav, sr


def compute_compression_metrics(
    model: CompressionModel,
    codes: torch.Tensor,
    original_audio: torch.Tensor,
    sample_rate: int
) -> dict:
    """Compute various compression metrics.
    
    Args:
        model: The compression model
        codes: Encoded discrete codes [B, K, T]
        original_audio: Original audio tensor [B, C, T]
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary of metrics
    """
    B, K, T_codes = codes.shape
    _, C, T_audio = original_audio.shape
    
    duration = T_audio / sample_rate
    
    # Calculate compression ratio
    original_bits = T_audio * C * 16  # Assuming 16-bit audio
    
    # Each code is from a codebook with 'cardinality' entries
    bits_per_code = torch.log2(torch.tensor(model.cardinality)).item()
    compressed_bits = K * T_codes * bits_per_code
    
    compression_ratio = original_bits / compressed_bits
    
    # Calculate bandwidth
    bandwidth_kbps = (compressed_bits / duration) / 1000
    
    # Frame rate
    frame_rate = model.frame_rate
    
    metrics = {
        'num_codebooks': K,
        'num_frames': T_codes,
        'codebook_cardinality': model.cardinality,
        'frame_rate': frame_rate,
        'compression_ratio': compression_ratio,
        'bandwidth_kbps': bandwidth_kbps,
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'duration_seconds': duration,
    }
    
    return metrics


def compute_reconstruction_quality(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> dict:
    """Compute reconstruction quality metrics.
    
    Args:
        original: Original audio [B, C, T]
        reconstructed: Reconstructed audio [B, C, T]
    
    Returns:
        Dictionary of quality metrics
    """
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]
    
    # Signal-to-Noise Ratio (SNR)
    noise = original - reconstructed
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # Mean Squared Error (MSE)
    mse = torch.mean((original - reconstructed) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = torch.max(torch.abs(original))
    psnr_db = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    
    metrics = {
        'snr_db': snr_db.item(),
        'psnr_db': psnr_db.item(),
        'mse': mse.item(),
    }
    
    return metrics


def test_compression(
    checkpoint_path: str,
    audio_path: str,
    output_path: tp.Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_codebooks: tp.Optional[int] = None
) -> dict:
    """Test compression on an audio file.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.bin file)
        audio_path: Path to the input audio file
        output_path: Path to save the reconstructed audio (optional)
        device: Device to use for inference
        num_codebooks: Number of codebooks to use (optional, uses max by default)
    
    Returns:
        Dictionary containing all metrics and results
    """
    logger.info(f"Using device: {device}")
    
    # Load the compression model
    logger.info(f"Loading compression model from: {checkpoint_path}")
    try:
        # Try loading as a CompressionSolver checkpoint
        model = CompressionSolver.model_from_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        logger.warning(f"Failed to load with CompressionSolver: {e}")
        logger.info("Trying to load as a direct checkpoint...")
        # Try loading as a direct model checkpoint
        model = CompressionModel.get_pretrained(checkpoint_path, device=device)
    
    model.eval()
    logger.info(f"Model loaded successfully!")
    logger.info(f"Model properties:")
    logger.info(f"  - Sample rate: {model.sample_rate} Hz")
    logger.info(f"  - Channels: {model.channels}")
    logger.info(f"  - Frame rate: {model.frame_rate} Hz")
    logger.info(f"  - Total codebooks: {model.total_codebooks}")
    logger.info(f"  - Codebook cardinality: {model.cardinality}")
    
    # Set number of codebooks if specified
    if num_codebooks is not None:
        logger.info(f"Setting number of codebooks to: {num_codebooks}")
        model.set_num_codebooks(num_codebooks)
    else:
        logger.info(f"Using default number of codebooks: {model.num_codebooks}")
    
    # Load audio
    audio, sr = load_audio(Path(audio_path), target_sr=model.sample_rate)
    
    # Handle channels
    if audio.shape[1] != model.channels:
        if model.channels == 1 and audio.shape[1] == 2:
            logger.info("Converting stereo to mono")
            audio = audio.mean(dim=1, keepdim=True)
        elif model.channels == 2 and audio.shape[1] == 1:
            logger.info("Converting mono to stereo")
            audio = audio.repeat(1, 2, 1)
    
    audio = audio.to(device)
    
    # Encode
    logger.info("Encoding audio...")
    with torch.no_grad():
        codes, scale = model.encode(audio)
    
    logger.info(f"Encoded codes shape: {codes.shape}")  # [B, K, T]
    logger.info(f"Scale: {scale.shape if scale is not None else None}")
    
    # Compute compression metrics
    compression_metrics = compute_compression_metrics(model, codes, audio, sr)
    
    # Decode
    logger.info("Decoding audio...")
    with torch.no_grad():
        reconstructed = model.decode(codes, scale)
    
    logger.info(f"Reconstructed audio shape: {reconstructed.shape}")
    
    # Compute reconstruction quality metrics
    quality_metrics = compute_reconstruction_quality(audio, reconstructed)
    
    # Combine all metrics
    all_metrics = {
        **compression_metrics,
        **quality_metrics,
    }
    
    # Save reconstructed audio if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving reconstructed audio to: {output_path}")
        # Move to CPU and remove batch dimension
        reconstructed_cpu = reconstructed.squeeze(0).cpu()
        torchaudio.save(str(output_path), reconstructed_cpu, sr)
        
        # Also save the codes for inspection
        codes_path = output_path.with_suffix('.codes.pt')
        logger.info(f"Saving codes to: {codes_path}")
        torch.save({
            'codes': codes.cpu(),
            'scale': scale.cpu() if scale is not None else None,
            'metadata': {
                'num_codebooks': model.num_codebooks,
                'frame_rate': model.frame_rate,
                'sample_rate': model.sample_rate,
            }
        }, codes_path)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test Encodec compression model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.bin file)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input audio file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the reconstructed audio (if not provided, will not save)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference (cuda or cpu)'
    )
    parser.add_argument(
        '--num-codebooks',
        type=int,
        default=None,
        help='Number of codebooks to use (default: use maximum available)'
    )
    
    args = parser.parse_args()
    
    # Run compression test
    metrics = test_compression(
        checkpoint_path=args.checkpoint,
        audio_path=args.input,
        output_path=args.output,
        device=args.device,
        num_codebooks=args.num_codebooks
    )
    
    # Print results
    print("\n" + "="*80)
    print("COMPRESSION TEST RESULTS")
    print("="*80)
    print("\nCompression Metrics:")
    print(f"  Number of codebooks: {metrics['num_codebooks']}")
    print(f"  Codebook cardinality: {metrics['codebook_cardinality']}")
    print(f"  Frame rate: {metrics['frame_rate']:.2f} Hz")
    print(f"  Number of frames: {metrics['num_frames']}")
    print(f"  Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  Bandwidth: {metrics['bandwidth_kbps']:.2f} kbps")
    print(f"  Original size: {metrics['original_bits'] / 8 / 1024:.2f} KB")
    print(f"  Compressed size: {metrics['compressed_bits'] / 8 / 1024:.2f} KB")
    
    print("\nReconstruction Quality Metrics:")
    print(f"  Signal-to-Noise Ratio (SNR): {metrics['snr_db']:.2f} dB")
    print(f"  Peak SNR (PSNR): {metrics['psnr_db']:.2f} dB")
    print(f"  Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
