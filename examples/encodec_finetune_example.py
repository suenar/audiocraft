#!/usr/bin/env python3
# Run with: python3 examples/encodec_finetune_example.py
"""
Example script demonstrating how to finetune and use a custom EnCodec model.

This script shows:
1. How to prepare a small dataset
2. How to finetune an EnCodec model programmatically
3. How to load and use the finetuned model
4. How to evaluate reconstruction quality
"""

import torch
import torchaudio
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset_example():
    """Example of preparing a dataset programmatically."""
    logger.info("Preparing example dataset...")
    
    # Use the assets folder as a simple example
    assets_dir = Path('assets')
    
    if not assets_dir.exists():
        logger.warning(f"Assets directory not found: {assets_dir}")
        return False
    
    # Run the preparation script
    import subprocess
    result = subprocess.run([
        'python', 'scripts/prepare_music_dataset.py',
        str(assets_dir),
        'egs/music_finetune_example',
        '--split', '0.8'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Dataset preparation failed: {result.stderr}")
        return False
    
    logger.info("Dataset prepared successfully!")
    return True


def train_encodec_programmatic():
    """Example of training EnCodec programmatically (for advanced users)."""
    logger.info("This example shows how to train EnCodec programmatically...")
    
    # For most users, using dora is recommended:
    # dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune
    
    # Advanced: Direct solver usage
    try:
        from audiocraft import train
        from pathlib import Path
        import os
        
        # Change to audiocraft root
        os.chdir(Path(train.__file__).parent.parent)
        
        # Note: This is advanced usage. Most users should use dora run instead.
        logger.info("For training, use: dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune")
        
    except Exception as e:
        logger.error(f"Training setup failed: {e}")
        return False
    
    return True


def load_and_test_model(checkpoint_path: str = '//pretrained/facebook/encodec_32khz'):
    """Example of loading and testing an EnCodec model."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    from audiocraft.solvers import CompressionSolver
    
    # Load the model
    model = CompressionSolver.model_from_checkpoint(checkpoint_path)
    model.eval()
    
    logger.info(f"Model loaded successfully!")
    logger.info(f"  Sample rate: {model.sample_rate} Hz")
    logger.info(f"  Channels: {model.channels}")
    logger.info(f"  Codebooks: {model.num_codebooks}")
    logger.info(f"  Frame rate: {model.frame_rate} Hz")
    
    return model


def encode_decode_example(model, audio_path: str = 'assets/bach.mp3'):
    """Example of encoding and decoding audio."""
    logger.info(f"Testing encode/decode with: {audio_path}")
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != model.sample_rate:
        logger.info(f"Resampling from {sr} Hz to {model.sample_rate} Hz")
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    
    # Ensure correct number of channels
    if wav.shape[0] != model.channels:
        if model.channels == 1 and wav.shape[0] == 2:
            # Convert stereo to mono
            wav = wav.mean(dim=0, keepdim=True)
        elif model.channels == 2 and wav.shape[0] == 1:
            # Convert mono to stereo
            wav = wav.repeat(2, 1)
    
    # Add batch dimension: [B, C, T]
    wav = wav.unsqueeze(0)
    
    # Encode and decode
    with torch.no_grad():
        logger.info("Encoding audio...")
        codes, scale = model.encode(wav)
        logger.info(f"  Codes shape: {codes.shape}")
        logger.info(f"  Codebooks used: {codes.shape[1]}")
        logger.info(f"  Time steps: {codes.shape[2]}")
        
        logger.info("Decoding audio...")
        reconstructed = model.decode(codes, scale)
        logger.info(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    mse = torch.mean((wav - reconstructed[..., :wav.shape[-1]]) ** 2)
    logger.info(f"  Reconstruction MSE: {mse.item():.6f}")
    
    # Save reconstructed audio
    output_path = Path('outputs/encodec_reconstruction.wav')
    output_path.parent.mkdir(exist_ok=True)
    torchaudio.save(
        str(output_path),
        reconstructed[0, :, :wav.shape[-1]].cpu(),
        model.sample_rate
    )
    logger.info(f"Saved reconstructed audio to: {output_path}")
    
    return reconstructed


def compare_models(original_model, finetuned_model, audio_path: str):
    """Compare reconstruction quality between two models."""
    logger.info("Comparing models...")
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != original_model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, original_model.sample_rate)
    if wav.shape[0] != original_model.channels:
        wav = wav.mean(dim=0, keepdim=True) if original_model.channels == 1 else wav.repeat(2, 1)
    wav = wav.unsqueeze(0)
    
    # Test original model
    with torch.no_grad():
        codes1, scale1 = original_model.encode(wav)
        recon1 = original_model.decode(codes1, scale1)
        mse1 = torch.mean((wav - recon1[..., :wav.shape[-1]]) ** 2)
    
    # Test finetuned model
    with torch.no_grad():
        codes2, scale2 = finetuned_model.encode(wav)
        recon2 = finetuned_model.decode(codes2, scale2)
        mse2 = torch.mean((wav - recon2[..., :wav.shape[-1]]) ** 2)
    
    logger.info(f"Original model MSE: {mse1.item():.6f}")
    logger.info(f"Finetuned model MSE: {mse2.item():.6f}")
    logger.info(f"Improvement: {((mse1 - mse2) / mse1 * 100).item():.2f}%")


def main():
    """Main example function."""
    logger.info("=" * 60)
    logger.info("EnCodec Finetuning Example")
    logger.info("=" * 60)
    
    # Example 1: Load pretrained model
    logger.info("\n1. Loading pretrained EnCodec model...")
    model = load_and_test_model()
    
    # Example 2: Test encoding/decoding
    logger.info("\n2. Testing encode/decode...")
    encode_decode_example(model)
    
    # Example 3: Show how to prepare dataset
    logger.info("\n3. Dataset preparation example...")
    logger.info("To prepare your dataset, run:")
    logger.info("  python scripts/prepare_music_dataset.py /path/to/music egs/music_finetune --split 0.9")
    
    # Example 4: Show training command
    logger.info("\n4. Training example...")
    logger.info("To finetune the model, run:")
    logger.info("  dora run solver=compression/encodec_musicgen_32khz dset=audio/music_finetune \\")
    logger.info("      continue_from=//pretrained/facebook/encodec_32khz \\")
    logger.info("      optim.lr=1e-4 dataset.batch_size=32")
    
    # Example 5: Show how to load finetuned model
    logger.info("\n5. Loading finetuned model...")
    logger.info("After training, load your model with:")
    logger.info("  model = CompressionSolver.model_from_checkpoint('//sig/<YOUR_SIGNATURE>')")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Prepare your music dataset")
    logger.info("2. Run the quick start script: ./scripts/finetune_encodec_quickstart.sh /path/to/music")
    logger.info("3. Monitor training and evaluate results")
    logger.info("\nFor more details, see ENCODEC_FINETUNING.md")


if __name__ == '__main__':
    main()
