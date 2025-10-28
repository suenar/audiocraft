#!/usr/bin/env python3
"""
Example MusicGen Finetuning Usage

This script demonstrates how to use the finetuned MusicGen model for music generation.
"""

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path


def generate_music_from_pretrained():
    """Example: Generate music using a pretrained MusicGen model."""
    print("=" * 80)
    print("Example 1: Generate music with pretrained model")
    print("=" * 80)
    
    # Load pretrained model
    print("Loading model...")
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    
    # Set generation parameters
    model.set_generation_params(
        duration=10,        # 10 seconds
        temperature=1.0,    # Sampling temperature
        top_k=250,          # Top-k sampling
        cfg_coef=3.0        # Classifier-free guidance
    )
    
    # Generate music with text prompts
    descriptions = [
        'A happy upbeat electronic dance music track',
        'A calm and relaxing acoustic guitar melody',
        'An energetic rock song with electric guitars and drums'
    ]
    
    print(f"Generating {len(descriptions)} music samples...")
    with torch.no_grad():
        wav = model.generate(descriptions)
    
    # Save generated audio
    output_dir = Path('generated_samples')
    output_dir.mkdir(exist_ok=True)
    
    for idx, one_wav in enumerate(wav):
        output_path = output_dir / f'pretrained_{idx}'
        audio_write(
            str(output_path),
            one_wav.cpu(),
            model.sample_rate,
            strategy='loudness',
            loudness_compressor=True
        )
        print(f"✓ Saved: {output_path}.wav - {descriptions[idx]}")
    
    print(f"\nGenerated samples saved to: {output_dir}/")


def generate_music_from_finetuned(checkpoint_path):
    """Example: Generate music using your finetuned model."""
    print("\n" + "=" * 80)
    print("Example 2: Generate music with finetuned model")
    print("=" * 80)
    
    # Load your finetuned model
    print(f"Loading finetuned model from: {checkpoint_path}")
    model = MusicGen.get_pretrained(checkpoint_path)
    
    # Set generation parameters
    model.set_generation_params(
        duration=15,        # 15 seconds
        temperature=1.0,
        top_k=250,
        cfg_coef=3.0
    )
    
    # Generate with custom prompts
    descriptions = [
        'Your custom music style description here',
    ]
    
    print(f"Generating music...")
    with torch.no_grad():
        wav = model.generate(descriptions)
    
    # Save
    output_dir = Path('generated_samples')
    output_dir.mkdir(exist_ok=True)
    
    for idx, one_wav in enumerate(wav):
        output_path = output_dir / f'finetuned_{idx}'
        audio_write(
            str(output_path),
            one_wav.cpu(),
            model.sample_rate,
            strategy='loudness',
            loudness_compressor=True
        )
        print(f"✓ Saved: {output_path}.wav")


def unconditional_generation():
    """Example: Generate music without text conditioning (unconditional)."""
    print("\n" + "=" * 80)
    print("Example 3: Unconditional music generation")
    print("=" * 80)
    
    # Load model
    print("Loading model...")
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    model.set_generation_params(duration=8)
    
    # Generate unconditional samples
    print("Generating unconditional samples...")
    with torch.no_grad():
        wav = model.generate_unconditional(num_samples=3)
    
    # Save
    output_dir = Path('generated_samples')
    output_dir.mkdir(exist_ok=True)
    
    for idx, one_wav in enumerate(wav):
        output_path = output_dir / f'unconditional_{idx}'
        audio_write(
            str(output_path),
            one_wav.cpu(),
            model.sample_rate,
            strategy='loudness',
            loudness_compressor=True
        )
        print(f"✓ Saved: {output_path}.wav")


def melody_guided_generation():
    """Example: Generate music guided by a melody."""
    print("\n" + "=" * 80)
    print("Example 4: Melody-guided generation")
    print("=" * 80)
    
    # This requires the melody model
    print("Loading melody model...")
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=10)
    
    # Load a reference melody (e.g., from your dataset)
    import torchaudio
    melody_path = 'assets/bach.mp3'  # Example melody file
    
    if Path(melody_path).exists():
        print(f"Loading melody from: {melody_path}")
        melody, sr = torchaudio.load(melody_path)
        
        # Generate music based on the melody + text
        descriptions = ['An electronic remix']
        
        print("Generating melody-guided music...")
        with torch.no_grad():
            wav = model.generate_with_chroma(
                descriptions,
                melody[None].expand(len(descriptions), -1, -1),
                sr
            )
        
        # Save
        output_dir = Path('generated_samples')
        output_dir.mkdir(exist_ok=True)
        
        for idx, one_wav in enumerate(wav):
            output_path = output_dir / f'melody_guided_{idx}'
            audio_write(
                str(output_path),
                one_wav.cpu(),
                model.sample_rate,
                strategy='loudness',
                loudness_compressor=True
            )
            print(f"✓ Saved: {output_path}.wav")
    else:
        print(f"⚠ Melody file not found: {melody_path}")
        print("  Skipping melody-guided generation example")


def load_from_checkpoint_example():
    """Example: Load model from a training checkpoint."""
    print("\n" + "=" * 80)
    print("Example 5: Load from training checkpoint")
    print("=" * 80)
    
    # This shows how to load from a checkpoint during or after training
    example_sig = "abc123def"  # Replace with your experiment signature
    checkpoint_path = f"./experiments/audiocraft/outputs/{example_sig}"
    
    print(f"To load your finetuned model:")
    print(f"  model = MusicGen.get_pretrained('{checkpoint_path}')")
    print()
    print("Or export it first:")
    print(f"""
from audiocraft.utils import export
from audiocraft import train

xp = train.main.get_xp_from_sig('{example_sig}')
export.export_lm(
    xp.folder / 'checkpoint.th',
    '/my/models/my_musicgen/state_dict.bin'
)
export.export_pretrained_compression_model(
    'facebook/encodec_32khz',
    '/my/models/my_musicgen/compression_state_dict.bin'
)

# Then load:
model = MusicGen.get_pretrained('/my/models/my_musicgen')
    """)


def main():
    """Run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MusicGen example usage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific example (1-5)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to finetuned model checkpoint (for example 2)'
    )
    
    args = parser.parse_args()
    
    print("MusicGen Usage Examples")
    print("This script demonstrates various ways to use MusicGen")
    print()
    
    if not args.example:
        print("Available examples:")
        print("  1. Generate with pretrained model")
        print("  2. Generate with finetuned model (requires --checkpoint)")
        print("  3. Unconditional generation")
        print("  4. Melody-guided generation")
        print("  5. Load from checkpoint (reference)")
        print()
        print("Usage: python example_usage.py --example 1")
        print("       python example_usage.py --example 2 --checkpoint /path/to/checkpoint")
        return
    
    try:
        if args.example == 1:
            generate_music_from_pretrained()
        elif args.example == 2:
            if not args.checkpoint:
                print("Error: --checkpoint required for example 2")
                print("Usage: python example_usage.py --example 2 --checkpoint /path/to/checkpoint")
                return
            generate_music_from_finetuned(args.checkpoint)
        elif args.example == 3:
            unconditional_generation()
        elif args.example == 4:
            melody_guided_generation()
        elif args.example == 5:
            load_from_checkpoint_example()
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all dependencies (run install_dependencies.sh)")
        print("  2. A CUDA-capable GPU (for faster generation)")
        print("  3. Internet connection (to download pretrained models)")


if __name__ == '__main__':
    main()
