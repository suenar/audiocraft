#!/usr/bin/env python3
"""
MusicGen Finetuning Script

This script provides a convenient wrapper for finetuning MusicGen models
using the audiocraft framework with Dora experiment management.

Usage:
    # Basic finetuning with default settings
    python scripts/finetune_musicgen.py
    
    # Finetune with custom dataset
    python scripts/finetune_musicgen.py --dataset custom_music
    
    # Continue from a specific pretrained model
    python scripts/finetune_musicgen.py --model medium
    
    # Finetune with custom batch size and learning rate
    python scripts/finetune_musicgen.py --batch-size 16 --lr 5e-5
    
    # Resume training from a checkpoint
    python scripts/finetune_musicgen.py --continue-from //sig/SIGNATURE
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Finetune MusicGen model for music generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Finetune the medium model on custom dataset
  python scripts/finetune_musicgen.py --model medium --dataset custom_music
  
  # Finetune with smaller batch size for limited GPU memory
  python scripts/finetune_musicgen.py --model small --batch-size 4
  
  # Continue training from a checkpoint
  python scripts/finetune_musicgen.py --continue-from //sig/abc123def
  
  # Finetune for stereo generation
  python scripts/finetune_musicgen.py --stereo --model medium
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        default='medium',
        choices=['small', 'medium', 'large'],
        help='Pretrained model size to finetune (default: medium)'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset',
        type=str,
        default='custom_music',
        help='Dataset configuration name (default: custom_music)'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size (default: 8)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--segment-duration',
        type=int,
        default=None,
        help='Audio segment duration in seconds (default: 30)'
    )
    
    # Checkpoint and continuation
    parser.add_argument(
        '--continue-from',
        type=str,
        default=None,
        help='Continue training from a checkpoint (e.g., //pretrained/facebook/musicgen-medium or //sig/SIGNATURE)'
    )
    
    # Audio configuration
    parser.add_argument(
        '--stereo',
        action='store_true',
        help='Enable stereo training (default: mono)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=32000,
        choices=[16000, 24000, 32000, 48000],
        help='Audio sample rate (default: 32000)'
    )
    
    # Experiment management
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name'
    )
    
    parser.add_argument(
        '--distributed',
        '-d',
        action='store_true',
        help='Use distributed training (multi-GPU)'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear previous checkpoint and start fresh'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show the command without executing'
    )
    
    args = parser.parse_args()
    
    # Build the dora run command
    cmd = ['dora', 'run']
    
    # Add distributed flag if requested
    if args.distributed:
        cmd.append('-d')
    
    # Add clear flag if requested
    if args.clear:
        cmd.append('--clear')
    
    # Add experiment name if provided
    if args.name:
        cmd.extend(['--name', args.name])
    
    # Base solver configuration
    cmd.append('solver=musicgen/finetune_32khz')
    
    # Model scale
    cmd.append(f'model/lm/model_scale={args.model}')
    
    # Dataset configuration
    cmd.append(f'dset=audio/{args.dataset}')
    
    # Continue from checkpoint
    if args.continue_from:
        continue_from = args.continue_from
    else:
        # Default: continue from pretrained model
        continue_from = f'//pretrained/facebook/musicgen-{args.model}'
    
    cmd.append(f'continue_from={continue_from}')
    
    # Conditioner (important for compatibility)
    cmd.append('conditioner=text2music')
    
    # Audio configuration
    if args.stereo:
        cmd.append('channels=2')
        cmd.append('interleave_stereo_codebooks.use=True')
        cmd.append('transformer_lm.n_q=8')
        cmd.append('codebooks_pattern.delay.delays=[0,0,1,1,2,2,3,3]')
    else:
        cmd.append('channels=1')
    
    cmd.append(f'sample_rate={args.sample_rate}')
    
    # Optional overrides
    if args.batch_size:
        cmd.append(f'dataset.batch_size={args.batch_size}')
    
    if args.lr:
        cmd.append(f'optim.lr={args.lr}')
    
    if args.epochs:
        cmd.append(f'optim.epochs={args.epochs}')
    
    if args.segment_duration:
        cmd.append(f'dataset.segment_duration={args.segment_duration}')
    
    # Print command
    print("=" * 80)
    print("MusicGen Finetuning Command:")
    print("=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    
    if args.dry_run:
        print("\nDry run mode - command not executed")
        return 0
    
    # Execute the command
    print("\nStarting finetuning...")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 130


if __name__ == '__main__':
    sys.exit(main())
