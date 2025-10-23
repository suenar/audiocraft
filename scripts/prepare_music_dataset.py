#!/usr/bin/env python3
# Run with: python3 scripts/prepare_music_dataset.py [args]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to prepare music dataset for EnCodec finetuning.

This script will:
1. Scan a directory for audio files
2. Optionally split the dataset into train/valid splits
3. Generate manifest files (.jsonl) for AudioCraft

Usage:
    # Basic usage - create manifest from a folder
    python scripts/prepare_music_dataset.py /path/to/music/folder egs/music_finetune

    # With train/valid split (90% train, 10% valid)
    python scripts/prepare_music_dataset.py /path/to/music/folder egs/music_finetune --split 0.9

    # Recursively scan subdirectories
    python scripts/prepare_music_dataset.py /path/to/music/folder egs/music_finetune --recursive

    # Filter by file extensions
    python scripts/prepare_music_dataset.py /path/to/music/folder egs/music_finetune --extensions .mp3 .wav .flac
"""

import argparse
import json
import logging
from pathlib import Path
import random
from typing import List, Optional
import sys

# Add audiocraft to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torchaudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_audio_files(
    root_dir: Path,
    extensions: List[str] = ['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
    recursive: bool = True
) -> List[Path]:
    """
    Get all audio files from a directory.
    
    Args:
        root_dir: Root directory to search
        extensions: List of file extensions to include
        recursive: Whether to search recursively
    
    Returns:
        List of Path objects for audio files
    """
    audio_files = []
    
    if recursive:
        for ext in extensions:
            audio_files.extend(root_dir.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            audio_files.extend(root_dir.glob(f'*{ext}'))
    
    return sorted(audio_files)


def create_manifest_entry(audio_path: Path, root_dir: Optional[Path] = None) -> dict:
    """
    Create a manifest entry for an audio file.
    
    Args:
        audio_path: Path to audio file
        root_dir: Root directory (to create relative paths)
    
    Returns:
        Dictionary with manifest entry
    """
    try:
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        sample_rate = info.sample_rate
        
        # Use relative path if root_dir is provided
        if root_dir:
            try:
                path = audio_path.relative_to(root_dir)
            except ValueError:
                path = audio_path
        else:
            path = audio_path
        
        entry = {
            "path": str(path),
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None,
            "info_path": None
        }
        
        return entry
    
    except Exception as e:
        logger.warning(f"Error processing {audio_path}: {e}")
        return None


def write_manifest(entries: List[dict], output_path: Path):
    """
    Write manifest entries to a .jsonl file.
    
    Args:
        entries: List of manifest entry dictionaries
        output_path: Path to output .jsonl file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare music dataset for EnCodec finetuning')
    parser.add_argument('input_dir', type=str, help='Directory containing audio files')
    parser.add_argument('output_dir', type=str, help='Output directory for manifest files')
    parser.add_argument('--split', type=float, default=None,
                        help='Train split ratio (e.g., 0.9 for 90%% train, 10%% valid). '
                             'If not provided, all data goes to train.')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively scan subdirectories')
    parser.add_argument('--extensions', nargs='+', default=['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
                        help='Audio file extensions to include')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/valid split')
    parser.add_argument('--relative-paths', action='store_true',
                        help='Use relative paths in manifest (relative to input_dir)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Get all audio files
    logger.info(f"Scanning {input_dir} for audio files...")
    audio_files = get_audio_files(input_dir, args.extensions, args.recursive)
    logger.info(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        logger.error("No audio files found!")
        return 1
    
    # Create manifest entries
    logger.info("Creating manifest entries...")
    root_dir = input_dir if args.relative_paths else None
    entries = []
    for audio_file in audio_files:
        entry = create_manifest_entry(audio_file, root_dir)
        if entry is not None:
            entries.append(entry)
    
    logger.info(f"Created {len(entries)} valid manifest entries")
    
    # Split into train/valid if requested
    if args.split is not None:
        random.seed(args.seed)
        random.shuffle(entries)
        
        split_idx = int(len(entries) * args.split)
        train_entries = entries[:split_idx]
        valid_entries = entries[split_idx:]
        
        logger.info(f"Split: {len(train_entries)} train, {len(valid_entries)} valid")
        
        # Write train manifest
        train_path = output_dir / 'train' / 'data.jsonl'
        write_manifest(train_entries, train_path)
        
        # Write valid manifest
        valid_path = output_dir / 'valid' / 'data.jsonl'
        write_manifest(valid_entries, valid_path)
    else:
        # Write all to train
        train_path = output_dir / 'train' / 'data.jsonl'
        write_manifest(entries, train_path)
        
        # Also create valid with same data (for testing)
        valid_path = output_dir / 'valid' / 'data.jsonl'
        write_manifest(entries[:min(100, len(entries))], valid_path)
        logger.info("Created valid split with first 100 samples for testing")
    
    logger.info("Done!")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Update config/dset/audio/music_finetune.yaml to point to {output_dir}")
    logger.info(f"2. Run training: dora grid compression.encodec_finetune_music")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
