#!/usr/bin/env python3
"""
Music Dataset Preparation Script

This script helps prepare your music dataset for finetuning MusicGen.
It creates the necessary manifest files (JSONL format) and metadata (JSON files).

Usage:
    # Create manifest from a directory of audio files
    python scripts/prepare_music_dataset.py /path/to/music/folder --output-dir egs/custom_music
    
    # With metadata CSV file
    python scripts/prepare_music_dataset.py /path/to/music/folder --metadata metadata.csv --output-dir egs/custom_music
    
    # Split into train/validation sets
    python scripts/prepare_music_dataset.py /path/to/music/folder --split 0.9 --output-dir egs/custom_music
"""

import argparse
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional
import csv


def create_manifest(
    audio_dir: Path,
    output_dir: Path,
    split_name: str = "train",
    metadata_dict: Optional[Dict[str, Dict]] = None
):
    """Create a JSONL manifest file for the dataset."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"data.jsonl"
    
    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.rglob(f'*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create manifest entries
    entries = []
    for audio_file in audio_files:
        # Get relative path from workspace root
        try:
            rel_path = audio_file.relative_to(Path.cwd())
        except ValueError:
            rel_path = audio_file
        
        # Get audio duration using librosa or torchaudio
        try:
            import torchaudio
            info = torchaudio.info(str(audio_file))
            duration = info.num_frames / info.sample_rate
            sample_rate = info.sample_rate
        except Exception as e:
            print(f"Warning: Could not get info for {audio_file}: {e}")
            duration = 0.0
            sample_rate = 44100
        
        entry = {
            "path": str(rel_path),
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None,
            "info_path": None
        }
        entries.append(entry)
        
        # Create metadata JSON file if metadata is provided
        if metadata_dict and audio_file.stem in metadata_dict:
            metadata = metadata_dict[audio_file.stem]
            json_path = audio_file.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Created metadata: {json_path}")
    
    # Write manifest file
    with open(manifest_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest: {manifest_path} with {len(entries)} entries")
    return manifest_path


def load_metadata_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load metadata from a CSV file.
    
    Expected CSV format:
    filename,description,genre,bpm,mood,artist,title
    song1,A happy upbeat song,pop,120,happy,Artist Name,Song Title
    """
    metadata_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.pop('filename', None)
            if filename:
                # Remove extension if present
                filename = Path(filename).stem
                metadata_dict[filename] = dict(row)
    
    return metadata_dict


def main():
    parser = argparse.ArgumentParser(
        description="Prepare music dataset for MusicGen finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Basic usage
  python scripts/prepare_music_dataset.py /data/music --output-dir egs/my_music
  
  # With metadata CSV
  python scripts/prepare_music_dataset.py /data/music \\
      --metadata /data/music/metadata.csv \\
      --output-dir egs/my_music
  
  # Split into train/val with 90% train, 10% validation
  python scripts/prepare_music_dataset.py /data/music \\
      --split 0.9 \\
      --output-dir egs/my_music

Metadata CSV Format:
  filename,description,genre,bpm,mood,artist,title
  song1.mp3,"Happy upbeat electronic music",electronic,128,happy,Artist1,Song1
  song2.mp3,"Calm acoustic guitar",acoustic,90,relaxing,Artist2,Song2
        """
    )
    
    parser.add_argument(
        'audio_dir',
        type=Path,
        help='Directory containing audio files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('egs/custom_music'),
        help='Output directory for manifest files (default: egs/custom_music)'
    )
    
    parser.add_argument(
        '--metadata',
        type=Path,
        help='Path to metadata CSV file (optional)'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        default=None,
        help='Train/validation split ratio (e.g., 0.9 for 90%% train, 10%% val)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Load metadata if provided
    metadata_dict = None
    if args.metadata:
        print(f"Loading metadata from {args.metadata}")
        metadata_dict = load_metadata_csv(args.metadata)
        print(f"Loaded metadata for {len(metadata_dict)} files")
    
    # Create manifest
    if args.split:
        # Create train/validation split
        import random
        random.seed(args.seed)
        
        # Get all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(args.audio_dir.rglob(f'*{ext}'))
        
        # Shuffle and split
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * args.split)
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]
        
        print(f"Split: {len(train_files)} train, {len(val_files)} validation")
        
        # Create temporary directories for split
        train_dir = args.output_dir / 'train_tmp'
        val_dir = args.output_dir / 'valid_tmp'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks or copy files
        for f in train_files:
            (train_dir / f.name).symlink_to(f.absolute())
        for f in val_files:
            (val_dir / f.name).symlink_to(f.absolute())
        
        # Create manifests
        train_output = args.output_dir / 'train'
        val_output = args.output_dir / 'valid'
        
        create_manifest(train_dir, train_output, 'train', metadata_dict)
        create_manifest(val_dir, val_output, 'valid', metadata_dict)
        
        # Cleanup temporary directories
        for f in train_dir.iterdir():
            f.unlink()
        for f in val_dir.iterdir():
            f.unlink()
        train_dir.rmdir()
        val_dir.rmdir()
    else:
        # Create single manifest
        create_manifest(args.audio_dir, args.output_dir / 'train', 'train', metadata_dict)
        
        # Create symlink for validation
        val_dir = args.output_dir / 'valid'
        val_dir.mkdir(parents=True, exist_ok=True)
        val_manifest = val_dir / 'data.jsonl'
        if val_manifest.exists():
            val_manifest.unlink()
        val_manifest.symlink_to((args.output_dir / 'train' / 'data.jsonl').absolute())
        print(f"Created validation manifest (symlink to train)")
    
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)
    print(f"\nTo use this dataset, update your config:")
    print(f"  dset=audio/custom_music")
    print(f"\nOr edit: config/dset/audio/custom_music.yaml")
    print(f"  train: {args.output_dir / 'train'}")
    print(f"  valid: {args.output_dir / 'valid'}")


if __name__ == '__main__':
    main()
