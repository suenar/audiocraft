#!/usr/bin/env python3
"""
Script to split audio files into train/validation/test sets based on composer and opus number.
Ensures that each piece (composer + opus) appears only in one set.
Split ratio: 80% train, 10% validation, 10% test
Converts mp3 files to wav format.
"""

import os
import shutil
from collections import defaultdict
from pathlib import Path
import random
import subprocess
import sys


def parse_filename(filename):
    """
    Parse filename to extract composer and opus number.
    Format: Composer_OpXX-XX_PerformanceInfo.mp3
    
    Returns: (composer, opus, full_filename) or None if parsing fails
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        composer = parts[0]
        opus = parts[1]
        return (composer, opus, filename)
    return None


def group_files_by_piece(directory='.'):
    """
    Group all mp3 files by their piece (composer + opus number).
    
    Returns: dict mapping (composer, opus) -> list of filenames
    """
    pieces = defaultdict(list)
    
    # Find all mp3 files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            parsed = parse_filename(filename)
            if parsed:
                composer, opus, full_filename = parsed
                piece_key = (composer, opus)
                pieces[piece_key].append(full_filename)
    
    return pieces


def split_pieces(pieces, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split pieces into train, validation, and test sets.
    
    Args:
        pieces: dict mapping (composer, opus) -> list of filenames
        train_ratio: proportion for training set (default 0.8)
        val_ratio: proportion for validation set (default 0.1)
        test_ratio: proportion for test set (default 0.1)
    
    Returns: tuple of (train_files, val_files, test_files)
    """
    # Get all unique pieces
    piece_keys = list(pieces.keys())
    random.shuffle(piece_keys)  # Shuffle for random split
    
    total_pieces = len(piece_keys)
    train_count = int(total_pieces * train_ratio)
    val_count = int(total_pieces * val_ratio)
    
    # Split piece keys
    train_pieces = piece_keys[:train_count]
    val_pieces = piece_keys[train_count:train_count + val_count]
    test_pieces = piece_keys[train_count + val_count:]
    
    # Collect all files for each set
    train_files = []
    val_files = []
    test_files = []
    
    for piece_key in train_pieces:
        train_files.extend(pieces[piece_key])
    
    for piece_key in val_pieces:
        val_files.extend(pieces[piece_key])
    
    for piece_key in test_pieces:
        test_files.extend(pieces[piece_key])
    
    return train_files, val_files, test_files


def convert_mp3_to_wav(input_path, output_path):
    """
    Convert mp3 file to wav format using ffmpeg.
    
    Args:
        input_path: path to input mp3 file
        output_path: path to output wav file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use ffmpeg to convert mp3 to wav
        subprocess.run(
            ['ffmpeg', '-i', str(input_path), '-acodec', 'pcm_s16le', '-ar', '44100', str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg to convert audio files.")
        print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
        sys.exit(1)


def organize_files(train_files, val_files, test_files, source_dir='.', convert_to_wav=True):
    """
    Organize files into train/val/test directories and optionally convert to wav.
    
    Args:
        train_files: list of filenames for training set
        val_files: list of filenames for validation set
        test_files: list of filenames for test set
        source_dir: source directory containing the files
        convert_to_wav: if True, convert mp3 to wav format
    """
    # Create directories
    train_dir = Path('train')
    val_dir = Path('val')
    test_dir = Path('test')
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Organize training files
    print(f"Organizing {len(train_files)} files into train directory...")
    for i, filename in enumerate(train_files, 1):
        src = Path(source_dir) / filename
        if src.exists():
            if convert_to_wav:
                dst = train_dir / filename.replace('.mp3', '.wav')
                if i % 10 == 0 or i == len(train_files):
                    print(f"  Converting train files: {i}/{len(train_files)}")
                convert_mp3_to_wav(src, dst)
            else:
                dst = train_dir / filename
                shutil.copy2(str(src), str(dst))
    
    # Organize validation files
    print(f"Organizing {len(val_files)} files into val directory...")
    for i, filename in enumerate(val_files, 1):
        src = Path(source_dir) / filename
        if src.exists():
            if convert_to_wav:
                dst = val_dir / filename.replace('.mp3', '.wav')
                if i % 10 == 0 or i == len(val_files):
                    print(f"  Converting val files: {i}/{len(val_files)}")
                convert_mp3_to_wav(src, dst)
            else:
                dst = val_dir / filename
                shutil.copy2(str(src), str(dst))
    
    # Organize test files
    print(f"Organizing {len(test_files)} files into test directory...")
    for i, filename in enumerate(test_files, 1):
        src = Path(source_dir) / filename
        if src.exists():
            if convert_to_wav:
                dst = test_dir / filename.replace('.mp3', '.wav')
                if i % 10 == 0 or i == len(test_files):
                    print(f"  Converting test files: {i}/{len(test_files)}")
                convert_mp3_to_wav(src, dst)
            else:
                dst = test_dir / filename
                shutil.copy2(str(src), str(dst))


def main():
    """Main function to split the dataset."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Group files by piece
    print("Scanning directory for mp3 files...")
    pieces = group_files_by_piece('.')
    
    if not pieces:
        print("No mp3 files found in the current directory!")
        return
    
    print(f"\nFound {len(pieces)} unique pieces (composer + opus combinations)")
    print(f"Total files: {sum(len(files) for files in pieces.values())}")
    
    # Display some statistics
    print("\nPiece distribution:")
    for piece_key, files in sorted(pieces.items()):
        composer, opus = piece_key
        print(f"  {composer}_{opus}: {len(files)} performances")
    
    # Split pieces
    print("\nSplitting pieces into train/val/test sets...")
    train_files, val_files, test_files = split_pieces(pieces)
    
    # Calculate actual piece counts for each set
    train_piece_count = sum(1 for key in pieces.keys() if any(f in train_files for f in pieces[key]))
    val_piece_count = sum(1 for key in pieces.keys() if any(f in val_files for f in pieces[key]))
    test_piece_count = sum(1 for key in pieces.keys() if any(f in test_files for f in pieces[key]))
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_files)} files from {train_piece_count} pieces ({train_piece_count/len(pieces)*100:.1f}%)")
    print(f"  Val:   {len(val_files)} files from {val_piece_count} pieces ({val_piece_count/len(pieces)*100:.1f}%)")
    print(f"  Test:  {len(test_files)} files from {test_piece_count} pieces ({test_piece_count/len(pieces)*100:.1f}%)")
    
    # Organize and convert files
    print("\nOrganizing and converting files to WAV format...")
    print("(Original mp3 files will be preserved)")
    
    # Organize files and convert to wav
    organize_files(train_files, val_files, test_files, source_dir='.', convert_to_wav=True)
    
    print("\n✓ Done! Files have been converted to WAV and organized into train/, val/, and test/ directories.")


if __name__ == '__main__':
    main()
