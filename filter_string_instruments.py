#!/usr/bin/env python3
"""
Script to filter MIDI files containing violin, viola, or cello tracks from local directory.

This script:
1. Scans a local directory for MIDI files
2. Filters MIDI files that contain violin (program 40), viola (program 41), or cello (program 42)
3. Copies the filtered MIDI files to a specified output directory

MIDI Program Numbers for string instruments:
- Violin: 40
- Viola: 41
- Cello: 42
"""

import os
import argparse
from pathlib import Path
from typing import Set, List, Tuple
import mido
from tqdm import tqdm
import shutil


# MIDI program numbers for string instruments
STRING_INSTRUMENTS = {
    40: "Violin",
    41: "Viola", 
    42: "Cello"
}


def check_midi_for_instruments(midi_path: Path, target_programs: Set[int]) -> Tuple[bool, List[str]]:
    """
    Check if a MIDI file contains tracks with specified program numbers.
    
    Args:
        midi_path: Path to MIDI file
        target_programs: Set of MIDI program numbers to search for
        
    Returns:
        Tuple of (has_instrument: bool, found_instruments: List[str])
    """
    try:
        # Parse MIDI file
        midi_file = mido.MidiFile(midi_path)
        found_programs = set()
        
        # Iterate through all tracks and messages
        for track in midi_file.tracks:
            for msg in track:
                # Check for program_change messages
                if msg.type == 'program_change':
                    if msg.program in target_programs:
                        found_programs.add(msg.program)
        
        # Get instrument names
        found_instruments = [STRING_INSTRUMENTS[p] for p in found_programs if p in STRING_INSTRUMENTS]
        
        return len(found_programs) > 0, found_instruments
            
    except Exception as e:
        print(f"Error parsing MIDI file {midi_path.name}: {e}")
        return False, []


def find_midi_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """
    Find all MIDI files in a directory.
    
    Args:
        input_dir: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects for MIDI files
    """
    midi_extensions = {'.mid', '.midi', '.MID', '.MIDI'}
    midi_files = []
    
    if recursive:
        for ext in midi_extensions:
            midi_files.extend(input_dir.rglob(f'*{ext}'))
    else:
        for ext in midi_extensions:
            midi_files.extend(input_dir.glob(f'*{ext}'))
    
    return sorted(midi_files)


def filter_and_save_midis(
    input_dir: str,
    output_dir: str,
    recursive: bool = True,
    max_files: int = None,
    copy_files: bool = True
):
    """
    Filter MIDI files from local directory and save those containing string instruments.
    
    Args:
        input_dir: Directory containing MIDI files to process
        output_dir: Directory to save filtered MIDI files
        recursive: Whether to search input directory recursively
        max_files: Maximum number of files to process (None for all)
        copy_files: If True, copy files; if False, create symlinks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (output_path / "violin").mkdir(exist_ok=True)
    (output_path / "viola").mkdir(exist_ok=True)
    (output_path / "cello").mkdir(exist_ok=True)
    (output_path / "mixed").mkdir(exist_ok=True)
    
    print(f"Scanning for MIDI files in: {input_path}")
    print(f"Recursive search: {recursive}")
    
    # Find all MIDI files
    midi_files = find_midi_files(input_path, recursive)
    total_files = len(midi_files)
    
    if total_files == 0:
        print(f"No MIDI files found in {input_path}")
        return
    
    print(f"Found {total_files} MIDI files")
    
    if max_files:
        midi_files = midi_files[:max_files]
        print(f"Processing first {len(midi_files)} files")
    
    # Statistics
    total_processed = 0
    total_matched = 0
    total_errors = 0
    instrument_counts = {inst: 0 for inst in STRING_INSTRUMENTS.values()}
    
    print(f"\nFiltering MIDI files for violin, viola, and cello...")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Process files with progress bar
    for midi_file in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        total_processed += 1
        
        # Check for string instruments
        has_strings, found_instruments = check_midi_for_instruments(
            midi_file, 
            set(STRING_INSTRUMENTS.keys())
        )
        
        if has_strings:
            total_matched += 1
            
            # Update statistics
            for inst in found_instruments:
                instrument_counts[inst] += 1
            
            # Determine subdirectory based on instruments found
            if len(found_instruments) > 1:
                subdir = "mixed"
                dest_filename = f"{'_'.join(found_instruments).lower()}_{midi_file.name}"
            else:
                subdir = found_instruments[0].lower()
                dest_filename = midi_file.name
            
            # Save the MIDI file
            output_file = output_path / subdir / dest_filename
            
            # Handle filename conflicts
            if output_file.exists():
                base = output_file.stem
                ext = output_file.suffix
                counter = 1
                while output_file.exists():
                    output_file = output_path / subdir / f"{base}_{counter}{ext}"
                    counter += 1
            
            try:
                if copy_files:
                    shutil.copy2(midi_file, output_file)
                else:
                    output_file.symlink_to(midi_file.absolute())
                
                tqdm.write(f"✓ [{total_matched:4d}] {midi_file.name} -> {subdir}/ ({', '.join(found_instruments)})")
            except Exception as e:
                tqdm.write(f"✗ Error copying {midi_file.name}: {e}")
                total_errors += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_processed}")
    print(f"Total files matched: {total_matched}")
    print(f"Total errors: {total_errors}")
    print(f"Match rate: {100 * total_matched / max(total_processed, 1):.2f}%")
    print("\nInstrument breakdown:")
    for instrument, count in instrument_counts.items():
        if count > 0:
            print(f"  {instrument}: {count}")
    print(f"\nFiltered MIDI files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter MIDI files with violin, viola, or cello from local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all MIDI files in a directory
  python filter_string_instruments.py --input-dir /path/to/midi/files --output-dir ./filtered

  # Process only first 1000 files
  python filter_string_instruments.py --input-dir ./midis --output-dir ./output --max-files 1000

  # Non-recursive search (only top-level directory)
  python filter_string_instruments.py --input-dir ./midis --output-dir ./output --no-recursive

  # Create symlinks instead of copying files
  python filter_string_instruments.py --input-dir ./midis --output-dir ./output --symlink
        """
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing MIDI files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save filtered MIDI files"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories recursively"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space)"
    )
    
    args = parser.parse_args()
    
    filter_and_save_midis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        max_files=args.max_files,
        copy_files=not args.symlink
    )


if __name__ == "__main__":
    main()
