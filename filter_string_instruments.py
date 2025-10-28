#!/usr/bin/env python3
"""
Script to filter MIDI files containing violin, viola, or cello tracks from GigaMIDI dataset.

This script:
1. Loads the GigaMIDI dataset from Hugging Face
2. Filters MIDI files that contain violin (program 40), viola (program 41), or cello (program 42)
3. Saves the filtered MIDI files to a specified output directory

MIDI Program Numbers for string instruments:
- Violin: 40
- Viola: 41
- Cello: 42
"""

import os
import argparse
from pathlib import Path
from typing import Set, List
import mido
from datasets import load_dataset
from tqdm import tqdm
import tempfile


# MIDI program numbers for string instruments
STRING_INSTRUMENTS = {
    40: "Violin",
    41: "Viola", 
    42: "Cello"
}


def check_midi_for_instruments(midi_data: bytes, target_programs: Set[int]) -> tuple[bool, List[str]]:
    """
    Check if a MIDI file contains tracks with specified program numbers.
    
    Args:
        midi_data: Raw MIDI file data as bytes
        target_programs: Set of MIDI program numbers to search for
        
    Returns:
        Tuple of (has_instrument: bool, found_instruments: List[str])
    """
    try:
        # Write MIDI data to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_file:
            tmp_file.write(midi_data)
            tmp_path = tmp_file.name
        
        try:
            # Parse MIDI file
            midi_file = mido.MidiFile(tmp_path)
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
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"Error parsing MIDI data: {e}")
        return False, []


def filter_and_save_midis(
    output_dir: str,
    split: str = "train",
    max_files: int = None,
    streaming: bool = True
):
    """
    Filter MIDI files from GigaMIDI dataset and save those containing string instruments.
    
    Args:
        output_dir: Directory to save filtered MIDI files
        split: Dataset split to use (default: "train")
        max_files: Maximum number of files to process (None for all)
        streaming: Whether to use streaming mode for the dataset
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (output_path / "violin").mkdir(exist_ok=True)
    (output_path / "viola").mkdir(exist_ok=True)
    (output_path / "cello").mkdir(exist_ok=True)
    (output_path / "mixed").mkdir(exist_ok=True)
    
    print(f"Loading GigaMIDI dataset (split: {split})...")
    
    # Load the dataset
    dataset = load_dataset(
        "Metacreation/GigaMIDI",
        split=split,
        streaming=streaming
    )
    
    # Statistics
    total_processed = 0
    total_matched = 0
    instrument_counts = {inst: 0 for inst in STRING_INSTRUMENTS.values()}
    
    print(f"\nFiltering MIDI files for violin, viola, and cello...")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Process dataset
    iterator = iter(dataset)
    
    try:
        while True:
            if max_files and total_processed >= max_files:
                break
                
            try:
                item = next(iterator)
            except StopIteration:
                break
            
            total_processed += 1
            
            # Get MIDI data
            # The dataset structure may vary - adjust field names as needed
            if 'midi' in item:
                midi_data = item['midi']
            elif 'audio' in item and 'bytes' in item['audio']:
                midi_data = item['audio']['bytes']
            else:
                print(f"Warning: Could not find MIDI data in item {total_processed}")
                continue
            
            # Check for string instruments
            has_strings, found_instruments = check_midi_for_instruments(
                midi_data, 
                set(STRING_INSTRUMENTS.keys())
            )
            
            if has_strings:
                total_matched += 1
                
                # Get filename (use index if no name available)
                filename = item.get('name', item.get('filename', f'midi_{total_processed:06d}.mid'))
                if not filename.endswith('.mid'):
                    filename += '.mid'
                
                # Update statistics
                for inst in found_instruments:
                    instrument_counts[inst] += 1
                
                # Determine subdirectory based on instruments found
                if len(found_instruments) > 1:
                    subdir = "mixed"
                    dest_filename = f"{'_'.join(found_instruments).lower()}_{filename}"
                else:
                    subdir = found_instruments[0].lower()
                    dest_filename = filename
                
                # Save the MIDI file
                output_file = output_path / subdir / dest_filename
                
                # Write MIDI data
                if isinstance(midi_data, bytes):
                    output_file.write_bytes(midi_data)
                else:
                    # Handle other data types if necessary
                    with open(output_file, 'wb') as f:
                        f.write(midi_data)
                
                print(f"✓ [{total_matched:4d}] {filename} -> {subdir}/ (instruments: {', '.join(found_instruments)})")
            
            # Progress update
            if total_processed % 100 == 0:
                print(f"Processed {total_processed} files, found {total_matched} matches...")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_processed}")
    print(f"Total files matched: {total_matched}")
    print(f"Match rate: {100 * total_matched / max(total_processed, 1):.2f}%")
    print("\nInstrument breakdown:")
    for instrument, count in instrument_counts.items():
        print(f"  {instrument}: {count}")
    print(f"\nFiltered MIDI files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter MIDI files with violin, viola, or cello from GigaMIDI dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save filtered MIDI files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download entire dataset instead of streaming (requires more disk space)"
    )
    
    args = parser.parse_args()
    
    filter_and_save_midis(
        output_dir=args.output_dir,
        split=args.split,
        max_files=args.max_files,
        streaming=not args.no_streaming
    )


if __name__ == "__main__":
    main()
