#!/usr/bin/env python3
"""
Script to load the GigaMIDI dataset and filter MIDI files containing violin.
The filtered MIDI files are saved to a specified output directory.

Violin MIDI programs (General MIDI):
- Program 40: Violin
- Program 41: Viola  
- Program 42: Cello (optional)
- Program 43: Contrabass (optional)

Usage:
    python filter_gigamidi_violin.py --output_dir ./violin_midi_files
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def filter_and_save_violin_midi(output_dir, include_strings=False, streaming=True):
    """
    Load GigaMIDI dataset, filter for violin tracks, and save MIDI files.
    
    Args:
        output_dir (str): Directory to save filtered MIDI files
        include_strings (bool): If True, include all string instruments (viola, cello, contrabass)
        streaming (bool): If True, use streaming mode to avoid downloading entire dataset
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path.absolute()}")
    print(f"Loading GigaMIDI dataset...")
    
    # Define violin-related MIDI program numbers (0-indexed)
    violin_programs = {40}  # Violin
    
    if include_strings:
        # Include other string instruments
        violin_programs.update({41, 42, 43})  # Viola, Cello, Contrabass
        print("Filtering for: Violin, Viola, Cello, Contrabass")
    else:
        print("Filtering for: Violin only")
    
    # Load dataset
    try:
        if streaming:
            print("Loading dataset in streaming mode...")
            dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)
            
            # Process dataset in streaming mode
            violin_count = 0
            total_processed = 0
            
            print("\nProcessing dataset...")
            for item in tqdm(dataset, desc="Filtering MIDI files"):
                total_processed += 1
                
                # Check if violin is in the instrument programs
                instrument_programs = item.get('instrument_programs', [])
                
                # Check if any violin program is in the list
                has_violin = any(prog in violin_programs for prog in instrument_programs)
                
                if has_violin:
                    # Get MIDI data and filename
                    midi_data = item.get('midi')
                    midi_filename = item.get('midi_filename', f'midi_{violin_count}.mid')
                    
                    if midi_data is not None:
                        # Save MIDI file
                        output_file = output_path / midi_filename
                        
                        # Create subdirectories if needed
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write MIDI data
                        with open(output_file, 'wb') as f:
                            f.write(midi_data)
                        
                        violin_count += 1
                        
                        if violin_count % 100 == 0:
                            print(f"\nSaved {violin_count} violin MIDI files so far...")
            
            print(f"\n{'='*60}")
            print(f"Processing complete!")
            print(f"Total files processed: {total_processed}")
            print(f"Violin files found: {violin_count}")
            print(f"Files saved to: {output_path.absolute()}")
            print(f"{'='*60}")
            
        else:
            # Load entire dataset (requires more memory and disk space)
            print("Loading entire dataset (this may take a while)...")
            dataset = load_dataset("Metacreation/GigaMIDI", split="train")
            
            # Filter dataset
            print("Filtering for violin tracks...")
            
            def has_violin(example):
                instrument_programs = example.get('instrument_programs', [])
                return any(prog in violin_programs for prog in instrument_programs)
            
            filtered_dataset = dataset.filter(has_violin)
            
            print(f"Found {len(filtered_dataset)} MIDI files with violin")
            
            # Save MIDI files
            print("Saving MIDI files...")
            for idx, item in enumerate(tqdm(filtered_dataset)):
                midi_data = item.get('midi')
                midi_filename = item.get('midi_filename', f'midi_{idx}.mid')
                
                if midi_data is not None:
                    output_file = output_path / midi_filename
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'wb') as f:
                        f.write(midi_data)
            
            print(f"\n{'='*60}")
            print(f"Successfully saved {len(filtered_dataset)} violin MIDI files")
            print(f"Output directory: {output_path.absolute()}")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This dataset requires authentication. Please ensure you:")
        print("1. Have access to the Metacreation/GigaMIDI dataset on HuggingFace")
        print("2. Are logged in with: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Filter and save MIDI files with violin from GigaMIDI dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./violin_midi_files',
        help='Directory to save filtered MIDI files (default: ./violin_midi_files)'
    )
    parser.add_argument(
        '--include_strings',
        action='store_true',
        help='Include all string instruments (violin, viola, cello, contrabass)'
    )
    parser.add_argument(
        '--no_streaming',
        action='store_true',
        help='Download entire dataset instead of streaming (requires more disk space)'
    )
    
    args = parser.parse_args()
    
    # Run the filtering and saving
    filter_and_save_violin_midi(
        output_dir=args.output_dir,
        include_strings=args.include_strings,
        streaming=not args.no_streaming
    )


if __name__ == "__main__":
    main()
