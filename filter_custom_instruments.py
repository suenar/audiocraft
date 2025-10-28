#!/usr/bin/env python3
"""
Customizable script to filter GigaMIDI dataset for specific instruments.
This is an example showing how to adapt the violin filter for other instruments.

Example usage:
    # Piano only
    python filter_custom_instruments.py --instruments 0 --output_dir ./piano_midi
    
    # Guitar (acoustic and electric)
    python filter_custom_instruments.py --instruments 24 25 26 27 28 29 30 31 --output_dir ./guitar_midi
    
    # Brass section (trumpet, trombone, french horn, tuba)
    python filter_custom_instruments.py --instruments 56 57 58 60 61 --output_dir ./brass_midi
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# General MIDI Instrument Reference
INSTRUMENT_REFERENCE = {
    # Piano
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet",
    
    # Chromatic Percussion
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone",
    12: "Marimba", 13: "Xylophone", 14: "Tubular Bells", 15: "Dulcimer",
    
    # Organ
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ",
    19: "Church Organ", 20: "Reed Organ", 21: "Accordion",
    22: "Harmonica", 23: "Tango Accordion",
    
    # Guitar
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)",
    26: "Electric Guitar (jazz)", 27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)", 29: "Overdriven Guitar",
    30: "Distortion Guitar", 31: "Guitar Harmonics",
    
    # Bass
    32: "Acoustic Bass", 33: "Electric Bass (finger)",
    34: "Electric Bass (pick)", 35: "Fretless Bass",
    36: "Slap Bass 1", 37: "Slap Bass 2",
    38: "Synth Bass 1", 39: "Synth Bass 2",
    
    # Strings
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    44: "Tremolo Strings", 45: "Pizzicato Strings",
    46: "Orchestral Harp", 47: "Timpani",
    
    # Ensemble
    48: "String Ensemble 1", 49: "String Ensemble 2",
    50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs",
    54: "Synth Choir", 55: "Orchestra Hit",
    
    # Brass
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section",
    62: "Synth Brass 1", 63: "Synth Brass 2",
    
    # Reed
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    
    # Pipe
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    
    # Synth Lead
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)", 84: "Lead 5 (charang)", 85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    
    # Synth Pad
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)",
    91: "Pad 4 (choir)", 92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)",
    94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    
    # Synth Effects
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)",
    99: "FX 4 (atmosphere)", 100: "FX 5 (brightness)", 101: "FX 6 (goblins)",
    102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    
    # Ethnic
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto",
    108: "Kalimba", 109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    
    # Percussive
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock",
    116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum", 119: "Reverse Cymbal",
    
    # Sound Effects
    120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore",
    123: "Bird Tweet", 124: "Telephone Ring", 125: "Helicopter",
    126: "Applause", 127: "Gunshot",
}


def print_instrument_reference():
    """Print the General MIDI instrument reference."""
    print("\nGeneral MIDI Instrument Programs (0-127):")
    print("=" * 70)
    
    categories = [
        ("Piano", range(0, 8)),
        ("Chromatic Percussion", range(8, 16)),
        ("Organ", range(16, 24)),
        ("Guitar", range(24, 32)),
        ("Bass", range(32, 40)),
        ("Strings", range(40, 48)),
        ("Ensemble", range(48, 56)),
        ("Brass", range(56, 64)),
        ("Reed", range(64, 72)),
        ("Pipe", range(72, 80)),
        ("Synth Lead", range(80, 88)),
        ("Synth Pad", range(88, 96)),
        ("Synth Effects", range(96, 104)),
        ("Ethnic", range(104, 112)),
        ("Percussive", range(112, 120)),
        ("Sound Effects", range(120, 128)),
    ]
    
    for category, prog_range in categories:
        print(f"\n{category}:")
        for prog in prog_range:
            if prog in INSTRUMENT_REFERENCE:
                print(f"  {prog:3d}: {INSTRUMENT_REFERENCE[prog]}")


def filter_and_save_midi(output_dir, instrument_programs, streaming=True):
    """
    Load GigaMIDI dataset, filter for specific instruments, and save MIDI files.
    
    Args:
        output_dir (str): Directory to save filtered MIDI files
        instrument_programs (set): Set of MIDI program numbers to filter
        streaming (bool): If True, use streaming mode
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nFiltering for instrument programs: {sorted(instrument_programs)}")
    print("Instruments:")
    for prog in sorted(instrument_programs):
        inst_name = INSTRUMENT_REFERENCE.get(prog, f"Unknown ({prog})")
        print(f"  - {prog}: {inst_name}")
    
    # Load dataset
    try:
        if streaming:
            print("\nLoading dataset in streaming mode...")
            dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)
            
            matched_count = 0
            total_processed = 0
            
            print("Processing dataset...\n")
            for item in tqdm(dataset, desc="Filtering MIDI files"):
                total_processed += 1
                
                # Check if any target instrument is present
                item_programs = item.get('instrument_programs', [])
                has_target = any(prog in instrument_programs for prog in item_programs)
                
                if has_target:
                    midi_data = item.get('midi')
                    midi_filename = item.get('midi_filename', f'midi_{matched_count}.mid')
                    
                    if midi_data is not None:
                        output_file = output_path / midi_filename
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_file, 'wb') as f:
                            f.write(midi_data)
                        
                        matched_count += 1
                        
                        if matched_count % 100 == 0:
                            print(f"\nSaved {matched_count} MIDI files so far...")
            
            print(f"\n{'='*60}")
            print(f"Processing complete!")
            print(f"Total files processed: {total_processed}")
            print(f"Matched files found: {matched_count}")
            print(f"Files saved to: {output_path.absolute()}")
            print(f"{'='*60}")
            
        else:
            print("\nLoading entire dataset...")
            dataset = load_dataset("Metacreation/GigaMIDI", split="train")
            
            def has_target_instrument(example):
                item_programs = example.get('instrument_programs', [])
                return any(prog in instrument_programs for prog in item_programs)
            
            filtered_dataset = dataset.filter(has_target_instrument)
            print(f"\nFound {len(filtered_dataset)} matching MIDI files")
            
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
            print(f"Successfully saved {len(filtered_dataset)} MIDI files")
            print(f"Output directory: {output_path.absolute()}")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This dataset requires authentication.")
        print("Please ensure you have access and are logged in with: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Filter GigaMIDI dataset for specific instruments',
        epilog='Example: python %(prog)s --instruments 0 1 2 --output_dir ./piano_midi'
    )
    parser.add_argument(
        '--instruments',
        type=int,
        nargs='+',
        required=True,
        help='MIDI program numbers to filter (space-separated)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save filtered MIDI files'
    )
    parser.add_argument(
        '--no_streaming',
        action='store_true',
        help='Download entire dataset instead of streaming'
    )
    parser.add_argument(
        '--show_instruments',
        action='store_true',
        help='Show General MIDI instrument reference and exit'
    )
    
    args = parser.parse_args()
    
    # Show instrument reference if requested
    if args.show_instruments:
        print_instrument_reference()
        return
    
    # Validate instrument programs
    instrument_programs = set(args.instruments)
    invalid_programs = [p for p in instrument_programs if p < 0 or p > 127]
    if invalid_programs:
        print(f"Error: Invalid MIDI program numbers: {invalid_programs}")
        print("Valid range is 0-127")
        print("\nUse --show_instruments to see the full reference")
        return
    
    # Run the filter
    filter_and_save_midi(
        output_dir=args.output_dir,
        instrument_programs=instrument_programs,
        streaming=not args.no_streaming
    )


if __name__ == "__main__":
    main()
