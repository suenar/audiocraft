#!/usr/bin/env python3
"""
Quick test script to view the structure of the GigaMIDI dataset.
This helps understand what fields are available before running the full filter.
"""

from datasets import load_dataset

def test_dataset_structure():
    """Load a sample from GigaMIDI to show its structure."""
    
    print("Loading a sample from GigaMIDI dataset...")
    print("Note: This requires authentication (huggingface-cli login)\n")
    
    try:
        # Load dataset in streaming mode and get first sample
        dataset = load_dataset("Metacreation/GigaMIDI", split="train", streaming=True)
        
        # Get first item
        first_item = next(iter(dataset))
        
        print("=" * 60)
        print("Dataset Structure - First Sample")
        print("=" * 60)
        
        # Show all available fields
        print("\nAvailable fields:")
        for key in first_item.keys():
            value = first_item[key]
            value_type = type(value).__name__
            
            # Show truncated preview for some fields
            if key == 'midi':
                print(f"  - {key}: <binary MIDI data, {len(value) if value else 0} bytes>")
            elif isinstance(value, list):
                print(f"  - {key}: {value_type} (length: {len(value)})")
                if len(value) > 0 and len(value) <= 10:
                    print(f"      Value: {value}")
                elif len(value) > 10:
                    print(f"      First 5 items: {value[:5]}")
            elif isinstance(value, str):
                preview = value[:100] + '...' if len(value) > 100 else value
                print(f"  - {key}: {value_type}")
                print(f"      Value: {preview}")
            else:
                print(f"  - {key}: {value_type}")
                print(f"      Value: {value}")
        
        print("\n" + "=" * 60)
        print("Key Fields for Filtering:")
        print("=" * 60)
        print(f"\nFilename: {first_item.get('midi_filename', 'N/A')}")
        print(f"Instrument Programs: {first_item.get('instrument_programs', [])}")
        print(f"Instrument Groups: {first_item.get('instrument_groups', [])}")
        
        # Show if this sample has violin
        instrument_programs = first_item.get('instrument_programs', [])
        has_violin = 40 in instrument_programs
        print(f"\nHas Violin (program 40): {has_violin}")
        
        print("\n" + "=" * 60)
        print("MIDI Program Numbers Reference:")
        print("=" * 60)
        print("  40 = Violin")
        print("  41 = Viola")
        print("  42 = Cello")
        print("  43 = Contrabass")
        print("  (See: https://en.wikipedia.org/wiki/General_MIDI)")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you:")
        print("1. Have access to Metacreation/GigaMIDI dataset")
        print("2. Are authenticated: huggingface-cli login")


if __name__ == "__main__":
    test_dataset_structure()
