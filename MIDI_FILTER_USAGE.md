# GigaMIDI String Instrument Filter

This script filters MIDI files from the [GigaMIDI dataset](https://huggingface.co/datasets/Metacreation/GigaMIDI) to find files containing violin, viola, or cello tracks.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements_midi_filter.txt
```

Or install manually:

```bash
pip install datasets mido tqdm huggingface-hub
```

## Usage

### Basic Usage

Filter MIDI files and save them to a specific directory:

```bash
python filter_string_instruments.py --output-dir ./filtered_midis
```

### Advanced Options

```bash
# Process only the first 1000 files (useful for testing)
python filter_string_instruments.py --output-dir ./filtered_midis --max-files 1000

# Use a different dataset split
python filter_string_instruments.py --output-dir ./filtered_midis --split validation

# Download entire dataset instead of streaming (faster but requires more disk space)
python filter_string_instruments.py --output-dir ./filtered_midis --no-streaming
```

### Command-line Arguments

- `--output-dir` (required): Directory where filtered MIDI files will be saved
- `--split`: Dataset split to use (default: "train")
- `--max-files`: Maximum number of files to process (default: all files)
- `--no-streaming`: Download entire dataset instead of streaming

## Output Structure

The script organizes filtered MIDI files into subdirectories:

```
output_dir/
├── violin/          # Files containing only violin
├── viola/           # Files containing only viola
├── cello/           # Files containing only cello
└── mixed/           # Files containing multiple string instruments
```

## MIDI Program Numbers

The script searches for these MIDI program numbers:
- **Violin**: Program 40
- **Viola**: Program 41
- **Cello**: Program 42

## How It Works

1. Loads the GigaMIDI dataset from Hugging Face (streaming by default)
2. Parses each MIDI file to check for program change messages
3. Identifies files containing the target instruments
4. Saves matching files to the appropriate subdirectory
5. Prints a summary with statistics

## Example Output

```
Loading GigaMIDI dataset (split: train)...

Filtering MIDI files for violin, viola, and cello...
Output directory: ./filtered_midis
------------------------------------------------------------
✓ [   1] symphony_no5.mid -> mixed/ (instruments: Violin, Viola, Cello)
✓ [   2] violin_sonata.mid -> violin/ (instruments: Violin)
Processed 100 files, found 15 matches...
...

============================================================
SUMMARY
============================================================
Total files processed: 1000
Total files matched: 157
Match rate: 15.70%

Instrument breakdown:
  Violin: 142
  Viola: 89
  Cello: 103

Filtered MIDI files saved to: ./filtered_midis
```

## Notes

- The script uses streaming by default to avoid downloading the entire dataset
- Streaming is slower but uses minimal disk space
- Use `--no-streaming` for faster processing if you have sufficient disk space
- You can interrupt the script with Ctrl+C and it will show results for files processed so far
- The script handles duplicate instrument tracks (counts each MIDI file once per instrument type)

## Troubleshooting

### Authentication Error

If you get an authentication error, you may need to login to Hugging Face:

```bash
huggingface-cli login
```

### Memory Issues

If you encounter memory issues, try:
- Using streaming mode (default)
- Processing fewer files with `--max-files`
- Processing the dataset in batches

### Dataset Structure Changes

If the GigaMIDI dataset structure changes, you may need to adjust the field names in the script where it accesses MIDI data (around line 105-112).
