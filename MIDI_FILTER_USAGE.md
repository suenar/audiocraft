# MIDI String Instrument Filter

This script filters MIDI files from a local directory to find files containing violin, viola, or cello tracks.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements_midi_filter.txt
```

Or install manually:

```bash
pip install mido tqdm
```

## Usage

### Basic Usage

Filter MIDI files from a local directory and save them to a specific directory:

```bash
python filter_string_instruments.py --input-dir /path/to/midi/files --output-dir ./filtered_midis
```

### Advanced Options

```bash
# Process only the first 1000 files (useful for testing)
python filter_string_instruments.py --input-dir ./midis --output-dir ./output --max-files 1000

# Non-recursive search (only top-level directory, no subdirectories)
python filter_string_instruments.py --input-dir ./midis --output-dir ./output --no-recursive

# Create symlinks instead of copying files (saves disk space)
python filter_string_instruments.py --input-dir ./midis --output-dir ./output --symlink
```

### Command-line Arguments

- `--input-dir` (required): Directory containing MIDI files to process
- `--output-dir` (required): Directory where filtered MIDI files will be saved
- `--max-files`: Maximum number of files to process (default: all files)
- `--no-recursive`: Do not search subdirectories recursively
- `--symlink`: Create symlinks instead of copying files (saves disk space)

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

1. Scans the input directory for MIDI files (recursively by default)
2. Parses each MIDI file to check for program change messages
3. Identifies files containing violin (program 40), viola (program 41), or cello (program 42)
4. Copies (or symlinks) matching files to the appropriate subdirectory
5. Prints a summary with statistics

## Example Output

```
Scanning for MIDI files in: /path/to/midi/files
Recursive search: True
Found 5000 MIDI files

Filtering MIDI files for violin, viola, and cello...
Output directory: ./filtered_midis
------------------------------------------------------------
Processing MIDI files: 100%|████████████████| 5000/5000 [02:15<00:00, 36.89file/s]
✓ [   1] symphony_no5.mid -> mixed/ (Violin, Viola, Cello)
✓ [   2] violin_sonata.mid -> violin/ (Violin)
✓ [   3] chamber_music.mid -> mixed/ (Violin, Cello)
...

============================================================
SUMMARY
============================================================
Total files processed: 5000
Total files matched: 782
Total errors: 0
Match rate: 15.64%

Instrument breakdown:
  Violin: 654
  Viola: 201
  Cello: 389

Filtered MIDI files saved to: ./filtered_midis
```

## Notes

- By default, the script searches subdirectories recursively
- Use `--no-recursive` to only search the top-level directory
- Use `--symlink` to create symbolic links instead of copying files (saves disk space)
- You can interrupt the script with Ctrl+C at any time
- The script handles duplicate filenames by adding a counter suffix
- Files are counted once per instrument type (one file can contain multiple instruments)

## Troubleshooting

### No MIDI Files Found

If the script reports no MIDI files found:
- Check that the input directory path is correct
- Verify that your MIDI files have `.mid` or `.midi` extensions (case insensitive)
- Try using an absolute path instead of a relative path

### Permission Errors

If you get permission errors when copying files:
- Check that you have read access to the input directory
- Check that you have write access to the output directory
- Try using `--symlink` instead of copying

### Memory Issues

If you encounter memory issues with very large MIDI files:
- Process files in batches using `--max-files`
- Consider filtering the input directory to process smaller subsets

### Parsing Errors

If specific MIDI files fail to parse:
- The script will skip corrupted files and continue processing
- Check the error messages for specific file names
- You may want to validate your MIDI files with a MIDI editor
