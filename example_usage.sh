#!/bin/bash
# Example usage script for MIDI string instrument filter

# Install dependencies (run once)
echo "Installing dependencies..."
pip install -q mido tqdm

# Example 1: Basic usage - process all MIDI files in a directory
echo -e "\n=== Example 1: Basic filtering ==="
python filter_string_instruments.py \
    --input-dir /path/to/your/gigamidi/files \
    --output-dir ./filtered_string_instruments

# Example 2: Test with first 100 files
echo -e "\n=== Example 2: Test with limited files ==="
python filter_string_instruments.py \
    --input-dir /path/to/your/gigamidi/files \
    --output-dir ./test_output \
    --max-files 100

# Example 3: Non-recursive search (only top-level directory)
echo -e "\n=== Example 3: Non-recursive search ==="
python filter_string_instruments.py \
    --input-dir /path/to/your/gigamidi/files \
    --output-dir ./filtered_output \
    --no-recursive

# Example 4: Create symlinks instead of copying (saves disk space)
echo -e "\n=== Example 4: Using symlinks ==="
python filter_string_instruments.py \
    --input-dir /path/to/your/gigamidi/files \
    --output-dir ./filtered_symlinks \
    --symlink

echo -e "\nDone! Check the output directories for filtered MIDI files."
