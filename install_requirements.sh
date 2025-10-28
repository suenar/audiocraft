#!/bin/bash

# Installation script for GigaMIDI filter requirements

echo "=========================================="
echo "Installing GigaMIDI Filter Requirements"
echo "=========================================="
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

echo "Installing required Python packages..."
pip install datasets tqdm huggingface-hub

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Request access to the dataset at:"
echo "   https://huggingface.co/datasets/Metacreation/GigaMIDI"
echo ""
echo "2. Once approved, authenticate with:"
echo "   huggingface-cli login"
echo ""
echo "3. Test the setup with:"
echo "   python test_gigamidi_structure.py"
echo ""
echo "4. Run the filter script:"
echo "   python filter_gigamidi_violin.py --output_dir ./violin_midi_files"
echo ""
