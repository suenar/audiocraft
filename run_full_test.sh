#!/bin/bash
# Complete workflow for testing Encodec compression
# This script runs the compression test and creates visualizations

set -e  # Exit on error

# Configuration
CHECKPOINT="/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/checkpoints/compression_32khz_violin.bin"
INPUT_AUDIO="/home/smg/v-sunan/VIOLIN-VALLE/egs/atepp/prompt_violin/prompt_A.wav"
OUTPUT_DIR="compression_test_results"
DEVICE="cuda"  # or "cpu"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Encodec Compression Test"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Step 1: Run compression test
echo "Step 1/2: Running compression test..."
python test_encodec_compression.py \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT_AUDIO" \
    --output "$OUTPUT_DIR/reconstructed.wav" \
    --device "$DEVICE"

echo ""
echo "✓ Compression test completed!"
echo ""

# Step 2: Create visualization
echo "Step 2/2: Creating visualization..."
python visualize_compression.py \
    --original "$INPUT_AUDIO" \
    --reconstructed "$OUTPUT_DIR/reconstructed.wav" \
    --codes "$OUTPUT_DIR/reconstructed.codes.pt" \
    --output "$OUTPUT_DIR/compression_comparison.png"

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files created:"
echo "  - reconstructed.wav          : Compressed and decompressed audio"
echo "  - reconstructed.codes.pt     : Discrete codes"
echo "  - compression_comparison.png : Visualization"
echo ""
echo "Next steps:"
echo "  1. Listen to the reconstructed audio:"
echo "     play $OUTPUT_DIR/reconstructed.wav"
echo ""
echo "  2. View the visualization:"
echo "     xdg-open $OUTPUT_DIR/compression_comparison.png"
echo ""
echo "  3. Compare with original:"
echo "     play $INPUT_AUDIO"
echo ""
