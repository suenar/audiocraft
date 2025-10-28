# GigaMIDI Violin Filter Script

This script filters the GigaMIDI dataset for MIDI files containing violin and saves them to a specified directory.

## Dataset Information

- **Dataset**: [Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI)
- **Size**: Over 2.1 million unique MIDI files
- **License**: CC BY-NC 4.0 (Non-commercial use only)

## Prerequisites

1. **Install required packages**:
   ```bash
   pip install datasets tqdm huggingface-hub
   ```

2. **Authentication** (REQUIRED):
   The GigaMIDI dataset is restricted and requires authentication. You need to:
   
   a. Request access to the dataset at: https://huggingface.co/datasets/Metacreation/GigaMIDI
   
   b. Once approved, log in with your HuggingFace credentials:
   ```bash
   huggingface-cli login
   ```
   
   You'll need to enter your HuggingFace token, which you can get from: https://huggingface.co/settings/tokens

## Usage

### Basic Usage (Violin only)

Filter for violin only and save to default directory (`./violin_midi_files`):

```bash
python filter_gigamidi_violin.py
```

### Custom Output Directory

Specify a custom output directory:

```bash
python filter_gigamidi_violin.py --output_dir /path/to/your/output/folder
```

### Include All String Instruments

To include violin, viola, cello, and contrabass:

```bash
python filter_gigamidi_violin.py --include_strings
```

### Download Mode (Non-streaming)

By default, the script uses streaming mode to avoid downloading the entire dataset. If you prefer to download the full dataset first (requires significant disk space):

```bash
python filter_gigamidi_violin.py --no_streaming
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output_dir` | Directory to save filtered MIDI files | `./violin_midi_files` |
| `--include_strings` | Include all string instruments (violin, viola, cello, contrabass) | False (violin only) |
| `--no_streaming` | Download entire dataset instead of streaming | False (streaming enabled) |

## Examples

1. **Filter violin only, save to custom directory**:
   ```bash
   python filter_gigamidi_violin.py --output_dir /data/violin_midis
   ```

2. **Include all string instruments**:
   ```bash
   python filter_gigamidi_violin.py --output_dir /data/string_midis --include_strings
   ```

3. **Use non-streaming mode**:
   ```bash
   python filter_gigamidi_violin.py --no_streaming
   ```

## How It Works

1. The script loads the GigaMIDI dataset using the HuggingFace `datasets` library
2. It filters MIDI files based on MIDI program numbers:
   - **Violin**: Program 40
   - **Viola**: Program 41 (if `--include_strings` is used)
   - **Cello**: Program 42 (if `--include_strings` is used)
   - **Contrabass**: Program 43 (if `--include_strings` is used)
3. Filtered MIDI files are saved to the specified output directory with their original filenames

## Expected Output

The script will:
- Show progress with a progress bar
- Display how many files have been processed
- Report the total number of violin MIDI files found
- Save all filtered MIDI files to the output directory

Example output:
```
Output directory: /workspace/violin_midi_files
Loading GigaMIDI dataset...
Filtering for: Violin only
Loading dataset in streaming mode...

Processing dataset...
Filtering MIDI files: 50000it [05:23, 154.67it/s]

Saved 100 violin MIDI files so far...
...

============================================================
Processing complete!
Total files processed: 2100000
Violin files found: 15432
Files saved to: /workspace/violin_midi_files
============================================================
```

## Troubleshooting

### Authentication Error

If you get an authentication error:
```
Error: Access to dataset Metacreation/GigaMIDI is restricted.
```

Solution:
1. Request access at: https://huggingface.co/datasets/Metacreation/GigaMIDI
2. Wait for approval
3. Run `huggingface-cli login` and enter your token

### Memory Issues

If you run out of memory with `--no_streaming`:
- Use streaming mode (default, don't use `--no_streaming` flag)
- Streaming processes one file at a time and doesn't require downloading the entire dataset

## Citation

If you use this dataset, please cite:

```bibtex
@article{lee2025gigamidi,
  title={The GigaMIDI Dataset with Features for Expressive Music Performance Detection},
  author={Lee, Keon Ju Maverick and Ens, Jeff and Adkins, Sara and Sarmento, Pedro and Barthet, Mathieu and Pasquier, Philippe},
  journal={Transactions of the International Society for Music Information Retrieval (TISMIR)},
  year={2025}
}
```

## License

The GigaMIDI dataset is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). This means you can use it for:
- ✅ Research purposes
- ✅ Educational purposes
- ❌ NOT for commercial purposes
