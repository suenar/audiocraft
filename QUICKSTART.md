# Quick Start Guide: Filter GigaMIDI for Violin

This guide helps you quickly set up and run the GigaMIDI violin filter.

## 📁 Files Created

1. **`filter_gigamidi_violin.py`** - Main script to filter and save violin MIDI files
2. **`test_gigamidi_structure.py`** - Test script to view dataset structure
3. **`install_requirements.sh`** - Installation helper script
4. **`GIGAMIDI_FILTER_README.md`** - Detailed documentation

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
bash install_requirements.sh
```

Or manually:
```bash
pip install datasets tqdm huggingface-hub
```

### Step 2: Get Dataset Access & Authenticate

1. **Request access** (one-time): 
   - Visit: https://huggingface.co/datasets/Metacreation/GigaMIDI
   - Click "Request Access" and wait for approval

2. **Login** (one-time):
   ```bash
   huggingface-cli login
   ```
   - Get your token from: https://huggingface.co/settings/tokens
   - Paste it when prompted

### Step 3: Run the Filter

**Basic usage** (violin only):
```bash
python filter_gigamidi_violin.py
```

**Custom output directory**:
```bash
python filter_gigamidi_violin.py --output_dir /path/to/save/violin_files
```

**Include all string instruments** (violin, viola, cello, contrabass):
```bash
python filter_gigamidi_violin.py --include_strings --output_dir /path/to/string_files
```

## 🧪 Test First (Optional)

Before running the full filter, test with:
```bash
python test_gigamidi_structure.py
```

This shows you:
- Dataset structure and available fields
- Sample instrument programs
- Whether your authentication works

## 📊 What to Expect

- **Dataset size**: 2.1+ million MIDI files
- **Violin files**: Estimated 10,000-50,000+ files (depends on dataset)
- **Processing time**: Hours (in streaming mode, depends on internet speed)
- **Disk space**: Varies, but much less than full dataset with streaming mode

The script uses **streaming mode** by default, which means:
- ✅ Doesn't download entire dataset first
- ✅ Lower disk space requirements
- ✅ Can start saving files immediately
- ⚠️ Slower overall, but more practical for large datasets

## 🎯 Examples

1. **Filter violin to default directory**:
   ```bash
   python filter_gigamidi_violin.py
   # Output: ./violin_midi_files/
   ```

2. **Filter to specific directory**:
   ```bash
   python filter_gigamidi_violin.py --output_dir ~/music/gigamidi/violin
   # Output: ~/music/gigamidi/violin/
   ```

3. **Get all string instruments**:
   ```bash
   python filter_gigamidi_violin.py --include_strings --output_dir ./strings
   # Output: ./strings/ (violin, viola, cello, contrabass)
   ```

## ⚙️ Command Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output_dir PATH` | Where to save MIDI files | `./violin_midi_files` |
| `--include_strings` | Include viola, cello, contrabass | `False` |
| `--no_streaming` | Download full dataset first | `False` (streaming ON) |

## 🎻 Instrument Programs

The filter uses MIDI General MIDI program numbers:

| Instrument | Program # | Included by Default |
|------------|-----------|---------------------|
| Violin | 40 | ✅ Yes |
| Viola | 41 | Only with `--include_strings` |
| Cello | 42 | Only with `--include_strings` |
| Contrabass | 43 | Only with `--include_strings` |

## 📝 Sample Output

```
Output directory: /workspace/violin_midi_files
Loading GigaMIDI dataset...
Filtering for: Violin only
Loading dataset in streaming mode...

Processing dataset...
Filtering MIDI files: 125000it [15:32, 134.12it/s]

Saved 100 violin MIDI files so far...
Saved 200 violin MIDI files so far...
...

============================================================
Processing complete!
Total files processed: 2100000
Violin files found: 18453
Files saved to: /workspace/violin_midi_files
============================================================
```

## ❓ Troubleshooting

**Problem**: `Access to dataset ... is restricted`
- **Solution**: Request access and authenticate (see Step 2)

**Problem**: Script is too slow
- **Solution**: This is normal with streaming mode and a huge dataset. Consider:
  - Running on a machine with faster internet
  - Using `--no_streaming` if you have enough disk space (~100GB+)
  - Running overnight

**Problem**: Out of memory with `--no_streaming`
- **Solution**: Use default streaming mode (remove `--no_streaming` flag)

## 📚 More Information

- Full documentation: `GIGAMIDI_FILTER_README.md`
- Dataset page: https://huggingface.co/datasets/Metacreation/GigaMIDI
- General MIDI spec: https://en.wikipedia.org/wiki/General_MIDI

## 📄 License

**Important**: The GigaMIDI dataset is under **CC BY-NC 4.0** license:
- ✅ Free for research and educational use
- ❌ NOT for commercial use
- Must provide attribution when publishing

## 🎵 Enjoy Your Violin MIDI Collection!

Once complete, you'll have a curated collection of violin MIDI files ready for:
- Music analysis
- Machine learning training
- Music generation research
- Educational purposes
