# GigaMIDI Violin Filter - Complete Summary

## 📦 What Was Created

I've created a complete toolkit to filter the GigaMIDI dataset for violin (and other instruments). Here's what you have:

### Main Scripts

1. **`filter_gigamidi_violin.py`** (6.3 KB) ⭐ **MAIN SCRIPT**
   - Filters GigaMIDI dataset for violin tracks
   - Saves MIDI files to specified directory
   - Uses streaming mode by default (memory efficient)
   - Options: violin-only or all string instruments

2. **`filter_custom_instruments.py`** (11 KB)
   - Flexible version for filtering ANY instruments
   - Includes complete General MIDI reference (0-127)
   - Great for filtering piano, guitar, brass, etc.

3. **`test_gigamidi_structure.py`** (3.0 KB)
   - Quick test to view dataset structure
   - Helps verify authentication works
   - Shows sample data before running full filter

### Documentation

4. **`QUICKSTART.md`** (5.0 KB) ⭐ **START HERE**
   - Step-by-step setup guide
   - Common use cases and examples
   - Troubleshooting tips

5. **`GIGAMIDI_FILTER_README.md`** (4.8 KB)
   - Comprehensive documentation
   - Detailed API reference
   - Citation information

6. **`SUMMARY.md`** (this file)
   - Overview of all files
   - Quick command reference

### Utilities

7. **`install_requirements.sh`** (1.1 KB)
   - One-command installation script
   - Installs all Python dependencies

## 🚀 Quick Start

### 1. Install & Setup (One Time)
```bash
# Install dependencies
bash install_requirements.sh

# Request dataset access
# Visit: https://huggingface.co/datasets/Metacreation/GigaMIDI

# Authenticate (after approval)
huggingface-cli login
```

### 2. Test (Optional but Recommended)
```bash
python test_gigamidi_structure.py
```

### 3. Run the Violin Filter
```bash
# Basic: Filter violin to ./violin_midi_files/
python filter_gigamidi_violin.py

# Custom directory:
python filter_gigamidi_violin.py --output_dir /path/to/save

# All string instruments:
python filter_gigamidi_violin.py --include_strings
```

## 🎯 Common Use Cases

### Violin Only (Default)
```bash
python filter_gigamidi_violin.py --output_dir ./violin_midi
```
**Filters**: Violin (MIDI program 40)  
**Output**: `./violin_midi/`

### All String Instruments
```bash
python filter_gigamidi_violin.py --include_strings --output_dir ./strings
```
**Filters**: Violin, Viola, Cello, Contrabass (programs 40-43)  
**Output**: `./strings/`

### Piano Only
```bash
python filter_custom_instruments.py --instruments 0 1 2 3 4 5 6 7 --output_dir ./piano
```
**Filters**: All piano types (programs 0-7)  
**Output**: `./piano/`

### Guitar (Acoustic & Electric)
```bash
python filter_custom_instruments.py --instruments 24 25 26 27 28 29 30 31 --output_dir ./guitar
```
**Filters**: All guitar types (programs 24-31)  
**Output**: `./guitar/`

### Brass Section
```bash
python filter_custom_instruments.py --instruments 56 57 58 59 60 61 --output_dir ./brass
```
**Filters**: Trumpet, Trombone, Tuba, French Horn, Brass Section (programs 56-61)  
**Output**: `./brass/`

### Show All Available Instruments
```bash
python filter_custom_instruments.py --show_instruments
```

## 📊 Dataset Information

- **Name**: GigaMIDI (Extended)
- **Size**: 2.1+ million MIDI files
- **Loops Detected**: 9.2M non-expressive + 2.3M expressive
- **License**: CC BY-NC 4.0 (Non-commercial only)
- **Access**: Restricted (requires approval)
- **URL**: https://huggingface.co/datasets/Metacreation/GigaMIDI

### Key Dataset Fields
- `midi`: Binary MIDI data
- `midi_filename`: Original filename
- `instrument_programs`: List of MIDI program numbers
- `instrument_groups`: List of instrument names
- `loop_*`: Loop detection features
- Many more metadata fields (see test script output)

## 🎻 MIDI Program Numbers Reference

### Strings (Programs 40-47)
| Program | Instrument |
|---------|------------|
| 40 | Violin |
| 41 | Viola |
| 42 | Cello |
| 43 | Contrabass |
| 44 | Tremolo Strings |
| 45 | Pizzicato Strings |
| 46 | Orchestral Harp |
| 47 | Timpani |

### Other Popular Instruments
| Program Range | Category |
|---------------|----------|
| 0-7 | Piano |
| 24-31 | Guitar |
| 32-39 | Bass |
| 56-63 | Brass |
| 64-71 | Reed (Saxophones, Clarinets) |
| 72-79 | Pipe (Flutes, Piccolos) |

**Full reference**: Run `python filter_custom_instruments.py --show_instruments`

## ⚙️ Script Options Comparison

### filter_gigamidi_violin.py
```bash
--output_dir PATH       # Where to save files (default: ./violin_midi_files)
--include_strings       # Include viola, cello, contrabass
--no_streaming          # Download full dataset first
```

### filter_custom_instruments.py
```bash
--instruments N [N...]  # MIDI program numbers (required)
--output_dir PATH       # Where to save files (required)
--no_streaming          # Download full dataset first
--show_instruments      # Show full MIDI reference
```

## 💡 Tips & Best Practices

### Performance
- **Use streaming mode** (default) for lower memory usage
- **Expect long processing time** (hours) for the full dataset
- **Run overnight** or on a server with good internet

### Storage
- **Streaming mode**: Only saves filtered files (much smaller)
- **Non-streaming mode**: Downloads full dataset first (~100GB+)
- **Estimate**: Violin files might be 500MB-2GB total

### Customization
- Modify `filter_gigamidi_violin.py` to add custom filtering logic
- Use `filter_custom_instruments.py` as template for complex filters
- Combine multiple instrument programs for ensemble filtering

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Access denied | Request access at HuggingFace, then login |
| Too slow | Normal with streaming; consider running overnight |
| Out of memory | Remove `--no_streaming` flag to use streaming mode |
| Authentication fails | Run `huggingface-cli login` with valid token |
| Missing dependencies | Run `bash install_requirements.sh` |

## 📚 File Details

### When to Use Each Script

**Use `filter_gigamidi_violin.py` when:**
- You want violin MIDI files specifically
- You want a simple, ready-to-use solution
- You might want other string instruments too

**Use `filter_custom_instruments.py` when:**
- You want instruments other than violin
- You need to filter multiple instrument types
- You want to see all available instruments

**Use `test_gigamidi_structure.py` when:**
- You want to verify your setup works
- You're curious about dataset structure
- You want to debug authentication issues

## 📖 Documentation Hierarchy

1. **QUICKSTART.md** - Read this first for step-by-step setup
2. **SUMMARY.md** (this file) - Overview and quick reference
3. **GIGAMIDI_FILTER_README.md** - Deep dive into main script

## 🎵 Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Install dependencies
bash install_requirements.sh

# 2. Get HuggingFace token and login
# Visit: https://huggingface.co/settings/tokens
huggingface-cli login

# 3. Test the setup
python test_gigamidi_structure.py

# 4. Filter violin tracks
python filter_gigamidi_violin.py --output_dir ~/music/violin_midi

# 5. Wait for completion (might take hours)
# The script shows progress and saves files as it goes

# 6. Use your MIDI files!
ls ~/music/violin_midi/*.mid
```

## 🎓 Educational Use

This toolkit is perfect for:
- Music information retrieval research
- Training machine learning models on symbolic music
- Music theory analysis
- Educational projects
- Non-commercial music generation research

## ⚖️ License & Citation

**Dataset License**: CC BY-NC 4.0 (Non-Commercial)
- ✅ Research use
- ✅ Educational use  
- ❌ Commercial use

**Citation**:
```bibtex
@article{lee2025gigamidi,
  title={The GigaMIDI Dataset with Features for Expressive Music Performance Detection},
  author={Lee, Keon Ju Maverick and Ens, Jeff and Adkins, Sara and Sarmento, Pedro and Barthet, Mathieu and Pasquier, Philippe},
  journal={Transactions of the International Society for Music Information Retrieval (TISMIR)},
  year={2025}
}
```

## 🆘 Need Help?

1. **Check documentation**: Start with `QUICKSTART.md`
2. **Test first**: Run `test_gigamidi_structure.py`
3. **Read errors carefully**: They usually indicate authentication or network issues
4. **Check HuggingFace**: Dataset page for updates or announcements

## 🎉 You're All Set!

You now have everything you need to:
- ✅ Filter the GigaMIDI dataset for violin
- ✅ Customize for any instrument
- ✅ Process the dataset efficiently
- ✅ Save MIDI files for your research

**Start here**: `QUICKSTART.md`  
**Main script**: `python filter_gigamidi_violin.py`

Happy music research! 🎻🎵
