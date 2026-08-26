# Dialogue Frame Finder

Finds the exact video frame where a given line of dialogue first appears on
screen, and reports the timestamp, frame number, extracted text and frame image.

```
Timestamp : 00:07:31.900
Frame     : 13553
Text      : "My mind rebels at stagnation"
Confidence: 0.912
Image     : data/results/frame_00013553.png
```

## Install

```bash
# ffmpeg + ffprobe must be on PATH
sudo apt install ffmpeg          # or: brew install ffmpeg

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Interactive Mode)

The easiest way to run the project:

```bash
python run_example.py
```

This will interactively ask you for:
1. Video URL (YouTube or local file path)
2. Target dialogue to search for
3. Whether to enable verbose output
4. Whether to skip audio processing

## Manual Command Line Usage

### Basic Syntax

```bash
python main.py --url "VIDEO_URL_OR_PATH" --text "target dialogue"
```

### Examples

#### Example 1: YouTube Video
```bash
python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --text "never gonna give you up"
```

#### Example 2: Local Video File
```bash
python main.py --url "C:\Users\YourName\Videos\movie.mp4" --text "hello world"
```

#### Example 3: With Verbose Logging
```bash
python main.py --url "https://youtube.com/watch?v=xyz" --text "my dialogue" --verbose
```

#### Example 4: Skip Audio (Faster - Only OCR)
```bash
python main.py --url "video.mp4" --text "on-screen text" --no-audio
```

#### Example 5: Full Frame OCR (For Title Cards)
```bash
python main.py --url "video.mp4" --text "centered title" --full-frame
```

## Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--url URL` | Video URL (YouTube, etc.) or local file path | ✅ Yes |
| `--text TEXT` | The dialogue phrase to search for | ✅ Yes |
| `--verbose` or `-v` | Show detailed debug logs | ❌ No |
| `--no-audio` | Skip speech recognition (faster, OCR only) | ❌ No |
| `--full-frame` | OCR entire frame instead of lower third (for title cards) | ❌ No |

Exit codes: `0` found, `1` not found, `2` error — so it can be scripted.

## Understanding the Output

### 1. Console Output

The script will show you a summary like this:

```
==============================================================
RESULT: found on-screen
==============================================================
Target    : "my mind rebels at stagnation"
Timestamp : 00:02:15.500
Frame     : 3238
Text      : "My mind rebels at stagnation"
Confidence: 0.95
Modality  : visual

Frame saved to: data/results/frame_00003238.png
Annotated:      data/results/frame_00003238_annotated.png
==============================================================
```

### 2. Result Files

After running, check the `data/results/` folder:

- **`result.json`** - Complete result data in JSON format
- **`frame_XXXXXXXX.png`** - The exact frame where dialogue appears
- **`frame_XXXXXXXX_annotated.png`** - Same frame with text highlighted

### 3. Result Types

#### ✅ **Found on Screen**
```
Modality: visual
```
The dialogue was found as on-screen text (subtitles, captions, etc.)

#### 🎤 **Found in Audio Only**
```
Modality: audio_only
```
The phrase was spoken but not shown on screen. Timestamp is from speech recognition.

#### 🔍 **Found in Both**
```
Modality: both
```
Both spoken AND shown on screen. Timestamp shows first visual appearance.

#### ❌ **Not Found**
```
found: false
```
The dialogue was not detected in audio or video.

## Folder Structure

```
quest1/
├── data/
│   ├── input/          # Downloaded videos
│   ├── audio/          # Extracted audio files (.wav)
│   ├── frames/         # (Usually empty - frames processed in memory)
│   └── results/        # Output: result.json + frame images
├── main.py             # Main entry point
└── run_example.py      # Interactive runner script
```

## Tips for Best Results

### 1. **Exact Phrase Matching**
The search uses fuzzy matching, so small variations are OK:
- Target: `"Hello world"`
- Will match: `"hello world"`, `"Hello World!"`, `"HELLO WORLD"`

### 2. **For On-Screen Text Only**
If you know the dialogue is definitely on screen, skip audio for faster results:
```bash
python main.py --url "video.mp4" --text "subtitle text" --no-audio
```

### 3. **For Title Cards or Centered Text**
By default, only the bottom 35% of the frame is scanned (for subtitles). For title cards:
```bash
python main.py --url "video.mp4" --text "title text" --full-frame
```

### 4. **Debugging Issues**
If results seem wrong, run with `--verbose` to see detailed logs:
```bash
python main.py --url "video.mp4" --text "dialogue" --verbose
```

## Troubleshooting

### Issue: "Video not found"
- Check the URL is correct
- For YouTube, make sure the video is public/available
- For local files, use absolute paths

### Issue: "Dialogue not found" but you know it's there
- Try with `--verbose` to see what's being detected
- Check if text is in the bottom third of frame, or use `--full-frame`
- Verify the exact wording matches

### Issue: Wrong timestamp
- Timestamps from audio (`modality: audio_only`) can have slight drift
- Visual timestamps (`modality: visual`) are frame-accurate
- Check if there are multiple instances of the phrase

### Issue: Slow processing
- Use `--no-audio` if you don't need speech recognition
- First run downloads models (PaddleOCR, Whisper) - subsequent runs are faster
- Higher resolution videos take longer

## Environment Variables

You can customize behavior with environment variables:

```bash
# Use different Whisper model (tiny/base/small/medium/large-v3)
export WHISPER_MODEL=tiny

# Use GPU for Whisper (if available)
export WHISPER_DEVICE=cuda

# Change OCR language
export OCR_LANG=en

# Then run:
python main.py --url "video.mp4" --text "dialogue"
```

## How it works

Three independent evidence sources locate a candidate time window cheaply
(subtitles, then audio, then a visual sweep), and only then does per-frame OCR
run on the few hundred frames that survive. Full reasoning in
[`docs/design.md`](docs/design.md); diagrams in
[`docs/architecture.md`](docs/architecture.md).

## Tests

```bash
pytest tests/ -q
```

42 tests covering normalisation, fuzzy matching, CFR/VFR frame arithmetic and
the first-appearance rule. All run in under a second with **no video, no models
and no network** — the pipeline's hardest logic is deliberately built from pure
functions over dataclasses so it can be tested this way.

### Ground-truth validation

```bash
python scripts/make_fixture.py --text "the quick brown fox" --at 3.0 --fps 25
python main.py --url tests/fixtures/fixture_25fps_480p.mp4 --text "the quick brown fox"
```

`make_fixture.py` burns known text into a generated clip at a known frame, at any
frame rate and resolution, with a configurable fade. Correctness is asserted
against real ground truth rather than assumed — offline, in CI, with no
copyrighted media in the repository.

## Layout

```
main.py              orchestrator -- read this first
config.py            every tunable threshold, in one place
schema.py            dataclasses passed between stages
report.py            console + JSON + PNG output

video/               downloader, metadata (CFR/VFR), audio, subtitles, frames
engines/             PaddleOCR and faster-whisper wrappers
matching/            normalisation and fuzzy scoring (pure functions)
search/              coarse -> fine -> resolve
scripts/             ground-truth fixture generator
tests/               42 tests, no I/O
docs/                design.md, architecture.md, prompts.md
```