# Dialogue Frame Finder 🎬

**Dialogue Frame Finder** is a high-accuracy, frame-accurate AI pipeline that takes a **YouTube video URL** and locates the exact video frame and timestamp where a given line of dialogue first appears on screen or is spoken aloud. 

It returns:
- **Timestamp** (HH:MM:SS.mmm)
- **Frame Number** (frame-accurate presentation index)
- **Extracted Text & Confidence Score**
- **Saved Frame Image** (`data/results/frame_XXXXXXXX.png`) & **Annotated Frame** showing text bounding boxes

---

## 🌟 Key Features

1. **Multi-Modal Evidence Search**:
   - 📜 **Subtitle Track Search** (Fastest)
   - 🎙️ **Speech Recognition (Whisper)** for spoken dialogue
   - 👁️ **Visual OCR Sweep (PaddleOCR)** for on-screen captions & title cards
2. **Frame-Accurate Seeking**: Uses PyAV PTS decoding rather than keyframe-approximated OpenCV seeking.
3. **Web UI & REST API**: Built-in FastAPI web app (`server.py`) with an intuitive browser interface.
4. **Command Line & Interactive CLI**: Simple terminal tools for scripting and automated runs.

---

## 🛠️ Prerequisites & Installation

### 1. System Dependencies
Make sure `ffmpeg` and `ffprobe` are installed and available on your system `PATH`:
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via `winget install ffmpeg` / `choco install ffmpeg`.
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y ffmpeg`
- **macOS**: `brew install ffmpeg`

### 2. Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required Python dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

There are three convenient ways to run Dialogue Frame Finder:

### Option 1: Web Interface (Recommended) 🌐

Start the FastAPI server using Uvicorn:

```bash
uvicorn server:app --reload
```

Once running, open your web browser and navigate to:
👉 **`http://127.0.0.1:8000/`**

Enter the **YouTube Video URL** and the **Dialogue Text**, then click **Find Frame**.

---

### Option 2: Interactive Terminal Mode 💬

Run the interactive wizard in your console:

```bash
python run_example.py
```
Follow the step-by-step prompts to input your YouTube URL, dialogue, and search preferences.

---

### Option 3: Command Line Interface (CLI) 💻

Run directly via `main.py`:

```bash
python main.py --url "YOUTUBE_VIDEO_URL" --text "TARGET_DIALOGUE"
```

#### CLI Examples

- **YouTube Video Search**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --text "never gonna give you up"
  ```
- **OCR Only (Skip Speech Recognition for Faster Results)**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --text "on-screen text" --no-audio
  ```
- **Full-Frame Scan (For Centered Titles or Title Cards)**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --text "title text" --full-frame
  ```
- **Verbose Debugging**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --text "dialogue" --verbose
  ```

---

## ⚙️ CLI Options & Exit Codes

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--url URL` | YouTube video URL | ✅ Yes | N/A |
| `--text TEXT` | Target dialogue to find | ✅ Yes | N/A |
| `--no-audio` | Skip ASR/Whisper (faster; scans OCR only) | ❌ No | `False` |
| `--full-frame` | Scan 100% of frame height (instead of bottom 35%) | ❌ No | `False` |
| `--verbose`, `-v` | Output debug level logs | ❌ No | `False` |

**Shell Exit Codes**:
- `0`: Dialogue found and saved to `data/results/`
- `1`: Target dialogue not found
- `2`: Runtime error occurred

---

## 📁 Output Artifacts

All run outputs are written to the `data/results/` directory:

- **`result.json`**: Structured response containing timestamp, frame number, detected text, confidence, and modality.
- **`frame_XXXXXXXX.png`**: Original frame image.
- **`frame_XXXXXXXX_annotated.png`**: Frame image with bounding box highlight over the detected text.

---

## 📂 Project Architecture

```
quest1/
├── server.py            # FastAPI Web Application backend
├── static/
│   └── index.html       # Web UI front-end interface
├── main.py              # Core CLI orchestrator pipeline
├── run_example.py       # Interactive terminal runner
├── config.py            # Global tunables, thresholds, and paths
├── schema.py            # Dataclasses (Candidate, Result, VideoMeta)
├── report.py            # Output formatting (Console, JSON, PNG)
├── video/               # Video metadata probing, frame decoding, subtitles
├── engines/             # PaddleOCR and faster-whisper AI model wrappers
├── matching/            # Fuzzy matching & text normalisation
├── search/              # Coarse sweep -> Fine window scan -> Resolution
├── tests/               # Unit test suite
└── requirements.txt     # Dependency definitions
```

---

## 🧪 Running Unit Tests

Run the offline pytest test suite (executes in ~1 second without loading AI models or network):

```bash
pytest tests/ -q
```

To run ground-truth synthetic clip generation and end-to-end verification:
```bash
python scripts/make_fixture.py --text "the quick brown fox" --at 3.0 --fps 25
python main.py --url tests/fixtures/fixture_25fps_480p.mp4 --text "the quick brown fox"
```