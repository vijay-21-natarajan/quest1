#config.py
"""
config.py
=========
Every tunable number in the project lives here and nowhere else.

The rule: if a value could reasonably be argued about, it belongs in this file.
When an interviewer says "make the matching stricter" or "sample more finely",
you change one line here while still talking to them. If thresholds were
scattered across five modules you would be hunting instead of answering.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Anchor everything to this file's location, not the current working directory.
# Otherwise `python main.py` works but `python /full/path/main.py` writes files
# into whatever directory you happened to be standing in.
ROOT = Path(__file__).parent.resolve()

DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"      # downloaded video
AUDIO_DIR = DATA_DIR / "audio"      # extracted 16 kHz wav
FRAMES_DIR = DATA_DIR / "frames"    # debug / sampled frames
RESULTS_DIR = DATA_DIR / "results"  # final PNG + result.json


def ensure_dirs() -> None:
    """
    Create the working directories if they are missing.

    Called once from main.py. Empty directories do not survive git, so the repo
    ships without them and we recreate them at runtime. This is why data/ can be
    fully gitignored -- nothing downstream ever has to handle a missing folder.
    """
    for d in (INPUT_DIR, AUDIO_DIR, FRAMES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Coarse search -- the cheap pass over the whole video
# ---------------------------------------------------------------------------

# Seconds between sampled frames in the visual sweep.
# 0.5 rather than 1.0 deliberately: subtitles typically display for 1.5-2s, but a
# short line can be on screen for well under a second and would fall cleanly
# between two 1-second samples. Missing it entirely is a silent failure.
COARSE_SAMPLE_INTERVAL = 0.5

# Fuzzy score (0-100) a sampled frame must reach to become a candidate window.
# Kept loose on purpose -- this stage should over-produce candidates. A false
# positive costs a few hundred frames of OCR in the fine pass; a false negative
# loses the answer forever.
FUZZY_THRESHOLD_COARSE = 70

# Seconds of padding added either side of a candidate before the fine search.
# Covers ASR timestamp drift and the fade-in that precedes full opacity.
WINDOW_PADDING_S = 2.0

# Safety valve: never fine-search a window longer than this. Protects against a
# pathological merge swallowing half the video.
MAX_WINDOW_S = 12.0


# ---------------------------------------------------------------------------
# Fine search and resolution -- the expensive per-frame pass
# ---------------------------------------------------------------------------

# Fuzzy score a decoded frame must reach to count as "the text is present".
# Stricter than the coarse threshold because here we are committing to an answer.
FUZZY_THRESHOLD_FINE = 82

# The stability rule. A frame only counts as the first appearance if it AND the
# next (STABILITY_FRAMES - 1) frames all clear the threshold.
#
# This is what makes "first appears" a definition rather than a guess. Subtitles
# fade in over several frames, and OCR on a half-opacity frame produces noisy
# scores that can spike above threshold for a single frame and drop back.
# Requiring persistence kills those flukes.
STABILITY_FRAMES = 3

# Below this score a frame is considered to have no text at all. Used to measure
# the fade ramp: the gap between "first frame with any signal" and "first frame
# that is stable" is reported as fade_frames.
FADE_FLOOR_SCORE = 40


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

# Fraction of the frame height, measured from the bottom, that gets OCR'd.
# Subtitles live in the lower third. Cropping cuts OCR cost roughly threefold
# AND removes false positives from signage, credits and on-screen captions.
# Set to 1.0 to disable cropping when the target might be a title card.
SUBTITLE_CROP_RATIO = 0.35

# PaddleOCR language pack. Whisper detects spoken language automatically, but
# OCR must be told. Exposed here so a non-English evaluation video is a one-line
# change rather than a code change.
OCR_LANG = os.getenv("OCR_LANG", "en")

# Minimum per-detection OCR confidence. Below this the text region is discarded
# as noise before it ever reaches the fuzzy matcher.
OCR_MIN_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# ASR (faster-whisper)
# ---------------------------------------------------------------------------

# "tiny" / "base" / "small" / "medium" / "large-v3".
# "small" is the sweet spot on CPU: good enough to locate a phrase within a few
# seconds, and it does not need a GPU. We only need a rough time window from
# audio -- the exact frame is re-derived from pixels afterwards -- so paying for
# a larger model here buys almost nothing.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")

# "cpu" or "cuda". int8 quantisation on CPU, float16 on GPU.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = "int8" if WHISPER_DEVICE == "cpu" else "float16"

# Score a transcript window must reach to produce an audio candidate.
# Lower than the visual threshold because ASR mishears more than OCR misreads,
# and an audio hit only has to get us near the right place.
FUZZY_THRESHOLD_AUDIO = 65

# Longest gap (seconds) allowed between two consecutive words inside one
# find_best_span() window before that window is rejected.
#
# words[] is a FLAT list -- when vad_filter skips a long silent/musical stretch,
# the words right before and after that stretch are still adjacent in the list
# even though they are a minute or more apart in real time. Without this guard,
# a sliding window can bridge that gap and stitch two unrelated lines into one
# "match" that scores high but spans nonsense -- and that span's wide time range
# then poisons merge_candidates() into fusing two real, distinct candidates into
# one giant window. Set above the longest natural in-line pause (e.g. a dramatic
# "that was... odd" trails ~1.2s) and well below a real scene-silence gap.
MAX_WORD_GAP_S = 3.0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# yt-dlp format selector. Cap at 720p: higher resolutions cost download time and
# decode time without helping OCR, since subtitles are large on-screen text.
# Falls back progressively so an unusual stream layout still downloads.
YTDLP_FORMAT = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"

# Seconds before a download is abandoned.
DOWNLOAD_TIMEOUT = 600