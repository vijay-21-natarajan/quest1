#audio.py
"""
video/audio.py
==============
Extracts an audio track suitable for Whisper. One job, one function.

Whisper was trained on 16 kHz mono audio and resamples internally to that rate
regardless of what you give it. Feeding it 48 kHz stereo means an extra
resampling step and, in some pipelines, a small accuracy loss for zero benefit.
So we produce exactly what the model wants.
"""

import logging
import subprocess
from pathlib import Path

import config

log = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract 16 kHz mono PCM WAV from the video. Returns the wav path.

    Returns an existing file without re-extracting, so re-running the pipeline
    during development does not repeat the work.
    """
    video_path = Path(video_path)
    output_path = Path(output_path) if output_path else \
        config.AUDIO_DIR / f"{video_path.stem}.wav"

    if output_path.exists() and output_path.stat().st_size > 0:
        log.info("Reusing existing audio: %s", output_path)
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                      # overwrite without prompting
        "-i", str(video_path),
        "-vn",                     # drop the video stream entirely -- we only
                                   # want audio, and skipping video decode makes
                                   # this many times faster
        "-acodec", "pcm_s16le",    # uncompressed 16-bit PCM: no decode step for
                                   # Whisper, no lossy artefacts
        "-ar", "16000",            # 16 kHz -- Whisper's native rate
        "-ac", "1",                # mono; stereo carries no extra speech info
        "-loglevel", "error",
        str(output_path),
    ]

    log.info("Extracting audio -> %s", output_path)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        # Not fatal. A video with no audio stream is a legitimate input -- the
        # target may be a silent title card. We return None and let the pipeline
        # fall back to the other evidence sources.
        log.warning("Audio extraction failed: %s", proc.stderr.strip())
        return None

    return str(output_path)


def has_audio_stream(video_path: str) -> bool:
    """
    Cheap check for an audio stream before attempting extraction.

    Saves loading the Whisper model (several seconds and hundreds of MB) for a
    video that has no audio at all.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",       # audio streams only
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return bool(proc.stdout.strip())