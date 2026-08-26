#subtitles.py
"""
video/subtitles.py
==================
Reads subtitle cues from wherever they might be hiding.

Subtitles are the cheapest possible evidence source. When a track exists we get
the target's timestamp for essentially zero compute -- no model loading, no
decoding, no OCR. Always check here first.

Two possible locations, both handled:
  * embedded  -- a subtitle stream inside the video container (ffmpeg extracts it)
  * sidecar   -- a separate .srt/.vtt file yt-dlp downloaded alongside the video

Parsing is delegated to pysubs2, which handles SRT, VTT, ASS and SSA. Hand-rolling
an SRT parser is a classic way to lose an hour to encoding and timecode edge cases.
"""

import logging
import subprocess
from pathlib import Path

import pysubs2

import config

log = logging.getLogger(__name__)


def extract_embedded(video_path: str, output_path: str = None) -> str:
    """
    Pull an embedded subtitle stream out of the container into an .srt file.

    Returns the path, or None when there is no subtitle stream -- which is a
    normal outcome, not an error. The pipeline must continue either way.
    """
    video_path = Path(video_path)
    output_path = Path(output_path) if output_path else \
        config.INPUT_DIR / f"{video_path.stem}.embedded.srt"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:s:0",           # stream 0, subtitle type, first one
        "-c:s", "srt",             # normalise to SRT whatever the source format
        "-loglevel", "error",
        str(output_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0 or not output_path.exists():
        log.info("No embedded subtitle stream extracted (this is normal)")
        return None

    return str(output_path)


def parse(subtitle_path: str) -> list:
    """
    Parse a subtitle file into [(start_seconds, end_seconds, text), ...].

    pysubs2 stores times in MILLISECONDS as integers. Dividing by 1000.0 converts
    to the seconds everything else in this project uses. Mixing the two units is
    an easy and very confusing bug.
    """
    try:
        subs = pysubs2.load(subtitle_path)
    except Exception as e:
        # Malformed subtitle files are common. Degrade to "no subtitles" rather
        # than crashing the run -- the other two evidence sources still work.
        log.warning("Could not parse %s: %s", subtitle_path, e)
        return []

    cues = []
    for line in subs:
        # plaintext strips ASS/SSA styling tags such as {\an8} and \N line breaks,
        # which would otherwise pollute the text we hand to the fuzzy matcher.
        text = line.plaintext.strip()
        if text:
            cues.append((line.start / 1000.0, line.end / 1000.0, text))

    log.info("Parsed %d subtitle cues from %s", len(cues), Path(subtitle_path).name)
    return cues


def gather_cues(video_path: str, sidecar_paths: list = None) -> list:
    """
    Collect cues from every available subtitle source, embedded and sidecar.

    Returns a single merged list. Duplicates across sources are harmless: the
    coarse search deduplicates overlapping candidate windows anyway, so a phrase
    found in two subtitle files simply produces one window with a higher score.
    """
    cues = []

    embedded = extract_embedded(video_path)
    if embedded:
        cues.extend(parse(embedded))

    for path in (sidecar_paths or []):
        cues.extend(parse(path))

    return cues