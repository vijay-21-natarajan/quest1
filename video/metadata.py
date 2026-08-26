#metadata.py
"""
video/metadata.py
=================
ffprobe wrapper. Answers "what is inside this file?"

This is the highest-risk file in the project. Every number reported as an answer
-- the frame index especially -- passes through the conversions here, and the
failure mode is silent: a wrong frame number looks exactly like a right one.

The critical piece is the CFR/VFR determination. Almost every tutorial computes
frame numbers as `round(timestamp * fps)`. That is only valid for constant frame
rate video. For variable frame rate the fps figure is an average and the formula
drifts, producing an answer that is plausibly wrong.
"""

import json
import logging
import subprocess
from fractions import Fraction

from schema import VideoMeta

log = logging.getLogger(__name__)


def _parse_rate(rate_str: str) -> float:
    """
    ffprobe reports frame rates as fractions like "30000/1001" (NTSC 29.97).

    Parsing as a Fraction rather than eval-ing the string keeps this safe and
    exact. Returns 0.0 for the "0/0" that ffprobe emits for streams with no
    meaningful rate.
    """
    if not rate_str or rate_str == "0/0":
        return 0.0
    try:
        return float(Fraction(rate_str))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(video_path: str) -> VideoMeta:
    """
    Run ffprobe and return a populated VideoMeta.

    -show_streams gives per-stream data (fps, resolution, codec)
    -show_format  gives container-level data (duration)
    -of json      machine-readable output instead of ffprobe's text format
    """
    cmd = [
        "ffprobe",
        "-v", "error",           # suppress banner noise, keep real errors
        "-show_streams",
        "-show_format",
        "-of", "json",
        video_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])

    # First video stream. Files occasionally carry a second one (embedded cover
    # art is stored as a video stream), so we take the first rather than assuming
    # there is exactly one.
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    v = video_streams[0]

    has_subs = any(s.get("codec_type") == "subtitle" for s in streams)

    # --- The CFR/VFR determination ------------------------------------------
    #
    # r_frame_rate   : the "real base" rate -- the lowest rate that can represent
    #                  every frame timestamp exactly
    # avg_frame_rate : total frames divided by duration
    #
    # For constant frame rate video these are identical. When they diverge, the
    # file has variable frame rate and frame index CANNOT be derived from a
    # timestamp arithmetically -- it must be counted.
    r_fps = _parse_rate(v.get("r_frame_rate", "0/0"))
    avg_fps = _parse_rate(v.get("avg_frame_rate", "0/0"))

    # Tolerance rather than exact equality: rounding in the container metadata
    # produces tiny differences (29.970030 vs 29.970000) that are not real VFR.
    is_cfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) < 0.01)

    if not is_cfr:
        log.warning(
            "Variable frame rate detected (r=%.4f, avg=%.4f). Frame numbers will "
            "be counted from decoded packets rather than computed.", r_fps, avg_fps
        )

    # Duration: prefer the container's value, fall back to the stream's. Some
    # remuxed files report one but not the other.
    duration = float(data.get("format", {}).get("duration")
                     or v.get("duration") or 0.0)

    # nb_frames is frequently absent for streamed or remuxed files. None is a
    # legitimate value here -- callers must not assume it exists.
    nb_frames = v.get("nb_frames")
    nb_frames = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else None

    meta = VideoMeta(
        path=video_path,
        fps=r_fps or avg_fps,      # r_fps preferred; fall back if it is missing
        avg_fps=avg_fps or r_fps,
        is_cfr=is_cfr,
        duration=duration,
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        nb_frames=nb_frames,
        has_subtitle_stream=has_subs,
    )

    log.info(
        "Probed: %dx%d, %.3f fps (%s), %.1fs, subs=%s",
        meta.width, meta.height, meta.fps,
        "CFR" if meta.is_cfr else "VFR", meta.duration, meta.has_subtitle_stream,
    )

    return meta


def time_to_frame(meta: VideoMeta, seconds: float) -> int:
    """
    Convert a timestamp to a frame index.

    ONLY valid for CFR. For VFR this is an estimate and callers must treat it as
    such -- video/frames.py counts real frames instead.

    round() rather than int(): int() truncates, so a frame presented at exactly
    1.0000001s would be assigned to the previous frame. round() picks the nearest,
    which is what "the frame at this instant" means.
    """
    if meta.fps <= 0:
        return 0
    return int(round(seconds * meta.fps))


def frame_to_time(meta: VideoMeta, frame_index: int) -> float:
    """
    Convert a frame index back to a timestamp. Same CFR caveat as above.

    In practice the pipeline avoids this: frames.py reads the true presentation
    timestamp off each decoded packet, which is exact for both CFR and VFR. This
    function exists for logging and for estimating window boundaries.
    """
    if meta.fps <= 0:
        return 0.0
    return frame_index / meta.fps


def format_timestamp(seconds: float) -> str:
    """
    Seconds -> "HH:MM:SS.sss", the exact output format the brief requires.

    Built with divmod rather than datetime/strftime because strftime cannot
    format durations over 24 hours and silently wraps. divmod is unbounded.
    """
    if seconds < 0:
        seconds = 0.0

    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)

    # :06.3f -> width 6 including the decimal point and 3 decimals, zero-padded.
    # Gives "07.900" rather than "7.900", keeping the column aligned.
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"