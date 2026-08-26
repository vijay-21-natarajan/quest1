#downloader.py
"""
video/downloader.py
===================
URL in, local file path out.

Build and test this module FIRST. Source sites are the biggest unknown in the
whole project -- extractors break, videos get region-locked, formats change. You
want to discover that in the first hour, not the sixth.

Uses yt-dlp as a Python library rather than a subprocess. That gives us the info
dict (which tells us whether subtitle tracks exist before we bother asking
ffmpeg) and real exceptions instead of parsing stderr.
"""

import logging
from pathlib import Path

import yt_dlp

import config

log = logging.getLogger(__name__)


def probe_url(url: str) -> dict:
    """
    Fetch metadata WITHOUT downloading.

    Cheap sanity check that runs before committing to a multi-hundred-megabyte
    download. Returns the raw info dict. Also tells us up front whether the site
    is serving subtitle tracks, which can short-circuit the entire audio stage.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        # download=False makes this metadata-only.
        info = ydl.extract_info(url, download=False)

    return info


def download(url: str, output_dir: Path = None) -> tuple:
    """
    Download the video and return (local_path, info_dict).

    Format selection is capped at 720p (see config.YTDLP_FORMAT). Higher
    resolutions cost download and decode time without helping OCR -- subtitles
    are large on-screen text and read perfectly well at 720p. If the evaluation
    video has unusually small text, raise the cap in config, not here.
    """
    output_dir = output_dir or config.INPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # %(id)s rather than %(title)s: titles contain slashes, quotes and unicode
    # that break paths on some filesystems. IDs are always filesystem-safe.
    outtmpl = str(output_dir / "%(id)s.%(ext)s")

    opts = {
        "format": config.YTDLP_FORMAT,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": config.DOWNLOAD_TIMEOUT,
        # Merge video+audio into mp4 when the format selector picks separate
        # streams. Without this we can end up with a video-only file and no audio
        # for Whisper to transcribe.
        "merge_output_format": "mp4",
        # Ask for subtitles if the site offers them. Costs nothing when absent
        # and can save the entire audio stage when present.
        "writesubtitles": True,
        "writeautomaticsub": False,   # auto-generated subs are ASR output with
                                      # worse timing than our own Whisper pass
        "subtitleslangs": ["all"],
    }

    log.info("Downloading %s", url)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # prepare_filename gives the path BEFORE post-processing. If streams were
        # merged the real extension may differ, so we verify below.
        path = Path(ydl.prepare_filename(info))

    if not path.exists():
        # Merging changed the container extension. Find the actual file by
        # matching the stem, which the ID-based template guarantees is unique.
        matches = list(output_dir.glob(f"{path.stem}.*"))
        # Exclude sidecar subtitle files that yt-dlp may have written alongside.
        matches = [m for m in matches if m.suffix not in (".vtt", ".srt", ".ass")]
        if not matches:
            raise FileNotFoundError(
                f"yt-dlp reported success but no video file found for {path.stem}"
            )
        path = matches[0]

    log.info("Downloaded to %s (%.1f MB)", path, path.stat().st_size / 1e6)

    return str(path), info


def find_sidecar_subtitles(video_path: str) -> list:
    """
    Return any .srt/.vtt/.ass files yt-dlp wrote next to the video.

    These are separate from EMBEDDED subtitle streams inside the container --
    video/subtitles.py handles those. Both paths are worth checking because
    different sites deliver subtitles differently.
    """
    p = Path(video_path)
    subs = []
    for ext in (".srt", ".vtt", ".ass", ".ssa"):
        # glob on the stem catches language-tagged names like "video.en.srt"
        subs.extend(p.parent.glob(f"{p.stem}*{ext}"))
    return [str(s) for s in subs]
