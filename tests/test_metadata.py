#test_metadata.py
"""
tests/test_metadata.py
======================
Frame-number arithmetic and timestamp formatting.

The most important test file in the project, because these failures are SILENT.
A wrong frame number looks exactly like a right one -- no exception, no warning,
just an answer that is quietly incorrect. Tests are the only thing that catches it.

No video needed: VideoMeta is a plain dataclass, so we construct one directly.
"""

import pytest

from schema import VideoMeta
from video.metadata import time_to_frame, frame_to_time, format_timestamp, _parse_rate


def make_meta(fps=25.0, is_cfr=True):
    return VideoMeta(
        path="dummy.mp4", fps=fps, avg_fps=fps, is_cfr=is_cfr,
        duration=600.0, width=1280, height=720,
        nb_frames=int(600 * fps), has_subtitle_stream=False,
    )


def test_parse_ntsc_fraction():
    # ffprobe reports NTSC as 30000/1001, not 29.97. Naive float parsing of the
    # string would raise; the Fraction path handles it exactly.
    assert _parse_rate("30000/1001") == pytest.approx(29.97, abs=0.01)


def test_parse_handles_zero_rate():
    # ffprobe emits "0/0" for streams with no meaningful rate. Must not raise
    # ZeroDivisionError -- it should degrade to 0.0 and let the caller decide.
    assert _parse_rate("0/0") == 0.0
    assert _parse_rate("") == 0.0


def test_time_to_frame_at_25fps():
    meta = make_meta(25.0)
    assert time_to_frame(meta, 0.0) == 0
    assert time_to_frame(meta, 1.0) == 25
    assert time_to_frame(meta, 60.0) == 1500


def test_time_to_frame_rounds_not_truncates():
    # int() truncation would assign a frame presented at 1.0000001s to frame 24.
    # round() picks the nearest, which is what "the frame at this instant" means.
    meta = make_meta(25.0)
    assert time_to_frame(meta, 0.99999) == 25
    assert time_to_frame(meta, 1.00001) == 25


def test_roundtrip_time_frame_time():
    meta = make_meta(30.0)
    for frame in (0, 1, 500, 13553):
        t = frame_to_time(meta, frame)
        assert time_to_frame(meta, t) == frame


def test_ntsc_dropframe_roundtrip():
    # 29.97 fps is where naive fps math accumulates visible error over long
    # durations. At 20 minutes in, a 30-vs-29.97 mistake is off by ~36 frames.
    meta = make_meta(30000 / 1001)
    frame = 36000
    t = frame_to_time(meta, frame)
    assert time_to_frame(meta, t) == frame


def test_zero_fps_does_not_crash():
    # Corrupt or unusual files report no frame rate. Returning 0 is better than
    # a ZeroDivisionError that takes down the whole run.
    meta = make_meta(0.0)
    assert time_to_frame(meta, 10.0) == 0
    assert frame_to_time(meta, 100) == 0.0


def test_format_timestamp_matches_required_format():
    # The brief specifies HH:MM:SS.sss exactly.
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(451.9) == "00:07:31.900"
    assert format_timestamp(3661.5) == "01:01:01.500"


def test_format_timestamp_zero_pads_seconds():
    # "00:00:07.900", not "00:00:7.900" -- keeps output columns aligned.
    assert format_timestamp(7.9) == "00:00:07.900"


def test_format_timestamp_beyond_24_hours():
    # datetime.strftime silently wraps past 24h. divmod does not. Unlikely input,
    # but a wrong answer here would be baffling to debug.
    assert format_timestamp(90000.0).startswith("25:")


def test_format_timestamp_negative_clamps():
    assert format_timestamp(-5.0) == "00:00:00.000"