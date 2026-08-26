#fine.py
"""
search/fine.py
==============
The expensive pass. Decodes EVERY frame in one candidate window and scores it.

Deliberately dumb: this module SCORES frames, it does not DECIDE anything.
Choosing which frame is "the first appearance" belongs to resolve.py.

That split exists so the ambiguity logic -- the part the evaluator will press
hardest on -- lives in one small readable file instead of being tangled up with
decode loops and OCR calls.

WHY NOT BINARY SEARCH
---------------------
Tempting: "is the text present at the midpoint?" then halve the window. It is
wrong here. Binary search needs monotonicity -- once true, always true. Subtitles
fade in, so the score ramps through a noisy band where a frame can spike above
threshold and drop back. Binary search on a non-monotonic signal lands
unpredictably.

The window is small anyway (a few hundred frames), so a linear scan is cheap and
correct. Choosing the correct-but-boring algorithm over the clever-but-wrong one
is worth being able to explain.
"""

import logging

import config
from matching import similarity
from schema import FrameHit, VideoMeta
from video import frames as frames_mod

log = logging.getLogger(__name__)


def scan_window(video_path: str, meta: VideoMeta, candidate, target: str,
                ocr_engine, pts_index: dict = None) -> list:
    """
    OCR every frame in the candidate window. Returns FrameHits in time order.

    Time order matters -- resolve.py walks this list forward looking for the
    first sustained run above threshold, which is only meaningful if the list is
    chronological. decode_window yields in presentation order, so it already is;
    this docstring exists so nobody "optimises" by sorting it differently.
    """
    hits = []

    for t, idx, frame in frames_mod.decode_window(
        video_path, meta, candidate.start_s, candidate.end_s, pts_index
    ):
        # Same crop as the coarse pass. Consistency matters: if the two passes
        # cropped differently, a frame could score above threshold in one and
        # below in the other, which is impossible to debug.
        cropped, y_offset = frames_mod.crop_lower(frame, config.SUBTITLE_CROP_RATIO)

        text, conf, bbox = ocr_engine.read_joined(cropped)

        # Translate the bounding box from crop coordinates back to full-frame
        # coordinates. Skipping this puts the annotation box in the wrong place
        # on the output image -- a small bug that looks very bad in a demo.
        if bbox and y_offset:
            bbox = [[pt[0], pt[1] + y_offset] for pt in bbox]

        hits.append(FrameHit(
            frame_index=idx,
            pts_seconds=t,
            text=text,
            ocr_confidence=conf,
            match_score=similarity.score(target, text) if text else 0.0,
            bbox=bbox,
        ))

    log.info("Fine scan: %d frames in window %.2f-%.2fs",
             len(hits), candidate.start_s, candidate.end_s)

    if hits:
        best = max(hits, key=lambda h: h.match_score)
        log.info("  best frame %d @ %.3fs score=%.1f",
                 best.frame_index, best.pts_seconds, best.match_score)

    return hits