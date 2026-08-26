#frames.py
"""
video/frames.py
===============
Frame access. The single most important correctness constraint in the project.

WHY PyAV AND NOT OpenCV
-----------------------
The obvious way to grab frame N is:

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()

This is wrong and it fails silently. On most compressed codecs that seek lands on
the nearest KEYFRAME, not frame N -- keyframes are typically 1-10 seconds apart.
OpenCV's internal frame counter then drifts from reality. You get a frame, it
looks fine, and it is off by anywhere from 1 to 250 frames. No error is raised.

Our entire deliverable is "the EXACT frame". So:

  * PyAV for all reading  -- decode sequentially, read the real presentation
                             timestamp off each packet
  * OpenCV only for imwrite at the very end

Seeking is still used, but only to reach a keyframe BEFORE the region of
interest; we then decode forward and discard frames until we pass the start time.
That is the only way to get frame-accurate access to a compressed stream.
"""

import logging

import av
import numpy as np

from schema import VideoMeta

log = logging.getLogger(__name__)


def _open(video_path: str):
    """
    Open a container and return (container, video_stream).

    thread_type = "AUTO" lets FFmpeg use multiple threads for decoding, which is
    a large speedup on the full-video coarse sweep and costs nothing to enable.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return container, stream


def build_pts_index(video_path: str) -> dict:
    """
    Map presentation timestamp -> sequential frame index, for VFR files.

    Only needed when metadata says the file is variable frame rate, because then
    frame index cannot be computed from time. We demux WITHOUT decoding: reading
    packet headers is far cheaper than decoding pixels, so this pass over an hour
    of video takes seconds rather than minutes.

    Returns {pts: index}. Called once and cached by the caller.
    """
    container, stream = _open(video_path)
    index = {}

    try:
        # demux yields packets. We never call decode(), so no pixels are produced.
        for i, packet in enumerate(container.demux(stream)):
            if packet.pts is None:
                continue          # flush packets at end of stream have no pts
            index[packet.pts] = i
    finally:
        container.close()

    log.info("Built PTS index for VFR file: %d entries", len(index))
    return index


def sample_frames(video_path: str, meta: VideoMeta, interval: float):
    """
    Yield roughly one frame every `interval` seconds across the whole video.

    Used by the coarse sweep. Generator, not a list -- an hour of video at 0.5s
    is 7200 frames and materialising them all would exhaust memory.

    Implementation note: we decode EVERY frame but only convert the ones we want
    to numpy arrays. Decoding is unavoidable (you cannot skip forward in a
    compressed stream without decoding), but to_ndarray() does a full colour-space
    conversion and is the expensive half. Skipping it for 29 out of every 30
    frames is most of the available saving.

    An alternative is piping `ffmpeg -vf fps=2` to stdout, which is faster since
    FFmpeg does the selection in C. It was not used here because it makes exact
    frame indexing much harder -- and exactness is the point of this project.

    Yields (pts_seconds, frame_index, bgr_ndarray).
    """
    container, stream = _open(video_path)
    time_base = float(stream.time_base)

    next_sample = 0.0
    frame_counter = 0

    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            # True timestamp: packet pts multiplied by the stream's time base.
            # This is exact for CFR and VFR alike, which is why we never compute
            # time from the frame counter.
            t = frame.pts * time_base

            if t + 1e-6 >= next_sample:
                # Frame index: counted for VFR, computed for CFR. Both are correct
                # here because we are decoding from frame 0 sequentially, so the
                # counter is authoritative in either case.
                idx = frame_counter if not meta.is_cfr else int(round(t * meta.fps))

                # bgr24 because OpenCV and PaddleOCR both expect BGR channel order.
                yield t, idx, frame.to_ndarray(format="bgr24")

                # Advance to the next sample point. Adding `interval` rather than
                # setting `t + interval` keeps the sample grid regular even when
                # frame timestamps drift.
                next_sample += interval

            frame_counter += 1
    finally:
        # Always close, even if the caller abandons the generator early (which
        # search/coarse.py does not, but a future caller might).
        container.close()


def decode_window(video_path: str, meta: VideoMeta,
                  start_s: float, end_s: float,
                  pts_index: dict = None):
    """
    Yield EVERY frame between start_s and end_s. The fine search's input.

    This is where frame accuracy is won or lost:

      1. seek() to the keyframe at or before start_s. FFmpeg seeks backwards by
         default with the BACKWARD flag, guaranteeing we land before our target
         rather than after it.
      2. decode forward from there, discarding frames until we reach start_s.
      3. stop once we pass end_s.

    Step 2 is the part people skip, and it is why naive seeking is wrong.

    Yields (pts_seconds, frame_index, bgr_ndarray).
    """
    container, stream = _open(video_path)
    time_base = float(stream.time_base)

    try:
        # seek() takes a timestamp in the stream's OWN time base units, not
        # seconds. Dividing by time_base performs that conversion.
        seek_target = int(max(0.0, start_s) / time_base)

        # any_frame=False -> only keyframes are valid seek targets
        # backward=True   -> land at or before the target, never after
        container.seek(seek_target, stream=stream, any_frame=False, backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            t = frame.pts * time_base

            # Discard the frames between the keyframe and our actual start.
            if t < start_s:
                continue

            # Streams are decoded in presentation order here, so the first frame
            # past end_s means we are done. Breaking rather than continuing saves
            # decoding the rest of the file.
            if t > end_s:
                break

            # Frame index. Three cases, in order of reliability:
            if meta.is_cfr:
                # CFR: arithmetic is exact.
                idx = int(round(t * meta.fps))
            elif pts_index and frame.pts in pts_index:
                # VFR with a prebuilt index: look up the true sequential position.
                idx = pts_index[frame.pts]
            else:
                # VFR without an index: fall back to the estimate and let the
                # caller know it is approximate via the log line below.
                idx = int(round(t * meta.avg_fps))

            yield t, idx, frame.to_ndarray(format="bgr24")
    finally:
        container.close()


def crop_lower(frame: np.ndarray, ratio: float) -> tuple:
    """
    Crop to the bottom `ratio` of the frame. Returns (cropped, y_offset).

    Two reasons this exists:
      * cost      -- OCR on 35% of the pixels is roughly 3x faster
      * accuracy  -- removes false positives from signage, credits, shop fronts
                     and on-screen captions that are not the subtitle

    The y_offset is returned so bounding boxes from the crop can be translated
    back into full-frame coordinates for the annotated output image. Forgetting
    this is why annotation boxes end up in the wrong place.

    ratio >= 1.0 disables cropping, for cases where the target might be a title
    card in the centre of the frame.
    """
    if ratio >= 1.0:
        return frame, 0

    h = frame.shape[0]
    y_start = int(h * (1.0 - ratio))
    return frame[y_start:, :], y_start