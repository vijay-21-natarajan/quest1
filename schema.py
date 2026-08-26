#schema.py
"""
schema.py
=========
Pure data containers passed between pipeline stages.

Why a separate file: every function signature in this project becomes
self-documenting once these exist. `def refine(window: Candidate) -> list[FrameHit]`
tells you more than `def refine(window) -> list`.

Deliberately has NO logic and NO heavy imports (no cv2, no paddle, no av).
That means tests can import it instantly, and there is no circular-import risk.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoMeta:
    """Everything ffprobe told us about the downloaded file."""

    path: str                    # absolute path to the local video file
    fps: float                   # nominal frames per second (r_frame_rate)
    avg_fps: float               # average frames per second (avg_frame_rate)
    is_cfr: bool                 # True when fps == avg_fps -> constant frame rate.
                                 # This single flag decides how frame numbers are
                                 # computed. Getting it wrong gives silently wrong
                                 # answers, which is the worst kind.
    duration: float              # total length in seconds
    width: int                   # frame width in pixels
    height: int                  # frame height in pixels
    nb_frames: Optional[int]     # total frame count if the container reports it
                                 # (often None for streamed/remuxed files)
    has_subtitle_stream: bool    # True if ffprobe found an embedded subtitle track


@dataclass
class Candidate:
    """
    A stretch of time where one evidence source thinks the target might live.

    Produced by search/coarse.py, consumed by search/fine.py. This is the
    hand-off between the cheap search and the expensive search.
    """

    start_s: float               # window start in seconds
    end_s: float                 # window end in seconds
    source: str                  # "subtitle" | "audio" | "visual" -- which locator
                                 # found it. Kept because agreement between two
                                 # independent sources is itself a confidence signal.
    score: float                 # 0-100 fuzzy match score that produced this window
    evidence: str = ""           # the actual text that matched, for debugging and
                                 # for explaining the result to a human

    def merge(self, other: "Candidate") -> "Candidate":
        """
        Combine two overlapping windows into one.

        Takes the union of the time ranges and the higher score. Sources get
        joined with "+" so a window backed by both audio and subtitles is
        visibly stronger than one backed by only a single source.
        """
        return Candidate(
            start_s=min(self.start_s, other.start_s),
            end_s=max(self.end_s, other.end_s),
            source="+".join(sorted({self.source, other.source})),
            score=max(self.score, other.score),
            evidence=self.evidence or other.evidence,
        )


@dataclass
class FrameHit:
    """
    One decoded frame that has been OCR'd and scored against the target.

    search/fine.py produces a list of these. It does NOT decide which one is
    the answer -- that is resolve.py's job. Keeping scoring and deciding in
    separate files is what makes the "how do you handle ambiguity" question
    answerable by pointing at a single file.
    """

    frame_index: int             # absolute frame number from the start of the video
    pts_seconds: float           # true presentation timestamp, read off the packet.
                                 # NOT frame_index/fps -- that assumes CFR.
    text: str                    # raw text OCR returned for this frame
    ocr_confidence: float        # 0-1, how sure the OCR model is of the characters
    match_score: float           # 0-100, how well `text` matches the target phrase
    bbox: Optional[list] = None  # polygon of the matched text region, for the
                                 # annotated output image


@dataclass
class Result:
    """The final answer. report.py turns this into console output, JSON and PNGs."""

    found: bool                  # False when nothing crossed the threshold
    timestamp: str = ""          # "HH:MM:SS.sss" -- the required output format
    pts_seconds: float = 0.0     # same instant as a raw float, for machines
    frame_index: int = -1        # the required frame number
    text: str = ""               # the extracted dialogue text
    confidence: float = 0.0      # 0-1 blended confidence (see resolve.py)
    modality: str = ""           # "visual" | "audio_only" | "subtitle_only" --
                                 # tells the user WHICH channel produced the answer,
                                 # so an audio-only hit is never mistaken for a
                                 # confirmed on-screen detection
    image_path: str = ""         # saved frame PNG
    annotated_path: str = ""     # same frame with the OCR box drawn on it
    fade_frames: int = 0         # how many frames the text took to ramp in.
                                 # Non-zero means "first appears" was a judgement
                                 # call, and we are being honest about it.
    alternates: list = field(default_factory=list)
                                 # other plausible hits, so an uncertain run
                                 # reports options instead of guessing
    notes: list = field(default_factory=list)
                                 # human-readable warnings collected during the run