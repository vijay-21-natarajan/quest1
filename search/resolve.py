#resolve.py
"""
search/resolve.py
=================
The decision file. Turns a list of scored frames into THE answer.

This is the smallest file in the project and the one that matters most in the
interview, because it is where "how do you handle ambiguity" gets answered.

THE PROBLEM WITH "FIRST APPEARS"
--------------------------------
Naively: the first frame whose score clears the threshold. Consider a real fade-in:

    frame 13550  score  8    nothing on screen
    frame 13551  score 31    text starting to fade in
    frame 13552  score 55    half opacity, OCR guessing
    frame 13553  score 91    readable
    frame 13554  score 96    fully opaque
    frame 13555  score 96

Where does the text "first appear"? 13551 is when photons changed. 13553 is when
it became machine-readable. There is no objectively correct answer -- so the
answer must be a DEFINITION, stated and defended, not a threshold picked by feel.

OUR DEFINITION
--------------
The first frame that clears the threshold AND is followed by (STABILITY_FRAMES-1)
consecutive frames that also clear it.

The persistence requirement is the important half. Without it, a single noisy
frame at half opacity that happens to OCR well becomes the answer. Requiring the
text to STAY readable filters those flukes out.

We then report the fade ramp length alongside the answer, so the user can see
that a judgement was made rather than having it hidden.
"""

import logging

import config
from schema import FrameHit, Result
from video.metadata import format_timestamp

log = logging.getLogger(__name__)


def find_first_stable(hits: list) -> tuple:
    """
    Apply the stability rule. Returns (first_stable_hit, fade_frame_count).

    Walks the chronological hit list. For each frame above threshold, looks ahead
    to confirm the next STABILITY_FRAMES-1 frames are also above it. The first
    frame that passes both checks is the answer.

    Returns (None, 0) when nothing qualifies -- a legitimate outcome that the
    caller reports honestly rather than papering over.
    """
    if not hits:
        return None, 0

    n = len(hits)
    need = config.STABILITY_FRAMES

    for i, hit in enumerate(hits):
        if hit.match_score < config.FUZZY_THRESHOLD_FINE:
            continue

        # Look-ahead window. Clamped to the list end so a match in the final few
        # frames of the window is not rejected purely for lack of room -- the
        # text may simply continue past our window boundary.
        lookahead = hits[i:min(i + need, n)]

        if all(h.match_score >= config.FUZZY_THRESHOLD_FINE for h in lookahead):
            # Measure the fade: walk BACKWARDS from the stable frame counting
            # frames that carried some signal but not enough to qualify. That
            # count is how many frames the text spent ramping in, and it is the
            # honest measure of how much judgement went into this answer.
            fade = 0
            j = i - 1
            while j >= 0 and hits[j].match_score >= config.FADE_FLOOR_SCORE:
                fade += 1
                j -= 1

            return hit, fade

    return None, 0


def compute_confidence(hit: FrameHit, candidate, fade_frames: int) -> float:
    """
    Blend the available signals into a single 0-1 confidence.

    Four components, weighted by how much each is worth trusting:

      match score (0.40)  -- how well the OCR text matches the target. The
                             primary signal; everything else modifies it.
      OCR confidence (0.25) -- how sure the model is of the characters. A perfect
                             fuzzy match on text the model barely read is weaker
                             than the same match on crisp text.
      cross-source (0.25) -- did more than one independent channel point here?
                             Audio and OCR agreeing is strong evidence because
                             their failure modes are completely unrelated.
      fade penalty (0.10) -- a long fade means the first-appearance frame was
                             more of a judgement call, so confidence drops.

    Surfacing this number rather than hiding it is deliberate. A wrong confident
    answer is worse than an honest uncertain one.
    """
    match_component = (hit.match_score / 100.0) * 0.40
    ocr_component = hit.ocr_confidence * 0.25

    # source is "audio+visual" when two channels merged, so counting "+" gives
    # the number of extra agreeing sources.
    source_count = candidate.source.count("+") + 1
    agreement_component = min(source_count / 2.0, 1.0) * 0.25

    # 0 fade frames -> full 0.10. 5+ fade frames -> 0. Linear in between.
    fade_component = max(0.0, 1.0 - fade_frames / 5.0) * 0.10

    return round(
        match_component + ocr_component + agreement_component + fade_component, 3
    )


def resolve(hits: list, candidate, target: str) -> Result:
    """
    Turn scored frames into a Result.

    Three outcomes, all handled explicitly:

      1. A stable match       -> the answer, with confidence and fade info
      2. Frames scored well but never stabilised -> report the best frame as an
         ALTERNATE with found=False. This is the "uncertain" path: we return
         options with scores instead of guessing, because a plausible wrong
         answer is worse than an honest "here are three maybes".
      3. Nothing scored at all -> found=False, no alternates

    Never fabricates an answer to avoid returning empty-handed.
    """
    result = Result(found=False)

    if not hits:
        result.notes.append("No frames decoded in the candidate window.")
        return result

    best_hit, fade = find_first_stable(hits)

    if best_hit is None:
        # Outcome 2 or 3. Report the strongest frames we saw so a human can judge.
        ranked = sorted(hits, key=lambda h: h.match_score, reverse=True)[:3]
        top = ranked[0]

        result.notes.append(
            f"No frame reached the stability threshold "
            f"(score >= {config.FUZZY_THRESHOLD_FINE} for "
            f"{config.STABILITY_FRAMES} consecutive frames). "
            f"Best observed score was {top.match_score:.1f}."
        )

        result.alternates = [
            {
                "frame": h.frame_index,
                "timestamp": format_timestamp(h.pts_seconds),
                "score": round(h.match_score, 1),
                "text": h.text,
            }
            for h in ranked if h.match_score >= config.FADE_FLOOR_SCORE
        ]

        return result

    # Outcome 1: we have an answer.
    result.found = True
    result.frame_index = best_hit.frame_index
    result.pts_seconds = best_hit.pts_seconds
    result.timestamp = format_timestamp(best_hit.pts_seconds)
    result.text = best_hit.text
    result.confidence = compute_confidence(best_hit, candidate, fade)
    result.fade_frames = fade

    # Modality tells the user WHICH channel confirmed the answer. An audio-only
    # hit must never be presented as a confirmed on-screen detection -- they are
    # different claims about the world.
    if "visual" in candidate.source:
        result.modality = "visual"
    elif "subtitle" in candidate.source:
        result.modality = "subtitle_confirmed_visually"
    else:
        result.modality = "audio_located_visually_confirmed"

    if fade > 0:
        result.notes.append(
            f"Text faded in over {fade} frame(s) before becoming stable; "
            f"the reported frame is the first stably readable one."
        )

    if result.confidence < 0.6:
        result.notes.append(
            "Confidence is low -- review the saved frame before relying on it."
        )

    return result


def resolve_audio_only(candidate, target: str) -> Result:
    """
    Fallback when the target is spoken but never rendered on screen.

    The audio locator found the phrase, but the fine search found no matching
    text in any frame. That is a real and valid outcome -- the line is dialogue,
    not a subtitle.

    We report the audio timestamp and mark the modality clearly. The caller
    attaches the frame decoded at that timestamp purely for reference -- it
    shows what was on screen when the line was spoken, not a claim that the
    text appears in it. `found` stays False because the text itself was never
    confirmed on screen; inventing a "confirmed" result would be a fabricated
    answer dressed up as a real one.
    """
    result = Result(found=False)

    result.timestamp = format_timestamp(candidate.start_s)
    result.pts_seconds = candidate.start_s
    result.text = target
    result.modality = "audio_only"
    result.confidence = round(candidate.score / 100.0 * 0.5, 3)
    result.notes.append(
        "Phrase found in the audio transcript but NOT as on-screen text. "
        "The attached frame is what was on screen at that timestamp -- it "
        "does not show the target text, since the line was spoken, not written."
    )

    return result