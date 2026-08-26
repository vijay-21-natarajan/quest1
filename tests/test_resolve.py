#test_resolve.py
"""
tests/test_resolve.py
=====================
Tests the hardest logic in the project with NO video at all.

resolve.py consumes a list of FrameHit objects. We can construct those by hand,
which means the fade-in rule, the stability requirement and the confidence blend
are all testable in milliseconds.

This is the payoff for keeping scoring (fine.py) separate from deciding
(resolve.py). If they were one function, none of this would be testable without
a real video and a loaded OCR model.
"""

import config
from schema import Candidate, FrameHit
from search.resolve import find_first_stable, resolve, compute_confidence


def hits_from_scores(scores, start_frame=1000, fps=25.0):
    """Build a chronological FrameHit list from a list of match scores."""
    return [
        FrameHit(
            frame_index=start_frame + i,
            pts_seconds=(start_frame + i) / fps,
            text="my mind rebels at stagnation" if s > 50 else "",
            ocr_confidence=0.9 if s > 50 else 0.0,
            match_score=float(s),
        )
        for i, s in enumerate(scores)
    ]


def a_candidate(source="visual", score=90.0):
    return Candidate(start_s=40.0, end_s=44.0, source=source, score=score)


def test_picks_first_frame_of_a_sustained_run():
    # Clean appearance, no fade. The answer is unambiguous.
    hits = hits_from_scores([0, 0, 95, 96, 96, 95])
    best, fade = find_first_stable(hits)
    assert best.frame_index == 1002
    assert fade == 0


def test_ignores_single_frame_spike():
    # THE core test. A lone frame at half opacity OCRs well by luck, then drops
    # back. Without the stability rule this spike becomes the answer and the
    # reported frame is wrong by however long the real text takes to appear.
    hits = hits_from_scores([0, 95, 10, 5, 96, 97, 96])
    best, _ = find_first_stable(hits)
    assert best.frame_index == 1004, "spike at index 1001 must be rejected"


def test_fade_in_reports_ramp_length():
    # The realistic case: score climbs through the fade before stabilising.
    # We must report the first STABLE frame and disclose the ramp length.
    #
    # 45 and 55 are both above FADE_FLOOR_SCORE (40) -- some text is visible but
    # not yet readable. 5 is below it: nothing on screen. So the ramp is 2 frames.
    hits = hits_from_scores([5, 45, 55, 91, 96, 96])
    best, fade = find_first_stable(hits)
    assert best.frame_index == 1003
    assert fade == 2, "frames scoring 45 and 55 are the fade ramp"


def test_fade_floor_excludes_noise_frames():
    # A frame scoring below FADE_FLOOR_SCORE is "no text at all", not a faint
    # first frame. It must NOT be counted into the ramp -- otherwise random OCR
    # noise before the subtitle inflates the reported fade length.
    hits = hits_from_scores([5, 31, 55, 91, 96, 96])
    _, fade = find_first_stable(hits)
    assert fade == 1, "score 31 is below the floor and is not part of the ramp"


def test_mid_run_dropout_does_not_restart_the_answer():
    # Lighting change or a cut-away briefly damages OCR mid-subtitle. The first
    # stable run is still the correct answer.
    hits = hits_from_scores([0, 95, 96, 95, 20, 95, 96])
    best, _ = find_first_stable(hits)
    assert best.frame_index == 1001


def test_no_stable_run_returns_none():
    hits = hits_from_scores([10, 20, 15, 30, 25])
    best, fade = find_first_stable(hits)
    assert best is None and fade == 0


def test_match_at_very_end_of_window_still_accepted():
    # Text appears in the last frames of our window and continues past it. The
    # look-ahead clamps to the list end rather than rejecting for lack of room.
    hits = hits_from_scores([0, 0, 0, 95, 96])
    best, _ = find_first_stable(hits)
    assert best is not None and best.frame_index == 1003


def test_resolve_returns_found_result_with_timestamp():
    hits = hits_from_scores([0, 0, 95, 96, 96])
    result = resolve(hits, a_candidate(), "my mind rebels at stagnation")
    assert result.found is True
    assert result.frame_index == 1002
    assert result.timestamp == "00:00:40.080"
    assert result.confidence > 0.0
    assert result.modality == "visual"


def test_resolve_reports_alternates_when_nothing_stabilises():
    # The uncertainty path: return ranked options with scores rather than
    # guessing. A plausible wrong answer is worse than an honest "here are maybes".
    hits = hits_from_scores([45, 60, 55, 48])
    result = resolve(hits, a_candidate(), "my mind rebels at stagnation")
    assert result.found is False
    assert result.alternates, "expected alternates to be reported"
    assert result.notes


def test_resolve_handles_empty_hit_list():
    result = resolve([], a_candidate(), "target")
    assert result.found is False
    assert result.notes


def test_cross_source_agreement_raises_confidence():
    # Two independent channels agreeing is the strongest signal available,
    # because their failure modes are unrelated.
    hit = hits_from_scores([96])[0]
    single = compute_confidence(hit, a_candidate(source="visual"), 0)
    both = compute_confidence(hit, a_candidate(source="audio+visual"), 0)
    assert both > single


def test_long_fade_lowers_confidence():
    # A long ramp means the first-appearance frame was more of a judgement call.
    hit = hits_from_scores([96])[0]
    crisp = compute_confidence(hit, a_candidate(), fade_frames=0)
    blurry = compute_confidence(hit, a_candidate(), fade_frames=6)
    assert crisp > blurry