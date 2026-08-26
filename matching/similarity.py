#similarity.py
"""
matching/similarity.py
======================
Scores how well a piece of detected text matches the target phrase.

Two responsibilities:
  1. score()          -- compare two strings, return 0-100
  2. find_best_span() -- slide a window across a word stream and find the best
                         matching stretch, with its time range

Both are pure. The target phrase arrives as an ARGUMENT, never a constant --
nothing in this repo hardcodes what we are searching for.
"""

from rapidfuzz import fuzz

import config
from matching.normalize import normalize, fold_ocr_confusions, word_list


def score(target: str, candidate: str) -> float:
    """
    Similarity of `candidate` to `target`, 0-100.

    Uses partial_ratio, not ratio. This is the important choice:

      ratio()          compares the whole strings. If OCR returns two subtitle
                       lines and the target is one of them, ratio() punishes the
                       extra text and the real match scores low.
      partial_ratio()  finds the best-matching substring of the longer string.
                       "my mind rebels at stagnation" inside "holmes my mind
                       rebels at stagnation watson" still scores ~100.

    Since detected text is almost always longer than the target (extra subtitle
    lines, surrounding transcript words), partial_ratio is the right tool.
    """
    t = normalize(target)
    c = normalize(candidate)

    # Empty input is a definite non-match. Guarding here means every caller can
    # pass raw OCR output without checking first.
    if not t or not c:
        return 0.0

    primary = fuzz.partial_ratio(t, c)

    # Second chance for OCR-damaged text. Only attempted when the primary score
    # is in the "nearly matched" band -- running it always would let unrelated
    # numeric strings score high after o->0 folding.
    if 50 <= primary < 90:
        folded = fuzz.partial_ratio(fold_ocr_confusions(target),
                                    fold_ocr_confusions(candidate))
        # Small penalty on the folded score: it is weaker evidence than a clean
        # match, so it should never outrank one.
        return max(primary, folded - 3)

    return float(primary)


def find_best_span(words: list, target: str, threshold: float = 0.0) -> list:
    """
    Slide a window across a timestamped word stream and return matching spans.

    `words` is a list of (word, start_seconds, end_seconds) tuples -- the flat
    output of engines/whisper_engine.py.

    Why a sliding window instead of joining the whole transcript and searching
    once: we need the TIME of the match, not just whether it exists. Joining
    loses the mapping from character position back to timestamp. Sliding keeps
    every window anchored to real word boundaries with real times attached.

    The window is sized to the target's word count, then widened by 2 to tolerate
    ASR inserting filler words ("uh", "and") inside the phrase.

    Returns a list of (score, start_seconds, end_seconds), best first.
    """
    target_words = word_list(target)
    if not target_words or not words:
        return []

    n = len(target_words)

    results = []

    # Try several window sizes rather than one. A window of exactly n words is
    # the tightest possible fit; n+1 and n+2 tolerate ASR inserting filler words
    # ("uh", "and") inside the phrase.
    #
    # This matters more than it looks. With only a wide window, partial_ratio
    # scores 100 for ANY window that CONTAINS the phrase -- including one that
    # starts several words early. The reported span would then begin seconds
    # before the phrase actually does, and we would fine-search the wrong place.
    for window_size in range(n, min(n + 3, len(words) + 1)):
        if window_size <= 0:
            continue

        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]

            # Reject windows that silently jump across a long silence. Words are
            # only adjacent in THIS LIST, not necessarily in time -- vad_filter
            # drops silent/musical stretches, so the word right before a dropped
            # stretch and the word right after it sit next to each other here
            # despite being a scene apart in the actual video. Scoring that
            # pairing as one span produces a "match" that bridges two unrelated
            # lines (see config.MAX_WORD_GAP_S for the full story).
            if any(window[k][1] - window[k - 1][2] > config.MAX_WORD_GAP_S
                   for k in range(1, len(window))):
                continue

            # Rebuild the window's text from its word tokens. These are already
            # individual words, so a plain join is correct.
            window_text = " ".join(w[0] for w in window)

            s = score(target, window_text)

            if s >= threshold:
                # Time range spans the first word's start to the last word's end.
                results.append((s, window[0][1], window[-1][2]))

    if not results:
        return []

    # Sort by score descending, then by DURATION ascending. The tie-break is the
    # important half: several windows score 100 when they all contain the phrase,
    # and among those the shortest is the one that actually brackets it.
    results.sort(key=lambda r: (-r[0], r[2] - r[1]))

    # Collapse near-duplicate spans. A phrase match produces several overlapping
    # windows with similar scores; reporting all of them as separate candidates
    # would waste fine-search passes on the same moment.
    deduped = []
    for s, start, end in results:
        if any(abs(start - d[1]) < 1.0 for d in deduped):
            continue
        deduped.append((s, start, end))

    return deduped


def best_score(target: str, candidates: list) -> tuple:
    """
    Score `target` against several strings, return (best_score, best_string).

    Convenience used when OCR returns multiple text regions for one frame and we
    want the strongest one. Returns (0.0, "") for an empty list so callers never
    have to special-case it.
    """
    if not candidates:
        return 0.0, ""

    scored = [(score(target, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0]