#test_similarity.py
"""
tests/test_similarity.py
========================
Guards BOTH failure directions:

  false negatives -- realistic OCR corruption must still score above threshold,
                     or we miss the answer entirely
  false positives -- unrelated text must score below, or we return a confident
                     wrong frame

The second is the one people forget to test, and it is the more embarrassing
failure in a live demo.
"""

import config
from matching.similarity import score, find_best_span, best_score

TARGET = "my mind rebels at stagnation"


def test_exact_match_scores_100():
    assert score(TARGET, TARGET) == 100


def test_case_and_punctuation_insensitive():
    assert score(TARGET, "My mind rebels at stagnation!") >= 99


def test_survives_realistic_ocr_corruption():
    # Character-level damage of the kind PaddleOCR actually produces on
    # low-contrast or motion-blurred subtitles.
    assert score(TARGET, "my mind rebeis at stagnatlon") >= config.FUZZY_THRESHOLD_FINE
    assert score(TARGET, "rny mind rebels at stagnation") >= config.FUZZY_THRESHOLD_FINE


def test_matches_when_embedded_in_longer_text():
    # This is why partial_ratio is used instead of ratio. OCR returns the whole
    # visible subtitle block, which may contain more than the target line.
    detected = "HOLMES: My mind rebels at stagnation. Give me problems!"
    assert score(TARGET, detected) >= config.FUZZY_THRESHOLD_FINE


def test_unrelated_text_scores_low():
    # The false-positive guard. Credits, signage and captions appear constantly.
    assert score(TARGET, "directed by guy ritchie") < config.FUZZY_THRESHOLD_COARSE
    assert score(TARGET, "chapter seven the adventure begins") < config.FUZZY_THRESHOLD_COARSE


def test_empty_inputs_score_zero():
    assert score(TARGET, "") == 0.0
    assert score("", "anything") == 0.0


def test_find_best_span_returns_correct_time_range():
    # Simulated Whisper output: (word, start, end)
    words = [
        ("the", 0.0, 0.2), ("game", 0.2, 0.5), ("is", 0.5, 0.7),
        ("afoot", 0.7, 1.1),
        ("my", 5.0, 5.2), ("mind", 5.2, 5.5), ("rebels", 5.5, 5.9),
        ("at", 5.9, 6.0), ("stagnation", 6.0, 6.7),
        ("give", 9.0, 9.3), ("me", 9.3, 9.5), ("problems", 9.5, 10.0),
    ]

    spans = find_best_span(words, TARGET, threshold=60)

    assert spans, "expected at least one matching span"
    top_score, start, end = spans[0]

    assert top_score >= 80
    # The span must bracket the phrase's real position at 5.0-6.7s.
    assert 4.5 <= start <= 5.5
    assert 6.0 <= end <= 7.5


def test_find_best_span_empty_inputs():
    assert find_best_span([], TARGET) == []
    assert find_best_span([("a", 0.0, 0.1)], "") == []


def test_best_score_picks_strongest_candidate():
    s, text = best_score(TARGET, ["directed by", "my mind rebels at stagnation", "1899"])
    assert s >= 99
    assert "rebels" in text


def test_best_score_handles_empty_list():
    assert best_score(TARGET, []) == (0.0, "")