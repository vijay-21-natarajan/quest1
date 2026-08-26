#test_normalize.py
"""
tests/test_normalize.py
=======================
Normalisation is the foundation everything else stands on. If it is wrong, every
comparison in the project is wrong in a way that is very hard to trace back.

Pure functions, so these run in milliseconds with no video, no models, no network.
Write these first -- they are the fastest feedback loop you will have.
"""

from matching.normalize import normalize, fold_ocr_confusions, word_list


def test_lowercases_and_strips_punctuation():
    assert normalize("My Mind Rebels At Stagnation!!!") == "my mind rebels at stagnation"


def test_collapses_whitespace():
    # OCR returns multiple spaces where the renderer used wide letter spacing.
    assert normalize("my   mind\n\trebels") == "my mind rebels"


def test_folds_typographic_quotes():
    # A subtitle renderer emits a curly apostrophe; a user types an ASCII one.
    # Without folding, these two strings would fail to match on that character.
    assert normalize("don\u2019t") == normalize("don't")


def test_strips_accents():
    assert normalize("caf\u00e9") == "cafe"


def test_hyphen_becomes_space_not_deletion():
    # "well-known" must become two words, not one. Deleting the hyphen would
    # create a token that appears in neither the target nor the transcript.
    assert normalize("well-known") == "well known"


def test_empty_and_none_safe():
    # Every caller passes raw OCR output straight in, which is frequently empty.
    assert normalize("") == ""
    assert normalize(None) == ""


def test_idempotent():
    # Normalising twice must equal normalising once, or scores would depend on
    # how many times a string happened to pass through the function.
    once = normalize("My Mind, Rebels!")
    assert normalize(once) == once


def test_ocr_confusion_folding():
    # 'rn' rendered small is visually identical to 'm'. After folding, the
    # corrupted and clean forms must collapse to the same string.
    assert fold_ocr_confusions("modern") == fold_ocr_confusions("rnodern")


def test_word_list_splits_after_normalising():
    # Punctuation attached to a word must not create a phantom token.
    assert word_list("My mind, rebels.") == ["my", "mind", "rebels"]