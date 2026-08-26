#normalize.py
"""
matching/normalize.py
=====================
Turns messy real-world text into a canonical form for comparison.

Both sides of EVERY comparison pass through here -- the target phrase the user
typed, the subtitle cue, the Whisper transcript, and the OCR output. If only one
side were normalised the comparison would be meaningless.

Pure functions: no I/O, no models, no config beyond what is passed in. That makes
this the easiest file in the project to unit test, and the first one to write.
"""

import re
import unicodedata

# Characters that are visually identical or near-identical but different code
# points. Video subtitles are full of typographic quotes and dashes; a user
# typing the target on a keyboard will use ASCII. Without this, "don't" and
# "don’t" score as a mismatch on the apostrophe.
UNICODE_FOLD = {
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2013": "-", "\u2014": "-",   # en dash, em dash
    "\u2026": "...",                # ellipsis character
    "\u00a0": " ",                  # non-breaking space
}

# Classic OCR confusions. Applied only by fold_ocr_confusions(), which is used
# as a SECOND-CHANCE comparison -- never as the primary one.
#
# Why second-chance: folding l->1 and o->0 destroys real information. "1000" and
# "looo" become identical, which would create false positives on numeric text.
# So we score normally first, and only retry with folding if the normal score
# lands just under threshold.
OCR_CONFUSIONS = [
    ("rn", "m"),   # 'rn' rendered small looks exactly like 'm'
    ("l", "1"),
    ("i", "1"),
    ("o", "0"),
    ("s", "5"),
    ("b", "6"),
    ("g", "9"),
]


def normalize(text: str) -> str:
    """
    Canonical form: lowercase, no punctuation, single-spaced, no accents.

    Order matters here:
      1. NFKC first, so composed and decomposed accents collapse to one form
      2. fold the typographic characters
      3. lowercase
      4. strip anything that is not a letter, digit or space
      5. collapse runs of whitespace

    Doing step 4 before step 2 would delete the curly quotes rather than
    converting them, which is fine for this use but loses the ability to reason
    about what was removed.
    """
    if not text:
        return ""

    # NFKC also expands ligatures (fi -> fi) which OCR models sometimes emit.
    text = unicodedata.normalize("NFKC", text)

    for src, dst in UNICODE_FOLD.items():
        text = text.replace(src, dst)

    text = text.lower()

    # Strip accents: decompose, then drop the combining marks. "café" -> "cafe".
    # Subtitle renderers and OCR disagree about accents more often than you would
    # expect, and the underlying word is the same.
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Keep letters, digits and spaces. Everything else becomes a space rather
    # than being deleted, so "well-known" -> "well known" not "wellknown".
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace runs and trim. OCR frequently returns multiple spaces
    # where the renderer used wide letter spacing.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fold_ocr_confusions(text: str) -> str:
    """
    Aggressive second-chance normalisation for suspected OCR damage.

    Maps every confusable character to one representative. Note this is lossy and
    NOT reversible -- it exists purely so two differently-corrupted strings can be
    compared. Only called by similarity.py when the primary score is borderline.
    """
    text = normalize(text)

    # 'rn' -> 'm' must run before the single-character rules, otherwise the 'r'
    # and 'n' get folded independently and the pair is never recognised.
    for src, dst in OCR_CONFUSIONS:
        text = text.replace(src, dst)

    return text


def word_list(text: str) -> list:
    """
    Normalised text split into words.

    Used by similarity.find_best_span() to slide a window across a Whisper
    transcript. Splitting after normalising means punctuation attached to words
    ("stagnation." -> "stagnation") does not create phantom tokens.
    """
    normalized = normalize(text)
    return normalized.split() if normalized else []