#ocr_engine.py
"""
engines/ocr_engine.py
=====================
PaddleOCR wrapper. Frame in, text out.

Two design decisions worth defending:

1. SINGLETON. Instantiating PaddleOCR loads detection and recognition models and
   costs 2-3 seconds. The coarse sweep calls this thousands of times. Loading per
   call would dominate the entire runtime.

2. DETECTION-ONLY MODE. PaddleOCR is really two models: a detector that finds
   text regions and a recogniser that reads them. The detector alone is roughly
   5x cheaper. The coarse sweep only needs "is there text here at all?", so it
   uses detection only and skips recognition entirely on empty frames.

This module knows NOTHING about the target phrase. It reports what it sees;
matching happens elsewhere. That separation is what lets the same engine serve
both the coarse and fine passes.
"""

import logging

import numpy as np

import config

log = logging.getLogger(__name__)

_ENGINE = None       # module-level singleton
_API_STYLE = None    # "v3" or "v2" -- detected once, see _detect_api_style


def get_engine():
    """
    Return the shared PaddleOCR instance, creating it on first use.

    Lazy rather than import-time: a run that hits a subtitle track never needs
    OCR at all, and should not pay the model-loading cost.
    """
    global _ENGINE, _API_STYLE

    if _ENGINE is not None:
        return _ENGINE

    # pyrefly: ignore [missing-import]
    from paddleocr import PaddleOCR

    log.info("Loading PaddleOCR (lang=%s)...", config.OCR_LANG)

    # PaddleOCR 3.x renamed several constructor arguments. Rather than pinning to
    # one version and breaking on the evaluator's machine, try the new signature
    # and fall back to the old one. Version drift in this library is frequent
    # enough that this defensiveness is worth the six lines.
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    try:
        _ENGINE = PaddleOCR(lang=config.OCR_LANG, use_textline_orientation=True, show_log=False)
        _API_STYLE = "v3"
    except TypeError:
        _ENGINE = PaddleOCR(lang=config.OCR_LANG, use_angle_cls=True, show_log=False)
        _API_STYLE = "v2"

    log.info("PaddleOCR loaded (API style: %s)", _API_STYLE)
    return _ENGINE


def _run(image: np.ndarray) -> list:
    """
    Call PaddleOCR and normalise the result to [(text, confidence, bbox), ...].

    The two API generations return completely different shapes:

      v2: [[[bbox, (text, score)], ...]]           nested lists
      v3: [{"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}]  dicts

    Normalising here means every caller sees one stable format regardless of
    which PaddleOCR version is installed.
    """
    engine = get_engine()

    # Call OCR - both APIs use the same method
    raw = engine.ocr(image, cls=True)
    out = []

    # Handle empty results
    if not raw:
        return out

    for page in raw:
        if not page:
            continue

        # Check if this is the dictionary format (v3) or list format (v2)
        if isinstance(page, dict):
            # v3 dictionary format
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("dt_polys", [])
            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 0.0
                bbox = polys[i].tolist() if i < len(polys) and hasattr(polys[i], "tolist") else None
                out.append((text, conf, bbox))
        else:
            # v2 list format
            for entry in page:
                bbox, (text, conf) = entry[0], entry[1]
                out.append((text, float(conf), bbox))

    return out


def has_text(image: np.ndarray) -> bool:
    """
    Cheap presence check for the coarse sweep.

    Ideally this calls the detector alone. PaddleOCR does not expose a stable
    detection-only entry point across versions, so we run the full pipeline and
    check whether anything cleared the confidence floor. Still worth having as a
    named function: it documents intent, and it is the one place to optimise if
    the coarse pass turns out to be the bottleneck.
    """
    results = _run(image)
    return any(conf >= config.OCR_MIN_CONFIDENCE for _, conf, _ in results)


def read(image: np.ndarray) -> list:
    """
    Full recognition. Returns [(text, confidence, bbox), ...] above the floor.

    Filtering by confidence here rather than in the caller means low-confidence
    garbage never reaches the fuzzy matcher, where it could accidentally score
    well against a short target phrase.
    """
    results = _run(image)
    return [(t, c, b) for t, c, b in results if c >= config.OCR_MIN_CONFIDENCE]


def read_joined(image: np.ndarray) -> tuple:
    """
    Read a frame and return (joined_text, mean_confidence, best_bbox).

    Subtitles routinely wrap onto two lines and PaddleOCR returns each as a
    separate detection. Joining them with a space before matching is essential:
    a two-line subtitle compared line-by-line against a one-line target scores
    poorly on both halves, and the real match is missed.
    """
    results = read(image)

    if not results:
        return "", 0.0, None

    text = " ".join(r[0] for r in results)
    mean_conf = sum(r[1] for r in results) / len(results)

    # Bounding box of the highest-confidence detection, used to draw the
    # annotation on the output image.
    best_bbox = max(results, key=lambda r: r[1])[2]

    return text, mean_conf, best_bbox