#whisper_engine.py
"""
engines/whisper_engine.py
=========================
faster-whisper wrapper. Audio in, timestamped words out.

WHY faster-whisper AND NOT WhisperX
-----------------------------------
WhisperX adds a wav2vec2 forced-alignment pass that sharpens word timings from
roughly one second of drift to about 100 ms. That sounds essential for
frame-accurate work, and it is not.

The audio stage only produces a CANDIDATE WINDOW. We pad it by two seconds and
then re-derive the exact frame from pixels in the fine search. Paying for a
second model to sharpen a timestamp we are about to discard is wasted work and
a wasted dependency.

faster-whisper's own word_timestamps=True is sufficient and comes free.

Like the OCR engine, this module knows nothing about the target phrase.
"""

import logging

import config

log = logging.getLogger(__name__)

_MODEL = None


def get_model():
    """
    Load the Whisper model once and cache it.

    Lazy for the same reason as the OCR engine: a run that finds its answer in a
    subtitle track should never pay to load hundreds of megabytes of ASR weights.
    """
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    from faster_whisper import WhisperModel

    log.info("Loading Whisper '%s' on %s (%s)...",
             config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE,
             config.WHISPER_COMPUTE_TYPE)

    _MODEL = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    return _MODEL


def transcribe(audio_path: str) -> tuple:
    """
    Transcribe the audio and return (words, language).

    `words` is a FLAT list of (word, start_seconds, end_seconds) covering the
    whole file. Flattening the segment structure is deliberate: it turns the
    transcript into a simple sequence that similarity.find_best_span() can slide
    a window across. Keeping the nested segment shape would mean a phrase
    straddling a segment boundary could never be found.
    """
    model = get_model()

    log.info("Transcribing %s ...", audio_path)

    # word_timestamps=True is the whole point -- without it we get segment-level
    # times that drift by seconds and are useless for locating a specific line.
    #
    # vad_filter skips silent regions. On a feature-length video with music and
    # pauses this is a meaningful speedup and it reduces hallucinated text in
    # silence, which Whisper is prone to.
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    words = []

    # segments is a GENERATOR -- transcription happens lazily as we iterate.
    # Nothing has actually been computed until this loop runs.
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            # .word carries a leading space from the tokeniser; strip it here so
            # downstream joins do not produce doubled spaces.
            words.append((w.word.strip(), w.start, w.end))

    log.info("Transcribed %d words, detected language: %s",
             len(words), info.language)

    return words, info.language