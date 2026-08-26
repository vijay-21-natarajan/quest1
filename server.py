#server.py
"""
server.py
=========
HTTP front door for the pipeline. Holds no pipeline logic of its own -- it
only translates an HTTP request into a call to main.run() and turns the
result.json that run() already writes into a JSON response.

Runs the pipeline in FastAPI's threadpool (plain `def`, not `async def`)
because main.run() is blocking, CPU/IO-heavy work (download, ASR, OCR).

Usage:
    uvicorn server:app --reload
    -> open http://127.0.0.1:8000/
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import main as pipeline

log = logging.getLogger("dialogue-frame-finder.server")
pipeline.setup_logging(verbose=False)

config.ensure_dirs()

app = FastAPI(title="Dialogue Frame Finder")

STATIC_DIR = Path(__file__).parent / "static"

# Serve the saved frame/annotated PNGs so the frontend <img> tag can load them.
app.mount("/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")


class ProcessRequest(BaseModel):
    url: str
    dialogue: str


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/process")
def process(payload: ProcessRequest):
    url = payload.url.strip()
    target = payload.dialogue.strip()

    if not url or not target:
        raise HTTPException(status_code=400, detail="url and dialogue are required")

    try:
        pipeline.run(url, target, use_audio=True)
    except Exception as e:
        log.exception("Pipeline failed for url=%r target=%r", url, target)
        raise HTTPException(status_code=500, detail=str(e))

    result_path = config.RESULTS_DIR / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="Pipeline produced no result")

    data = json.loads(result_path.read_text())

    found = bool(data.get("found"))
    modality = data.get("modality") or ""

    # "found" is False in two very different situations, and only one of them
    # is an error:
    #   - modality == "audio_only": the phrase WAS located (in the audio
    #     transcript) and main.run() already resolved a timestamp, text and an
    #     illustrative frame for it. It just was not confirmed as on-screen
    #     text. That is a real, useful result -- return it as one.
    #   - anything else: no evidence source found the phrase at all, or no
    #     candidate frame ever stabilised. There is nothing to show.
    if not found and modality != "audio_only":
        detail = (data.get("notes") or ["Target dialogue not found in video"])[0]
        raise HTTPException(status_code=404, detail=detail)

    # Prefer the annotated frame (shows the OCR box) when one was produced.
    image_source = data.get("annotated_path") or data.get("image_path")
    image_url = f"/results/{Path(image_source).name}" if image_source else None
    frame_index = data.get("frame_index", -1)

    return {
        "found": found,
        "timestamp": data.get("timestamp"),
        "frame_number": frame_index if frame_index is not None and frame_index >= 0 else None,
        "text": data.get("text"),
        "confidence": data.get("confidence"),
        "modality": modality,
        "image": image_url,
        "note": (data.get("notes") or [None])[0],
    }
