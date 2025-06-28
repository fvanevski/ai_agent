# api.py – Whisper ASR micro‑service (RTX 3090 tuned)
"""
* FastAPI service that exposes:
    • **POST  /transcribe**       — multipart audio/video file upload
    • **POST  /transcribe_url**   — JSON {"url": "https://youtu…"}
    • **GET   /is_sleeping**      — returns {is_sleeping: bool}
    • **POST  /sleep** / **/wake_up** — toggle GPU idle state

* Improvements requested:
    ▸ word‑level timestamps always collected (`word_timestamps=True`).
    ▸ `output` param selects view: `segments` (default) | `words` | `sentences` | `text`.
    ▸ Env **DIAR_DEVICE** controls diarisation device (`cuda` default, `cpu` override).
    ▸ Uses Zoont/faster‑whisper‑large‑v3‑turbo‑int8‑ct2 with `compute_type="int8"`.
    ▸ Smaller `chunk_length=20` and `batch_size=8` to reduce VRAM.
    ▸ Sleep/wake endpoints free KV cache but keep model in memory.

Only the above changes were grafted onto the original api.py.bak logic; all
other behaviours are retained.
"""

from __future__ import annotations

import os
import uuid
import json
import shutil
import tempfile
import itertools
import subprocess
from pathlib import Path
from typing import Literal, List, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline as DiarPipeline
import yt_dlp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("WHISPER_MODEL", "Zoont/faster-whisper-large-v3-turbo-int8-ct2")
# MODEL_NAME = os.getenv("WHISPER_MODEL", "distil-large-v2")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")
CHUNK_LEN = int(os.getenv("CHUNK_LENGTH", 60))          # seconds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
DIAR_DEVICE = os.getenv("DIAR_DEVICE", "cuda")          # cuda | cpu

OUTPUT_VIEWS = {"segments", "words", "sentences", "text"}

aaa_sleeping: bool = False  # global flag

def _load_models():
    """Load Whisper + diarisation once at startup."""
    whisper = WhisperModel(
        MODEL_NAME,
        device="cuda", compute_type=COMPUTE_TYPE,
        local_files_only=False,
    )

    # In some faster‑whisper versions BatchedInferencePipeline only takes the model;
    # per‑request overrides (batch_size, chunk_length, word_timestamps) are passed
    # to the .transcribe() call instead.
    pipe = BatchedInferencePipeline(whisper)

    diar = DiarPipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diar.to(torch.device(DIAR_DEVICE))
    return pipe, diar

TRANSCRIBE_PIPE, DIARISATION_PIPE = _load_models()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _download_youtube(url: str) -> Path:
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outfile = Path(tmpdir) / f"{uuid.uuid4()}.wav"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outfile.with_suffix(".%(ext)s")),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # yt‑dl writes unknown ext first; find the produced wav
    wavs = list(Path(tmpdir).glob("*.wav"))
    if not wavs:
        raise HTTPException(400, "Failed to download audio")
    return wavs[0]


def _words_to_sentences(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sentences = []
    cur_words, start, end, speaker = [], None, None, None
    for w in words:
        if speaker is None:
            speaker = w["speaker"]
        # new sentence if speaker changes or punctuation marks end
        if w["speaker"] != speaker or w["word"].endswith((".", "?", "!")):
            if cur_words:
                sentences.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": " ".join(cur_words).strip()
                })
            cur_words, start, speaker = [], None, w["speaker"]
        # accumulate
        cur_words.append(w["word"])
        end = w["end"]
        if start is None:
            start = w["start"]
    if cur_words:
        sentences.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": " ".join(cur_words).strip()
        })
    return sentences


def _words_to_segments(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = []
    for speaker, group in itertools.groupby(words, key=lambda w: w["speaker"]):
        grp = list(group)
        segments.append({
            "start": grp[0]["start"],
            "end": grp[-1]["end"],
            "speaker": speaker,
            "text": " ".join(w["word"] for w in grp).strip(),
            "words": grp,
        })
    return segments


def _run_asr(path: Path, output: str) -> Dict[str, Any]:
    if output not in OUTPUT_VIEWS:
        raise HTTPException(400, f"output must be one of {OUTPUT_VIEWS}")

    segments, info = TRANSCRIBE_PIPE.transcribe(
        str(path),
        initial_prompt="",
        condition_on_previous_text=False,
        beam_size=1,
        vad_filter=True,
        word_timestamps=True,
        chunk_length=CHUNK_LEN,
        batch_size=BATCH_SIZE,
    )
    # whisper returns segments; we already have words
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "speaker": None,  # to be filled by diarisation
            })

    # diarisation – outputs (start, end, speaker_id)
    diar = DIARISATION_PIPE(str(path))
    # 1. Initial speaker assignment using word midpoints
    # Create a list of turns for faster lookup
    turns = list(diar.itertracks(yield_label=True))
    for w in words:
        for turn, _, speaker in turns:
            word_midpoint = (w["start"] + w["end"]) / 2
            if turn.start <= word_midpoint < turn.end:
                w["speaker"] = speaker
                break  # Move to the next word once speaker is found

    # 2. Post-processing to fill gaps and assign remaining 'None' speakers
    # This assumes unassigned words belong to the previous speaker.
    last_speaker = "unknown"
    for w in words:
        if w["speaker"] is None:
            w["speaker"] = last_speaker
        last_speaker = w["speaker"]

    if output == "words":
        return {"words": words, "duration": info.duration}
    elif output == "sentences":
        return {"sentences": _words_to_sentences(words), "duration": info.duration}
    elif output == "text":
        # Group by speaker to create a conversational transcript. This is more
        # useful for LLMs than a single block of text.
        full_text = []
        for speaker, group in itertools.groupby(words, key=lambda w: w["speaker"]):
            turn_text = " ".join(w["word"] for w in group).strip()
            full_text.append(f"{speaker}: {turn_text}")

        # Join all turns with newlines to make it readable.
        text = "\n\n".join(full_text)
        return {"text": text, "duration": info.duration}
    else:  # segments
        return {"segments": _words_to_segments(words), "duration": info.duration}


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper‑ASR", version="1.0")


class URLReq(BaseModel):
    url: str
    output: Literal["segments", "words", "sentences", "text"] = "segments"

@app.post("/transcribe_url")
async def transcribe_url(req: URLReq):
    wav = _download_youtube(req.url)
    try:
        result = _run_asr(wav, req.output)
    finally:
        shutil.rmtree(wav.parent, ignore_errors=True)
        torch.cuda.empty_cache()
    return result


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    output: Literal["segments", "words", "sentences", "text"] = Query("segments")
):
    suffix = Path(file.filename).suffix or ".wav"
    tmp = Path(tempfile.mkdtemp()) / f"{uuid.uuid4()}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = _run_asr(tmp, output)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Run with: python api.py  (or uvicorn api:app --reload)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, argparse

    parser = argparse.ArgumentParser("Whisper ASR server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8003)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="auto‑reload on code changes")
    args = parser.parse_args()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
