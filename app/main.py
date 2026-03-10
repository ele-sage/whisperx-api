"""Standalone speech-to-text FastAPI application.

A self-contained app that exposes a single ``POST /speech-to-text``
endpoint. Requests are processed synchronously (the transcription is
returned in the HTTP response) and GPU access is serialized with a
single lock so the server cannot be overwhelmed by concurrent callers.

Start with::

    uvicorn app.main:app --host 0.0.0.0 --port 8001
"""

import gc
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile


import torch
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.audio import get_audio_duration_from_file, safe_remove_file, validate_extension
from app.config import ALLOWED_EXTENSIONS
from app.processing import run_speech_to_text
from app.schemas import (
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    VADOptions,
    WhisperModelParams,
)

# ── Logging ──────────────────────────────────────────────────────────

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Standalone speech-to-text app started")
    logger.info("Allowed extensions: %s", sorted(ALLOWED_EXTENSIONS))
    yield
    logger.info("Shutting down standalone app")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Standalone WhisperX Speech-to-Text",
    description=(
        "A minimal, self-contained speech-to-text API. "
        "Returns the transcription directly in the response. "
        "Concurrent requests are queued behind a single GPU lock."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Endpoints ────────────────────────────────────────────────────────

# NOTE: This is a plain `def`, NOT `async def`.
# FastAPI automatically runs sync endpoints in a threadpool, so
# the event loop stays free to serve /health and accept new connections
# while a long transcription is in progress.
@app.post("/speech-to-text", tags=["Speech-to-Text"])
def speech_to_text(
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options: ASROptions = Depends(),
    vad_options: VADOptions = Depends(),
    file: UploadFile = File(...),
    split_audio: bool = Query(
        default=False,
        description="Split stereo audio into separate channels for individual processing",
    ),
) -> JSONResponse:
    """Transcribe an uploaded audio file and return the result synchronously.

    The response contains the full aligned (and optionally diarized)
    transcription. If another request is already being processed, this
    one will block until the GPU is free.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is missing")

    # Validate extension
    try:
        validate_extension(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Save upload to temp file
    _, ext = os.path.splitext(file.filename)
    temp = NamedTemporaryFile(suffix=ext, delete=False)
    temp.write(file.file.read())
    temp.close()
    temp_path = temp.name

    logger.info("Received file: %s (%s)", file.filename, temp_path)

    try:
        duration = get_audio_duration_from_file(temp_path)
        start_time = time.time()
        
        result = run_speech_to_text(
            temp_file=temp_path,
            model_params=model_params,
            align_params=align_params,
            diarize_params=diarize_params,
            asr_options=asr_options,
            vad_options=vad_options,
            split_audio=split_audio,
        )
        
        processing_time = time.time() - start_time
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "duration": duration,
                "processing_time": round(processing_time, 3),
                **result,
            },
        )
    except Exception as exc:
        logger.exception("Processing failed for %s", file.filename)
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(exc),
                "segments": [],
                "word_segments": [],
                "duration": 0.0,
                "processing_time": 0.0,
                "is_stereo": False,
            },
        )
    finally:
        safe_remove_file(temp_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Simple liveness check."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "timestamp": time.time()},
    )
