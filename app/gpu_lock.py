"""Single global GPU lock for serializing all ML work.

Only one request can use the GPU at a time. Concurrent requests queue
behind the lock and are served sequentially, preventing OOM errors.
"""

import threading
from contextlib import contextmanager
from typing import Generator

import logging

logger = logging.getLogger("app")

_transcription_lock = threading.Lock()
_alignment_lock = threading.Lock()
_diarization_lock = threading.Lock()


@contextmanager
def transcription_lock() -> Generator[None, None, None]:
    """Acquire the GPU lock for transcription."""
    logger.debug("Waiting to acquire transcription lock...")
    _transcription_lock.acquire()
    logger.debug("Transcription lock acquired.")
    try:
        yield
    finally:
        _transcription_lock.release()
        logger.debug("Transcription lock released.")


@contextmanager
def alignment_lock() -> Generator[None, None, None]:
    """Acquire the GPU lock for alignment."""
    logger.debug("Waiting to acquire alignment lock...")
    _alignment_lock.acquire()
    logger.debug("Alignment lock acquired.")
    try:
        yield
    finally:
        _alignment_lock.release()
        logger.debug("Alignment lock released.")


@contextmanager
def diarization_lock() -> Generator[None, None, None]:
    """Acquire the GPU lock for diarization."""
    logger.debug("Waiting to acquire diarization lock...")
    _diarization_lock.acquire()
    logger.debug("Diarization lock acquired.")
    try:
        yield
    finally:
        _diarization_lock.release()
        logger.debug("Diarization lock released.")
