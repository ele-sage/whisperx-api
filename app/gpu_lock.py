"""Single global GPU lock for serializing all ML work.

Only one request can use the GPU at a time. Concurrent requests queue
behind the lock and are served sequentially, preventing OOM errors.
"""

import threading
from contextlib import contextmanager
from typing import Generator

import logging

logger = logging.getLogger("app")

_gpu_lock = threading.Lock()


@contextmanager
def gpu_lock() -> Generator[None, None, None]:
    """Acquire the global GPU lock, blocking until it is available."""
    logger.debug("Waiting to acquire GPU lock...")
    _gpu_lock.acquire()
    logger.debug("GPU lock acquired.")
    try:
        yield
    finally:
        _gpu_lock.release()
        logger.debug("GPU lock released.")
