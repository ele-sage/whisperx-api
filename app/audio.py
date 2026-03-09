"""Audio processing utilities for the standalone app."""

import logging
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Union

import numpy as np
from whisperx import load_audio
from whisperx.audio import SAMPLE_RATE

from app.config import ALLOWED_EXTENSIONS, VIDEO_EXTENSIONS

logger = logging.getLogger("app")


def validate_extension(filename: str) -> str:
    """Validate and return the file extension, raising ValueError if unsupported."""
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    return ext


def convert_video_to_audio(file: str) -> str:
    """Convert a video file to a 16 kHz mono WAV."""
    temp_filename = NamedTemporaryFile(delete=False, suffix=".wav").name
    subprocess.call(
        [
            "ffmpeg", "-y", "-i", file,
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            temp_filename,
        ]
    )
    return temp_filename


def process_audio_file(audio_file: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Load audio, converting from video if necessary."""
    import os

    ext = os.path.splitext(audio_file)[1].lower()
    converted_file: str | None = None
    if ext in VIDEO_EXTENSIONS:
        converted_file = convert_video_to_audio(audio_file)
        audio_file = converted_file
    audio = load_audio(audio_file)
    if converted_file:
        safe_remove_file(converted_file)
    return audio  # type: ignore[no-any-return]


def get_audio_duration_from_file(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        logger.error("Could not parse duration from ffprobe: %s", result.stdout)
        return float(len(load_audio(file_path))) / SAMPLE_RATE


def is_stereo_audio(file_path: str) -> bool:
    """Check if the first audio stream in a file is stereo (2 channels)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    try:
        return int(result.stdout.strip()) == 2
    except ValueError:
        return False


def split_stereo_to_mono(audio_file_path: Union[str, Path]) -> tuple[str, str]:
    """Split a stereo file into two mono WAV files (left, right)."""
    input_path = Path(audio_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    left_file = NamedTemporaryFile(delete=False, suffix=".wav").name
    right_file = NamedTemporaryFile(delete=False, suffix=".wav").name

    command = [
        "ffmpeg", "-i", str(input_path),
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[left][right]",
        "-map", "[left]", "-ac", "1", "-ar", "16000", "-y", left_file,
        "-map", "[right]", "-ac", "1", "-ar", "16000", "-y", right_file,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return left_file, right_file


def safe_remove_file(file_path: str) -> None:
    """Best-effort removal of a temporary file."""
    import os

    if not file_path:
        return
    try:
        os.remove(file_path)
    except OSError:
        pass
