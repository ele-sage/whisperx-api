"""Minimal configuration for the standalone speech-to-text app."""

import os

import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "int8"

AUDIO_EXTENSIONS: set[str] = {
    ".mp3", ".wav", ".awb", ".aac", ".ogg", ".oga", ".m4a", ".wma", ".amr",
}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".wmv", ".mkv"}
ALLOWED_EXTENSIONS: set[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
