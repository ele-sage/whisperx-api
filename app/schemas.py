"""Pydantic schemas for the standalone speech-to-text app."""

from enum import Enum
from typing import Any

from fastapi import Query
from pydantic import BaseModel, Field, field_validator
from whisperx import utils  # pyright: ignore[reportMissingTypeStubs]


# ── Enums ────────────────────────────────────────────────────────────

class ComputeType(str, Enum):
    float16 = "float16"
    float32 = "float32"
    int8 = "int8"


class WhisperModel(str, Enum):
    tiny = "tiny"
    tiny_en = "tiny.en"
    base = "base"
    base_en = "base.en"
    small = "small"
    small_en = "small.en"
    medium = "medium"
    medium_en = "medium.en"
    large = "large"
    large_v1 = "large-v1"
    large_v2 = "large-v2"
    large_v3 = "large-v3"
    large_v3_turbo = "large-v3-turbo"
    distil_large_v2 = "distil-large-v2"
    distil_medium_en = "distil-medium.en"
    distil_small_en = "distil-small.en"
    distil_large_v3 = "distil-large-v3"
    faster_crisper_whisper = "nyrahealth/faster_CrisperWhisper"


class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


class TaskEnum(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class InterpolateMethod(str, Enum):
    nearest = "nearest"
    linear = "linear"
    ignore = "ignore"


# ── Query-parameter models ───────────────────────────────────────────

class ASROptions(BaseModel):
    beam_size: int = Field(Query(5, description="Number of beams in beam search"))
    best_of: int = Field(Query(5, description="Number of candidates to keep"))
    patience: float = Field(Query(1.0, description="Beam decoding patience"))
    length_penalty: float = Field(Query(1.0, description="Token length penalty"))
    temperatures: float = Field(Query(0.0, description="Sampling temperature"))
    compression_ratio_threshold: float = Field(Query(2.4, description="Gzip compression ratio threshold"))
    log_prob_threshold: float = Field(Query(-1.0, description="Average log probability threshold"))
    no_speech_threshold: float = Field(Query(0.6, description="No-speech probability threshold"))
    initial_prompt: str | None = Field(Query(None, description="Initial prompt for first window"))
    suppress_tokens: list[int] = Field(Query([-1], description="Token IDs to suppress"))
    suppress_numerals: bool | None = Field(Query(False, description="Whether to suppress numerals"))
    hotwords: str | None = Field(Query(None, description="Hotwords prompt"))

    @field_validator("suppress_tokens", mode="before")
    @classmethod
    def parse_suppress_tokens(cls, value: str | list[int]) -> list[int]:
        if isinstance(value, str):
            return [int(x) for x in value.split(",")]
        return value


class VADOptions(BaseModel):
    vad_onset: float = Field(Query(0.500, description="VAD onset threshold"))
    vad_offset: float = Field(Query(0.363, description="VAD offset threshold"))


class WhisperModelParams(BaseModel):
    language: str | None = Field(
        Query(default=None, description="Language to transcribe (auto-detect if omitted)", enum=list(utils.LANGUAGES.keys())),
    )
    task: TaskEnum = Field(Query(default="transcribe", description="Task: transcribe or translate"))
    model: WhisperModel = Field(Query(default="large-v3-turbo", description="Whisper model name"))
    device: Device = Field(Query(default="cuda", description="Device for inference"))
    device_index: int = Field(Query(0, description="Device index for FasterWhisper"))
    threads: int = Field(Query(0, description="CPU threads (0 = auto)"))
    batch_size: int = Field(Query(8, description="Batch size for inference"))
    chunk_size: int = Field(Query(12, description="VAD chunk size"))
    compute_type: ComputeType = Field(Query("float16", description="Compute type"))


class AlignmentParams(BaseModel):
    align_model: str | None = Field(Query(None, description="Phoneme-level alignment model name"))
    interpolate_method: InterpolateMethod = Field(Query("nearest", description="Interpolation method"))
    return_char_alignments: bool = Field(Query(False, description="Return char-level alignments"))


class DiarizationParams(BaseModel):
    min_speakers: int | None = Field(Query(None, description="Minimum speakers"))
    max_speakers: int | None = Field(Query(None, description="Maximum speakers"))


# ── Result models ────────────────────────────────────────────────────

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    segments: list[TranscriptionSegment]
    language: str


class Word(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


class LabeledWord(Word):
    speaker: str | None = None


class AlignmentSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word]


class LabeledSegment(AlignmentSegment):
    speaker: str | None = None
    words: list[LabeledWord]


class AlignedTranscription(BaseModel):
    segments: list[LabeledSegment]
    word_segments: list[LabeledWord]
