"""Synchronous speech-to-text processing pipeline with cached models.

Models are loaded lazily on first use and cached across requests for
performance.  All GPU work runs inside a single global lock so concurrent
requests are queued safely.

Optimizations:
- Audio loading (CPU/IO) runs OUTSIDE the GPU lock.
- Speaker assignment (CPU-only) runs OUTSIDE the GPU lock.
- Transcription model: cached, NOT reloaded on language change (Whisper
  is multilingual — language is a transcribe-time parameter).
- Alignment model: caches up to 2 languages simultaneously (LRU eviction).
- Diarization model: cached, reloaded only on device change.
"""

import gc
import logging
import time
from collections import OrderedDict
from typing import Any, List

import numpy as np
import torch
import whisperx


from whisperx import align, load_align_model, load_audio, load_model
from whisperx.diarize import DiarizationPipeline

from app.audio import (
    is_stereo_audio,
    process_audio_file,
    safe_remove_file,
    split_stereo_to_mono,
)
from app.config import HF_TOKEN
from app.gpu_lock import alignment_lock, diarization_lock, transcription_lock
from app.schemas import (
    AlignedTranscription,
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    LabeledSegment,
    LabeledWord,
    VADOptions,
    WhisperModelParams,
)
from app.transcript import filter_aligned_transcription

logger = logging.getLogger("app")

_MAX_CACHED_ALIGNMENT_LANGUAGES = 2


# ── Cached model holders ─────────────────────────────────────────────


class TranscriptionModel:
    """Lazily loaded, cached WhisperX transcription model.

    The Whisper model is multilingual — language is only a parameter
    passed at transcribe time, NOT at model load time.  So we do NOT
    reload the model when the requested language changes.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self._config: dict[str, Any] | None = None

    def _should_reload(
        self,
        model: str,
        device: str,
        device_index: int,
        compute_type: str,
        task: str,
    ) -> bool:
        if self.model is None or self._config is None:
            return True
        return self._config != {
            "model": model,
            "device": device,
            "device_index": device_index,
            "compute_type": compute_type,
            "task": task,
        }

    def transcribe(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        params: WhisperModelParams,
        asr_options: ASROptions,
        vad_options: VADOptions,
    ) -> dict[str, Any]:
        """Transcribe audio. Must be called while holding the GPU lock."""
        faster_whisper_threads = 4
        if params.threads > 0:
            torch.set_num_threads(params.threads)
            faster_whisper_threads = params.threads

        if self._should_reload(
            params.model.value,
            params.device.value,
            params.device_index,
            params.compute_type.value,
            params.task.value,
        ):
            if self.model is not None:
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(
                "Loading transcription model %s on %s",
                params.model.value,
                params.device.value,
            )
            self.model = load_model(
                params.model.value,
                params.device.value,
                device_index=params.device_index,
                compute_type=params.compute_type.value,
                asr_options=asr_options.model_dump(),
                vad_options=vad_options.model_dump(),
                language=None,
                task=params.task.value,
                threads=faster_whisper_threads,
            )
            self._config = {
                "model": params.model.value,
                "device": params.device.value,
                "device_index": params.device_index,
                "compute_type": params.compute_type.value,
                "task": params.task.value,
            }
        else:
            logger.debug("Reusing cached transcription model")
        logger.debug(
            "Transcribing audio with batch_size=%s, chunk_size=%s, language=%s",
            params.batch_size,
            params.chunk_size,
            params.language,
        )
        result = self.model.transcribe(
            audio=audio,
            batch_size=params.batch_size,
            chunk_size=params.chunk_size,
            language=params.language,
        )
        return result  # type: ignore[no-any-return]


class AlignmentModel:
    """Lazily loaded, multi-language cached WhisperX alignment model.

    Caches up to ``_MAX_CACHED_ALIGNMENT_LANGUAGES`` alignment models
    simultaneously (keyed by language + device).  When the cache is
    full, the least-recently-used model is evicted.
    """

    def __init__(self, max_cached: int = _MAX_CACHED_ALIGNMENT_LANGUAGES) -> None:
        self._max_cached = max_cached
        # key = (language_code, device), value = (model, metadata)
        self._cache: OrderedDict[tuple[str, str], tuple[Any, Any]] = OrderedDict()

    def _get_or_load(
        self,
        language_code: str,
        device: str,
        align_model_name: str | None,
    ) -> tuple[Any, Any]:
        """Return a cached (model, metadata) or load a new one."""
        key = (language_code, device)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug("Reusing cached alignment model for %s", language_code)
            return self._cache[key]

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_cached:
            evicted_key, (old_model, old_meta) = self._cache.popitem(last=False)
            logger.info("Evicting alignment model for %s (LRU)", evicted_key[0])
            del old_model, old_meta
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Loading alignment model for %s on %s", language_code, device)
        model, metadata = load_align_model(
            language_code=language_code,
            device=device,
            model_name=align_model_name,
        )
        self._cache[key] = (model, metadata)
        return model, metadata

    def align(
        self,
        transcript: list[dict[str, Any]],
        audio: np.ndarray[Any, np.dtype[np.float32]],
        language_code: str,
        device: str,
        align_params: AlignmentParams,
    ) -> dict[str, Any]:
        """Align transcript to audio. Must be called while holding the GPU lock."""
        start_time = time.time()
        model, metadata = self._get_or_load(
            language_code, device, align_params.align_model,
        )

        result = align(
            transcript,
            model,
            metadata,
            audio,
            device,
            interpolate_method=align_params.interpolate_method.value,
            return_char_alignments=align_params.return_char_alignments,
        )

        logger.debug("Completed alignment in %.2fs", time.time() - start_time)
        return result  # type: ignore[no-any-return]


class DiarizationModel:
    """Lazily loaded, cached PyAnnote diarization pipeline."""

    def __init__(self, hf_token: str | None) -> None:
        self.hf_token = hf_token
        self.model: Any = None
        self._device: str | None = None

    def diarize(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        device: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Any:
        """Diarize audio. Must be called while holding the GPU lock."""
        if self.model is None or self._device != device:
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()

            logger.info("Loading diarization model on %s", device)
            self.model = DiarizationPipeline(
                token=self.hf_token, device=device
            )
            self._device = device
        else:
            logger.debug("Reusing cached diarization model")

        start_time = time.time()
        result = self.model(
            audio=audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        logger.debug("Completed diarization in %.2fs", time.time() - start_time)
        return result


# ── Global model instances ───────────────────────────────────────────

transcription_model = TranscriptionModel()
alignment_model = AlignmentModel()
diarization_model = DiarizationModel(hf_token=HF_TOKEN)


# ── Pipeline functions ───────────────────────────────────────────────


def run_speech_to_text(
    temp_file: str,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
    split_audio: bool = False,
) -> dict[str, Any]:
    """Run the full speech-to-text pipeline synchronously.

    Audio loading happens OUTSIDE the GPU lock so the next request's
    audio can be read while the current one is still on the GPU.
    """
    if split_audio and is_stereo_audio(temp_file):
        result = _run_split_audio(
            temp_file, model_params, align_params, diarize_params,
            asr_options, vad_options,
        )
        result["is_stereo"] = True
        return result

    # ── Audio loading (CPU/IO — no lock needed) ──
    start_cpu_io = time.time()
    audio = process_audio_file(temp_file)
    audio_load_time = time.time() - start_cpu_io
    logger.debug("Audio loaded in %.2fs", audio_load_time)

    # ── GPU work (transcribe + align + diarize) ──
    start_gpu = time.time()
    
    with transcription_lock():
        start_transcribe = time.time()
        raw = transcription_model.transcribe(audio, model_params, asr_options, vad_options)
        logger.debug("Transcription step took %.2fs", time.time() - start_transcribe)

    with alignment_lock():
        aligned = alignment_model.align(
            transcript=raw["segments"],
            audio=audio,
            language_code=raw["language"],
            device=model_params.device.value,
            align_params=align_params,
        )

    transcript = AlignedTranscription(**aligned)
    filtered = filter_aligned_transcription(transcript)
    transcript_dict = filtered.model_dump()

    with diarization_lock():
        diarization_segments = diarization_model.diarize(
            audio=audio,
            device=model_params.device.value,
            min_speakers=diarize_params.min_speakers,
            max_speakers=diarize_params.max_speakers,
        )
    logger.debug("Total GPU processing time: %.2fs", time.time() - start_gpu)

    # Free memory explicitly before CPU intensive speaker assignment
    del audio
    
    # ── Speaker assignment (CPU-only — no lock needed) ──
    start_assign = time.time()
    result = whisperx.assign_word_speakers(diarization_segments, transcript_dict)
    logger.debug("Speaker assignment took %.2fs", time.time() - start_assign)

    result["is_stereo"] = stereo
    return result  # type: ignore[no-any-return]


def _run_split_audio(
    temp_file: str,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
) -> dict[str, Any]:
    """Split stereo → process L/R channels → merge results."""
    left_file, right_file = split_stereo_to_mono(temp_file)

    channels: dict[str, AlignedTranscription] = {}
    
    # ── Audio loading (CPU/IO) ──
    # Load both channels outside the GPU lock simultaneously
    audio_data = {}
    try:
        for channel_name, channel_file in [("left", left_file), ("right", right_file)]:
            audio_data[channel_name] = load_audio(channel_file)
    finally:
        safe_remove_file(left_file)
        safe_remove_file(right_file)

    from concurrent.futures import ThreadPoolExecutor

    def _process_channel(channel_name: str) -> dict[str, Any]:
        """Transcribe and align a single channel sequentially."""
        # 1. Transcribe (mutually exclusive with other transcriptions)
        with transcription_lock():
            raw = transcription_model.transcribe(
                audio_data[channel_name], model_params, asr_options, vad_options,
            )
            
        # 2. Align (mutually exclusive with other alignments, but CAN overlap 
        #    with the OTHER channel's transcription)
        with alignment_lock():
            return alignment_model.align(
                transcript=raw["segments"],
                audio=audio_data[channel_name],
                language_code=raw["language"],
                device=model_params.device.value,
                align_params=align_params,
            )

    # ── GPU work (transcribe + align interleaved) ──
    # By running both channels concurrently in threads, the separate locks 
    # naturally create an interleaved pipeline:
    # Thread L: [Transcribe Lock] -> [Align Lock]
    # Thread R: (Waits for Trans Lock) -> [Transcribe Lock] -> [Align Lock]
    aligned_results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for channel_name, result in zip(["left", "right"], executor.map(_process_channel, ["left", "right"])):
            aligned_results[channel_name] = result

    # ── Filtering (CPU — no lock needed) ──
    for channel_name in ["left", "right"]:
        transcript = AlignedTranscription(**aligned_results[channel_name])
        filtered = filter_aligned_transcription(transcript, channel_name)
        channels[channel_name] = filtered

    merged = _merge_channel_results(channels)
    return merged.model_dump()


def _merge_channel_results(
    channels: dict[str, AlignedTranscription],
) -> AlignedTranscription:
    """Merge transcription results from multiple channels."""
    merged_segments: List[LabeledSegment] = []
    merged_words: List[LabeledWord] = []

    for channel, transcription in channels.items():
        for segment in transcription.segments:
            new_segment = LabeledSegment(
                **segment.model_dump(exclude={"speaker", "words"}),
                words=[],
                speaker=channel,
            )
            if segment.words:
                new_segment.words = [
                    LabeledWord(**w.model_dump(exclude={"speaker"}), speaker=channel)
                    for w in segment.words
                ]
            merged_segments.append(new_segment)

        if transcription.word_segments:
            for word in transcription.word_segments:
                merged_words.append(
                    LabeledWord(**word.model_dump(exclude={"speaker"}), speaker=channel)
                )

    merged_segments.sort(key=lambda s: s.start or 0.0)
    merged_words.sort(key=lambda w: w.start or 0.0)

    return AlignedTranscription(segments=merged_segments, word_segments=merged_words)
