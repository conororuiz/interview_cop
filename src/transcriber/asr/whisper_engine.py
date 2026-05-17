"""ASR engine built on `faster-whisper` (CTranslate2).

Why faster-whisper over openai-whisper:
  * 4-5x faster on the same hardware with no quality loss.
  * Native CUDA / CPU / int8 quantization without extra dependencies.
  * Stable streaming-friendly API: returns segments + per-word timestamps
    along with the auto-detected language and detection probability.

Why a thin wrapper:
  * The orchestrator must work in mono float32 numpy arrays (the same format
    the rest of the pipeline uses).
  * We need to disable the library's built-in VAD: we do our own upstream
    so segments arrive already trimmed to speech.
  * We want to apply a `condition_on_previous_text` strategy that resets on
    long pauses to avoid runaway hallucinations across segments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import get_settings
from ..hardware.accel import AccelProfile, detect_accel

log = logging.getLogger(__name__)


@dataclass
class TranscriptWord:
    word: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    language: str                       # ISO 639-1, e.g. "en", "es"
    language_prob: float
    words: list[TranscriptWord]
    duration_s: float
    asr_seconds: float                  # wall-time spent in Whisper
    rtf: float                          # asr_seconds / duration_s

    @property
    def is_spanish(self) -> bool:
        return self.language.lower().startswith("es")


class WhisperEngine:
    def __init__(
        self,
        model_name: str | None = None,
        accel: AccelProfile | None = None,
    ):
        from faster_whisper import WhisperModel  # type: ignore

        s = get_settings()
        self._settings = s
        self._accel = accel or detect_accel()
        self._model_name = model_name or s.whisper_model

        log.info(
            "Loading faster-whisper '%s' on %s (%s)...",
            self._model_name, self._accel.ct2_device, self._accel.ct2_compute_type,
        )
        t0 = time.monotonic()
        # download_root keeps models inside the project, predictable for users.
        self._model = WhisperModel(
            self._model_name,
            device=self._accel.ct2_device,
            compute_type=self._accel.ct2_compute_type,
            download_root=str(s.models_dir / "faster-whisper"),
        )
        load_s = time.monotonic() - t0
        log.info("Whisper loaded in %.2fs.", load_s)

        # Carry-over prompt across consecutive segments improves coherence,
        # but we drop it after long silence to avoid hallucination drift.
        self._previous_text: Optional[str] = None
        self._previous_lang: Optional[str] = None

    # --- public API ---

    def transcribe(
        self,
        pcm_16k_mono: np.ndarray,
        language_hint: str | None = None,
        carry_context: bool = True,
        preview: bool = False,
    ) -> Transcript:
        """Transcribe a numpy float32 mono 16 kHz buffer.

        Args:
            preview: if True, use fast settings (beam_size=1, no word timestamps)
                suitable for in-progress segment previews. Lower quality than
                the final pass but typically 2-3x faster on the GPU.
        """
        s = self._settings
        if pcm_16k_mono.dtype != np.float32:
            pcm_16k_mono = pcm_16k_mono.astype(np.float32, copy=False)
        if pcm_16k_mono.ndim != 1:
            pcm_16k_mono = pcm_16k_mono.reshape(-1)
        duration = pcm_16k_mono.size / 16000.0
        if duration <= 0.0:
            return Transcript(
                text="", language="", language_prob=0.0, words=[],
                duration_s=0.0, asr_seconds=0.0, rtf=0.0,
            )

        # Previews never consume the carry-over context — they are throwaway.
        initial_prompt = self._previous_text if (carry_context and not preview) else None

        beam = 1 if preview else s.whisper_beam_size
        want_words = not preview

        t0 = time.monotonic()
        segments_gen, info = self._model.transcribe(
            pcm_16k_mono,
            language=language_hint,
            task="transcribe",
            beam_size=beam,
            temperature=s.whisper_temperature,
            condition_on_previous_text=(
                s.whisper_condition_on_previous and carry_context and not preview
            ),
            initial_prompt=initial_prompt,
            vad_filter=False,            # we already segmented upstream
            word_timestamps=want_words,
            # Light stability tweaks: penalise repetition, allow some no-speech.
            repetition_penalty=1.1,
            no_speech_threshold=0.6,
        )
        # `transcribe` returns a generator; iterate to actually run inference.
        text_parts: list[str] = []
        words: list[TranscriptWord] = []
        for seg in segments_gen:
            text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append(TranscriptWord(
                        word=w.word, start=float(w.start), end=float(w.end),
                        prob=float(w.probability),
                    ))
        elapsed = time.monotonic() - t0

        text = "".join(text_parts).strip()
        # Update carry-over context — never from previews.
        if carry_context and text and not preview:
            # Keep only last ~400 chars to stay within the prompt budget.
            self._previous_text = (self._previous_text or "")[-200:] + " " + text
            self._previous_text = self._previous_text[-400:]
            self._previous_lang = info.language

        return Transcript(
            text=text,
            language=info.language,
            language_prob=float(info.language_probability),
            words=words,
            duration_s=duration,
            asr_seconds=elapsed,
            rtf=elapsed / duration if duration > 0 else 0.0,
        )

    def reset_context(self) -> None:
        """Forget previous-text conditioning (call on long pauses or topic change)."""
        self._previous_text = None
        self._previous_lang = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def accel(self) -> AccelProfile:
        return self._accel
