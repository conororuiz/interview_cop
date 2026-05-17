"""Application configuration.

Centralized, env-driven settings using pydantic-settings. All modules read
configuration through `get_settings()` to keep behaviour reproducible and
overridable from `.env` or environment variables without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ComputeDevice = Literal["auto", "cuda", "mps", "cpu"]
WhisperModel = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]
TranslationBackend = Literal["nllb-1.3b", "nllb-600m", "deepl"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Hardware ---
    compute_device: ComputeDevice = "auto"
    compute_type: str | None = None  # e.g. "float16", "int8_float16". None = auto.

    # --- Audio ---
    sample_rate: int = 16000
    channels: int = 1
    audio_device_hint: str | None = None
    capture_block_ms: int = 30  # PortAudio callback block size in ms

    # --- VAD / segmentation ---
    vad_aggressiveness: float = Field(0.5, ge=0.0, le=1.0)
    min_segment_ms: int = 1500
    max_segment_ms: int = 10000          # tighter -> lower worst-case latency
    segment_overlap_ms: int = 400
    silence_tail_ms: int = 300           # close faster on brief pauses

    # --- Live preview (interim transcription while a segment is still open) ---
    preview_enabled: bool = True
    preview_interval_ms: int = 1500       # how often to re-transcribe the open segment
    preview_min_audio_ms: int = 1500      # show first preview after this much audio

    # --- ASR ---
    whisper_model: WhisperModel = "large-v3"
    whisper_beam_size: int = 5
    whisper_temperature: float = 0.0
    whisper_condition_on_previous: bool = True

    # --- Translation ---
    translation_backend: TranslationBackend = "nllb-1.3b"
    target_language: str = "es"  # ISO 639-1
    deepl_api_key: str | None = None

    # --- Paths ---
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- Logging ---
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
