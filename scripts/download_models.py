"""Pre-download all models so the first `transcriber` run is instant.

Downloads (~8 GB total on a fresh install):
  * Silero VAD                           ~  2 MB  (CPU ONNX)
  * faster-whisper large-v3              ~3.0 GB
  * NLLB-200 distilled-1.3B              ~5.2 GB

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-nllb     # ASR only
    python scripts/download_models.py --skip-whisper  # translation only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcriber.config import get_settings  # noqa: E402
from transcriber.hardware.accel import detect_accel, describe  # noqa: E402
from transcriber.logging_setup import setup_logging  # noqa: E402


def _download_silero() -> None:
    print("\n=> Silero VAD")
    from transcriber.pipeline.vad import SileroVAD
    t0 = time.monotonic()
    SileroVAD()
    print(f"   ✓ ready ({time.monotonic() - t0:.1f}s)")


def _download_whisper() -> None:
    print("\n=> faster-whisper (this can take a while on first run)")
    from transcriber.asr.whisper_engine import WhisperEngine
    s = get_settings()
    print(f"   model: {s.whisper_model}")
    t0 = time.monotonic()
    WhisperEngine(accel=detect_accel())
    print(f"   ✓ ready ({time.monotonic() - t0:.1f}s)")


def _download_nllb() -> None:
    print("\n=> NLLB-200 translator")
    from transcriber.translation.nllb_engine import NLLBTranslator
    t0 = time.monotonic()
    NLLBTranslator()
    print(f"   ✓ ready ({time.monotonic() - t0:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-nllb", action="store_true")
    args = parser.parse_args()

    setup_logging()
    print(describe())

    if not args.skip_vad:
        _download_silero()
    if not args.skip_whisper:
        _download_whisper()
    if not args.skip_nllb:
        _download_nllb()

    print("\nAll requested models are cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
