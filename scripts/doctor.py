"""Diagnose the environment.

Checks:
  * Python version
  * torch / CUDA / GPU
  * CTranslate2 / faster-whisper import
  * Silero VAD import
  * Transformers / NLLB tokenizer load
  * Capture backend availability (without actually opening a stream)
  * Model cache presence

Usage:
    python scripts/doctor.py
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def section(name: str) -> None:
    print(f"\n[{name}]")


def main() -> int:
    bad = 0

    section("Python")
    print(f"  {sys.version.split()[0]} on {platform.system()} {platform.release()} ({platform.machine()})")
    if sys.version_info < (3, 12):
        warn("Project targets Python 3.12+. Consider upgrading.")

    section("PyTorch / CUDA")
    try:
        import torch
        print(f"  torch={torch.__version__}  cuda_built={torch.version.cuda}  available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB)")
        else:
            warn("CUDA not available — will run on CPU. Whisper large-v3 will be slow.")
    except Exception as e:
        fail(f"torch import failed: {e}")
        bad += 1

    section("CTranslate2 / faster-whisper")
    try:
        import ctranslate2  # noqa: F401
        import faster_whisper  # noqa: F401
        ok("imported")
    except Exception as e:
        fail(f"{e}")
        bad += 1

    section("Silero VAD")
    try:
        from silero_vad import load_silero_vad  # noqa: F401
        ok("silero-vad importable")
    except Exception as e:
        fail(f"{e}")
        bad += 1

    section("Transformers (NLLB)")
    try:
        import transformers  # noqa: F401
        ok(f"transformers={transformers.__version__}")
    except Exception as e:
        fail(f"{e}")
        bad += 1

    section("Audio capture backend")
    sysname = platform.system().lower()
    if sysname == "windows":
        try:
            import pyaudiowpatch  # type: ignore # noqa: F401
            ok("PyAudioWPatch importable (WASAPI loopback)")
        except Exception as e:
            fail(f"PyAudioWPatch missing: {e}")
            bad += 1
    elif sysname == "linux":
        if shutil.which("pactl"):
            ok("pactl found (PulseAudio/PipeWire tools)")
        else:
            fail("`pactl` not found. Install `pulseaudio-utils`.")
            bad += 1
        try:
            import sounddevice  # noqa: F401
            ok("sounddevice importable")
        except Exception as e:
            fail(f"sounddevice missing: {e}")
            bad += 1
    elif sysname == "darwin":
        try:
            import sounddevice as sd  # type: ignore
            devs = sd.query_devices()
            blackhole = any("blackhole" in d["name"].lower() for d in devs)
            if blackhole:
                ok("BlackHole detected (CoreAudio loopback)")
            else:
                warn("BlackHole NOT detected. Install with: brew install blackhole-2ch")
        except Exception as e:
            fail(f"sounddevice missing or audio scan failed: {e}")
            bad += 1

    section("Model cache")
    from transcriber.config import get_settings
    s = get_settings()
    silero = s.models_dir / "silero_vad" / "silero_vad.onnx"
    whisper_cache = s.models_dir / "faster-whisper"
    nllb_cache = s.models_dir / "nllb"
    if silero.exists():
        ok(f"silero VAD downloaded ({silero.stat().st_size // 1024} KB)")
    else:
        warn(f"silero VAD not yet downloaded ({silero})")
    if whisper_cache.exists() and any(whisper_cache.iterdir()):
        ok(f"faster-whisper cache populated under {whisper_cache}")
    else:
        warn(f"faster-whisper cache empty — first run will download ~3 GB")
    if nllb_cache.exists() and any(nllb_cache.iterdir()):
        ok(f"NLLB cache populated under {nllb_cache}")
    else:
        warn(f"NLLB cache empty — first run will download ~5 GB. "
             f"Tip: `python scripts/download_models.py` to pre-download.")

    print()
    if bad == 0:
        print(f"{GREEN}All critical checks passed.{RESET}")
        return 0
    print(f"{RED}{bad} critical check(s) failed. See above.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
