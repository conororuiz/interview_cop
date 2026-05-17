"""Hardware acceleration detection.

Returns a normalised `AccelProfile` describing the best execution backend
for both faster-whisper (CTranslate2) and PyTorch (NLLB). We probe carefully
so a broken CUDA install falls back to CPU rather than crashing at runtime.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Literal

from ..config import get_settings

log = logging.getLogger(__name__)

CTDevice = Literal["cuda", "cpu"]
TorchDevice = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class AccelProfile:
    ct2_device: CTDevice          # device string for CTranslate2 / faster-whisper
    ct2_compute_type: str         # e.g. "float16", "int8_float16", "int8"
    torch_device: TorchDevice     # device string for PyTorch / NLLB
    torch_dtype: str              # "float16" | "bfloat16" | "float32"
    gpu_name: str | None
    notes: str

    @property
    def using_gpu(self) -> bool:
        return self.ct2_device == "cuda" or self.torch_device in ("cuda", "mps")


def _probe_cuda() -> tuple[bool, str | None]:
    """Return (is_available, gpu_name)."""
    try:
        import torch  # noqa: WPS433  (heavy import deferred)
    except Exception as e:  # pragma: no cover
        log.warning("torch not importable: %s", e)
        return False, None
    try:
        ok = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if ok else None
        return ok, name
    except Exception as e:  # pragma: no cover
        log.warning("CUDA probe failed: %s", e)
        return False, None


def _probe_mps() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def detect_accel() -> AccelProfile:
    """Choose the best available acceleration profile, honouring user override."""
    s = get_settings()
    forced = s.compute_device

    cuda_ok, gpu_name = _probe_cuda()
    mps_ok = _probe_mps()
    is_apple_silicon = platform.machine().lower() in {"arm64", "aarch64"} and platform.system() == "Darwin"

    # User override path
    if forced != "auto":
        if forced == "cuda" and not cuda_ok:
            log.warning("CUDA forced but unavailable, falling back to CPU.")
            forced = "cpu"
        if forced == "mps" and not mps_ok:
            log.warning("MPS forced but unavailable, falling back to CPU.")
            forced = "cpu"

    # Automatic choice
    if forced == "cuda" or (forced == "auto" and cuda_ok):
        ct = "float16"  # best balance on modern NVIDIA
        return AccelProfile(
            ct2_device="cuda",
            ct2_compute_type=s.compute_type or ct,
            torch_device="cuda",
            torch_dtype="float16",
            gpu_name=gpu_name,
            notes=f"CUDA enabled ({gpu_name})",
        )

    if forced == "mps" or (forced == "auto" and is_apple_silicon and mps_ok):
        # CTranslate2 has no MPS backend; falls back to CPU int8 for whisper.
        return AccelProfile(
            ct2_device="cpu",
            ct2_compute_type=s.compute_type or "int8",
            torch_device="mps",
            torch_dtype="float16",
            gpu_name="Apple Silicon (MPS)",
            notes="Whisper runs on CPU int8; NLLB on MPS",
        )

    # CPU fallback
    return AccelProfile(
        ct2_device="cpu",
        ct2_compute_type=s.compute_type or "int8",
        torch_device="cpu",
        torch_dtype="float32",
        gpu_name=None,
        notes="CPU fallback (int8 Whisper)",
    )


def describe() -> str:
    p = detect_accel()
    return (
        f"ASR  : CT2 device={p.ct2_device} compute_type={p.ct2_compute_type}\n"
        f"NLLB : torch device={p.torch_device} dtype={p.torch_dtype}\n"
        f"GPU  : {p.gpu_name or 'n/a'}\n"
        f"Notes: {p.notes}"
    )
