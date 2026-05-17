"""Silero VAD v5 wrapper using the official `silero-vad` package.

Originally we hand-rolled an onnxruntime wrapper. That turned out to be
brittle (the model's I/O signature has shifted across releases), so we now
delegate to the official `silero_vad` Python package which guarantees a
stable API and bundles the matching model weights.

The package exposes a JIT-scripted PyTorch model with a `reset_states()`
method and `__call__(chunk_tensor, sr)` returning speech probability. We
keep this wrapper minimal and synchronous so the streaming pipeline owns
all chunking logic.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)

SILERO_CHUNK = 512   # samples (= 32 ms at 16 kHz). Required by Silero v5 @ 16 kHz.
SILERO_SR = 16000


class SileroVAD:
    """Stateful Silero VAD v5. Process exactly 512-sample 16 kHz chunks."""

    def __init__(self, threshold: float | None = None, use_onnx: bool = True):
        # Import lazily so that `import transcriber.config` doesn't pull torch.
        import torch  # noqa: WPS433
        from silero_vad import load_silero_vad  # type: ignore

        self._torch = torch
        self._settings = get_settings()

        if threshold is None:
            # Silero recommends 0.5 as default. Map our aggressiveness 0..1 to
            # 0.35..0.65 so users can be lenient (more speech) or strict.
            threshold = 0.35 + 0.30 * self._settings.vad_aggressiveness
        self.threshold = float(threshold)

        # Force CPU — VAD overhead is negligible and we want CUDA memory for Whisper.
        self._model = load_silero_vad(onnx=use_onnx)
        self._sr = SILERO_SR

        # Move torch backend to CPU (no-op for onnx); reset state to known zero.
        if not use_onnx:
            self._model.to("cpu")
            self._model.eval()
        self.reset()

        log.info(
            "Silero VAD ready (backend=%s, threshold=%.2f).",
            "onnx" if use_onnx else "torch", self.threshold,
        )

    def reset(self) -> None:
        """Reset internal LSTM state (call when starting a fresh stream)."""
        try:
            self._model.reset_states()
        except AttributeError:  # pragma: no cover  (some backends differ)
            pass

    def process_chunk(self, chunk_512: np.ndarray) -> float:
        """Return speech probability in [0, 1] for one 512-sample 16 kHz chunk."""
        if chunk_512.size != SILERO_CHUNK:
            raise ValueError(
                f"SileroVAD.process_chunk expected {SILERO_CHUNK} samples, got {chunk_512.size}"
            )
        if chunk_512.dtype != np.float32:
            chunk_512 = chunk_512.astype(np.float32, copy=False)

        tensor = self._torch.from_numpy(chunk_512)
        with self._torch.no_grad():
            prob = self._model(tensor, self._sr).item()
        return float(prob)

    def is_speech(self, prob: float) -> bool:
        return prob >= self.threshold
