"""Windows WASAPI loopback capture using PyAudioWPatch.

WASAPI loopback lets us read the audio mix being sent to *any* render
endpoint (speakers / headphones) without virtual cables. PyAudioWPatch
exposes loopback-only devices as separate entries; we prefer the loopback
counterpart of the system's default output device.

The capture runs in a callback on PortAudio's audio thread. We resample
inline to the target sample rate (default 16 kHz mono) because Whisper /
Silero are trained at 16 kHz and we want a stable contract upstream.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .capture import AudioCapture, CaptureChunk, CaptureConfig, DeviceInfo

log = logging.getLogger(__name__)


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap mono linear resampler. We do NOT need high-quality SRC here —
    Whisper is robust and the cost matters in the audio callback."""
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * dst_sr / src_sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    base = np.floor(idx).astype(np.int64)
    frac = (idx - base).astype(np.float32)
    base_next = np.minimum(base + 1, len(x) - 1)
    out = (1 - frac) * x[base] + frac * x[base_next]
    return out.astype(np.float32, copy=False)


class WasapiLoopbackCapture(AudioCapture):
    """Capture from the loopback device matching the default WASAPI output."""

    def __init__(self, config: CaptureConfig):
        super().__init__(config)
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._src_sr: int = 0
        self._src_channels: int = 0
        self._block_start_t: float = 0.0
        self._lock = threading.Lock()

    # --- backend probing ---

    def _find_loopback_device(self) -> tuple[int, dict]:
        import pyaudiowpatch as pa  # type: ignore

        self._pa = self._pa or pa.PyAudio()
        # Prefer the loopback of the default WASAPI output.
        try:
            default_out = self._pa.get_default_wasapi_loopback()
            log.info("Default WASAPI loopback: %s", default_out.get("name"))
            return int(default_out["index"]), default_out
        except OSError:
            log.debug("No default loopback returned, scanning manually...")

        # Manual scan: pick first loopback device, biased by hint.
        hint = (self.config.device_hint or "").lower().strip()
        candidates = []
        for info in self._pa.get_loopback_device_info_generator():
            name = info.get("name", "")
            if hint and hint in name.lower():
                return int(info["index"]), info
            candidates.append((int(info["index"]), info))

        if not candidates:
            raise RuntimeError(
                "No WASAPI loopback devices found. Ensure Windows audio service "
                "is running and that PyAudioWPatch is installed correctly."
            )
        idx, info = candidates[0]
        log.info("Falling back to first loopback device: %s", info.get("name"))
        return idx, info

    # --- lifecycle ---

    def start(self) -> None:
        import pyaudiowpatch as pa  # type: ignore

        idx, info = self._find_loopback_device()
        self._src_sr = int(info["defaultSampleRate"])
        self._src_channels = int(info["maxInputChannels"])
        self._device = DeviceInfo(
            name=info["name"],
            sample_rate=self._src_sr,
            channels=self._src_channels,
            api_name="WASAPI",
            is_loopback=True,
        )

        # PortAudio block size at the source SR; resampling happens in the callback.
        src_block = int(self._src_sr * self.config.block_ms / 1000)

        self._block_start_t = time.monotonic()

        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            if status:
                log.debug("PortAudio status flags: %r", status)
            # Interleaved float32 frames from WASAPI loopback.
            arr = np.frombuffer(in_data, dtype=np.float32)
            if self._src_channels > 1:
                arr = arr.reshape(-1, self._src_channels).mean(axis=1)
            mono16k = _resample_linear(arr, self._src_sr, self.config.sample_rate)
            with self._lock:
                ts = self._block_start_t
                # advance by the duration actually consumed
                self._block_start_t += frame_count / self._src_sr
            self._publish(CaptureChunk(
                pcm=mono16k,
                timestamp=ts,
                device_name=self._device.name if self._device else "",
            ))
            return (None, pa.paContinue)

        self._stream = self._pa.open(  # type: ignore[union-attr]
            format=pa.paFloat32,
            channels=self._src_channels,
            rate=self._src_sr,
            frames_per_buffer=src_block,
            input=True,
            input_device_index=idx,
            stream_callback=callback,
        )
        self._stream.start_stream()
        log.info(
            "WASAPI loopback stream started: %s @ %d Hz x%d -> %d Hz mono",
            self._device.name, self._src_sr, self._src_channels, self.config.sample_rate,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception as e:  # pragma: no cover
            log.warning("Error stopping WASAPI stream: %s", e)
        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception as e:  # pragma: no cover
            log.warning("Error terminating PyAudio: %s", e)
        self._stream = None
        self._pa = None
        log.info("WASAPI loopback stream stopped.")
