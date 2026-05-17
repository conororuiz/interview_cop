"""Audio capture abstractions.

Defines a uniform interface across platforms. Concrete implementations live
under `windows_wasapi.py`, `linux_pulse.py`, `macos_coreaudio.py`. The
`AudioCapture` protocol guarantees:

  * synchronous `start()` / `stop()` lifecycle
  * a thread-safe iterator yielding mono float32 frames at the configured
    sample rate, with monotonically increasing timestamps.

Frames are emitted as small fixed-size blocks (default 30 ms) so that the
downstream ring-buffer + VAD can process audio with minimal latency.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 30
    device_hint: str | None = None

    @property
    def block_frames(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass(frozen=True)
class CaptureChunk:
    """A small block of mono PCM audio."""
    pcm: np.ndarray            # float32 [-1, 1], shape (block_frames,)
    timestamp: float           # seconds, monotonic, start of the block
    device_name: str


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    sample_rate: int
    channels: int
    api_name: str               # e.g. "WASAPI", "PulseAudio", "CoreAudio"
    is_loopback: bool


class AudioCapture(ABC):
    """Common interface for all platform-specific capture backends."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._queue: queue.Queue[CaptureChunk] = queue.Queue(maxsize=200)
        self._stop_evt = threading.Event()
        self._device: DeviceInfo | None = None

    # --- public API ---

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    def chunks(self, timeout: float = 1.0) -> Iterator[CaptureChunk]:
        """Yield chunks until `stop()` is called."""
        while not self._stop_evt.is_set():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue

    # --- helpers for subclasses ---

    def _publish(self, chunk: CaptureChunk) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest to keep us close to realtime — log at debug to avoid spam.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
                log.debug("Capture queue full; dropped oldest block.")
            except queue.Empty:  # pragma: no cover
                pass


def cli_record_test() -> None:
    """Tiny CLI: record 30 s of system audio to ./capture_test.wav.

    Entry-point: `transcriber-capture-test`. Useful for the Hito 1 check.
    """
    import argparse
    import time
    from pathlib import Path

    import soundfile as sf

    from ..logging_setup import setup_logging
    from .device_picker import select_capture_backend

    parser = argparse.ArgumentParser(description="Record 30 s of system audio to WAV.")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("capture_test.wav"))
    parser.add_argument("--device-hint", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    cap = select_capture_backend(CaptureConfig(device_hint=args.device_hint))
    cap.start()
    dev = cap.device
    log.info("Recording %ds from device: %s (api=%s, loopback=%s)",
             args.seconds, dev.name if dev else "?", dev.api_name if dev else "?",
             dev.is_loopback if dev else "?")

    buf: list[np.ndarray] = []
    started = time.monotonic()
    try:
        for chunk in cap.chunks():
            buf.append(chunk.pcm)
            if time.monotonic() - started >= args.seconds:
                break
    finally:
        cap.stop()

    audio = np.concatenate(buf) if buf else np.zeros(1, dtype=np.float32)
    sf.write(str(args.output), audio, cap.config.sample_rate, subtype="PCM_16")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    log.info("Wrote %s | duration=%.2fs peak=%.3f rms=%.4f",
             args.output, audio.size / cap.config.sample_rate, peak, rms)
    if peak < 0.001:
        log.warning("Audio looks silent — make sure something was actually playing!")
