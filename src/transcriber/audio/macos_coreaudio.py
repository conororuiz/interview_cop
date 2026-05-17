"""macOS capture via a virtual loopback device (BlackHole).

CoreAudio does not expose system audio for capture natively. The standard
solution is the open-source `BlackHole` audio driver, which creates a virtual
output device that simultaneously appears as an input. Users either:

  * route system audio to BlackHole directly (loses local playback), or
  * create a Multi-Output Device combining BlackHole + speakers (recommended).

The installer script gives the user one-line instructions. Here we detect
BlackHole among input devices and open it via sounddevice.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .capture import AudioCapture, CaptureChunk, CaptureConfig, DeviceInfo

log = logging.getLogger(__name__)


class CoreAudioLoopbackCapture(AudioCapture):
    def __init__(self, config: CaptureConfig):
        super().__init__(config)
        self._stream = None

    def _find_blackhole(self) -> tuple[int, str]:
        import sounddevice as sd  # type: ignore

        hint = (self.config.device_hint or "blackhole").lower()
        devs = sd.query_devices()
        # First pass: explicit match.
        for i, d in enumerate(devs):
            if d["max_input_channels"] >= 1 and hint in d["name"].lower():
                return i, d["name"]
        # Fallback: any device whose name suggests loopback.
        for i, d in enumerate(devs):
            n = d["name"].lower()
            if d["max_input_channels"] >= 1 and ("loopback" in n or "soundflower" in n):
                return i, d["name"]
        raise RuntimeError(
            "No loopback device found on macOS. Install BlackHole:\n"
            "    brew install blackhole-2ch\n"
            "Then create a Multi-Output Device in Audio MIDI Setup that combines "
            "BlackHole 2ch with your speakers, and set it as the system output."
        )

    def start(self) -> None:
        import sounddevice as sd  # type: ignore

        idx, name = self._find_blackhole()
        log.info("Opening macOS loopback device: %s (index=%d)", name, idx)
        self._device = DeviceInfo(
            name=name,
            sample_rate=self.config.sample_rate,
            channels=1,
            api_name="CoreAudio",
            is_loopback=True,
        )

        block = self.config.block_frames
        start_t = time.monotonic()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("sounddevice status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self._publish(CaptureChunk(
                pcm=np.asarray(mono, dtype=np.float32, order="C").copy(),
                timestamp=start_t,
                device_name=name,
            ))

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            blocksize=block,
            dtype="float32",
            device=idx,
            callback=callback,
        )
        self._stream.start()
        log.info("CoreAudio loopback capture started @ %d Hz mono", self.config.sample_rate)

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as e:  # pragma: no cover
            log.warning("Error stopping CoreAudio stream: %s", e)
        self._stream = None
        log.info("CoreAudio capture stopped.")
