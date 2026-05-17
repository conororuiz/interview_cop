"""Linux capture using the PulseAudio/PipeWire monitor source.

On modern Linux desktops (Ubuntu, Fedora, Arch, etc.) PipeWire — or PulseAudio
in compatibility mode — automatically creates a `.monitor` source for every
sink. Reading that source gives us the audio mix going to that sink, which is
exactly what we need.

Detection order:
 1. If `device_hint` is provided, match a source whose name contains the hint.
 2. Else parse `pactl info` to find the default sink, then open
    `<default_sink>.monitor`.
 3. Else fall back to the first available `.monitor` source.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import numpy as np

from .capture import AudioCapture, CaptureChunk, CaptureConfig, DeviceInfo

log = logging.getLogger(__name__)


def _pactl_default_sink() -> str | None:
    if shutil.which("pactl") is None:
        return None
    try:
        out = subprocess.check_output(["pactl", "info"], text=True, timeout=3)
        for line in out.splitlines():
            if line.lower().startswith("default sink:"):
                return line.split(":", 1)[1].strip()
    except Exception as e:
        log.debug("pactl info failed: %s", e)
    return None


def _list_monitor_sources() -> list[str]:
    if shutil.which("pactl") is None:
        return []
    try:
        out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True, timeout=3)
        return [line.split("\t")[1] for line in out.splitlines() if "\t" in line and ".monitor" in line]
    except Exception as e:  # pragma: no cover
        log.debug("pactl list sources failed: %s", e)
        return []


class PulseMonitorCapture(AudioCapture):
    def __init__(self, config: CaptureConfig):
        super().__init__(config)
        self._stream = None
        self._thread: threading.Thread | None = None

    def _pick_source(self) -> str:
        hint = (self.config.device_hint or "").lower().strip()
        sources = _list_monitor_sources()
        if hint:
            for s in sources:
                if hint in s.lower():
                    return s
        default_sink = _pactl_default_sink()
        if default_sink:
            target = f"{default_sink}.monitor"
            if target in sources or not sources:
                return target
        if sources:
            return sources[0]
        raise RuntimeError(
            "No PulseAudio/PipeWire monitor sources detected. "
            "Make sure pulseaudio-utils is installed and a sound server is running."
        )

    def start(self) -> None:
        import sounddevice as sd  # type: ignore

        source = self._pick_source()
        log.info("Opening Pulse monitor source: %s", source)
        # sounddevice uses PortAudio's PulseAudio host API; passing the
        # device name routes to the matching source.
        sd.default.device = (source, None)

        self._device = DeviceInfo(
            name=source,
            sample_rate=self.config.sample_rate,
            channels=1,
            api_name="PulseAudio/PipeWire",
            is_loopback=True,
        )

        block = self.config.block_frames
        start_t = time.monotonic()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("sounddevice status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            ts = start_t + (time_info.inputBufferAdcTime if hasattr(time_info, "inputBufferAdcTime") else 0)
            self._publish(CaptureChunk(
                pcm=np.asarray(mono, dtype=np.float32, order="C").copy(),
                timestamp=ts,
                device_name=source,
            ))

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            blocksize=block,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        log.info("PulseAudio/PipeWire capture started @ %d Hz mono", self.config.sample_rate)

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as e:  # pragma: no cover
            log.warning("Error stopping Pulse stream: %s", e)
        self._stream = None
        log.info("PulseAudio capture stopped.")
