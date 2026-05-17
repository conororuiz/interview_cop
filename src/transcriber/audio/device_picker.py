"""Selects the right capture backend for the current platform."""

from __future__ import annotations

import logging
import platform

from .capture import AudioCapture, CaptureConfig

log = logging.getLogger(__name__)


def select_capture_backend(config: CaptureConfig) -> AudioCapture:
    system = platform.system().lower()
    if system == "windows":
        from .windows_wasapi import WasapiLoopbackCapture
        log.info("Selecting WASAPI loopback backend (Windows).")
        return WasapiLoopbackCapture(config)
    if system == "linux":
        from .linux_pulse import PulseMonitorCapture
        log.info("Selecting PulseAudio/PipeWire monitor backend (Linux).")
        return PulseMonitorCapture(config)
    if system == "darwin":
        from .macos_coreaudio import CoreAudioLoopbackCapture
        log.info("Selecting CoreAudio loopback backend (macOS, requires BlackHole).")
        return CoreAudioLoopbackCapture(config)
    raise RuntimeError(f"Unsupported platform: {system}")
