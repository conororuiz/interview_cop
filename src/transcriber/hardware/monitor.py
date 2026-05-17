"""Live hardware monitoring (CPU + GPU).

We use:
  * `psutil` for CPU utilisation (cross-platform, no compilation).
  * `pynvml` for NVIDIA GPU utilisation + VRAM (much more accurate than
    parsing `nvidia-smi` and avoids subprocess overhead).

Falls back gracefully when pynvml is missing (e.g. on macOS) or no NVIDIA
device is present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareSample:
    cpu_percent: float          # 0..100
    ram_percent: float          # 0..100
    gpu_percent: float | None   # 0..100 or None
    vram_used_mb: float | None
    vram_total_mb: float | None
    gpu_temp_c: float | None    # degrees Celsius

    @property
    def has_gpu(self) -> bool:
        return self.gpu_percent is not None


class HardwareMonitor:
    def __init__(self):
        self._psutil_ok = False
        self._nvml_ok = False
        self._nvml_handle = None
        try:
            import psutil  # noqa: F401
            self._psutil_ok = True
            # First call returns 0, prime it.
            import psutil as _p
            _p.cpu_percent(interval=None)
        except Exception as e:
            log.debug("psutil not available: %s", e)

        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
        except Exception as e:
            log.debug("pynvml not available: %s", e)

    def sample(self) -> HardwareSample:
        cpu = 0.0
        ram = 0.0
        if self._psutil_ok:
            import psutil
            cpu = float(psutil.cpu_percent(interval=None))
            ram = float(psutil.virtual_memory().percent)

        gpu = None
        vram_used = None
        vram_total = None
        gpu_temp = None
        if self._nvml_ok:
            try:
                pynvml = self._pynvml
                h = self._nvml_handle
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpu = float(util.gpu)
                vram_used = mem.used / (1024 * 1024)
                vram_total = mem.total / (1024 * 1024)
                try:
                    gpu_temp = float(pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    ))
                except Exception:
                    pass
            except Exception as e:
                log.debug("NVML sample failed: %s", e)

        return HardwareSample(
            cpu_percent=cpu, ram_percent=ram,
            gpu_percent=gpu, vram_used_mb=vram_used, vram_total_mb=vram_total,
            gpu_temp_c=gpu_temp,
        )

    def shutdown(self) -> None:
        if self._nvml_ok:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
