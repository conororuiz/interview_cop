"""Auto-detect the host hardware and write a safe `.env` accordingly.

Run at the end of every install script (or by hand at any time):

    python scripts/autoconfig.py             # detect + write .env
    python scripts/autoconfig.py --dry-run   # print plan, do not write
    python scripts/autoconfig.py --force     # overwrite existing user values

The goal is "it just works on whatever box you install it on":
  * NVIDIA GPU  → pick a CUDA-friendly Whisper / NLLB combo that fits the
    detected VRAM, with the right CTranslate2 compute_type.
  * Apple Silicon → MPS for NLLB, int8 Whisper on CPU.
  * Anything else → CPU-only with much smaller models and looser segment
    timings so the app stays responsive (and doesn't OOM).

Existing values in `.env` are preserved unless `--force` is passed, so user
overrides (API keys, custom Whisper model, etc.) survive a re-install.

Detection uses `nvidia-smi` first (works without torch installed). If that
fails we fall back to pynvml / torch.cuda when present. Everything else
(`psutil`, `platform`) is stdlib-friendly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


# ---------------------------------------------------------------------------
# Colors (best-effort; degrade silently on dumb terminals)
# ---------------------------------------------------------------------------
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


_C = _supports_color()
GREEN  = "\033[92m" if _C else ""
RED    = "\033[91m" if _C else ""
YELLOW = "\033[93m" if _C else ""
CYAN   = "\033[96m" if _C else ""
DIM    = "\033[2m"  if _C else ""
BOLD   = "\033[1m"  if _C else ""
RESET  = "\033[0m"  if _C else ""


# ---------------------------------------------------------------------------
# Hardware probing
# ---------------------------------------------------------------------------
@dataclass
class Hardware:
    system: str
    machine: str
    cpu_cores: int
    ram_gb: float
    has_nvidia: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: float = 0.0
    gpu_compute_capability: Optional[str] = None  # e.g. "8.6"
    is_apple_silicon: bool = False
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command quietly, returning (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, "", ""


def _detect_ram_gb() -> float:
    """Total RAM in GB. Uses psutil if present, else /proc/meminfo or wmic."""
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # Linux fallback
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    # Windows fallback via wmic (deprecated but still present on most boxes)
    if platform.system() == "Windows":
        rc, out, _ = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"])
        if rc == 0:
            m = re.search(r"=(\d+)", out)
            if m:
                return round(int(m.group(1)) / (1024 ** 3), 1)
    # macOS fallback
    if platform.system() == "Darwin":
        rc, out, _ = _run(["sysctl", "-n", "hw.memsize"])
        if rc == 0 and out.isdigit():
            return round(int(out) / (1024 ** 3), 1)
    return 0.0


def _detect_nvidia() -> tuple[bool, Optional[str], float, Optional[str]]:
    """Return (has_nvidia, name, vram_gb, compute_capability)."""
    # Preferred path: nvidia-smi (works pre-torch-install)
    if shutil.which("nvidia-smi"):
        rc, out, _ = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        if rc == 0 and out:
            # First GPU only — multi-GPU users tend to set CUDA_VISIBLE_DEVICES.
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    vram_mb = float(parts[1])
                    vram_gb = round(vram_mb / 1024, 1)
                except ValueError:
                    vram_gb = 0.0
                cc = parts[2] if len(parts) >= 3 and parts[2] else None
                return True, name, vram_gb, cc

    # Fallback: pynvml if installed
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                vram_gb = round(mem.total / (1024 ** 3), 1)
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                    cc = f"{major}.{minor}"
                except Exception:
                    cc = None
                return True, name, vram_gb, cc
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    # Last resort: torch
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            try:
                cap = torch.cuda.get_device_capability(0)
                cc = f"{cap[0]}.{cap[1]}"
            except Exception:
                cc = None
            return True, name, vram_gb, cc
    except Exception:
        pass

    return False, None, 0.0, None


def detect_hardware() -> Hardware:
    hw = Hardware(
        system=platform.system(),
        machine=platform.machine(),
        cpu_cores=os.cpu_count() or 0,
        ram_gb=_detect_ram_gb(),
    )
    hw.is_apple_silicon = (hw.system == "Darwin" and hw.machine.lower() in {"arm64", "aarch64"})
    has_nv, name, vram, cc = _detect_nvidia()
    hw.has_nvidia = has_nv
    hw.gpu_name = name
    hw.gpu_vram_gb = vram
    hw.gpu_compute_capability = cc
    if has_nv and cc:
        try:
            major = int(cc.split(".")[0])
            if major <= 6:
                hw.notes.append(
                    "Older NVIDIA architecture detected (Pascal or earlier). "
                    "float16 may be slow on this card; using int8_float16."
                )
        except Exception:
            pass
    return hw


# ---------------------------------------------------------------------------
# Tier picker
# ---------------------------------------------------------------------------
@dataclass
class Tier:
    name: str                # human label (e.g. "GPU-HIGH (large-v3 / NLLB-1.3B)")
    settings: dict[str, str] # {ENV_KEY: value}
    rationale: str


def _gpu_tier_from_vram(vram: float, cc: Optional[str]) -> Tier:
    """Pick a tier for an NVIDIA GPU based on VRAM and architecture."""
    # Pascal (cc 6.x) and older have weak / no fp16 — int8 is faster there.
    is_old_arch = False
    if cc:
        try:
            is_old_arch = int(cc.split(".")[0]) <= 6
        except Exception:
            pass

    if vram >= 12:
        return Tier(
            name="GPU-HIGH (large-v3 fp16 + NLLB-1.3B)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cuda",
                "TRANSCRIBER_WHISPER_MODEL": "large-v3",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-1.3b",
                "TRANSCRIBER_COMPUTE_TYPE": "float16",
                "TRANSCRIBER_MAX_SEGMENT_MS": "10000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "1500",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "1500",
            },
            rationale=f"{vram:.0f} GB VRAM is enough for large-v3 in fp16 plus NLLB-1.3B with headroom.",
        )
    if vram >= 8:
        return Tier(
            name="GPU-UPPER-MID (large-v3 int8_fp16 + NLLB-1.3B)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cuda",
                "TRANSCRIBER_WHISPER_MODEL": "large-v3",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-1.3b",
                "TRANSCRIBER_COMPUTE_TYPE": "int8_float16",
                "TRANSCRIBER_MAX_SEGMENT_MS": "10000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "1500",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "1500",
            },
            rationale=f"{vram:.0f} GB VRAM fits large-v3 if Whisper is int8_fp16 (~1 GB saved).",
        )
    if vram >= 6:
        return Tier(
            name="GPU-MID (medium fp16 + NLLB-1.3B)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cuda",
                "TRANSCRIBER_WHISPER_MODEL": "medium",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-1.3b",
                "TRANSCRIBER_COMPUTE_TYPE": "int8_float16" if is_old_arch else "float16",
                "TRANSCRIBER_MAX_SEGMENT_MS": "10000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "2000",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "2000",
            },
            rationale=f"{vram:.0f} GB VRAM → step down to medium so NLLB-1.3B still fits.",
        )
    if vram >= 4:
        return Tier(
            name="GPU-LOW (small int8_fp16 + NLLB-600M)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cuda",
                "TRANSCRIBER_WHISPER_MODEL": "small",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
                "TRANSCRIBER_COMPUTE_TYPE": "int8_float16" if not is_old_arch else "int8",
                "TRANSCRIBER_MAX_SEGMENT_MS": "12000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "2500",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "2500",
            },
            rationale=f"{vram:.0f} GB VRAM → small Whisper + NLLB-600M is the safe combo.",
        )
    if vram >= 2:
        return Tier(
            name="GPU-VERY-LOW (small int8 + NLLB-600M)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cuda",
                "TRANSCRIBER_WHISPER_MODEL": "small",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
                "TRANSCRIBER_COMPUTE_TYPE": "int8",
                "TRANSCRIBER_MAX_SEGMENT_MS": "14000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "3000",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "3000",
            },
            rationale=(
                f"{vram:.0f} GB VRAM is tight (GTX 1060-class). Pure int8 + smallest "
                "viable models, with looser segment limits to keep memory steady."
            ),
        )
    # <2 GB VRAM → CPU is actually safer.
    return _cpu_tier(ram_gb=999, reason=f"GPU has only {vram:.1f} GB VRAM — CPU is safer.")


def _apple_tier(ram_gb: float) -> Tier:
    if ram_gb >= 16:
        return Tier(
            name="APPLE-HIGH (medium MPS + NLLB-1.3B)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "mps",
                "TRANSCRIBER_WHISPER_MODEL": "medium",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-1.3b",
                "TRANSCRIBER_MAX_SEGMENT_MS": "12000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "2500",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "2500",
            },
            rationale=f"Apple Silicon with {ram_gb:.0f} GB unified memory — NLLB on MPS, Whisper int8 on CPU.",
        )
    return Tier(
        name="APPLE-LOW (small MPS + NLLB-600M)",
        settings={
            "TRANSCRIBER_COMPUTE_DEVICE": "mps",
            "TRANSCRIBER_WHISPER_MODEL": "small",
            "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
            "TRANSCRIBER_MAX_SEGMENT_MS": "12000",
            "TRANSCRIBER_PREVIEW_INTERVAL_MS": "3000",
            "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "3000",
        },
        rationale=f"Apple Silicon with {ram_gb:.0f} GB unified memory — using the lighter combo.",
    )


def _cpu_tier(ram_gb: float, reason: str = "") -> Tier:
    base_reason = reason or "No usable GPU detected."
    if ram_gb >= 16:
        return Tier(
            name="CPU-HIGH (medium int8 + NLLB-600M)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cpu",
                "TRANSCRIBER_WHISPER_MODEL": "medium",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
                "TRANSCRIBER_COMPUTE_TYPE": "int8",
                "TRANSCRIBER_MAX_SEGMENT_MS": "12000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "3000",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "3000",
            },
            rationale=f"{base_reason} {ram_gb:.0f} GB RAM is enough for Whisper medium on CPU.",
        )
    if ram_gb >= 8:
        return Tier(
            name="CPU-MID (small int8 + NLLB-600M)",
            settings={
                "TRANSCRIBER_COMPUTE_DEVICE": "cpu",
                "TRANSCRIBER_WHISPER_MODEL": "small",
                "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
                "TRANSCRIBER_COMPUTE_TYPE": "int8",
                "TRANSCRIBER_MAX_SEGMENT_MS": "12000",
                "TRANSCRIBER_PREVIEW_INTERVAL_MS": "3000",
                "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "3000",
            },
            rationale=f"{base_reason} {ram_gb:.0f} GB RAM → small Whisper is the right tradeoff.",
        )
    return Tier(
        name="CPU-LOW (base int8 + NLLB-600M, looser segments)",
        settings={
            "TRANSCRIBER_COMPUTE_DEVICE": "cpu",
            "TRANSCRIBER_WHISPER_MODEL": "base",
            "TRANSCRIBER_TRANSLATION_BACKEND": "nllb-600m",
            "TRANSCRIBER_COMPUTE_TYPE": "int8",
            "TRANSCRIBER_MAX_SEGMENT_MS": "14000",
            "TRANSCRIBER_PREVIEW_INTERVAL_MS": "4000",
            "TRANSCRIBER_PREVIEW_MIN_AUDIO_MS": "4000",
        },
        rationale=f"{base_reason} Only {ram_gb:.1f} GB RAM — using the lightest viable models.",
    )


def pick_tier(hw: Hardware) -> Tier:
    if hw.has_nvidia and hw.gpu_vram_gb >= 2:
        return _gpu_tier_from_vram(hw.gpu_vram_gb, hw.gpu_compute_capability)
    if hw.is_apple_silicon:
        return _apple_tier(hw.ram_gb)
    return _cpu_tier(hw.ram_gb)


# ---------------------------------------------------------------------------
# .env writer
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=(.*)$")


def _read_env(path: Path) -> tuple[list[str], dict[str, int]]:
    """Return (lines, {key: line_index}) for active (uncommented) assignments."""
    if not path.is_file():
        return [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    keys: dict[str, int] = {}
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        m = _KEY_RE.match(line)
        if m:
            keys[m.group(1)] = i
    return lines, keys


def write_env(tier: Tier, hw: Hardware, *, dry_run: bool, force: bool) -> tuple[list[str], list[str]]:
    """Write `tier.settings` to .env. Return (changed_keys, preserved_keys)."""
    # Seed from .env.example if .env is missing and example exists.
    if not ENV_FILE.is_file() and ENV_EXAMPLE.is_file() and not dry_run:
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    lines, keys = _read_env(ENV_FILE)

    changed: list[str] = []
    preserved: list[str] = []
    to_append: list[str] = []

    for k, v in tier.settings.items():
        if k in keys and not force:
            preserved.append(k)
            continue
        if k in keys:
            lines[keys[k]] = f"{k}={v}"
            changed.append(k)
        else:
            to_append.append(f"{k}={v}")
            changed.append(k)

    if to_append:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"# --- autoconfig: {tier.name} ---")
        lines.extend(to_append)

    if not dry_run and (changed or to_append):
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return changed, preserved


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def _print_report(hw: Hardware, tier: Tier,
                  changed: list[str], preserved: list[str], dry_run: bool) -> None:
    print()
    print(f"{BOLD}Realtime Transcriber — autoconfig{RESET}")
    print()
    print(f"  {BOLD}Host{RESET}    : {hw.system} / {hw.machine}  ({hw.cpu_cores} cores, {hw.ram_gb:.1f} GB RAM)")
    if hw.has_nvidia:
        cc = f" cc={hw.gpu_compute_capability}" if hw.gpu_compute_capability else ""
        print(f"  {BOLD}GPU{RESET}     : {GREEN}{hw.gpu_name}{RESET}  ({hw.gpu_vram_gb:.1f} GB VRAM{cc})")
    elif hw.is_apple_silicon:
        print(f"  {BOLD}GPU{RESET}     : {GREEN}Apple Silicon (MPS){RESET}")
    else:
        print(f"  {BOLD}GPU{RESET}     : {YELLOW}none detected — CPU mode{RESET}")
    for n in hw.notes:
        print(f"           {DIM}{n}{RESET}")

    print()
    print(f"  {BOLD}Tier{RESET}    : {CYAN}{tier.name}{RESET}")
    print(f"  {BOLD}Why{RESET}     : {tier.rationale}")
    print()
    print(f"  {BOLD}Settings{RESET}:")
    for k, v in tier.settings.items():
        flag = ""
        if k in changed:
            flag = f"  {GREEN}[written]{RESET}"
        elif k in preserved:
            flag = f"  {YELLOW}[kept existing]{RESET}"
        print(f"    {k:<40} = {v}{flag}")

    if preserved and not dry_run:
        print()
        print(f"  {DIM}Existing values in .env were preserved. Pass --force to overwrite them.{RESET}")
    if dry_run:
        print()
        print(f"  {YELLOW}--dry-run: no changes written to .env{RESET}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-configure .env for this host.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, do not write.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing values in .env (default: preserve them).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the human report.")
    args = ap.parse_args()

    hw = detect_hardware()
    tier = pick_tier(hw)
    changed, preserved = write_env(tier, hw, dry_run=args.dry_run, force=args.force)

    if args.json:
        print(json.dumps({
            "hardware": asdict(hw),
            "tier": tier.name,
            "rationale": tier.rationale,
            "settings": tier.settings,
            "changed": changed,
            "preserved": preserved,
            "dry_run": args.dry_run,
        }, indent=2))
    else:
        _print_report(hw, tier, changed, preserved, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
