#!/usr/bin/env bash
# Install script for Linux (NVIDIA GPU optional).
# Usage:
#   ./scripts/install_linux.sh
#
# Requires Python 3.12+ and pulseaudio-utils (provides `pactl`).
# On Debian/Ubuntu:
#   sudo apt install python3.12 python3.12-venv pulseaudio-utils
#
# The script auto-detects whether an NVIDIA GPU is present and installs the
# matching PyTorch wheel:
#   * NVIDIA found      -> CUDA 12.8 wheels (Blackwell-compatible)
#   * NVIDIA NOT found  -> CPU wheels (no CUDA toolkit needed)
#
# Then it runs `scripts/autoconfig.py` which detects the hardware tier
# (GPU VRAM, system RAM) and writes a safe .env so the app starts cleanly
# regardless of the machine spec.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment .venv"
python3.12 -m venv .venv

echo "==> Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# GPU detection (BEFORE installing torch — picks the right wheel)
# ---------------------------------------------------------------------------
HAS_NVIDIA=0
GPU_INFO=""
if command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) && [ -n "$GPU_INFO" ]; then
        HAS_NVIDIA=1
    fi
fi

if [ "$HAS_NVIDIA" = "1" ]; then
    echo "==> NVIDIA GPU detected: $GPU_INFO"
    echo "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)"
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
else
    echo "==> No NVIDIA GPU detected (nvidia-smi missing or returned nothing)."
    echo "==> Installing PyTorch CPU wheels — no CUDA toolkit needed."
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
fi

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Installing project in editable mode"
pip install -e .

# ---------------------------------------------------------------------------
# Auto-configure .env for THIS machine
# ---------------------------------------------------------------------------
echo
echo "==> Auto-detecting hardware and writing .env"
if ! python scripts/autoconfig.py; then
    echo "(!) autoconfig failed but install completed; you can run it later:"
    echo "    python scripts/autoconfig.py"
fi

echo
echo "==> Done. Activate the venv next time with:"
echo "    source .venv/bin/activate"
echo
echo "Quick checks:"
echo "  python scripts/doctor.py"
echo "  transcriber-gui"
