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
# Three branches:
#   1. NVIDIA found    -> cu128 wheels (full GPU acceleration)
#   2. AMD/Radeon only -> CPU wheels by default. PyTorch ROCm exists on Linux
#                        but faster-whisper / CTranslate2 has no ROCm backend,
#                        so Whisper would land on CPU anyway. We default to a
#                        pure-CPU install to keep things bulletproof. Advanced
#                        users on RDNA2+ can re-install torch with the ROCm
#                        wheels later to accelerate NLLB only.
#   3. Nothing useful  -> CPU wheels.
# ---------------------------------------------------------------------------
HAS_NVIDIA=0
NVIDIA_INFO=""
if command -v nvidia-smi >/dev/null 2>&1; then
    if NVIDIA_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) && [ -n "$NVIDIA_INFO" ]; then
        HAS_NVIDIA=1
    fi
fi

HAS_AMD=0
AMD_INFO=""
if command -v lspci >/dev/null 2>&1; then
    # Vendor 1002 = AMD/ATI. Match VGA/Display/3D controllers only.
    AMD_LINE=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i '\[1002:' | head -n1 || true)
    if [ -n "$AMD_LINE" ]; then
        HAS_AMD=1
        # Extract human-readable name between the colon and the [1002:...] tag.
        AMD_INFO=$(echo "$AMD_LINE" | sed -E 's/^[^:]+:\s*//; s/\s*\[1002:[^]]+\].*//')
        [ -z "$AMD_INFO" ] && AMD_INFO="$AMD_LINE"
    fi
fi

if [ "$HAS_NVIDIA" = "1" ]; then
    echo "==> NVIDIA GPU detected: $NVIDIA_INFO"
    echo "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)"
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
elif [ "$HAS_AMD" = "1" ]; then
    echo "==> AMD GPU detected: $AMD_INFO"
    echo "    faster-whisper has no ROCm backend, so Whisper runs on CPU regardless."
    echo "    Installing PyTorch CPU wheels for a bulletproof setup."
    echo "    (Advanced users on RDNA2+ can later reinstall torch with the ROCm 6.x"
    echo "     wheels to accelerate NLLB only:"
    echo "       pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchaudio)"
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
else
    echo "==> No discrete GPU detected — installing PyTorch CPU wheels."
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
