#!/usr/bin/env bash
# Install script for Linux + NVIDIA GPU.
# Usage:
#   ./scripts/install_linux.sh
#
# Requires Python 3.12+, recent NVIDIA driver, and pulseaudio-utils
# (provides `pactl`). On Debian/Ubuntu:
#   sudo apt install python3.12 python3.12-venv pulseaudio-utils

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment .venv"
python3.12 -m venv .venv

echo "==> Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)"
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Installing project in editable mode"
pip install -e .

echo "==> Done. Activate the venv next time with:"
echo "    source .venv/bin/activate"
