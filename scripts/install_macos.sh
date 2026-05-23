#!/usr/bin/env bash
# Install script for macOS (Apple Silicon recommended).
#
# Requires Python 3.12+, Homebrew, and BlackHole for loopback capture.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh first." >&2
  exit 1
fi

if ! system_profiler SPAudioDataType 2>/dev/null | grep -qi blackhole; then
  echo "==> Installing BlackHole (virtual loopback device)"
  brew install blackhole-2ch
  echo
  echo "NOTE: open 'Audio MIDI Setup', create a Multi-Output Device"
  echo "      combining 'BlackHole 2ch' + your speakers, and set it as the"
  echo "      system output. Otherwise system audio won't be captured."
fi

echo "==> Creating virtual environment .venv"
python3.12 -m venv .venv

echo "==> Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip / wheel / setuptools"
pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch (MPS-enabled on Apple Silicon)"
pip install torch

echo "==> Installing project requirements"
pip install -r requirements.txt

echo "==> Installing project in editable mode"
pip install -e .

# ---------------------------------------------------------------------------
# Auto-configure .env for THIS machine (Apple Silicon -> MPS, else CPU tier)
# ---------------------------------------------------------------------------
echo
echo "==> Auto-detecting hardware and writing .env"
if ! python scripts/autoconfig.py; then
    echo "(!) autoconfig failed but install completed; you can run it later:"
    echo "    python scripts/autoconfig.py"
fi

echo
echo "==> Done. Activate the venv with: source .venv/bin/activate"
echo
echo "Quick checks:"
echo "  python scripts/doctor.py"
echo "  transcriber-gui"
