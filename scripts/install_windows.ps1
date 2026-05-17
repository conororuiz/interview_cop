# Install script for Windows + NVIDIA GPU.
# Run from PowerShell in the project root:
#   .\scripts\install_windows.ps1
#
# Requires Python 3.12+ and an NVIDIA GPU with recent drivers (CUDA 12.x runtime
# is bundled inside the torch wheel; you do NOT need to install CUDA toolkit
# separately for PyTorch). faster-whisper / CTranslate2 ship their own
# cuBLAS / cuDNN binaries on Windows since CT2 4.4+.

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment .venv" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "==> Activating venv" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip / wheel / setuptools" -ForegroundColor Cyan
python -m pip install --upgrade pip wheel setuptools

Write-Host "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)" -ForegroundColor Cyan
# cu128 is the minimum CUDA toolkit version that supports Blackwell GPUs
# (RTX 50-series, sm_120). Older indexes like cu124/cu121 will fall back to
# CPU on Blackwell and PyTorch won't see the GPU.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio

Write-Host "==> Installing project requirements" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Installing project in editable mode" -ForegroundColor Cyan
pip install -e .

Write-Host "==> Done. Activate the venv next time with:" -ForegroundColor Green
Write-Host "    . .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "Quick checks:" -ForegroundColor Yellow
Write-Host "  python -c `"import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')`""
Write-Host "  transcriber-capture-test --seconds 30 --output capture_test.wav"
