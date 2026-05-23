# Install script for Windows (NVIDIA GPU optional).
# Run from PowerShell in the project root:
#   .\scripts\install_windows.ps1
#
# Requires Python 3.12+. The script auto-detects whether an NVIDIA GPU is
# present and installs the matching PyTorch wheel:
#   * NVIDIA found      -> CUDA 12.8 wheels (Blackwell-compatible, ~3 GB)
#   * NVIDIA NOT found  -> CPU wheels (much smaller, no CUDA dependency)
#
# After dependencies are in place it runs `scripts/autoconfig.py` which
# detects the hardware (GPU VRAM, RAM) and writes a safe .env tier so the
# app starts cleanly regardless of the machine spec.

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment .venv" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "==> Activating venv" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip / wheel / setuptools" -ForegroundColor Cyan
python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# GPU detection (BEFORE installing torch — picks the right wheel)
# ---------------------------------------------------------------------------
$hasNvidia = $false
$gpuInfo = ""
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuInfo) {
            $hasNvidia = $true
        }
    } catch {
        $hasNvidia = $false
    }
}

if ($hasNvidia) {
    Write-Host "==> NVIDIA GPU detected: $gpuInfo" -ForegroundColor Green
    Write-Host "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)" -ForegroundColor Cyan
    # cu128 is the minimum CUDA toolkit version that supports Blackwell GPUs
    # (RTX 50-series, sm_120). Older indexes like cu124/cu121 will fall back to
    # CPU on Blackwell and PyTorch won't see the GPU.
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
} else {
    Write-Host "==> No NVIDIA GPU detected (nvidia-smi missing or returned nothing)." -ForegroundColor Yellow
    Write-Host "==> Installing PyTorch CPU wheels — no CUDA toolkit needed." -ForegroundColor Cyan
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
}

Write-Host "==> Installing project requirements" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Installing project in editable mode" -ForegroundColor Cyan
pip install -e .

# ---------------------------------------------------------------------------
# Auto-configure .env for THIS machine (VRAM tier, model sizes, segment ms)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==> Auto-detecting hardware and writing .env" -ForegroundColor Cyan
python scripts/autoconfig.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "(!) autoconfig failed but install completed; you can run it later:" -ForegroundColor Yellow
    Write-Host "    python scripts/autoconfig.py" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> Done. Activate the venv next time with:" -ForegroundColor Green
Write-Host "    . .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "Quick checks:" -ForegroundColor Yellow
Write-Host "  python -c `"import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')`""
Write-Host "  python scripts/doctor.py"
Write-Host "  transcriber-gui"
