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
# Three branches:
#   1. NVIDIA found    -> cu128 wheels (full GPU acceleration)
#   2. AMD/Radeon only -> CPU wheels (PyTorch ROCm has no Windows wheels and
#                        faster-whisper/CTranslate2 has no ROCm backend
#                        anywhere — so the only safe option is CPU torch).
#   3. Nothing useful  -> CPU wheels.
# ---------------------------------------------------------------------------
$hasNvidia = $false
$nvidiaInfo = ""
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $nvidiaInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaInfo) {
            $hasNvidia = $true
        }
    } catch {
        $hasNvidia = $false
    }
}

# AMD / Radeon detection via WMI (no separate tool needed on Windows).
$hasAmd = $false
$amdInfo = ""
try {
    $videoCtrls = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                  Select-Object -ExpandProperty Name
    foreach ($n in $videoCtrls) {
        if ($n -match "AMD|Radeon|ATI ") {
            $hasAmd = $true
            $amdInfo = $n
            break
        }
    }
} catch {
    $hasAmd = $false
}

if ($hasNvidia) {
    Write-Host "==> NVIDIA GPU detected: $nvidiaInfo" -ForegroundColor Green
    Write-Host "==> Installing PyTorch with CUDA 12.8 wheels (Blackwell-compatible)" -ForegroundColor Cyan
    # cu128 is the minimum CUDA toolkit version that supports Blackwell GPUs
    # (RTX 50-series, sm_120). Older indexes like cu124/cu121 will fall back to
    # CPU on Blackwell and PyTorch won't see the GPU.
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
} elseif ($hasAmd) {
    Write-Host "==> AMD GPU detected: $amdInfo" -ForegroundColor Yellow
    Write-Host "    PyTorch ROCm has no Windows wheels and faster-whisper has no" -ForegroundColor Yellow
    Write-Host "    ROCm backend — installing CPU torch instead (the app will work," -ForegroundColor Yellow
    Write-Host "    just on CPU). Expect 5-10x slower transcription vs NVIDIA." -ForegroundColor Yellow
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
} else {
    Write-Host "==> No discrete GPU detected — installing PyTorch CPU wheels." -ForegroundColor Yellow
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
