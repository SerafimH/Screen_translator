# ==============================================================================
# Screen Translator - OCR Helper Setup Script
# ==============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Setting up Python virtual environment for OCR Helper..." -ForegroundColor Cyan

# 1. Check Python installation
$SystemPython = Get-Command python -ErrorAction SilentlyContinue
if (-not $SystemPython) {
    Write-Error "Python 3 is not installed or not in PATH. Please install Python 3.8-3.13."
    exit 1
}

# 2. Create .venv if not already present
$VenvDir = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir..."
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Virtual environment already exists in $VenvDir"
}

# 3. Locate venv python.exe
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python executable not found at $VenvPython"
    exit 1
}

# 4. Install dependencies
Write-Host "Installing dependencies from requirements.txt..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit $LASTEXITCODE
}

Write-Host "OCR Helper environment setup completed successfully!" -ForegroundColor Green
Write-Host "Now run download_models.py to download OCR models." -ForegroundColor Yellow
