# ==============================================================================
# Screen Translator - OCR Helper PyInstaller Build Script
# Compiles server.py into a single standalone ocr_helper.exe binary.
# ==============================================================================
param(
    [switch]$IncludeModels = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== Building Standalone ocr_helper.exe ===" -ForegroundColor Cyan

# 1. Locate Python executable (check .venv first, then system python)
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "Using virtual environment Python: $PythonExe"
} else {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $SystemPython) {
        Write-Error "Python not found. Please install Python or run setup.ps1 first."
        exit 1
    }
    $PythonExe = "python"
    Write-Host "Using system Python: $((Get-Command python).Source)"
}

# 2. Check PyInstaller availability
& $PythonExe -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not found in environment. Installing PyInstaller..." -ForegroundColor Yellow
    & $PythonExe -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller."
        exit $LASTEXITCODE
    }
}

# 3. Check models directory
$ModelsDir = Join-Path $ScriptDir "models"
$AddDataArgs = @()
if ($IncludeModels) {
    if (Test-Path $ModelsDir) {
        $ModelFiles = Get-ChildItem -Path $ModelsDir -Filter "*.onnx"
        if ($ModelFiles.Count -gt 0) {
            Write-Host "Found $($ModelFiles.Count) model(s) in $ModelsDir. Bundling into binary..." -ForegroundColor Green
            $AddDataArgs = @("--add-data", "models;models")
        } else {
            Write-Warning "Models directory exists but contains no .onnx files. Run download_models.py first if models are needed inside .exe."
        }
    } else {
        Write-Warning "Models directory not found at $ModelsDir. Run download_models.py first if models are needed inside .exe."
    }
}

# 4. Target output directory
$DistDir = Join-Path (Split-Path -Parent $ScriptDir) "dist"
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
}

Write-Host "Running PyInstaller (onefile mode)..." -ForegroundColor Cyan
& $PythonExe -m PyInstaller `
    --noconfirm `
    --onefile `
    --clean `
    --name "ocr_helper" `
    --collect-all "rapidocr" `
    --collect-binaries "onnxruntime" `
    --hidden-import "shapely" `
    --hidden-import "pyclipper" `
    --distpath $DistDir `
    @AddDataArgs `
    server.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

$OutputExe = Join-Path $DistDir "ocr_helper.exe"
if (Test-Path $OutputExe) {
    $SizeMb = [math]::Round((Get-Item $OutputExe).Length / 1MB, 2)
    Write-Host "`n[SUCCESS] Built standalone binary: $OutputExe ($SizeMb MB)" -ForegroundColor Green
    Write-Host "Screen Translator will automatically discover this binary." -ForegroundColor Cyan
} else {
    Write-Error "Expected output binary not found at $OutputExe"
    exit 1
}
