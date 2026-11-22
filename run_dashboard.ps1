# NiftyAlpha Streamlit Launcher for Windows PowerShell
$VenvStreamlit = Join-Path $PSScriptRoot "venv\Scripts\streamlit.exe"

if (-not (Test-Path $VenvStreamlit)) {
    Write-Error "Virtual environment streamlit executable not found at $VenvStreamlit. Please run 'python -m venv venv' and install requirements."
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Launching NiftyAlpha Streamlit Dashboard via venv..." -ForegroundColor Cyan
Write-Host " Dashboard URL: http://localhost:8501" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& $VenvStreamlit run (Join-Path $PSScriptRoot "dashboard\app.py")
