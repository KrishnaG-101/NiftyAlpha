# NiftyAlpha Streamlit Launcher for Windows PowerShell
$ScriptDir = $PSScriptRoot
$AppFile = Join-Path $ScriptDir "dashboard\app.py"

$VenvStreamlit = Join-Path $ScriptDir "venv\Scripts\streamlit.exe"
$DotVenvStreamlit = Join-Path $ScriptDir ".venv\Scripts\streamlit.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Launching NiftyAlpha Streamlit Dashboard..." -ForegroundColor Cyan
Write-Host " Dashboard URL: http://localhost:8501" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

if (Test-Path $VenvStreamlit) {
    & $VenvStreamlit run $AppFile
} elseif (Test-Path $DotVenvStreamlit) {
    & $DotVenvStreamlit run $AppFile
} else {
    try {
        streamlit run $AppFile
    } catch {
        python -m streamlit run $AppFile
    }
}
