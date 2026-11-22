@echo off
REM NiftyAlpha Streamlit Launcher for Windows CMD / Double-Click
setlocal
cd /d "%~dp0"

set "STREAMLIT_EXE=%~dp0venv\Scripts\streamlit.exe"
set "APP_FILE=%~dp0dashboard\app.py"

if not exist "%STREAMLIT_EXE%" (
    echo [ERROR] Streamlit executable not found at "%STREAMLIT_EXE%".
    echo Please ensure the virtual environment is set up.
    pause
    exit /b 1
)

echo ==========================================================
echo  Launching NiftyAlpha Streamlit Dashboard...
echo  Local URL: http://localhost:8501
echo ==========================================================

"%STREAMLIT_EXE%" run "%APP_FILE%"
pause
