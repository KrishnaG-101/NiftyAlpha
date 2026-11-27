@echo off
REM NiftyAlpha Streamlit Launcher for Windows CMD / Double-Click
setlocal
cd /d "%~dp0"

set "STREAMLIT_CMD="
if exist "%~dp0venv\Scripts\streamlit.exe" (
    set "STREAMLIT_CMD=%~dp0venv\Scripts\streamlit.exe"
) else if exist "%~dp0.venv\Scripts\streamlit.exe" (
    set "STREAMLIT_CMD=%~dp0.venv\Scripts\streamlit.exe"
) else (
    set "STREAMLIT_CMD=streamlit"
)

set "APP_FILE=%~dp0dashboard\app.py"

echo ==========================================================
echo  Launching NiftyAlpha Streamlit Dashboard...
echo  Local URL: http://localhost:8501
echo ==========================================================

%STREAMLIT_CMD% run "%APP_FILE%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [INFO] Direct streamlit command failed, attempting via python -m streamlit...
    python -m streamlit run "%APP_FILE%"
)
pause
