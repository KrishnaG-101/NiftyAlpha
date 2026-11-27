@echo off
rem ==============================================================================
rem NiftyAlpha - Daily Scheduled Task Wrapper Script (Windows Task Scheduler)
rem ==============================================================================
rem Triggers automated inference at Indian market close (15:45 IST).
rem Generates tomorrow's trade orders in reports\daily_orders_<YYYY-MM-DD>.csv.
rem
rem Scheduled Task setup (PowerShell):
rem $Action = New-ScheduledTaskAction -Execute "%CD%\scripts\cron_daily_predict.bat"
rem $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 3:45PM
rem Register-ScheduledTask -Action $Action -Trigger $Trigger -TaskName "NiftyAlpha_Daily_Predict"
rem ==============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"
set "LOG_FILE=%PROJECT_ROOT%\logs\cron_daily_predict.log"

echo ====================================================================== >> "%LOG_FILE%"
echo [%DATE% %TIME%] Starting NiftyAlpha Scheduled Daily Predictor >> "%LOG_FILE%"
echo ====================================================================== >> "%LOG_FILE%"

rem Activate virtual environment if present
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
) else if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
)

python "%PROJECT_ROOT%\scripts\daily_predict.py" --processed-dir "%PROJECT_ROOT%\data\processed" --raw-dir "%PROJECT_ROOT%\data\raw" --models-dir "%PROJECT_ROOT%\models" --reports-dir "%PROJECT_ROOT%\reports" --threshold 0.50 >> "%LOG_FILE%" 2>&1

set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 (
    echo [%DATE% %TIME%] SUCCESS: Daily orders successfully persisted. >> "%LOG_FILE%"
) else (
    echo [%DATE% %TIME%] ERROR: Daily prediction failed with exit code %EXIT_CODE% >> "%LOG_FILE%"
)

exit /b %EXIT_CODE%
