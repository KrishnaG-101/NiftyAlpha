#!/usr/bin/env bash
# ==============================================================================
# NiftyAlpha - Daily Cron Job Wrapper Script (Linux / macOS)
# ==============================================================================
# Triggers automated inference at Indian market close (15:45 IST / 10:15 UTC).
# Generates tomorrow's trade orders in reports/daily_orders_<YYYY-MM-DD>.csv.
#
# Crontab configuration (crontab -e):
# 45 15 * * 1-5 /bin/bash /path/to/NiftyAlpha/scripts/cron_daily_predict.sh >> /path/to/NiftyAlpha/logs/cron_daily_predict.log 2>&1
# ==============================================================================

set -euo pipefail

# Resolve project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"

cd "$PROJECT_ROOT"

# Ensure logs directory exists
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/cron_daily_predict.log"

echo "======================================================================" >> "$LOG_FILE"
echo " [$(date -u +"%Y-%m-%d %H:%M:%S UTC")] Starting NiftyAlpha Cron Daily Predictor" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"

# Activate Python virtual environment if present
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Execute daily prediction script
python "$PROJECT_ROOT/scripts/daily_predict.py" \
    --processed-dir "$PROJECT_ROOT/data/processed" \
    --raw-dir "$PROJECT_ROOT/data/raw" \
    --models-dir "$PROJECT_ROOT/models" \
    --reports-dir "$PROJECT_ROOT/reports" \
    --threshold 0.50 >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] SUCCESS: Daily orders successfully persisted." >> "$LOG_FILE"
else
    echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] ERROR: Daily prediction failed with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

exit $EXIT_CODE
