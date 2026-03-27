#!/usr/bin/env bash
# sports-analytics/scripts/run_daily.sh
# Daily MLB Insights pipeline runner for cron.
# Runs the pipeline, then triggers Zip with a leaderboard summary.
#
# Usage:
#   ./scripts/run_daily.sh              # run for today
#   ./scripts/run_daily.sh 2025-04-01   # run for specific date

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="/opt/sports-analytics-venv"
PYTHON="$VENV/bin/python"
LOG_DIR="/var/log/openclaw"
LOG_FILE="$LOG_DIR/mlb-insights.log"
ZIP_TRIGGER="/opt/zip-tools/sports/trigger_zip.sh"

DATE="${1:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== MLB Insights Pipeline Starting ==="

cd "$REPO_ROOT"

# Run the pipeline
if [[ -n "$DATE" ]]; then
    log "Running for date: $DATE"
    $PYTHON -m mlb_insights.run_daily --date "$DATE" 2>&1 | tee -a "$LOG_FILE"
else
    log "Running for today"
    $PYTHON -m mlb_insights.run_daily 2>&1 | tee -a "$LOG_FILE"
fi

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    log "Pipeline completed successfully."

    # Trigger Zip with insights summary
    if [[ -x "$ZIP_TRIGGER" ]]; then
        log "Sending insights to Zip..."
        bash "$ZIP_TRIGGER" --mlb-insights 2>&1 | tee -a "$LOG_FILE"
        log "Zip notification sent."
    else
        log "WARNING: trigger_zip.sh not found or not executable at $ZIP_TRIGGER"
    fi
else
    log "ERROR: Pipeline failed with exit code $EXIT_CODE"
    # Still try to alert via Zip
    if [[ -x "$ZIP_TRIGGER" ]]; then
        OPENCLAW_TRIGGER="$(dirname "$ZIP_TRIGGER")/../shared/openclaw_trigger.sh"
        if [[ -x "$OPENCLAW_TRIGGER" ]]; then
            bash "$OPENCLAW_TRIGGER" "MLB Insights pipeline FAILED at $(date). Check $LOG_FILE"
        fi
    fi
    exit $EXIT_CODE
fi

log "=== MLB Insights Pipeline Complete ==="
