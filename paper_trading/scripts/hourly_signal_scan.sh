#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/analysis/0.87"
cd "$ROOT"

# Load optional runtime secrets/config from local env file.
if [[ -f "$ROOT/paper_trading/config/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/paper_trading/config/.env"
  set +a
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PAPER_MODE=true
export START_EQUITY_EUR="${START_EQUITY_EUR:-320}"
export BINANCE_MODE="${BINANCE_MODE:-marketdata_only}"
POSTURE_FREEZE_DIR="${REPAIRED_POSTURE_FREEZE_DIR:-$ROOT/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126}"
export REPAIRED_POSTURE_FREEZE_DIR="$POSTURE_FREEZE_DIR"
export REPAIRED_ACTIVE_SUBSET_CSV="${REPAIRED_ACTIVE_SUBSET_CSV:-$POSTURE_FREEZE_DIR/repaired_active_3m_subset.csv}"
export REPAIRED_ACTIVE_PARAMS_DIR="${REPAIRED_ACTIVE_PARAMS_DIR:-$POSTURE_FREEZE_DIR/repaired_active_3m_params}"
export REQUIRE_REPAIRED_POSTURE_PACK="${REQUIRE_REPAIRED_POSTURE_PACK:-true}"
export PAPER_SYMBOL_ALLOWLIST="${PAPER_SYMBOL_ALLOWLIST:-SOLUSDT}"
export REQUIRED_ACTIVE_STRATEGY_ID="${REQUIRED_ACTIVE_STRATEGY_ID:-M1_ENTRY_ONLY_PASSIVE_BASELINE}"
export REPAIRED_CONTRACT_DEFER_EXIT_TO_NEXT_BAR="${REPAIRED_CONTRACT_DEFER_EXIT_TO_NEXT_BAR:-true}"

LOG_DIR="$ROOT/paper_trading/logs"
mkdir -p "$LOG_DIR"

LOCK_FILE="/tmp/paper_trading_hourly_scan.lock"

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] hourly_scan_start"
  flock -n 200 || {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] hourly_scan_skip lock_held"
    exit 0
  }

  "$ROOT/.venv/bin/python" -m paper_trading.app.main \
    --once \
    --max-cycles 1 \
    --no-startup-reset

  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] hourly_scan_done"
} 200>"$LOCK_FILE" >>"$LOG_DIR/hourly_scan.log" 2>>"$LOG_DIR/errors.log"
