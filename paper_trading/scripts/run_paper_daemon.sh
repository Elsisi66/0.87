#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/analysis/0.87"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PAPER_MODE=true
export BINANCE_MODE="${BINANCE_MODE:-marketdata_only}"
export START_EQUITY_EUR="${START_EQUITY_EUR:-320}"
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

if [[ -f "$ROOT/paper_trading/state/daemon.pid" ]]; then
  OLD_PID="$(cat "$ROOT/paper_trading/state/daemon.pid" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Daemon already running with PID $OLD_PID"
    exit 0
  fi
fi

nohup "$ROOT/.venv/bin/python" -m paper_trading.app.main \
  >> "$LOG_DIR/service.log" \
  2>> "$LOG_DIR/errors.log" &

PID=$!
echo "$PID" > "$ROOT/paper_trading/state/daemon.pid"

echo "Paper daemon started (PID=$PID)"
echo "Logs: $LOG_DIR/service.log"
