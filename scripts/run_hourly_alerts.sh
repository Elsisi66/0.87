#!/usr/bin/env bash
set -euo pipefail
cd /root/analysis/0.87
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
python3 scripts/telegram_signal_alert_live.py --symbols BTCUSDT --tf 1h --only-cycles 3 --cycle-shift 1 >> logs/telegram_alerts.log 2>&1
