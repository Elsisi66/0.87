# Phase T4 Smoke Report

- Smoke command executed in repaired mode:
  - `BINANCE_MODE=local_only PYTHONPATH=/root/analysis/0.87 /root/analysis/0.87/.venv/bin/python /root/analysis/0.87/paper_trading/scripts/smoke_test.py`
- Run 1 (reset + immediate cycle):
  - fills open/close: `0/0`
  - processed bars count: `0`
  - `start_from_bar_ts=2025-12-31T23:00:00+00:00`
  - `last_processed_bar_ts=null`
- Run 2 (restart without reset):
  - fills open/close: `0/0`
  - duplicate delta: `0/0`
  - `last_processed_bar_ts=null`
- Result:
  - no historical backlog replay after reset: `true`
  - processed-bar idempotency after restart: `true`
