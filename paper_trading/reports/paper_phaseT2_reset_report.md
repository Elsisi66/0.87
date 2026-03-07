# Phase T2 Reset Report

- Reset command executed:
  - `PYTHONPATH=/root/analysis/0.87 /root/analysis/0.87/.venv/bin/python /root/analysis/0.87/paper_trading/scripts/reset_paper_state.py --hard-reset-now --start-capital 320`
- Latest reset result:
  - `reset_mode=local_hard_reset`
  - `reset_flag=local_hard_reset_applied`
  - `start_equity_eur=320.0`
  - `start_from_bar_ts=2025-12-31T23:00:00+00:00`
- State archive created:
  - `/root/analysis/0.87/paper_trading/state/archive/hard_reset_20260228_010346`
- Reset report:
  - `/root/analysis/0.87/paper_trading/reports/startup_reset_report_20260228_010346.md`
- Reset script now supports:
  - `--hard-reset-now`
  - `--start-capital 320`
