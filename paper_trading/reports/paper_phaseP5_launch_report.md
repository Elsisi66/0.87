# Phase P5 Launch Report

- Generated UTC: `2026-02-27T11:41:30+00:00`

## Launch Mode
- Switched to hourly cron execution mode (recommended for 1h strategy cadence).
- Cron runs one paper cycle per hour, with no startup reset.
- Script:
  - `/root/analysis/0.87/paper_trading/scripts/hourly_signal_scan.sh`

## Installed Cron Entry
- `3 * * * * /root/analysis/0.87/paper_trading/scripts/hourly_signal_scan.sh`
- Verified via `crontab -l`.

## Execution Semantics
- Command executed by cron uses:
  - `python -m paper_trading.app.main --once --max-cycles 1 --no-startup-reset`
- If a buy signal occurs, paper execution simulator opens position in local ledger.
- If exit conditions trigger, paper execution simulator closes position and records realized PnL.

## Safety Guards Active
- `PAPER_MODE=true` forced in script.
- Local paper ledger used; no live order placement.
- Production Binance URL is treated as unsafe for live execution and flagged degraded mode.
- Overlap protection with `flock` lock file:
  - `/tmp/paper_trading_hourly_scan.lock`

## Validation
- Manual test run of hourly script completed successfully.
- Hourly scan log:
  - `/root/analysis/0.87/paper_trading/logs/hourly_scan.log`

