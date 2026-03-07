# Phase P3 Notifier Report

- Generated UTC: `2026-02-27T11:29:30+00:00`

## Notification Policy
- Only daily summary messages are supported/sent.
- No trade-by-trade Telegram sends.
- Daily summary gate:
  - one message/day based on `DAILY_SUMMARY_HOUR_UTC`
  - prevents repeat sends via `last_summary_date`.

## Summary Artifact Evidence
- JSON summary generated:
  - `/root/analysis/0.87/paper_trading/reports/daily_summary_20260227.json`
- Markdown summary generated:
  - `/root/analysis/0.87/paper_trading/reports/daily_summary_20260227.md`

## Telegram Path Status
- Telegram credentials currently missing in environment.
- Send path is functional but safely degraded (no send attempt beyond credential guard).
- Summary event is still journaled locally.

