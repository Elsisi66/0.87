# Phase P3 Recovery Report

- Generated UTC: `2026-02-27T11:29:30+00:00`

## Recovery Controls Implemented
- Retry with exponential backoff + jitter:
  - `paper_trading/app/utils/retry.py`
- API circuit breaker:
  - threshold: `API_CIRCUIT_FAIL_THRESHOLD` (default 5)
  - cooldown: `API_CIRCUIT_COOLDOWN_SEC` (default 600)
- Symbol quarantine:
  - threshold: `SYMBOL_ERROR_QUARANTINE_THRESHOLD` (default 4)
  - quarantine period: `SYMBOL_QUARANTINE_MINUTES` (default 120)
- Dead-letter queue for unrecoverable events:
  - `paper_trading/state/dead_letter_queue.jsonl`
- Restart reconciliation:
  - reload state
  - optional replay (`--replay-bars`)
  - idempotent processing via `processed_bars.json`

## Runtime Evidence
- API retry/failure counters observed:
  - `api_retries=15`, `api_failures=5`
- Circuit breaker open events logged when DNS/API unavailable.
- Current health counters:
  - `/root/analysis/0.87/paper_trading/state/health_counters.json`
- Error logs:
  - `/root/analysis/0.87/paper_trading/logs/errors.log`

## Degraded Mode Behavior
- With production Binance URL and unavailable DNS/API, daemon continues in paper mode using local data and marks `degraded_mode=true`.
- No live orders are submitted.

