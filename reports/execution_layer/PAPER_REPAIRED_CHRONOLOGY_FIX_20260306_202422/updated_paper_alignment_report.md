# Updated Paper Alignment Report (Chronology + Fallback Fix)

## Scope
- Window replayed: 2026-03-04 to 2026-03-05 UTC (exact requested window)
- Symbol scope: SOLUSDT only (repaired posture active subset)
- Data source: Binance 1h klines fetched for 2025-12-01..2026-03-06 to support causal indicator warmup

## Chronology Rule Implemented
- `defer_exit_to_next_bar=True` guard in execution simulator.
- If a close condition appears on the entry bar (`hold_hours == 0`), exit is deferred and counted.
- Exits with invalid pre-entry ordering (`hold_hours < 0`) are blocked and counted.

## Replay Metrics (No Guard vs Guard)
- signals: 31 (same source for both)
- opens: no_guard=31 guard=11
- closes: no_guard=31 guard=11
- same_bar_close_rate: no_guard=1.000000 guard=0.000000
- close win rate: no_guard=0.000000 guard=0.090909
- all-negative close pathology: no_guard=True guard=False

## Guard Counters (Guarded Replay)
- same_bar_exit_attempts=11
- exits_deferred_to_next_bar=11
- exits_blocked_pre_entry=0

## Timing Alignment Checks (Guarded Replay)
- entry_before_exit_violations=0
- entry_on_signal_violations=11

## Before vs After (Previous Report vs Patched Replay)
- previous same-bar close rate: 1.000000 (67/67)
- patched same-bar close rate: 0.000000 (0/11)
- previous non-SOL rows: 32

## Startup Guard Confirmation
- Startup now logs: `REPAIRED_POSTURE_MODE: fallback disabled`
- Universe resolves from repaired posture active subset only; fallback to legacy best_by_symbol is disabled in repaired posture mode.

## Final Outcome
- PAPER_MISMATCH_REMAINS
