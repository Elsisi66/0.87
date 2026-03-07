# Phase D3 Alert Rules

Immediate red alerts:
- Any `hybrid_exit_mutation_detected_count > 0`.
- Any missing or duplicated 1h signal relative to the frozen signal path.
- Any fill marked impossible under the configured Model A entry-only rules.
- Any repeated runtime exception on the same candidate for 2 consecutive cycles.

Economic caution alerts:
- `taker_share > 0.10` on either candidate intraday.
- `p95_fill_delay_min > 6.0` intraday on either candidate.
- `realized_pnl_pct_net < 0` on a rolling 5-trade window versus the 1h reference expectation.
- `primary_minus_backup_expectancy < -0.0010` on a rolling 10-trade window.
