# Phase D4 Rollback Rules

## Infra Rollback
- Soft warning: any single runtime mismatch, stale data event, or transient signal gap.
- Hard stop: any freeze-lock failure, any hybrid exit mutation, any impossible fill path, or 2 consecutive runtime exceptions.
- Rollback action: stop both `paper_primary` and `shadow_backup`; do not fail over.

## Economic Caution / Stop
- Taker share
  - soft warning: `> 0.10` on any rolling 10 trades
  - hard stop: `> 0.25` on any rolling 10 trades
  - action: switch primary to backup only if infra is clean and backup remains within band; otherwise stop both
- Fill delay
  - soft warning: `p95_fill_delay_min > 6.0`
  - hard stop: `p95_fill_delay_min > 12.0`
  - action: switch primary to backup only if backup remains <= 6.0; otherwise stop both
- Realized expectancy vs frozen 1h reference
  - soft warning: rolling 5-trade delta <= 0.0
  - hard stop: rolling 10-trade delta < -0.0010
  - action: demote primary to shadow if backup remains positive; otherwise stop both
- Trade-handling mismatch
  - soft warning: 1 mismatch event in a day
  - hard stop: 2 mismatch events in a day
  - action: stop both immediately
- SL clustering
  - soft warning: rolling SL hit rate > 0.70 over 10 trades
  - hard stop: rolling SL hit rate > 0.85 over 10 trades
  - action: keep backup only if no purity violations; otherwise stop both
