# Paper Instant-Exit Diagnosis

- Monitor dir: `/root/analysis/0.87/reports/execution_layer/PAPER_SOL_FORWARD_MONITOR_20260306_205431`
- Paper files loaded: `2`
- Sample window (UTC dates from filenames): `20260306` to `20260307`
- Closed trades in sample: `0`

## Decision
`METRICS_DON'T_MATCH_CLAIM`

## Evidence
- win_rate: `n/a`
- profit_factor: `n/a`
- expectancy_per_trade_eur: `n/a`
- avg_win_eur / avg_loss_eur / win_loss_ratio: `n/a` / `n/a` / `n/a`
- hold_minutes mean / median: `n/a` / `n/a`
- instant_exit_rate A (hold <= 3m): `n/a`
- instant_exit_rate B (hold <= 60m): `n/a`
- exit_reason breakdown: `{}`
- loss concentration worst10 / worst25 / bottom-decile-share: `n/a` / `n/a` / `n/a`

## Sanity Cross-Checks
- same_bar exits total: `0`
- exit-before-entry total: `0`
- entry-on-signal violations total: `0`
- chronology guard deferrals (same_bar_exit_attempts / exits_deferred_to_next_bar / exits_blocked_pre_entry): `0` / `0` / `0`

## Interpretation
- No closed paper trades in sampled files; cannot observe 80% losers or instant exits.
- With zero closed trades in the sampled paper-forward window, the data does not support a claim of "~80% losers" or "instant exits" in this monitor run.

## Recommended Next Lever
- Measurement/sample lever: keep PAPER_ALIGNED monitor running until at least 30-50 closed trades before attributing failure mode; retain chronology guard checks in daily truthpack.

_Generated UTC: `2026-03-07T00:13:25.723204+00:00`_
