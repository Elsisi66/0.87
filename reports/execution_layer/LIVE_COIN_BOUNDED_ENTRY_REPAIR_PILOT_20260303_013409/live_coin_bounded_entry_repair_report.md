# Live-Coin Bounded Entry Repair Pilot

- Generated UTC: `2026-03-03T01:35:45.008068+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409`
- Frozen diagnosis dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Repaired multicoin source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`

## Control / Acceptance Rules

- Control is the current repaired best candidate per live coin.
- Tolerances: expectancy >= control - 0.00005; cvar_5 >= control - 0.00010; maxDD >= control - 0.01.
- Trade retention floor: `0.85`.
- Instant-loser reduction must be at least max(5 trades, 5% of control instant losers).
- Parity must remain clean (no same-parent-bar exits, no invalid stop/TP geometry, no lookahead).

- Priority gate result: No repair on LINKUSDT/NEARUSDT survived the acceptance rules, so DOGEUSDT and LTCUSDT were not fully evaluated.

## Variant Results

| symbol | variant_id | instant_loser_count | fast_loser_count | meaningful_winner_count | expectancy_net | cvar_5 | max_drawdown | trade_count_retention_vs_control | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LINKUSDT | CONTROL | 194 | 46 | 24 | 0.00215659 | -0.00215288 | -0.181624 | 1 | 0 |
| LINKUSDT | GAP_CAP_FILTER | 63 | 45 | 24 | 0.00300216 | -0.00208293 | -0.0887874 | 0.544828 | 0 |
| LINKUSDT | GAP_CAP_PLUS_ROOM15 | 60 | 45 | 25 | 0.00280651 | -0.00258275 | -0.102537 | 0.544828 | 0 |
| LINKUSDT | UPPER_WICK_FILTER | 85 | 21 | 16 | 0.0018989 | -0.00187306 | -0.0869377 | 0.458621 | 0 |
| NEARUSDT | CONTROL | 297 | 48 | 43 | 0.00277844 | -0.00209958 | -0.0798885 | 1 | 0 |
| NEARUSDT | ROOM15_ON_NONCHASE | 285 | 56 | 44 | 0.0026318 | -0.00254945 | -0.0908857 | 1 | 0 |
| NEARUSDT | ROOM20_ON_NONCHASE_BREAKOUTSAFE | 289 | 53 | 46 | 0.00281087 | -0.00294938 | -0.0978838 | 1 | 0 |

## Best Vs Control

| symbol | decision | best_variant_id | before_instant_loser_count | after_instant_loser_count | before_fast_loser_count | after_fast_loser_count | before_expectancy_net | after_expectancy_net | trade_count_retention_vs_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LINKUSDT | NO_REPAIR_APPROVED | CONTROL | 194 | 194 | 46 | 46 | 0.00215659 | 0.00215659 | 1 |
| NEARUSDT | NO_REPAIR_APPROVED | CONTROL | 297 | 297 | 48 | 48 | 0.00277844 | 0.00277844 | 1 |

## Deployment Posture

- approved_paper: `LINKUSDT`
- shadow_only: `NEARUSDT, DOGEUSDT, LTCUSDT`
- disabled: `(none)`