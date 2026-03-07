# LINK Retention Cliff Forensics

- Generated UTC: `2026-03-03T01:49:41.345511+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/LINK_RETENTION_CLIFF_FORENSICS_20260303_014856`

## Discovered Inputs

- Repaired 1h baseline dir: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Rebased Model A dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Winner-vs-loser diagnosis dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Latest entry-repair pilot dir: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409`
- LINK control trade CSV: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409/_trade_sources/LINKUSDT_M3_ENTRY_ONLY_FASTER_control.csv`
- LINK failed hard-gap trade CSV: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409/_trade_sources/LINKUSDT_GAP_CAP_FILTER.csv`

## Baseline Vs Failed Hard-Gap

- Repaired 1h baseline expectancy for LINKUSDT: `0.0028439573`
- Current repaired Model A control expectancy for LINKUSDT: `0.0021565911`
- Failed hard-gap expectancy for LINKUSDT: `0.0030021591`
- Control trades: `290`
- Failed hard-gap trades: `158`
- Failed hard-gap retention: `0.544828`

## Hard-Gap Removal Decomposition

- Removed trades: `132`
- Removed instant losers: `131`
- Removed fast losers: `1`
- Removed meaningful winners: `0`
- Removed positive-PnL trades: `0`
- Removed non-positive-PnL trades: `132`

## Gap Decile Decomposition

| action_gap_decile | trades | removed_by_hard_gap | removed_share | instant_losers | fast_losers | meaningful_winners | median_gap_pct | mean_control_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 29 | 0 | 0 | 3 | 12 | 7 | -0.0176707 | 0.0147454 |
| 2 | 29 | 0 | 0 | 7 | 8 | 5 | -0.00849118 | 0.00290296 |
| 3 | 29 | 0 | 0 | 7 | 11 | 5 | -0.00548986 | 0.00562262 |
| 4 | 29 | 0 | 0 | 18 | 7 | 2 | -0.00311333 | 0.000769086 |
| 5 | 29 | 0 | 0 | 18 | 6 | 4 | -0.00136488 | 0.00427295 |
| 6 | 29 | 16 | 0.551724 | 25 | 2 | 1 | 7.01066e-05 | 0.000675931 |
| 7 | 29 | 29 | 1 | 29 | 0 | 0 | 0.00249532 | -0.0018618 |
| 8 | 29 | 29 | 1 | 29 | 0 | 0 | 0.00476072 | -0.00184974 |
| 9 | 29 | 29 | 1 | 29 | 0 | 0 | 0.00822368 | -0.0018618 |
| 10 | 29 | 29 | 1 | 29 | 0 | 0 | 0.0155834 | -0.00184974 |

## Bounded Soft Variant Scan

| variant_id | variant_kind | instant_loser_count | fast_loser_count | expectancy_net | cvar_5 | max_drawdown | trade_count_retention_vs_control | valid_for_ranking | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONTROL | control | 194 | 46 | 0.00215659 | -0.00215288 | -0.181624 | 1 | 1 | 0 |
| GAP_CAP_0BPS | gap_cap | 63 | 45 | 0.00300216 | -0.00208293 | -0.0887874 | 0.544828 | 0 | 0 |
| GAP_CAP_2BPS | gap_cap | 65 | 45 | 0.0029894 | -0.00208293 | -0.0924869 | 0.551724 | 0 | 0 |
| GAP_CAP_4BPS | gap_cap | 67 | 45 | 0.00297665 | -0.00208293 | -0.0961864 | 0.558621 | 0 | 0 |
| PCTL_DROP_TOP10_POSGAP | percentile_drop | 180 | 46 | 0.00224589 | -0.00215288 | -0.179774 | 0.951724 | 1 | 1 |
| PCTL_DROP_TOP25_POSGAP | percentile_drop | 161 | 46 | 0.00236708 | -0.00215288 | -0.166826 | 0.886207 | 1 | 1 |
| SIZE_50_TOP10_POSGAP | size_haircut | 194 | 46 | 0.00220124 | -0.00215288 | -0.180699 | 1 | 1 | 0 |
| SIZE_75_ALL_POSGAP | size_haircut | 194 | 46 | 0.00236798 | -0.00208293 | -0.158415 | 1 | 1 | 0 |

## Decision

- Best surviving variant: `PCTL_DROP_TOP10_POSGAP`
- Decision reason: Accepted under the existing pilot acceptance logic.