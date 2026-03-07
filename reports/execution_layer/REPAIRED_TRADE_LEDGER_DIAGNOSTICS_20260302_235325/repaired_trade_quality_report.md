# Repaired Trade Quality Report

- Generated UTC: `2026-03-02T23:55:59.817554+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_TRADE_LEDGER_DIAGNOSTICS_20260302_235325`

## Source Files Used

- Canonical repaired 1h trade ledger: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650/repaired_1h_trades.csv`
- Canonical repaired 1h summary: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650/repaired_1h_reference_summary.csv`
- Repaired Model A priority reference-vs-best: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206/repaired_modelA_reference_vs_best_priority.csv`
- Repaired Model A priority results: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206/repaired_modelA_results_priority.csv`
- Repaired Model A priority manifest: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206/repaired_modelA_run_manifest.json`
- Universal data foundation signal timeline: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929/universe_signal_timeline.csv`
- Universal data foundation 3m manifest: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929/universe_3m_download_manifest.csv`
- Universal data foundation readiness: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929/universe_symbol_readiness.csv`
- Repaired multicoin Model A dir present: `1`
- Repaired multicoin Model A dir complete/usable: `1`
- Repaired multicoin Model A dir checked: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`

## Label Rules

- Labels are assigned from repaired 1h full-trade metrics only, so every coin uses the widest repaired trade sample available.
- `KEEP`: expectancy_per_trade >= repaired-universe 75th percentile and profit_factor >= repaired-universe 75th percentile.
- `WATCH`: expectancy_per_trade >= repaired-universe median and profit_factor >= repaired-universe median, but below KEEP.
- `TRASH`: expectancy_per_trade <= repaired-universe 25th percentile and profit_factor <= repaired-universe 25th percentile and edge_vs_max_drawdown <= repaired-universe 25th percentile.
- `WEAK`: positive data but not strong enough for WATCH and not weak enough for TRASH.
- `DATA_BLOCKED`: no repaired trade sample available.

## Numeric Thresholds

- expectancy_q1: `0.0014060191`
- expectancy_q2: `0.0028582919`
- expectancy_q3: `0.0035115449`
- profit_factor_q1: `1.7017980050`
- profit_factor_q2: `2.4255287378`
- profit_factor_q3: `2.7544379677`
- edge_vs_max_drawdown_q1: `0.0055352614`

## Per-Coin Judgment Snapshot

| symbol | wins | losses | win_rate_pct | loss_rate_pct | avg_loss | expectancy_per_trade | net_pnl | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADAUSDT | 337 | 2323 | 12.669173 | 87.330827 | 0.002428 | 0.002858 | 7.603057 | WEAK |
| AVAXUSDT | 245 | 1487 | 14.145497 | 85.854503 | 0.002200 | 0.003346 | 5.795090 | WATCH |
| AXSUSDT | 64 | 509 | 11.169284 | 88.830716 | 0.002200 | 0.005019 | 2.875984 | KEEP |
| BCHUSDT | 150 | 1581 | 8.665511 | 91.334489 | 0.002245 | 0.001448 | 2.506269 | WEAK |
| BNBUSDT | 55 | 562 | 8.914100 | 91.085900 | 0.002200 | 0.001406 | 0.867514 | TRASH |
| BTCUSDT | 103 | 802 | 11.381215 | 88.618785 | 0.002376 | 0.001049 | 0.949743 | WEAK |
| CRVUSDT | 129 | 1457 | 8.133670 | 91.866330 | 0.002200 | 0.003545 | 5.622449 | KEEP |
| DOGEUSDT | 145 | 1246 | 10.424155 | 89.575845 | 0.002200 | 0.005187 | 7.215596 | KEEP |
| LINKUSDT | 90 | 878 | 9.297521 | 90.702479 | 0.002200 | 0.002844 | 2.752951 | WEAK |
| LTCUSDT | 67 | 873 | 7.127660 | 92.872340 | 0.002200 | 0.000812 | 0.762923 | TRASH |
| NEARUSDT | 119 | 1261 | 8.623188 | 91.376812 | 0.002198 | 0.003512 | 4.845932 | WATCH |
| OGUSDT | 110 | 1212 | 8.320726 | 91.679274 | 0.002200 | 0.004525 | 5.982482 | KEEP |
| PAXGUSDT | 29 | 202 | 12.554113 | 87.445887 | 0.002166 | -0.001374 | -0.317503 | TRASH |
| SOLUSDT | 503 | 4531 | 9.992054 | 90.007946 | 0.002202 | 0.003198 | 16.096292 | WATCH |
| TRXUSDT | 102 | 874 | 10.450820 | 89.549180 | 0.002372 | 0.001225 | 1.195823 | TRASH |
| XRPUSDT | 153 | 1881 | 7.522124 | 92.477876 | 0.002200 | 0.001770 | 3.600889 | WEAK |
| ZECUSDT | 129 | 1245 | 9.388646 | 90.611354 | 0.002200 | 0.003276 | 4.501682 | WATCH |

## Priority Repaired 1h Vs Repaired Model A

| symbol | best_candidate_id | trade_count_diff | win_rate_diff_pct | avg_loss_diff | expectancy_diff | max_drawdown_diff | cvar_diff | behavior_call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | M2_ENTRY_ONLY_MORE_PASSIVE | -19 | -0.280593 | -0.000333 | 0.000377 | 0.000507 | 0.000067 | REAL_TRADE_BEHAVIOR_IMPROVED |
| NEARUSDT | M1_ENTRY_ONLY_PASSIVE_BASELINE | -4 | -0.140214 | -0.000335 | 0.000389 | 0.014691 | 0.000100 | REAL_TRADE_BEHAVIOR_IMPROVED |
| SOLUSDT | M3_ENTRY_ONLY_FASTER | -153 | 0.451278 | -0.000336 | 0.000304 | 0.044325 | 0.000093 | REAL_TRADE_BEHAVIOR_IMPROVED |

## Best Coins

| symbol | expectancy_per_trade | profit_factor | net_pnl | label |
| --- | --- | --- | --- | --- |
| DOGEUSDT | 0.005187 | 3.632851 | 7.215596 | KEEP |
| AXSUSDT | 0.005019 | 3.568862 | 2.875984 | KEEP |
| OGUSDT | 0.004525 | 3.244145 | 5.982482 | KEEP |
| CRVUSDT | 0.003545 | 2.754438 | 5.622449 | KEEP |
| NEARUSDT | 0.003512 | 2.748087 | 4.845932 | WATCH |

## Worst Coins

| symbol | expectancy_per_trade | profit_factor | net_pnl | label |
| --- | --- | --- | --- | --- |
| PAXGUSDT | -0.001374 | 0.274371 | -0.317503 | TRASH |
| LTCUSDT | 0.000812 | 1.397318 | 0.762923 | TRASH |
| BTCUSDT | 0.001049 | 1.498451 | 0.949743 | WEAK |
| TRXUSDT | 0.001225 | 1.576776 | 1.195823 | TRASH |
| BNBUSDT | 0.001406 | 1.701798 | 0.867514 | TRASH |

## Label Counts

- KEEP: `4`
- TRASH: `4`
- WATCH: `4`
- WEAK: `5`