# Repaired Universe Pre-Execution Pathology Audit

This is a repaired 1h-only preflight diagnostic. It intentionally excludes any 3m execution overlay, local repair search, or new optimization.

## Inputs Used
- Frozen repaired universe dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207`
- Frozen universe table: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_best_by_symbol.csv`
- Frozen selected params dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_universe_selected_params`
- Repaired fee model: `/root/analysis/0.87/reports/execution_layer/MULTICOIN_MODELA_AUDIT_20260228_180250/fee_model.json`
- Reused code paths: `scripts/backtest_exec_phasec_sol.py`, `scripts/repaired_frontier_contamination_audit.py`, `scripts/instant_loser_vs_winner_entry_forensics.py`

## Bucket Definitions
- `instant_loser`: valid repaired 1h trade, stop-loss exit, hold <= 60 minutes.
- `fast_loser`: valid repaired 1h trade, stop-loss exit, 60 < hold <= 240 minutes.
- `meaningful_winner`: valid repaired 1h trade with net PnL >= 0.0020.

## Per-Symbol Summary
| priority_rank | symbol | trade_count | instant_loser_rate_pct | fast_loser_rate_pct | meaningful_winner_rate_pct | classification | priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NEARUSDT | 1839 | 73.9532 | 11.963 | 11.5824 | HIGH_INSTANT_LOSER_BURDEN | 0.627162 |
| 2 | OGUSDT | 976 | 75.2049 | 9.83607 | 11.7828 | HIGH_INSTANT_LOSER_BURDEN | 0.623924 |
| 3 | DOGEUSDT | 1594 | 72.9611 | 11.1041 | 12.9235 | HIGH_INSTANT_LOSER_BURDEN | 0.623526 |
| 4 | SOLUSDT | 4773 | 72.3444 | 11.8165 | 12.445 | HIGH_INSTANT_LOSER_BURDEN | 0.622282 |
| 5 | LTCUSDT | 940 | 69.7872 | 14.5745 | 12.0213 | MIXED | 0.621702 |
| 6 | AXSUSDT | 819 | 74.6032 | 10.8669 | 10.5006 | HIGH_INSTANT_LOSER_BURDEN | 0.621368 |
| 7 | ZECUSDT | 1605 | 69.4081 | 13.4579 | 13.5826 | MIXED | 0.620436 |
| 8 | LINKUSDT | 1039 | 67.3725 | 15.5919 | 13.1858 | HIGH_FAST_LOSER_BURDEN | 0.61949 |
| 9 | CRVUSDT | 1540 | 72.987 | 13.1169 | 9.02597 | MIXED | 0.619058 |
| 10 | BCHUSDT | 1966 | 69.3795 | 13.3266 | 13.2757 | MIXED | 0.618642 |
| 11 | XRPUSDT | 2030 | 74.0887 | 12.1675 | 8.32512 | MIXED | 0.617291 |
| 12 | AVAXUSDT | 1645 | 70.9422 | 10.0304 | 13.8602 | HIGH_INSTANT_LOSER_BURDEN | 0.612888 |
| 13 | TRXUSDT | 1156 | 67.0415 | 13.3218 | 14.1869 | MIXED | 0.611289 |
| 14 | ADAUSDT | 2470 | 69.9595 | 12.0243 | 9.47368 | MIXED | 0.602085 |
| 15 | BTCUSDT | 905 | 61.989 | 15.8011 | 8.83978 | MIXED | 0.580994 |

## Observed Asymmetry
- `NEARUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.014058, winner first2 MAE median=0.006495, loser first2 close median=-0.003280, winner first2 close median=0.024588).
- `OGUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.016008, winner first2 MAE median=0.006058, loser first2 close median=-0.005108, winner first2 close median=0.027580).
- `DOGEUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.011813, winner first2 MAE median=0.004955, loser first2 close median=-0.002645, winner first2 close median=0.024862).
- `SOLUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.012323, winner first2 MAE median=0.005013, loser first2 close median=-0.003154, winner first2 close median=0.022154).
- `LTCUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.009386, winner first2 MAE median=0.003797, loser first2 close median=-0.001693, winner first2 close median=0.016523).
- `AXSUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.013968, winner first2 MAE median=0.004996, loser first2 close median=-0.003267, winner first2 close median=0.021983).
- `ZECUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.011155, winner first2 MAE median=0.005600, loser first2 close median=-0.002547, winner first2 close median=0.022740).
- `LINKUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.010081, winner first2 MAE median=0.004831, loser first2 close median=-0.001595, winner first2 close median=0.022361).
- `CRVUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.013661, winner first2 MAE median=0.007289, loser first2 close median=-0.003655, winner first2 close median=0.028708).
- `BCHUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.009035, winner first2 MAE median=0.003440, loser first2 close median=-0.002094, winner first2 close median=0.016076).
- `XRPUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.009810, winner first2 MAE median=0.004509, loser first2 close median=-0.002616, winner first2 close median=0.017931).
- `AVAXUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.012150, winner first2 MAE median=0.006473, loser first2 close median=-0.003932, winner first2 close median=0.024180).
- `TRXUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.007117, winner first2 MAE median=0.002317, loser first2 close median=-0.002157, winner first2 close median=0.011199).
- `ADAUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.010760, winner first2 MAE median=0.004888, loser first2 close median=-0.002818, winner first2 close median=0.028095).
- `BTCUSDT`: winners absorb less early adverse excursion and close stronger by the second eligible bar (loser first2 MAE median=-0.005240, winner first2 MAE median=0.001434, loser first2 close median=-0.001099, winner first2 close median=0.010795).

## Proven vs Assumed
- Proven: all symbols were evaluated only from the frozen repaired universe candidates using the repaired 1h chronology-valid simulator with `defer_exit_to_next_bar=True`.
- Proven: counts and rates come from the actual repaired 1h trade path, not from legacy or 3m-derived artifacts.
- Assumed: downstream 3m execution work is most valuable where loser burden is material but meaningful winners still exist and early-path asymmetry is visible.

## Recommendation
- Final recommendation: `RUN_3M_ON_PRIORITY_SUBSET_FIRST`
- Priority symbols for the next downstream 3m scope: `NEARUSDT, OGUSDT, DOGEUSDT, SOLUSDT, LTCUSDT, AXSUSDT, ZECUSDT`