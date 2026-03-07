# Multi-Coin Model A Audit

- Generated UTC: 2026-02-28T18:15:04.414140+00:00
- Foundation source: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`
- Universe size: `17`
- Variants tested per coin: `5`
- Contract parity:
  - 1h signal owner: `src/bot087/optim/ga.py` via frozen universal timeline
  - 1h exit owner: `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` semantics via `phase_a_model_a_audit.simulate_frozen_1h_exit`
  - 3m entry executor: `phase_a_model_a_audit.simulate_entry_only_fill`
  - forbidden hybrid exit knobs blocked: `['tp_mult', 'sl_mult', 'time_stop_min', 'break_even_enabled', 'break_even_trigger_r', 'break_even_offset_bps', 'trailing_enabled', 'trail_start_r', 'trail_step_bps', 'partial_take_enabled', 'partial_take_r', 'partial_take_pct']`

## Per-Coin Classification

| symbol | classification | best_candidate_id | best_delta_expectancy_vs_1h_reference | best_cvar_improve_ratio | best_maxdd_improve_ratio | best_valid_for_ranking | foundation_integrity_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADAUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003751429359 | -0.2259599361 | 0.7852224302 | 0 | PARTIAL |
| AVAXUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003634460567 | 0.03058207172 | 0.7172237455 | 0 | PARTIAL |
| AXSUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.004588079813 | 0.05300892431 | 0.6776604214 | 0 | PARTIAL |
| BCHUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002343547835 | 8.169806546e-14 | 0.8667042103 | 0 | PARTIAL |
| BNBUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002682901686 | 0.04770803188 | 0.7500607863 | 0 | PARTIAL |
| BTCUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001062492385 | 3.332468022e-14 | 0.6421479485 | 1 | PARTIAL |
| CRVUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0057763594 | 0.05963503985 | 0.8174203749 | 0 | PARTIAL |
| DOGEUSDT | MODEL_A_WEAK_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.006348098904 | 0.09087244167 | 0.7563528554 | 1 | PARTIAL |
| LINKUSDT | MODEL_A_STRONG_GO | M3_ENTRY_ONLY_FASTER | 0.00303401847 | 0.02120356972 | 0.4411160329 | 1 | PARTIAL |
| LTCUSDT | MODEL_A_WEAK_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.002062230775 | 0.07421249403 | 0.7039325006 | 1 | PARTIAL |
| NEARUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.003848069509 | 0.04543622084 | 0.8333147414 | 1 | PARTIAL |
| OGUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003667407225 | 0.09541606376 | 0.5606253067 | 1 | PARTIAL |
| PAXGUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0002447860214 | 3.785677021e-14 | 0.215541315 | 0 | PARTIAL |
| SOLUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002537399098 | 0.054403896 | 0.8546979974 | 0 | PARTIAL |
| TRXUSDT | MODEL_A_NO_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001144275681 | 0.06897092599 | 0.4728068052 | 1 | PARTIAL |
| XRPUSDT | MODEL_A_WEAK_GO | M3_ENTRY_ONLY_FASTER | 0.003770878421 | 0.05129895901 | 0.5715503693 | 1 | PARTIAL |
| ZECUSDT | MODEL_A_STRONG_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.002674727917 | 7.453050625e-14 | 0.2223602392 | 1 | PARTIAL |

## Top Improving Coins

| symbol | classification | best_candidate_id | delta_expectancy_vs_1h_reference | cvar_improve_ratio | maxdd_improve_ratio | best_valid_for_ranking |
| --- | --- | --- | --- | --- | --- | --- |
| DOGEUSDT | MODEL_A_WEAK_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.006348098904 | 0.09087244167 | 0.7563528554 | 1 |
| CRVUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0057763594 | 0.05963503985 | 0.8174203749 | 0 |
| AXSUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.004588079813 | 0.05300892431 | 0.6776604214 | 0 |
| NEARUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.003848069509 | 0.04543622084 | 0.8333147414 | 1 |
| XRPUSDT | MODEL_A_WEAK_GO | M3_ENTRY_ONLY_FASTER | 0.003770878421 | 0.05129895901 | 0.5715503693 | 1 |
| ADAUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003751429359 | -0.2259599361 | 0.7852224302 | 0 |
| OGUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003667407225 | 0.09541606376 | 0.5606253067 | 1 |
| AVAXUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003634460567 | 0.03058207172 | 0.7172237455 | 0 |
| LINKUSDT | MODEL_A_STRONG_GO | M3_ENTRY_ONLY_FASTER | 0.00303401847 | 0.02120356972 | 0.4411160329 | 1 |
| BNBUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002682901686 | 0.04770803188 | 0.7500607863 | 0 |

## Largest DD Improvements

| symbol | classification | best_candidate_id | maxdd_improve_ratio | delta_expectancy_vs_1h_reference | best_valid_for_ranking |
| --- | --- | --- | --- | --- | --- |
| BCHUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.8667042103 | 0.002343547835 | 0 |
| SOLUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.8546979974 | 0.002537399098 | 0 |
| NEARUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.8333147414 | 0.003848069509 | 1 |
| CRVUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.8174203749 | 0.0057763594 | 0 |
| ADAUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.7852224302 | 0.003751429359 | 0 |
| DOGEUSDT | MODEL_A_WEAK_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.7563528554 | 0.006348098904 | 1 |
| BNBUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.7500607863 | 0.002682901686 | 0 |
| AVAXUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.7172237455 | 0.003634460567 | 0 |
| LTCUSDT | MODEL_A_WEAK_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.7039325006 | 0.002062230775 | 1 |
| AXSUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.6776604214 | 0.004588079813 | 0 |

## Data-Blocked Coins

_(none)_

## Not Tradable After Execution

| symbol | classification | classification_reason | best_candidate_id | best_valid_for_ranking | best_delta_expectancy_vs_1h_reference |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.003751429359 |
| AVAXUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.003634460567 |
| AXSUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.004588079813 |
| BCHUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M3_ENTRY_ONLY_FASTER | 0 | 0.002343547835 |
| BNBUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M3_ENTRY_ONLY_FASTER | 0 | 0.002682901686 |
| CRVUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.0057763594 |
| OGUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 1 | 0.003667407225 |
| PAXGUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.0002447860214 |
| SOLUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M3_ENTRY_ONLY_FASTER | 0 | 0.002537399098 |
| TRXUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001144275681 |
