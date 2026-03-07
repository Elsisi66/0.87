# Phase G SOL Pathology-First Rehab

- Generated UTC: 2026-02-22T14:38:43.325566+00:00
- Symbol: SOLUSDT
- Final verdict: **HOLD**

## 1) Frozen setup confirmation (hashes + contract lock checks)

- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- setup_checks: {"contract_id_match": 1, "fee_hash_match": 1, "metrics_hash_match": 1, "split_integrity": 1, "subset_hash_match": 1, "symbol_match": 1}

## 2) G0 drawdown forensics summary (root-cause ranking)

- baseline_variant: V4R_EXEC_3M_PHASEC_BEST
- comp_total_return/maxDD: -1.000000 / -1.000000
- fixed_total_return/maxDD: -14.047683 / -14.395908
- adverse_loss_share: 0.297872
- sl_loss_share: 0.999135
- amplification_ratio: 0.071186

| rank | root_cause | evidence |
| --- | --- | --- |
| 1 | signal quality issue | expectancy_net=-0.000868, sl_loss_share=0.9991, win_rate=0.0367 |
| 2 | regime concentration issue | adverse_loss_share=0.5683, dominant_worst_regime=sl |
| 3 | sizing/compounding amplification issue | comp_total_return=-1.000000, fixed_total_return=-14.047683, amplification_ratio=0.0712 |

## 3) G1 parameterization design (global prior + SOL offsets + constraints)

- prior_symbols: AVAXUSDT, BCHUSDT, CRVUSDT, NEARUSDT, SOLUSDT
- selected_offset_scale: 0.250
- uc_tp_vector: [1.035128, 1.118766, 1.076191, 1.065451, 1.024473]
- uc_sl_vector: [0.964951, 0.999, 0.965836, 0.97533, 0.934108]
- tp_offset_cap/sl_offset_cap: 0.080000 / 0.050000
- constraints(min_trades/min_split/min_bucket): 120/40/30
- fail_fast_streak_used: 3

## 4) G2 ablation results table

| variant | signals_total | trades_total | expectancy_net | total_return | max_drawdown_pct | cvar_5 | profit_factor | fatal_gate | absolute_practical_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1 | 365 | 365 | 0.000727 | -0.942191 | -0.940891 | -0.007198 | 1.241123 | 0 | 0 |
| plus_uc_params_plus_regime_mod_plus_regime_gate | 657 | 657 | 0.000155 | -0.996317 | -0.996235 | -0.007198 | 1.050695 | 1 | 0 |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2 | 365 | 365 | -0.000001 | -0.961517 | -0.963663 | -0.007198 | 0.999669 | 1 | 0 |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h | 365 | 365 | -0.000288 | -0.944847 | -0.947816 | -0.007198 | 0.907252 | 0 | 0 |
| plus_uc_params_plus_regime_mod | 1200 | 1200 | -0.000423 | -0.999996 | -0.999996 | -0.007342 | 0.847979 | 1 | 0 |
| plus_regime_gate_plus_cooldown4h_plus_delay1 | 365 | 365 | -0.000639 | -0.967600 | -0.966872 | -0.002200 | 0.697966 | 1 | 0 |
| baseline_signal | 1200 | 1200 | -0.001027 | -1.000000 | -1.000000 | -0.002200 | 0.514693 | 1 | 0 |
| plus_regime_gate | 657 | 657 | -0.001044 | -0.999723 | -0.999717 | -0.002200 | 0.507514 | 1 | 0 |
| plus_uc_params | 1200 | 1200 | -0.001044 | -1.000000 | -1.000000 | -0.002200 | 0.506670 | 1 | 0 |
| plus_regime_gate_plus_cooldown4h | 365 | 365 | -0.001059 | -0.988690 | -0.988435 | -0.002200 | 0.497849 | 1 | 0 |
| plus_regime_gate_plus_cooldown4h_plus_delay0 | 365 | 365 | -0.001059 | -0.988690 | -0.988435 | -0.002200 | 0.497849 | 1 | 0 |
| plus_regime_gate_plus_cooldown4h_plus_delay2 | 365 | 365 | -0.001149 | -0.992918 | -0.992759 | -0.002200 | 0.464217 | 1 | 0 |

## 5) G3 practical gate decisions

| variant | delta_expectancy_net_vs_baseline | delta_total_return_vs_baseline | delta_max_drawdown_pct_vs_baseline | relative_improvement_pass | absolute_practical_pass | final_gate_pass | reason_classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1 | 0.001754 | 0.057809 | 0.059109 | 1 | 0 | 0 | interaction issue |
| plus_uc_params_plus_regime_mod_plus_regime_gate | 0.001183 | 0.003682 | 0.003765 | 0 | 0 | 0 | interaction issue |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2 | 0.001026 | 0.038483 | 0.036337 | 1 | 0 | 0 | signal quality issue |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h | 0.000739 | 0.055153 | 0.052184 | 1 | 0 | 0 | signal quality issue |
| plus_uc_params_plus_regime_mod | 0.000605 | 0.000003 | 0.000004 | 0 | 0 | 0 | signal quality issue |
| plus_regime_gate_plus_cooldown4h_plus_delay1 | 0.000389 | 0.032399 | 0.033128 | 1 | 0 | 0 | signal quality issue |
| baseline_signal | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | signal quality issue |
| plus_regime_gate | -0.000016 | 0.000276 | 0.000283 | 0 | 0 | 0 | signal quality issue |
| plus_uc_params | -0.000017 | -0.000000 | -0.000000 | 0 | 0 | 0 | signal quality issue |
| plus_regime_gate_plus_cooldown4h | -0.000032 | 0.011310 | 0.011565 | 0 | 0 | 0 | signal quality issue |
| plus_regime_gate_plus_cooldown4h_plus_delay0 | -0.000032 | 0.011310 | 0.011565 | 0 | 0 | 0 | signal quality issue |
| plus_regime_gate_plus_cooldown4h_plus_delay2 | -0.000122 | 0.007081 | 0.007241 | 0 | 0 | 0 | signal quality issue |

## 6) Final recommendation + next exact prompt

- Final recommendation: **HOLD**

Next exact prompt:

```text
Phase G follow-up (SOL, contract-locked): keep the same representative subset/hash and rerun only signal-layer changes. Implement regime gate + 4h cooldown + delayed 1h entry modes with bounded universe-conditioned tp/sl offsets; reject any candidate that fails absolute practical gates (expectancy>0, total_return>0, maxDD>-0.35, cvar5>-0.0015). If all fail absolute gates again, freeze optimization and prepare FORK/HOLD memo with root-cause evidence from Phase G0 forensics.
```
