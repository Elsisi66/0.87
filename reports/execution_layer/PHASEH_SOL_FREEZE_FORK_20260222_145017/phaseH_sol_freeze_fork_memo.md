# Phase H SOL Freeze and Fork Memo

- Generated UTC: 2026-02-22T14:50:18.088716+00:00
- Phase G source (frozen): `/root/analysis/0.87/reports/execution_layer/PHASEG_SOL_PATHOLOGY_REHAB_20260222_143826`
- Symbol: SOLUSDT

## Freeze Confirmation

- Representative subset/hash and contract hashes are unchanged from Phase G trusted setup.
- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- setup_checks: {"contract_id_match": 1, "fee_hash_match": 1, "metrics_hash_match": 1, "split_integrity": 1, "subset_hash_match": 1, "symbol_match": 1}

## Phase G Coverage

- variants_tested_total: 12
- variants_with_relative_improvement_pass: 4
- variants_with_absolute_practical_pass: 0
- variants_with_final_gate_pass: 0
- variants_triggering_fatal_gate: 10

Variants tested:
- baseline_signal
- plus_regime_gate
- plus_regime_gate_plus_cooldown4h
- plus_regime_gate_plus_cooldown4h_plus_delay0
- plus_regime_gate_plus_cooldown4h_plus_delay1
- plus_regime_gate_plus_cooldown4h_plus_delay2
- plus_uc_params
- plus_uc_params_plus_regime_mod
- plus_uc_params_plus_regime_mod_plus_regime_gate
- plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h
- plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1
- plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2

Top relative uplifts observed (still non-deployable):
| variant | delta_expectancy_net_vs_baseline | delta_total_return_vs_baseline | delta_max_drawdown_pct_vs_baseline | relative_improvement_pass | absolute_practical_pass |
| --- | --- | --- | --- | --- | --- |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1 | 0.001754 | 0.057809 | 0.059109 | 1 | 0 |
| plus_uc_params_plus_regime_mod_plus_regime_gate | 0.001183 | 0.003682 | 0.003765 | 0 | 0 |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2 | 0.001026 | 0.038483 | 0.036337 | 1 | 0 |

## Terminal Decision

- Absolute practical gates failed universally despite relative uplift in some variants.
- Branch status: **HOLD current optimization branch** (terminal for tp/sl polishing in this line).
- Forward action: **FORK_SIGNAL_LAYER** under new signal-definition scope only.
- Deployment status: **NO_DEPLOY** until absolute practical gates pass.

## Next Exact Prompt (Fork)

```text
SOL signal-definition FORK (contract-locked Phase H): create a new SOL-only signal layer branch that keeps the same representative subset/hash and downstream 3m execution contract unchanged. Implement only: (1) trend alignment gate, (2) volatility regime gate, (3) de-clustering cooldown, (4) delayed 1h entry modes {0,1,2}. Evaluate every candidate in fixed-size/capped-risk mode first; enforce absolute gates before compounding (expectancy_net>0, total_return>0, maxDD>-0.35, cvar_5>-0.0015, PF>=1.05, support_ok=1). Reintroduce compounding only for fixed-size passers. Include multiple-testing accounting, DSR/PSR significance, and a reality-check benchmark in shortlist decisions. If no fixed-size candidate passes absolute gates, return HOLD/NO_DEPLOY with root-cause evidence and stop.
```
