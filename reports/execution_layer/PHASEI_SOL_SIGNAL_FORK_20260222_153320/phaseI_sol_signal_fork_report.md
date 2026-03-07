# Phase I SOL Signal-Definition Fork Report

- Generated UTC: 2026-02-22T15:33:23.216071+00:00
- Symbol: SOLUSDT
- Source Phase H dir: `/root/analysis/0.87/reports/execution_layer/PHASEH_SOL_FREEZE_FORK_20260222_145017`
- Source Phase G dir: `/root/analysis/0.87/reports/execution_layer/PHASEG_SOL_PATHOLOGY_REHAB_20260222_143826`

## Frozen Setup Confirmation

- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- setup_checks: {"contract_id_match": 1, "fee_hash_match": 1, "metrics_hash_match": 1, "selected_model_set_hash_match": 1, "split_integrity": 1, "subset_hash_match": 1, "symbol_match": 1}

## Fork Components Implemented

- trend alignment gate (slow trend threshold filter for long entries)
- volatility regime gate (supported bucket filtering)
- de-clustering cooldown windows
- delayed 1h entry modes {0,1,2}

## Fixed-Size Stage (Mandatory First)

Metric units:
- trade-level notional metrics: decimal per-trade returns on position notional (`expectancy_net_trade_notional_dec`, `cvar_5_trade_notional_dec`, `profit_factor_trade`)
- equity-path metrics: decimal account equity outcomes (`total_return_fixed_equity_dec`, `max_drawdown_pct_fixed_equity_dec`, `geometric_equity_step_return_fixed_clean`)

- candidates_total: 3
- fixed_size_passers: 0
- baseline_adverse_loss_share: 0.572414
- best_variant_adverse_loss_share: 0.273713
- baseline_dominant_worst_regime_bucket: `unknown|down`
- best_variant: `fork_trend0.50_all_supported_cd4h_d1`
- best_variant_dominant_worst_regime_bucket: `high|up`

| variant | signals_total | trades_total | expectancy_net_trade_notional_dec | cvar_5_trade_notional_dec | profit_factor_trade | geometric_equity_step_return_fixed | geometric_equity_step_return_fixed_clean | ruin_event_fixed | total_return_fixed_equity_dec | max_drawdown_pct_fixed_equity_dec | support_ok | fixed_absolute_practical_pass | fixed_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fork_trend0.50_all_supported_cd4h_d1 | 384 | 384 | -0.000675 | -0.002200 | 0.680458 | -1.000000 | nan | 1 | -2.593473 | -3.041135 | 1 | 0 | nan |
| fork_trend0.60_all_supported_cd4h_d1 | 384 | 384 | -0.000675 | -0.002200 | 0.680458 | -1.000000 | nan | 1 | -2.593473 | -3.041135 | 1 | 0 | nan |
| baseline_contract_locked | 1200 | 1200 | -0.001264 | -0.002200 | 0.405539 | -1.000000 | nan | 1 | -15.443836 | -15.768676 | 1 | 0 | nan |

Ranking policy used (fixed-size survivors only):
1. primary: `geometric_equity_step_return_fixed_clean` (with `ruin_event_fixed`=0 required for meaningful ranking)
2. secondary: `max_drawdown_pct_fixed_equity_dec`, `cvar_5_trade_notional_dec`, support/stability
3. tertiary: `expectancy_net_trade_notional_dec`, `profit_factor_trade`

## Compounding Follow-up (Conditional)

- fixed-size survivors promoted: 0
- compounded passers: 0
_(empty)_

## Multiple-Testing and Significance Controls

- raw_trials: 2
- effective_trials_estimate: 1.000000
- shortlist_rows: 3
- shortlist_significance_file: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_153320/phaseI_sol_shortlist_significance.csv`
- multiple_testing_summary_file: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_153320/phaseI_sol_multiple_testing_summary.md`

## Final Decision

- final_decision: **NO_DEPLOY**
- deployment_status: **NO_DEPLOY**
- root_cause_classification: regime concentration / path-quality issue, signal quality issue

Stop condition:
- If no fixed-size candidate passes absolute gates, optimization stops in this branch with HOLD/NO_DEPLOY.
