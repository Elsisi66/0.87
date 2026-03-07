# Phase I SOL Signal-Definition Fork Report

- Generated UTC: 2026-02-22T15:13:15.686010+00:00
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
- equity-path metrics: decimal account equity outcomes (`total_return_fixed_equity_dec`, `max_drawdown_pct_fixed_equity_dec`, `geometric_equity_step_return_fixed`)

- candidates_total: 121
- fixed_size_passers: 0
- baseline_adverse_loss_share: 0.572414
- best_variant_adverse_loss_share: 1.000000
- baseline_dominant_worst_regime_bucket: `unknown|down`
- best_variant: `fork_trend0.50_high_only_cd6h_d2`
- best_variant_dominant_worst_regime_bucket: `high|up`

| variant | signals_total | trades_total | expectancy_net_trade_notional_dec | cvar_5_trade_notional_dec | profit_factor_trade | geometric_equity_step_return_fixed | total_return_fixed_equity_dec | max_drawdown_pct_fixed_equity_dec | support_ok | fixed_absolute_practical_pass | fixed_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fork_trend0.50_high_only_cd6h_d2 | 87 | 87 | -0.000173 | -0.002200 | 0.917762 | -0.001868 | -0.150134 | -0.721947 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd6h_d2 | 87 | 87 | -0.000173 | -0.002200 | 0.917762 | -0.001868 | -0.150134 | -0.721947 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd4h_d2 | 104 | 104 | -0.000467 | -0.002200 | 0.778972 | -0.006382 | -0.486156 | -0.960914 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd4h_d2 | 104 | 104 | -0.000467 | -0.002200 | 0.778972 | -0.006382 | -0.486156 | -0.960914 | 0 | 0 | nan |
| fork_trend0.50_mid_only_cd2h_d1 | 192 | 192 | -0.000131 | -0.002200 | 0.936939 | -1.000000 | -0.251056 | -1.245563 | 0 | 0 | nan |
| fork_trend0.60_mid_only_cd2h_d1 | 192 | 192 | -0.000131 | -0.002200 | 0.936939 | -1.000000 | -0.251056 | -1.245563 | 0 | 0 | nan |
| fork_trend0.50_mid_only_cd6h_d1 | 125 | 125 | -0.000485 | -0.002200 | 0.770091 | -1.000000 | -0.606829 | -1.469648 | 0 | 0 | nan |
| fork_trend0.60_mid_only_cd6h_d1 | 125 | 125 | -0.000485 | -0.002200 | 0.770091 | -1.000000 | -0.606829 | -1.469648 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd6h_d0 | 87 | 87 | -0.001835 | -0.002200 | 0.155850 | -1.000000 | -1.596783 | -1.513214 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd6h_d0 | 87 | 87 | -0.001835 | -0.002200 | 0.155850 | -1.000000 | -1.596783 | -1.513214 | 0 | 0 | nan |
| fork_trend0.50_mid_only_cd4h_d1 | 156 | 156 | -0.000685 | -0.002200 | 0.675982 | -1.000000 | -1.069025 | -1.672361 | 0 | 0 | nan |
| fork_trend0.60_mid_only_cd4h_d1 | 156 | 156 | -0.000685 | -0.002200 | 0.675982 | -1.000000 | -1.069025 | -1.672361 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd2h_d1 | 134 | 134 | -0.001279 | -0.002200 | 0.405421 | -1.000000 | -1.691208 | -1.729243 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd2h_d1 | 134 | 134 | -0.001279 | -0.002200 | 0.405421 | -1.000000 | -1.691208 | -1.729243 | 0 | 0 | nan |
| fork_trend0.50_mid_only_cd0h_d1 | 262 | 262 | -0.000540 | -0.002200 | 0.742906 | -1.000000 | -1.413706 | -1.782155 | 0 | 0 | nan |
| fork_trend0.60_mid_only_cd0h_d1 | 262 | 262 | -0.000540 | -0.002200 | 0.742906 | -1.000000 | -1.413706 | -1.782155 | 0 | 0 | nan |
| fork_trend0.50_low_mid_cd4h_d1 | 286 | 286 | -0.000064 | -0.002200 | 0.969212 | -1.000000 | -0.183516 | -1.823637 | 0 | 0 | nan |
| fork_trend0.60_low_mid_cd4h_d1 | 286 | 286 | -0.000064 | -0.002200 | 0.969212 | -1.000000 | -0.183516 | -1.823637 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd4h_d0 | 104 | 104 | -0.001895 | -0.002200 | 0.130128 | -1.000000 | -1.970701 | -1.850867 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd4h_d0 | 104 | 104 | -0.001895 | -0.002200 | 0.130128 | -1.000000 | -1.970701 | -1.850867 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd4h_d1 | 104 | 104 | -0.001776 | -0.002200 | 0.184876 | -1.000000 | -1.846667 | -1.865709 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd4h_d1 | 104 | 104 | -0.001776 | -0.002200 | 0.184876 | -1.000000 | -1.846667 | -1.865709 | 0 | 0 | nan |
| fork_trend0.50_low_mid_cd4h_d0 | 286 | 286 | -0.000605 | -0.002200 | 0.708528 | -1.000000 | -1.730969 | -1.882151 | 0 | 0 | nan |
| fork_trend0.60_low_mid_cd4h_d0 | 286 | 286 | -0.000605 | -0.002200 | 0.708528 | -1.000000 | -1.730969 | -1.882151 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd6h_d1 | 87 | 87 | -0.002200 | -0.002200 | 0.000000 | -1.000000 | -1.913582 | -1.934129 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd6h_d1 | 87 | 87 | -0.002200 | -0.002200 | 0.000000 | -1.000000 | -1.913582 | -1.934129 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd0h_d2 | 180 | 180 | -0.000825 | -0.002200 | 0.612053 | -1.000000 | -1.484739 | -1.955460 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd0h_d2 | 180 | 180 | -0.000825 | -0.002200 | 0.612053 | -1.000000 | -1.484739 | -1.955460 | 0 | 0 | nan |
| fork_trend0.50_high_only_cd2h_d2 | 134 | 134 | -0.001165 | -0.002200 | 0.458029 | -1.000000 | -1.561620 | -2.034071 | 0 | 0 | nan |
| fork_trend0.60_high_only_cd2h_d2 | 134 | 134 | -0.001165 | -0.002200 | 0.458029 | -1.000000 | -1.561620 | -2.034071 | 0 | 0 | nan |

Ranking policy used (fixed-size survivors only):
1. primary: `geometric_equity_step_return_fixed`
2. secondary: `max_drawdown_pct_fixed_equity_dec`, `cvar_5_trade_notional_dec`, support/stability
3. tertiary: `expectancy_net_trade_notional_dec`, `profit_factor_trade`

## Compounding Follow-up (Conditional)

- fixed-size survivors promoted: 0
- compounded passers: 0
_(empty)_

## Multiple-Testing and Significance Controls

- raw_trials: 120
- effective_trials_estimate: 100.947173
- shortlist_rows: 5
- shortlist_significance_file: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_151206/phaseI_sol_shortlist_significance.csv`
- multiple_testing_summary_file: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_151206/phaseI_sol_multiple_testing_summary.md`

## Final Decision

- final_decision: **NO_DEPLOY**
- deployment_status: **NO_DEPLOY**
- root_cause_classification: regime concentration / path-quality issue, signal quality issue, support/stability issue

Stop condition:
- If no fixed-size candidate passes absolute gates, optimization stops in this branch with HOLD/NO_DEPLOY.
