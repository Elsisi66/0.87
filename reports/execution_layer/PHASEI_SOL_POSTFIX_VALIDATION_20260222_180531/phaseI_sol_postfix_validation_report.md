# Phase I SOL Post-Fix Validation Report

- Generated UTC: 2026-02-22T18:05:44.744842+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_POSTFIX_VALIDATION_20260222_180531`
- Source Phase H dir: `/root/analysis/0.87/reports/execution_layer/PHASEH_SOL_FREEZE_FORK_20260222_145017`
- Source Phase G dir: `/root/analysis/0.87/reports/execution_layer/PHASEG_SOL_PATHOLOGY_REHAB_20260222_143826`

## Frozen Setup

- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- setup_checks: {"contract_id_match": 1, "fee_hash_match": 1, "metrics_hash_match": 1, "selected_model_set_hash_match": 1, "split_integrity": 1, "subset_hash_match": 1, "symbol_match": 1}

## Step 1: Knob Sensitivity (OAT)

- trend threshold effect detected: 0
- volatility bucket effect detected: 1
- cooldown effect detected: 1
- delay effect detected: 1
- overall sensitivity exists: 1
- OAT rows: 17

| variant | stage | signals_total | trades_total | expectancy_net_trade_notional_dec | cvar_5_trade_notional_dec | geometric_equity_step_return_fixed_clean | total_return_fixed_equity_dec | max_drawdown_pct_fixed_equity_dec | support_ok | fixed_absolute_practical_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oat_trend_t0.30 | oat_trend | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_trend_t0.50 | oat_trend | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_trend_t0.60 | oat_trend | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_trend_t0.75 | oat_trend | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_vol_high_low_mid | oat_vol | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_cooldown_cd4h | oat_cooldown | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_delay_d1 | oat_delay | 384 | 384 | -0.000675 | -0.002200 | nan | -2.593473 | -3.041135 | 1 | 0 |
| oat_delay_d0 | oat_delay | 384 | 384 | -0.000930 | -0.002200 | nan | -3.569699 | -3.140373 | 1 | 0 |
| oat_vol_high_mid | oat_vol | 254 | 254 | -0.001370 | -0.002200 | nan | -3.478981 | -3.534733 | 1 | 0 |
| oat_cooldown_cd6h | oat_cooldown | 313 | 313 | -0.001245 | -0.002200 | nan | -3.897375 | -4.024261 | 1 | 0 |
| oat_cooldown_cd2h | oat_cooldown | 509 | 509 | -0.000512 | -0.002200 | nan | -2.606522 | -4.236846 | 1 | 0 |
| oat_delay_d2 | oat_delay | 384 | 384 | -0.001201 | -0.002200 | nan | -4.613243 | -4.794486 | 1 | 0 |
| oat_cooldown_cd0h | oat_cooldown | 701 | 701 | -0.000705 | -0.002200 | nan | -4.941110 | -5.413226 | 1 | 0 |
| oat_vol_mid | oat_vol | 156 | 156 | -0.000685 | -0.002200 | nan | -1.069025 | -1.672361 | 0 | 0 |
| oat_vol_low_mid | oat_vol | 286 | 286 | -0.000064 | -0.002200 | nan | -0.183516 | -1.823637 | 0 | 0 |
| oat_vol_high | oat_vol | 104 | 104 | -0.001776 | -0.002200 | nan | -1.846667 | -1.865709 | 0 | 0 |

## Step 2: Duplicate Pruning

- tested_variants_total: 19
- duplicate_rows_detected: 8
- nonduplicate_variants_total: 11
- effective_trials_after_pruning: 6.926704
- avg_pairwise_step_return_corr_nondup: 0.341477

## Step 3: Reduced Cross Search

- reduced_cross_run: 1
- reduced_cross_rows: 2
Stop rule:
- no fixed-size absolute passers => stop search for this run.

## Step 4: Edge vs Cost Decomposition

- edge_vs_fee_verdict: **some_gross_edge_killed_by_costs**
| variant | is_baseline | expectancy_gross_trade_notional_dec | expectancy_net_trade_notional_dec | cost_drag_expectancy_dec | avg_total_cost_bps | trades_total | adverse_loss_share | dominant_worst_regime_bucket |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_contract_locked | 1 | -0.000064 | -0.001264 | 0.001200 | 11.998944 | 1200 | 0.572414 | unknown|down |
| cross_t0.30_vhigh_low_mid_cd4h_d1 | 0 | 0.000525 | -0.000675 | 0.001200 | 12.001299 | 384 | 0.273713 | mid|up |
| oat_delay_d0 | 0 | 0.000270 | -0.000930 | 0.001200 | 12.000282 | 384 | 0.275204 | mid|up |
| oat_vol_high_mid | 0 | -0.000170 | -0.001370 | 0.001200 | 11.998521 | 254 | 0.407258 | mid|up |
| oat_cooldown_cd6h | 0 | -0.000045 | -0.001245 | 0.001200 | 11.999019 | 313 | 0.272131 | mid|up |
| oat_cooldown_cd2h | 0 | 0.000688 | -0.000512 | 0.001200 | 12.001952 | 509 | 0.266940 | low|up |

## Best Non-Duplicate Candidates

| variant | stage | signals_total | trades_total | geometric_equity_step_return_fixed_clean | total_return_fixed_equity_dec | max_drawdown_pct_fixed_equity_dec | cvar_5_trade_notional_dec | fixed_absolute_practical_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_t0.30_vhigh_low_mid_cd4h_d1 | cross | 384 | 384 | nan | -2.593473 | -3.041135 | -0.002200 | 0 |
| oat_delay_d0 | oat_delay | 384 | 384 | nan | -3.569699 | -3.140373 | -0.002200 | 0 |
| oat_vol_high_mid | oat_vol | 254 | 254 | nan | -3.478981 | -3.534733 | -0.002200 | 0 |
| oat_cooldown_cd6h | oat_cooldown | 313 | 313 | nan | -3.897375 | -4.024261 | -0.002200 | 0 |
| oat_cooldown_cd2h | oat_cooldown | 509 | 509 | nan | -2.606522 | -4.236846 | -0.002200 | 0 |

## Decision

- fixed_size_passers_nonduplicate: 0
- recommendation: **pivot_to_execution_exit_optimization**
- deployment_status: **NO_DEPLOY**
