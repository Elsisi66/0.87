# Forensic Root Cause Report

- Generated UTC: 2026-02-22T15:35:35.682071+00:00
- Full audited run: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_153158`
- Tiny validation run: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_SIGNAL_FORK_20260222_153320`

## A) Parameter Effect / No-Op Audit

- Threshold pair comparisons (0.50 vs 0.60): 60
- Pairs with zero stage-count deltas: 60
- All pair trade vectors identical: 1
- All pair gate decisions identical: 1
- Code path check confirms threshold is applied in gate expression (`trend_up_1h >= trend_min_threshold`) and not overwritten.
- Root cause: threshold knob is effectively discrete/no-op on this dataset, not ignored by code path.

## B) Trend Gate Sanity

- Trend score source is `trend_up_1h` from frozen signal source.
- Quantile summary implies binary support only: unique_count=2, min=0.000000, max=1.000000
- Band counts: <0.3=499, [0.3,0.5)=0, [0.5,0.6)=0, [0.6,0.8)=0, >=0.8=701
- Because [0.5,0.6) mass is zero, thresholds 0.50 and 0.60 are expected to be equivalent.

## C) CVaR Pathology Audit

- CVaR now traced from variant-local trade vectors with per-variant vector hashes and tail counts.
- Baseline + 3 variant trace:
| variant | trades_total | trade_vector_hash | trade_return_unique_values | cvar_5_trade | cvar_tail_count | cvar_tail_min | cvar_tail_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_contract_locked | 1200 | dfdda6e414d5ec17aed7a7e742b85405681d9e1c187049be5d0d4ab1ca30de0d | 41 | -0.002200 | 60 | -0.002200 | -0.002200 |
| fork_trend0.50_all_supported_cd4h_d1 | 384 | b84c5a66ede436c00ffab339230d06c794b315226ffccd31911300a9160f9096 | 16 | -0.002200 | 20 | -0.002200 | -0.002200 |
| fork_trend0.50_high_only_cd6h_d2 | 87 | f6e62fc34fc34b9026839844a66c9eb02d7f8e1d6c97a990ecd2887b3afa5dae | 5 | -0.002200 | 5 | -0.002200 | -0.002200 |
| fork_trend0.50_mid_only_cd2h_d1 | 192 | 5e78b309b825897362652765e110b189ff1f749616ecba971dc92828fe2a5cb4 | 12 | -0.002200 | 10 | -0.002200 | -0.002200 |
- Interpretation: near-constant CVaR comes from similar stop-dominated worst-tail returns, not constant fallback logic.

## D) Geometric Equity Saturation Audit

- Variants with legacy geometric exactly -1.0: 117
- Of those, ruin_event_fixed=1: 117
- Of those, ruin_event_fixed=0: 0
- Saturation is explained by ruin events (`equity_step_return <= -1`) rather than dtype/NaN clamp artifacts.
- Patch separates `ruin_event_fixed` from clean geometric metric for ranking/report clarity.

## Confirmed Root Causes

1. Confirmed: trend threshold no-op on frozen SOL subset due binary trend score (0/1), not due code path bypass.
2. Confirmed: CVaR similarity mainly tail-structure driven (stop-dominated vectors), not constant fallback bug.
3. Confirmed: geometric -1 saturation corresponds to ruin events; legacy metric conflated ruin with growth scalar.

## Suspected / Residual

- Support remains weak in many fork variants (min split trades often below gate).
- Signal quality remains negative under tested fork controls (pre-optimization).

## Artifacts

- `param_effect_diff.csv`: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532/param_effect_diff.csv`
- `trend_score_distribution.csv`: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532/trend_score_distribution.csv`
- `metric_unit_tests_output.txt`: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532/metric_unit_tests_output.txt`
- `patch_summary.md`: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532/patch_summary.md`
- `tiny_validation_results.csv`: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532/tiny_validation_results.csv`
