# Baseline Mismatch Audit

- Generated UTC: 2026-02-21T21:24:15.650231+00:00
- Tight dir: `/root/analysis/0.87/reports/execution_layer/20260220_123314`
- GA dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402`

## Cause Summary

- Signal universe mismatch: tight overall uses 1312 test signals; GA run uses 60 test signals.
- GA run uses symbol/time subset defined by GA walkforward splits (from ga_config signal CSV + split indices).
- When tight baseline is re-measured on the exact GA test timestamps, baseline metrics reconcile within tolerance.
- Fee/slippage parameters are consistent between compared pipelines.

## Raw Overall (Different Universes)

- tight_baseline_expectancy_per_signal: -0.000565
- ga_baseline_expectancy_per_signal: -0.001990
- tight_signals_total: 1312
- ga_signals_total: 60

## Aligned Universe Reconciliation

- aligned_signals_total: 60
- aligned_tight_expectancy_per_signal: -0.001990
- ga_reported_baseline_expectancy_per_signal: -0.001990
- aligned_tight_cvar5: -0.002200
- ga_reported_baseline_cvar5: -0.002200
- aligned_tight_maxdd: -0.117204
- ga_reported_baseline_maxdd: -0.117204

## Per-Symbol Aligned Details

- SOLUSDT: signals 60, expectancy tight/ga -0.001990/-0.001990, cvar5 tight/ga -0.002200/-0.002200, maxdd tight/ga -0.117204/-0.117204

## Fee Model

- fee_model_exact_match: 1
- tight source: walkforward_exec_limit parser defaults (assumed; run metadata missing explicit fee snapshot)
- ga source: /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/ga_config.yaml
- values (maker/taker/limit_slip/market_slip): 2.0/4.0/0.5/2.0 vs 2.0/4.0/0.5/2.0

## Pass Criteria

- abs delta expectancy <= 1.0e-04: 1
- abs delta cvar5 <= 1.0e-04: 1
- abs delta maxDD <= 0.01: 1
- aligned signal counts match: 1
- Phase A decision: **PASS**

- baseline_audit.csv: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_212415/baseline_audit.csv`
- metrics_definition.md: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_212415/metrics_definition.md`
- fee_model.json: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_212415/fee_model.json`
