# SOL Reconciliation Report

- Generated UTC: 2026-02-22T01:13:08.903560+00:00
- Symbol: SOLUSDT
- Phase C dir: `/root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430`
- Phase A dir: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310`

## Contract Verification

- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- signal_subset_hash: `5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf`
- wf_split_hash: `388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01`

## Core Contradiction

- V1 native fullscan total_return: 678.304825 (final_equity=6793048.25)
- V2 frozen 1h reference total_return: -0.996799
- V4 frozen Phase C best total_return: -0.993151

## Root Cause Summary

- Fullscan and frozen evaluations are not the same universe or contract.
- Fullscan uses endogenous 1h signal generation + legacy fee/slip and native ATR sizing.
- Frozen evaluation uses exported signal subset + 3m path simulation + Phase A fee contract + fixed-risk equity simulator.
- Phase C best improves over frozen control but remains deeply negative in absolute equity terms on this frozen sample.

## Reproduction Check (V2/V3/V4)

| variant | expectancy_actual | expectancy_expected | expectancy_delta | maxdd_actual | maxdd_expected | maxdd_delta | pass_tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE | -0.000649 | -0.000649 | 0.000000 | -0.999175 | -0.999175 | 0.000000 | 1 |
| V3_EXEC_3M_PHASEC_CONTROL_FROZEN | -0.000643 | -0.000643 | 0.000000 | -0.998986 | -0.998986 | 0.000000 | 1 |
| V4_EXEC_3M_PHASEC_BEST_FROZEN | -0.000559 | -0.000559 | 0.000000 | -0.998117 | -0.998117 | 0.000000 | 1 |

## Universe Comparison

| variant_scope | date_start | date_end | duration_days | sample_type | sample_count | signals_count | trades_total | fee_model | sizing_model | split_scope | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1_1H_FULLSCAN_REFERENCE | 2020-08-11 06:00:00+00:00 | 2025-12-31 23:00:00+00:00 | 1968.708333 | 1h bars + endogenous signals | 47230 | 5034 | 2551 | legacy_fullscan_fee_bps=7;slippage_bps=2 (both sides) | native_atr_position_sizing_compounding | full_period_no_wf_split | Source used by params_scan/best_by_symbol |
| FROZEN_PHASEC_TEST_UNIVERSE | 2025-04-11 18:00:00+00:00 | 2025-11-11 04:00:00+00:00 | 213.416667 | frozen_exported_signals_test_only | 600 | 600 | 600 | phase_a_maker_taker_fee_model | fixed_fractional_risk_per_trade_compounding | walkforward_test_splits_only | Universe used by Phase C/Phase D decisioning |
| V3_EXEC_3M_PHASEC_CONTROL_FROZEN | 2025-04-11 18:00:00+00:00 | 2025-11-11 04:00:00+00:00 | 213.416667 | same_frozen_test_signals | 600 | 600 | 600 | phase_a_maker_taker_fee_model | fixed_fractional_risk_per_trade_compounding | walkforward_test_splits_only | Phase C control baseline_exit |
| V4_EXEC_3M_PHASEC_BEST_FROZEN | 2025-04-11 18:00:00+00:00 | 2025-11-11 04:00:00+00:00 | 213.416667 | same_frozen_test_signals | 600 | 600 | 600 | phase_a_maker_taker_fee_model | fixed_fractional_risk_per_trade_compounding | walkforward_test_splits_only | Phase C best global exit |

## Metric Glossary Snapshot

- `expectancy_net`: mean net return per valid trade.
- `total_return`: final_equity / initial_equity - 1.
- `max_drawdown_pct`: most negative peak-to-trough drawdown fraction.
- `cvar_5`: mean of worst 5% trade outcomes.

## Decision Inputs

- delta_expectancy(V4 - V3): 0.000084
- delta_maxdd(V4 - V3): 0.000869
- delta_expectancy(V4 - V1): -0.003376
- deploy_status_candidate: HOLD
