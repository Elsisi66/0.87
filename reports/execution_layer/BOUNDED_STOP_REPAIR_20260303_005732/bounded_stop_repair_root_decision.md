# Bounded Stop Repair Decision

- Generated UTC: 2026-03-03T01:03:53.428208+00:00
- Frozen repaired multicoin source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Frozen survivor forensics source: `/root/analysis/0.87/reports/execution_layer/SURVIVOR_COIN_STOP_FORENSICS_20260303_003559`
- Outcome: `ENTRY_LAYER_NOW_BECOMES_PRIMARY_PROBLEM`
- Universal winning variant across primary survivors: `CONTROL_SIGNAL_MULT`
- Approved: `[]`
- Shadow only: `['DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT']`
- Disabled: `['SOLUSDT', 'AVAXUSDT']`

## Best Variant Per Coin

| symbol | variant_id | modelA_delta_expectancy_vs_control_modelA | modelA_delta_expectancy_vs_repaired_1h | modelA_pct_exit_within_3h_reduction | modelA_pct_exit_within_4h_reduction | modelA_cvar_improve_ratio_vs_control | modelA_maxdd_improve_ratio_vs_control | modelA_valid_for_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | CONTROL_SIGNAL_MULT | 0 | 0.0002581020542 | 0 | 0 | 0 | 0 | 0 |
| DOGEUSDT | CONTROL_SIGNAL_MULT | 0 | 0.0006817630797 | 0 | 0 | 0 | 0 | 1 |
| LINKUSDT | CONTROL_SIGNAL_MULT | 0 | 0.0002978493519 | 0 | 0 | 0 | 0 | 1 |
| LTCUSDT | CONTROL_SIGNAL_MULT | 0 | 0.0004895665814 | 0 | 0 | 0 | 0 | 1 |
| NEARUSDT | CONTROL_SIGNAL_MULT | 0 | 0.0003615505286 | 0 | 0 | 0 | 0 | 1 |
| SOLUSDT | SIGDIST_X2_CAP60BPS | 3.932637008e-05 | 8.505086578e-05 | 0.03610906411 | 0.03979366249 | -0.4216570239 | -0.6805123803 | 0 |