# Bounded Stop Repair Paper Parity Check

- Generated UTC: 2026-03-03T01:03:53.430641+00:00
- Timing guard preserved: `searchsorted(..., side="right")` starts exit evaluation on the first full 1h bar after fill.
- Trigger rule preserved: `wick touch on 1h low<=sl or high>=tp; same-bar dual touch still resolves to SL`.
- Hybrid exit mutation: `disabled` (entry logic unchanged, TP formula unchanged, only stop-construction formula varies).
- Reproducible config loading: `phase_v.build_exec_args(...)` + frozen fee/metric lock reused.

## Winning Variant Checks

| symbol | variant_id | parity_same_parent_bar_violations | parity_stop_trigger_mismatches | modelA_valid_for_ranking | modelA_invalid_reason |
| --- | --- | --- | --- | --- | --- |
| AVAXUSDT | CONTROL_SIGNAL_MULT | 0 | 0 | 0 | AVAXUSDT:missing_slice_rate>0.0200 |
| DOGEUSDT | CONTROL_SIGNAL_MULT | 0 | 0 | 1 |  |
| LINKUSDT | CONTROL_SIGNAL_MULT | 0 | 0 | 1 |  |
| LTCUSDT | CONTROL_SIGNAL_MULT | 0 | 0 | 1 |  |
| NEARUSDT | CONTROL_SIGNAL_MULT | 0 | 0 | 1 |  |
| SOLUSDT | SIGDIST_X2_CAP60BPS | 0 | 0 | 0 | SOLUSDT:missing_slice_rate>0.0200 |