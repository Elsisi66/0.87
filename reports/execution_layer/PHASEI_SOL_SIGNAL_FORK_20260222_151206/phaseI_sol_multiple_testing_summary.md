# Phase I Multiple-Testing Summary

- Generated UTC: 2026-02-22T15:13:15.684636+00:00
- Baseline variant: `baseline_contract_locked`
- Raw fork trials evaluated: 120
- Effective trials estimate (correlation-adjusted): 100.947173
- Average pairwise correlation across fork trial step-returns: 0.160108
- Shortlist size: 5

## Reality-Check Benchmark Note

Shortlisted variants are compared against the contract-locked baseline for directional uplift context.
A full White reality-check bootstrap is not implemented in this run; this remains a required control before any deployment escalation.

| variant | delta_geom_step_vs_baseline | delta_total_return_fixed_vs_baseline |
| --- | --- | --- |
| fork_trend0.50_high_only_cd6h_d2 | 0.998132 | 15.293702 |
| fork_trend0.60_high_only_cd6h_d2 | 0.998132 | 15.293702 |
| fork_trend0.50_high_only_cd4h_d2 | 0.993618 | 14.957680 |
| fork_trend0.60_high_only_cd4h_d2 | 0.993618 | 14.957680 |
| fork_trend0.50_mid_only_cd2h_d1 | 0.000000 | 15.192780 |
