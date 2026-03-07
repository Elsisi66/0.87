# NY1 Economics Upper Bound

- Generated UTC: 2026-02-27T12:34:28.122991+00:00
- NX input dir: `/root/analysis/0.87/reports/execution_layer/PHASENX_EXEC_FAMILY_DISCOVERY_20260227_115329`
- Best variant by net expectancy: `REGIME_ROUTED_EXEC_mid_00` (-0.0004218327)
- Economic feasibility verdict: `infeasible`
- Edge near zero band hit (abs(net)<=1 bp/signal and net<0): `0`
- Best-variant required cost reduction to reach net>=0: `48.081%`

- Families with zero valid_for_ranking variants in NX3 (excluded from detailed re-eval): `['STAGED_ENTRY_RISKSHAPE']`

## Economics Table

| record_type | family_id | variant_id | valid_for_ranking | expectancy_gross | expectancy_net | fee_drag_per_signal | required_gross_edge_lift_to_net_zero_bps | required_cost_reduction_pct | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | BASELINE | BASELINE_FROZEN | 1 | 0.000362209189 | -0.0008378556817 | 0.001200064871 | 8.378556817 | 69.81753255 |  |
| variant | REGIME_ROUTED_EXEC | REGIME_ROUTED_EXEC_mid_00 | 1 | 0.0004555032218 | -0.000421832692 | 0.0008773359138 | 4.21832692 | 48.08109247 |  |
| variant | PASSIVE_LADDER_ADAPTIVE | PASSIVE_LADDER_ADAPTIVE_conservative_03 | 1 | 0.0003901163209 | -0.0004658096944 | 0.0008559260153 | 4.658096944 | 54.42172409 |  |

Gross/Net fields are computed from detailed signal rows (`*_pnl_gross_pct`, `*_pnl_net_pct`) using the same expectancy convention as the evaluator (zeros for non-filled/non-valid entries).
