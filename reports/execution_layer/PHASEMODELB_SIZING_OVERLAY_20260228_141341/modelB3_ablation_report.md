# Model B3 Ablation Report

## Baseline Reference
- Frozen Model A primary expectancy_net: `0.001211949481977627`
- Frozen Model A primary cvar_5: `-0.0019663317074100134`
- Frozen Model A primary max_drawdown: `-0.14088281136390174`
- Frozen Model A primary max_consecutive_losses: `60`
- Frozen Model A primary loss_cluster_worst_burden: `-0.11098425078746454`
- Frozen Model A primary loss_cluster_avg_burden: `-0.022393358929102526`

## Top Model B Variants
| variant_id | family | expectancy_net | delta_expectancy_vs_modelA | cvar_improve_ratio | maxdd_improve_ratio | loss_cluster_avg_burden_improve_ratio | route_pass | stress_lite_pass | bootstrap_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_cap_size_delay_tiered | regime_cap_size | 0.001234498935 | 2.254945341e-05 | 0.05592952826 | 0.0105036945 | 0.02147711198 | 1 | 1 | 0.95 |

## Interim Readout
- Preliminary classification: `MODEL_B_GO`
- Reason: `material risk-shape improvement with no meaningful expectancy damage`
