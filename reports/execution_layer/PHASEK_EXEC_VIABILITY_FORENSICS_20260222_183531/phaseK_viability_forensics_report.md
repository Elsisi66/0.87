# Phase K Viability Forensics Report

- Generated UTC: 2026-02-22T18:35:33.279933+00:00
- Analyzed run: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_181715`
- Candidates analyzed: 1830

## Root Cause Class

- Classification: **D**
- Reason: No valid candidates appear even under ±20% participation-gate counterfactuals while sampled behavior remains overwhelmingly low-participation.

## Gate Thresholds (Diagnostic Baseline)

- symbol: SOLUSDT (mode=tight)
- overall entry floor: 0.7000
- symbol entry floor (from execution config): 0.9700
- overall min trades floor: max(200, ceil(0.15 * signals))
- symbol min trades floor: max(50, ceil(0.10 * signals))
- symbol max taker share cap (before genome cap): 0.2500
- symbol max median fill delay min: 45.00
- symbol max p95 fill delay min: 180.00

## Failure Incidence

```csv
gate,source,threshold_expr,failed_count,failed_pct,pass_count,earliest_binding_only_fail_count,near_binding_fail_count_le2,slack_p05,slack_p50,slack_p95,slack_mean_failed_only
fail_symbol_entry_rate,computed_gate,overall_entry_rate >= th_symbol_entry_rate_base,1830,1.0,0,1,1,-0.97,-0.9422222222222223,-0.7906944444444446,-0.9122358834244081
fail_overall_entry_rate,computed_gate,overall_entry_rate >= th_overall_entry_rate_base,1829,0.9994535519125683,1,0,0,-0.7,-0.6722222222222223,-0.5206944444444446,-0.6426006925460178
fail_overall_min_trades,computed_gate,overall_entries_valid >= th_overall_min_trades_base,1829,0.9994535519125683,1,0,0,-200.0,-190.0,-135.45000000000005,-179.33624931656644
fail_symbol_min_trades,computed_gate,overall_entries_valid >= th_symbol_min_trades_base,1599,0.8737704918032787,231,0,0,-50.0,-40.0,14.549999999999955,-36.347091932457786
fail_symbol_taker_share,computed_gate,overall_exec_taker_share <= th_symbol_max_taker_share_row,484,0.2644808743169399,1346,0,0,0.0046081620416219,0.0827280000173193,0.21677145441181636,-0.2668207227133615
fail_symbol_median_fill_delay,computed_gate,overall_exec_median_fill_delay_min <= th_symbol_max_median_delay_base,472,0.25792349726775954,1358,0,0,21.0,45.0,45.0,-15.75
fail_nan,computed_gate,nan_pass == 1,468,0.25573770491803277,1362,0,0,,,,
fail_symbol_p95_fill_delay,computed_gate,overall_exec_p95_fill_delay_min <= th_symbol_max_p95_delay_base,468,0.25573770491803277,1362,0,0,140.1,168.0,177.0,
```

## Co-Failure Highlights

```csv
gate_a,gate_b,cofail_count
fail_overall_min_trades,fail_overall_entry_rate,1829
fail_symbol_entry_rate,fail_overall_entry_rate,1829
fail_symbol_entry_rate,fail_overall_min_trades,1829
fail_overall_entry_rate,fail_symbol_min_trades,1599
fail_symbol_min_trades,fail_symbol_entry_rate,1599
fail_overall_min_trades,fail_symbol_min_trades,1599
```

## Earliest-Binding Gate Estimate

```csv
gate,failed_count,failed_pct,earliest_binding_only_fail_count,near_binding_fail_count_le2,slack_p50
fail_symbol_entry_rate,1830,1.0,1,1,-0.9422222222222223
fail_overall_entry_rate,1829,0.9994535519125683,0,0,-0.6722222222222223
fail_overall_min_trades,1829,0.9994535519125683,0,0,-190.0
fail_symbol_min_trades,1599,0.8737704918032787,0,0,-40.0
fail_symbol_taker_share,484,0.2644808743169399,0,0,0.0827280000173193
fail_symbol_median_fill_delay,472,0.25792349726775954,0,0,45.0
fail_nan,468,0.25573770491803277,0,0,
fail_symbol_p95_fill_delay,468,0.25573770491803277,0,0,168.0
```

## Near-Feasible Frontier (Invalid Candidates)

- near_feasible_rows: 200
```csv
genome_hash,near_source,fail_count_total,failed_gates,frontier_norm_deficit_total,overall_exec_expectancy_net,overall_cvar_improve_ratio,overall_maxdd_improve_ratio,slack_overall_entry_rate,slack_overall_min_trades,slack_symbol_taker_share
4a97bd80bcdc1db375e045fb,closest_frontier_invalid|fail_count_1_to_2,1,fail_symbol_entry_rate,0.25257731958762886,-0.0004360527301996,-0.1063303596965868,0.637247191153054,0.025000000000000022,61.0,0.1074928653321543
abb426f167c9afae1120b5d0,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,0.7755948289968908,-0.000381561345265,-0.2139200111251776,0.6744627218293743,-0.1777777777777777,-12.0,0.1177702022153564
1522f5ae1135fe28f8b6cfe8,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,0.7874267713958434,-0.0004179727669968,0.1664733090868252,0.6565506137860347,-0.18055555555555547,-13.0,0.0551240194197732
47b957ff92d0c8a4bacd4851,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.0004017345769924,-0.0003380620685587,-0.0396464182763283,0.7194394330759636,-0.23055555555555557,-31.0,0.0356498052973774
030c1043c0549913bc7f9d9c,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.0122336769759448,-0.0003464390270876,0.0588454996686212,0.7181898276091117,-0.23333333333333328,-32.0,0.1855365496857257
6f7498864754ae5b0ea7901e,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.0240656193748976,-0.0004043171054119,0.2957410809345669,0.6693073639419829,-0.23611111111111105,-33.0,0.1094547161424575
73b33e174f4ba5d842df1e6c,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.1068892161675667,-0.0003938958931357,0.4553346122825486,0.6842806422442801,-0.25555555555555554,-40.0,0.1798529815782819
246593c19181f67065f97996,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.1068892161675667,-0.0004267927203982,-0.1620857925533906,0.6356134925467446,-0.25555555555555554,-40.0,0.0502317794718474
d9425b17b4c0bc39f69efb3d,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.248872524954999,-0.0001953281082927,0.042062864260605,0.8417675825583304,-0.28888888888888886,-52.0,0.1865137824830776
b89753f142f5f8977980eab8,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.248872524954999,-0.0002684410039129,-0.1639156670792626,0.7739153613304385,-0.28888888888888886,-52.0,0.0308204908548902
9fcccbeec211c207d83b9232,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.3198641793487156,-0.0002034316677817,-0.0860945024600968,0.8253826085263003,-0.3055555555555556,-58.0,0.1363552499289264
3d54b3250dcc239a20ab9e75,closest_frontier_invalid,3,fail_overall_entry_rate|fail_overall_min_trades|fail_symbol_entry_rate,1.3671919489445263,-0.0002288615930644,0.0938659820648941,0.792801297903769,-0.31666666666666665,-62.0,0.2086325060523434
```

## Participation-Gate Counterfactuals (DIAGNOSTIC ONLY)

- baseline pass_count (all deltas 0): 0
- max pass_count over tested perturbations: 0
- minimum shift that yields >0 pass (if any): none
```csv
delta_overall_entry_floor,delta_overall_min_trades,delta_symbol_entry_floor,delta_symbol_min_trades,counterfactual_pass_count,counterfactual_pass_pct,counterfactual_participation_pass_count,counterfactual_other_fixed_pass_count,counterfactual_new_valid_from_invalid_count
-0.2,-0.2,-0.2,-0.2,0,0.0,0,1342,0
-0.2,-0.2,-0.2,-0.1,0,0.0,0,1342,0
-0.2,-0.2,-0.2,0.0,0,0.0,0,1342,0
-0.2,-0.2,-0.2,0.1,0,0.0,0,1342,0
-0.2,-0.2,-0.2,0.2,0,0.0,0,1342,0
-0.2,-0.2,-0.1,-0.2,0,0.0,0,1342,0
-0.2,-0.2,-0.1,-0.1,0,0.0,0,1342,0
-0.2,-0.2,-0.1,0.0,0,0.0,0,1342,0
-0.2,-0.2,-0.1,0.1,0,0.0,0,1342,0
-0.2,-0.2,-0.1,0.2,0,0.0,0,1342,0
-0.2,-0.2,0.0,-0.2,0,0.0,0,1342,0
-0.2,-0.2,0.0,-0.1,0,0.0,0,1342,0
```

## Sampler/Search-Space Feasibility Check

- parameters_analyzed: 28
- participation-collapse knobs flagged: 5
- dead knobs flagged: 1
- dead knob list (first 20): g_entry_mode
```csv
parameter,unique_count,spearman_corr_entry_rate,spearman_corr_entries_valid,fail_rate_delta_high_minus_low,entry_rate_delta_high_minus_low,entries_valid_delta_high_minus_low,association_strength,collapses_participation_flag,dead_knob_flag
g_min_signal_quality_gate,480,-0.20992196495991655,-0.20992196495991655,0.0,-0.03586194800248563,-12.910301280894823,1.4122703189425905,1,0
g_mss_displacement_gate,2,-0.5612310030230371,-0.5612310030230371,,,,1.1224620060460742,1,0
g_max_taker_share,485,-0.19203897552379576,-0.19203897552379576,0.0,-0.025800582241630274,-9.288209606986896,1.0980709183836406,1,0
g_vol_threshold,467,-0.1953180410188805,-0.1953180410188805,0.0,-0.016491038166432284,-5.936773739915626,0.8470011812681352,1,0
g_killzone_filter,2,-0.4164186307529174,-0.4164186307529174,,,,0.8328372615058348,1,0
g_spread_guard_bps,477,0.5280726926512818,0.5280726926512818,0.0,0.0741181411181411,26.682530802530803,3.1072553569599215,0,0
g_partial_take_pct,434,0.4132253215372269,0.4132253215372269,0.0,0.06115025329597802,22.0140911865521,2.5186934922958115,0,0
g_sl_mult,451,0.3103326015004673,0.3103326015004673,0.0,0.037764660692030314,13.59527784913092,1.6657463525431635,0,0
g_partial_take_r,477,0.2295859384558241,0.2295859384558241,0.0,0.0316027390344065,11.376986052386345,1.3337309549899157,0,0
g_trail_start_r,451,0.19638828113504403,0.19638828113504403,0.0,0.024920733780223782,8.971464160880563,1.0824210242940344,0,0
g_trail_step_bps,462,0.15654813718760988,0.15654813718760988,0.0,0.020670136000297164,7.441248960106982,0.8851117273551417,0,0
g_cooldown_min,197,-0.07398423186191233,-0.07398423186191233,0.0,-0.02110166363920392,-7.596598910113414,0.7319258060962041,0,0
```

## Decision

- Final class: **D**
- Recommendation: treat current branch as likely dead unless major model/execution definition changes are introduced.
