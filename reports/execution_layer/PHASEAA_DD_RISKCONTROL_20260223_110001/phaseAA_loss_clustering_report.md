# Phase AA Loss Clustering Report

- Generated UTC: 2026-02-23T11:00:32.655732+00:00
- Baseline genome hash: `862c940746de0da984862d95` (`E1`)
- Source PhaseS dir: `/root/analysis/0.87/reports/execution_layer/PHASEQRS_AUTORUN_20260222_201536`
- Source PhaseV dir: `/root/analysis/0.87/reports/execution_layer/PHASEV_BRANCHB_PORTABILITY_DD_20260222_235009`

## Baseline Reproduction

- valid_for_ranking: `1`
- exec_expectancy_net: `0.00005649`
- exec_cvar_5: `-0.00177975`
- exec_max_drawdown: `-0.18691674`
- entries_valid / entry_rate: `355` / `0.986111`

## Loss Clustering

- max_consecutive_losses: `33`
- streak>=3 count: `24`
- streak>=5 count: `20`
- streak>=10 count: `12`
- unconditional_loss_rate: `0.909859`
- conditional_loss_rate_after_loss: `0.909938`
- conditional_loss_rate_after_nonloss: `0.906250`
- sl_loss_share: `0.884618`

## Worst 3 Loss Streak Segments

| start_entry_time | end_entry_time | length | pnl_sum | mean_pnl | sl_share_in_segment | dominant_exit_reason | dominant_session_bucket | dominant_vol_bucket | median_fill_delay_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-07-16 21:03:00+00:00 | 2025-08-09 14:00:00+00:00 | 33 | -0.047087134 | -0.0014268829 | 0.90909091 | sl | 12_17 | mid | 0 |
| 2025-08-09 20:00:00+00:00 | 2025-08-27 18:00:00+00:00 | 31 | -0.043464168 | -0.0014020699 | 0.87096774 | sl | 06_11 | high | 0 |
| 2025-10-03 06:00:00+00:00 | 2025-11-10 18:00:00+00:00 | 28 | -0.039348982 | -0.0014053208 | 0.82142857 | sl | 12_17 | low | 0 |

## Tail Attribution Highlights

| tail_name | axis | bucket | trades_count | loss_abs_sum | share_of_tail_loss_abs |
| --- | --- | --- | --- | --- | --- |
| cvar_5 | entry_mechanic | market_fallback | 17 | 0.030586087 | 0.95475824 |
| cvar_5 | entry_mechanic | limit | 1 | 0.001449339 | 0.045241757 |
| cvar_5 | session_bucket | 00_05 | 5 | 0.008995908 | 0.28081125 |
| cvar_5 | session_bucket | 06_11 | 5 | 0.008995908 | 0.28081125 |
| cvar_5 | session_bucket | 18_23 | 4 | 0.0071967264 | 0.224649 |
| cvar_5 | sl_hit | sl_hit_1 | 18 | 0.032035426 | 1 |
| cvar_5 | vol_bucket | unknown | 6 | 0.01079509 | 0.3369735 |
| cvar_5 | vol_bucket | mid | 5 | 0.008995908 | 0.28081125 |
| cvar_5 | vol_bucket | low | 4 | 0.0068468838 | 0.21372851 |
