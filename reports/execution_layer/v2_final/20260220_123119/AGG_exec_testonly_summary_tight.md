# AGG Exec Test-Only Tight

- Generated UTC: 2026-02-20T12:31:34.370249+00:00
- Run dirs used (deduped): 3

## Per Symbol

- AVAXUSDT: entry_rate=0.968539, expectancy_net=-0.000660, baseline_expectancy_net=-0.000381, taker_share=0.155452, median_fill_delay_min=0.00
- NEARUSDT: entry_rate=0.951311, expectancy_net=-0.000837, baseline_expectancy_net=-0.000695, taker_share=0.122047, median_fill_delay_min=0.00
- SOLUSDT: entry_rate=0.988333, expectancy_net=-0.000856, baseline_expectancy_net=-0.000643, taker_share=0.116358, median_fill_delay_min=0.00

## Overall

- entry_rate=0.974085
- exec_expectancy_net=-0.000786
- baseline_expectancy_net_proxy=-0.000565
- taker_share=0.130673
- median_fill_delay_min_weighted=0.00

## Rubric Decision

- Decision: **exec not worth it; focus on 1h edge/stops**
- Rule thresholds: tail_improvement>=15%, taker_share<=0.25, median_fill_delay<=45
- CVaR improvement ratio: 0.000000
- MaxDD improvement ratio: -0.358125
- Exec expectancy >= baseline expectancy: 0
- Taker condition met: 1
- Delay condition met: 1
