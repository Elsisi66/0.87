# Phase S Paper Monitoring Spec

- Scope: SOLUSDT only, paper/shadow mode.
- Minimum monitoring horizon: 30 calendar days or 300 executed entries (whichever is later).
- Core KPIs: exec_expectancy_net, delta_expectancy_vs_baseline, cvar_improve_ratio, maxdd_improve_ratio, entry_rate, taker_share, p95_fill_delay.
- Success condition: delta_expectancy_vs_baseline remains > 0 on rolling 2-week windows and risk improvements remain non-negative.
- Fail condition: two consecutive rolling windows with delta_expectancy<=0 or risk-improve ratios <=0.
- Drift checks: taker_share drift > +5pp, p95_fill_delay drift > +20 min, entry_rate drop below 0.90.
