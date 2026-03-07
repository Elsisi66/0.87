# Phase S Guardrails And Rollback

- Hard guardrail: keep production gate definitions unchanged.
- Immediate rollback triggers (paper mode disable):
  - taker_share > 0.25
  - p95_fill_delay > 180 min
  - delta_expectancy_vs_baseline <= 0 for 2 consecutive review windows
  - maxdd_improve_ratio <= 0 for 2 consecutive review windows
- Incident checklist: freeze lock check, data integrity check, execution feed check, config hash check, scenario replay.
