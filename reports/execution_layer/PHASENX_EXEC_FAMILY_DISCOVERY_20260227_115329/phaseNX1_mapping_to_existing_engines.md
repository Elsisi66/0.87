# NX1 Mapping To Existing Engines

- Core scorer/gates reused from `src/execution/ga_exec_3m_opt.py`:
  - unchanged aggregation and hard-gate validity logic
  - unchanged CVaR/maxDD/expectancy definitions
- New family mechanics implemented as wrapper-level entry simulators feeding equivalent per-signal rows into existing scorer.
- Exit path remains `ga_exec._simulate_dynamic_exit_long` for all families.
- Contract lock remains `ga_exec._validate_and_lock_frozen_artifacts` with `allow_freeze_hash_mismatch=0`.

Telemetry additions:
- Funnel counters: time/micro-vol/state eligibility, limit placement/fill, fallback/market fill counts.
- Route bucket traces for regime-routed family.
- Duplicate/effective-trial accounting at ablation + GA pilot stages.
