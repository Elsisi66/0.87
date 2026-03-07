# Next Experiment Plan

- Generated UTC: 2026-02-22T18:11:36.882494+00:00
- Chosen branch: **A**
- Branch reason: Implementation patches/tests are present, but fixed-size survivors remain zero; signal fork still non-deployable.

## Evidence Basis

- Phase I postfix source: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_POSTFIX_VALIDATION_20260222_180531`
- Phase I forensic source: `/root/analysis/0.87/reports/execution_layer/PHASEI_FORENSIC_DEBUG_20260222_153532`
- fixed_size_passers_nonduplicate: 0
- bug_fix_confirmed_from_forensic_tests: 1
- postfix recommendation: pivot_to_execution_exit_optimization

## Frozen Setup Confirmation

- representative_subset_path: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv`
- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- setup_checks_vs_expected: `{"fee_hash_match_expected": 1, "metrics_hash_match_expected": 1, "model_set_hash_match_expected": 1, "rep_hash_match_expected": 1}`

## Execution/Exit Pivot Design

Entry search family:
- delay and fill control (`max_fill_delay_min`, `fallback_to_market`, `fallback_delay_min`)
- entry price discipline (`limit_offset_bps`, `min_entry_improvement_bps_gate`, `spread_guard_bps`)
- entry quality gating (`mss_displacement_gate`, `micro_vol_filter`, `skip_if_vol_gate`, `killzone_filter`)
- flow pacing (`cooldown_min`, taker-share limits)

Exit search family:
- target/stop geometry (`tp_mult`, `sl_mult`)
- time-based exits (`time_stop_min`)
- adaptive protection (`break_even_*`, `trailing_*`)
- staged exits (`partial_take_*`)

Objective hierarchy for full run:
- Primary: fixed-size geometric equity-step return (when exported by runner)
- Secondary: drawdown and CVaR floors plus split stability/support
- Tertiary: expectancy and fill-quality
- Penalties: overtrading, taker-share excess, regime concentration

Hard constraints:
- minimum split trade support
- support/validity gates
- no NaN/metric pathology
- no single-regime dominance beyond threshold (requires regime audit column in full run)

Anti-overfit controls:
- duplicate candidate collapse
- effective trial count after pruning
- PSR/DSR proxy reporting for shortlist
- reality-check bootstrap placeholder (explicit TODO)

## Pilot Spec

- Scope: SOLUSDT only
- Budget: 48 candidates (single generation)
- Engine: `src/execution/ga_exec_3m_opt.py`
- Run root: `/root/analysis/0.87/reports/execution_layer/PHASEJ_EXEC_PIVOT_20260222_181136/pilot_ga_runs`
- Stop condition: if zero viable candidates after hard constraints, mark NO_GO for full marathon

## Notes

- No downstream contract, fee model, or subset edits were introduced in this phase.
- This pilot is a diversity/viability probe, not a full optimization campaign.
