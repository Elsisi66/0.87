# Phase H Diagnostics Taxonomy Audit

- Generated UTC: 2026-02-22T14:50:18.091419+00:00
- Source report: `/root/analysis/0.87/reports/execution_layer/PHASEG_SOL_PATHOLOGY_REHAB_20260222_143826/phaseG_report.md`

## Regime vs Exit-Reason Labeling

- Reported in G0 root-cause evidence: `dominant_worst_regime=sl`
- `sl` is an exit reason, not a regime bucket label.
- Corrected dominant worst regime bucket (from `phaseG0_sol_drawdown_buckets.csv`): `unknown|down`

## Adverse Loss Share Reconciliation

- `phaseG_report.md` top-level `adverse_loss_share`: 0.297872
- G0 root-cause `adverse_loss_share` (baseline forensic context): 0.568300
- G2 baseline row `adverse_loss_share`: 0.569697
- Last-evaluated G2 variant: `plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2` with `adverse_loss_share=0.297872`

Reconciliation:
- Top-level value in Phase G report aligns with the last evaluated variant, not baseline forensic context.
- Root-cause table value aligns with baseline forensic interpretation (and baseline G2 magnitude).
- For terminal memos, baseline context should be explicit and separate from per-variant diagnostics.

## Taxonomy Fix Standard for Future Reports

- Use `dominant_worst_regime_bucket` for regime fields and reserve `exit_reason` labels for stop/exit attribution sections.
- Emit both `baseline_adverse_loss_share` and `best_variant_adverse_loss_share` as separate named fields to prevent leakage.
- Gate decision rows should carry variant-local diagnostics only; summary headers should carry baseline-local diagnostics only.

## Verification Notes

- Loss clusters file present and readable: `/root/analysis/0.87/reports/execution_layer/PHASEG_SOL_PATHOLOGY_REHAB_20260222_143826/phaseG0_sol_loss_clusters.csv` (44 rows).
- Regime bucket rows present: 7.
