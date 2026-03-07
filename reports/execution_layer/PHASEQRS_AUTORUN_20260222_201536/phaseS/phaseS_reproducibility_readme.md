# Phase S Reproducibility Readme

- Generated UTC: 2026-02-22T20:22:21.160792+00:00
- Source full run: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`
- Q result: Q_PASS_WEAK
- R result: R_PASS
- S result: S_PROMOTE_PAPER_CAUTION

## Frozen Locks

- representative_subset_csv: /root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv
- canonical_fee_model: /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json
- canonical_metrics_definition: /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md
- canonical_fee_sha256: b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a
- canonical_metrics_sha256: d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99
- expected_fee_sha256: b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a
- expected_metrics_sha256: d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99
- fee_hash_match: 1
- metrics_hash_match: 1

## Reproduce

1) Use source full run artifacts and selected candidate hashes from `phaseS_final_candidates.csv`.
2) Re-run OOS routes from Phase Q using the same scripts and locked files.
3) Re-run Phase R stress matrix on survivor set.
4) Verify manifests and hashes match before any paper-mode activation.
