# Phase D2 Report

- Generated UTC: 2026-02-23T13:44:03.147309+00:00
- Decision: **PASS**
- Reason: tail label stable and monotonic
- Combined support: `478`
- Combined Spearman(tail_risk_score, y_tail_loss): `0.291276`
- Combined stable_sign_frac: `1.0000` (rule >= 0.75)
- Combined monotonic violations: `0` (rule <= 1)

## Per-route tail stability

| route_id | support | tail_rate | overall_spearman_tail | stable_sign_frac | split_eligible | split_positive | score_nan_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| route1_holdout | 70 | 0.28571429 | 0.27392601 | 0 | 0 | 0 | 0 |
| route2_reslice | 408 | 0.06127451 | 0.177319 | 1 | 6 | 6 | 0 |

## Leakage check

- y_tail_loss is created from route outcomes with split-aware train-thresholding (no same-split threshold leakage).
- tail_risk_score is a monotonic mapping from pre-entry risk score bins.
