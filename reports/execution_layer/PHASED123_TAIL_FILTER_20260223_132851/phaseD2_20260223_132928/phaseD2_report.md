# Phase D2 Report

- Generated UTC: 2026-02-23T13:29:28.436850+00:00
- Decision: **NO_GO**
- Reason: tail label stability below threshold or pathology
- Tail label fraction (worst-X%): `0.10`
- Combined support: `478`
- Combined Spearman(risk_score, y_tail): `-0.026726`
- Combined stable_sign_frac: `0.4286`

## Per-route tail stability

| route_id | support | tail_rate | overall_spearman_tail | stable_sign_frac | split_eligible | split_positive | score_nan_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| route1_holdout | 70 | 0.28571429 | 0.0015650813 | 0 | 0 | 0 | 0 |
| route2_reslice | 408 | 0.31862745 | -0.032290591 | 0.42857143 | 7 | 3 | 0 |

## Leakage check

- y_tail_loss_route defined from realized route outcomes only (label), not used as pre-entry feature.
- Risk score uses pre-entry and prior-only features from AE family.
