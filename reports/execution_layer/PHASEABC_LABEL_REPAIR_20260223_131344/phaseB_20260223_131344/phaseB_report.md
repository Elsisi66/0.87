# Phase B Report

- Generated UTC: 2026-02-23T13:14:46.390766+00:00
- Decision: **PASS**
- Reason: stable risk score
- E1 support: `355`
- E1 stable_sign_frac: `0.8000` (rule >= 0.60)
- E1 overall spearman(y_toxic): `0.190673`
- E1 monotonic violations: `1`
- E2 overall spearman(y_toxic): `0.190673`
- Backup direction consistent: `1`

## Split Spearman (E1)

| split_id | support | spearman |
| --- | --- | --- |
| 0 | 71 | 0.37451581 |
| 1 | 71 | 0.13832083 |
| 2 | 72 | -0.2771655 |
| 3 | 71 | 0.17751174 |
| 4 | 70 | 0.062223451 |

## Split Spearman (E2)

| split_id | support | spearman |
| --- | --- | --- |
| 0 | 71 | 0.37451581 |
| 1 | 71 | 0.13832083 |
| 2 | 72 | -0.2771655 |
| 3 | 71 | 0.17751174 |
| 4 | 70 | 0.062223451 |

## Leakage Audit

- Feature set uses only pre-entry fields and prior-trade context fields from AE.
- Excluded post-entry outcome fields from risk score construction.
