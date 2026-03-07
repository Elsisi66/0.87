# Repaired Frontier Contamination Report

- Generated UTC: `2026-03-03T23:35:24.631551+00:00`
- Audit scope: repaired 1h-only upstream contamination audit; intentionally excludes the 3m execution layer.

## Discovered Artifacts
- Phase I frontier: `/root/analysis/0.87/reports/execution_layer/PHASEI_EXECAWARE_1H_GA_EXPANSION_20260224_012237`
- Params scan universe: `/root/analysis/0.87/reports/params_scan/20260220_044949`
- Repaired baseline: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Phase J0 appendix: `/root/analysis/0.87/reports/execution_layer/PHASEJ0_POST_PHASEI_RECOVERY_20260224_025200`
- Repaired signal root / fee model source: `/root/analysis/0.87/reports/execution_layer/MULTICOIN_MODELA_AUDIT_20260228_180250`

## Discovery Validation
- Phase I discovery target matched expected: `1`
- Params scan discovery target matched expected: `1`
- Repaired baseline discovery target matched expected: `1`

## Coverage
- Phase I coverage: `215/215` = `100.0000%`
- Universe coverage: `18/18` = `100.0000%`

## Legacy Reconstruction Parity
- Phase I signal reconstruction parity: `ok=1`, sample=`25`, mismatches=`0`
- Universe legacy evaluator parity: `ok=1`, sample=`18`, mismatches=`0`
- Repaired evaluator smoke check (SOL/NEAR/AVAX): `ok=1`, mismatches=`0`

## Phase I Frontier Comparison
- Old score = legacy GA objective `OJ2`.
- New score = repaired 1h symbol-equity score `(cagr_pct * profit_factor) / (1 + max_dd_pct)`.
- Spearman(old rank, repaired rank): `-0.262587`
- top10_legacy_retained_pct: `0.0`
- top10_shared: `0`
- top10_turnover: `1.0`
- top20_legacy_retained_pct: `0.0`
- top20_shared: `0`
- top20_turnover: `1.0`
- top5_legacy_retained_pct: `0.0`
- top5_shared: `0`
- top5_turnover: `1.0`
- Top-20 repaired pass-proxy flips (secondary annotation): `16` / `20` = `80.0000%`

## Universe Membership Comparison
- Passed-set Jaccard: `0.800000`
- STAY_PASS: `12`
- DROP_FROM_PASS: `1`
- NEW_PASS: `2`
- STAY_FAIL: `3`
- Drop rate vs legacy pass set: `7.6923%`
- Shared-pass rank Spearman: `0.843409`
- top10_shared: `9`
- top10_turnover: `0.09999999999999998`
- top5_shared: `4`
- top5_turnover: `0.19999999999999996`
- Legacy pass set: `ADAUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, CRVUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, SOLUSDT, TRXUSDT, XRPUSDT, ZECUSDT`
- Repaired pass set: `ADAUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, OGUSDT, SOLUSDT, TRXUSDT, ZECUSDT`

## Decision
- Contamination severity: `HIGH_CONTAMINATION`
- Final recommendation: `FULL_1H_GA_RERUN_RECOMMENDED`
