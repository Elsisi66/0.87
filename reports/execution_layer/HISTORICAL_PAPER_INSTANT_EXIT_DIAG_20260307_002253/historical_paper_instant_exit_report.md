# Historical Paper Instant-Exit Diagnosis

**Data Source Label: PRIMARY (historical paper bot trade logs)**

## Scope
- Window label: `default`
- Window UTC: `2026-03-01T00:00:00+00:00` to `2026-03-06T00:00:00+00:00` (end exclusive)
- Closed trades analyzed: `107`

## Hard Conclusion
`LOSSES_DOMINATED_BY_INSTANT_EXITS`
- Basis: Losses are concentrated in very short holds (<=3m) and same-parent-bar closes.

## Core Metrics
- win_rate: `0.000000`
- profit_factor: `0.000000`
- expectancy per trade (EUR): `-0.597985`
- avg_win / avg_loss / win_loss_ratio: `n/a` / `-0.597985` / `n/a`

## Hold-Time and Instant Exit
- hold minutes mean / median / p10 / p90: `0.000000` / `0.000000` / `0.000000` / `0.000000`
- instant_exit_rate A (hold <= 3 min): `1.000000`
- instant_exit_rate B (hold <= 60 min): `1.000000`

## Exit Reason and Loss Concentration
- exit_reason breakdown: `{"sl": 107}`
- worst10 sum (EUR): `-7.015668`
- worst25 sum (EUR): `-16.684589`
- bottom-decile negative pnl share: `0.120047` (decile count `11`)

## Plumbing-Era Artifact Checks
- same parent 1h bar close fraction: `1.000000` (`107` / `107`)
- exit_before_entry count: `0`
- entry_time < signal_time check: `n/a` (signal_time not present in source rows)

## Symbol Mix
- `{"ADAUSDT": 1, "AVAXUSDT": 19, "CRVUSDT": 5, "DOGEUSDT": 1, "LINKUSDT": 4, "LTCUSDT": 2, "NEARUSDT": 17, "SOLUSDT": 48, "XRPUSDT": 7, "ZECUSDT": 3}`

## Recommendation
- Primary issue is execution/plumbing behavior, not weak alpha in this window; prioritize bounded mechanical controls (chronology-safe entry/exit gating and volatility-normalized stop floor) before alpha redesign.

_Generated UTC: `2026-03-07T00:22:53.121596+00:00`_
