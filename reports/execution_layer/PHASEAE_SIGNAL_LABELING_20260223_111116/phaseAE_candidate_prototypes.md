# Phase AE Candidate Prototypes

- Generated UTC: 2026-02-23T11:12:18.008688+00:00
- No session/time veto rule is used.
- Hard filters are exact replay; soft sizing is proxy replay only.

## H1_vol_trend_guard

- Type: `hard_filter_exact`
- Rule: block if pre3m_atr_z>=q(pre3m_atr_z,0.8) AND trend_up_1h<=q(trend_up_1h,0.2)
- Expected impact target: tail + cluster risk in weak-trend high-vol conditions
- Risk: participation drop, possible over-filtering

## H2_spread_vol_guard

- Type: `hard_filter_exact`
- Rule: block if pre3m_spread_proxy_bps>=q80 AND pre3m_realized_vol_12>=q80
- Expected impact target: fee/slippage and bad fills in stressed microstructure
- Risk: entry starvation if too broad

## H3_impulse_vol_guard

- Type: `hard_filter_exact`
- Rule: block if pre3m_body_bps_abs>=q80 AND pre3m_atr_z>=q80
- Expected impact target: impulsive entries likely to mean-revert into SL
- Risk: removes momentum winners too

## S1_risk_score_half_size

- Type: `soft_size_proxy`
- Rule: size=0.5 if risk_score>=2 else 1.0
- Expected impact target: reduce tail exposure without hard blocking
- Risk: proxy only (engine does not natively support per-trade sizing)

## S2_risk_score_quarter_size

- Type: `soft_size_proxy`
- Rule: size=0.25 if risk_score>=3 else 1.0
- Expected impact target: aggressive tail-risk suppression under compounded risk
- Risk: may suppress too much gross edge
