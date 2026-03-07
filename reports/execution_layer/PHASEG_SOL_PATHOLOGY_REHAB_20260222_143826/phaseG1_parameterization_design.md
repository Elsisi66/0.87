# Phase G1 Universe-Conditioned Parameterization

- Generated UTC: 2026-02-22T14:38:33.989717+00:00
- Universe summary source: `/root/analysis/0.87/artifacts/reports/universe_20260211_172937/summary.json`
- Selected model set hash: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- Candidate scan rows: 3
- Candidate scan fail-fast streak threshold: 3
- Selected offset_scale: 0.250

## Global Prior (Passed Long Universe)

- symbols: AVAXUSDT, BCHUSDT, CRVUSDT, NEARUSDT, SOLUSDT
- tp_prior_median: [1.0395, 1.125048, 1.067449, 1.080213, 1.030945]
- sl_prior_median: [0.964229, 0.999, 0.967705, 0.991295, 0.937986]

## SOL Anchor and Bounded Offsets

- sol_anchor_tp: [1.022013, 1.099919, 1.102416, 1.021166, 1.005057]
- sol_anchor_sl: [0.967117, 0.999, 0.960227, 0.927437, 0.922472]
- tp_offset_cap: 0.080000
- sl_offset_cap: 0.050000
- uc_tp_vector: [1.035128, 1.118766, 1.076191, 1.065451, 1.024473]
- uc_sl_vector: [0.964951, 0.999, 0.965836, 0.97533, 0.934108]

## Stability and Practical Constraints

- min_trades: 120
- min_split_trades: 40
- min_bucket_support: 30
- parameter_drift_limit: 0.030000 (measured drift=0.0 under fixed-vector design)
- fatal gates: max_dd<=-0.950000 or total_return<=-0.950000

## Candidate Scan

| offset_scale | signals_total | trades_total | expectancy_net | total_return | max_drawdown_pct | cvar_5 | profit_factor | min_split_trades | support_ok | fatal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 1200.000000 | 1200.000000 | -0.001073 | -1.000000 | -1.000000 | -0.002456 | 0.496960 | 240.000000 | 1.000000 | 1.000000 |
| 0.250000 | 1200.000000 | 1200.000000 | -0.001044 | -1.000000 | -1.000000 | -0.002200 | 0.506670 | 240.000000 | 1.000000 | 1.000000 |
| 0.500000 | 1200.000000 | 1200.000000 | -0.001044 | -1.000000 | -1.000000 | -0.002200 | 0.506670 | 240.000000 | 1.000000 | 1.000000 |
