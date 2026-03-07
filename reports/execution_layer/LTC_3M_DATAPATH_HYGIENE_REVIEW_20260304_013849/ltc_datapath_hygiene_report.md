# LTC 3m Data-Path Hygiene Review

This is a strict LTC-only data-path hygiene review. It does not change the winning 3m config or strategy logic.

## Code Paths Used
- `scripts/repaired_universe_3m_exec_subset1_confirm.py` for window-building and repaired-branch signal reconstruction
- `scripts/phase_v_multicoin_model_a_audit.py` for bundle construction and route-enabled 3m evaluation

## Core Finding
- The current `940/940` partial label is driven by the merged-window metadata path: merged `coverage_ratio=0.998504292111 < 1.0`, so every non-empty slice is labeled partial by the current builder.
- Fresh merged LTC parquet and prior cached LTC parquet are identical: `1`
- True exact slice completeness is not fully clean: `928` full slices, `12` partial slices

## Winner Comparison
| path_id | download_source | coverage_ratio | winner_candidate_id | winner_expectancy | winner_delta_vs_1h | winner_route_pass | signals_partial_3m_data | signals_missing_3m_data |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fresh_merged_foundation | foundation_merged_3m | 0.9985042921 | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0005884738669 | 0.0004895665814 | 1 | 940 | 0 |
| prior_cached_subset1 | prior_subset1_merged_3m | 1 | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0005884738669 | 0.0004895665814 | 1 | 0 | 0 |

## Exact Coverage Trace
- foundation windows used: `348`
- foundation coverage mean/min/max: `0.998504292111` / `0.707692307692` / `1.000000000000`
- windows below 1.0 coverage: `6`
- true partial slices: `12`
- max missing rows in any true partial slice: `152`

## Decision
- final LTC posture: `LTC_REMAINS_BLOCKED`
- active-subset implication: `FREEZE_SOL_ONLY_AND_KEEP_LTC_BLOCKED`

## Proven vs Assumed
- Proven: the fresh merged path and the prior cached path produce identical LTC 3m parquet data and identical winning 3m metrics.
- Proven: the blanket 940/940 partial label is a conservative metadata artifact caused by applying a single merged window `coverage_ratio < 1.0` to every slice.
- Proven: despite that artifact, there are still 12 truly partial LTC signal slices under exact row-count checking.
- Assumed: exact 3m slice completeness is the operational cleanliness standard; under that standard, any non-zero true partial slice count blocks promotion.