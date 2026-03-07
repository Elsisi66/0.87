# R2 Revalidation Report

- Generated UTC: 2026-02-28T00:54:29.565433+00:00
- Candidate count (incl baseline): `5`
- Repaired route count: `3`
- Route-feasible routes: `3` / `3`
- Surviving non-baseline candidates (valid + route-pass + positive delta): `0`

## Revalidation Results

| candidate_id | candidate_type | valid_for_ranking | expectancy_net | delta_expectancy_vs_baseline | cvar_improve_ratio | maxdd_improve_ratio | route_pass | min_subperiod_delta | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASELINE_FROZEN | baseline_reference | 1 | -0.0008378556817 | 0 | 0 | 0 | 1 | 0 |  |
| E1 | historical_exec | 1 | 5.649190711e-05 | 0.0008943475888 | 0.1908480812 | 0.5802892197 | 0 | 0.0008237879328 |  |
| E2 | historical_exec | 1 | 5.552012202e-05 | 0.0008933758037 | 0.1820117438 | 0.5795036682 | 0 | 0.0008237879328 |  |
| REGIME_ROUTED_EXEC_mid_00 | nx_variant | 1 | -0.000421832692 | 0.0004160229898 | 0.1820117438 | 0.4440767163 | 0 | -0.0004743130023 |  |
| PASSIVE_LADDER_ADAPTIVE_conservative_03 | nx_variant | 1 | -0.0004658096944 | 0.0003720459873 | 0.29688413 | 0.3945810244 | 0 | -0.0003948739808 |  |
