# S2 Mapping To Existing 1h Engine

- Generated UTC: 2026-02-28T01:22:35.845695+00:00
- Existing 1h optimization engine: `src/bot087/optim/ga.py`.
- This branch does not replace that engine; it specifies how to change its objective / labeling layer:
  - add execution-aware post-score penalties as additional fitness terms
  - enforce route-balance/support minimum constraints before accepting a candidate
  - include center-route anti-collapse penalties in validation score
  - use repaired-route slices as mandatory validation cohorts
- In this run, the spec is benchmarked as deterministic prototypes over the frozen subset before spending GA compute.
