# NY2 Decision Next Step

- Generated UTC: 2026-02-27T12:34:28.127253+00:00
- Route feasibility: `infeasible`
- Economic feasibility: `infeasible`
- Classification: `ROUTE_INFEASIBLE`
- Best variant net expectancy: `-0.0004218327`
- Best variant required cost reduction to net>=0: `48.081%`
- Single best next step:
  - Redesign route construction/harness so every route evaluated under walkforward has at least 200 scored test signals (or remove undersized routes from route-pass criteria); then rerun NX3 with unchanged hard gates.
