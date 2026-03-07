# Decision Phase M2 Pilot

- Generated UTC: 2026-02-22T19:15:05.950071+00:00
- Decision: **GO**

## Evidence

- valid_for_ranking_count: 18 (Phase L=0, Phase M=0)
- duplicate_rate improved to 0.1889 from 0.5208/0.5167
- effective_trials_proxy improved to 147.0 from 24.0/30.0
- entry-rate slack p90 (vs 0.97) moved to -0.0003 from -0.0811/-0.0667

## Guardrails for Phase M2

- Keep hard gates unchanged.
- Keep frozen subset/fee/metrics hashes unchanged.
- Keep repaired sampler + telemetry + dedupe controls enabled.
- Use small pilot budget first; stop if valid_for_ranking returns to zero over multiple generations.
