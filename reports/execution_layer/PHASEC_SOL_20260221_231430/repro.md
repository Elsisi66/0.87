# Repro

```bash
python3 scripts/phase_c_sol_runner.py \
  --symbol SOLUSDT \
  --phase-a-contract-dir /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310 \
  --signals-csv /root/analysis/0.87/data/signals/SOLUSDT_signals_1h.csv \
  --max-signals 2000 \
  --wf-splits 5 \
  --coarse-max-configs 1200 \
  --refine-max-configs 1500 \
  --coarse-seed 42 \
  --refine-seed 314159 \
  --workers 3
```
