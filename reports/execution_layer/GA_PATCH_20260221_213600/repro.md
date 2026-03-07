# Repro

```bash
cd /root/analysis/0.87
.venv/bin/python -m src.execution.ga_exec_3m_opt \
  --symbol SOLUSDT \
  --signals-csv data/signals/SOLUSDT_signals_1h.csv \
  --max-signals 2000 \
  --mode normal \
  --force-no-skip 1 \
  --pop 24 \
  --gens 2 \
  --workers 1 \
  --seed 42 \
  --hard-max-taker-share 1.0 \
  --hard-max-median-fill-delay-min 180 \
  --hard-max-p95-fill-delay-min 360 \
  --outdir reports/execution_layer
```
