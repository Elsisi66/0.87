# Repro

- Generated UTC: 2026-02-21T21:35:18.942165+00:00
- Seed: 42
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342`

## Fresh Run

```bash
python3 -m src.execution.ga_exec_3m_opt \
  --symbol SOLUSDT \
  --mode normal \
  --pop 24 \
  --gens 2 \
  --workers 1 \
  --seed 42 \
  --export-topk 20 \
  --execution-config configs/execution_configs.yaml \
  --timeframe 3m \
  --pre-buffer-hours 6.0 \
  --exec-horizon-hours 12.0 \
  --max-signals 2000
```

## Resume

```bash
python3 -m src.execution.ga_exec_3m_opt --resume /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342
```
