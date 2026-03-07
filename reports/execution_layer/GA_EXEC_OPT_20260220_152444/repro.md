# Repro

- Generated UTC: 2026-02-20T15:24:46.114096+00:00
- Seed: 11
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152444`

## Fresh Run

```bash
python3 -m src.execution.ga_exec_3m_opt \
  --symbol SOLUSDT \
  --mode tight \
  --pop 4 \
  --gens 1 \
  --workers 3 \
  --seed 11 \
  --export-topk 2 \
  --execution-config configs/execution_configs.yaml \
  --timeframe 3m \
  --pre-buffer-hours 6.0 \
  --exec-horizon-hours 12.0 \
  --max-signals 40
```

## Resume

```bash
python3 -m src.execution.ga_exec_3m_opt --resume /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152444
```
