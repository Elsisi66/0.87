# Repro

- Generated UTC: 2026-02-20T15:24:37.824899+00:00
- Seed: 7
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152432`

## Fresh Run

```bash
python3 -m src.execution.ga_exec_3m_opt \
  --symbols SOLUSDT,AVAXUSDT,NEARUSDT \
  --mode tight \
  --pop 6 \
  --gens 1 \
  --workers 1 \
  --seed 7 \
  --export-topk 3 \
  --execution-config configs/execution_configs.yaml \
  --timeframe 3m \
  --pre-buffer-hours 6.0 \
  --exec-horizon-hours 12.0 \
  --max-signals 60
```

## Resume

```bash
python3 -m src.execution.ga_exec_3m_opt --resume /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152432
```
