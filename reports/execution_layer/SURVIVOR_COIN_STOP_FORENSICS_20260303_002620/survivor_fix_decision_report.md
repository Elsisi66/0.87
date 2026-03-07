# Survivor-Coin Stop Forensics

- Generated UTC: 2026-03-03T00:30:20.296436+00:00
- Frozen repaired multicoin source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Frozen repaired 1h baseline source: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Repaired contract guards:
  - `scripts/backtest_exec_phasec_sol.py:247` uses `defer_exit_to_next_bar=True`
  - `scripts/backtest_exec_phasec_sol.py:316` starts 1h exit evaluation at `idx + 1`
  - `scripts/phase_a_model_a_audit.py:423` uses `searchsorted(..., side="right")` after fill
  - `scripts/phase_a_model_a_audit.py:445` anchors stop to `fill_price * sl_mult_sig`
  - `scripts/execution_layer_3m_ict.py:780-805` uses wick-touch triggers and resolves same-bar SL/TP conflicts in favor of SL
  - `paper_trading/app/model_a_runtime.py:672-674` blocks same-parent-bar exit checks after fill

## Per-Coin Summary

| symbol | total_trades | trades_exit_within_3h | pct_exit_within_3h | trades_exit_within_4h | pct_exit_within_4h | main_short_hold_exit_reason | root_cause_classification | recommended_action | mean_hold_min | median_hold_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | 501 | 409 | 0.8163672655 | 425 | 0.8483033932 | sl | SHARED_STOP_DEFINITION_BUG | PATCH_GLOBAL_STOP_LOGIC | 138.3113772 | 60 |
| DOGEUSDT | 415 | 323 | 0.778313253 | 336 | 0.8096385542 | sl | STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT | MOVE_TO_SHADOW_ONLY | 172.5903614 | 60 |
| LINKUSDT | 290 | 230 | 0.7931034483 | 240 | 0.8275862069 | sl | STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT | KEEP_AS_IS | 165.7241379 | 60 |
| LTCUSDT | 282 | 237 | 0.8404255319 | 245 | 0.8687943262 | sl | STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT | MOVE_TO_SHADOW_ONLY | 140.2553191 | 60 |
| NEARUSDT | 410 | 342 | 0.8341463415 | 347 | 0.8463414634 | sl | STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT | MOVE_TO_SHADOW_ONLY | 155.5463415 | 60 |
| SOLUSDT | 1357 | 1087 | 0.8010316875 | 1119 | 0.8246131172 | sl | STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT | DISABLE_COIN | 170.5512159 | 60 |

## Decision Layer

- universal_stop_issue_scope: `universal`
- approved_keep_as_is: `['LINKUSDT']`
- shadow_only: `['DOGEUSDT', 'LTCUSDT', 'NEARUSDT']`
- disabled: `['SOLUSDT']`