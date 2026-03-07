# Phase E4 Smoke Report

- Smoke anchor start_from_bar_ts: `2025-11-10T16:00:00+00:00`
- Latest closed 1h bar processed: `2025-12-31T23:00:00+00:00`
- Market meta: `{'one_h_source': 'local', 'one_h_degraded': 0, 'three_m_source': 'local', 'three_m_degraded': 0, 'fx_source': 'fallback_1_to_1'}`
- Telegram probe: `telegram_error:HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: /bot8305***og/sendMessage (Caused by NameResolutionError("HTTPSConnection(host='api.telegram.org', port=443): Failed to resolve 'api.telegram.org' ([Errno -2] Name or service not known)"))`
- Cycle summary JSON: `/root/analysis/0.87/reports/execution_layer/PHASEE_MODEL_A_PAPER_RUNTIME_20260228_030145/model_a_cycle_summary_20260228_030148.json`
- Cycle summary MD: `/root/analysis/0.87/reports/execution_layer/PHASEE_MODEL_A_PAPER_RUNTIME_20260228_030145/model_a_cycle_summary_20260228_030148.md`

## paper_primary
- Candidate: `M3_ENTRY_ONLY_FASTER_C_WIN_02`
- Signals seen: `96`
- Entries attempted / filled: `4` / `4`
- Exits processed: `4`
- Runtime errors: `0`
- Recovery actions: `0`
- Open positions: `0`
- Realized PnL EUR: `-2.2432403335539397`
- Taker share: `0.0`
- Avg / P95 fill delay: `0.0` / `0.0`
- State consistent: `1`

## shadow_backup
- Candidate: `M2_ENTRY_ONLY_MORE_PASSIVE_NOFB_C_FB_ON`
- Signals seen: `96`
- Entries attempted / filled: `4` / `4`
- Exits processed: `4`
- Runtime errors: `0`
- Recovery actions: `0`
- Open positions: `0`
- Realized PnL EUR: `-2.2432403335540068`
- Taker share: `0.0`
- Avg / P95 fill delay: `0.0` / `0.0`
- State consistent: `1`

## overall
- Candidate: `ALL`
- Signals seen: `192`
- Entries attempted / filled: `8` / `8`
- Exits processed: `8`
- Runtime errors: `0`
- Recovery actions: `0`
- Open positions: `0`
- Realized PnL EUR: `-4.486480667107946`
- Taker share: `0.0`
- Avg / P95 fill delay: `0.0` / `0.0`
- State consistent: `1`
