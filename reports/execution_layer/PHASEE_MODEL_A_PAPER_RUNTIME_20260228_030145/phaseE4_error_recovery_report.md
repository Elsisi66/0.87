# Phase E4 Error Recovery Report

- Telegram probe reason: `telegram_error:HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: /bot8305***og/sendMessage (Caused by NameResolutionError("HTTPSConnection(host='api.telegram.org', port=443): Failed to resolve 'api.telegram.org' ([Errno -2] Name or service not known)"))`
- Runtime health counters: `{'api_retries': 0, 'api_failures': 0, 'signal_errors': 0, 'execution_errors': 0, 'recovery_events': 0, 'telegram_errors': 0, 'quarantined_symbols': 0, 'degraded_mode': False, 'strategy_health': 'GREEN'}`
- Coordinator dead-letter path: `/root/analysis/0.87/paper_trading/state/model_a_runtime/coordinator_dead_letter.jsonl`
- Coordinator journal path: `/root/analysis/0.87/paper_trading/state/model_a_runtime/coordinator_journal.jsonl`

## paper_primary
- Runtime errors: `0`
- Recovery actions: `0`
- State consistent: `1`
- Open positions after smoke: `0`

## shadow_backup
- Runtime errors: `0`
- Recovery actions: `0`
- State consistent: `1`
- Open positions after smoke: `0`
