# Instant Loser Vs Winner Entry Forensics

- Generated UTC: `2026-03-03T01:25:26.718010+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Repaired multicoin source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Foundation source: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`

## Bucket Definitions

- `instant_loser`: valid trade, stop-loss exit, hold <= 60 minutes.
- `fast_loser`: valid trade, stop-loss exit, 60 < hold <= 240 minutes.
- `meaningful_winner`: valid trade, net pnl >= 0.0020 (20 bps).
- `neutral_small_win`: all other valid trades.

## Live-Coin Bucket Counts

| symbol | live_status | best_candidate_id | total_trades | instant_losers | fast_losers | meaningful_winners | neutral_small_win |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DOGEUSDT | shadow | M2_ENTRY_ONLY_MORE_PASSIVE | 415 | 290 | 43 | 51 | 31 |
| LINKUSDT | approved | M3_ENTRY_ONLY_FASTER | 290 | 194 | 46 | 24 | 26 |
| LTCUSDT | shadow | M2_ENTRY_ONLY_MORE_PASSIVE | 282 | 210 | 35 | 18 | 19 |
| NEARUSDT | shadow | M1_ENTRY_ONLY_PASSIVE_BASELINE | 410 | 297 | 48 | 43 | 22 |

## Strongest Separating Features

| symbol | top_feature_1_call | top_feature_2_call | top_feature_3_call | decision |
| --- | --- | --- | --- | --- |
| DOGEUSDT | winners show better first-2-bar close return | shallower first-2-bar adverse excursion | winners show better first eligible-bar close return | FILTER_AND_GIVE_ROOM |
| LINKUSDT | winners show better first-2-bar close return | shallower first-2-bar adverse excursion | winners show better first eligible-bar close return | FILTER_AND_GIVE_ROOM |
| LTCUSDT | shallower first-2-bar adverse excursion | shallower first eligible-bar adverse excursion | winners show better first-2-bar close return | FILTER_AND_GIVE_ROOM |
| NEARUSDT | shallower first-2-bar adverse excursion | shallower first eligible-bar adverse excursion | winners show better first eligible-bar close return | GIVE_MORE_INITIAL_ROOM |

## Repair Decisions

| symbol | decision | bounded_follow_up_repair_set |
| --- | --- | --- |
| DOGEUSDT | FILTER_AND_GIVE_ROOM | test 1-2 entry filters first, then a single conditional wider-initial-risk rule for filtered passes |
| LINKUSDT | FILTER_AND_GIVE_ROOM | test 1-2 entry filters first, then a single conditional wider-initial-risk rule for filtered passes |
| LTCUSDT | FILTER_AND_GIVE_ROOM | test 1-2 entry filters first, then a single conditional wider-initial-risk rule for filtered passes |
| NEARUSDT | GIVE_MORE_INITIAL_ROOM | test one conditional wider-initial-risk rule on high-quality entries only |