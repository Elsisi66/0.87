# Phase G2 Diagnostics Report

- Generated UTC: 2026-02-23T22:02:30.720760+00:00
- Valid rows used: `850`

## Legacy Score Correlation vs Post-Exec PnL

| exec_choice_id | spearman_legacy_vs_metric | support_n |
| --- | --- | --- |
| E1 | 0.037436516 | 425 |
| E2 | 0.035660782 | 425 |

## Feature Stability (sample)

| exec_choice_id | feature | outcome | overall_spearman | stable_sign_frac | split_count |
| --- | --- | --- | --- | --- | --- |
| E1 | atr_percentile_1h | pnl_net_trade_notional_dec | 0.0081556641 | 0.4 | 5 |
| E1 | atr_percentile_1h | y_cluster_loss | 0.11122795 | 0.8 | 5 |
| E1 | atr_percentile_1h | y_tail_loss | -0.12554894 | 0.8 | 5 |
| E1 | atr_percentile_1h | y_toxic_trade | 0.10479324 | 0.8 | 5 |
| E1 | f1h_adx | pnl_net_trade_notional_dec | 0.061334147 | 0.6 | 5 |
| E1 | f1h_adx | y_cluster_loss | -0.085826373 | 0.6 | 5 |
| E1 | f1h_adx | y_tail_loss | -0.028215435 | 0.6 | 5 |
| E1 | f1h_adx | y_toxic_trade | -0.11078917 | 0.6 | 5 |
| E1 | f1h_ema_200_slope | pnl_net_trade_notional_dec | 0.048785549 | 0.6 | 5 |
| E1 | f1h_ema_200_slope | y_cluster_loss | 0.11648781 | 0.8 | 5 |
| E1 | f1h_ema_200_slope | y_tail_loss | -0.074653338 | 0.8 | 5 |
| E1 | f1h_ema_200_slope | y_toxic_trade | 0.074404536 | 0.8 | 5 |
| E1 | f1h_rsi | pnl_net_trade_notional_dec | -0.008389637 | 0.8 | 5 |
| E1 | f1h_rsi | y_cluster_loss | -0.0073125609 | 0.4 | 5 |
| E1 | f1h_rsi | y_tail_loss | 0.047515576 | 0.6 | 5 |
| E1 | f1h_rsi | y_toxic_trade | -0.024392696 | 0.4 | 5 |
| E1 | f1h_rv_24 | pnl_net_trade_notional_dec | 0.063300377 | 0.6 | 5 |
| E1 | f1h_rv_24 | y_cluster_loss | -0.1763995 | 0.8 | 5 |
| E1 | f1h_rv_24 | y_tail_loss | -0.074653338 | 0.8 | 5 |
| E1 | f1h_rv_24 | y_toxic_trade | -0.13122998 | 0.8 | 5 |
