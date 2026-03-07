# Phase AB Risk Controls Spec

- Generated UTC: 2026-02-23T11:00:32.677091+00:00
- Scope: structural controls in execution engine only; no session/killzone veto overlays.
- Hard gates: unchanged.

## AA Evidence Used

- max_consecutive_losses: `33`
- streak>=5 count: `20`
- sl_loss_share: `0.884618`
- conditional_loss_rate_after_loss: `0.909938`

## Control Family

### C1_cooldown_decluster

- Description: Increase deterministic cooldown to reduce bursty consecutive entries.
- Default params for AC ablation: `{"cooldown_min": 45}`
- Expected impact channels: `max_consecutive_losses, streak_ge5_count`
- Risks: `lower participation, possible entry-rate pressure`

### C2_break_even_early

- Description: Arm break-even earlier to cut SL-heavy reversals without session veto.
- Default params for AC ablation: `{"break_even_enabled": 1, "break_even_offset_bps": 0.0, "break_even_trigger_r": 0.55}`
- Expected impact channels: `sl_loss_share, cvar_5, max_drawdown`
- Risks: `premature stop-outs, reduced winner tail`

### C3_trailing_tail_guard

- Description: Enable earlier trailing protection to cap adverse reversals after favorable excursion.
- Default params for AC ablation: `{"trail_start_r": 1.0, "trail_step_bps": 12.0, "trailing_enabled": 1}`
- Expected impact channels: `cvar_5, max_drawdown, loss_run_ge3_count`
- Risks: `winner truncation, higher exit churn`

### C4_time_stop_tighten

- Description: Tighten long horizon time-stop to reduce slow-bleed losses and long exposure tails.
- Default params for AC ablation: `{"time_stop_min": 1026}`
- Expected impact channels: `sl_loss_share, max_consecutive_losses, max_drawdown`
- Risks: `cuts late recoveries, can lower expectancy if too tight`
