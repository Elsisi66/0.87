# Intrabar Ambiguity Sensitivity

- Scope: frozen Phase C test universe, 1h-bar approximation around 3m-entry references.
- `optimistic`: TP precedence on same-bar TP/SL touch.
- `pessimistic`: SL precedence on same-bar TP/SL touch.
- `neutral`: deterministic candle-direction tie-break.

| mode | expectancy_net | total_return | max_drawdown_pct | cvar_5 | delta_expectancy_vs_intrabar3m | delta_total_return_vs_intrabar3m | delta_maxdd_vs_intrabar3m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| intrabar_3m_reference | -0.000649 | -0.996799 | -0.999175 | -0.002200 | 0.000000 | 0.000000 | 0.000000 |
| bar_1h_optimistic | -0.000649 | -0.996799 | -0.999175 | -0.002200 | 0.000000 | 0.000000 | 0.000000 |
| bar_1h_neutral | -0.000649 | -0.996799 | -0.999175 | -0.002200 | 0.000000 | 0.000000 | 0.000000 |
| bar_1h_pessimistic | -0.000649 | -0.996799 | -0.999175 | -0.002200 | 0.000000 | 0.000000 | 0.000000 |
