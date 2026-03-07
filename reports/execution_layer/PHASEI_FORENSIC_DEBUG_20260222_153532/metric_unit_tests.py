import numpy as np
from scripts import phase_i_sol_signal_fork as p1

# Run: analysis/0.87/venv/bin/python metric_unit_tests.py

vec_id = np.array([-0.01] * 100, dtype=float)
print('cvar_identical', p1._compute_cvar5_trade_notional(vec_id))
vec_mix = np.array([0.01] * 95 + [-0.2] * 5, dtype=float)
print('cvar_mixed', p1._compute_cvar5_trade_notional(vec_mix))
vec_ruin = np.array([-0.5, -1.2, 0.1], dtype=float)
print('geom_legacy', p1._geom_mean_return(vec_ruin))
print('geom_clean', p1._geom_mean_return_with_ruin_flag(vec_ruin))
