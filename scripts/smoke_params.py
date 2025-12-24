from src.bot087.strategy.param_store import load_active_params

sp = load_active_params("BTCUSDT")
print(sp)
print("willr_by_cycle:", sp.willr_by_cycle)
print("tp_mult_by_cycle:", sp.tp_mult_by_cycle)
print("sl_mult_by_cycle:", sp.sl_mult_by_cycle)
print("exit_rsi_by_cycle:", sp.exit_rsi_by_cycle)
print("trade_cycles:", sp.trade_cycles)
