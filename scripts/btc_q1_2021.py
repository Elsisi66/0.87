# 0.87/scripts/btc_q1_2021.py

import pandas as pd

from src.bot087.datafeed.cache_loader import load_full_parquet
from src.bot087.strategy.param_store import load_active_params
from src.bot087.backtest.engine import run_backtest_long_only

df = load_full_parquet("BTCUSDT", tf="1h")

# Q1 2021 window (fast iteration)
df = df[
    (df["Timestamp"] >= pd.to_datetime("2017-01-01", utc=True))
    & (df["Timestamp"] < pd.to_datetime("2025-12-01", utc=True))
].reset_index(drop=True)

p = load_active_params("BTCUSDT")

trades, metrics = run_backtest_long_only(
    df,
    symbol="BTCUSDT",
    p=p,
    initial_equity=10_000.0,
    fee_bps=7.0,
)

print(metrics)
print("n_trades:", len(trades))
print("first trade:", trades[0] if trades else None)
print("last trade:", trades[-1] if trades else None)
