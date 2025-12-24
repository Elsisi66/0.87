import pandas as pd

from src.bot087.strategy.param_store import load_active_params
from src.bot087.strategy.logic import can_enter_long
from src.bot087.datafeed.cache_loader import load_full_parquet

df = load_full_parquet("BTCUSDT", tf="1h")

# take a small slice to keep it fast
df = df[(df["Timestamp"] >= pd.to_datetime("2021-01-01", utc=True)) &
        (df["Timestamp"] <  pd.to_datetime("2021-03-01", utc=True))].reset_index(drop=True)

p = load_active_params("BTCUSDT")

hits = []
for i in range(len(df)):
    if can_enter_long(df.iloc[i], p):
        hits.append(df.iloc[i]["Timestamp"])

print("bars checked:", len(df))
print("entry hits:", len(hits))
print("first 10 hits:", hits[:10])
