import pandas as pd

from src.bot087.datafeed.loader import load_symbol_history
from src.bot087.features.build import build_features

df = load_symbol_history("BTCUSDT", start="2017-12-20", end="2018-01-10", keep_extra_columns=True)

# Keep the old values from the file
old = df[["Timestamp", "EMA_200", "RSI"]].copy()

# Recompute continuous-history indicators
df2 = build_features(df)

cmp = old.merge(df2[["Timestamp", "EMA_200", "RSI"]], on="Timestamp", suffixes=("_old", "_new"))
cmp["EMA200_diff"] = (cmp["EMA_200_new"] - cmp["EMA_200_old"]).abs()
cmp["RSI_diff"] = (cmp["RSI_new"] - cmp["RSI_old"]).abs()

print("max EMA200 diff:", cmp["EMA200_diff"].max())
print("max RSI diff:", cmp["RSI_diff"].max())

# show a few rows around 2018-01-01
mask = (cmp["Timestamp"] >= pd.to_datetime("2017-12-31", utc=True)) & (cmp["Timestamp"] <= pd.to_datetime("2018-01-02", utc=True))
print(cmp.loc[mask, ["Timestamp", "EMA_200_old", "EMA_200_new", "RSI_old", "RSI_new"]].head(20))
