from src.bot087.datafeed.dataset import load_build_cache

df, rep = load_build_cache("BTCUSDT", start="2017-08-01", end="2025-12-01", tf="1h", cache=True)

print(rep)
print(df[["Timestamp", "Close", "EMA_200", "RSI", "WILLR"]].head(3))
print(df[["Timestamp", "Close", "EMA_200", "RSI", "WILLR"]].tail(3))
