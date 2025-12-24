import pandas as pd
import ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute indicators on the FULL concatenated history.
    Overwrites existing feature columns so year-boundary resets don't screw results.
    """
    out = df.copy()
    out = out.sort_values("Timestamp").reset_index(drop=True)

    close = out["Close"]

    # EMAs
    out["EMA_20"] = ta.trend.EMAIndicator(close, window=20, fillna=True).ema_indicator()
    out["EMA_35"] = ta.trend.EMAIndicator(close, window=35, fillna=True).ema_indicator()
    out["EMA_50"] = ta.trend.EMAIndicator(close, window=50, fillna=True).ema_indicator()
    out["EMA_120"] = ta.trend.EMAIndicator(close, window=120, fillna=True).ema_indicator()
    out["EMA_200"] = ta.trend.EMAIndicator(close, window=200, fillna=True).ema_indicator()
    out["EMA_200_SLOPE"] = out["EMA_200"].diff()

    # RSI
    out["RSI"] = ta.momentum.RSIIndicator(close, window=14, fillna=True).rsi()

    # ATR
    atr = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14, fillna=True)
    out["ATR"] = atr.average_true_range()

    # ADX + DI
    adx = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"], window=14, fillna=True)
    out["ADX"] = adx.adx()
    out["PLUS_DI"] = adx.adx_pos()
    out["MINUS_DI"] = adx.adx_neg()

    # MACD
    macd = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["MACD"] = macd.macd()
    out["MACD_SIGNAL"] = macd.macd_signal()

    # Williams %R
    out["WILLR"] = ta.momentum.WilliamsRIndicator(out["High"], out["Low"], out["Close"], lbp=14, fillna=True).williams_r()

    # Year (optional)
    out["Year"] = pd.to_datetime(out["Timestamp"], utc=True).dt.year

    return out
