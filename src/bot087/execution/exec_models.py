from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FillResult:
    fill_raw: float
    first_open: float
    vwap_proxy: float
    high_max: float
    low_min: float
    volume_sum: float
    bars: int


def _window(sec_df: pd.DataFrame, ts, window_sec: int) -> pd.DataFrame:
    start = pd.Timestamp(ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    start = start.tz_convert("UTC")
    end = start + pd.Timedelta(seconds=int(window_sec))
    x = sec_df[(sec_df["Timestamp"] >= start) & (sec_df["Timestamp"] < end)]
    return x.reset_index(drop=True)


def _vwap_proxy(x: pd.DataFrame) -> Tuple[float, float]:
    if x.empty:
        return np.nan, 0.0
    typ = (x["High"].astype(float) + x["Low"].astype(float) + x["Close"].astype(float)) / 3.0
    if "Volume" in x.columns:
        vol = x["Volume"].astype(float).fillna(0.0)
    else:
        vol = pd.Series(np.zeros(len(x)), index=x.index, dtype=float)

    v_sum = float(vol.sum())
    if v_sum > 0:
        vwap = float((typ * vol).sum() / v_sum)
    else:
        vwap = float(x["Close"].iloc[0])
    return vwap, v_sum


def fill_default(sec_df: pd.DataFrame, ts, side: str, window_sec: int, fallback_first_open: float) -> FillResult:
    x = _window(sec_df, ts, window_sec)
    if x.empty:
        return FillResult(
            fill_raw=float(fallback_first_open),
            first_open=float(fallback_first_open),
            vwap_proxy=float(fallback_first_open),
            high_max=float(fallback_first_open),
            low_min=float(fallback_first_open),
            volume_sum=0.0,
            bars=0,
        )

    first_open = float(x["Open"].iloc[0])
    vwap, v_sum = _vwap_proxy(x)
    high_max = float(x["High"].max())
    low_min = float(x["Low"].min())

    s = side.lower()
    if s == "buy":
        fill = max(first_open, float(vwap))
    elif s == "sell":
        fill = min(first_open, float(vwap))
    else:
        raise ValueError(f"Unknown side: {side}")

    return FillResult(
        fill_raw=float(fill),
        first_open=first_open,
        vwap_proxy=float(vwap),
        high_max=high_max,
        low_min=low_min,
        volume_sum=float(v_sum),
        bars=int(len(x)),
    )


def fill_stress(sec_df: pd.DataFrame, ts, side: str, window_sec: int, fallback_first_open: float) -> FillResult:
    x = _window(sec_df, ts, window_sec)
    if x.empty:
        return FillResult(
            fill_raw=float(fallback_first_open),
            first_open=float(fallback_first_open),
            vwap_proxy=float(fallback_first_open),
            high_max=float(fallback_first_open),
            low_min=float(fallback_first_open),
            volume_sum=0.0,
            bars=0,
        )

    first_open = float(x["Open"].iloc[0])
    vwap, v_sum = _vwap_proxy(x)
    high_max = float(x["High"].max())
    low_min = float(x["Low"].min())

    s = side.lower()
    if s == "buy":
        fill = high_max
    elif s == "sell":
        fill = low_min
    else:
        raise ValueError(f"Unknown side: {side}")

    return FillResult(
        fill_raw=float(fill),
        first_open=first_open,
        vwap_proxy=float(vwap),
        high_max=high_max,
        low_min=low_min,
        volume_sum=float(v_sum),
        bars=int(len(x)),
    )


def fill_model(sec_df: pd.DataFrame, ts, side: str, window_sec: int, fallback_first_open: float, model: str) -> FillResult:
    m = model.lower()
    if m == "default":
        return fill_default(sec_df, ts, side, window_sec, fallback_first_open)
    if m == "stress":
        return fill_stress(sec_df, ts, side, window_sec, fallback_first_open)
    raise ValueError(f"Unsupported execution model: {model}")
