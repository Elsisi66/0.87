from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from src.bot087.strategy.params import StrategyParams


def get_cycle(row: pd.Series) -> int:
    """
    Temporary: if your dataframe already has 'cycle' use it, otherwise default to 1.
    (We will plug your real cycle/regime logic next.)
    """
    for k in ("cycle", "Cycle", "CYCLE"):
        if k in row.index and pd.notna(row[k]):
            try:
                return int(row[k])
            except Exception:
                pass
    return 1  # default so you can test end-to-end TODAY


def _ema_col(span: int) -> str:
    return f"EMA_{int(span)}"


def can_enter_long(row: pd.Series, p: StrategyParams) -> bool:
    # hour filter (if provided)
    ts = row["Timestamp"]
    if p.allow_hours and int(ts.hour) not in set(p.allow_hours):
        return False

    cycle = get_cycle(row)
    if cycle not in set(p.trade_cycles):
        return False

    # RSI band with buffer
    rsi = float(row["RSI"])
    lo = p.entry_rsi_min - p.entry_rsi_buffer
    hi = p.entry_rsi_max + p.entry_rsi_buffer
    if not (lo <= rsi <= hi):
        return False

    # WILLR: must be between floor and cycle threshold
    willr = float(row["WILLR"])
    willr_thr = float(p.willr_by_cycle[cycle])
    if not (p.willr_floor <= willr <= willr_thr):
        return False

    # EMA alignment (EMA_span > EMA_trend_long)
    if p.ema_align:
        c_fast = _ema_col(p.ema_span)
        c_slow = _ema_col(p.ema_trend_long)
        if c_fast not in row.index or c_slow not in row.index:
            return False
        if float(row[c_fast]) <= float(row[c_slow]):
            return False

    # EMA200 slope (optional)
    if p.require_ema200_slope:
        slope = row.get("EMA_200_SLOPE", None)
        if slope is None or pd.isna(slope) or float(slope) <= 0:
            return False

    return True


def exit_levels(entry_px: float, cycle: int, p: StrategyParams) -> Tuple[float, float]:
    tp = entry_px * float(p.tp_mult_by_cycle[cycle])
    sl = entry_px * float(p.sl_mult_by_cycle[cycle])
    return tp, sl


def should_exit_long(
    row: pd.Series,
    entry_px: float,
    entry_ts: pd.Timestamp,
    p: StrategyParams,
) -> Optional[str]:
    cycle = get_cycle(row)

    # TP/SL by cycle
    tp_px, sl_px = exit_levels(entry_px, cycle, p)
    px = float(row["Close"])
    if px >= tp_px:
        return "tp"
    if px <= sl_px:
        return "sl"

    # RSI exit (we'll refine direction later if needed)
    rsi = float(row["RSI"])
    if rsi <= float(p.exit_rsi_by_cycle[cycle]):
        return "rsi_exit"

    # max hold
    hold_hours = (row["Timestamp"] - entry_ts).total_seconds() / 3600.0
    if hold_hours >= float(p.max_hold_hours):
        return "max_hold"

    return None
