from __future__ import annotations

from dataclasses import dataclass
from typing import List


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class Bounds:
    # RSI entry clamps
    entry_rsi_min: tuple[float, float] = (30.0, 70.0)
    entry_rsi_max: tuple[float, float] = (35.0, 85.0)
    entry_rsi_buffer: tuple[float, float] = (0.0, 12.0)

    # WILLR per cycle
    willr: tuple[float, float] = (-100.0, -1.0)

    # TP/SL multipliers
    tp_mult: tuple[float, float] = (1.005, 1.20)
    sl_mult: tuple[float, float] = (0.80, 0.999)

    # Exit RSI per cycle
    exit_rsi: tuple[float, float] = (35.0, 85.0)

    # Hold
    max_hold_hours: tuple[int, int] = (6, 80)


B = Bounds()
