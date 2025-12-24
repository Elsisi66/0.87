from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StrategyParams:
    # entry RSI window constraints
    entry_rsi_min: float
    entry_rsi_max: float
    entry_rsi_buffer: float

    # WILLR
    willr_floor: float
    willr_max: float
    willr_by_cycle: List[float]  # len=5

    # EMA / trend
    ema_span: int
    ema_trend_long: int
    ema_align: bool
    require_ema200_slope: bool

    # risk / exits
    profit_target_mult: float
    stop_loss_mult: float
    max_hold_hours: int

    tp_mult_by_cycle: List[float]  # len=5
    sl_mult_by_cycle: List[float]  # len=5
    exit_rsi_by_cycle: List[float]  # len=5

    risk_per_trade: float
    max_allocation: float
    atr_k: float

    use_vol_filter: bool
    vol_tail_percentile: float

    allow_hours: List[int]

    # which cycles are tradable (your rule)
    trade_cycles: List[int]
