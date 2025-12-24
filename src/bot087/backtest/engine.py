# src/bot087/backtest/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.bot087.strategy.params import StrategyParams
from src.bot087.strategy.logic import can_enter_long, should_exit_long, get_cycle, exit_levels


@dataclass
class Trade:
    symbol: str
    cycle: int
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: pd.Timestamp
    exit_px: float
    size: float
    reason: str
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    net_pnl: float
    hold_hours: float


def _fee(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10000.0)


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        if peak > 0:
            dd = (peak - x) / peak
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def run_backtest_long_only(
    df: pd.DataFrame,
    symbol: str,
    p: StrategyParams,
    initial_equity: float = 10_000.0,
    fee_bps: float = 7.0,
    two_candle_confirm: bool = True,
) -> Tuple[List[Trade], Dict[str, float]]:
    """
    Long-only. One position at a time.
    Entry/exit on Close.
    Includes 2-candle confirmation for entries when enabled.
    """
    if df.empty:
        return [], {
            "initial_equity": float(initial_equity),
            "final_equity": float(initial_equity),
            "net_profit": 0.0,
            "trades": 0.0,
            "win_rate_pct": 0.0,
            "avg_net_pnl": 0.0,
            "max_dd_pct": 0.0,
        }

    equity = float(initial_equity)
    equity_curve: List[float] = [equity]

    in_pos = False
    entry_ts: Optional[pd.Timestamp] = None
    entry_px: float = 0.0
    entry_cycle: int = 1
    size: float = 0.0
    entry_fee_paid: float = 0.0

    trades: List[Trade] = []
    exit_reasons: Dict[str, int] = {}

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["Timestamp"]
        px = float(row["Close"])

        # mark-to-market curve
        equity_curve.append(equity)

        if not in_pos:
            # 2-candle confirmation: require previous bar AND current bar to pass entry
            if two_candle_confirm:
                if i == 0:
                    continue
                prev_row = df.iloc[i - 1]
                if not (can_enter_long(prev_row, p) and can_enter_long(row, p)):
                    continue
            else:
                if not can_enter_long(row, p):
                    continue

            entry_cycle = int(get_cycle(row))
            entry_ts = ts
            entry_px = px

            # cycle-aware TP/SL determines distance used for sizing
            tp_px, sl_px = exit_levels(entry_px, entry_cycle, p)
            risk_per_unit = max(entry_px - sl_px, 1e-9)

            risk_budget = equity * float(p.risk_per_trade)
            raw_units = risk_budget / risk_per_unit

            max_notional = equity * float(p.max_allocation)
            cap_units = max_notional / entry_px

            size = max(0.0, min(raw_units, cap_units))
            if size <= 0.0:
                continue

            entry_fee_paid = _fee(size * entry_px, fee_bps)
            equity -= entry_fee_paid

            in_pos = True
            continue

        # ---- in position ----
        assert entry_ts is not None

        reason = should_exit_long(row, entry_px=entry_px, entry_ts=entry_ts, p=p)
        if reason is None:
            continue

        exit_ts = ts
        exit_px = px

        gross_pnl = (exit_px - entry_px) * size
        exit_fee = _fee(size * exit_px, fee_bps)
        net_pnl = gross_pnl - entry_fee_paid - exit_fee  # full round-trip net

        # equity update: entry fee already deducted at entry, so add (gross - exit_fee)
        equity += (gross_pnl - exit_fee)

        hold_hours = (exit_ts - entry_ts).total_seconds() / 3600.0

        trades.append(
            Trade(
                symbol=symbol.upper(),
                cycle=entry_cycle,
                entry_ts=entry_ts,
                entry_px=entry_px,
                exit_ts=exit_ts,
                exit_px=exit_px,
                size=size,
                reason=reason,
                gross_pnl=gross_pnl,
                entry_fee=entry_fee_paid,
                exit_fee=exit_fee,
                net_pnl=net_pnl,
                hold_hours=hold_hours,
            )
        )

        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # flat
        in_pos = False
        entry_ts = None
        entry_px = 0.0
        size = 0.0
        entry_cycle = 1
        entry_fee_paid = 0.0

    # metrics
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = (wins / n * 100.0) if n else 0.0
    total_net = sum(t.net_pnl for t in trades)

    max_dd = _max_drawdown_pct(equity_curve)

    metrics: Dict[str, float] = {
        "initial_equity": float(initial_equity),
        "final_equity": float(equity),
        "net_profit": float(equity - float(initial_equity)),
        "trades": float(n),
        "win_rate_pct": float(win_rate),
        "avg_net_pnl": float(total_net / n) if n else 0.0,
        "max_dd_pct": float(max_dd),
    }

    for k, v in exit_reasons.items():
        metrics[f"exit_{k}_count"] = float(v)

    return trades, metrics
