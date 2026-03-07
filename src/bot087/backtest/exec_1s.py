# src/bot087/backtest/exec_1s.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

from src.bot087.optim.ga import (
    _ensure_indicators,
    _norm_params,
    build_entry_signal,
    compute_cycles,
    _shift_cycles,
    _apply_cost,
    _position_size,
)

@dataclass
class Exec1SConfig:
    # entry confirmation
    confirm_window_sec: int = 600          # watch up to 10m after hour open
    confirm_bps: float = 5.0               # require +5 bps above hour open
    abort_bps: float = 30.0                # if we dump -30 bps before confirm => skip

    # pullback fill after confirm
    pullback_bps: float = 6.0              # limit buy 6 bps below confirm price
    pullback_window_sec: int = 300         # wait up to 5m for pullback fill

    # execution choices
    market_on_no_pullback: bool = True     # if no limit fill, buy at confirm price
    use_close_for_market: bool = True      # market price uses 1s Close else 1s Open
    market_on_no_confirm: bool = False     # if no confirm cross, buy at hour open/first 1s open


def _as_utc_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


class SecIndex:
    def __init__(self, sec_df: pd.DataFrame):
        required = ["Timestamp", "Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in sec_df.columns]
        if missing:
            raise ValueError(f"sec_df missing required columns: {missing}")

        # Keep only required execution columns to reduce memory footprint.
        df = sec_df[required].copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        self.ts = df["Timestamp"].to_numpy(dtype="datetime64[ns]")
        self.o = df["Open"].to_numpy(dtype=float)
        self.h = df["High"].to_numpy(dtype=float)
        self.l = df["Low"].to_numpy(dtype=float)
        self.c = df["Close"].to_numpy(dtype=float)

        if self.ts.size == 0:
            raise ValueError("sec_df has no valid timestamp rows after parsing")

        self.start = _as_utc_ts(self.ts[0])
        self.end = _as_utc_ts(self.ts[-1])

    def slice_i(self, start: pd.Timestamp, end_excl: pd.Timestamp) -> Tuple[int, int]:
        s = np.datetime64(_as_utc_ts(start).to_datetime64())
        e = np.datetime64(_as_utc_ts(end_excl).to_datetime64())
        i0 = int(np.searchsorted(self.ts, s, side="left"))
        i1 = int(np.searchsorted(self.ts, e, side="left"))
        return i0, i1


def _find_confirm_then_entry(
    sec: SecIndex,
    hour_start: pd.Timestamp,
    hour_open_px: float,
    cfg: Exec1SConfig,
) -> Optional[Tuple[pd.Timestamp, float]]:
    """
    Returns (entry_time, entry_price_raw) or None (skip).
    """
    hour_start = _as_utc_ts(hour_start)
    # confirm window is bounded inside the hour
    t_confirm_end = min(hour_start + pd.Timedelta(seconds=cfg.confirm_window_sec),
                        hour_start + pd.Timedelta(seconds=3600))

    i0, i1 = sec.slice_i(hour_start, t_confirm_end)
    if i1 <= i0:
        return None

    open_px = float(hour_open_px)
    up_thr = open_px * (1.0 + cfg.confirm_bps / 1e4)
    abort_thr = open_px * (1.0 - cfg.abort_bps / 1e4)

    lows = sec.l[i0:i1]
    closes = sec.c[i0:i1]
    opens = sec.o[i0:i1]
    tss = sec.ts[i0:i1]

    # if we dump too hard before confirm => skip
    # (this is not lookahead; it's a rule evaluated as time unfolds)
    min_low_so_far = np.minimum.accumulate(lows)
    aborted = min_low_so_far < abort_thr

    # confirm at first second where close (or open) crosses up_thr,
    # BUT only if not aborted before that point.
    cross = closes >= up_thr
    idxs = np.flatnonzero(cross)
    if idxs.size == 0:
        return None

    k = int(idxs[0])
    if aborted[k]:
        return None

    confirm_px = float(closes[k] if cfg.use_close_for_market else opens[k])
    confirm_ts = _as_utc_ts(tss[k])

    # pullback limit after confirm
    limit_px = confirm_px * (1.0 - cfg.pullback_bps / 1e4)
    t_pb_end = min(confirm_ts + pd.Timedelta(seconds=cfg.pullback_window_sec),
                   hour_start + pd.Timedelta(seconds=3600))
    j0, j1 = sec.slice_i(confirm_ts, t_pb_end)
    if j1 > j0:
        lows2 = sec.l[j0:j1]
        tss2 = sec.ts[j0:j1]
        hit = np.flatnonzero(lows2 <= limit_px)
        if hit.size:
            hh = int(hit[0])
            fill_ts = _as_utc_ts(tss2[hh])
            return fill_ts, float(limit_px)

    if cfg.market_on_no_pullback:
        return confirm_ts, float(confirm_px)

    return None


def _find_confirm_then_entry_with_reason(
    sec: SecIndex,
    hour_start: pd.Timestamp,
    hour_open_px: float,
    cfg: Exec1SConfig,
) -> Tuple[Optional[Tuple[pd.Timestamp, float]], str]:
    """
    Returns ((entry_time, entry_price_raw) | None, reason_code).
    reason_code:
      - no_sec_data
      - no_confirm_cross
      - market_on_no_confirm
      - aborted_before_confirm
      - pullback_fill
      - market_fallback
      - no_pullback_no_market
    """
    hour_start = _as_utc_ts(hour_start)
    t_confirm_end = min(
        hour_start + pd.Timedelta(seconds=cfg.confirm_window_sec),
        hour_start + pd.Timedelta(seconds=3600),
    )

    i0, i1 = sec.slice_i(hour_start, t_confirm_end)
    if i1 <= i0:
        return None, "no_sec_data"

    open_px = float(hour_open_px)
    up_thr = open_px * (1.0 + cfg.confirm_bps / 1e4)
    abort_thr = open_px * (1.0 - cfg.abort_bps / 1e4)

    lows = sec.l[i0:i1]
    closes = sec.c[i0:i1]
    opens = sec.o[i0:i1]
    tss = sec.ts[i0:i1]

    min_low_so_far = np.minimum.accumulate(lows)
    aborted = min_low_so_far < abort_thr

    idxs = np.flatnonzero(closes >= up_thr)
    if idxs.size == 0:
        if cfg.market_on_no_confirm:
            j0, j1 = sec.slice_i(hour_start, hour_start + pd.Timedelta(seconds=1))
            fallback_px = float(sec.o[j0]) if j1 > j0 else open_px
            return (hour_start, fallback_px), "market_on_no_confirm"
        return None, "no_confirm_cross"

    k = int(idxs[0])
    if aborted[k]:
        return None, "aborted_before_confirm"

    confirm_px = float(closes[k] if cfg.use_close_for_market else opens[k])
    confirm_ts = _as_utc_ts(tss[k])

    limit_px = confirm_px * (1.0 - cfg.pullback_bps / 1e4)
    t_pb_end = min(
        confirm_ts + pd.Timedelta(seconds=cfg.pullback_window_sec),
        hour_start + pd.Timedelta(seconds=3600),
    )
    j0, j1 = sec.slice_i(confirm_ts, t_pb_end)
    if j1 > j0:
        lows2 = sec.l[j0:j1]
        tss2 = sec.ts[j0:j1]
        hit = np.flatnonzero(lows2 <= limit_px)
        if hit.size:
            hh = int(hit[0])
            fill_ts = _as_utc_ts(tss2[hh])
            return (fill_ts, float(limit_px)), "pullback_fill"

    if cfg.market_on_no_pullback:
        return (confirm_ts, float(confirm_px)), "market_fallback"

    return None, "no_pullback_no_market"


def _first_touch_tp_sl(
    sec: SecIndex,
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    tp_px: float,
    sl_px: float,
) -> Optional[Tuple[pd.Timestamp, str, float]]:
    """
    Find earliest second that hits SL or TP between [start, end_excl).
    Returns (ts, reason, exec_px_raw) or None.
    If both hit, choose the one that happens first in time; if same second, choose SL (conservative).
    """
    i0, i1 = sec.slice_i(start, end_excl)
    if i1 <= i0:
        return None

    h = sec.h[i0:i1]
    l = sec.l[i0:i1]
    tss = sec.ts[i0:i1]

    hit_sl = np.flatnonzero(l <= sl_px)
    hit_tp = np.flatnonzero(h >= tp_px)

    if hit_sl.size == 0 and hit_tp.size == 0:
        return None

    sl_i = int(hit_sl[0]) if hit_sl.size else None
    tp_i = int(hit_tp[0]) if hit_tp.size else None

    if sl_i is not None and tp_i is not None:
        if sl_i < tp_i:
            ts = _as_utc_ts(tss[sl_i])
            return ts, "sl", float(sl_px)
        if tp_i < sl_i:
            ts = _as_utc_ts(tss[tp_i])
            return ts, "tp", float(tp_px)
        # same second -> conservative
        ts = _as_utc_ts(tss[sl_i])
        return ts, "sl", float(sl_px)

    if sl_i is not None:
        ts = _as_utc_ts(tss[sl_i])
        return ts, "sl", float(sl_px)

    ts = _as_utc_ts(tss[tp_i])
    return ts, "tp", float(tp_px)


def run_backtest_long_only_exec_1s(
    df_1h: pd.DataFrame,
    sec_df: pd.DataFrame,
    symbol: str,
    params: Dict[str, Any],
    exec_cfg: Exec1SConfig,
    initial_equity: float = 10_000.0,
    fee_bps: float = 7.0,
    slippage_bps: float = 2.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Backtest long-only on 1h signals, but executes entries/exits using 1s path.
    """
    p = _norm_params(dict(params))

    df = df_1h.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    sec = SecIndex(sec_df)

    # Constrain 1h universe to windows where 1s execution exists.
    sec_start_hour = sec.start.floor("h")
    sec_end_hour = sec.end.floor("h")
    df = df[(df["Timestamp"] >= sec_start_hour) & (df["Timestamp"] <= sec_end_hour)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No 1h rows overlap with 1s data. "
            f"1h=[{df_1h['Timestamp'].min()},{df_1h['Timestamp'].max()}] "
            f"1s=[{sec.start},{sec.end}]"
        )

    df = _ensure_indicators(df, p)

    sig = np.asarray(build_entry_signal(df, p), dtype=bool)

    # cycles shifted for hour-open entries
    cycles_raw = compute_cycles(df, p)
    cycles = _shift_cycles(cycles_raw, int(p.get("cycle_shift", 1)), int(p.get("cycle_fill", 2)))

    ts_1h = df["Timestamp"].to_numpy()
    o_1h = df["Open"].to_numpy(dtype=float)
    c_1h = df["Close"].to_numpy(dtype=float)

    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()

    cash = float(initial_equity)
    units = 0.0
    entry_px = 0.0
    entry_ts = None
    entry_i = -1
    entry_cycle = None
    tp_mult = 1.0
    sl_mult = 1.0

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    equity_curve = []
    trades: List[Dict[str, Any]] = []
    diag = {
        "diag_overlap_hours": float(len(df)),
        "diag_signal_hours": float(sig.sum()),
        "diag_entry_attempts": 0.0,
        "diag_entry_no_sec_data": 0.0,
        "diag_entry_no_confirm_cross": 0.0,
        "diag_entry_market_on_no_confirm": 0.0,
        "diag_entry_aborted_before_confirm": 0.0,
        "diag_entry_pullback_fill": 0.0,
        "diag_entry_market_fallback": 0.0,
        "diag_entry_no_pullback_no_market": 0.0,
        "diag_entries_opened": 0.0,
        "diag_exit_tp": 0.0,
        "diag_exit_sl": 0.0,
        "diag_exit_maxhold": 0.0,
        "diag_exit_rsi_exit": 0.0,
        "diag_exit_eod": 0.0,
    }

    for i in range(len(df)):
        # mark-to-market on 1h close (you can swap to mid if you want)
        equity = cash + units * c_1h[i]
        equity_curve.append(equity)

        hour_start = _as_utc_ts(ts_1h[i])
        hour_end = hour_start + pd.Timedelta(hours=1)

        # ENTRY: 1h signal at hour open -> execute via 1s
        if units == 0.0 and sig[i]:
            diag["diag_entry_attempts"] += 1.0
            entry, reason_code = _find_confirm_then_entry_with_reason(
                sec=sec,
                hour_start=hour_start,
                hour_open_px=float(o_1h[i]),
                cfg=exec_cfg,
            )
            diag_key = f"diag_entry_{reason_code}"
            if diag_key in diag:
                diag[diag_key] += 1.0
            if entry is None:
                continue
            fill_ts, fill_px_raw = entry
            diag["diag_entries_opened"] += 1.0

            buy_px = _apply_cost(float(fill_px_raw), fee_bps, slippage_bps, "buy")
            atrv = float(atr_prev[i])
            size = _position_size(equity, buy_px, atrv, risk_per_trade, max_alloc, atr_k)
            if size <= 0.0:
                continue
            cost = size * buy_px
            if cost > cash:
                size = cash / buy_px
                cost = size * buy_px
            if size <= 0.0:
                continue

            units = size
            cash -= cost
            entry_px = buy_px
            entry_ts = fill_ts
            entry_i = i
            entry_cycle = int(cycles[i])
            if entry_cycle < 0 or entry_cycle > 4:
                entry_cycle = int(p.get("cycle_fill", 2))
            entry_cycle = max(0, min(4, entry_cycle))

            tp_mult = float(p["tp_mult_by_cycle"][entry_cycle])
            sl_mult = float(p["sl_mult_by_cycle"][entry_cycle])
            continue

        # EXIT: check TP/SL using 1s within THIS hour
        if units > 0.0 and entry_ts is not None:
            hold = i - entry_i
            tp_px = entry_px * tp_mult
            sl_px = entry_px * sl_mult

            # priority: tp/sl first-touch via 1s
            touch = _first_touch_tp_sl(sec, hour_start, hour_end, tp_px=tp_px, sl_px=sl_px)
            exit_reason = None
            exit_px_raw = None
            exit_time = None

            if touch is not None:
                exit_time, exit_reason, exit_px_raw = touch

            # time exit at hour open (if no tp/sl first)
            if exit_reason is None and hold >= max_hold:
                # market at first second open of the hour
                i0, i1 = sec.slice_i(hour_start, hour_start + pd.Timedelta(seconds=1))
                if i1 > i0:
                    exit_time = hour_start
                    exit_reason = "maxhold"
                    exit_px_raw = float(sec.o[i0])
                else:
                    exit_time = hour_start
                    exit_reason = "maxhold"
                    exit_px_raw = float(o_1h[i])

            # RSI profit-protect exit at hour open (if no tp/sl/time)
            if exit_reason is None:
                ex = float(p["exit_rsi_by_cycle"][int(entry_cycle)]) if entry_cycle is not None else 50.0
                pnl_ratio = c_1h[i] / entry_px if entry_px > 0 else 1.0
                if (rsi_prev[i] < ex) and (pnl_ratio > 1.0):
                    i0, i1 = sec.slice_i(hour_start, hour_start + pd.Timedelta(seconds=1))
                    exit_time = hour_start
                    exit_reason = "rsi_exit"
                    exit_px_raw = float(sec.o[i0]) if i1 > i0 else float(o_1h[i])

            if exit_reason is not None and exit_px_raw is not None and exit_time is not None:
                diag_exit_key = f"diag_exit_{exit_reason}"
                if diag_exit_key in diag:
                    diag[diag_exit_key] += 1.0
                sell_px = _apply_cost(float(exit_px_raw), fee_bps, slippage_bps, "sell")
                proceeds = units * sell_px
                cash += proceeds

                gross = (sell_px - entry_px) * units
                trades.append({
                    "symbol": symbol,
                    "cycle": int(entry_cycle) if entry_cycle is not None else None,
                    "entry_ts": str(pd.to_datetime(entry_ts, utc=True)),
                    "exit_ts": str(pd.to_datetime(exit_time, utc=True)),
                    "entry_px": float(entry_px),
                    "exit_px": float(sell_px),
                    "units": float(units),
                    "reason": exit_reason,
                    "net_pnl": float(gross),
                    "hold_hours": float(hold),
                })

                units = 0.0
                entry_px = 0.0
                entry_ts = None
                entry_cycle = None
                entry_i = -1

    # liquidate at end
    if units > 0.0:
        diag["diag_exit_eod"] += 1.0
        last_t = _as_utc_ts(ts_1h[-1]) + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        i0, i1 = sec.slice_i(last_t, last_t + pd.Timedelta(seconds=1))
        last_px = float(sec.c[i0]) if i1 > i0 else float(c_1h[-1])
        sell_px = _apply_cost(last_px, fee_bps, slippage_bps, "sell")
        cash += units * sell_px
        gross = (sell_px - entry_px) * units
        trades.append({
            "symbol": symbol,
            "cycle": int(entry_cycle) if entry_cycle is not None else None,
            "entry_ts": str(pd.to_datetime(entry_ts, utc=True)) if entry_ts is not None else None,
            "exit_ts": str(pd.to_datetime(last_t, utc=True)),
            "entry_px": float(entry_px),
            "exit_px": float(sell_px),
            "units": float(units),
            "reason": "eod",
            "net_pnl": float(gross),
            "hold_hours": float(max(0, len(df) - 1 - entry_i)),
        })

    final_equity = float(cash)
    net_profit = float(final_equity - initial_equity)

    eq = np.array(equity_curve, dtype=float) if equity_curve else np.array([initial_equity], dtype=float)
    runmax = np.maximum.accumulate(eq)
    dd = (runmax - eq) / np.maximum(runmax, 1e-9)
    max_dd = float(dd.max()) if dd.size else 0.0

    pnls = np.array([t["net_pnl"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins_n = int((pnls > 0.0).sum()) if pnls.size else 0
    losses_n = int((pnls < 0.0).sum()) if pnls.size else 0
    trades_n = int(pnls.size) if pnls.size else 0

    gross_profit = float(pnls[pnls > 0.0].sum()) if pnls.size else 0.0
    gross_loss = float(pnls[pnls < 0.0].sum()) if pnls.size else 0.0

    win_rate = float((wins_n / max(1, wins_n + losses_n)) * 100.0) if (wins_n + losses_n) else 0.0
    pf = float(gross_profit / abs(gross_loss)) if gross_loss < -1e-9 else (10.0 if gross_profit > 0 else 0.0)

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "net_profit": float(net_profit),
        "trades": float(trades_n),
        "wins": float(wins_n),
        "losses": float(losses_n),
        "win_rate_pct": float(win_rate),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "max_dd": float(max_dd),
        "profit_factor": float(pf),
    }
    metrics.update(diag)
    return trades, metrics
