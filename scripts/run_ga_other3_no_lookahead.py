#!/usr/bin/env python3
"""
Run GA for ETH/AVAX/BNB with:
- NO-LOOKAHEAD cycle computation (cycle at bar t uses t-1 info)
- NO-LOOKAHEAD RSI profit-exit check (uses open[t], not close[t])
- Better cycle subset exploration (not just [1,2] / [1,2,3])

Outputs go to the normal GA locations:
  artifacts/ga/<SYMBOL>/...
  data/metadata/params/<SYMBOL>_active_params.json
"""

from __future__ import annotations

import os, sys, json, argparse, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim import ga as g


def _load_df(symbol: str, tf: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet: {fp}")
    return pd.read_parquet(fp)


def _load_params_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    raw = json.load(open(path, "r"))
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    if isinstance(raw, dict):
        return raw
    return None


def _load_seed(symbol: str) -> Dict[str, Any]:
    meta = PROJECT_ROOT / "data" / "metadata" / "params"
    # prefer coin active, else BTC active, else fallback
    for fp in [
        meta / f"{symbol}_active_params.json",
        meta / f"{symbol}_seed_params.json",
        meta / "BTCUSDT_active_params.json",
    ]:
        got = _load_params_json(fp)
        if got:
            return got

    # hard fallback baseline
    return {
        "entry_rsi_min": 54.0,
        "entry_rsi_max": 62.0,
        "entry_rsi_buffer": 3.0,
        "willr_floor": -100.0,
        "willr_by_cycle": [-76.0, -28.0, -91.0, -16.0, -46.0],
        "ema_span": 35,
        "ema_trend_long": 120,
        "ema_align": True,
        "require_ema200_slope": True,
        "adx_min": 18.0,
        "require_plus_di": True,
        "tp_mult_by_cycle": [1.04, 1.07, 1.05, 1.06, 1.06],
        "sl_mult_by_cycle": [0.98, 0.98, 0.95, 0.99, 0.965],
        "exit_rsi_by_cycle": [52.8, 55.6, 49.1, 65.6, 54.7],
        "risk_per_trade": 0.02,
        "max_allocation": 0.7,
        "atr_k": 1.0,
        "allow_hours": None,
        "trade_cycles": [1, 2, 3],
        "max_hold_hours": 48,
    }


# ----------------------------
# NO-LOOKAHEAD cycles
# ----------------------------
def compute_cycles_no_lookahead(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    """
    Returns cycles 0..4 aligned to bar t OPEN decisions.
    Cycle at index t is computed using t-1 data (no lookahead).
    """
    close = df["Close"].astype(float)

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_long = df[ema_long_col].astype(float) if ema_long_col in df.columns else df["EMA_200"].astype(float)

    slope = df.get("EMA_200_SLOPE", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    adx = df.get("ADX", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    rsi = df.get("RSI", pd.Series(np.full(len(df), 50.0), index=df.index)).astype(float)
    atr = df.get("ATR", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)

    # shift everything by 1: what you know at bar t open is bar t-1 close/indicators
    close_prev = close.shift(1).to_numpy()
    ema_prev = ema_long.shift(1).to_numpy()
    slope_prev = slope.shift(1).to_numpy()
    adx_prev = adx.shift(1).to_numpy()
    rsi_prev = rsi.shift(1).to_numpy()
    atr_prev = atr.shift(1).to_numpy()

    above = close_prev > ema_prev
    s_up = slope_prev > 0.0
    s_down = slope_prev < 0.0

    adx_thr = float(p.get("adx_min", 18.0))
    adx_high = adx_prev >= adx_thr
    rsi_high = rsi_prev > float(p.get("entry_rsi_max", 65.0))

    # breakout uses previous close vs rolling high up to t-2
    w = 20
    recent_high_prev = close.shift(2).rolling(w, min_periods=w).max().to_numpy()
    gap_prev = close_prev - recent_high_prev
    mult = float(p.get("breakout_atr_mult", 1.5))
    breakout_up = (close_prev > recent_high_prev) & (gap_prev > atr_prev * mult)
    breakout_up = np.nan_to_num(breakout_up, nan=False).astype(bool)

    n = len(df)
    cycles = np.full(n, 2, dtype=np.int8)

    cycles[breakout_up] = 3
    mask_down = (~breakout_up) & s_down & (~above) & adx_high
    cycles[mask_down] = 0
    mask_exp = (~breakout_up) & (~mask_down) & s_up & above & adx_high
    cycles[mask_exp] = 1
    mask_dist = (~breakout_up) & (~mask_down) & (~mask_exp) & rsi_high & (~adx_high)
    cycles[mask_dist] = 4

    # first bars: force default accumulation
    if n > 0:
        cycles[0] = 2

    return cycles


def build_entry_signal_no_lookahead(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    cycles = compute_cycles_no_lookahead(df, p)

    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()
    willr_prev = df["WILLR"].astype(float).shift(1).fillna(-50.0).to_numpy()
    close_prev = df["Close"].astype(float).shift(1).fillna(df["Close"].astype(float)).to_numpy()

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_span_col = f"EMA_{int(p['ema_span'])}"
    ema_long_prev = (
        df[ema_long_col].astype(float).shift(1).ffill().to_numpy()
        if ema_long_col in df.columns
        else df["EMA_200"].astype(float).shift(1).ffill().to_numpy()
    )
    ema_span_prev = (
        df[ema_span_col].astype(float).shift(1).ffill().to_numpy()
        if ema_span_col in df.columns
        else ema_long_prev
    )

    slope_prev = df.get("EMA_200_SLOPE", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    adx_prev = df.get("ADX", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    plus_prev = df.get("PLUS_DI", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    minus_prev = df.get("MINUS_DI", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()

    close_ok = close_prev > ema_long_prev
    ema_ok = (ema_span_prev > ema_long_prev) if bool(p.get("ema_align", True)) else True
    adx_ok = adx_prev >= float(p.get("adx_min", 18.0))
    di_ok = (plus_prev > minus_prev) if bool(p.get("require_plus_di", True)) else True
    slope_ok = (slope_prev > 0.0) if bool(p.get("require_ema200_slope", True)) else True
    trend_ok = close_ok & ema_ok & adx_ok & di_ok & slope_ok

    trade_cycles = set(int(x) for x in p.get("trade_cycles", [1, 2, 3]))
    if bool(p.get("require_trade_cycles", True)):
        cyc_ok = np.array([int(c) in trade_cycles for c in cycles], dtype=bool)
    else:
        cyc_ok = np.ones(len(df), dtype=bool)

    # hour filter (reuse original)
    hour_ok = g._hour_mask(df["Timestamp"], p.get("allow_hours"))

    rsi_min = float(p["entry_rsi_min"])
    rsi_max = float(p["entry_rsi_max"])
    rsi_band = (rsi_prev >= rsi_min) & (rsi_prev <= rsi_max)

    willr_floor = float(p.get("willr_floor", -100.0))
    willr_thr = np.array([float(p["willr_by_cycle"][int(c)]) for c in cycles], dtype=float)
    willr_ok = (willr_prev >= willr_floor) & (willr_prev < willr_thr)

    return (rsi_band & willr_ok & trend_ok & cyc_ok & hour_ok).astype(bool)


def run_backtest_long_only_no_lookahead(
    df: pd.DataFrame,
    symbol: str,
    p: Dict[str, Any],
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    sig = build_entry_signal_no_lookahead(df, p)
    sig = np.asarray(sig, dtype=bool)
    if sig.ndim != 1 or len(sig) != len(df):
        raise RuntimeError(f"Bad sig shape/len: shape={sig.shape} len={len(sig)} df={len(df)}")

    ts = df["Timestamp"].to_numpy()
    o = df["Open"].astype(float).to_numpy()
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    c = df["Close"].astype(float).to_numpy()
    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()

    cycles = compute_cycles_no_lookahead(df, p)

    cash = float(initial_equity)
    units = 0.0
    entry_px = 0.0
    entry_ts = None
    entry_cycle = None
    entry_i = -1
    tp_mult = 1.0
    sl_mult = 1.0

    equity_curve = []
    trades: List[Dict[str, Any]] = []

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    for i in range(len(df)):
        mid = c[i]
        equity = cash + units * mid
        equity_curve.append(equity)

        # ENTRY
        if units == 0.0 and sig[i]:
            buy_px = g._apply_cost(o[i], fee_bps, slippage_bps, "buy")
            atrv = float(atr_prev[i])
            size = g._position_size(equity, buy_px, atrv, risk_per_trade, max_alloc, atr_k)
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
            entry_ts = ts[i]
            entry_cycle = int(cycles[i])
            entry_i = i

            tp_mult = float(p["tp_mult_by_cycle"][entry_cycle])
            sl_mult = float(p["sl_mult_by_cycle"][entry_cycle])
            continue

        # EXIT
        if units > 0.0 and entry_ts is not None:
            hold = i - entry_i
            tp_px = entry_px * tp_mult
            sl_px = entry_px * sl_mult

            exit_reason = None
            exit_exec_px = None

            hit_sl = l[i] <= sl_px
            hit_tp = h[i] >= tp_px

            if hit_sl and hit_tp:
                exit_reason = "sl"
                exit_exec_px = sl_px
            elif hit_sl:
                exit_reason = "sl"
                exit_exec_px = sl_px
            elif hit_tp:
                exit_reason = "tp"
                exit_exec_px = tp_px
            elif hold >= max_hold:
                exit_reason = "maxhold"
                exit_exec_px = o[i]
            else:
                # NO-LOOKAHEAD: use open[i] (known) to check "in profit"
                ex = float(p["exit_rsi_by_cycle"][int(entry_cycle)]) if entry_cycle is not None else 50.0
                pnl_ratio = (o[i] / entry_px) if entry_px > 0 else 1.0
                if (rsi_prev[i] < ex) and (pnl_ratio > 1.0):
                    exit_reason = "rsi_exit"
                    exit_exec_px = o[i]

            if exit_reason is not None and exit_exec_px is not None:
                sell_px = g._apply_cost(float(exit_exec_px), fee_bps, slippage_bps, "sell")
                proceeds = units * sell_px
                cash += proceeds

                gross = (sell_px - entry_px) * units
                hold_hours = float(hold)

                trades.append({
                    "symbol": symbol,
                    "cycle": int(entry_cycle) if entry_cycle is not None else None,
                    "entry_ts": str(pd.to_datetime(entry_ts, utc=True)),
                    "exit_ts": str(pd.to_datetime(ts[i], utc=True)),
                    "entry_px": float(entry_px),
                    "exit_px": float(sell_px),
                    "units": float(units),
                    "reason": exit_reason,
                    "net_pnl": float(gross),
                    "hold_hours": float(hold_hours),
                })

                units = 0.0
                entry_px = 0.0
                entry_ts = None
                entry_cycle = None
                entry_i = -1

    # Close any open at end
    if units > 0.0:
        sell_px = g._apply_cost(float(c[-1]), fee_bps, slippage_bps, "sell")
        cash += units * sell_px
        gross = (sell_px - entry_px) * units
        trades.append({
            "symbol": symbol,
            "cycle": int(entry_cycle) if entry_cycle is not None else None,
            "entry_ts": str(pd.to_datetime(entry_ts, utc=True)) if entry_ts is not None else None,
            "exit_ts": str(pd.to_datetime(ts[-1], utc=True)),
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
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    win_rate = float((pnls > 0.0).mean() * 100.0) if pnls.size else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 1e-9 else (10.0 if wins.size else 0.0)

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "net_profit": float(net_profit),
        "trades": float(len(trades)),
        "win_rate_pct": float(win_rate),
        "max_dd": float(max_dd),
        "profit_factor": float(pf),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
    }
    return trades, metrics


def _random_trade_cycles_v2() -> List[int]:
    # GA cycles are 0..4. For longs, avoid 0 by default.
    choices = [
        [1, 2],
        [1, 2, 3],
        [2, 3],
        [1, 3],
        [3],
        [2],
        [4],
        [2, 4],
        [1, 2, 4],
        [1, 3, 4],
    ]
    return random.choice(choices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,AVAXUSDT,BNBUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--generations", type=int, default=35)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()

    # monkeypatch GA module to remove lookahead + broaden cycle search
    g.compute_cycles = compute_cycles_no_lookahead
    g.build_entry_signal = build_entry_signal_no_lookahead
    g.run_backtest_long_only = run_backtest_long_only_no_lookahead
    g._random_trade_cycles = _random_trade_cycles_v2

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        df = _load_df(sym, args.tf)

        seed = _load_seed(sym)
        # broaden by default for these coins
        seed = dict(seed)
        seed.setdefault("trade_cycles", [1, 2, 3, 4])
        # make breakouts less “unicorn” on some coins (still conservative)
        seed.setdefault("breakout_atr_mult", 1.0)

        cfg = g.GAConfig(
            pop_size=args.pop,
            generations=args.generations,
            elite_k=6,
            mutation_rate=0.35,
            mutation_strength=1.0,
            n_procs=args.procs,

            mc_splits=6,
            train_days=540,
            val_days=180,
            test_days=180,
            seed=42,

            fee_bps=7.0,
            slippage_bps=2.0,
            initial_equity=10_000.0,

            # looser for these coins so GA doesn't “force churn”
            min_trades_train=20,
            min_trades_val=8,

            require_trade_cycles=True,

            w_train=0.7,
            w_val=0.3,
            dd_penalty=0.45,
            trade_penalty=0.8,
            bad_val_penalty=1200.0,

            resume=True,
        )

        print(f"\n=== RUN GA (NO LOOKAHEAD) {sym} tf={args.tf} ===", flush=True)
        best_p, report = g.run_ga_montecarlo(symbol=sym, df=df, seed_params=seed, cfg=cfg)
        print("Saved:", report["saved"], flush=True)
        print("Best trade_cycles:", best_p.get("trade_cycles"), flush=True)


if __name__ == "__main__":
    main()
