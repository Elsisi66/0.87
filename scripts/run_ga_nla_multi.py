#!/usr/bin/env python3
"""
Run LONG GA with NO-LOOKAHEAD cycles + no-lookahead-safe exit profit check.
Runs multiple symbols and saves under <SYMBOL>_NLA (so it won't overwrite your normal results).

Default: trade cycle is FIXED to [3].
"""
from __future__ import annotations

import os, sys, json, argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim import ga as g


def _load_params_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    raw = json.load(open(path, "r"))
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    if isinstance(raw, dict):
        return raw
    return None


def _load_seed(base_symbol: str) -> Dict[str, Any]:
    meta = PROJECT_ROOT / "data" / "metadata" / "params"
    for fp in [
        meta / f"{base_symbol}_active_params.json",
        meta / f"{base_symbol}_seed_params.json",
        meta / "BTCUSDT_active_params.json",
    ]:
        got = _load_params_json(fp)
        if got:
            return got
    raise FileNotFoundError("Could not find seed params in data/metadata/params.")


def _load_df(base_symbol: str, tf: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{base_symbol}_{tf}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet: {fp}")
    return pd.read_parquet(fp)


def compute_cycles_no_lookahead(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    """
    Regimes aligned to decisions at bar t OPEN.
    Everything uses t-1 info (no lookahead).
    """
    close = df["Close"].astype(float)

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_long = df[ema_long_col].astype(float) if ema_long_col in df.columns else df["EMA_200"].astype(float)

    slope = df.get("EMA_200_SLOPE", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    adx = df.get("ADX", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)
    rsi = df.get("RSI", pd.Series(np.full(len(df), 50.0), index=df.index)).astype(float)
    atr = df.get("ATR", pd.Series(np.zeros(len(df)), index=df.index)).astype(float)

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

    # breakout uses only past info:
    # recent_high_prev = max(close[t-2 .. t-21]) compared to close_prev (=close[t-1])
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

    if n > 0:
        cycles[0] = 2
    return cycles


def run_backtest_long_only_no_lookahead(
    df: pd.DataFrame,
    symbol: str,
    p: Dict[str, Any],
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
):
    """
    Uses same OHLC bar model as your GA backtest, but:
    - cycles are computed with t-1 info
    - RSI exit profit-check uses OPEN (available at decision time), not CLOSE
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    sig = g.build_entry_signal(df, p)
    sig = np.asarray(sig, dtype=bool)

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
    trades = []

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    for i in range(len(df)):
        equity = cash + units * c[i]
        equity_curve.append(equity)

        # ENTRY at OPEN
        if units == 0.0 and sig[i]:
            buy_px = g._apply_cost(o[i], fee_bps, slippage_bps, "buy")
            size = g._position_size(equity, buy_px, float(atr_prev[i]), risk_per_trade, max_alloc, atr_k)
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

            # conservative if both
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
                ex = float(p["exit_rsi_by_cycle"][int(entry_cycle)]) if entry_cycle is not None else 50.0
                pnl_ratio = o[i] / entry_px if entry_px > 0 else 1.0  # OPEN, not CLOSE
                if (rsi_prev[i] < ex) and (pnl_ratio > 1.0):
                    exit_reason = "rsi_exit"
                    exit_exec_px = o[i]

            if exit_reason is not None and exit_exec_px is not None:
                sell_px = g._apply_cost(float(exit_exec_px), fee_bps, slippage_bps, "sell")
                cash += units * sell_px
                gross = (sell_px - entry_px) * units

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
                    "hold_hours": float(hold),
                })

                units = 0.0
                entry_px = 0.0
                entry_ts = None
                entry_cycle = None
                entry_i = -1

    # mark-to-market at end
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma list like BTCUSDT,ADAUSDT,SOLUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--suffix", default="_NLA")
    ap.add_argument("--cycles", default="3", help="Comma list of cycles to allow, default 3")
    ap.add_argument("--generations", type=int, default=35)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()

    base_syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fixed_cycles = [int(x) for x in args.cycles.split(",") if x.strip()]

    # Patch GA functions to NLA variants
    g.compute_cycles = compute_cycles_no_lookahead
    g.run_backtest_long_only = run_backtest_long_only_no_lookahead

    # Force cycles fixed (prevents "GA wins by barely trading the right cycle")
    _orig_mut = g.mutate_params
    def _mut_fixed(p, strength, rate):
        q = _orig_mut(p, strength, rate)
        q["trade_cycles"] = list(fixed_cycles)
        return g._norm_params(q)
    g.mutate_params = _mut_fixed

    for base_symbol in base_syms:
        out_symbol = base_symbol + args.suffix

        print(f"\n=== NLA GA: {out_symbol} (data={base_symbol}, tf={args.tf}, cycles={fixed_cycles}) ===", flush=True)

        df = _load_df(base_symbol, tf=args.tf)
        seed = _load_seed(base_symbol)
        seed = g._norm_params(seed)
        seed["trade_cycles"] = list(fixed_cycles)

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

            min_trades_train=40,
            min_trades_val=15,

            require_trade_cycles=True,

            w_train=0.7,
            w_val=0.3,
            dd_penalty=0.45,
            trade_penalty=0.8,
            bad_val_penalty=1200.0,

            resume=True,
        )

        best_p, report = g.run_ga_montecarlo(symbol=out_symbol, df=df, seed_params=seed, cfg=cfg)
        print("Saved:", report["saved"], flush=True)
        print("Best trade_cycles:", best_p.get("trade_cycles"), flush=True)


if __name__ == "__main__":
    main()
