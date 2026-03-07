#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import ccxt
except Exception as e:
    raise SystemExit("Missing ccxt. Install with: pip install ccxt") from e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.telegram_signal_alert_live import (
    sym_to_ccxt, tf_to_ms, norm_params, ensure_indicators,
    compute_cycles, shift_cycles, build_entry_signal
)

def utc_now():
    return datetime.now(timezone.utc)

def fetch_ohlcv_range(exchange, symbol_ccxt: str, timeframe: str, start_dt: datetime, end_dt: datetime, limit=1000) -> pd.DataFrame:
    ms_tf = tf_to_ms(timeframe)
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    rows = []
    last_ts = None
    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        for r in batch:
            if r[0] >= end_ms:
                break
            rows.append(r)
        new_last = batch[-1][0]
        if last_ts is not None and new_last == last_ts:
            break
        last_ts = new_last
        since = new_last + ms_tf

    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"])
    df["Timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    return df

def apply_cost(px: float, fee_bps: float, slip_bps: float, side: str) -> float:
    fee = fee_bps / 1e4
    slip = slip_bps / 1e4
    if side == "buy":
        return px * (1.0 + fee + slip)
    return px * (1.0 - fee - slip)

def position_size(equity: float, entry_px: float, atr: float, risk_per_trade: float, max_alloc: float, atr_k: float) -> float:
    if equity <= 0 or entry_px <= 0 or atr <= 0:
        return 0.0
    u_risk = (equity * risk_per_trade) / (atr_k * atr)
    u_afford = (equity * max_alloc) / entry_px
    return max(0.0, min(u_risk, u_afford))

def run_backtest_long(df: pd.DataFrame, p: dict, fee_bps: float, slip_bps: float, initial: float, only_cycles=None, cycle_shift=1):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df = ensure_indicators(df, p)

    sig, cycles = build_entry_signal(df, p, cycle_shift=cycle_shift, two_candle=False)

    if only_cycles is not None:
        only = set(int(x) for x in only_cycles)
    else:
        only = None

    ts = df["Timestamp"].to_numpy()
    o = df["Open"].astype(float).to_numpy()
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    c = df["Close"].astype(float).to_numpy()
    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()

    cash = float(initial)
    units = 0.0
    entry_px = 0.0
    entry_i = -1
    entry_cycle = None

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    equity_curve = []
    trades = []

    for i in range(len(df)):
        equity = cash + units * c[i]
        equity_curve.append(equity)

        if units == 0.0 and sig[i]:
            cyc = int(cycles[i])
            if only is not None and cyc not in only:
                continue

            buy_px = apply_cost(float(o[i]), fee_bps, slip_bps, "buy")
            size = position_size(equity, buy_px, float(atr_prev[i]), risk_per_trade, max_alloc, atr_k)
            if size <= 0:
                continue
            cost = size * buy_px
            if cost > cash:
                size = cash / buy_px
                cost = size * buy_px
            if size <= 0:
                continue

            units = size
            cash -= cost
            entry_px = buy_px
            entry_i = i
            entry_cycle = cyc
            continue

        if units > 0.0 and entry_i >= 0:
            hold = i - entry_i
            tp_mult = float(p["tp_mult_by_cycle"][int(entry_cycle)])
            sl_mult = float(p["sl_mult_by_cycle"][int(entry_cycle)])
            tp_px = entry_px * tp_mult
            sl_px = entry_px * sl_mult

            hit_sl = l[i] <= sl_px
            hit_tp = h[i] >= tp_px

            exit_reason = None
            exit_exec_px = None

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
                exit_exec_px = float(o[i])
            else:
                ex = float(p["exit_rsi_by_cycle"][int(entry_cycle)])
                pnl_ratio = c[i] / entry_px if entry_px > 0 else 1.0
                if (rsi_prev[i] < ex) and (pnl_ratio > 1.0):
                    exit_reason = "rsi_exit"
                    exit_exec_px = float(o[i])

            if exit_reason is not None:
                sell_px = apply_cost(float(exit_exec_px), fee_bps, slip_bps, "sell")
                proceeds = units * sell_px
                cash += proceeds
                net_pnl = (sell_px - entry_px) * units

                trades.append({
                    "entry_ts": str(pd.to_datetime(ts[entry_i], utc=True)),
                    "exit_ts":  str(pd.to_datetime(ts[i], utc=True)),
                    "cycle": int(entry_cycle),
                    "entry_px": float(entry_px),
                    "exit_px": float(sell_px),
                    "units": float(units),
                    "reason": exit_reason,
                    "net_pnl": float(net_pnl),
                    "hold_bars": int(hold),
                })

                units = 0.0
                entry_px = 0.0
                entry_i = -1
                entry_cycle = None

    if units > 0.0:
        sell_px = apply_cost(float(c[-1]), fee_bps, slip_bps, "sell")
        cash += units * sell_px
        net_pnl = (sell_px - entry_px) * units
        trades.append({
            "entry_ts": str(pd.to_datetime(ts[entry_i], utc=True)),
            "exit_ts":  str(pd.to_datetime(ts[-1], utc=True)),
            "cycle": int(entry_cycle) if entry_cycle is not None else None,
            "entry_px": float(entry_px),
            "exit_px": float(sell_px),
            "units": float(units),
            "reason": "eod",
            "net_pnl": float(net_pnl),
            "hold_bars": int(max(0, len(df)-1-entry_i)),
        })

    eq = np.array(equity_curve, dtype=float) if equity_curve else np.array([initial], dtype=float)
    runmax = np.maximum.accumulate(eq)
    dd = (runmax - eq) / np.maximum(runmax, 1e-9)
    max_dd = float(dd.max()) if dd.size else 0.0

    pnls = np.array([t["net_pnl"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float((pnls > 0).mean() * 100.0) if pnls.size else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 1e-9 else (float("inf") if wins.size else 0.0)

    metrics = {
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "trades": float(len(trades)),
        "wins": float((pnls > 0).sum()) if pnls.size else 0.0,
        "losses": float((pnls < 0).sum()) if pnls.size else 0.0,
        "win_rate_pct": float(win_rate),
        "net_profit": float(cash - initial),
        "final_equity": float(cash),
        "profit_factor": float(pf) if np.isfinite(pf) else float("inf"),
        "max_dd_pct": float(max_dd),
        "fee_bps": float(fee_bps),
        "slip_bps": float(slip_bps),
        "cycle_shift": int(cycle_shift),
        "only_cycles": list(sorted(only)) if only is not None else None,
    }
    return trades, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--params", default=str((ROOT/"artifacts/ga/BTCUSDT/best_params.json").resolve()))
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--only-cycles", default="3")
    ap.add_argument("--cycle-shift", type=int, default=1)
    args = ap.parse_args()

    sym_ccxt = sym_to_ccxt(args.symbol)
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})

    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=int(args.days))

    df = fetch_ohlcv_range(exchange, sym_ccxt, args.tf, start_dt, end_dt, limit=1000)

    p = norm_params(json.load(open(args.params, "r")))
    only_cycles = [int(x) for x in args.only_cycles.split(",") if x.strip()] if args.only_cycles else None

    trades, m = run_backtest_long(
        df, p, fee_bps=args.fee_bps, slip_bps=args.slip_bps,
        initial=args.initial, only_cycles=only_cycles, cycle_shift=args.cycle_shift
    )

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / "backtests_live" / args.symbol.upper() / ts_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    print(f"\n=== {args.symbol.upper()} LIVE BACKTEST last {args.days}d ===")
    print(f"Saved: {out_dir}")
    for k in ["rows","start","end","trades","wins","losses","win_rate_pct","net_profit","profit_factor","max_dd_pct","final_equity","fee_bps","slip_bps","cycle_shift","only_cycles"]:
        print(f"{k}: {m[k]}")

if __name__ == "__main__":
    main()
