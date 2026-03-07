#!/usr/bin/env python3
"""
No-lookahead backtest runner.

Key NLA rules enforced:
- Entry decision uses t-1 features (RSI/WILLR/ADX/DI/slope/EMA conditions).
- Regime/cycle used for filtering & thresholds can be shifted forward by N bars
  (so bar t uses the cycle label from bar t-1). Use --cycle-shift 1 for strict NLA.
- RSI exit decision is based on PREVIOUS BAR state and executes on next open
  (fixes a real lookahead leak where current close was used to decide an exit at same-bar open).

Outputs:
  output/backtests_nla/<SYMBOL>/<UTCSTAMP>/
    - trades.csv
    - metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Project root + import path
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------
# Helpers: IO
# ----------------------------
def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(fp: Path) -> Dict[str, Any]:
    raw = json.loads(fp.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Bad json format: {fp}")


def _find_params(symbol: str) -> Path:
    """
    Preference order:
      1) artifacts/ga/<SYMBOL>/best_params.json
      2) data/metadata/params/<SYMBOL>_active_params.json
    """
    p1 = PROJECT_ROOT / "artifacts" / "ga" / symbol / "best_params.json"
    p2 = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"No params found for {symbol}. Tried:\n- {p1}\n- {p2}")


def _load_df(symbol: str, tf: str) -> pd.DataFrame:
    """
    Preferred:
      data/processed/_full/<SYMBOL>_<TF>_full.parquet
    Fallback:
      data/parquet/<SYMBOL>.parquet
    Fallback:
      concat data/processed/<SYMBOL>_*_proc.csv
    """
    full_fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if full_fp.exists():
        df = pd.read_parquet(full_fp)
        return df

    par_fp = PROJECT_ROOT / "data" / "parquet" / f"{symbol}.parquet"
    if par_fp.exists():
        df = pd.read_parquet(par_fp)
        # try to restore column casing expected by indicators/backtest
        if "timestamp" in df.columns and "Timestamp" not in df.columns:
            df = df.rename(columns={"timestamp": "Timestamp"})
        if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        if "Timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Timestamp"})
        return df

    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if not files:
        raise FileNotFoundError(f"No data found for {symbol}. Looked in:\n- {full_fp}\n- {par_fp}\n- {proc_dir}/{symbol}_*_proc.csv")
    dfs = [pd.read_csv(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ----------------------------
# NLA: Entry signal + backtest
# ----------------------------
def _hour_mask(ts: pd.Series, allow_hours: Optional[List[int]]) -> np.ndarray:
    if not allow_hours:
        return np.ones(len(ts), dtype=bool)
    allowed = set(int(h) for h in allow_hours)
    return ts.dt.hour.apply(lambda h: int(h) in allowed).to_numpy(dtype=bool)


def _apply_cost(price: float, fee_bps: float, slip_bps: float, side: str) -> float:
    fee = fee_bps / 1e4
    slip = slip_bps / 1e4
    if side == "buy":
        return price * (1.0 + fee + slip)
    return price * (1.0 - fee - slip)


def _position_size(equity: float, entry_px: float, atr: float, risk_per_trade: float, max_alloc: float, atr_k: float) -> float:
    if equity <= 0.0 or entry_px <= 0.0 or atr <= 0.0:
        return 0.0
    u_risk = (equity * risk_per_trade) / (atr_k * atr)
    u_afford = (equity * max_alloc) / entry_px
    return max(0.0, min(u_risk, u_afford))


def _shift_cycles(cycles: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return cycles
    out = np.roll(cycles, shift)
    # prevent wraparound leak: copy earliest stable label
    out[:shift] = out[shift]
    return out


def build_entry_signal_nla(df: pd.DataFrame, p: Dict[str, Any], cycle_shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds entry signal using t-1 features, and cycles shifted by cycle_shift.

    Returns:
      sig: bool array len(df)
      cycles_nla: int array len(df) used for filtering/thresholds
    """
    from src.bot087.optim.ga import compute_cycles  # use your existing regime logic

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    cycles = compute_cycles(df, p)
    cycles_nla = _shift_cycles(cycles, int(cycle_shift))

    # t-1 features
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()
    willr_prev = df["WILLR"].astype(float).shift(1).fillna(-50.0).to_numpy()

    close_prev = df["Close"].astype(float).shift(1).fillna(df["Close"].astype(float)).to_numpy()

    ema_long_col = f"EMA_{int(p.get('ema_trend_long', 120))}"
    ema_span_col = f"EMA_{int(p.get('ema_span', 35))}"

    if ema_long_col in df.columns:
        ema_long_prev = df[ema_long_col].astype(float).shift(1).ffill().to_numpy()
    else:
        ema_long_prev = df["EMA_200"].astype(float).shift(1).ffill().to_numpy()

    if ema_span_col in df.columns:
        ema_span_prev = df[ema_span_col].astype(float).shift(1).ffill().to_numpy()
    else:
        ema_span_prev = ema_long_prev

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

    trade_cycles = set(int(x) for x in p.get("trade_cycles", [1, 2]))
    require_trade_cycles = bool(p.get("require_trade_cycles", True))
    if require_trade_cycles:
        cyc_ok = np.array([int(c) in trade_cycles for c in cycles_nla], dtype=bool)
    else:
        cyc_ok = np.ones(len(df), dtype=bool)

    hour_ok = _hour_mask(df["Timestamp"], p.get("allow_hours"))

    rsi_min = float(p.get("entry_rsi_min", 50.0))
    rsi_max = float(p.get("entry_rsi_max", 65.0))
    rsi_band = (rsi_prev >= rsi_min) & (rsi_prev <= rsi_max)

    willr_floor = float(p.get("willr_floor", -100.0))
    willr_by_cycle = p.get("willr_by_cycle", [-80, -80, -80, -80, -80])
    willr_thr = np.array([float(willr_by_cycle[int(c)]) for c in cycles_nla], dtype=float)
    willr_ok = (willr_prev >= willr_floor) & (willr_prev < willr_thr)

    sig = (rsi_band & willr_ok & trend_ok & cyc_ok & hour_ok).astype(bool)
    return sig, cycles_nla


def run_backtest_nla(
    df: pd.DataFrame,
    symbol: str,
    p: Dict[str, Any],
    initial_equity: float,
    fee_bps: float,
    slip_bps: float,
    cycle_shift: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    sig, cycles_nla = build_entry_signal_nla(df, p, cycle_shift=cycle_shift)

    ts = df["Timestamp"].to_numpy()
    o = df["Open"].astype(float).to_numpy()
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    c = df["Close"].astype(float).to_numpy()

    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()
    close_prev = df["Close"].astype(float).shift(1).fillna(df["Close"].astype(float)).to_numpy()

    cash = float(initial_equity)
    units = 0.0
    entry_px = 0.0
    entry_ts = None
    entry_cycle = None
    entry_i = -1

    tp_mult = 1.0
    sl_mult = 1.0

    equity_curve: List[float] = []
    trades: List[Dict[str, Any]] = []

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    tp_by = p.get("tp_mult_by_cycle", [1.05] * 5)
    sl_by = p.get("sl_mult_by_cycle", [0.98] * 5)
    exit_rsi_by = p.get("exit_rsi_by_cycle", [50.0] * 5)

    for i in range(len(df)):
        # mark-to-market at close (fine for reporting)
        equity = cash + units * c[i]
        equity_curve.append(float(equity))

        # ENTRY at bar open if signal true
        if units == 0.0 and bool(sig[i]):
            buy_px = _apply_cost(float(o[i]), fee_bps, slip_bps, "buy")
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

            units = float(size)
            cash -= float(cost)

            entry_px = float(buy_px)
            entry_ts = ts[i]
            entry_cycle = int(cycles_nla[i])  # IMPORTANT: shifted cycle
            entry_i = i

            tp_mult = float(tp_by[entry_cycle])
            sl_mult = float(sl_by[entry_cycle])
            continue

        # EXIT logic
        if units > 0.0 and entry_ts is not None and entry_cycle is not None:
            hold = i - entry_i
            tp_px = entry_px * tp_mult
            sl_px = entry_px * sl_mult

            exit_reason = None
            exit_exec_px = None

            hit_sl = float(l[i]) <= float(sl_px)
            hit_tp = float(h[i]) >= float(tp_px)

            # pessimistic if both hit same bar
            if hit_sl and hit_tp:
                exit_reason = "sl"
                exit_exec_px = float(sl_px)
            elif hit_sl:
                exit_reason = "sl"
                exit_exec_px = float(sl_px)
            elif hit_tp:
                exit_reason = "tp"
                exit_exec_px = float(tp_px)
            elif hold >= max_hold:
                exit_reason = "maxhold"
                exit_exec_px = float(o[i])
            else:
                # STRICT NLA: decision based on PREVIOUS BAR state, execute on this bar open
                ex = float(exit_rsi_by[int(entry_cycle)])
                pnl_ratio_prev_close = float(close_prev[i]) / float(entry_px) if entry_px > 0 else 1.0
                if (float(rsi_prev[i]) < ex) and (pnl_ratio_prev_close > 1.0):
                    exit_reason = "rsi_exit"
                    exit_exec_px = float(o[i])

            if exit_reason is not None and exit_exec_px is not None:
                sell_px = _apply_cost(float(exit_exec_px), fee_bps, slip_bps, "sell")
                proceeds = units * sell_px
                cash += float(proceeds)

                net_pnl = (sell_px - entry_px) * units

                trades.append({
                    "symbol": symbol,
                    "cycle": int(entry_cycle),
                    "entry_ts": str(pd.to_datetime(entry_ts, utc=True)),
                    "exit_ts": str(pd.to_datetime(ts[i], utc=True)),
                    "entry_px": float(entry_px),
                    "exit_px": float(sell_px),
                    "units": float(units),
                    "reason": str(exit_reason),
                    "net_pnl": float(net_pnl),
                    "hold_hours": float(hold),
                })

                # flat
                units = 0.0
                entry_px = 0.0
                entry_ts = None
                entry_cycle = None
                entry_i = -1

    # force close at end (at close)
    if units > 0.0:
        sell_px = _apply_cost(float(c[-1]), fee_bps, slip_bps, "sell")
        cash += float(units * sell_px)
        net_pnl = (sell_px - entry_px) * units
        trades.append({
            "symbol": symbol,
            "cycle": int(entry_cycle) if entry_cycle is not None else None,
            "entry_ts": str(pd.to_datetime(entry_ts, utc=True)) if entry_ts is not None else None,
            "exit_ts": str(pd.to_datetime(ts[-1], utc=True)),
            "entry_px": float(entry_px),
            "exit_px": float(sell_px),
            "units": float(units),
            "reason": "eod",
            "net_pnl": float(net_pnl),
            "hold_hours": float(max(0, len(df) - 1 - entry_i)),
        })
        units = 0.0

    final_equity = float(cash)
    net_profit = float(final_equity - float(initial_equity))

    eq = np.array(equity_curve, dtype=float) if equity_curve else np.array([initial_equity], dtype=float)
    runmax = np.maximum.accumulate(eq)
    dd = (runmax - eq) / np.maximum(runmax, 1e-9)
    max_dd = float(dd.max()) if dd.size else 0.0

    pnls = np.array([t["net_pnl"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]

    win_rate = float((pnls > 0.0).mean() * 100.0) if pnls.size else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 1e-9 else (float("inf") if wins.size else 0.0)

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "net_profit": float(net_profit),
        "trades": float(len(trades)),
        "wins": float((pnls > 0.0).sum()) if pnls.size else 0.0,
        "losses": float((pnls < 0.0).sum()) if pnls.size else 0.0,
        "win_rate_pct": float(win_rate),
        "profit_factor": float(pf) if np.isfinite(pf) else float("inf"),
        "max_dd_pct": float(max_dd),
        "min_pnl": float(pnls.min()) if pnls.size else 0.0,
        "max_pnl": float(pnls.max()) if pnls.size else 0.0,
        "cycle_shift": float(cycle_shift),
        "fee_bps": float(fee_bps),
        "slip_bps": float(slip_bps),
    }
    return trades, metrics


# ----------------------------
# Reporting
# ----------------------------
def _pf_from_pnls(pnls: np.ndarray) -> float:
    wins = pnls[pnls > 0.0].sum()
    losses = pnls[pnls < 0.0].sum()
    if losses < 0:
        return float(wins / abs(losses)) if abs(losses) > 1e-12 else float("inf")
    return float("inf") if wins > 0 else 0.0


def _print_cycle_table(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("\n--- BY CYCLE ---\n(no trades)")
        return
    g = trades_df.groupby("cycle")["net_pnl"]
    out = pd.DataFrame({
        "trades": g.size(),
        "net": g.sum(),
        "winrate_pct": trades_df.assign(win=trades_df["net_pnl"] > 0).groupby("cycle")["win"].mean() * 100.0,
        "pf": trades_df.groupby("cycle")["net_pnl"].apply(lambda x: _pf_from_pnls(x.to_numpy(dtype=float))),
    }).reset_index().sort_values("cycle")
    print("\n--- BY CYCLE ---")
    print(out.to_string(index=False))


def _print_yearly_table(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("\n--- YEARLY ---\n(no trades)")
        return
    trades_df["entry_year"] = pd.to_datetime(trades_df["entry_ts"], utc=True).dt.year
    g = trades_df.groupby("entry_year")["net_pnl"]
    out = pd.DataFrame({
        "trades": g.size(),
        "net": g.sum(),
        "avg": g.mean(),
        "winrate_pct": trades_df.assign(win=trades_df["net_pnl"] > 0).groupby("entry_year")["win"].mean() * 100.0,
        "pf": trades_df.groupby("entry_year")["net_pnl"].apply(lambda x: _pf_from_pnls(x.to_numpy(dtype=float))),
    }).reset_index().rename(columns={"entry_year": "year"}).sort_values("year")
    print("\n--- YEARLY ---")
    print(out.to_string(index=False))


def _ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize column names if needed
    rename = {}
    for a, b in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
        if a in df.columns and b not in df.columns:
            rename[a] = b
    if "timestamp" in df.columns and "Timestamp" not in df.columns:
        rename["timestamp"] = "Timestamp"
    if rename:
        df = df.rename(columns=rename)

    needed = {"Timestamp", "Open", "High", "Low", "Close"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Data missing columns {miss}. Has: {list(df.columns)[:30]}")
    return df


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--tf", default="1h", help="Timeframe label used in filename (default: 1h)")
    ap.add_argument("--params", default=None, help="Optional params json path to use for ALL symbols")
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--override-cycles", default=None, help="Override trade cycles, e.g. 3 or 1,3")
    ap.add_argument("--cycle-shift", type=int, default=1, help="Shift computed cycles forward by N bars (strict NLA: 1).")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        raise SystemExit("No symbols provided")

    run_tag = _utc_tag()

    for symbol in syms:
        # ---- params ----
        params_fp = Path(args.params) if args.params else _find_params(symbol)
        p = _read_json(params_fp)

        # ---- normalize ----
        from src.bot087.optim.ga import _norm_params, _ensure_indicators
        p = _norm_params(p)

        if args.override_cycles:
            cyc = [int(x.strip()) for x in str(args.override_cycles).split(",") if x.strip()]
            p["trade_cycles"] = cyc
            p["require_trade_cycles"] = True

        # ---- data ----
        df = _load_df(symbol, tf=args.tf)
        df = _ensure_ohlc_cols(df)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        if args.start:
            start_ts = pd.to_datetime(args.start, utc=True)
            df = df[df["Timestamp"] >= start_ts].reset_index(drop=True)
        if args.end:
            end_ts = pd.to_datetime(args.end, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df["Timestamp"] <= end_ts].reset_index(drop=True)

        if df.empty:
            print(f"\n=== {symbol} (NO LOOKAHEAD) ===")
            print("No data after filtering.")
            continue

        # indicators
        df = _ensure_indicators(df, p)

        # ---- backtest ----
        trades, m = run_backtest_nla(
            df=df,
            symbol=symbol,
            p=p,
            initial_equity=float(args.initial),
            fee_bps=float(args.fee_bps),
            slip_bps=float(args.slip_bps),
            cycle_shift=int(args.cycle_shift),
        )

        # ---- save ----
        out_dir = PROJECT_ROOT / "output" / "backtests_nla" / symbol / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(out_dir / "trades.csv", index=False)

        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

        # ---- print ----
        print(f"\n=== {symbol} (NO LOOKAHEAD) ===")
        print(f"Params: {params_fp}")
        print(f"Saved: {out_dir}")
        for k in ["trades", "wins", "losses", "win_rate_pct", "net_profit", "profit_factor", "max_dd_pct", "final_equity", "cycle_shift"]:
            if k in m:
                print(f"{k}: {m[k]}")

        _print_cycle_table(trades_df.copy())
        _print_yearly_table(trades_df.copy())


if __name__ == "__main__":
    main()
