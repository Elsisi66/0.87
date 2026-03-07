# scripts/eval_symbol_detailed.py
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only


def _load_df(symbol: str, tf: str) -> pd.DataFrame:
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    parquet_fp = full_dir / f"{symbol}_{tf}_full.parquet"
    if parquet_fp.exists():
        return pd.read_parquet(parquet_fp)

    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if not files:
        raise FileNotFoundError(f"No data found for {symbol}")
    dfs = [pd.read_csv(fp) for fp in files]
    return pd.concat(dfs, ignore_index=True)


def _load_active_params(symbol: str) -> Dict[str, Any]:
    fp = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    raw = json.load(open(fp, "r"))
    p = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
    return _norm_params(p)


def _equity_from_trades(trades_df: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    """
    Build an equity curve based on realized PnL only (close-to-close).
    Good enough for evaluation and drawdown/streak stats.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["time", "equity", "pnl", "drawdown"])

    t = trades_df.sort_values("exit_ts").copy()
    t["pnl"] = t["net_pnl"].astype(float)
    t["equity"] = initial_equity + t["pnl"].cumsum()
    t["peak"] = t["equity"].cummax()
    t["drawdown"] = (t["peak"] - t["equity"]) / t["peak"].replace(0, np.nan)
    return t[["exit_ts", "equity", "pnl", "drawdown"]].rename(columns={"exit_ts": "time"})


def _max_losing_streak(pnls: np.ndarray) -> Tuple[int, float]:
    max_len = 0
    max_sum = 0.0
    cur_len = 0
    cur_sum = 0.0
    for x in pnls:
        if x < 0:
            cur_len += 1
            cur_sum += x
            if cur_len > max_len:
                max_len = cur_len
                max_sum = cur_sum
        else:
            cur_len = 0
            cur_sum = 0.0
    return max_len, float(max_sum)


def _max_winning_streak(pnls: np.ndarray) -> Tuple[int, float]:
    max_len = 0
    max_sum = 0.0
    cur_len = 0
    cur_sum = 0.0
    for x in pnls:
        if x > 0:
            cur_len += 1
            cur_sum += x
            if cur_len > max_len:
                max_len = cur_len
                max_sum = cur_sum
        else:
            cur_len = 0
            cur_sum = 0.0
    return max_len, float(max_sum)


def _agg_report(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnls = trades["net_pnl"].astype(float).to_numpy()
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float((pnls > 0).mean() * 100.0)
    pf = float(wins.sum() / abs(losses.sum())) if losses.size else (10.0 if wins.size else 0.0)
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    max_ls_len, max_ls_sum = _max_losing_streak(pnls)
    max_ws_len, max_ws_sum = _max_winning_streak(pnls)

    return {
        "trades": int(len(trades)),
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_losing_streak_len": int(max_ls_len),
        "max_losing_streak_sum": float(max_ls_sum),
        "max_winning_streak_len": int(max_ws_len),
        "max_winning_streak_sum": float(max_ws_sum),
    }


def _run_one(symbol: str, tf: str, fee_bps: float, slippage_bps: float, initial_equity: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = _load_active_params(symbol)
    df = _load_df(symbol, tf)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df = _ensure_indicators(df, p)

    trades_list, m = run_backtest_long_only(
        df, symbol=symbol, p=p,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps
    )
    trades = pd.DataFrame(trades_list)
    if trades.empty:
        return trades, {"metrics": m, "extras": _agg_report(trades)}

    # Normalize timestamps
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")

    # Add derived columns
    trades["gross_notional"] = trades["entry_px"].astype(float) * trades["units"].astype(float)
    trades["pnl_pct"] = trades["net_pnl"].astype(float) / trades["gross_notional"].replace(0, np.nan)
    trades["win"] = trades["net_pnl"].astype(float) > 0

    # Attach entry ATR/RSI at entry time (nearest match)
    idx = pd.Index(df["Timestamp"])
    entry_pos = idx.get_indexer(trades["entry_ts"], method="nearest")
    entry_pos = np.clip(entry_pos, 0, len(df) - 1)
    trades["entry_i"] = entry_pos

    # Use ATR/RSI from df at entry index (ATR is already computed; your strategy uses prev features but this is fine for eval)
    trades["entry_atr"] = df["ATR"].astype(float).to_numpy()[entry_pos]
    trades["entry_rsi"] = df["RSI"].astype(float).to_numpy()[entry_pos]

    # Time buckets
    trades["year"] = trades["exit_ts"].dt.year
    trades["month"] = trades["exit_ts"].dt.to_period("M").astype(str)

    extras = _agg_report(trades)
    return trades, {"metrics": m, "extras": extras}


def main():
    # ---- EDIT THESE ----
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1h"
    initial_equity = 10_000.0

    # cost scenarios you want to stress
    scenarios = [
        ("base_7_2", 7.0, 2.0),
        ("stress_14_4", 14.0, 4.0),
        ("stress_20_6", 20.0, 6.0),
    ]

    out_dir = PROJECT_ROOT / "artifacts" / "eval" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for tag, fee_bps, slip_bps in scenarios:
        trades, rep = _run_one(symbol, tf, fee_bps, slip_bps, initial_equity)
        metrics = rep["metrics"]
        extras = rep["extras"]

        # Save trades
        trades_fp = out_dir / f"trades_{tag}.csv"
        trades.to_csv(trades_fp, index=False)

        # Equity curve (realized)
        eq = _equity_from_trades(trades, initial_equity)
        eq_fp = out_dir / f"equity_{tag}.csv"
        eq.to_csv(eq_fp, index=False)

        # Group reports
        if not trades.empty:
            by_year = trades.groupby("year").agg(
                trades=("net_pnl", "size"),
                net=("net_pnl", "sum"),
                win_rate=("win", "mean"),
                pf=("net_pnl", lambda x: (x[x>0].sum() / abs(x[x<0].sum())) if (x[x<0].sum() != 0) else (10.0 if (x[x>0].sum() > 0) else 0.0)),
                avg_pnl=("net_pnl", "mean"),
                avg_hold=("hold_hours", "mean"),
            ).reset_index()
            by_year["win_rate"] = by_year["win_rate"] * 100.0
            by_year.to_csv(out_dir / f"by_year_{tag}.csv", index=False)

            by_month = trades.groupby("month").agg(
                trades=("net_pnl", "size"),
                net=("net_pnl", "sum"),
                win_rate=("win", "mean"),
                pf=("net_pnl", lambda x: (x[x>0].sum() / abs(x[x<0].sum())) if (x[x<0].sum() != 0) else (10.0 if (x[x>0].sum() > 0) else 0.0)),
            ).reset_index()
            by_month["win_rate"] = by_month["win_rate"] * 100.0
            by_month.to_csv(out_dir / f"by_month_{tag}.csv", index=False)

            by_cycle = trades.groupby("cycle").agg(
                trades=("net_pnl", "size"),
                net=("net_pnl", "sum"),
                win_rate=("win", "mean"),
                pf=("net_pnl", lambda x: (x[x>0].sum() / abs(x[x<0].sum())) if (x[x<0].sum() != 0) else (10.0 if (x[x>0].sum() > 0) else 0.0)),
                avg_hold=("hold_hours", "mean"),
            ).reset_index()
            by_cycle["win_rate"] = by_cycle["win_rate"] * 100.0
            by_cycle.to_csv(out_dir / f"by_cycle_{tag}.csv", index=False)

            by_reason = trades.groupby("reason").agg(
                trades=("net_pnl", "size"),
                net=("net_pnl", "sum"),
                win_rate=("win", "mean"),
                avg_hold=("hold_hours", "mean"),
            ).reset_index()
            by_reason["win_rate"] = by_reason["win_rate"] * 100.0
            by_reason.to_csv(out_dir / f"by_reason_{tag}.csv", index=False)

            # Worst drawdown trade-window (realized) - show top 10 worst points
            if not eq.empty:
                worst = eq.sort_values("drawdown", ascending=False).head(10)
                worst.to_csv(out_dir / f"worst_drawdown_points_{tag}.csv", index=False)

        row = {
            "scenario": tag,
            "fee_bps": fee_bps,
            "slippage_bps": slip_bps,
            **{f"m_{k}": v for k, v in metrics.items()},
            **{f"x_{k}": v for k, v in extras.items()},
        }
        summary_rows.append(row)

        # Print headline
        print(f"\n=== {symbol} {tf} | {tag} ===")
        print("metrics:", metrics)
        print("extras:", extras)
        print("saved:", str(out_dir))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Dump best params used
    params_fp = out_dir / "active_params_used.json"
    json.dump(_load_active_params(symbol), open(params_fp, "w"), indent=2)

    print("\nDONE. Reports in:", out_dir)


if __name__ == "__main__":
    main()
