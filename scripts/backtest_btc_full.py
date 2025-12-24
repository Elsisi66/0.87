import pandas as pd
from pathlib import Path

from src.bot087.datafeed.cache_loader import load_full_parquet
from src.bot087.strategy.param_store import load_active_params
from src.bot087.backtest.engine import run_backtest_long_only


def trades_to_df(trades):
    if not trades:
        return pd.DataFrame(columns=[
            "symbol", "cycle", "entry_ts", "entry_px", "exit_ts", "exit_px",
            "size", "reason", "gross_pnl", "entry_fee", "exit_fee", "net_pnl", "hold_hours"
        ])
    rows = []
    for t in trades:
        rows.append({
            "symbol": t.symbol,
            "cycle": t.cycle,
            "entry_ts": pd.to_datetime(t.entry_ts, utc=True),
            "entry_px": float(t.entry_px),
            "exit_ts": pd.to_datetime(t.exit_ts, utc=True),
            "exit_px": float(t.exit_px),
            "size": float(t.size),
            "reason": t.reason,
            "gross_pnl": float(t.gross_pnl),
            "entry_fee": float(t.entry_fee),
            "exit_fee": float(t.exit_fee),
            "net_pnl": float(t.net_pnl),
            "hold_hours": float(t.hold_hours),
        })
    return pd.DataFrame(rows)


def yearly_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    trades_df = trades_df.copy()
    trades_df["exit_year"] = trades_df["exit_ts"].dt.year
    trades_df["win"] = trades_df["net_pnl"] > 0

    g = trades_df.groupby("exit_year")
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate_pct": g["win"].mean() * 100.0,
        "net_profit": g["net_pnl"].sum(),
        "avg_net_pnl": g["net_pnl"].mean(),
        "avg_hold_hours": g["hold_hours"].mean(),
    }).reset_index()

    # profit factor (gross wins / gross losses abs)
    wins = g.apply(lambda x: x.loc[x["net_pnl"] > 0, "net_pnl"].sum())
    losses = g.apply(lambda x: x.loc[x["net_pnl"] <= 0, "net_pnl"].sum())
    out["profit_factor"] = (wins / (-losses)).replace([pd.NA, pd.NaT, float("inf")], 0.0).fillna(0.0)

    return out.sort_values("exit_year")


def main():
    symbol = "BTCUSDT"
    tf = "1h"

    df = load_full_parquet(symbol, tf=tf)
    p = load_active_params(symbol)

    trades, metrics = run_backtest_long_only(
        df,
        symbol=symbol,
        p=p,
        initial_equity=10_000.0,
        fee_bps=7.0,
    )

    print("=== OVERALL METRICS ===")
    print(metrics)
    print("n_trades:", len(trades))

    tdf = trades_to_df(trades)

    print("\n=== EXIT REASONS ===")
    if not tdf.empty:
        print(tdf["reason"].value_counts())

    print("\n=== YEARLY SUMMARY (by exit year) ===")
    y = yearly_summary(tdf)
    print(y.to_string(index=False))

    print("\n=== BEST 5 TRADES (net_pnl) ===")
    if not tdf.empty:
        print(tdf.sort_values("net_pnl", ascending=False).head(5)[
            ["entry_ts", "exit_ts", "reason", "net_pnl", "hold_hours"]
        ].to_string(index=False))

    print("\n=== WORST 5 TRADES (net_pnl) ===")
    if not tdf.empty:
        print(tdf.sort_values("net_pnl", ascending=True).head(5)[
            ["entry_ts", "exit_ts", "reason", "net_pnl", "hold_hours"]
        ].to_string(index=False))

    # Save trades for analysis
    out_dir = Path("artifacts") / "reports" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{symbol}_{tf}_full_trades.csv"
    tdf.to_csv(out_csv, index=False)
    print(f"\nSaved trades to: {out_csv}")


if __name__ == "__main__":
    main()
