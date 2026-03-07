import argparse
from pathlib import Path
import pandas as pd

from src.bot087.optim.ga_short import GAConfig, run_ga_montecarlo


def parse_cycles(s: str):
    s = (s or "").strip()
    if not s:
        return [0, 4]
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="BTCUSDT", help="Symbol like BTCUSDT / ADAUSDT / SOLUSDT")
    ap.add_argument("--cycles", default="0,4", help="Trade cycles for short (default: 0,4)")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    args = ap.parse_args()

    root = Path("/root/analysis/0.87")
    sym = args.sym.upper()
    cycles = parse_cycles(args.cycles)

    parquet = root / "data" / "processed" / "_full" / f"{sym}_1h_full.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing parquet: {parquet}")

    df = pd.read_parquet(parquet)
    if "Timestamp" not in df.columns:
        raise SystemExit("Parquet missing Timestamp column")

    # Seed params (SHORT defaults)
    seed = {
        # per-cycle WILLR thresholds for SHORT: require WILLR > threshold (closer to 0)
        # start around -25 and let GA move it
        "willr_by_cycle": [-25.0, -25.0, -25.0, -25.0, -25.0],

        # per-cycle TP/SL for SHORT:
        # TP < 1, SL > 1
        "tp_mult_by_cycle": [0.94, 0.94, 0.94, 0.94, 0.94],
        "sl_mult_by_cycle": [1.03, 1.03, 1.03, 1.03, 1.03],

        # exit RSI after profit (rebound exit)
        "exit_rsi_by_cycle": [55.0, 55.0, 55.0, 55.0, 55.0],

        # trend filters
        "ema_span": 35,
        "ema_trend_long": 120,
        "ema_align": True,
        "require_ema200_slope": True,
        "adx_min": 18.0,
        "require_minus_di": True,

        # RSI band (short prefers higher RSI)
        "entry_rsi_min": 55.0,
        "entry_rsi_max": 82.0,

        # risk
        "max_hold_hours": 48,
        "risk_per_trade": 0.02,
        "max_allocation": 0.7,
        "atr_k": 1.0,

        # cycles
        "trade_cycles": cycles,

        # strict NLA
        "cycle_shift": 1,
        "cycle_fill": 2,

        # no 2-candle confirm
        "two_candle_confirm": False,
        "require_trade_cycles": True,
    }

    cfg = GAConfig(
        pop_size=args.pop,
        generations=args.gen,
        n_procs=args.procs,
        resume=bool(args.resume),

        # strict NLA
        cycle_shift=1,
        cycle_fill=2,

        # keep it off
        two_candle_confirm=False,
    )

    print(f"=== GA SHORT (NLA) {sym} cycles={cycles} ===", flush=True)
    print(f"rows: {len(df)}", flush=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    print("time:", df["Timestamp"].min(), "->", df["Timestamp"].max(), flush=True)

    best, report = run_ga_montecarlo(sym, df, seed, cfg)
    print("DONE")
    print("best saved:", report["saved"]["best_params"])
    print("active saved:", report["saved"]["active_params"])


if __name__ == "__main__":
    main()
