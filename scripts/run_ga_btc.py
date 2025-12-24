# scripts/run_ga_btc.py
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_seed(symbol: str) -> Dict[str, Any]:
    """
    Tries:
      data/metadata/params/<SYMBOL>_active_params.json
      data/metadata/params/<SYMBOL>_seed_params.json
    Else falls back to a safe baseline (BTC-ish).
    """
    meta_dir = PROJECT_ROOT / "data" / "metadata" / "params"
    p1 = meta_dir / f"{symbol}_active_params.json"
    p2 = meta_dir / f"{symbol}_seed_params.json"

    for fp in [p1, p2]:
        if fp.exists():
            with open(fp, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
                return raw["params"]
            if isinstance(raw, dict):
                return raw

    # Hard fallback baseline (so it never crashes)
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
        "allow_hours": [1, 6, 8, 9, 12, 15, 17, 20, 21],
        "trade_cycles": [1, 2],
        "use_sig_baseline": True,
        "use_sig_breakout": False,
        "use_sig_pullback": False,
        "breakout_window": 20,
        "breakout_atr_mult": 1.5,
        "max_hold_hours": 48,
    }


def _load_df(symbol: str, tf: str) -> pd.DataFrame:
    """
    Preferred: data/processed/_full/<SYMBOL>_<TF>_full.parquet
    Fallback: concatenate yearly processed CSVs in data/processed/
    """
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    parquet_fp = full_dir / f"{symbol}_{tf}_full.parquet"

    if parquet_fp.exists():
        try:
            df = pd.read_parquet(parquet_fp)
            return df
        except Exception as e:
            print(f"[WARN] read_parquet failed ({e}). Falling back to yearly CSVs...", flush=True)

    # fallback to yearly processed
    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if not files:
        raise FileNotFoundError(f"No parquet and no yearly processed CSVs found for {symbol} under {proc_dir}")

    dfs: List[pd.DataFrame] = []
    for fp in files:
        d = pd.read_csv(fp)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    symbol = "BTCUSDT"
    tf = "1h"

    from src.bot087.optim.ga import GAConfig, run_ga_montecarlo

    df = _load_df(symbol, tf=tf)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    print(f"[DATA] rows={len(df)} first={df['Timestamp'].iloc[0]} last={df['Timestamp'].iloc[-1]}", flush=True)

    seed = _load_seed(symbol)

    cfg = GAConfig(
        pop_size=40,
        generations=35,
        elite_k=6,
        mutation_rate=0.35,
        mutation_strength=1.0,
        n_procs=3,

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

        two_candle_confirm=True,
        require_trade_cycles=True,

        # fitness knobs
        w_train=0.7,
        w_val=0.3,
        dd_penalty=0.45,
        trade_penalty=0.8,
        bad_val_penalty=1200.0,

        resume=True,
    )

    best_p, report = run_ga_montecarlo(symbol=symbol, df=df, seed_params=seed, cfg=cfg)

    print("\n=== DONE ===", flush=True)
    print("Saved to:", flush=True)
    for k, v in report["saved"].items():
        print(f"  {k}: {v}", flush=True)

    print("\nBest params (top-level):", flush=True)
    for k in sorted(best_p.keys()):
        if isinstance(best_p[k], list):
            print(f"  {k}: {best_p[k]}", flush=True)
        else:
            print(f"  {k}: {best_p[k]}", flush=True)


if __name__ == "__main__":
    main()
