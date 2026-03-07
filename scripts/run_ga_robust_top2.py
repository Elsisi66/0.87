import argparse
import json
from pathlib import Path

import pandas as pd

from src.bot087.optim.ga import GAConfig, run_ga_montecarlo


PROJECT_ROOT = Path("/root/analysis/0.87").resolve()


def load_seed_params(symbol: str) -> dict:
    # Prefer your current active params store
    fp = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    if fp.exists():
        raw = json.loads(fp.read_text())
        return raw["params"] if isinstance(raw, dict) and "params" in raw else raw

    # Fallback: GA best params if present
    fp2 = PROJECT_ROOT / "artifacts" / "ga" / symbol / "best_params.json"
    if fp2.exists():
        return json.loads(fp2.read_text())

    raise FileNotFoundError(f"No seed params found for {symbol} in data/metadata/params or artifacts/ga.")


def load_df(symbol: str, tf: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet: {fp}")
    return pd.read_parquet(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated, e.g. ADAUSDT,SOLUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--fee-bps", type=float, default=25.0)   # STRESS by default
    ap.add_argument("--slip-bps", type=float, default=10.0)  # STRESS by default
    ap.add_argument("--cycle-shift", type=int, default=1)    # strict NLA cycles
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        print(f"\n=== ROBUST GA: {sym} | tf={args.tf} | fee={args.fee_bps} slip={args.slip_bps} | cycle_shift={args.cycle_shift} ===", flush=True)

        df = load_df(sym, args.tf)
        seed = load_seed_params(sym)

        # enforce strict NLA cycles for the whole GA run
        seed["cycle_shift"] = int(args.cycle_shift)

        cfg = GAConfig(
            pop_size=args.pop,
            generations=args.gens,
            n_procs=args.procs,

            # Optimize under STRESS costs so it doesn’t die instantly
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),

            # keep default constraints unless you know what you're doing
            resume=True,
        )

        best, report = run_ga_montecarlo(sym, df, seed, cfg)

        out_root = PROJECT_ROOT / "artifacts" / "ga" / sym
        print(f"[DONE] {sym} best saved to: {out_root / 'best_params.json'}", flush=True)
        print(f"[DONE] {sym} runs in:       {out_root / 'runs'}", flush=True)
        print(f"[DONE] {sym} ACTIVE at:     {PROJECT_ROOT / 'data' / 'metadata' / 'params' / f'{sym}_active_params.json'}", flush=True)


if __name__ == "__main__":
    main()
