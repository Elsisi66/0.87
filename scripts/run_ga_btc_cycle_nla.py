import argparse
from pathlib import Path
import json
import pandas as pd

from src.bot087.optim.ga import GAConfig, run_ga_montecarlo

ROOT = Path(__file__).resolve().parents[1]

def load_seed(seed_path: Path) -> dict:
    with open(seed_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True, choices=[1,2,3], help="Which cycle to optimize")
    ap.add_argument("--tf", type=str, default="1h")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--seed", type=str, default=str(ROOT/"artifacts/ga/BTCUSDT/best_params.json"))
    ap.add_argument("--data", type=str, default=str(ROOT/"data/processed/_full/BTCUSDT_1h_full.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if "Timestamp" not in df.columns:
        raise SystemExit("Parquet missing Timestamp column")

    seed = load_seed(Path(args.seed))

    # hard-focus cycle
    seed["trade_cycles"] = [int(args.cycle)]
    seed["require_trade_cycles"] = True

    # If you optimize cycle 1/2, breakout logic is usually noise.
    # Keep baseline/pullback enabled; GA can still flip them if you left mutations on.
    if int(args.cycle) in (1,2):
        seed["use_sig_baseline"] = True
        seed["use_sig_pullback"] = True
        seed["use_sig_breakout"] = False

    # IMPORTANT: isolate outputs by using an alias symbol
    alias = f"BTCUSDT_C{int(args.cycle)}"

    # Tune min trades so GA doesn't "win" with 3 lucky trades
    if int(args.cycle) == 2:
        min_tr_train, min_tr_val = 20, 8
    else:
        min_tr_train, min_tr_val = 40, 15

    cfg = GAConfig(
        pop_size=int(args.pop),
        generations=int(args.gen),
        n_procs=int(args.procs),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial),

        min_trades_train=min_tr_train,
        min_trades_val=min_tr_val,

        # keep resume ON for long runs
        resume=True,
    )

    best_params, report = run_ga_montecarlo(alias, df, seed, cfg)

    print("\n=== DONE ===")
    print("Alias:", alias)
    print("Best params saved:", report["saved"]["best_params"])
    print("Active params saved:", report["saved"]["active_params"])
    print("Run dir:", report["saved"]["run_dir"])

if __name__ == "__main__":
    main()
