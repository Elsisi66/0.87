#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_dir(project_root: Path, run_id: str) -> Path:
    base = project_root / "artifacts" / "ga"
    candidates = list(base.glob(f"*/runs/{run_id}"))
    if not candidates:
        candidates = [p for p in base.rglob(run_id) if p.is_dir() and p.parent.name == "runs"]
    if not candidates:
        raise FileNotFoundError(f"Could not find run_id={run_id} under {base}")
    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]


def _list_gen_files(run_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in run_dir.glob("gen_*.json"):
        try:
            gen = int(p.stem.split("gen_", 1)[1])
            out[gen] = p
        except Exception:
            continue
    return out


def _get_best_ind(gen_file: Path) -> Dict[str, Any]:
    j = _load_json(gen_file)
    return j["best"]["ind"]


def _force_cycle(p: Dict[str, Any], cycle: Optional[int]) -> Dict[str, Any]:
    p = _norm_params(dict(p))

    # Strict NLA defaults (keep you honest)
    p["cycle_shift"] = 1
    p["cycle_fill"] = 2
    p["two_candle_confirm"] = False
    p["require_trade_cycles"] = True

    if cycle is not None:
        p["trade_cycles"] = [int(cycle)]

    return _norm_params(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--cycle", type=int, default=None, help="force trade_cycles=[cycle]")
    ap.add_argument("--gen_start", type=int, default=20)
    ap.add_argument("--gen_end", type=int, default=80)
    ap.add_argument("--fee_bps", type=float, default=7.0)
    ap.add_argument("--slip_bps", type=float, default=2.0)
    ap.add_argument("--initial_equity", type=float, default=10000.0)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_best_json", default="")
    args = ap.parse_args()

    root = Path.cwd().resolve()
    run_dir = _find_run_dir(root, args.run_id)
    gens = _list_gen_files(run_dir)
    if not gens:
        raise RuntimeError(f"No gen_*.json files found in {run_dir}")

    gens_in_range = sorted([g for g in gens.keys() if args.gen_start <= g <= args.gen_end])
    if not gens_in_range:
        raise RuntimeError("No generations found in requested range.")

    print(f"[sweep] run_dir: {run_dir}")
    print(f"[sweep] gens: min={min(gens)} max={max(gens)} count={len(gens)}")
    print(f"[sweep] eval gens in [{args.gen_start},{args.gen_end}]: {len(gens_in_range)}")

    df = pd.read_parquet(args.data)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected Timestamp column in 1h parquet")

    # build indicators once
    df = _ensure_indicators(df, _norm_params({}))

    rows: List[Dict[str, Any]] = []
    best_row = None
    best_params = None

    for gen in gens_in_range:
        ind = _get_best_ind(gens[gen])
        ind = _force_cycle(ind, args.cycle)

        _, m = run_backtest_long_only(
            df=df,
            symbol=args.symbol,
            p=ind,
            initial_equity=args.initial_equity,
            fee_bps=args.fee_bps,
            slippage_bps=args.slip_bps,
        )

        row = {
            "gen": gen,
            "net": float(m["net_profit"]),
            "trades": float(m["trades"]),
            "win_rate": float(m["win_rate_pct"]),
            "pf": float(m["profit_factor"]),
            "max_dd": float(m["max_dd"]),
        }
        rows.append(row)

        if best_row is None or row["net"] > best_row["net"]:
            best_row = row
            best_params = ind

        print(
            f"[gen {gen:03d}] net={row['net']:.2f} trades={row['trades']:.0f} "
            f"win={row['win_rate']:.2f}% pf={row['pf']:.2f} dd={row['max_dd']*100:.2f}%"
        )

    out_df = pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)

    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / f"gen_sweep_cycle{args.cycle or 'NA'}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print("\n=== TOP 10 by net ===")
    print(out_df.head(10).to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")

    if best_params is not None:
        out_best = Path(args.out_best_json) if args.out_best_json else (run_dir / f"best_by_sweep_cycle{args.cycle or 'NA'}.json")
        with open(out_best, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved BEST params (by net): {out_best}")
        print(f"Best row: {best_row}")


if __name__ == "__main__":
    main()
