#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only

CYCLE_LIST_KEYS = ["willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"]


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_dir(project_root: Path, run_id: str) -> Path:
    base = project_root / "artifacts" / "ga"
    candidates = list(base.glob(f"*/runs/{run_id}"))
    if not candidates:
        # deeper search just in case
        candidates = [p for p in base.rglob(run_id) if p.is_dir() and p.parent.name == "runs"]
    if not candidates:
        raise FileNotFoundError(f"Could not find run_id={run_id} under {base}")
    if len(candidates) > 1:
        # pick the shortest path (usually the direct one)
        candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]


def _list_gen_files(run_dir: Path) -> Dict[int, Path]:
    """
    Supports gen_003.json and gen_3.json and similar.
    Returns mapping gen -> file path.
    """
    out: Dict[int, Path] = {}
    for p in run_dir.glob("gen_*.json"):
        name = p.stem  # gen_003
        try:
            gen_str = name.split("gen_", 1)[1]
            gen = int(gen_str)
            out[gen] = p
        except Exception:
            continue
    return out


def _get_best_ind_from_genfile(gen_file: Path) -> Dict[str, Any]:
    j = _load_json(gen_file)
    return j["best"]["ind"]


def _merge_c1_c3(p1: Dict[str, Any], p3: Dict[str, Any]) -> Dict[str, Any]:
    a = _norm_params(dict(p1))
    b = _norm_params(dict(p3))

    out = dict(a)

    for k in CYCLE_LIST_KEYS:
        la = list(out.get(k, [0.0] * 5))
        lb = list(b.get(k, la))
        if len(la) != 5 or len(lb) != 5:
            raise ValueError(f"Bad per-cycle list len for {k}: {len(la)} {len(lb)}")
        la[3] = float(lb[3])  # cycle 3 from p3
        out[k] = la

    out["trade_cycles"] = [1, 3]
    out["require_trade_cycles"] = True

    # strict NLA
    out["cycle_shift"] = 1
    out["cycle_fill"] = 2
    out["two_candle_confirm"] = False

    return _norm_params(out)


def _params_hash(p: Dict[str, Any]) -> str:
    payload = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--data", default="data/processed/_full/BTCUSDT_1h_full.parquet")
    ap.add_argument("--run_c1", required=True)
    ap.add_argument("--run_c3", required=True)
    ap.add_argument("--gen_start", type=int, default=20)
    ap.add_argument("--gen_end", type=int, default=80)
    ap.add_argument("--fee_bps", type=float, default=7.0)
    ap.add_argument("--slip_bps", type=float, default=2.0)
    ap.add_argument("--initial_equity", type=float, default=10000.0)
    ap.add_argument("--out", default="", help="optional output CSV path")
    args = ap.parse_args()

    root = PROJECT_ROOT

    # locate run dirs anywhere under artifacts/ga/*/runs/<run_id>
    run_dir_c1 = _find_run_dir(root, args.run_c1)
    run_dir_c3 = _find_run_dir(root, args.run_c3)

    print(f"[sweep] run_c1 dir: {run_dir_c1}")
    print(f"[sweep] run_c3 dir: {run_dir_c3}")

    gens1 = _list_gen_files(run_dir_c1)
    gens3 = _list_gen_files(run_dir_c3)

    if not gens1:
        raise RuntimeError(f"No gen_*.json files found in {run_dir_c1}")
    if not gens3:
        raise RuntimeError(f"No gen_*.json files found in {run_dir_c3}")

    # intersection of available gens
    common = sorted(set(gens1.keys()) & set(gens3.keys()))
    common = [g for g in common if args.gen_start <= g <= args.gen_end]

    print(f"[sweep] gens in c1: min={min(gens1)} max={max(gens1)} count={len(gens1)}")
    print(f"[sweep] gens in c3: min={min(gens3)} max={max(gens3)} count={len(gens3)}")
    print(f"[sweep] common gens in range [{args.gen_start},{args.gen_end}]: {len(common)}")

    if not common:
        raise RuntimeError("No common generations between the two runs in the requested range. "
                           "Lower --gen_start or check run_ids.")

    df = pd.read_parquet(args.data)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected Timestamp column in 1h parquet")

    # indicators once
    df = _ensure_indicators(df, _norm_params({}))

    rows: List[Dict[str, Any]] = []
    metrics_cache: Dict[str, Dict[str, float]] = {}
    cache_hits = 0

    for gen in common:
        p1 = _get_best_ind_from_genfile(gens1[gen])
        p3 = _get_best_ind_from_genfile(gens3[gen])
        merged = _merge_c1_c3(p1, p3)
        key = _params_hash(merged)

        if key in metrics_cache:
            m = metrics_cache[key]
            cache_hits += 1
            reused = True
        else:
            _, m = run_backtest_long_only(
                df=df,
                symbol=args.symbol,
                p=merged,
                initial_equity=args.initial_equity,
                fee_bps=args.fee_bps,
                slippage_bps=args.slip_bps,
                collect_trades=False,
            )
            metrics_cache[key] = m
            reused = False

        rows.append({
            "gen": gen,
            "net": float(m["net_profit"]),
            "trades": float(m["trades"]),
            "win_rate": float(m["win_rate_pct"]),
            "pf": float(m["profit_factor"]),
            "max_dd": float(m["max_dd"]),
            "cached": int(reused),
        })

        tag = "cached" if reused else "eval"
        print(f"[gen {gen:03d}|{tag}] net={m['net_profit']:.2f} trades={m['trades']:.0f} "
              f"win={m['win_rate_pct']:.2f}% pf={m['profit_factor']:.2f} dd={m['max_dd']*100:.2f}%")

    out_df = pd.DataFrame(rows).sort_values(["net", "max_dd", "pf"], ascending=[False, True, False]).reset_index(drop=True)

    out_path = Path(args.out) if args.out else (root / "artifacts" / "ga" / args.symbol / "gen_sweep_c13.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if cache_hits:
        print(f"\n[sweep] cache hits: {cache_hits} / {len(common)}")

    print("\n=== TOP 10 by net ===")
    print(out_df.head(10).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
