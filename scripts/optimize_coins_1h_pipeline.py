#!/usr/bin/env python3
"""
Optimize 1h long-only strategies for multiple coins with strict no-lookahead rules.

How to run:
  scripts/venv/bin/python scripts/optimize_coins_1h_pipeline.py \
    --n-procs 3 --pop-size 48 --generations 120 --mc-splits 6

  # Include BTC as read-only baseline row at the end (never writes BTC params)
  scripts/venv/bin/python scripts/optimize_coins_1h_pipeline.py \
    --include-btc-read-only

Notes:
- Auto-discovers symbols from data/processed/_full/*_1h_full.parquet.
- Excludes BTC by default.
- Writes per-coin run folder: artifacts/ga/<SYMBOL>/<RUN_ID>/
- Appends generation history to: artifacts/ga/all_coins_history.csv
- Appends phase summary rows to: artifacts/ga/all_coins_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim.ga import (  # noqa: E402
    GAConfig,
    _ensure_indicators,
    _norm_params,
    build_entry_signal,
    run_ga_montecarlo,
    run_backtest_long_only,
)

CYCLE_ARRAY_KEYS = ["willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"]
PHASE_SUMMARY_COLUMNS = [
    "symbol",
    "phase",
    "val_net",
    "val_pf",
    "val_dd",
    "val_trades",
    "stability",
    "test_net",
    "fitness",
    "PASS/FAIL",
    "run_id",
    "timestamp",
]
HISTORY_COLUMNS = [
    "symbol",
    "phase",
    "run_id",
    "timestamp",
    "gen",
    "fitness",
    "train_net",
    "train_pf",
    "train_dd",
    "train_trades",
    "val_net",
    "val_pf",
    "val_dd",
    "val_trades",
    "test_net",
    "test_pf",
    "test_dd",
    "test_trades",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_rows_csv(path: Path, rows: Iterable[Dict[str, Any]], ordered_columns: Optional[List[str]] = None) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if ordered_columns is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        ordered_columns = seen

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def discover_symbols(full_dir: Path, include_btc: bool = False) -> List[str]:
    out: List[str] = []
    for fp in sorted(full_dir.glob("*_1h_full.parquet")):
        stem = fp.name
        if "_" not in stem:
            continue
        symbol = stem.split("_", 1)[0].upper()
        if symbol == "BTCUSDT" and not include_btc:
            continue
        out.append(symbol)
    return sorted(set(out))


def load_df_1h(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {fp}")
    df = pd.read_parquet(fp)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing Timestamp column in {fp}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{symbol}: empty dataframe after Timestamp parse")
    return df


def _unwrap_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def load_seed_params(symbol: str) -> Dict[str, Any]:
    params_dir = PROJECT_ROOT / "data" / "metadata" / "params"
    candidates = [
        params_dir / f"{symbol}_active_params.json",
        params_dir / f"{symbol}_seed_params.json",
    ]
    for fp in candidates:
        if fp.exists():
            return _norm_params(_unwrap_params(_read_json(fp)))

    return _norm_params(
        {
            "entry_rsi_min": 52.0,
            "entry_rsi_max": 64.0,
            "entry_rsi_buffer": 2.5,
            "willr_floor": -100.0,
            "willr_by_cycle": [-78.0, -32.0, -90.0, -18.0, -50.0],
            "ema_span": 35,
            "ema_trend_long": 120,
            "ema_align": True,
            "require_ema200_slope": True,
            "adx_min": 18.0,
            "require_plus_di": True,
            "tp_mult_by_cycle": [1.035, 1.08, 1.05, 1.07, 1.05],
            "sl_mult_by_cycle": [0.985, 0.98, 0.96, 0.99, 0.97],
            "exit_rsi_by_cycle": [50.0, 56.0, 50.0, 62.0, 52.0],
            "risk_per_trade": 0.02,
            "max_allocation": 0.7,
            "atr_k": 1.0,
            "trade_cycles": [1, 3],
            "max_hold_hours": 48,
            "cycle_shift": 1,
            "cycle_fill": 2,
            "two_candle_confirm": False,
            "require_trade_cycles": True,
        }
    )


def enforce_nla_and_validate(df: pd.DataFrame, params: Dict[str, Any], symbol: str) -> pd.DataFrame:
    p = _norm_params(dict(params))
    p["cycle_shift"] = 1
    p["two_candle_confirm"] = False
    p["require_trade_cycles"] = True

    dfi = _ensure_indicators(df.copy(), p)
    required = ["RSI", "ATR", "WILLR", "ADX", "PLUS_DI", "MINUS_DI", "EMA_200", "EMA_200_SLOPE"]
    missing = [c for c in required if c not in dfi.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required indicators: {missing}")

    nan_cols = [c for c in required if dfi[c].isna().any()]
    if nan_cols:
        raise ValueError(f"{symbol}: NaNs in required indicators: {nan_cols}")

    sig = build_entry_signal(dfi, p)
    if len(sig) != len(dfi):
        raise ValueError(f"{symbol}: signal length mismatch (sig={len(sig)} df={len(dfi)})")

    return dfi


@dataclass
class Paths:
    root: Path
    run_id: str
    all_history_csv: Path
    all_report_csv: Path


@dataclass
class PhaseResult:
    symbol: str
    phase: str
    trade_cycles: List[int]
    best_params: Dict[str, Any]
    report: Dict[str, Any]
    phase_dir: Path


def build_paths(run_id: str) -> Paths:
    ga_root = PROJECT_ROOT / "artifacts" / "ga"
    ga_root.mkdir(parents=True, exist_ok=True)
    return Paths(
        root=ga_root,
        run_id=run_id,
        all_history_csv=ga_root / "all_coins_history.csv",
        all_report_csv=ga_root / "all_coins_report.csv",
    )


def init_coin_run_dir(paths: Paths, symbol: str, args: argparse.Namespace, df: pd.DataFrame) -> Path:
    run_dir = paths.root / symbol / paths.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config_used.json",
        {
            "timestamp_utc": _utc_now_iso(),
            "symbol": symbol,
            "run_id": paths.run_id,
            "args": vars(args),
            "data": {
                "rows": int(len(df)),
                "start": str(df["Timestamp"].min()),
                "end": str(df["Timestamp"].max()),
            },
            "fees": {
                "fee_bps": float(args.fee_bps),
                "slippage_bps": float(args.slippage_bps),
            },
            "nla": {
                "cycle_shift": 1,
                "entry_rule": "t-1 features -> bar t Open",
            },
        },
    )
    return run_dir


def _phase_seed(base_params: Dict[str, Any], trade_cycles: List[int]) -> Dict[str, Any]:
    p = _norm_params(dict(base_params))
    p["trade_cycles"] = [int(x) for x in trade_cycles]
    p["require_trade_cycles"] = True
    p["two_candle_confirm"] = False
    p["cycle_shift"] = 1
    p["cycle_fill"] = int(p.get("cycle_fill", 2))
    return _norm_params(p)


def _ga_cfg_from_args(args: argparse.Namespace) -> GAConfig:
    return GAConfig(
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        elite_k=6,
        mutation_rate=0.35,
        mutation_strength=1.0,
        n_procs=int(args.n_procs),
        mc_splits=int(args.mc_splits),
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        initial_equity=10_000.0,
        min_trades_train=40,
        min_trades_val=15,
        two_candle_confirm=False,
        require_trade_cycles=True,
        cycle_shift=1,
        cycle_fill=2,
        w_train=0.7,
        w_val=0.3,
        dd_penalty=0.45,
        trade_penalty=0.8,
        bad_val_penalty=1200.0,
        resume=False,
        early_stop_patience=int(args.early_stop_patience),
    )


def _safe_alias_symbol(symbol: str, phase: str) -> str:
    return f"{symbol}__{phase.upper()}"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _snapshot_phase_artifacts(report: Dict[str, Any], phase_dir: Path) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(str(report["saved"]["run_dir"]))
    if run_dir.exists():
        for fp in sorted(run_dir.glob("gen_*.json")):
            shutil.copy2(fp, phase_dir / fp.name)
        for fp in sorted(run_dir.glob("best_gen_*.json")):
            shutil.copy2(fp, phase_dir / fp.name)
        _copy_if_exists(run_dir / "final_report.json", phase_dir / "final_report.json")

    saved = report.get("saved", {})
    if "history_csv" in saved:
        _copy_if_exists(Path(str(saved["history_csv"])), phase_dir / "ga_history.csv")
    if "checkpoint_latest" in saved:
        _copy_if_exists(Path(str(saved["checkpoint_latest"])), phase_dir / "checkpoint_latest.json")


def _history_rows_from_phase(symbol: str, phase: str, run_id: str, phase_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(phase_dir.glob("gen_*.json")):
        payload = _read_json(fp)
        best = payload.get("best", {})
        avg = best.get("avg", {})
        tr = avg.get("train", {})
        va = avg.get("val", {})
        te = avg.get("test", {})
        rows.append(
            {
                "symbol": symbol,
                "phase": phase,
                "run_id": run_id,
                "timestamp": payload.get("utc", _utc_now_iso()),
                "gen": int(payload.get("gen", -1)),
                "fitness": float(best.get("fitness", 0.0)),
                "train_net": float(tr.get("net", 0.0)),
                "train_pf": float(tr.get("pf", 0.0)),
                "train_dd": float(tr.get("dd", 0.0)),
                "train_trades": float(tr.get("trades", 0.0)),
                "val_net": float(va.get("net", 0.0)),
                "val_pf": float(va.get("pf", 0.0)),
                "val_dd": float(va.get("dd", 0.0)),
                "val_trades": float(va.get("trades", 0.0)),
                "test_net": float(te.get("net", 0.0)),
                "test_pf": float(te.get("pf", 0.0)),
                "test_dd": float(te.get("dd", 0.0)),
                "test_trades": float(te.get("trades", 0.0)),
            }
        )
    return rows


def _run_phase_ga(
    symbol: str,
    phase: str,
    trade_cycles: List[int],
    seed_params: Dict[str, Any],
    df: pd.DataFrame,
    args: argparse.Namespace,
    coin_run_dir: Path,
    paths: Paths,
) -> PhaseResult:
    phase_seed = _phase_seed(seed_params, trade_cycles=trade_cycles)
    phase_dir = coin_run_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ga_cfg_from_args(args)
    alias_symbol = _safe_alias_symbol(symbol, phase)

    _write_json(
        phase_dir / "phase_config.json",
        {
            "timestamp_utc": _utc_now_iso(),
            "symbol": symbol,
            "phase": phase,
            "alias_symbol": alias_symbol,
            "trade_cycles": trade_cycles,
            "cfg": asdict(cfg),
            "seed_params": phase_seed,
        },
    )

    if args.dry_run:
        empty_report = {"symbol": alias_symbol, "run_id": "dryrun", "saved": {}, "best_overall": {"ind": phase_seed}}
        _write_json(phase_dir / "final_report.json", empty_report)
        return PhaseResult(
            symbol=symbol,
            phase=phase,
            trade_cycles=list(trade_cycles),
            best_params=phase_seed,
            report=empty_report,
            phase_dir=phase_dir,
        )

    best_params, report = run_ga_montecarlo(
        symbol=alias_symbol,
        df=df,
        seed_params=phase_seed,
        cfg=cfg,
    )
    best_params = _phase_seed(best_params, trade_cycles=trade_cycles)
    _snapshot_phase_artifacts(report, phase_dir)
    _write_json(phase_dir / "best_params.json", {"symbol": symbol, "phase": phase, "params": best_params})

    hist_rows = _history_rows_from_phase(symbol, phase, paths.run_id, phase_dir)
    _append_rows_csv(paths.all_history_csv, hist_rows, ordered_columns=HISTORY_COLUMNS)

    if args.cleanup_temp_ga:
        saved = report.get("saved", {})
        run_dir = Path(str(saved.get("run_dir", "")))
        history_csv = Path(str(saved.get("history_csv", "")))
        best_params_fp = Path(str(saved.get("best_params", "")))
        ckpt_fp = Path(str(saved.get("checkpoint_latest", "")))
        active_fp = Path(str(saved.get("active_params", "")))
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        for fp in [history_csv, best_params_fp, ckpt_fp, active_fp]:
            if fp.exists():
                try:
                    fp.unlink()
                except Exception:
                    pass

    return PhaseResult(
        symbol=symbol,
        phase=phase,
        trade_cycles=list(trade_cycles),
        best_params=best_params,
        report=report,
        phase_dir=phase_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbol list. Default: auto-discover from data/processed/_full")
    ap.add_argument("--include-btc-read-only", action="store_true", help="Append BTC summary row using existing params only")
    ap.add_argument("--run-id", default="", help="Optional fixed run id; default UTC timestamp")
    ap.add_argument("--n-procs", type=int, default=3)
    ap.add_argument("--pop-size", type=int, default=48)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--mc-splits", type=int, default=6)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--early-stop-patience", type=int, default=40)
    ap.add_argument("--cleanup-temp-ga", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.pop_size > 120:
        raise SystemExit("pop-size hard limit is 120")
    if args.generations > 150:
        raise SystemExit("generations hard limit is 150")

    run_id = args.run_id.strip() or _utc_tag()
    paths = build_paths(run_id)

    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    if args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        symbols = [s for s in symbols if s != "BTCUSDT"]
    else:
        symbols = discover_symbols(full_dir, include_btc=False)

    if not symbols:
        raise SystemExit("No non-BTC symbols found.")

    print(f"[pipeline] run_id={run_id}")
    print(f"[pipeline] symbols={symbols}")

    phase_summary_rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        df = load_df_1h(symbol)
        seed = load_seed_params(symbol)
        dfi = enforce_nla_and_validate(df, seed, symbol)
        run_dir = init_coin_run_dir(paths, symbol, args, dfi)
        print(f"[{symbol}] cycle1 optimization ...", flush=True)
        c1 = _run_phase_ga(
            symbol=symbol,
            phase="cycle1",
            trade_cycles=[1],
            seed_params=seed,
            df=dfi,
            args=args,
            coin_run_dir=run_dir,
            paths=paths,
        )
        _write_json(run_dir / f"{symbol}_cycle1.json", {"symbol": symbol, "params": c1.best_params, "source": c1.phase})

        print(f"[{symbol}] cycle3 optimization ...", flush=True)
        c3 = _run_phase_ga(
            symbol=symbol,
            phase="cycle3",
            trade_cycles=[3],
            seed_params=seed,
            df=dfi,
            args=args,
            coin_run_dir=run_dir,
            paths=paths,
        )
        _write_json(run_dir / f"{symbol}_cycle3.json", {"symbol": symbol, "params": c3.best_params, "source": c3.phase})

        phase_summary_rows.extend(
            [
                {
                    "symbol": symbol,
                    "phase": "cycle1",
                    "val_net": "",
                    "val_pf": "",
                    "val_dd": "",
                    "val_trades": "",
                    "stability": "",
                    "test_net": "",
                    "fitness": float(c1.report.get("best_overall", {}).get("fitness", 0.0)),
                    "PASS/FAIL": "",
                    "run_id": run_id,
                    "timestamp": _utc_now_iso(),
                },
                {
                    "symbol": symbol,
                    "phase": "cycle3",
                    "val_net": "",
                    "val_pf": "",
                    "val_dd": "",
                    "val_trades": "",
                    "stability": "",
                    "test_net": "",
                    "fitness": float(c3.report.get("best_overall", {}).get("fitness", 0.0)),
                    "PASS/FAIL": "",
                    "run_id": run_id,
                    "timestamp": _utc_now_iso(),
                },
            ]
        )
        print(f"[{symbol}] cycle1/cycle3 complete: {run_dir}", flush=True)

    _append_rows_csv(paths.all_report_csv, phase_summary_rows, ordered_columns=PHASE_SUMMARY_COLUMNS)
    print(f"[pipeline] cycle1/cycle3 phase complete. Appended {len(phase_summary_rows)} rows to {paths.all_report_csv}")


if __name__ == "__main__":
    main()
