#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "REPAIRED_TRADE_LEDGER_DIAGNOSTICS"
UNIVERSE = [
    "SOLUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "CRVUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "AXSUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "OGUSDT",
    "PAXGUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "ZECUSDT",
]
PRIORITY_SYMBOLS = ["SOLUSDT", "AVAXUSDT", "NEARUSDT"]
CANONICAL_1H_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650"
).resolve()
PRIORITY_RUN_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206"
).resolve()
REPAIRED_MULTICOIN_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108"
).resolve()


CANONICAL_SCHEMA = {
    "filled": "filled",
    "valid": "valid_for_metrics",
    "pnl_net": "pnl_net_pct",
    "pnl_gross": "pnl_gross_pct",
    "sl_hit": "sl_hit",
    "exit_reason": "exit_reason",
    "entry_time": "entry_time",
    "exit_time": "exit_time",
    "hold_minutes": "hold_minutes",
}
BUNDLE_BASELINE_SCHEMA = {
    "filled": "baseline_filled",
    "valid": "baseline_valid_for_metrics",
    "pnl_net": "baseline_pnl_net_pct",
    "pnl_gross": "baseline_pnl_gross_pct",
    "sl_hit": "baseline_sl_hit",
    "exit_reason": "baseline_exit_reason",
    "entry_time": "baseline_entry_time",
    "exit_time": "baseline_exit_time",
    "hold_minutes": "",
}
BUNDLE_EXEC_SCHEMA = {
    "filled": "exec_filled",
    "valid": "exec_valid_for_metrics",
    "pnl_net": "exec_pnl_net_pct",
    "pnl_gross": "exec_pnl_gross_pct",
    "sl_hit": "exec_sl_hit",
    "exit_reason": "exec_exit_reason",
    "entry_time": "exec_entry_time",
    "exit_time": "exec_exit_time",
    "hold_minutes": "",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    print(msg, flush=True)


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (set, tuple)):
            return list(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def tail_mean(values: pd.Series, frac: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    k = max(1, int(math.ceil(arr.size * float(frac))))
    return float(arr[:k].mean())


def max_drawdown(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    cum = np.cumsum(arr)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cum[:-1])))
    dd = cum - peaks
    return float(dd.min())


def max_consecutive_losses(values: pd.Series) -> int:
    best = 0
    run = 0
    for x in pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float):
        if float(x) < 0.0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def exit_category(reason: str) -> str:
    r = str(reason).strip().lower()
    if not r:
        return "unknown"
    if ("sl" in r) or ("stop" in r):
        return "stop_loss"
    if ("tp" in r) or ("target" in r):
        return "target"
    if "indicator" in r:
        return "indicator"
    if ("window_end" in r) or ("time" in r) or ("timeout" in r):
        return "time_or_window"
    return "other"


def hold_minutes_from_times(df: pd.DataFrame, entry_col: str, exit_col: str) -> pd.Series:
    entry = pd.to_datetime(df.get(entry_col, pd.Series(dtype=object)), utc=True, errors="coerce")
    exit_ = pd.to_datetime(df.get(exit_col, pd.Series(dtype=object)), utc=True, errors="coerce")
    mins = (exit_ - entry).dt.total_seconds() / 60.0
    return pd.to_numeric(mins, errors="coerce")


def metric_or_blank(x: Any) -> str:
    if isinstance(x, float) and np.isfinite(x):
        return f"{x:.6f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return ""


def simulate_repaired_reference_from_signals(
    *,
    symbol: str,
    signal_rows: pd.DataFrame,
    fee: modela.phasec_bt.FeeModel,
    exec_horizon_hours: float,
) -> pd.DataFrame:
    use = signal_rows.copy().sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
    sig_df = pd.DataFrame(
        {
            "signal_id": use["signal_id"].astype(str),
            "signal_time": pd.to_datetime(use["signal_time_utc"], utc=True, errors="coerce"),
            "tp_mult": pd.to_numeric(use["strategy_tp_mult"], errors="coerce"),
            "sl_mult": pd.to_numeric(use["strategy_sl_mult"], errors="coerce"),
        }
    )
    split_lookup = {str(sid): 0 for sid in sig_df["signal_id"].tolist()}
    return modela.phasec_bt._simulate_1h_reference(  # pylint: disable=protected-access
        signals_df=sig_df,
        split_lookup=split_lookup,
        fee=fee,
        exec_horizon_hours=float(exec_horizon_hours),
        symbol=str(symbol).upper(),
    )


def compute_trade_metrics(
    *,
    symbol: str,
    df: pd.DataFrame,
    schema: Dict[str, str],
    dataset_name: str,
    scope: str,
    source_kind: str,
    source_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    filled = pd.to_numeric(df.get(schema["filled"], 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(df.get(schema["valid"], 0), errors="coerce").fillna(0).astype(int)
    pnl_net = pd.to_numeric(df.get(schema["pnl_net"], np.nan), errors="coerce")
    pnl_gross = pd.to_numeric(df.get(schema["pnl_gross"], np.nan), errors="coerce")
    trade_mask = (filled == 1) & (valid == 1) & pnl_net.notna()

    trade_df = df.loc[trade_mask].copy()
    trade_pnl = pd.to_numeric(trade_df.get(schema["pnl_net"], pd.Series(dtype=float)), errors="coerce").dropna()
    gross_pnl_ser = pd.to_numeric(trade_df.get(schema["pnl_gross"], pd.Series(dtype=float)), errors="coerce").dropna()
    wins = trade_pnl[trade_pnl > 0.0]
    losses = trade_pnl[trade_pnl < 0.0]
    breakeven_count = int((trade_pnl == 0.0).sum())

    if schema.get("hold_minutes"):
        hold = pd.to_numeric(trade_df.get(schema["hold_minutes"], np.nan), errors="coerce")
    else:
        hold = hold_minutes_from_times(trade_df, schema["entry_time"], schema["exit_time"])
    hold = pd.to_numeric(hold, errors="coerce").dropna()

    raw_reasons = trade_df.get(schema["exit_reason"], pd.Series(dtype=object)).fillna("").astype(str).str.lower()
    raw_counts = raw_reasons.value_counts().to_dict()
    category_counts: Dict[str, int] = {}
    for reason, count in raw_counts.items():
        cat = exit_category(reason)
        category_counts[cat] = int(category_counts.get(cat, 0) + int(count))

    breakdown_rows: List[Dict[str, Any]] = []
    total_trades = int(len(trade_pnl))
    for reason, count in sorted(raw_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        breakdown_rows.append(
            {
                "symbol": symbol,
                "dataset_name": dataset_name,
                "scope": scope,
                "source_kind": source_kind,
                "raw_exit_reason": str(reason),
                "exit_category": exit_category(reason),
                "exit_count": int(count),
                "exit_share_pct": float(100.0 * int(count) / max(1, total_trades)),
            }
        )

    avg_win = float(wins.mean()) if not wins.empty else float("nan")
    avg_loss_abs = float(abs(losses.mean())) if not losses.empty else float("nan")
    med_win = float(wins.median()) if not wins.empty else float("nan")
    med_loss_abs = float(abs(losses.median())) if not losses.empty else float("nan")
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss_abs = float(abs(losses.sum())) if not losses.empty else 0.0
    profit_factor = (
        float(gross_profit / gross_loss_abs)
        if gross_loss_abs > 0.0
        else (float("inf") if gross_profit > 0.0 else float("nan"))
    )
    payoff_ratio = (
        float(avg_win / avg_loss_abs)
        if np.isfinite(avg_win) and np.isfinite(avg_loss_abs) and avg_loss_abs > 0.0
        else float("nan")
    )

    metrics = {
        "symbol": symbol,
        "dataset_name": dataset_name,
        "scope": scope,
        "source_kind": source_kind,
        "source_path": source_path,
        "total_trades": total_trades,
        "wins": int((trade_pnl > 0.0).sum()),
        "losses": int((trade_pnl < 0.0).sum()),
        "breakeven": breakeven_count,
        "win_rate_pct": float(100.0 * ((trade_pnl > 0.0).sum()) / max(1, total_trades)),
        "loss_rate_pct": float(100.0 * ((trade_pnl < 0.0).sum()) / max(1, total_trades)),
        "gross_pnl": float(gross_pnl_ser.sum()) if not gross_pnl_ser.empty else float("nan"),
        "net_pnl": float(trade_pnl.sum()) if total_trades > 0 else float("nan"),
        "avg_win": avg_win,
        "avg_loss": avg_loss_abs,
        "median_win": med_win,
        "median_loss": med_loss_abs,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy_per_trade": float(trade_pnl.mean()) if total_trades > 0 else float("nan"),
        "cvar_5": tail_mean(trade_pnl, 0.05) if total_trades > 0 else float("nan"),
        "max_drawdown": max_drawdown(trade_pnl) if total_trades > 0 else float("nan"),
        "max_consecutive_losses": max_consecutive_losses(trade_pnl) if total_trades > 0 else 0,
        "stop_loss_exit_count": int(category_counts.get("stop_loss", 0)),
        "target_exit_count": int(category_counts.get("target", 0)),
        "indicator_exit_count": int(category_counts.get("indicator", 0)),
        "time_or_window_exit_count": int(category_counts.get("time_or_window", 0)),
        "other_exit_count": int(category_counts.get("other", 0)),
        "average_hold_minutes": float(hold.mean()) if not hold.empty else float("nan"),
        "median_hold_minutes": float(hold.median()) if not hold.empty else float("nan"),
        "edge_vs_max_drawdown": (
            float((trade_pnl.mean()) / abs(max_drawdown(trade_pnl)))
            if total_trades > 0 and np.isfinite(max_drawdown(trade_pnl)) and abs(max_drawdown(trade_pnl)) > 0.0
            else float("nan")
        ),
    }
    return metrics, breakdown_rows


def classify_coin(
    row: pd.Series,
    *,
    exp_q1: float,
    exp_q2: float,
    exp_q3: float,
    pf_q1: float,
    pf_q2: float,
    pf_q3: float,
    edge_q1: float,
) -> str:
    trades = int(row.get("total_trades", 0))
    exp = finite_float(row.get("expectancy_per_trade"))
    pf = finite_float(row.get("profit_factor"))
    edge = finite_float(row.get("edge_vs_max_drawdown"))
    if trades <= 0 or not np.isfinite(exp):
        return "DATA_BLOCKED"
    if exp <= exp_q1 and pf <= pf_q1 and edge <= edge_q1:
        return "TRASH"
    if exp >= exp_q3 and pf >= pf_q3:
        return "KEEP"
    if exp >= exp_q2 and pf >= pf_q2:
        return "WATCH"
    return "WEAK"


def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    use = [c for c in cols if c in df.columns]
    if df.empty or not use:
        return "_(none)_"
    x = df.loc[:, use].copy()
    out = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-coin repaired trade ledger diagnostics")
    ap.add_argument("--canonical-1h-dir", default=str(CANONICAL_1H_DIR_DEFAULT))
    ap.add_argument("--priority-run-dir", default=str(PRIORITY_RUN_DIR_DEFAULT))
    ap.add_argument("--repaired-multicoin-dir", default=str(REPAIRED_MULTICOIN_DIR_DEFAULT))
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=20260228)
    args_cli = ap.parse_args()

    canonical_dir = Path(args_cli.canonical_1h_dir).resolve()
    priority_run_dir = Path(args_cli.priority_run_dir).resolve()
    repaired_multicoin_dir = Path(args_cli.repaired_multicoin_dir).resolve()
    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = ensure_dir(run_root / f"{RUN_PREFIX}_{utc_tag()}")

    canonical_trades_path = canonical_dir / "repaired_1h_trades.csv"
    canonical_summary_path = canonical_dir / "repaired_1h_reference_summary.csv"
    if not canonical_trades_path.exists():
        raise FileNotFoundError(f"Missing canonical repaired trade ledger: {canonical_trades_path}")
    if not canonical_summary_path.exists():
        raise FileNotFoundError(f"Missing canonical repaired summary: {canonical_summary_path}")

    priority_best_path = priority_run_dir / "repaired_modelA_reference_vs_best_priority.csv"
    priority_results_path = priority_run_dir / "repaired_modelA_results_priority.csv"
    priority_manifest_path = priority_run_dir / "repaired_modelA_run_manifest.json"
    if not priority_best_path.exists():
        raise FileNotFoundError(f"Missing repaired Model A priority compare file: {priority_best_path}")

    log(f"[1/5] Loading repaired source files into memory")
    canonical_trades = pd.read_csv(canonical_trades_path)
    canonical_trades["symbol"] = canonical_trades["symbol"].astype(str).str.upper()
    canonical_summary = pd.read_csv(canonical_summary_path)
    canonical_summary["symbol"] = canonical_summary["symbol"].astype(str).str.upper()
    priority_best = pd.read_csv(priority_best_path)
    priority_best["symbol"] = priority_best["symbol"].astype(str).str.upper()
    priority_results = pd.read_csv(priority_results_path) if priority_results_path.exists() else pd.DataFrame()
    if not priority_results.empty:
        priority_results["symbol"] = priority_results["symbol"].astype(str).str.upper()
    priority_manifest = {}
    if priority_manifest_path.exists():
        priority_manifest = json.loads(priority_manifest_path.read_text(encoding="utf-8"))

    foundation_dir = Path(str(priority_manifest.get("foundation_dir", ""))).resolve() if priority_manifest.get("foundation_dir") else phase_v.find_latest_foundation_dir()
    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state=foundation_state, seed=int(args_cli.seed))
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )
    variant_map = {str(cfg["candidate_id"]): dict(cfg) for cfg in modela.build_model_a_variants()}
    readiness_map = phase_v.symbol_readiness_map(foundation_state)
    quality_map = phase_v.symbol_quality_map(foundation_state)

    log(f"[2/5] Building repaired 1h per-coin trade metrics for {len(UNIVERSE)} symbols")
    baseline_rows: List[Dict[str, Any]] = []
    exit_rows: List[Dict[str, Any]] = []
    baseline_source_meta: Dict[str, Any] = {}
    canonical_symbols = set(canonical_trades["symbol"].unique().tolist())
    missing_from_canonical = [s for s in UNIVERSE if s not in canonical_symbols]

    for symbol in UNIVERSE:
        if symbol in canonical_symbols:
            sym_df = canonical_trades[canonical_trades["symbol"] == symbol].copy().reset_index(drop=True)
            metrics, breakdown = compute_trade_metrics(
                symbol=symbol,
                df=sym_df,
                schema=CANONICAL_SCHEMA,
                dataset_name="repaired_1h_reference_full",
                scope="full",
                source_kind="canonical_repaired_export",
                source_path=str(canonical_trades_path),
            )
            baseline_source_meta[symbol] = {
                "source_kind": "canonical_repaired_export",
                "source_path": str(canonical_trades_path),
                "trade_rows": int(len(sym_df)),
            }
        else:
            sig_rows = foundation_state.signal_timeline[
                foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol
            ].copy()
            sig_rows = sig_rows.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
            if sig_rows.empty:
                metrics = {
                    "symbol": symbol,
                    "dataset_name": "repaired_1h_reference_full",
                    "scope": "full",
                    "source_kind": "data_blocked",
                    "source_path": str(foundation_dir / "universe_signal_timeline.csv"),
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "breakeven": 0,
                    "win_rate_pct": float("nan"),
                    "loss_rate_pct": float("nan"),
                    "gross_pnl": float("nan"),
                    "net_pnl": float("nan"),
                    "avg_win": float("nan"),
                    "avg_loss": float("nan"),
                    "median_win": float("nan"),
                    "median_loss": float("nan"),
                    "payoff_ratio": float("nan"),
                    "profit_factor": float("nan"),
                    "expectancy_per_trade": float("nan"),
                    "cvar_5": float("nan"),
                    "max_drawdown": float("nan"),
                    "max_consecutive_losses": 0,
                    "stop_loss_exit_count": 0,
                    "target_exit_count": 0,
                    "indicator_exit_count": 0,
                    "time_or_window_exit_count": 0,
                    "other_exit_count": 0,
                    "average_hold_minutes": float("nan"),
                    "median_hold_minutes": float("nan"),
                    "edge_vs_max_drawdown": float("nan"),
                }
                breakdown = []
                baseline_source_meta[symbol] = {
                    "source_kind": "data_blocked",
                    "source_path": str(foundation_dir / "universe_signal_timeline.csv"),
                    "reason": "no_signal_rows",
                }
            else:
                log(f"  - reconstructing repaired 1h baseline for {symbol} (not present in canonical export)")
                sim_df = simulate_repaired_reference_from_signals(
                    symbol=symbol,
                    signal_rows=sig_rows,
                    fee=fee,
                    exec_horizon_hours=float(exec_args.exec_horizon_hours),
                )
                sim_df["symbol"] = symbol
                metrics, breakdown = compute_trade_metrics(
                    symbol=symbol,
                    df=sim_df,
                    schema=CANONICAL_SCHEMA,
                    dataset_name="repaired_1h_reference_full",
                    scope="full",
                    source_kind="runtime_reconstructed_repaired_contract",
                    source_path=str(foundation_dir / "universe_signal_timeline.csv"),
                )
                baseline_source_meta[symbol] = {
                    "source_kind": "runtime_reconstructed_repaired_contract",
                    "source_path": str(foundation_dir / "universe_signal_timeline.csv"),
                    "trade_rows": int(len(sim_df)),
                    "market_path": str(PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"),
                }
        metrics["bucket_1h"] = str(readiness_map.get(symbol, {}).get("bucket_1h", ""))
        metrics["foundation_integrity_status"] = str(
            quality_map.get(symbol, {}).get("integrity_status", readiness_map.get(symbol, {}).get("integrity_status", ""))
        )
        baseline_rows.append(metrics)
        exit_rows.extend(breakdown)

    baseline_df = pd.DataFrame(baseline_rows).sort_values("symbol").reset_index(drop=True)
    valid_baseline = baseline_df[(baseline_df["total_trades"] > 0) & baseline_df["expectancy_per_trade"].notna()].copy()
    exp_q1 = float(valid_baseline["expectancy_per_trade"].quantile(0.25))
    exp_q2 = float(valid_baseline["expectancy_per_trade"].quantile(0.50))
    exp_q3 = float(valid_baseline["expectancy_per_trade"].quantile(0.75))
    pf_q1 = float(valid_baseline["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.25))
    pf_q2 = float(valid_baseline["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.50))
    pf_q3 = float(valid_baseline["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.75))
    edge_q1 = float(valid_baseline["edge_vs_max_drawdown"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.25))

    baseline_df["label"] = baseline_df.apply(
        lambda r: classify_coin(
            r,
            exp_q1=exp_q1,
            exp_q2=exp_q2,
            exp_q3=exp_q3,
            pf_q1=pf_q1,
            pf_q2=pf_q2,
            pf_q3=pf_q3,
            edge_q1=edge_q1,
        ),
        axis=1,
    )

    log(f"[3/5] Reconstructing trade-level repaired Model A comparisons for {len(PRIORITY_SYMBOLS)} priority symbols")
    comparison_rows: List[Dict[str, Any]] = []
    priority_trade_meta: Dict[str, Any] = {}
    for symbol in PRIORITY_SYMBOLS:
        ref_row = priority_best[priority_best["symbol"] == symbol]
        if ref_row.empty:
            continue
        best_row = ref_row.iloc[0]
        best_candidate_id = str(best_row.get("best_candidate_id", ""))
        if best_candidate_id not in variant_map:
            continue

        log(f"  - rebuilding matched reference and best Model A trades for {symbol} ({best_candidate_id})")
        sig_df = foundation_state.signal_timeline[
            foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol
        ].copy()
        win_df = foundation_state.download_manifest[
            foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol
        ].copy()
        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=win_df,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        base_full = modela.build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )
        split_mask = pd.to_numeric(base_full["split_id"], errors="coerce").fillna(-1).astype(int) >= 0
        base_subset = base_full.loc[split_mask].copy().reset_index(drop=True)
        one_h = modela.load_1h_market(symbol)
        modela_eval = modela.evaluate_model_a_variant(
            bundle=bundle,
            baseline_df=base_full,
            cfg=variant_map[best_candidate_id],
            one_h=one_h,
            args=exec_args,
        )
        modela_df = modela_eval["signal_rows_df"].copy().reset_index(drop=True)

        ref_metrics, ref_breakdown = compute_trade_metrics(
            symbol=symbol,
            df=base_subset,
            schema=BUNDLE_BASELINE_SCHEMA,
            dataset_name="repaired_1h_reference_matched_modelA_sample",
            scope="matched_modelA_sample",
            source_kind="runtime_reconstructed_repaired_contract",
            source_path=str(foundation_dir / "universe_signal_timeline.csv"),
        )
        exec_metrics, exec_breakdown = compute_trade_metrics(
            symbol=symbol,
            df=modela_df,
            schema=BUNDLE_EXEC_SCHEMA,
            dataset_name=f"repaired_modelA_best_{best_candidate_id}",
            scope="matched_modelA_sample",
            source_kind="reconstructed_from_priority_selection",
            source_path=str(priority_best_path),
        )
        exit_rows.extend(ref_breakdown)
        exit_rows.extend(exec_breakdown)

        trade_count_diff = int(exec_metrics["total_trades"]) - int(ref_metrics["total_trades"])
        win_rate_diff = float(exec_metrics["win_rate_pct"] - ref_metrics["win_rate_pct"])
        loss_rate_diff = float(exec_metrics["loss_rate_pct"] - ref_metrics["loss_rate_pct"])
        avg_loss_diff = (
            float(exec_metrics["avg_loss"] - ref_metrics["avg_loss"])
            if np.isfinite(exec_metrics["avg_loss"]) and np.isfinite(ref_metrics["avg_loss"])
            else float("nan")
        )
        expectancy_diff = float(exec_metrics["expectancy_per_trade"] - ref_metrics["expectancy_per_trade"])
        maxdd_diff = float(exec_metrics["max_drawdown"] - ref_metrics["max_drawdown"])
        cvar_diff = float(exec_metrics["cvar_5"] - ref_metrics["cvar_5"])

        if finite_float(best_row.get("delta_expectancy_vs_repaired_1h")) > 0.0 and expectancy_diff <= 0.0:
            behavior_call = "AGGREGATE_UP_ONLY_NO_REAL_TRADE_EDGE"
        elif expectancy_diff > 0.0 and maxdd_diff >= 0.0 and cvar_diff >= 0.0:
            behavior_call = "REAL_TRADE_BEHAVIOR_IMPROVED"
        elif expectancy_diff > 0.0:
            behavior_call = "TRADE_EDGE_UP_BUT_RISK_MIXED"
        else:
            behavior_call = "NO_REAL_TRADE_IMPROVEMENT"

        comparison_rows.append(
            {
                "symbol": symbol,
                "best_candidate_id": best_candidate_id,
                "priority_classification": str(best_row.get("classification", "")),
                "best_valid_for_ranking": int(best_row.get("best_valid_for_ranking", 0)),
                "aggregate_signal_expectancy_delta": finite_float(best_row.get("delta_expectancy_vs_repaired_1h")),
                "reference_total_trades": int(ref_metrics["total_trades"]),
                "modelA_total_trades": int(exec_metrics["total_trades"]),
                "trade_count_diff": trade_count_diff,
                "reference_win_rate_pct": float(ref_metrics["win_rate_pct"]),
                "modelA_win_rate_pct": float(exec_metrics["win_rate_pct"]),
                "win_rate_diff_pct": win_rate_diff,
                "reference_loss_rate_pct": float(ref_metrics["loss_rate_pct"]),
                "modelA_loss_rate_pct": float(exec_metrics["loss_rate_pct"]),
                "loss_rate_diff_pct": loss_rate_diff,
                "reference_avg_loss": float(ref_metrics["avg_loss"]),
                "modelA_avg_loss": float(exec_metrics["avg_loss"]),
                "avg_loss_diff": avg_loss_diff,
                "reference_expectancy_per_trade": float(ref_metrics["expectancy_per_trade"]),
                "modelA_expectancy_per_trade": float(exec_metrics["expectancy_per_trade"]),
                "expectancy_diff": expectancy_diff,
                "reference_net_pnl": float(ref_metrics["net_pnl"]),
                "modelA_net_pnl": float(exec_metrics["net_pnl"]),
                "net_pnl_diff": float(exec_metrics["net_pnl"] - ref_metrics["net_pnl"]),
                "reference_max_drawdown": float(ref_metrics["max_drawdown"]),
                "modelA_max_drawdown": float(exec_metrics["max_drawdown"]),
                "max_drawdown_diff": maxdd_diff,
                "reference_cvar_5": float(ref_metrics["cvar_5"]),
                "modelA_cvar_5": float(exec_metrics["cvar_5"]),
                "cvar_diff": cvar_diff,
                "behavior_call": behavior_call,
                "reference_source_path": str(foundation_dir / "universe_signal_timeline.csv"),
                "modelA_source_path": str(priority_best_path),
            }
        )
        priority_trade_meta[symbol] = {
            "best_candidate_id": best_candidate_id,
            "priority_compare_row_source": str(priority_best_path),
            "build_meta": build_meta,
            "modela_metrics": modela_eval["metrics"],
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values("symbol").reset_index(drop=True)

    if not comparison_df.empty:
        compare_lookup = {
            str(r["symbol"]).upper(): dict(r)
            for _, r in comparison_df.iterrows()
        }
        baseline_df["modelA_behavior_call"] = baseline_df["symbol"].map(
            lambda s: compare_lookup.get(str(s).upper(), {}).get("behavior_call", "")
        )
        baseline_df["modelA_best_candidate_id"] = baseline_df["symbol"].map(
            lambda s: compare_lookup.get(str(s).upper(), {}).get("best_candidate_id", "")
        )
    else:
        baseline_df["modelA_behavior_call"] = ""
        baseline_df["modelA_best_candidate_id"] = ""

    log(f"[4/5] Writing CSV artifacts")
    judgment_cols = [
        "symbol",
        "bucket_1h",
        "foundation_integrity_status",
        "source_kind",
        "source_path",
        "total_trades",
        "wins",
        "losses",
        "breakeven",
        "win_rate_pct",
        "loss_rate_pct",
        "gross_pnl",
        "net_pnl",
        "avg_win",
        "avg_loss",
        "median_win",
        "median_loss",
        "payoff_ratio",
        "profit_factor",
        "expectancy_per_trade",
        "cvar_5",
        "max_drawdown",
        "max_consecutive_losses",
        "stop_loss_exit_count",
        "target_exit_count",
        "indicator_exit_count",
        "time_or_window_exit_count",
        "average_hold_minutes",
        "median_hold_minutes",
        "edge_vs_max_drawdown",
        "modelA_best_candidate_id",
        "modelA_behavior_call",
        "label",
    ]
    baseline_df.loc[:, judgment_cols].to_csv(run_dir / "repaired_trade_judgment_by_coin.csv", index=False)
    comparison_df.to_csv(run_dir / "repaired_trade_judgment_reference_vs_modelA.csv", index=False)
    pd.DataFrame(exit_rows).sort_values(["symbol", "dataset_name", "exit_count"], ascending=[True, True, False]).to_csv(
        run_dir / "repaired_trade_exit_reason_breakdown.csv",
        index=False,
    )

    best_by_quality = baseline_df.sort_values(
        ["expectancy_per_trade", "profit_factor", "edge_vs_max_drawdown", "net_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    worst_by_quality = baseline_df.sort_values(
        ["expectancy_per_trade", "profit_factor", "edge_vs_max_drawdown", "net_pnl"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    label_counts = baseline_df["label"].astype(str).value_counts().to_dict()

    log(f"[5/5] Writing markdown report and manifest")
    multicoin_present = repaired_multicoin_dir.exists()
    multicoin_required = [
        repaired_multicoin_dir / "repaired_multicoin_modelA_results.csv",
        repaired_multicoin_dir / "repaired_multicoin_modelA_reference_vs_best.csv",
        repaired_multicoin_dir / "repaired_multicoin_modelA_run_manifest.json",
    ]
    multicoin_complete = multicoin_present and all(p.exists() for p in multicoin_required)

    report_lines = [
        "# Repaired Trade Quality Report",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        "",
        "## Source Files Used",
        "",
        f"- Canonical repaired 1h trade ledger: `{canonical_trades_path}`",
        f"- Canonical repaired 1h summary: `{canonical_summary_path}`",
        f"- Repaired Model A priority reference-vs-best: `{priority_best_path}`",
        f"- Repaired Model A priority results: `{priority_results_path}`",
        f"- Repaired Model A priority manifest: `{priority_manifest_path}`",
        f"- Universal data foundation signal timeline: `{foundation_dir / 'universe_signal_timeline.csv'}`",
        f"- Universal data foundation 3m manifest: `{foundation_dir / 'universe_3m_download_manifest.csv'}`",
        f"- Universal data foundation readiness: `{foundation_dir / 'universe_symbol_readiness.csv'}`",
        f"- Repaired multicoin Model A dir present: `{int(multicoin_present)}`",
        f"- Repaired multicoin Model A dir complete/usable: `{int(multicoin_complete)}`",
        f"- Repaired multicoin Model A dir checked: `{repaired_multicoin_dir}`",
        "",
        "## Label Rules",
        "",
        "- Labels are assigned from repaired 1h full-trade metrics only, so every coin uses the widest repaired trade sample available.",
        "- `KEEP`: expectancy_per_trade >= repaired-universe 75th percentile and profit_factor >= repaired-universe 75th percentile.",
        "- `WATCH`: expectancy_per_trade >= repaired-universe median and profit_factor >= repaired-universe median, but below KEEP.",
        "- `TRASH`: expectancy_per_trade <= repaired-universe 25th percentile and profit_factor <= repaired-universe 25th percentile and edge_vs_max_drawdown <= repaired-universe 25th percentile.",
        "- `WEAK`: positive data but not strong enough for WATCH and not weak enough for TRASH.",
        "- `DATA_BLOCKED`: no repaired trade sample available.",
        "",
        "## Numeric Thresholds",
        "",
        f"- expectancy_q1: `{exp_q1:.10f}`",
        f"- expectancy_q2: `{exp_q2:.10f}`",
        f"- expectancy_q3: `{exp_q3:.10f}`",
        f"- profit_factor_q1: `{pf_q1:.10f}`",
        f"- profit_factor_q2: `{pf_q2:.10f}`",
        f"- profit_factor_q3: `{pf_q3:.10f}`",
        f"- edge_vs_max_drawdown_q1: `{edge_q1:.10f}`",
        "",
        "## Per-Coin Judgment Snapshot",
        "",
        markdown_table(
            baseline_df.loc[:, ["symbol", "wins", "losses", "win_rate_pct", "loss_rate_pct", "avg_loss", "expectancy_per_trade", "net_pnl", "label"]]
            .sort_values("symbol")
            .reset_index(drop=True),
            ["symbol", "wins", "losses", "win_rate_pct", "loss_rate_pct", "avg_loss", "expectancy_per_trade", "net_pnl", "label"],
        ),
        "",
        "## Priority Repaired 1h Vs Repaired Model A",
        "",
        markdown_table(
            comparison_df,
            [
                "symbol",
                "best_candidate_id",
                "trade_count_diff",
                "win_rate_diff_pct",
                "avg_loss_diff",
                "expectancy_diff",
                "max_drawdown_diff",
                "cvar_diff",
                "behavior_call",
            ],
        ),
        "",
        "## Best Coins",
        "",
        markdown_table(
            best_by_quality.loc[:, ["symbol", "expectancy_per_trade", "profit_factor", "net_pnl", "label"]].head(5),
            ["symbol", "expectancy_per_trade", "profit_factor", "net_pnl", "label"],
        ),
        "",
        "## Worst Coins",
        "",
        markdown_table(
            worst_by_quality.loc[:, ["symbol", "expectancy_per_trade", "profit_factor", "net_pnl", "label"]].head(5),
            ["symbol", "expectancy_per_trade", "profit_factor", "net_pnl", "label"],
        ),
        "",
        "## Label Counts",
        "",
        "\n".join([f"- {k}: `{int(v)}`" for k, v in sorted(label_counts.items())]),
    ]
    (run_dir / "repaired_trade_quality_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "universe": list(UNIVERSE),
        "priority_symbols": list(PRIORITY_SYMBOLS),
        "source_paths": {
            "canonical_repaired_1h_trade_csv": str(canonical_trades_path),
            "canonical_repaired_1h_summary_csv": str(canonical_summary_path),
            "priority_repaired_modelA_reference_vs_best_csv": str(priority_best_path),
            "priority_repaired_modelA_results_csv": str(priority_results_path),
            "priority_repaired_modelA_manifest_json": str(priority_manifest_path),
            "foundation_signal_timeline_csv": str(foundation_dir / "universe_signal_timeline.csv"),
            "foundation_download_manifest_csv": str(foundation_dir / "universe_3m_download_manifest.csv"),
            "foundation_readiness_csv": str(foundation_dir / "universe_symbol_readiness.csv"),
            "foundation_quality_csv": str(foundation_dir / "universe_3m_data_quality.csv"),
            "repaired_multicoin_dir_checked": str(repaired_multicoin_dir),
        },
        "repaired_multicoin_status": {
            "present": int(multicoin_present),
            "complete_usable": int(multicoin_complete),
            "missing_required_files": [str(p) for p in multicoin_required if not p.exists()],
        },
        "baseline_source_meta": baseline_source_meta,
        "priority_trade_meta": priority_trade_meta,
        "label_rules": {
            "basis": "repaired_1h_full_trade_metrics_only",
            "keep": "expectancy>=q75 and profit_factor>=q75",
            "watch": "expectancy>=q50 and profit_factor>=q50",
            "trash": "expectancy<=q25 and profit_factor<=q25 and edge_vs_max_drawdown<=q25",
            "weak": "everything else with repaired data",
            "data_blocked": "no repaired trade sample",
            "thresholds": {
                "expectancy_q1": exp_q1,
                "expectancy_q2": exp_q2,
                "expectancy_q3": exp_q3,
                "profit_factor_q1": pf_q1,
                "profit_factor_q2": pf_q2,
                "profit_factor_q3": pf_q3,
                "edge_vs_max_drawdown_q1": edge_q1,
            },
        },
        "outputs": {
            "repaired_trade_judgment_by_coin_csv": str(run_dir / "repaired_trade_judgment_by_coin.csv"),
            "repaired_trade_judgment_reference_vs_modelA_csv": str(run_dir / "repaired_trade_judgment_reference_vs_modelA.csv"),
            "repaired_trade_exit_reason_breakdown_csv": str(run_dir / "repaired_trade_exit_reason_breakdown.csv"),
            "repaired_trade_quality_report_md": str(run_dir / "repaired_trade_quality_report.md"),
            "repaired_trade_judgment_manifest_json": str(run_dir / "repaired_trade_judgment_manifest.json"),
        },
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "best_symbols_by_quality": best_by_quality["symbol"].head(5).tolist(),
        "worst_symbols_by_quality": worst_by_quality["symbol"].head(5).tolist(),
        "missing_from_canonical_repaired_export": missing_from_canonical,
    }
    json_dump(run_dir / "repaired_trade_judgment_manifest.json", manifest)

    log(f"Diagnostics complete: {run_dir}")


if __name__ == "__main__":
    main()
