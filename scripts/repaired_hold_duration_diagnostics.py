#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


RUN_PREFIX = "REPAIRED_HOLD_DURATION_DIAGNOSTICS"
SELECTED = [
    "AXSUSDT",
    "CRVUSDT",
    "DOGEUSDT",
    "OGUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "ZECUSDT",
]
RANKING_STATUS = {
    "AXSUSDT": "KEEP",
    "CRVUSDT": "KEEP",
    "DOGEUSDT": "KEEP",
    "OGUSDT": "KEEP",
    "AVAXUSDT": "WATCH",
    "NEARUSDT": "WATCH",
    "SOLUSDT": "WATCH",
    "ZECUSDT": "WATCH",
}
CANONICAL_1H_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650"
).resolve()
PRIORITY_RUN_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206"
).resolve()
TRADE_JUDGMENT_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/REPAIRED_TRADE_LEDGER_DIAGNOSTICS_20260302_235325"
).resolve()

BASELINE_SCHEMA = {
    "filled": "filled",
    "valid": "valid_for_metrics",
    "exit_reason": "exit_reason",
    "entry_time": "entry_time",
    "exit_time": "exit_time",
    "hold_minutes": "hold_minutes",
    "pnl_net": "pnl_net_pct",
}
MATCHED_BASELINE_SCHEMA = {
    "filled": "baseline_filled",
    "valid": "baseline_valid_for_metrics",
    "exit_reason": "baseline_exit_reason",
    "entry_time": "baseline_entry_time",
    "exit_time": "baseline_exit_time",
    "hold_minutes": "",
    "pnl_net": "baseline_pnl_net_pct",
}
MODELA_SCHEMA = {
    "filled": "exec_filled",
    "valid": "exec_valid_for_metrics",
    "exit_reason": "exec_exit_reason",
    "entry_time": "exec_entry_time",
    "exit_time": "exec_exit_time",
    "hold_minutes": "",
    "pnl_net": "exec_pnl_net_pct",
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
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def hold_minutes_from_times(df: pd.DataFrame, entry_col: str, exit_col: str) -> pd.Series:
    entry = pd.to_datetime(df.get(entry_col, pd.Series(dtype=object)), utc=True, errors="coerce")
    exit_ = pd.to_datetime(df.get(exit_col, pd.Series(dtype=object)), utc=True, errors="coerce")
    return pd.to_numeric((exit_ - entry).dt.total_seconds() / 60.0, errors="coerce")


def exit_category(reason: str) -> str:
    r = str(reason).strip().lower()
    if not r:
        return "other"
    if ("sl" in r) or ("stop" in r):
        return "stop_loss"
    if ("tp" in r) or ("target" in r):
        return "target"
    if "indicator" in r:
        return "indicator_exit"
    return "other"


def dominant_reason(counts: Dict[str, int]) -> str:
    nonzero = [(k, int(v)) for k, v in counts.items() if int(v) > 0]
    if not nonzero:
        return "none"
    nonzero.sort(key=lambda kv: (-kv[1], kv[0]))
    return str(nonzero[0][0])


def compute_hold_stats(
    *,
    symbol: str,
    ranking_status: str,
    trade_quality_label: str,
    repaired_modelA_classification: str,
    raw_df: pd.DataFrame,
    schema: Dict[str, str],
    dataset_name: str,
    strategy_version: str,
    source_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    filled = pd.to_numeric(raw_df.get(schema["filled"], 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(raw_df.get(schema["valid"], 0), errors="coerce").fillna(0).astype(int)
    pnl = pd.to_numeric(raw_df.get(schema["pnl_net"], np.nan), errors="coerce")
    mask = (filled == 1) & (valid == 1) & pnl.notna()

    trades = raw_df.loc[mask].copy().reset_index(drop=True)
    if schema["hold_minutes"]:
        hold = pd.to_numeric(trades.get(schema["hold_minutes"], np.nan), errors="coerce")
    else:
        hold = hold_minutes_from_times(trades, schema["entry_time"], schema["exit_time"])
    trades["_hold_minutes_calc"] = pd.to_numeric(hold, errors="coerce")
    trades["_exit_reason_norm"] = trades.get(schema["exit_reason"], pd.Series(dtype=object)).fillna("").astype(str).str.lower()

    h = trades["_hold_minutes_calc"]
    le_1h = trades[h <= 60.0]
    gt_1_le_2 = trades[(h > 60.0) & (h <= 120.0)]
    gt_2_le_3 = trades[(h > 120.0) & (h <= 180.0)]
    gt_3_le_4 = trades[(h > 180.0) & (h <= 240.0)]
    gt_4 = trades[h > 240.0]
    within_3 = trades[h <= 180.0]
    within_4 = trades[h <= 240.0]

    def _reason_counts(df: pd.DataFrame) -> Dict[str, int]:
        out = {"stop_loss": 0, "target": 0, "indicator_exit": 0, "other": 0}
        for reason in df["_exit_reason_norm"].tolist():
            out[exit_category(reason)] += 1
        return out

    within_3_counts = _reason_counts(within_3)
    within_4_counts = _reason_counts(within_4)

    total = int(len(trades))
    summary = {
        "symbol": symbol,
        "ranking_status": ranking_status,
        "trade_quality_label": trade_quality_label,
        "repaired_modelA_classification": repaired_modelA_classification,
        "dataset_name": dataset_name,
        "strategy_version": strategy_version,
        "source_path": source_path,
        "total_trades": total,
        "trades_exit_le_1h": int(len(le_1h)),
        "trades_exit_gt_1h_le_2h": int(len(gt_1_le_2)),
        "trades_exit_gt_2h_le_3h": int(len(gt_2_le_3)),
        "trades_exit_gt_3h_le_4h": int(len(gt_3_le_4)),
        "trades_exit_gt_4h": int(len(gt_4)),
        "pct_exit_le_1h": float(100.0 * len(le_1h) / max(1, total)),
        "pct_exit_le_2h": float(100.0 * (len(le_1h) + len(gt_1_le_2)) / max(1, total)),
        "pct_exit_le_3h": float(100.0 * len(within_3) / max(1, total)),
        "pct_exit_le_4h": float(100.0 * len(within_4) / max(1, total)),
        "median_hold_minutes": float(h.median()) if not h.dropna().empty else float("nan"),
        "mean_hold_minutes": float(h.mean()) if not h.dropna().empty else float("nan"),
        "trades_exit_within_3h": int(len(within_3)),
        "pct_exit_within_3h": float(100.0 * len(within_3) / max(1, total)),
        "trades_exit_within_4h": int(len(within_4)),
        "pct_exit_within_4h": float(100.0 * len(within_4) / max(1, total)),
        "within_3h_stop_loss_count": int(within_3_counts["stop_loss"]),
        "within_3h_target_count": int(within_3_counts["target"]),
        "within_3h_indicator_exit_count": int(within_3_counts["indicator_exit"]),
        "within_3h_other_exit_count": int(within_3_counts["other"]),
        "within_3h_main_exit_reason": dominant_reason(within_3_counts),
        "within_4h_stop_loss_count": int(within_4_counts["stop_loss"]),
        "within_4h_target_count": int(within_4_counts["target"]),
        "within_4h_indicator_exit_count": int(within_4_counts["indicator_exit"]),
        "within_4h_other_exit_count": int(within_4_counts["other"]),
        "within_4h_main_exit_reason": dominant_reason(within_4_counts),
        "expectancy_per_trade": float(pnl[mask].mean()) if total > 0 else float("nan"),
    }

    rows = []
    for window_name, window_df, counts in [
        ("within_3h", within_3, within_3_counts),
        ("within_4h", within_4, within_4_counts),
    ]:
        rows.append(
            {
                "symbol": symbol,
                "ranking_status": ranking_status,
                "trade_quality_label": trade_quality_label,
                "repaired_modelA_classification": repaired_modelA_classification,
                "dataset_name": dataset_name,
                "strategy_version": strategy_version,
                "source_path": source_path,
                "window_name": window_name,
                "window_trade_count": int(len(window_df)),
                "stop_loss_count": int(counts["stop_loss"]),
                "target_count": int(counts["target"]),
                "indicator_exit_count": int(counts["indicator_exit"]),
                "other_exit_count": int(counts["other"]),
                "dominant_exit_reason": dominant_reason(counts),
            }
        )
    return summary, rows, trades


def hold_profile_flag(summary: Dict[str, Any]) -> str:
    total = int(summary.get("total_trades", 0))
    if total <= 0:
        return "INCONCLUSIVE"
    pct_3 = finite_float(summary.get("pct_exit_within_3h"))
    pct_4 = finite_float(summary.get("pct_exit_within_4h"))
    within_4 = int(summary.get("trades_exit_within_4h", 0))
    stop_4 = int(summary.get("within_4h_stop_loss_count", 0))
    stop_share_4 = float(100.0 * stop_4 / max(1, within_4))
    if within_4 > 0 and pct_4 >= 60.0 and stop_share_4 >= 70.0:
        return "TOO_MANY_EARLY_STOPS"
    if pct_4 >= 80.0 or pct_3 >= 70.0:
        return "SHORT_HOLD_HEAVY"
    if 40.0 <= pct_4 <= 70.0 and stop_share_4 < 70.0:
        return "BALANCED_HOLD_PROFILE"
    return "INCONCLUSIVE"


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
    ap = argparse.ArgumentParser(description="Hold-duration diagnostics for selected repaired-contract coins")
    ap.add_argument("--canonical-1h-dir", default=str(CANONICAL_1H_DIR_DEFAULT))
    ap.add_argument("--priority-run-dir", default=str(PRIORITY_RUN_DIR_DEFAULT))
    ap.add_argument("--trade-judgment-dir", default=str(TRADE_JUDGMENT_DIR_DEFAULT))
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=20260228)
    args_cli = ap.parse_args()

    canonical_dir = Path(args_cli.canonical_1h_dir).resolve()
    priority_run_dir = Path(args_cli.priority_run_dir).resolve()
    trade_judgment_dir = Path(args_cli.trade_judgment_dir).resolve()
    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = ensure_dir(run_root / f"{RUN_PREFIX}_{utc_tag()}")
    trade_source_dir = ensure_dir(run_dir / "_trade_sources")

    canonical_trades_path = canonical_dir / "repaired_1h_trades.csv"
    priority_best_path = priority_run_dir / "repaired_modelA_reference_vs_best_priority.csv"
    priority_manifest_path = priority_run_dir / "repaired_modelA_run_manifest.json"
    trade_quality_path = trade_judgment_dir / "repaired_trade_judgment_by_coin.csv"
    if not canonical_trades_path.exists():
        raise FileNotFoundError(f"Missing repaired 1h trade export: {canonical_trades_path}")
    if not priority_best_path.exists():
        raise FileNotFoundError(f"Missing repaired Model A priority selection file: {priority_best_path}")
    if not trade_quality_path.exists():
        raise FileNotFoundError(f"Missing repaired trade quality file: {trade_quality_path}")

    log("[1/5] Loading repaired source files and labels")
    canonical = pd.read_csv(canonical_trades_path)
    canonical["symbol"] = canonical["symbol"].astype(str).str.upper()
    priority_best = pd.read_csv(priority_best_path)
    priority_best["symbol"] = priority_best["symbol"].astype(str).str.upper()
    trade_quality = pd.read_csv(trade_quality_path)
    trade_quality["symbol"] = trade_quality["symbol"].astype(str).str.upper()

    priority_manifest = json.loads(priority_manifest_path.read_text(encoding="utf-8")) if priority_manifest_path.exists() else {}
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
    quality_map = {str(r["symbol"]).upper(): str(r["label"]) for _, r in trade_quality.iterrows()}
    modela_class_map = {
        str(r["symbol"]).upper(): str(r["classification"])
        for _, r in priority_best.iterrows()
    }

    log("[2/5] Building repaired 1h reference hold-duration buckets")
    summary_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    source_catalog: List[Dict[str, Any]] = []

    for symbol in SELECTED:
        ranking_status = RANKING_STATUS[symbol]
        trade_quality_label = quality_map.get(symbol, "")
        repaired_modelA_classification = modela_class_map.get(symbol, "")
        if symbol in set(canonical["symbol"].unique()):
            coin_df = canonical[canonical["symbol"] == symbol].copy().reset_index(drop=True)
            coin_path = trade_source_dir / f"{symbol}_repaired_1h_reference.csv"
            coin_df.to_csv(coin_path, index=False)
            source_catalog.append(
                {
                    "symbol": symbol,
                    "strategy_version": "repaired_1h_reference",
                    "source_path": str(coin_path),
                    "upstream_path": str(canonical_trades_path),
                    "notes": "filtered from canonical repaired_1h_trades.csv",
                }
            )
        else:
            log(f"  - reconstructing repaired 1h reference trade export for {symbol}")
            sig_rows = foundation_state.signal_timeline[
                foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol
            ].copy()
            coin_df = simulate_repaired_reference_from_signals(
                symbol=symbol,
                signal_rows=sig_rows,
                fee=fee,
                exec_horizon_hours=float(exec_args.exec_horizon_hours),
            )
            coin_df["symbol"] = symbol
            coin_path = trade_source_dir / f"{symbol}_repaired_1h_reference.csv"
            coin_df.to_csv(coin_path, index=False)
            source_catalog.append(
                {
                    "symbol": symbol,
                    "strategy_version": "repaired_1h_reference",
                    "source_path": str(coin_path),
                    "upstream_path": str(foundation_dir / "universe_signal_timeline.csv"),
                    "notes": "runtime reconstruction using repaired 1h contract",
                }
            )

        summary, reason_rows, _trades = compute_hold_stats(
            symbol=symbol,
            ranking_status=ranking_status,
            trade_quality_label=trade_quality_label,
            repaired_modelA_classification=repaired_modelA_classification,
            raw_df=coin_df,
            schema=BASELINE_SCHEMA,
            dataset_name="repaired_1h_reference_full",
            strategy_version="repaired_1h_reference",
            source_path=str(coin_path),
        )
        summary["hold_profile_flag"] = hold_profile_flag(summary)
        summary_rows.append(summary)
        breakdown_rows.extend(reason_rows)

    baseline_df = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)

    log("[3/5] Rebuilding repaired Model A hold comparisons where a repaired candidate exists")
    for symbol in [s for s in SELECTED if s in set(priority_best["symbol"].unique())]:
        best_row = priority_best[priority_best["symbol"] == symbol].iloc[0]
        best_candidate_id = str(best_row.get("best_candidate_id", ""))
        if best_candidate_id not in variant_map:
            continue
        ranking_status = RANKING_STATUS[symbol]
        trade_quality_label = quality_map.get(symbol, "")
        repaired_modelA_classification = str(best_row.get("classification", ""))

        log(f"  - rebuilding matched repaired 1h vs Model A for {symbol} ({best_candidate_id})")
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
        matched_ref_df = base_full.loc[split_mask].copy().reset_index(drop=True)
        ref_path = trade_source_dir / f"{symbol}_matched_repaired_1h_reference_for_{best_candidate_id}.csv"
        matched_ref_df.to_csv(ref_path, index=False)
        source_catalog.append(
            {
                "symbol": symbol,
                "strategy_version": "repaired_1h_reference_matched_modelA_sample",
                "source_path": str(ref_path),
                "upstream_path": str(foundation_dir / "universe_signal_timeline.csv"),
                "notes": "walk-forward matched repaired 1h sample used for Model A comparison",
            }
        )

        one_h = modela.load_1h_market(symbol)
        ev = modela.evaluate_model_a_variant(
            bundle=bundle,
            baseline_df=base_full,
            cfg=variant_map[best_candidate_id],
            one_h=one_h,
            args=exec_args,
        )
        modela_df = ev["signal_rows_df"].copy().reset_index(drop=True)
        modela_path = trade_source_dir / f"{symbol}_{best_candidate_id}_repaired_modelA_trade_level.csv"
        modela_df.to_csv(modela_path, index=False)
        source_catalog.append(
            {
                "symbol": symbol,
                "strategy_version": f"repaired_modelA_best_{best_candidate_id}",
                "source_path": str(modela_path),
                "upstream_path": str(priority_best_path),
                "notes": "trade-level reconstruction using repaired priority-selected candidate",
            }
        )

        ref_summary, ref_breakdown, _ = compute_hold_stats(
            symbol=symbol,
            ranking_status=ranking_status,
            trade_quality_label=trade_quality_label,
            repaired_modelA_classification=repaired_modelA_classification,
            raw_df=matched_ref_df,
            schema=MATCHED_BASELINE_SCHEMA,
            dataset_name="repaired_1h_reference_matched_modelA_sample",
            strategy_version="repaired_1h_reference_matched_modelA_sample",
            source_path=str(ref_path),
        )
        modela_summary, modela_breakdown, _ = compute_hold_stats(
            symbol=symbol,
            ranking_status=ranking_status,
            trade_quality_label=trade_quality_label,
            repaired_modelA_classification=repaired_modelA_classification,
            raw_df=modela_df,
            schema=MODELA_SCHEMA,
            dataset_name=f"repaired_modelA_best_{best_candidate_id}",
            strategy_version=f"repaired_modelA_best_{best_candidate_id}",
            source_path=str(modela_path),
        )
        breakdown_rows.extend(ref_breakdown)
        breakdown_rows.extend(modela_breakdown)

        median_diff = float(modela_summary["median_hold_minutes"] - ref_summary["median_hold_minutes"])
        mean_diff = float(modela_summary["mean_hold_minutes"] - ref_summary["mean_hold_minutes"])
        expectancy_diff = float(modela_summary["expectancy_per_trade"] - ref_summary["expectancy_per_trade"])
        comparison_rows.append(
            {
                "symbol": symbol,
                "ranking_status": ranking_status,
                "trade_quality_label": trade_quality_label,
                "repaired_modelA_classification": repaired_modelA_classification,
                "best_candidate_id": best_candidate_id,
                "reference_source_path": str(ref_path),
                "modelA_source_path": str(modela_path),
                "reference_total_trades": int(ref_summary["total_trades"]),
                "modelA_total_trades": int(modela_summary["total_trades"]),
                "reference_trades_exit_within_3h": int(ref_summary["trades_exit_within_3h"]),
                "modelA_trades_exit_within_3h": int(modela_summary["trades_exit_within_3h"]),
                "reference_pct_exit_within_3h": float(ref_summary["pct_exit_within_3h"]),
                "modelA_pct_exit_within_3h": float(modela_summary["pct_exit_within_3h"]),
                "reference_trades_exit_within_4h": int(ref_summary["trades_exit_within_4h"]),
                "modelA_trades_exit_within_4h": int(modela_summary["trades_exit_within_4h"]),
                "reference_pct_exit_within_4h": float(ref_summary["pct_exit_within_4h"]),
                "modelA_pct_exit_within_4h": float(modela_summary["pct_exit_within_4h"]),
                "reference_short_stop_loss_3h": int(ref_summary["within_3h_stop_loss_count"]),
                "modelA_short_stop_loss_3h": int(modela_summary["within_3h_stop_loss_count"]),
                "reference_short_stop_loss_4h": int(ref_summary["within_4h_stop_loss_count"]),
                "modelA_short_stop_loss_4h": int(modela_summary["within_4h_stop_loss_count"]),
                "reference_short_target_3h": int(ref_summary["within_3h_target_count"]),
                "modelA_short_target_3h": int(modela_summary["within_3h_target_count"]),
                "reference_short_target_4h": int(ref_summary["within_4h_target_count"]),
                "modelA_short_target_4h": int(modela_summary["within_4h_target_count"]),
                "reduces_short_stopouts": int(
                    modela_summary["within_3h_stop_loss_count"] < ref_summary["within_3h_stop_loss_count"]
                    or modela_summary["within_4h_stop_loss_count"] < ref_summary["within_4h_stop_loss_count"]
                ),
                "increases_short_target_captures": int(
                    modela_summary["within_3h_target_count"] > ref_summary["within_3h_target_count"]
                    or modela_summary["within_4h_target_count"] > ref_summary["within_4h_target_count"]
                ),
                "reference_median_hold_minutes": float(ref_summary["median_hold_minutes"]),
                "modelA_median_hold_minutes": float(modela_summary["median_hold_minutes"]),
                "median_hold_diff_minutes": median_diff,
                "reference_mean_hold_minutes": float(ref_summary["mean_hold_minutes"]),
                "modelA_mean_hold_minutes": float(modela_summary["mean_hold_minutes"]),
                "mean_hold_diff_minutes": mean_diff,
                "changes_median_hold_materially": int(abs(median_diff) >= 15.0),
                "reference_expectancy_per_trade": float(ref_summary["expectancy_per_trade"]),
                "modelA_expectancy_per_trade": float(modela_summary["expectancy_per_trade"]),
                "expectancy_diff": expectancy_diff,
                "exits_faster_without_improving_trade_quality": int(mean_diff < 0.0 and expectancy_diff <= 0.0),
                "comparison_readout": (
                    "FASTER_WITHOUT_QUALITY_GAIN"
                    if (mean_diff < 0.0 and expectancy_diff <= 0.0)
                    else ("SHORT_HOLD_BEHAVIOR_IMPROVED" if expectancy_diff > 0.0 else "MIXED")
                ),
                "bundle_signals_total": int(build_meta.get("signals_total", 0)),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("symbol").reset_index(drop=True)

    log("[4/5] Writing required artifact tables")
    baseline_out = baseline_df.loc[
        :,
        [
            "symbol",
            "ranking_status",
            "trade_quality_label",
            "repaired_modelA_classification",
            "strategy_version",
            "source_path",
            "total_trades",
            "trades_exit_le_1h",
            "trades_exit_gt_1h_le_2h",
            "trades_exit_gt_2h_le_3h",
            "trades_exit_gt_3h_le_4h",
            "trades_exit_gt_4h",
            "pct_exit_le_1h",
            "pct_exit_le_2h",
            "pct_exit_le_3h",
            "pct_exit_le_4h",
            "trades_exit_within_3h",
            "pct_exit_within_3h",
            "trades_exit_within_4h",
            "pct_exit_within_4h",
            "median_hold_minutes",
            "mean_hold_minutes",
            "within_3h_main_exit_reason",
            "within_4h_main_exit_reason",
            "hold_profile_flag",
        ],
    ]
    baseline_out.to_csv(run_dir / "repaired_hold_duration_by_coin.csv", index=False)
    pd.DataFrame(breakdown_rows).sort_values(
        ["symbol", "dataset_name", "window_name"],
        ascending=[True, True, True],
    ).to_csv(run_dir / "repaired_hold_duration_exit_reason_breakdown.csv", index=False)
    comparison_df.to_csv(run_dir / "repaired_hold_duration_reference_vs_modelA.csv", index=False)

    log("[5/5] Writing markdown report and manifest")
    highest_4h = baseline_df.sort_values(["pct_exit_within_4h", "pct_exit_within_3h"], ascending=[False, False]).head(3)
    lowest_4h = baseline_df.sort_values(["pct_exit_within_4h", "pct_exit_within_3h"], ascending=[True, True]).head(3)

    report_lines = [
        "# Repaired Hold Duration Report",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        "",
        "## Source Files Used",
        "",
    ]
    for row in source_catalog:
        report_lines.append(
            f"- `{row['symbol']}` | `{row['strategy_version']}` | source `{row['source_path']}` | upstream `{row['upstream_path']}` | {row['notes']}"
        )
    report_lines.extend(
        [
            "",
            "## Hold-Profile Flag Rules",
            "",
            "- `TOO_MANY_EARLY_STOPS`: pct_exit_within_4h >= 60% and stop-loss share inside 4h >= 70%.",
            "- `SHORT_HOLD_HEAVY`: pct_exit_within_4h >= 80% or pct_exit_within_3h >= 70%, excluding the stricter early-stop case.",
            "- `BALANCED_HOLD_PROFILE`: pct_exit_within_4h between 40% and 70% and early-stop share under 70%.",
            "- `INCONCLUSIVE`: everything else.",
            "",
            "## Per-Coin Hold Snapshot",
            "",
            markdown_table(
                baseline_out.sort_values("symbol").reset_index(drop=True),
                [
                    "symbol",
                    "ranking_status",
                    "trade_quality_label",
                    "total_trades",
                    "trades_exit_within_3h",
                    "pct_exit_within_3h",
                    "trades_exit_within_4h",
                    "pct_exit_within_4h",
                    "median_hold_minutes",
                    "mean_hold_minutes",
                    "hold_profile_flag",
                ],
            ),
            "",
            "## Repaired 1h Vs Repaired Model A",
            "",
            markdown_table(
                comparison_df,
                [
                    "symbol",
                    "best_candidate_id",
                    "reduces_short_stopouts",
                    "increases_short_target_captures",
                    "median_hold_diff_minutes",
                    "expectancy_diff",
                    "exits_faster_without_improving_trade_quality",
                    "comparison_readout",
                ],
            ),
            "",
            "## Highest Short-Hold Rate",
            "",
            markdown_table(highest_4h, ["symbol", "pct_exit_within_4h", "pct_exit_within_3h", "within_4h_main_exit_reason"]),
            "",
            "## Lowest Short-Hold Rate",
            "",
            markdown_table(lowest_4h, ["symbol", "pct_exit_within_4h", "pct_exit_within_3h", "within_4h_main_exit_reason"]),
        ]
    )
    (run_dir / "repaired_hold_duration_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "selected_symbols": list(SELECTED),
        "ranking_status": dict(RANKING_STATUS),
        "source_paths": {
            "canonical_repaired_1h_trades_csv": str(canonical_trades_path),
            "priority_repaired_modelA_reference_vs_best_csv": str(priority_best_path),
            "priority_repaired_modelA_manifest_json": str(priority_manifest_path),
            "trade_quality_csv": str(trade_quality_path),
            "foundation_signal_timeline_csv": str(foundation_dir / "universe_signal_timeline.csv"),
            "foundation_download_manifest_csv": str(foundation_dir / "universe_3m_download_manifest.csv"),
        },
        "trade_source_catalog": source_catalog,
        "flag_rules": {
            "TOO_MANY_EARLY_STOPS": "pct_exit_within_4h >= 60 and 4h stop-loss share >= 70",
            "SHORT_HOLD_HEAVY": "pct_exit_within_4h >= 80 or pct_exit_within_3h >= 70",
            "BALANCED_HOLD_PROFILE": "40 <= pct_exit_within_4h <= 70 and 4h stop-loss share < 70",
            "INCONCLUSIVE": "all other cases",
        },
        "outputs": {
            "repaired_hold_duration_by_coin_csv": str(run_dir / "repaired_hold_duration_by_coin.csv"),
            "repaired_hold_duration_exit_reason_breakdown_csv": str(run_dir / "repaired_hold_duration_exit_reason_breakdown.csv"),
            "repaired_hold_duration_reference_vs_modelA_csv": str(run_dir / "repaired_hold_duration_reference_vs_modelA.csv"),
            "repaired_hold_duration_report_md": str(run_dir / "repaired_hold_duration_report.md"),
            "repaired_hold_duration_manifest_json": str(run_dir / "repaired_hold_duration_manifest.json"),
        },
    }
    json_dump(run_dir / "repaired_hold_duration_manifest.json", manifest)
    log(f"Diagnostics complete: {run_dir}")


if __name__ == "__main__":
    main()
