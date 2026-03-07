#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import rebase_model_a_repaired_priority as repaired  # noqa: E402


RUN_PREFIX = "SURVIVOR_COIN_STOP_FORENSICS"
ANALYSIS_SET = ["LINKUSDT", "DOGEUSDT", "NEARUSDT", "LTCUSDT", "SOLUSDT", "AVAXUSDT"]
SHORT_1H_MIN = 60.0
SHORT_2H_MIN = 120.0
SHORT_3H_MIN = 180.0
SHORT_4H_MIN = 240.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def find_latest_multicoin_dir() -> Path:
    cands = sorted(
        [
            p
            for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("REPAIRED_MULTICOIN_MODELA_AUDIT_*")
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )
    if not cands:
        raise FileNotFoundError("No REPAIRED_MULTICOIN_MODELA_AUDIT_* directory found")
    return cands[-1].resolve()


def fmt_float(x: Any, nd: int = 10) -> str:
    xv = float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])
    return f"{xv:.{nd}f}" if np.isfinite(xv) else "nan"


def normalize_exit_reason(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"window_end", "timeout"}:
        return "timeout"
    if s in {"sl", "tp"}:
        return s
    if not s:
        return "other"
    return s


def _ts_from_ns(x: int) -> pd.Timestamp:
    return pd.to_datetime(int(x), utc=True)


def first_eligible_bar_after(fill_ts: pd.Timestamp, one_h: modela.OneHMarket) -> pd.Timestamp:
    idx = int(np.searchsorted(one_h.ts_ns, int(pd.to_datetime(fill_ts, utc=True).value), side="right"))
    if idx >= len(one_h.ts_ns):
        return pd.NaT
    return _ts_from_ns(int(one_h.ts_ns[idx]))


def row_metrics(
    *,
    row: pd.Series,
    ctx: Any,
    one_h: modela.OneHMarket,
) -> Dict[str, Any]:
    entry_ts = pd.to_datetime(row.get("exec_entry_time"), utc=True, errors="coerce")
    exit_ts = pd.to_datetime(row.get("exec_exit_time"), utc=True, errors="coerce")
    entry_px = float(pd.to_numeric(pd.Series([row.get("exec_entry_price", np.nan)]), errors="coerce").iloc[0])
    if pd.notna(entry_ts) and pd.notna(exit_ts):
        hold_min = float((exit_ts - entry_ts).total_seconds() / 60.0)
    else:
        hold_min = float("nan")

    sl_mult = float(getattr(ctx, "sl_mult_sig", np.nan))
    tp_mult = float(getattr(ctx, "tp_mult_sig", np.nan))
    sl_px = float(entry_px * sl_mult) if np.isfinite(entry_px) and np.isfinite(sl_mult) else float("nan")
    tp_px = float(entry_px * tp_mult) if np.isfinite(entry_px) and np.isfinite(tp_mult) else float("nan")
    stop_distance_bps = float((1.0 - sl_mult) * 1e4) if np.isfinite(sl_mult) else float("nan")
    first_bar_ts = first_eligible_bar_after(entry_ts, one_h) if pd.notna(entry_ts) else pd.NaT

    exit_bar_low = float("nan")
    exit_bar_high = float("nan")
    exit_bar_close = float("nan")
    stop_trigger_match = 0
    tp_trigger_match = 0
    if pd.notna(exit_ts):
        idx = int(np.searchsorted(one_h.ts_ns, int(exit_ts.value), side="left"))
        if idx < len(one_h.ts_ns) and int(one_h.ts_ns[idx]) == int(exit_ts.value):
            exit_bar_low = float(one_h.low_np[idx])
            exit_bar_high = float(one_h.high_np[idx])
            exit_bar_close = float(one_h.close_np[idx])
            stop_trigger_match = int(np.isfinite(sl_px) and np.isfinite(exit_bar_low) and exit_bar_low <= sl_px)
            tp_trigger_match = int(np.isfinite(tp_px) and np.isfinite(exit_bar_high) and exit_bar_high >= tp_px)

    exit_reason = normalize_exit_reason(row.get("exec_exit_reason", ""))
    return {
        "signal_id": str(row.get("signal_id", "")),
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "hold_min": hold_min,
        "exit_reason_norm": exit_reason,
        "sl_mult_sig": sl_mult,
        "tp_mult_sig": tp_mult,
        "sl_price": sl_px,
        "tp_price": tp_px,
        "stop_distance_bps": stop_distance_bps,
        "first_eligible_bar_ts": first_bar_ts,
        "triggered_on_first_eligible": int(pd.notna(first_bar_ts) and pd.notna(exit_ts) and exit_ts == first_bar_ts),
        "same_parent_bar_violation": int(pd.notna(first_bar_ts) and pd.notna(exit_ts) and exit_ts < first_bar_ts),
        "same_bar_dual_touch": int(pd.to_numeric(pd.Series([row.get("exec_same_bar_hit", 0)]), errors="coerce").fillna(0).iloc[0]),
        "invalid_stop_geometry": int(pd.to_numeric(pd.Series([row.get("exec_invalid_stop_geometry", 0)]), errors="coerce").fillna(0).iloc[0]),
        "invalid_tp_geometry": int(pd.to_numeric(pd.Series([row.get("exec_invalid_tp_geometry", 0)]), errors="coerce").fillna(0).iloc[0]),
        "exit_bar_low": exit_bar_low,
        "exit_bar_high": exit_bar_high,
        "exit_bar_close": exit_bar_close,
        "stop_trigger_match": stop_trigger_match,
        "tp_trigger_match": tp_trigger_match,
    }


def classify_coin(
    *,
    ranking_label: str,
    total_trades: int,
    short3_total: int,
    short4_total: int,
    short3_reason_counts: Dict[str, int],
    short3_sl_first_eligible_rate: float,
    short3_stop_trigger_mismatch_count: int,
    same_parent_violations: int,
    invalid_stop_geometry_count: int,
    median_stop_distance_bps_short_sl: float,
) -> Tuple[str, str, str]:
    if same_parent_violations > 0:
        return "WRONG_SOURCE_CANDLE", "PATCH_GLOBAL_STOP_LOGIC", "universal"
    if invalid_stop_geometry_count > 0:
        return "SHARED_STOP_DEFINITION_BUG", "PATCH_GLOBAL_STOP_LOGIC", "universal"
    if short3_stop_trigger_mismatch_count > 0:
        return "WRONG_TRIGGER_RULE", "PATCH_GLOBAL_STOP_LOGIC", "universal"

    short3_main_reason = max(short3_reason_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if short3_total > 0 else "other"
    short3_sl_share = float(short3_reason_counts.get("sl", 0) / max(1, short3_total))
    early_hold_share = float(short4_total / max(1, total_trades))

    if short3_main_reason == "sl":
        if short3_sl_first_eligible_rate >= 0.70 and np.isfinite(median_stop_distance_bps_short_sl) and median_stop_distance_bps_short_sl <= 175.0:
            if ranking_label == "MODEL_A_STRONG_GO_REPAIRED":
                return "STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT", "KEEP_AS_IS", "universal"
            if ranking_label == "MODEL_A_WEAK_GO_REPAIRED":
                return "STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT", "MOVE_TO_SHADOW_ONLY", "universal"
            return "STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT", "DISABLE_COIN", "universal"
        if short3_sl_first_eligible_rate >= 0.60 and short3_sl_share >= 0.60:
            if ranking_label in {"MODEL_A_STRONG_GO_REPAIRED", "MODEL_A_WEAK_GO_REPAIRED"}:
                return "ENTRY_QUALITY_PROBLEM", "MOVE_TO_SHADOW_ONLY", "unjustified"
            return "ENTRY_QUALITY_PROBLEM", "DISABLE_COIN", "unjustified"

    if ranking_label == "MODEL_A_STRONG_GO_REPAIRED":
        return "MIXED", "KEEP_AS_IS", "unjustified"
    if ranking_label == "MODEL_A_WEAK_GO_REPAIRED":
        if early_hold_share >= 0.65 and short3_sl_share >= 0.45:
            return "MIXED", "MOVE_TO_SHADOW_ONLY", "coin-selective"
        return "MIXED", "KEEP_AS_IS", "unjustified"
    if short3_total == 0:
        return "ENTRY_QUALITY_PROBLEM", "DISABLE_COIN", "unjustified"
    return "ENTRY_QUALITY_PROBLEM", "DISABLE_COIN", "unjustified"


def main() -> None:
    ap = argparse.ArgumentParser(description="Survivor-coin stop forensics under the repaired Model A contract")
    ap.add_argument("--multicoin-dir", default="", help="Path to REPAIRED_MULTICOIN_MODELA_AUDIT_*; defaults to latest")
    ap.add_argument("--foundation-dir", default="", help="Path to UNIVERSAL_DATA_FOUNDATION_*; defaults to latest completed run")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=20260228)
    args_cli = ap.parse_args()

    multicoin_dir = Path(args_cli.multicoin_dir).resolve() if str(args_cli.multicoin_dir).strip() else find_latest_multicoin_dir()
    foundation_dir = Path(args_cli.foundation_dir).resolve() if str(args_cli.foundation_dir).strip() else phase_v.find_latest_foundation_dir()

    class_fp = multicoin_dir / "repaired_multicoin_modelA_coin_classification.csv"
    best_fp = multicoin_dir / "repaired_multicoin_modelA_reference_vs_best.csv"
    if not class_fp.exists() or not best_fp.exists():
        raise FileNotFoundError(f"Missing repaired multicoin forensic sources in {multicoin_dir}")

    class_df = pd.read_csv(class_fp)
    best_df = pd.read_csv(best_fp)
    for df in (class_df, best_df):
        df["symbol"] = df["symbol"].astype(str).str.upper()

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state=foundation_state, seed=int(args_cli.seed))
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    variants = {str(v["candidate_id"]): dict(v) for v in phase_v.sanitize_variants()}
    qual_map = phase_v.symbol_quality_map(foundation_state)
    ready_map = phase_v.symbol_readiness_map(foundation_state)
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )

    hold_rows: List[Dict[str, Any]] = []
    short_reason_rows: List[Dict[str, Any]] = []
    stop_rows: List[Dict[str, Any]] = []
    root_rows: List[Dict[str, Any]] = []
    symbol_meta: Dict[str, Any] = {}

    for symbol in ANALYSIS_SET:
        best_row = best_df[best_df["symbol"] == symbol]
        class_row = class_df[class_df["symbol"] == symbol]
        if best_row.empty or class_row.empty:
            raise RuntimeError(f"Missing repaired multicoin classification for {symbol}")
        best_row_s = best_row.iloc[0]
        class_row_s = class_row.iloc[0]
        candidate_id = str(best_row_s.get("best_candidate_id", "")).strip()
        if not candidate_id:
            raise RuntimeError(f"No best candidate recorded for {symbol}")
        if candidate_id not in variants:
            raise KeyError(f"Unknown candidate_id={candidate_id} for {symbol}")

        qrow = qual_map.get(symbol, {})
        rrow = ready_map.get(symbol, {})
        sig_df = foundation_state.signal_timeline[foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol].copy()
        win_df = foundation_state.download_manifest[foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol].copy()
        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=win_df,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        one_h = modela.load_1h_market(symbol)
        baseline_full = modela.build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )
        ev = modela.evaluate_model_a_variant(
            bundle=bundle,
            baseline_df=baseline_full,
            cfg=variants[candidate_id],
            one_h=one_h,
            args=exec_args,
        )
        signal_df = ev["signal_rows_df"].copy()
        signal_df["exec_filled"] = pd.to_numeric(signal_df["exec_filled"], errors="coerce").fillna(0).astype(int)
        filled_df = signal_df[signal_df["exec_filled"] == 1].copy().reset_index(drop=True)
        ctx_map = {str(ctx.signal_id): ctx for ctx in bundle.contexts}

        metrics_rows: List[Dict[str, Any]] = []
        for row in filled_df.to_dict("records"):
            sid = str(row.get("signal_id", ""))
            ctx = ctx_map.get(sid)
            if ctx is None:
                continue
            metrics_rows.append(row_metrics(row=pd.Series(row), ctx=ctx, one_h=one_h))
        metrics_df = pd.DataFrame(metrics_rows)
        if metrics_df.empty:
            metrics_df = pd.DataFrame(
                columns=[
                    "signal_id",
                    "entry_ts",
                    "exit_ts",
                    "hold_min",
                    "exit_reason_norm",
                    "sl_mult_sig",
                    "tp_mult_sig",
                    "sl_price",
                    "tp_price",
                    "stop_distance_bps",
                    "first_eligible_bar_ts",
                    "triggered_on_first_eligible",
                    "same_parent_bar_violation",
                    "same_bar_dual_touch",
                    "invalid_stop_geometry",
                    "invalid_tp_geometry",
                    "exit_bar_low",
                    "exit_bar_high",
                    "exit_bar_close",
                    "stop_trigger_match",
                    "tp_trigger_match",
                ]
            )

        total_trades = int(len(metrics_df))
        short1 = metrics_df[pd.to_numeric(metrics_df["hold_min"], errors="coerce") <= SHORT_1H_MIN].copy()
        short2 = metrics_df[pd.to_numeric(metrics_df["hold_min"], errors="coerce") <= SHORT_2H_MIN].copy()
        short3 = metrics_df[pd.to_numeric(metrics_df["hold_min"], errors="coerce") <= SHORT_3H_MIN].copy()
        short4 = metrics_df[pd.to_numeric(metrics_df["hold_min"], errors="coerce") <= SHORT_4H_MIN].copy()

        mean_hold = float(pd.to_numeric(metrics_df["hold_min"], errors="coerce").mean()) if total_trades else float("nan")
        median_hold = float(pd.to_numeric(metrics_df["hold_min"], errors="coerce").median()) if total_trades else float("nan")

        short3_counts = {k: 0 for k in ["sl", "tp", "timeout", "other"]}
        short4_counts = {k: 0 for k in ["sl", "tp", "timeout", "other"]}
        for x in short3["exit_reason_norm"].astype(str).tolist():
            short3_counts[x if x in short3_counts else "other"] += 1
        for x in short4["exit_reason_norm"].astype(str).tolist():
            short4_counts[x if x in short4_counts else "other"] += 1
        main_short_reason = max(short3_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if len(short3) > 0 else "other"

        short3_sl = short3[short3["exit_reason_norm"].astype(str) == "sl"].copy()
        same_parent_violations = int(pd.to_numeric(metrics_df["same_parent_bar_violation"], errors="coerce").fillna(0).sum()) if total_trades else 0
        same_bar_dual_touch_count = int(pd.to_numeric(metrics_df["same_bar_dual_touch"], errors="coerce").fillna(0).sum()) if total_trades else 0
        invalid_stop_geometry_count = int(pd.to_numeric(metrics_df["invalid_stop_geometry"], errors="coerce").fillna(0).sum()) if total_trades else 0
        short3_stop_trigger_mismatch_count = int(
            ((short3_sl["stop_trigger_match"].fillna(0).astype(int) == 0)).sum()
        ) if not short3_sl.empty else 0
        short3_sl_first_eligible_count = int(pd.to_numeric(short3_sl["triggered_on_first_eligible"], errors="coerce").fillna(0).sum()) if not short3_sl.empty else 0
        short3_sl_first_eligible_rate = float(short3_sl_first_eligible_count / max(1, len(short3_sl))) if len(short3_sl) else float("nan")
        mean_stop_distance_bps = float(pd.to_numeric(metrics_df["stop_distance_bps"], errors="coerce").mean()) if total_trades else float("nan")
        median_stop_distance_bps = float(pd.to_numeric(metrics_df["stop_distance_bps"], errors="coerce").median()) if total_trades else float("nan")
        mean_stop_distance_bps_short_sl = float(pd.to_numeric(short3_sl["stop_distance_bps"], errors="coerce").mean()) if len(short3_sl) else float("nan")
        median_stop_distance_bps_short_sl = float(pd.to_numeric(short3_sl["stop_distance_bps"], errors="coerce").median()) if len(short3_sl) else float("nan")

        root_cause, action, fix_scope = classify_coin(
            ranking_label=str(class_row_s.get("classification", "")),
            total_trades=total_trades,
            short3_total=int(len(short3)),
            short4_total=int(len(short4)),
            short3_reason_counts=short3_counts,
            short3_sl_first_eligible_rate=short3_sl_first_eligible_rate,
            short3_stop_trigger_mismatch_count=short3_stop_trigger_mismatch_count,
            same_parent_violations=same_parent_violations,
            invalid_stop_geometry_count=invalid_stop_geometry_count,
            median_stop_distance_bps_short_sl=median_stop_distance_bps_short_sl,
        )

        hold_rows.append(
            {
                "symbol": symbol,
                "repaired_ranking_label": str(class_row_s.get("classification", "")),
                "best_candidate_id": candidate_id,
                "total_trades": total_trades,
                "trades_exit_within_1h": int(len(short1)),
                "trades_exit_within_2h": int(len(short2)),
                "trades_exit_within_3h": int(len(short3)),
                "trades_exit_within_4h": int(len(short4)),
                "pct_exit_within_3h": float(len(short3) / max(1, total_trades)),
                "pct_exit_within_4h": float(len(short4) / max(1, total_trades)),
                "mean_hold_min": mean_hold,
                "median_hold_min": median_hold,
                "main_short_hold_exit_reason": main_short_reason,
            }
        )
        short_reason_rows.append(
            {
                "symbol": symbol,
                "repaired_ranking_label": str(class_row_s.get("classification", "")),
                "best_candidate_id": candidate_id,
                "short_hold_threshold_min": int(SHORT_3H_MIN),
                "short_trades_total": int(len(short3)),
                "short_sl_count": int(short3_counts["sl"]),
                "short_tp_count": int(short3_counts["tp"]),
                "short_timeout_count": int(short3_counts["timeout"]),
                "short_other_count": int(short3_counts["other"]),
                "within_4h_total": int(len(short4)),
                "within_4h_sl_count": int(short4_counts["sl"]),
                "within_4h_tp_count": int(short4_counts["tp"]),
                "within_4h_timeout_count": int(short4_counts["timeout"]),
                "within_4h_other_count": int(short4_counts["other"]),
            }
        )

        sample_first_bar = pd.NaT
        sample_short_sl_exit = pd.NaT
        if not short3_sl.empty:
            sample_first_bar = short3_sl.iloc[0]["first_eligible_bar_ts"]
            sample_short_sl_exit = short3_sl.iloc[0]["exit_ts"]

        stop_rows.append(
            {
                "symbol": symbol,
                "repaired_ranking_label": str(class_row_s.get("classification", "")),
                "best_candidate_id": candidate_id,
                "stop_formula": "sl_price = realized_fill_price * signal_sl_mult",
                "stop_anchor_basis": "realized_fill_price",
                "stop_source_candle": "3m fill candle price with 1h signal sl_mult",
                "source_signal_candle": "1h signal at signal_time owns sl_mult/tp_mult",
                "first_candle_eligible_to_trigger_stop": "first full 1h bar strictly after fill_time",
                "first_candle_eligible_rule": "searchsorted(one_h.ts_ns, fill_time, side='right')",
                "trigger_rule": "wick touch on 1h low<=sl or high>=tp; if both hit same bar, SL wins",
                "same_parent_bar_violation_count": same_parent_violations,
                "same_bar_dual_touch_count": same_bar_dual_touch_count,
                "invalid_stop_geometry_count": invalid_stop_geometry_count,
                "short_sl_count": int(len(short3_sl)),
                "short_sl_first_eligible_count": short3_sl_first_eligible_count,
                "short_sl_first_eligible_rate": short3_sl_first_eligible_rate,
                "short_sl_trigger_mismatch_count": short3_stop_trigger_mismatch_count,
                "stop_distance_bps_mean_all": mean_stop_distance_bps,
                "stop_distance_bps_median_all": median_stop_distance_bps,
                "stop_distance_bps_mean_short_sl": mean_stop_distance_bps_short_sl,
                "stop_distance_bps_median_short_sl": median_stop_distance_bps_short_sl,
                "signal_sl_mult_mean": float(pd.to_numeric(metrics_df["sl_mult_sig"], errors="coerce").mean()) if total_trades else float("nan"),
                "signal_sl_mult_median": float(pd.to_numeric(metrics_df["sl_mult_sig"], errors="coerce").median()) if total_trades else float("nan"),
                "sample_first_eligible_bar_ts": str(pd.to_datetime(sample_first_bar, utc=True)) if pd.notna(sample_first_bar) else "",
                "sample_short_sl_trigger_exit_ts": str(pd.to_datetime(sample_short_sl_exit, utc=True)) if pd.notna(sample_short_sl_exit) else "",
            }
        )
        root_rows.append(
            {
                "symbol": symbol,
                "repaired_ranking_label": str(class_row_s.get("classification", "")),
                "best_candidate_id": candidate_id,
                "total_trades": total_trades,
                "trades_exit_within_3h": int(len(short3)),
                "pct_exit_within_3h": float(len(short3) / max(1, total_trades)),
                "trades_exit_within_4h": int(len(short4)),
                "pct_exit_within_4h": float(len(short4) / max(1, total_trades)),
                "main_short_hold_exit_reason": main_short_reason,
                "root_cause_classification": root_cause,
                "recommended_action": action,
                "fix_scope": fix_scope,
                "short_sl_first_eligible_rate": short3_sl_first_eligible_rate,
                "median_stop_distance_bps_short_sl": median_stop_distance_bps_short_sl,
                "short_sl_trigger_mismatch_count": short3_stop_trigger_mismatch_count,
                "same_parent_bar_violation_count": same_parent_violations,
                "same_bar_dual_touch_count": same_bar_dual_touch_count,
            }
        )

        symbol_meta[symbol] = {
            "bundle_build": build_meta,
            "foundation_integrity_status": str(qrow.get("integrity_status", rrow.get("integrity_status", ""))),
            "ranking_label": str(class_row_s.get("classification", "")),
            "best_candidate_id": candidate_id,
            "best_delta_expectancy_vs_repaired_1h": float(pd.to_numeric(pd.Series([class_row_s.get("best_delta_expectancy_vs_repaired_1h", np.nan)]), errors="coerce").iloc[0]),
            "root_cause_classification": root_cause,
            "recommended_action": action,
        }

    hold_df = pd.DataFrame(hold_rows).sort_values("symbol").reset_index(drop=True)
    short_reason_df = pd.DataFrame(short_reason_rows).sort_values("symbol").reset_index(drop=True)
    stop_df = pd.DataFrame(stop_rows).sort_values("symbol").reset_index(drop=True)
    root_df = pd.DataFrame(root_rows).sort_values("symbol").reset_index(drop=True)

    hold_df.to_csv(run_dir / "survivor_hold_duration_by_coin.csv", index=False)
    short_reason_df.to_csv(run_dir / "survivor_short_exit_reason_breakdown.csv", index=False)
    stop_df.to_csv(run_dir / "survivor_stop_definition_matrix.csv", index=False)
    root_df.to_csv(run_dir / "survivor_stop_root_cause_by_coin.csv", index=False)

    universal_stop_issue = "unjustified"
    if (root_df["root_cause_classification"] == "SHARED_STOP_DEFINITION_BUG").any() or (root_df["root_cause_classification"] == "WRONG_TRIGGER_RULE").any():
        universal_stop_issue = "universal"
    elif (
        not root_df.empty
        and (root_df["root_cause_classification"] == "STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT").all()
    ):
        universal_stop_issue = "universal"
    elif (root_df["root_cause_classification"] == "STOP_TOO_TIGHT_BUT_LOGICALLY_CORRECT").any():
        universal_stop_issue = "coin-selective"

    approved = root_df[root_df["recommended_action"] == "KEEP_AS_IS"]["symbol"].astype(str).tolist()
    shadow = root_df[root_df["recommended_action"] == "MOVE_TO_SHADOW_ONLY"]["symbol"].astype(str).tolist()
    disabled = root_df[root_df["recommended_action"] == "DISABLE_COIN"]["symbol"].astype(str).tolist()

    report_lines = [
        "# Survivor-Coin Stop Forensics",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Frozen repaired multicoin source: `{multicoin_dir}`",
        f"- Frozen repaired 1h baseline source: `{multicoin_dir.parent / '1H_CONTRACT_REPAIR_REBASELINE_20260301_140650'}`",
        "- Repaired contract guards:",
        "  - `scripts/backtest_exec_phasec_sol.py:247` uses `defer_exit_to_next_bar=True`",
        "  - `scripts/backtest_exec_phasec_sol.py:316` starts 1h exit evaluation at `idx + 1`",
        "  - `scripts/phase_a_model_a_audit.py:423` uses `searchsorted(..., side=\"right\")` after fill",
        "  - `scripts/phase_a_model_a_audit.py:445` anchors stop to `fill_price * sl_mult_sig`",
        "  - `scripts/execution_layer_3m_ict.py:780-805` uses wick-touch triggers and resolves same-bar SL/TP conflicts in favor of SL",
        "  - `paper_trading/app/model_a_runtime.py:672-674` blocks same-parent-bar exit checks after fill",
        "",
        "## Per-Coin Summary",
        "",
        repaired.markdown_table(
            root_df.merge(
                hold_df[
                    [
                        "symbol",
                        "mean_hold_min",
                        "median_hold_min",
                    ]
                ],
                on="symbol",
                how="left",
            ),
            [
                "symbol",
                "total_trades",
                "trades_exit_within_3h",
                "pct_exit_within_3h",
                "trades_exit_within_4h",
                "pct_exit_within_4h",
                "main_short_hold_exit_reason",
                "root_cause_classification",
                "recommended_action",
                "mean_hold_min",
                "median_hold_min",
            ],
            n=20,
        ),
        "",
        "## Decision Layer",
        "",
        f"- universal_stop_issue_scope: `{universal_stop_issue}`",
        f"- approved_keep_as_is: `{approved}`",
        f"- shadow_only: `{shadow}`",
        f"- disabled: `{disabled}`",
    ]
    repaired.write_text(run_dir / "survivor_fix_decision_report.md", "\n".join(report_lines))

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "frozen_repaired_multicoin_dir": str(multicoin_dir),
        "frozen_repaired_multicoin_classification_csv": str(class_fp),
        "frozen_repaired_multicoin_reference_vs_best_csv": str(best_fp),
        "foundation_dir": str(foundation_dir),
        "analysis_set": list(ANALYSIS_SET),
        "contract_validation": contract_validation,
        "repaired_contract_guards": {
            "repaired_1h_defer_exit_default": "scripts/backtest_exec_phasec_sol.py:247",
            "repaired_1h_eval_start": "scripts/backtest_exec_phasec_sol.py:316",
            "model_a_exit_after_fill_guard": "scripts/phase_a_model_a_audit.py:423",
            "model_a_stop_anchor_formula": "scripts/phase_a_model_a_audit.py:445",
            "wick_touch_trigger_rule": "scripts/execution_layer_3m_ict.py:780-805",
            "paper_runtime_guard": "paper_trading/app/model_a_runtime.py:672-674",
        },
        "legacy_pre_repair_results_used": 0,
        "symbol_meta": symbol_meta,
        "universal_stop_issue_scope": universal_stop_issue,
        "approved_keep_as_is": approved,
        "shadow_only": shadow,
        "disabled": disabled,
        "outputs": {
            "survivor_hold_duration_by_coin_csv": str(run_dir / "survivor_hold_duration_by_coin.csv"),
            "survivor_short_exit_reason_breakdown_csv": str(run_dir / "survivor_short_exit_reason_breakdown.csv"),
            "survivor_stop_definition_matrix_csv": str(run_dir / "survivor_stop_definition_matrix.csv"),
            "survivor_stop_root_cause_by_coin_csv": str(run_dir / "survivor_stop_root_cause_by_coin.csv"),
            "survivor_fix_decision_report_md": str(run_dir / "survivor_fix_decision_report.md"),
            "survivor_forensics_manifest_json": str(run_dir / "survivor_forensics_manifest.json"),
        },
    }
    repaired.json_dump(run_dir / "survivor_forensics_manifest.json", manifest)


if __name__ == "__main__":
    main()
