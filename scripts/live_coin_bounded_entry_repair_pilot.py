#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import instant_loser_vs_winner_entry_forensics as forensics  # noqa: E402
from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT"
DEFAULT_DIAG_DIR = Path(
    "/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335"
).resolve()
COIN_ORDER = ["LINKUSDT", "NEARUSDT", "DOGEUSDT", "LTCUSDT"]
CURRENT_STATUS = {
    "LINKUSDT": "approved",
    "NEARUSDT": "shadow",
    "DOGEUSDT": "shadow",
    "LTCUSDT": "shadow",
}
CONTROL_TO_REPAIR_DECISION = {
    "LINKUSDT": "FILTER+ROOM",
    "NEARUSDT": "ROOM",
    "DOGEUSDT": "FILTER+ROOM",
    "LTCUSDT": "FILTER+ROOM",
}
MEANINGFUL_WINNER_THRESHOLD = 0.0020
EXPECTANCY_TOL = 0.00005
CVAR_TOL = 0.00010
MAXDD_TOL = 0.01
MIN_RETENTION = 0.85
MIN_INSTANT_LOSER_REDUCTION_ABS = 5
MIN_INSTANT_LOSER_REDUCTION_FRAC = 0.05


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 30) -> str:
    if df.empty:
        return "_(none)_"
    use = [c for c in cols if c in df.columns]
    if not use:
        return "_(none)_"
    x = df.loc[:, use].head(n).copy()
    out = ["| " + " | ".join(x.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def find_latest_diag_dir() -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        req = {
            "instant_loser_vs_winner_trade_buckets.csv",
            "instant_loser_vs_winner_feature_matrix.csv",
            "entry_repair_recommendation_by_coin.csv",
        }
        names = {f.name for f in p.iterdir() if f.is_file()}
        if req.issubset(names):
            return p.resolve()
    if DEFAULT_DIAG_DIR.exists():
        return DEFAULT_DIAG_DIR
    raise FileNotFoundError("No completed INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS directory found")


def hold_minutes_from_row(row: pd.Series) -> float:
    return forensics.hold_minutes_from_times(row.get("exec_entry_time"), row.get("exec_exit_time"))


def is_stop(reason: Any) -> bool:
    r = str(reason).strip().lower()
    return ("sl" in r) or ("stop" in r)


def zero_exec_fields(out: Dict[str, Any], skip_reason: str) -> Dict[str, Any]:
    out["exec_filled"] = 0
    out["exec_valid_for_metrics"] = 0
    out["exec_sl_hit"] = 0
    out["exec_tp_hit"] = 0
    out["exec_pnl_net_pct"] = float("nan")
    out["exec_pnl_gross_pct"] = float("nan")
    out["exec_fill_liquidity_type"] = ""
    out["exec_fill_delay_min"] = float("nan")
    out["exec_mae_pct"] = float("nan")
    out["exec_mfe_pct"] = float("nan")
    out["entry_improvement_bps"] = float("nan")
    out["exec_skip_reason"] = str(skip_reason)
    out["exec_exit_reason"] = ""
    out["exec_entry_time"] = ""
    out["exec_exit_time"] = ""
    out["exec_entry_price"] = float("nan")
    out["exec_exit_price"] = float("nan")
    out["exec_same_bar_hit"] = 0
    out["exec_invalid_stop_geometry"] = 0
    out["exec_invalid_tp_geometry"] = 0
    out["exec_entry_type"] = ""
    return out


def build_split_lookup(bundle: modela.ga_exec.SymbolBundle) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        split_id = int(sp["split_id"])
        for ctx in bundle.contexts[idx0:idx1]:
            out[str(ctx.signal_id)] = split_id
    return out


def metric_or_blank(x: Any) -> str:
    if isinstance(x, float) and np.isfinite(x):
        return f"{x:.6f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return ""


def load_diagnosis_state(diag_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucket_df = pd.read_csv(diag_dir / "instant_loser_vs_winner_trade_buckets.csv")
    feature_df = pd.read_csv(diag_dir / "instant_loser_vs_winner_feature_matrix.csv")
    rec_df = pd.read_csv(diag_dir / "entry_repair_recommendation_by_coin.csv")
    for df in (bucket_df, feature_df, rec_df):
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()
    return bucket_df, feature_df, rec_df


def feature_subset(feature_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return feature_df[feature_df["symbol"] == str(symbol).upper()].copy().reset_index(drop=True)


def midpoint_threshold(winner_s: pd.Series, loser_s: pd.Series, lo: float, hi: float, fallback: float) -> float:
    w = pd.to_numeric(winner_s, errors="coerce").dropna()
    l = pd.to_numeric(loser_s, errors="coerce").dropna()
    if w.empty or l.empty:
        return float(fallback)
    mid = float((float(w.median()) + float(l.median())) / 2.0)
    return float(min(max(mid, lo), hi))


def compute_filter_thresholds(symbol: str, diag_features: pd.DataFrame) -> Dict[str, float]:
    winners = diag_features[diag_features["bucket"] == "meaningful_winner"].copy()
    losers = diag_features[diag_features["bucket"].isin(["instant_loser", "fast_loser"])].copy()
    wick_mid = midpoint_threshold(
        winners.get("upper_wick_ratio", pd.Series(dtype=float)),
        losers.get("upper_wick_ratio", pd.Series(dtype=float)),
        lo=0.08,
        hi=0.35,
        fallback=0.20,
    )
    breakout_mid = midpoint_threshold(
        winners.get("breakout_dist_pct", pd.Series(dtype=float)),
        losers.get("breakout_dist_pct", pd.Series(dtype=float)),
        lo=-0.08,
        hi=0.02,
        fallback=0.0,
    )
    return {
        "action_gap_cap": 0.0,
        "upper_wick_cap": float(wick_mid),
        "breakout_cap": float(breakout_mid),
        "room_floor_bps_small": 15.0,
        "room_floor_bps_large": 20.0,
    }


def build_variant_specs(symbol: str, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "variant_id": "CONTROL",
            "label": "Current repaired control",
            "filter_mode": "none",
            "room_floor_bps": 0.0,
            "room_condition": "none",
        }
    ]
    if CONTROL_TO_REPAIR_DECISION[symbol] == "FILTER+ROOM":
        specs.extend(
            [
                {
                    "variant_id": "GAP_CAP_FILTER",
                    "label": f"Reject action-gap > {thresholds['action_gap_cap']:.4f}",
                    "filter_mode": "gap",
                    "room_floor_bps": 0.0,
                    "room_condition": "none",
                },
                {
                    "variant_id": "UPPER_WICK_FILTER",
                    "label": f"Reject upper-wick > {thresholds['upper_wick_cap']:.4f}",
                    "filter_mode": "wick",
                    "room_floor_bps": 0.0,
                    "room_condition": "none",
                },
                {
                    "variant_id": "GAP_CAP_PLUS_ROOM15",
                    "label": f"Gap-cap filter + conditional min stop {thresholds['room_floor_bps_small']:.1f}bps",
                    "filter_mode": "gap",
                    "room_floor_bps": float(thresholds["room_floor_bps_small"]),
                    "room_condition": "allowed_after_filter",
                },
            ]
        )
    else:
        specs.extend(
            [
                {
                    "variant_id": "ROOM15_ON_NONCHASE",
                    "label": f"Min stop {thresholds['room_floor_bps_small']:.1f}bps on non-chase entries",
                    "filter_mode": "none",
                    "room_floor_bps": float(thresholds["room_floor_bps_small"]),
                    "room_condition": "non_chase",
                },
                {
                    "variant_id": "ROOM20_ON_NONCHASE_BREAKOUTSAFE",
                    "label": f"Min stop {thresholds['room_floor_bps_large']:.1f}bps on non-chase + non-breakout entries",
                    "filter_mode": "none",
                    "room_floor_bps": float(thresholds["room_floor_bps_large"]),
                    "room_condition": "non_chase_breakout_safe",
                },
            ]
        )
    return specs


def passes_filter(spec: Dict[str, Any], feature_row: Optional[pd.Series], thresholds: Dict[str, float]) -> bool:
    mode = str(spec.get("filter_mode", "none"))
    if mode == "none":
        return True
    if feature_row is None:
        return False
    if mode == "gap":
        return bool(finite_float(feature_row.get("action_gap_pct", np.nan), default=1.0) <= float(thresholds["action_gap_cap"]))
    if mode == "wick":
        return bool(finite_float(feature_row.get("upper_wick_ratio", np.nan), default=1.0) <= float(thresholds["upper_wick_cap"]))
    return True


def room_applies(spec: Dict[str, Any], feature_row: Optional[pd.Series], thresholds: Dict[str, float], allowed: bool) -> bool:
    floor_bps = float(spec.get("room_floor_bps", 0.0))
    if floor_bps <= 0.0:
        return False
    cond = str(spec.get("room_condition", "none"))
    if cond == "allowed_after_filter":
        return bool(allowed)
    if feature_row is None:
        return False
    gap = finite_float(feature_row.get("action_gap_pct", np.nan), default=1.0)
    breakout = finite_float(feature_row.get("breakout_dist_pct", np.nan), default=1.0)
    if cond == "non_chase":
        return bool(gap <= float(thresholds["action_gap_cap"]))
    if cond == "non_chase_breakout_safe":
        return bool(gap <= float(thresholds["action_gap_cap"]) and breakout <= float(thresholds["breakout_cap"]))
    return False


def stop_floor_to_sl_mult(current_sl_mult: float, floor_bps: float) -> float:
    current_dist = max(0.0, 1.0 - float(current_sl_mult))
    floor_dist = max(current_dist, float(floor_bps) / 1e4)
    return float(max(0.0, min(float(current_sl_mult), 1.0 - floor_dist)))


def rerun_exit_with_room(
    *,
    row_s: pd.Series,
    ctx: modela.ga_exec.SignalContext,
    one_h: modela.OneHMarket,
    exec_args: argparse.Namespace,
    floor_bps: float,
) -> Dict[str, Any]:
    fill_time = pd.to_datetime(row_s.get("exec_entry_time"), utc=True, errors="coerce")
    fill_price = finite_float(row_s.get("exec_entry_price", np.nan))
    if pd.isna(fill_time) or (not np.isfinite(fill_price)) or fill_price <= 0.0:
        return dict(row_s.to_dict())

    new_ctx = replace(ctx, sl_mult_sig=stop_floor_to_sl_mult(float(ctx.sl_mult_sig), float(floor_bps)))
    exit_res = modela.simulate_frozen_1h_exit(
        ctx=new_ctx,
        fill_time=fill_time,
        fill_price=float(fill_price),
        one_h=one_h,
        args=exec_args,
    )
    liq = str(row_s.get("exec_fill_liquidity_type", "")).strip().lower()
    liq = "taker" if liq == "taker" else "maker"
    cost = modela.exec3m._costed_pnl_long(  # pylint: disable=protected-access
        entry_price=float(fill_price),
        exit_price=float(exit_res.get("exit_price", np.nan)),
        entry_liquidity_type=str(liq),
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )

    out = dict(row_s.to_dict())
    out["exec_valid_for_metrics"] = int(exit_res.get("valid_for_metrics", 0))
    out["exec_sl_hit"] = int(exit_res.get("sl_hit", 0))
    out["exec_tp_hit"] = int(exit_res.get("tp_hit", 0))
    out["exec_pnl_net_pct"] = float(cost.get("pnl_net_pct", np.nan))
    out["exec_pnl_gross_pct"] = float(cost.get("pnl_gross_pct", np.nan))
    out["exec_mae_pct"] = float(exit_res.get("mae_pct", np.nan))
    out["exec_mfe_pct"] = float(exit_res.get("mfe_pct", np.nan))
    out["exec_exit_reason"] = str(exit_res.get("exit_reason", ""))
    out["exec_exit_time"] = str(exit_res.get("exit_time", ""))
    out["exec_exit_price"] = float(exit_res.get("exit_price", np.nan))
    out["exec_same_bar_hit"] = int(exit_res.get("same_bar_hit", 0))
    out["exec_invalid_stop_geometry"] = int(exit_res.get("invalid_stop_geometry", 0))
    out["exec_invalid_tp_geometry"] = int(exit_res.get("invalid_tp_geometry", 0))
    out["repair_room_floor_bps_applied"] = float(floor_bps)
    out["repair_effective_sl_mult"] = float(new_ctx.sl_mult_sig)
    return out


def apply_variant(
    *,
    control_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    ctx_map: Dict[str, modela.ga_exec.SignalContext],
    one_h: modela.OneHMarket,
    exec_args: argparse.Namespace,
    thresholds: Dict[str, float],
    spec: Dict[str, Any],
) -> pd.DataFrame:
    feature_lookup = {
        str(r["signal_id"]): r
        for _, r in feature_df.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for _, row_s in control_df.iterrows():
        sid = str(row_s.get("signal_id", ""))
        feat_row = feature_lookup.get(sid)
        allowed = passes_filter(spec=spec, feature_row=feat_row, thresholds=thresholds)
        out = dict(row_s.to_dict())
        out["repair_filter_pass"] = int(allowed)
        out["repair_variant_id"] = str(spec["variant_id"])
        out["repair_variant_label"] = str(spec["label"])
        out["repair_room_floor_bps_applied"] = 0.0
        out["repair_effective_sl_mult"] = float(ctx_map[sid].sl_mult_sig) if sid in ctx_map else float("nan")

        if not allowed:
            rows.append(zero_exec_fields(out, skip_reason=f"repair_filter_reject:{spec['variant_id']}"))
            continue

        room_hit = room_applies(spec=spec, feature_row=feat_row, thresholds=thresholds, allowed=allowed)
        out["repair_room_condition_hit"] = int(room_hit)
        if room_hit and int(row_s.get("exec_filled", 0)) == 1 and sid in ctx_map:
            out = rerun_exit_with_room(
                row_s=pd.Series(out),
                ctx=ctx_map[sid],
                one_h=one_h,
                exec_args=exec_args,
                floor_bps=float(spec.get("room_floor_bps", 0.0)),
            )
            out["repair_variant_id"] = str(spec["variant_id"])
            out["repair_variant_label"] = str(spec["label"])
            out["repair_filter_pass"] = int(allowed)
            out["repair_room_condition_hit"] = int(room_hit)
        else:
            out["repair_room_condition_hit"] = int(room_hit)
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def compute_split_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_id, g in df.groupby("split_id", dropna=False):
        agg = modela.ga_exec._aggregate_rows(g)  # pylint: disable=protected-access
        rows.append(
            {
                "split_id": int(split_id),
                "signals_total": int(agg["exec"]["signals_total"]),
                "delta_expectancy_exec_minus_baseline": float(agg["delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(agg["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(agg["maxdd_improve_ratio"]),
            }
        )
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)


def compute_valid_for_ranking(
    *,
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    bundle: modela.ga_exec.SymbolBundle,
    exec_args: argparse.Namespace,
    rollup: Dict[str, Any],
) -> Tuple[int, str]:
    e = rollup["exec"]
    signals_total = int(e["signals_total"])
    entries_valid = int(e["entries_valid"])
    min_trades_symbol = max(
        int(exec_args.hard_min_trades_symbol),
        int(math.ceil(float(exec_args.hard_min_trade_frac_symbol) * max(1, signals_total))),
    )
    min_trades_overall = max(
        int(exec_args.hard_min_trades_overall),
        int(math.ceil(float(exec_args.hard_min_trade_frac_overall) * max(1, signals_total))),
    )

    min_entry_symbol = max(float(exec_args.hard_min_entry_rate_symbol), float(bundle.constraints.get("min_entry_rate", 0.0)))
    min_entry_overall = float(exec_args.hard_min_entry_rate_overall)
    max_taker = min(float(exec_args.hard_max_taker_share), float(bundle.constraints.get("max_taker_share", 1.0)))
    max_median_delay = min(float(exec_args.hard_max_median_fill_delay_min), float(bundle.constraints.get("max_fill_delay_min", 1e9)))
    min_median_improve = float(bundle.constraints.get("min_median_entry_improvement_bps", -9999.0))

    missing_slice_rate = float(pd.to_numeric(df.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df.empty else 0.0
    split_delta = pd.to_numeric(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float)), errors="coerce")
    req = [
        float(e["mean_expectancy_net"]),
        float(e["cvar_5"]),
        float(e["max_drawdown"]),
        float(e["entry_rate"]),
        float(e["taker_share"]),
        float(e["median_fill_delay_min"]),
        float(e["p95_fill_delay_min"]),
        float(e["median_entry_improvement_bps"]),
    ]

    invalid: List[str] = []
    if not (np.isfinite(e["entry_rate"]) and float(e["entry_rate"]) >= min_entry_symbol):
        invalid.append(f"{bundle.symbol}:entry_rate")
    if entries_valid < int(min_trades_symbol):
        invalid.append(f"{bundle.symbol}:trades<{min_trades_symbol}")
    if not (np.isfinite(e["entry_rate"]) and float(e["entry_rate"]) >= min_entry_overall):
        invalid.append("overall:entry_rate")
    if entries_valid < int(min_trades_overall):
        invalid.append(f"overall:trades<{min_trades_overall}")
    if not (np.isfinite(e["taker_share"]) and float(e["taker_share"]) <= max_taker):
        invalid.append(f"{bundle.symbol}:taker_share")
    if not (np.isfinite(e["median_fill_delay_min"]) and float(e["median_fill_delay_min"]) <= max_median_delay):
        invalid.append(f"{bundle.symbol}:median_fill_delay")
    if not (np.isfinite(e["p95_fill_delay_min"]) and float(e["p95_fill_delay_min"]) <= float(exec_args.hard_max_p95_fill_delay_min)):
        invalid.append(f"{bundle.symbol}:p95_fill_delay")
    if not (np.isfinite(e["median_entry_improvement_bps"]) and float(e["median_entry_improvement_bps"]) >= min_median_improve):
        invalid.append(f"{bundle.symbol}:median_entry_improvement")
    if not (np.isfinite(missing_slice_rate) and missing_slice_rate <= float(exec_args.hard_max_missing_slice_rate)):
        invalid.append(f"{bundle.symbol}:missing_slice_rate>{float(exec_args.hard_max_missing_slice_rate):.4f}")
    if not all(np.isfinite(v) for v in req):
        invalid.append(f"{bundle.symbol}:nan_or_inf")
    if split_df.empty or not split_delta.notna().all():
        invalid.append("split_metrics_missing_or_nan")

    return int(len(invalid) == 0), "|".join(sorted(set(invalid)))


def compute_variant_metrics(
    *,
    df: pd.DataFrame,
    bundle: modela.ga_exec.SymbolBundle,
    exec_args: argparse.Namespace,
) -> Dict[str, Any]:
    agg = modela.ga_exec._aggregate_rows(df)  # pylint: disable=protected-access
    split_df = compute_split_metrics(df)
    valid_for_ranking, invalid_reason = compute_valid_for_ranking(
        df=df,
        split_df=split_df,
        bundle=bundle,
        exec_args=exec_args,
        rollup=agg,
    )

    mask = (
        pd.to_numeric(df.get("exec_filled", 0), errors="coerce").fillna(0).astype(int) == 1
    ) & (
        pd.to_numeric(df.get("exec_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1
    )
    pnl = pd.to_numeric(df.get("exec_pnl_net_pct", np.nan), errors="coerce")
    hold = pd.Series([hold_minutes_from_row(r) for _, r in df.iterrows()], index=df.index, dtype=float)
    exit_reason = df.get("exec_exit_reason", pd.Series(dtype=object)).fillna("").astype(str)
    valid_mask = mask & pnl.notna()
    valid_df = df.loc[valid_mask].copy()
    valid_hold = hold.loc[valid_mask]
    valid_pnl = pnl.loc[valid_mask]
    valid_reason = exit_reason.loc[valid_mask]

    instant_mask = valid_reason.map(is_stop) & (valid_hold <= 60.0)
    fast_mask = valid_reason.map(is_stop) & (valid_hold > 60.0) & (valid_hold <= 240.0)
    meaningful_mask = valid_pnl >= float(MEANINGFUL_WINNER_THRESHOLD)

    parity_clean = int(
        (
            pd.to_numeric(df.get("exec_same_bar_hit", 0), errors="coerce").fillna(0).astype(int).sum() == 0
        )
        and (
            pd.to_numeric(df.get("exec_invalid_stop_geometry", 0), errors="coerce").fillna(0).astype(int).sum() == 0
        )
        and (
            pd.to_numeric(df.get("exec_invalid_tp_geometry", 0), errors="coerce").fillna(0).astype(int).sum() == 0
        )
        and (
            pd.to_numeric(df.get("lookahead_violation", 0), errors="coerce").fillna(0).astype(int).sum() == 0
        )
    )

    return {
        "signals_total": int(agg["exec"]["signals_total"]),
        "total_trades": int(agg["exec"]["entries_valid"]),
        "instant_loser_count": int(instant_mask.sum()),
        "fast_loser_count": int(fast_mask.sum()),
        "meaningful_winner_count": int(meaningful_mask.sum()),
        "expectancy_net": float(agg["exec"]["mean_expectancy_net"]),
        "cvar_5": float(agg["exec"]["cvar_5"]),
        "max_drawdown": float(agg["exec"]["max_drawdown"]),
        "win_rate": float((valid_pnl > 0.0).mean()) if not valid_pnl.empty else float("nan"),
        "median_hold_minutes": float(valid_hold.median()) if not valid_hold.dropna().empty else float("nan"),
        "pct_exit_within_3h": float(100.0 * (valid_hold <= 180.0).sum() / max(1, len(valid_hold))),
        "pct_exit_within_4h": float(100.0 * (valid_hold <= 240.0).sum() / max(1, len(valid_hold))),
        "entry_rate": float(agg["exec"]["entry_rate"]),
        "taker_share": float(agg["exec"]["taker_share"]),
        "median_fill_delay_min": float(agg["exec"]["median_fill_delay_min"]),
        "p95_fill_delay_min": float(agg["exec"]["p95_fill_delay_min"]),
        "median_entry_improvement_bps": float(agg["exec"]["median_entry_improvement_bps"]),
        "valid_for_ranking": int(valid_for_ranking),
        "invalid_reason": str(invalid_reason),
        "parity_clean": int(parity_clean),
        "split_count": int(len(split_df)),
        "min_split_delta": float(pd.to_numeric(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float)), errors="coerce").min())
        if not split_df.empty
        else float("nan"),
    }


def accept_variant(control: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    instant_required = max(int(MIN_INSTANT_LOSER_REDUCTION_ABS), int(math.ceil(float(control["instant_loser_count"]) * MIN_INSTANT_LOSER_REDUCTION_FRAC)))
    instant_delta = int(candidate["instant_loser_count"]) - int(control["instant_loser_count"])
    expectancy_delta = float(candidate["expectancy_net"]) - float(control["expectancy_net"])
    cvar_delta = float(candidate["cvar_5"]) - float(control["cvar_5"])
    maxdd_delta = float(candidate["max_drawdown"]) - float(control["max_drawdown"])
    retention = float(candidate["total_trades"]) / max(1, int(control["total_trades"]))

    checks = {
        "instant_losers_materially_down": int(instant_delta <= -int(instant_required)),
        "expectancy_ok": int(expectancy_delta >= -float(EXPECTANCY_TOL)),
        "cvar_ok": int(cvar_delta >= -float(CVAR_TOL)),
        "maxdd_ok": int(maxdd_delta >= -float(MAXDD_TOL)),
        "retention_ok": int(retention >= float(MIN_RETENTION)),
        "parity_clean": int(candidate["parity_clean"] == 1),
    }
    accepted = bool(all(int(v) == 1 for v in checks.values()))
    checks.update(
        {
            "instant_required_abs": int(instant_required),
            "instant_delta": int(instant_delta),
            "expectancy_delta": float(expectancy_delta),
            "cvar_delta": float(cvar_delta),
            "maxdd_delta": float(maxdd_delta),
            "trade_count_retention_vs_control": float(retention),
        }
    )
    return accepted, checks


def decide_coin(
    *,
    symbol: str,
    status: str,
    control_variant_id: str,
    candidate_rows: List[Dict[str, Any]],
) -> Tuple[str, str]:
    non_control = [r for r in candidate_rows if r["variant_id"] != control_variant_id]
    approved = [r for r in non_control if r["accepted"]]
    if approved:
        approved.sort(
            key=lambda r: (
                int(r["valid_for_ranking"] == 0),
                -float(r["expectancy_delta_vs_control"]),
                float(r["instant_delta_vs_control"]),
                -float(r["trade_count_retention_vs_control"]),
            )
        )
        best = approved[0]
        decision = "APPROVED_REPAIR" if int(best["valid_for_ranking"]) == 1 else "SHADOW_REPAIR_ONLY"
        return decision, str(best["variant_id"])
    if status == "shadow":
        return "NO_REPAIR_APPROVED", "NO_REPAIR_APPROVED"
    return "NO_REPAIR_APPROVED", "NO_REPAIR_APPROVED"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded entry-repair pilot for repaired-contract live coins")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--diag-dir", default="", help="Path to INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_*; defaults to latest")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)

    diag_dir = Path(args.diag_dir).resolve() if args.diag_dir else find_latest_diag_dir()
    bucket_df, diag_feature_df, rec_df = load_diagnosis_state(diag_dir)
    diag_manifest_path = diag_dir / "instant_loser_vs_winner_manifest.json"
    diag_manifest = json.loads(diag_manifest_path.read_text(encoding="utf-8")) if diag_manifest_path.exists() else {}

    repaired_multicoin_dir = Path(diag_manifest.get("sources", {}).get("repaired_multicoin_dir", forensics.find_latest_repaired_multicoin_dir())).resolve()
    class_df = forensics.load_best_variant_lookup(repaired_multicoin_dir)
    repaired_manifest_path = repaired_multicoin_dir / "repaired_multicoin_modelA_run_manifest.json"
    repaired_manifest = json.loads(repaired_manifest_path.read_text(encoding="utf-8")) if repaired_manifest_path.exists() else {}
    foundation_dir = Path(repaired_manifest.get("foundation_dir", phase_v.find_latest_foundation_dir())).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state, seed=20260303)
    variant_map = {str(cfg["candidate_id"]): dict(cfg) for cfg in phase_v.sanitize_variants()}

    log("[1/6] Reconstructing control state and running priority pilot")
    result_rows: List[Dict[str, Any]] = []
    vs_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    per_coin_meta: Dict[str, Any] = {}

    trade_dir = ensure_dir(run_dir / "_trade_sources")
    continuation_allowed = False
    priority_symbols = ["LINKUSDT", "NEARUSDT"]
    processed_symbols: List[str] = []

    for symbol in COIN_ORDER:
        if symbol in ["DOGEUSDT", "LTCUSDT"] and not continuation_allowed and set(priority_symbols).issubset(set(processed_symbols)):
            log(f"  - early stop gate active: skipping deeper pilot on {symbol}")
            decision_rows.append(
                {
                    "symbol": symbol,
                    "current_status": CURRENT_STATUS[symbol],
                    "current_action_map": CONTROL_TO_REPAIR_DECISION[symbol],
                    "decision": "NO_REPAIR_APPROVED",
                    "best_variant_id": "NO_REPAIR_APPROVED",
                    "best_variant_label": "not_run_due_priority_gate",
                    "continuation_gate_blocked": 1,
                }
            )
            processed_symbols.append(symbol)
            continue

        row = class_df[class_df["symbol"] == symbol]
        if row.empty:
            raise KeyError(f"Missing repaired multicoin classification for {symbol}")
        best_row = row.iloc[0]
        best_candidate_id = str(best_row["best_candidate_id"]).strip()
        diag_features = feature_subset(diag_feature_df, symbol)
        thresholds = compute_filter_thresholds(symbol, diag_features)
        specs = build_variant_specs(symbol, thresholds)

        log(f"  - {symbol}: control {best_candidate_id}, {len(specs)} variants")
        trade_df, sig_timeline_df, _one_h_df, rebuild_meta = forensics.reconstruct_best_modela_trades(
            symbol=symbol,
            best_candidate_id=best_candidate_id,
            foundation_state=foundation_state,
            exec_args=exec_args,
            run_dir=run_dir,
            variant_map=variant_map,
        )
        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_timeline_df[sig_timeline_df["symbol"].astype(str).str.upper() == symbol].copy(),
            symbol_windows=foundation_state.download_manifest[
                foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol
            ].copy(),
            exec_args=exec_args,
            run_dir=run_dir,
        )
        control_path = trade_dir / f"{symbol}_{best_candidate_id}_control.csv"
        trade_df.to_csv(control_path, index=False)
        one_h = rebuild_meta["one_h_market"]
        ctx_map = {str(ctx.signal_id): ctx for ctx in bundle.contexts}

        variant_outputs: Dict[str, pd.DataFrame] = {}
        control_metrics: Optional[Dict[str, Any]] = None
        variant_rows_for_decision: List[Dict[str, Any]] = []

        for spec in specs:
            if str(spec["variant_id"]) == "CONTROL":
                var_df = trade_df.copy().reset_index(drop=True)
                var_df["repair_filter_pass"] = 1
                var_df["repair_variant_id"] = "CONTROL"
                var_df["repair_variant_label"] = str(spec["label"])
                var_df["repair_room_floor_bps_applied"] = 0.0
                var_df["repair_effective_sl_mult"] = var_df["signal_id"].map(lambda sid: float(ctx_map[str(sid)].sl_mult_sig))
                var_df["repair_room_condition_hit"] = 0
            else:
                var_df = apply_variant(
                    control_df=trade_df,
                    feature_df=diag_features,
                    ctx_map=ctx_map,
                    one_h=one_h,
                    exec_args=exec_args,
                    thresholds=thresholds,
                    spec=spec,
                )

            variant_outputs[str(spec["variant_id"])] = var_df
            metrics = compute_variant_metrics(df=var_df, bundle=bundle, exec_args=exec_args)
            if str(spec["variant_id"]) == "CONTROL":
                control_metrics = dict(metrics)
            if control_metrics is None:
                raise RuntimeError("Control metrics missing before evaluating repair variants")

            accepted, checks = (
                (False, {
                    "instant_losers_materially_down": 1,
                    "expectancy_ok": 1,
                    "cvar_ok": 1,
                    "maxdd_ok": 1,
                    "retention_ok": 1,
                    "parity_clean": int(metrics["parity_clean"]),
                    "instant_required_abs": 0,
                    "instant_delta": 0,
                    "expectancy_delta": 0.0,
                    "cvar_delta": 0.0,
                    "maxdd_delta": 0.0,
                    "trade_count_retention_vs_control": 1.0,
                }) if str(spec["variant_id"]) == "CONTROL" else accept_variant(control_metrics, metrics)
            )

            result_row = {
                "symbol": symbol,
                "current_status": CURRENT_STATUS[symbol],
                "current_action_map": CONTROL_TO_REPAIR_DECISION[symbol],
                "classification": str(best_row["classification"]),
                "control_candidate_id": best_candidate_id,
                "variant_id": str(spec["variant_id"]),
                "variant_label": str(spec["label"]),
                "instant_loser_count": int(metrics["instant_loser_count"]),
                "fast_loser_count": int(metrics["fast_loser_count"]),
                "meaningful_winner_count": int(metrics["meaningful_winner_count"]),
                "expectancy_net": float(metrics["expectancy_net"]),
                "cvar_5": float(metrics["cvar_5"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "win_rate": float(metrics["win_rate"]),
                "total_trades": int(metrics["total_trades"]),
                "median_hold_minutes": float(metrics["median_hold_minutes"]),
                "pct_exit_within_3h": float(metrics["pct_exit_within_3h"]),
                "pct_exit_within_4h": float(metrics["pct_exit_within_4h"]),
                "valid_for_ranking": int(metrics["valid_for_ranking"]),
                "invalid_reason": str(metrics["invalid_reason"]),
                "parity_clean": int(metrics["parity_clean"]),
                "room_floor_bps": float(spec.get("room_floor_bps", 0.0)),
                "accepted": int(accepted),
                "instant_required_abs": int(checks["instant_required_abs"]),
                "instant_delta_vs_control": int(checks["instant_delta"]),
                "expectancy_delta_vs_control": float(checks["expectancy_delta"]),
                "cvar_delta_vs_control": float(checks["cvar_delta"]),
                "maxdd_delta_vs_control": float(checks["maxdd_delta"]),
                "trade_count_retention_vs_control": float(checks["trade_count_retention_vs_control"]),
                "accept_instant_losers_materially_down": int(checks["instant_losers_materially_down"]),
                "accept_expectancy_ok": int(checks["expectancy_ok"]),
                "accept_cvar_ok": int(checks["cvar_ok"]),
                "accept_maxdd_ok": int(checks["maxdd_ok"]),
                "accept_retention_ok": int(checks["retention_ok"]),
                "accept_parity_clean": int(checks["parity_clean"]),
                "source_trade_csv": str(control_path if str(spec["variant_id"]) == "CONTROL" else trade_dir / f"{symbol}_{spec['variant_id']}.csv"),
            }
            result_rows.append(result_row)
            variant_rows_for_decision.append(result_row)

            out_path = trade_dir / f"{symbol}_{spec['variant_id']}.csv"
            var_df.to_csv(out_path, index=False)

        decision, best_variant_id = decide_coin(
            symbol=symbol,
            status=CURRENT_STATUS[symbol],
            control_variant_id="CONTROL",
            candidate_rows=variant_rows_for_decision,
        )
        best_row_dec = next((r for r in variant_rows_for_decision if str(r["variant_id"]) == str(best_variant_id)), None)
        if best_row_dec is None:
            best_row_dec = next((r for r in variant_rows_for_decision if str(r["variant_id"]) == "CONTROL"), None)
        decision_rows.append(
            {
                "symbol": symbol,
                "current_status": CURRENT_STATUS[symbol],
                "current_action_map": CONTROL_TO_REPAIR_DECISION[symbol],
                "decision": decision,
                "best_variant_id": best_variant_id,
                "best_variant_label": str(best_row_dec["variant_label"]) if best_row_dec is not None else "",
                "continuation_gate_blocked": 0,
                "best_variant_valid_for_ranking": int(best_row_dec["valid_for_ranking"]) if best_row_dec is not None else 0,
                "best_variant_accepted": int(best_row_dec["accepted"]) if best_row_dec is not None else 0,
            }
        )

        control_row = next(r for r in variant_rows_for_decision if str(r["variant_id"]) == "CONTROL")
        best_compare = best_row_dec if best_row_dec is not None else control_row
        vs_rows.append(
            {
                "symbol": symbol,
                "decision": decision,
                "best_variant_id": str(best_compare["variant_id"]),
                "best_variant_label": str(best_compare["variant_label"]),
                "before_instant_loser_count": int(control_row["instant_loser_count"]),
                "after_instant_loser_count": int(best_compare["instant_loser_count"]),
                "before_fast_loser_count": int(control_row["fast_loser_count"]),
                "after_fast_loser_count": int(best_compare["fast_loser_count"]),
                "before_meaningful_winner_count": int(control_row["meaningful_winner_count"]),
                "after_meaningful_winner_count": int(best_compare["meaningful_winner_count"]),
                "before_expectancy_net": float(control_row["expectancy_net"]),
                "after_expectancy_net": float(best_compare["expectancy_net"]),
                "before_cvar_5": float(control_row["cvar_5"]),
                "after_cvar_5": float(best_compare["cvar_5"]),
                "before_max_drawdown": float(control_row["max_drawdown"]),
                "after_max_drawdown": float(best_compare["max_drawdown"]),
                "trade_count_retention_vs_control": float(best_compare["trade_count_retention_vs_control"]),
                "before_total_trades": int(control_row["total_trades"]),
                "after_total_trades": int(best_compare["total_trades"]),
                "before_pct_exit_within_3h": float(control_row["pct_exit_within_3h"]),
                "after_pct_exit_within_3h": float(best_compare["pct_exit_within_3h"]),
                "before_pct_exit_within_4h": float(control_row["pct_exit_within_4h"]),
                "after_pct_exit_within_4h": float(best_compare["pct_exit_within_4h"]),
                "parity_clean_after": int(best_compare["parity_clean"]),
                "best_variant_valid_for_ranking": int(best_compare["valid_for_ranking"]),
            }
        )

        per_coin_meta[symbol] = {
            "thresholds": thresholds,
            "control_candidate_id": best_candidate_id,
            "build_meta": build_meta,
            "reconstruct_meta": {
                "build_meta": rebuild_meta["build_meta"],
                "metrics": rebuild_meta["metrics"],
            },
            "variant_ids_tested": [str(spec["variant_id"]) for spec in specs],
            "control_trade_source_csv": str(control_path),
            "diagnosis_trade_bucket_row": bucket_df[bucket_df["symbol"] == symbol].to_dict("records"),
            "diagnosis_recommendation_row": rec_df[rec_df["symbol"] == symbol].to_dict("records"),
        }
        processed_symbols.append(symbol)

        if symbol in priority_symbols:
            if any(int(r["accepted"]) == 1 for r in variant_rows_for_decision if str(r["variant_id"]) != "CONTROL"):
                continuation_allowed = True

    log("[2/6] Writing CSV artifacts")
    results_df = pd.DataFrame(result_rows).sort_values(["symbol", "variant_id"]).reset_index(drop=True)
    vs_df = pd.DataFrame(vs_rows).sort_values("symbol").reset_index(drop=True)
    decisions_df = pd.DataFrame(decision_rows).sort_values("symbol").reset_index(drop=True)

    results_df.to_csv(run_dir / "live_coin_bounded_entry_repair_results.csv", index=False)
    vs_df.to_csv(run_dir / "live_coin_bounded_entry_repair_vs_control.csv", index=False)
    decisions_df.to_csv(run_dir / "live_coin_bounded_entry_repair_decision.csv", index=False)

    log("[3/6] Deriving deployment posture")
    approved_paper: List[str] = []
    shadow_only: List[str] = []
    disabled: List[str] = []
    for symbol in COIN_ORDER:
        dec_row = decisions_df[decisions_df["symbol"] == symbol]
        decision = str(dec_row.iloc[0]["decision"]) if not dec_row.empty else "NO_REPAIR_APPROVED"
        prior = CURRENT_STATUS[symbol]
        if decision == "APPROVED_REPAIR":
            approved_paper.append(symbol)
        elif decision == "SHADOW_REPAIR_ONLY":
            shadow_only.append(symbol)
        elif decision == "DISABLE_COIN":
            disabled.append(symbol)
        else:
            if prior == "approved":
                approved_paper.append(symbol)
            elif prior == "shadow":
                shadow_only.append(symbol)
            else:
                disabled.append(symbol)

    log("[4/6] Writing markdown report")
    priority_gate_message = (
        "At least one repair on LINKUSDT/NEARUSDT survived, so DOGEUSDT and LTCUSDT were evaluated."
        if continuation_allowed
        else "No repair on LINKUSDT/NEARUSDT survived the acceptance rules, so DOGEUSDT and LTCUSDT were not fully evaluated."
    )
    report_lines = [
        "# Live-Coin Bounded Entry Repair Pilot",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        f"- Frozen diagnosis dir: `{diag_dir}`",
        f"- Repaired multicoin source: `{repaired_multicoin_dir}`",
        "",
        "## Control / Acceptance Rules",
        "",
        "- Control is the current repaired best candidate per live coin.",
        "- Tolerances: expectancy >= control - 0.00005; cvar_5 >= control - 0.00010; maxDD >= control - 0.01.",
        f"- Trade retention floor: `{MIN_RETENTION:.2f}`.",
        "- Instant-loser reduction must be at least max(5 trades, 5% of control instant losers).",
        "- Parity must remain clean (no same-parent-bar exits, no invalid stop/TP geometry, no lookahead).",
        "",
        f"- Priority gate result: {priority_gate_message}",
        "",
        "## Variant Results",
        "",
        markdown_table(
            results_df,
            [
                "symbol",
                "variant_id",
                "instant_loser_count",
                "fast_loser_count",
                "meaningful_winner_count",
                "expectancy_net",
                "cvar_5",
                "max_drawdown",
                "trade_count_retention_vs_control",
                "accepted",
            ],
        ),
        "",
        "## Best Vs Control",
        "",
        markdown_table(
            vs_df,
            [
                "symbol",
                "decision",
                "best_variant_id",
                "before_instant_loser_count",
                "after_instant_loser_count",
                "before_fast_loser_count",
                "after_fast_loser_count",
                "before_expectancy_net",
                "after_expectancy_net",
                "trade_count_retention_vs_control",
            ],
        ),
        "",
        "## Deployment Posture",
        "",
        f"- approved_paper: `{', '.join(approved_paper) if approved_paper else '(none)'}`",
        f"- shadow_only: `{', '.join(shadow_only) if shadow_only else '(none)'}`",
        f"- disabled: `{', '.join(disabled) if disabled else '(none)'}`",
    ]
    (run_dir / "live_coin_bounded_entry_repair_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log("[5/6] Writing manifest")
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "frozen_diagnosis_dir": str(diag_dir),
        "repaired_multicoin_dir": str(repaired_multicoin_dir),
        "repaired_multicoin_manifest_json": str(repaired_manifest_path),
        "foundation_dir": str(foundation_dir),
        "coin_order": list(COIN_ORDER),
        "current_status": dict(CURRENT_STATUS),
        "current_action_map": dict(CONTROL_TO_REPAIR_DECISION),
        "parity_guards": {
            "same_parent_bar_exit_disabled": int(repaired_manifest.get("contract_assumptions", {}).get("same_parent_bar_exit_disabled", 1)),
            "one_h_signal_owner": 1,
            "one_h_exit_owner": 1,
            "three_m_entry_only": 1,
        },
        "acceptance_rules": {
            "expectancy_tolerance": float(EXPECTANCY_TOL),
            "cvar_tolerance": float(CVAR_TOL),
            "maxdd_tolerance": float(MAXDD_TOL),
            "min_trade_count_retention": float(MIN_RETENTION),
            "instant_loser_reduction_abs": int(MIN_INSTANT_LOSER_REDUCTION_ABS),
            "instant_loser_reduction_frac": float(MIN_INSTANT_LOSER_REDUCTION_FRAC),
        },
        "continuation_allowed_after_priority": int(continuation_allowed),
        "priority_gate_message": priority_gate_message,
        "per_coin_meta": per_coin_meta,
        "deployment_posture": {
            "approved_paper": approved_paper,
            "shadow_only": shadow_only,
            "disabled": disabled,
        },
        "outputs": {
            "results_csv": str(run_dir / "live_coin_bounded_entry_repair_results.csv"),
            "vs_control_csv": str(run_dir / "live_coin_bounded_entry_repair_vs_control.csv"),
            "decision_csv": str(run_dir / "live_coin_bounded_entry_repair_decision.csv"),
            "report_md": str(run_dir / "live_coin_bounded_entry_repair_report.md"),
            "manifest_json": str(run_dir / "live_coin_bounded_entry_repair_manifest.json"),
        },
    }
    json_dump(run_dir / "live_coin_bounded_entry_repair_manifest.json", manifest)
    log("[6/6] Complete")
    log(str(run_dir))


if __name__ == "__main__":
    main()
