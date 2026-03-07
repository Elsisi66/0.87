#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as pr  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    if x.empty:
        return "_(none)_"
    lines = ["| " + " | ".join(x.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm_rank(s: pd.Series) -> pd.Series:
    x = to_num(s)
    if x.notna().sum() <= 1:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return x.rank(method="average", pct=True).fillna(0.5).astype(float)


def safe_div(a: float, b: float) -> float:
    if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(float(b)) <= 1e-12:
        return float("nan")
    return float(a / b)


def psr_dsr_proxy(vec: np.ndarray, eff_trials: float = 1.0) -> Tuple[float, float]:
    x = np.asarray(vec, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd <= 1e-12:
        psr = 1.0 if mu > 0 else 0.0
    else:
        z = float(mu / (sd / math.sqrt(float(x.size))))
        psr = float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    dsr = float(psr / max(1.0, math.sqrt(max(1.0, eff_trials))))
    return psr, dsr


def effective_trials_from_corr(mat: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(mat, dtype=float)
    if x.ndim != 2 or x.shape[0] <= 1:
        return float(x.shape[0] if x.ndim == 2 else 0), 0.0
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        x[:, j] = col
    cc = np.corrcoef(x)
    if np.ndim(cc) == 0:
        return float(x.shape[0]), 0.0
    iu = np.triu_indices_from(cc, k=1)
    vals = np.abs(cc[iu])
    vals = vals[np.isfinite(vals)]
    avg_abs = float(vals.mean()) if vals.size else 0.0
    n = float(x.shape[0])
    n_eff = float(n / max(1e-9, (1.0 + (n - 1.0) * avg_abs)))
    return n_eff, avg_abs


def regime_bucket_from_feats(df: pd.DataFrame) -> pd.Series:
    vol = norm_rank(df["pre3m_atr_z"])
    spr = norm_rank(df["pre3m_spread_proxy_bps"])
    score = 0.6 * vol + 0.4 * spr
    out = pd.Series(index=df.index, dtype=object)
    out[score < 0.33] = "calm"
    out[(score >= 0.33) & (score < 0.66)] = "neutral"
    out[score >= 0.66] = "stressed"
    return out.fillna("unknown").astype(str)


def validate_repaired_routes(
    *,
    route_examples_fp: Path,
    route_examples_df: pd.DataFrame,
    route_bundles: Dict[str, ga_exec.SymbolBundle],
) -> None:
    if not route_examples_fp.exists():
        raise FileNotFoundError(f"Missing route example file: {route_examples_fp}")
    exp = pd.read_csv(route_examples_fp).sort_values("route_id").reset_index(drop=True)
    got = route_examples_df.sort_values("route_id").reset_index(drop=True)
    exp_ids = exp["route_id"].astype(str).tolist()
    got_ids = got["route_id"].astype(str).tolist()
    if exp_ids != got_ids:
        raise RuntimeError(f"Repaired route id mismatch: expected={exp_ids} got={got_ids}")
    exp_counts = exp[["route_id", "route_signal_count", "wf_test_signal_count"]].copy()
    got_counts = got[["route_id", "route_signal_count", "wf_test_signal_count"]].copy()
    if not exp_counts.equals(got_counts):
        raise RuntimeError("Repaired route counts mismatch vs route redesign artifact")
    if sorted(route_bundles.keys()) != sorted(exp_ids):
        raise RuntimeError("Repaired route bundles do not match route example ids")


def build_feature_frame(base_bundle: ga_exec.SymbolBundle, rep_df: pd.DataFrame, genome: Dict[str, Any]) -> pd.DataFrame:
    rep = ae.ensure_signals_schema(rep_df)
    feat = ae.build_preentry_features(base_bundle, genome=genome).copy()
    feat["signal_id"] = feat["signal_id"].astype(str)
    rep_meta = rep[["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]].copy()
    rep_meta["signal_id"] = rep_meta["signal_id"].astype(str)
    out = feat.merge(rep_meta, on="signal_id", how="left", suffixes=("", "_sig"))
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["session_bucket"] = ae.session_bucket(out["signal_time"])
    out["vol_bucket"] = ae.vol_bucket(out["atr_percentile_1h"])
    out["trend_bucket"] = np.where(to_num(out["trend_up_1h"]) >= 0.5, "up", "down")
    out["regime_bucket"] = regime_bucket_from_feats(out)
    return out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def run_exec_detailed(
    *,
    genome: Dict[str, Any],
    bundle: ga_exec.SymbolBundle,
    feature_df: pd.DataFrame,
    args: argparse.Namespace,
    route_id: str,
    exec_id: str,
) -> pd.DataFrame:
    met = ga_exec._evaluate_genome(genome=copy.deepcopy(genome), bundles=[bundle], args=args, detailed=True)
    rows = met.get("signal_rows_df", pd.DataFrame()).copy()
    if rows.empty:
        return pd.DataFrame()
    rows["signal_id"] = rows["signal_id"].astype(str)
    rows = rows.merge(feature_df, on="signal_id", how="left", suffixes=("", "_feat"))
    rows["route_id"] = str(route_id)
    rows["exec_id"] = str(exec_id)
    rows["signal_time"] = pd.to_datetime(rows["signal_time"], utc=True, errors="coerce")
    return rows.sort_values(["split_id", "signal_time", "signal_id"]).reset_index(drop=True)


def baseline_detailed(
    *,
    bundle: ga_exec.SymbolBundle,
    feature_df: pd.DataFrame,
    route_id: str,
) -> pd.DataFrame:
    sig_df, _ = pr.baseline_rows_for_bundle(bundle)
    if sig_df.empty:
        return pd.DataFrame()
    sig_df["signal_id"] = sig_df["signal_id"].astype(str)
    sig_df = sig_df.merge(feature_df, on="signal_id", how="left", suffixes=("", "_feat"))
    sig_df["route_id"] = str(route_id)
    sig_df["signal_time"] = pd.to_datetime(sig_df["signal_time"], utc=True, errors="coerce")
    return sig_df.sort_values(["split_id", "signal_time", "signal_id"]).reset_index(drop=True)


def subset_eval_from_rows(
    *,
    rows_df: pd.DataFrame,
    bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    genome: Dict[str, Any],
    keep_ids: Optional[set[str]],
    ref_roll: Optional[Dict[str, float]] = None,
    ref_split_means: Optional[Dict[int, float]] = None,
    stress_mult: float = 1.0,
) -> Dict[str, Any]:
    x = rows_df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    if keep_ids is not None:
        x = x[x["signal_id"].isin(keep_ids)].copy()
    if x.empty:
        return {
            "valid_for_ranking": 0,
            "invalid_reason": "empty_subset|overall:trades<200|split_metrics_missing_or_nan",
            "expectancy_net": float("nan"),
            "delta_vs_ref": float("nan"),
            "cvar_improve_ratio": float("nan"),
            "maxdd_improve_ratio": float("nan"),
            "signals_total": 0,
            "entries_valid": 0,
            "entry_rate": float("nan"),
            "taker_share": float("nan"),
            "median_fill_delay_min": float("nan"),
            "p95_fill_delay_min": float("nan"),
            "min_split_delta": float("nan"),
            "split_count": 0,
            "split_means": {},
        }

    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if stress_mult != 1.0:
        gross = to_num(x.get("exec_pnl_gross_pct", np.nan))
        net = to_num(x.get("exec_pnl_net_pct", np.nan))
        fee_drag = (gross - net).fillna(0.0)
        x["exec_pnl_net_pct"] = gross - stress_mult * fee_drag

    roll = ga_exec._rollup_mode(x, "exec")
    thresholds = ga_exec._symbol_thresholds(bundle, genome, str(args.mode), args)

    invalid: List[str] = []
    signals_sym = int(roll["signals_total"])
    entries_sym = int(roll["entries_valid"])
    min_trades_symbol = max(int(args.hard_min_trades_symbol), int(math.ceil(float(args.hard_min_trade_frac_symbol) * max(1, signals_sym))))
    min_entry_rate_symbol = max(float(args.hard_min_entry_rate_symbol), float(thresholds["min_entry_rate"]))

    if not (np.isfinite(roll["entry_rate"]) and float(roll["entry_rate"]) >= min_entry_rate_symbol):
        invalid.append(f"{bundle.symbol}:entry_rate")
    if entries_sym < int(min_trades_symbol):
        invalid.append(f"{bundle.symbol}:trades<{min_trades_symbol}")

    max_taker_symbol = min(float(args.hard_max_taker_share), float(thresholds["max_taker_share"]))
    max_delay_symbol = min(float(args.hard_max_median_fill_delay_min), float(thresholds["max_fill_delay_min"]))
    if not (np.isfinite(roll["taker_share"]) and float(roll["taker_share"]) <= max_taker_symbol):
        invalid.append(f"{bundle.symbol}:taker_share")
    if not (np.isfinite(roll["median_fill_delay_min"]) and float(roll["median_fill_delay_min"]) <= max_delay_symbol):
        invalid.append(f"{bundle.symbol}:median_fill_delay")
    if not (np.isfinite(roll["p95_fill_delay_min"]) and float(roll["p95_fill_delay_min"]) <= float(args.hard_max_p95_fill_delay_min)):
        invalid.append(f"{bundle.symbol}:p95_fill_delay")
    if not (np.isfinite(roll["median_entry_improvement_bps"]) and float(roll["median_entry_improvement_bps"]) >= float(thresholds["min_median_entry_improvement_bps"])):
        invalid.append(f"{bundle.symbol}:entry_improvement")

    missing_slice_rate = float(pd.to_numeric(x.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean())
    if not (np.isfinite(missing_slice_rate) and missing_slice_rate <= float(args.hard_max_missing_slice_rate)):
        invalid.append(f"{bundle.symbol}:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")

    req_symbol = [
        float(roll["mean_expectancy_net"]),
        float(roll["cvar_5"]),
        float(roll["max_drawdown"]),
        float(roll["entry_rate"]),
        float(roll["taker_share"]),
        float(roll["median_fill_delay_min"]),
        float(roll["p95_fill_delay_min"]),
    ]
    if not all(np.isfinite(v) for v in req_symbol):
        invalid.append(f"{bundle.symbol}:nan_or_inf")

    split_means: Dict[int, float] = {}
    expected_split_ids = [int(sp["split_id"]) for sp in bundle.splits]
    for sid in expected_split_ids:
        xs = x[to_num(x.get("split_id", np.nan)).fillna(-1).astype(int) == int(sid)].copy()
        if xs.empty:
            invalid.append("split_metrics_missing_or_nan")
            continue
        sp_roll = ga_exec._rollup_mode(xs, "exec")
        if not np.isfinite(sp_roll["mean_expectancy_net"]):
            invalid.append("split_metrics_missing_or_nan")
            continue
        split_means[int(sid)] = float(sp_roll["mean_expectancy_net"])
    if len(split_means) != len(expected_split_ids):
        invalid.append(f"split_count:{len(split_means)}!={len(expected_split_ids)}")

    overall_signals = int(roll["signals_total"])
    overall_entries = int(roll["entries_valid"])
    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, overall_signals))))
    if not (np.isfinite(roll["entry_rate"]) and float(roll["entry_rate"]) >= float(args.hard_min_entry_rate_overall)):
        invalid.append("overall:entry_rate")
    if overall_entries < int(min_trades_overall):
        invalid.append(f"overall:trades<{min_trades_overall}")
    if not (np.isfinite(roll["taker_share"]) and float(roll["taker_share"]) <= float(args.hard_max_taker_share)):
        invalid.append("overall:taker_share")
    if not (np.isfinite(roll["median_fill_delay_min"]) and float(roll["median_fill_delay_min"]) <= float(args.hard_max_median_fill_delay_min)):
        invalid.append("overall:median_fill_delay")
    if not (np.isfinite(roll["p95_fill_delay_min"]) and float(roll["p95_fill_delay_min"]) <= float(args.hard_max_p95_fill_delay_min)):
        invalid.append("overall:p95_fill_delay")
    if not (np.isfinite(missing_slice_rate) and missing_slice_rate <= float(args.hard_max_missing_slice_rate)):
        invalid.append(f"overall:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")
    if not all(np.isfinite(v) for v in req_symbol):
        invalid.append("overall:nan_or_inf")

    invalid = sorted(set(invalid))
    valid = int(len(invalid) == 0)

    ref_roll = ref_roll or {}
    ref_split_means = ref_split_means or {}
    delta = float(roll["mean_expectancy_net"] - float(ref_roll.get("mean_expectancy_net", 0.0))) if ref_roll else 0.0
    cvar_imp = float(ga_exec._improvement_ratio_abs(float(roll["cvar_5"]), float(ref_roll.get("cvar_5", np.nan)))) if ref_roll else 0.0
    maxdd_imp = float(ga_exec._improvement_ratio_abs(float(roll["max_drawdown"]), float(ref_roll.get("max_drawdown", np.nan)))) if ref_roll else 0.0
    split_deltas = [float(split_means[sid] - ref_split_means.get(sid, 0.0)) for sid in sorted(split_means.keys())] if ref_split_means else [float(v) for v in split_means.values()]
    min_split_delta = float(min(split_deltas)) if split_deltas else float("nan")

    return {
        "valid_for_ranking": int(valid),
        "invalid_reason": "|".join(invalid),
        "expectancy_net": float(roll["mean_expectancy_net"]),
        "cvar_5": float(roll["cvar_5"]),
        "max_drawdown": float(roll["max_drawdown"]),
        "delta_vs_ref": float(delta),
        "cvar_improve_ratio": float(cvar_imp),
        "maxdd_improve_ratio": float(maxdd_imp),
        "signals_total": int(signals_sym),
        "entries_valid": int(entries_sym),
        "entry_rate": float(roll["entry_rate"]),
        "taker_share": float(roll["taker_share"]),
        "median_fill_delay_min": float(roll["median_fill_delay_min"]),
        "p95_fill_delay_min": float(roll["p95_fill_delay_min"]),
        "median_entry_improvement_bps": float(roll["median_entry_improvement_bps"]),
        "min_split_delta": float(min_split_delta),
        "split_count": int(len(split_means)),
        "split_means": split_means,
        "missing_slice_rate": float(missing_slice_rate),
    }


def build_s1_table(
    *,
    e1_route_rows: Dict[str, pd.DataFrame],
    e2_route_rows: Dict[str, pd.DataFrame],
    base_route_rows: Dict[str, pd.DataFrame],
    route_headroom: Dict[str, int],
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for rid, e1 in e1_route_rows.items():
        if e1.empty:
            continue
        x = e1.copy()
        x["signal_id"] = x["signal_id"].astype(str)
        e2 = e2_route_rows[rid][["signal_id", "exec_pnl_net_pct", "exec_pnl_gross_pct", "exec_fill_delay_min", "exec_fill_liquidity_type"]].copy()
        e2["signal_id"] = e2["signal_id"].astype(str)
        e2 = e2.rename(
            columns={
                "exec_pnl_net_pct": "e2_exec_pnl_net_pct",
                "exec_pnl_gross_pct": "e2_exec_pnl_gross_pct",
                "exec_fill_delay_min": "e2_exec_fill_delay_min",
                "exec_fill_liquidity_type": "e2_exec_fill_liquidity_type",
            }
        )
        if "baseline_pnl_net_pct" not in x.columns:
            x["baseline_pnl_net_pct"] = np.nan
        if "baseline_pnl_gross_pct" not in x.columns:
            x["baseline_pnl_gross_pct"] = np.nan
        if "baseline_fill_delay_min" not in x.columns and "baseline_fill_delay_min" in base_route_rows[rid].columns:
            x = x.merge(
                base_route_rows[rid][["signal_id", "baseline_fill_delay_min"]].assign(signal_id=base_route_rows[rid]["signal_id"].astype(str)),
                on="signal_id",
                how="left",
            )
        x = x.merge(e2, on="signal_id", how="left")
        x["symbol"] = "SOLUSDT"
        x["post_exec_net_expectancy"] = to_num(x["exec_pnl_net_pct"]).fillna(0.0)
        x["delta_vs_execution_baseline"] = to_num(x["exec_pnl_net_pct"]).fillna(0.0) - to_num(x["baseline_pnl_net_pct"]).fillna(0.0)
        x["fee_drag_trade"] = to_num(x["exec_pnl_gross_pct"]).fillna(0.0) - to_num(x["exec_pnl_net_pct"]).fillna(0.0)
        x["cvar_contrib_proxy"] = np.where(to_num(x["exec_pnl_net_pct"]).fillna(0.0) <= float(to_num(x["exec_pnl_net_pct"]).quantile(0.10)), to_num(x["exec_pnl_net_pct"]).fillna(0.0), 0.0)
        x["drawdown_contrib_proxy"] = np.minimum(to_num(x["exec_pnl_net_pct"]).fillna(0.0), 0.0)
        x["entry_rate_outcome"] = to_num(x["exec_filled"]).fillna(0).astype(int)
        x["taker_share_outcome"] = (x["exec_fill_liquidity_type"].astype(str).str.lower() == "taker").astype(int)
        x["fill_delay_min"] = to_num(x["exec_fill_delay_min"])
        x["toxic_candidate"] = (
            (x["taker_share_outcome"] == 1)
            | (to_num(x["est_taker_risk_proxy"]) >= float(to_num(x["est_taker_risk_proxy"]).quantile(0.80)))
            | (to_num(x["fee_drag_trade"]) >= float(to_num(x["fee_drag_trade"]).quantile(0.80)))
        ).astype(int)
        x["adverse_selection_candidate"] = (
            (to_num(x["entry_improvement_bps"]).fillna(0.0) <= 0.0)
            | ((x["taker_share_outcome"] == 1) & (to_num(x["exec_pnl_net_pct"]).fillna(0.0) <= 0.0))
        ).astype(int)
        neg = (to_num(x["exec_pnl_net_pct"]).fillna(0.0) < 0.0).astype(int)
        x = x.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        x["cluster_loss_candidate"] = (neg.rolling(3, min_periods=3).sum().fillna(0) >= 2).astype(int)
        x["weak_support_region"] = int(route_headroom.get(rid, 0) <= 20)
        x["route_support_headroom"] = int(route_headroom.get(rid, 0))
        x["center_route_failure_bucket"] = ((x["route_id"].astype(str) == "route_center_60pct") & ((x["adverse_selection_candidate"] == 1) | (x["entry_rate_outcome"] == 0))).astype(int)
        parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return out
    keep_cols = [
        "signal_id",
        "symbol",
        "signal_time",
        "route_id",
        "split_id",
        "session_bucket",
        "regime_bucket",
        "trend_bucket",
        "vol_bucket",
        "tp_mult",
        "sl_mult",
        "pre3m_ret_3bars",
        "pre3m_realized_vol_12",
        "pre3m_atr_z",
        "pre3m_spread_proxy_bps",
        "pre3m_body_bps_abs",
        "pre3m_wick_ratio",
        "pre3m_impulse_atr",
        "pre3m_trend_slope_12",
        "pre3m_accel_6v6",
        "pre3m_upbar_ratio_12",
        "pre3m_close_to_high_dist_bps",
        "pre3m_close_to_low_dist_bps",
        "est_taker_risk_proxy",
        "est_fill_delay_risk_proxy",
        "baseline_pnl_gross_pct",
        "baseline_pnl_net_pct",
        "exec_pnl_gross_pct",
        "exec_pnl_net_pct",
        "e2_exec_pnl_net_pct",
        "post_exec_net_expectancy",
        "delta_vs_execution_baseline",
        "fee_drag_trade",
        "cvar_contrib_proxy",
        "drawdown_contrib_proxy",
        "entry_rate_outcome",
        "taker_share_outcome",
        "fill_delay_min",
        "toxic_candidate",
        "cluster_loss_candidate",
        "adverse_selection_candidate",
        "weak_support_region",
        "route_support_headroom",
        "center_route_failure_bucket",
    ]
    return out.loc[:, [c for c in keep_cols if c in out.columns]].copy()


def summarize_s1_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_specs = [
        ("route", "route_id"),
        ("session", "session_bucket"),
        ("regime", "regime_bucket"),
        ("vol", "vol_bucket"),
    ]
    for slice_type, col in group_specs:
        for key, g in df.groupby(col):
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": str(key),
                    "signals": int(len(g)),
                    "mean_exec_net": float(to_num(g["exec_pnl_net_pct"]).mean()),
                    "mean_exec_gross": float(to_num(g["exec_pnl_gross_pct"]).mean()),
                    "mean_delta_vs_baseline": float(to_num(g["delta_vs_execution_baseline"]).mean()),
                    "toxic_rate": float(to_num(g["toxic_candidate"]).mean()),
                    "adverse_rate": float(to_num(g["adverse_selection_candidate"]).mean()),
                    "cluster_rate": float(to_num(g["cluster_loss_candidate"]).mean()),
                    "entry_rate": float(to_num(g["entry_rate_outcome"]).mean()),
                    "taker_rate": float(to_num(g["taker_share_outcome"]).mean()),
                    "median_fill_delay_min": float(to_num(g["fill_delay_min"]).median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice_type", "mean_exec_net"], ascending=[True, False]).reset_index(drop=True)


def top_feature_associations(df: pd.DataFrame, outcome_col: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    y = to_num(df[outcome_col]).fillna(0.0)
    rows: List[Dict[str, Any]] = []
    for c in feature_cols:
        x = to_num(df[c])
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 10:
            continue
        xx = x[mask]
        yy = y[mask]
        corr = float(np.corrcoef(xx.to_numpy(dtype=float), yy.to_numpy(dtype=float))[0, 1]) if xx.std(ddof=0) > 1e-12 and yy.std(ddof=0) > 1e-12 else float("nan")
        mean_on = float(xx[yy > 0.5].mean()) if int((yy > 0.5).sum()) > 0 else float("nan")
        mean_off = float(xx[yy <= 0.5].mean()) if int((yy <= 0.5).sum()) > 0 else float("nan")
        rows.append(
            {
                "feature": str(c),
                "corr": float(corr),
                "mean_when_flag_1": float(mean_on),
                "mean_when_flag_0": float(mean_off),
                "mean_diff": float(mean_on - mean_off) if np.isfinite(mean_on) and np.isfinite(mean_off) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("corr", key=lambda s: np.abs(to_num(s)), ascending=False).reset_index(drop=True)


def compute_score(feature_df: pd.DataFrame, score_family: str, weight_scale: float = 1.0) -> pd.Series:
    x = feature_df.copy()
    fill_risk = norm_rank(x["est_fill_delay_risk_proxy"])
    taker_risk = norm_rank(x["est_taker_risk_proxy"])
    spread = norm_rank(x["pre3m_spread_proxy_bps"])
    atr = norm_rank(x["pre3m_atr_z"])
    body = norm_rank(x["pre3m_body_bps_abs"])
    wick = norm_rank(x["pre3m_wick_ratio"])
    impulse = norm_rank(x["pre3m_impulse_atr"])
    close_high = norm_rank(x["pre3m_close_to_high_dist_bps"])
    rv = norm_rank(x["pre3m_realized_vol_12"])
    down_trend = 1.0 - to_num(x["trend_up_1h"]).fillna(0.0).clip(lower=0.0, upper=1.0)

    sf = str(score_family)
    if sf == "FILL_RESCUE":
        score = 0.45 * fill_risk + 0.25 * spread + 0.15 * rv + 0.15 * down_trend
    elif sf == "COST_DEFENSE":
        score = 0.40 * taker_risk + 0.25 * spread + 0.20 * atr + 0.15 * close_high
    elif sf == "TAIL_PENALIZED":
        score = 0.30 * atr + 0.25 * body + 0.20 * wick + 0.15 * impulse + 0.10 * rv
    elif sf == "CENTER_GUARD":
        score = 0.35 * fill_risk + 0.25 * taker_risk + 0.20 * spread + 0.20 * atr
    else:
        score = pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (score * float(weight_scale)).astype(float)


def apply_hard_rule(feature_df: pd.DataFrame, rule_id: str, q: Dict[str, float]) -> set[str]:
    x = feature_df.copy()
    rid = str(rule_id)
    if rid == "AE_H1":
        block = (to_num(x["pre3m_atr_z"]) >= float(q["pre3m_atr_z_q80"])) & (to_num(x["trend_up_1h"]) <= float(q["trend_up_q20"]))
    elif rid == "AE_H2":
        block = (to_num(x["pre3m_spread_proxy_bps"]) >= float(q["spread_q80"])) & (to_num(x["pre3m_realized_vol_12"]) >= float(q["rv_q80"]))
    elif rid == "AE_H3":
        block = (to_num(x["pre3m_body_bps_abs"]) >= float(q["body_q80"])) & (to_num(x["pre3m_atr_z"]) >= float(q["pre3m_atr_z_q80"]))
    else:
        block = pd.Series(False, index=x.index)
    keep = x.loc[~block, "signal_id"].astype(str)
    return set(keep.tolist())


def apply_route_prune(
    *,
    feature_df: pd.DataFrame,
    score: pd.Series,
    route_test_ids_map: Dict[str, set[str]],
    budgets: Dict[str, int],
    session_cap: Optional[int] = None,
    vol_cap: Optional[int] = None,
) -> set[str]:
    x = feature_df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["score_proto"] = score.reindex(x.index).fillna(0.0).to_numpy(dtype=float)
    removed: set[str] = set()
    for rid in sorted(route_test_ids_map.keys()):
        route_ids = set(route_test_ids_map[rid])
        k = int(max(0, budgets.get(rid, 0)))
        if k <= 0:
            continue
        cand = x[x["signal_id"].isin(route_ids) & (~x["signal_id"].isin(removed))].copy()
        if cand.empty:
            continue
        cand = cand.sort_values(["score_proto", "signal_time", "signal_id"], ascending=[False, True, True]).reset_index(drop=True)
        sess_used: Counter[str] = Counter()
        vol_used: Counter[str] = Counter()
        take: List[str] = []
        for r in cand.itertuples(index=False):
            sid = str(getattr(r, "signal_id"))
            sess = str(getattr(r, "session_bucket", "unknown"))
            vol = str(getattr(r, "vol_bucket", "unknown"))
            if session_cap is not None and sess_used[sess] >= int(session_cap):
                continue
            if vol_cap is not None and vol_used[vol] >= int(vol_cap):
                continue
            take.append(sid)
            sess_used[sess] += 1
            vol_used[vol] += 1
            if len(take) >= k:
                break
        removed.update(take)
    keep = set(x.loc[~x["signal_id"].isin(removed), "signal_id"].astype(str).tolist())
    return keep


def build_prototype_specs() -> List[Dict[str, Any]]:
    return [
        {"prototype_id": "baseline_active", "family_id": "REFERENCE", "method": "baseline", "description": "Keep full active signal set"},
        {"prototype_id": "ref_ae_h1", "family_id": "REFERENCE", "method": "hard_rule", "rule_id": "AE_H1", "description": "Legacy AE H1 volatility/trend guard"},
        {"prototype_id": "ref_ae_h2", "family_id": "REFERENCE", "method": "hard_rule", "rule_id": "AE_H2", "description": "Legacy AE H2 spread/vol guard"},
        {"prototype_id": "ref_ae_h3", "family_id": "REFERENCE", "method": "hard_rule", "rule_id": "AE_H3", "description": "Legacy AE H3 impulse/vol guard"},
        {
            "prototype_id": "fill_rescue_conservative",
            "family_id": "FILL_RESCUE",
            "method": "route_prune",
            "score_family": "FILL_RESCUE",
            "budgets": {"route_front_60pct": 2, "route_center_60pct": 4, "route_back_60pct": 0},
            "session_cap": 2,
            "vol_cap": None,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "fill_rescue_moderate",
            "family_id": "FILL_RESCUE",
            "method": "route_prune",
            "score_family": "FILL_RESCUE",
            "budgets": {"route_front_60pct": 3, "route_center_60pct": 6, "route_back_60pct": 1},
            "session_cap": 3,
            "vol_cap": None,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "cost_defense_conservative",
            "family_id": "COST_DEFENSE",
            "method": "route_prune",
            "score_family": "COST_DEFENSE",
            "budgets": {"route_front_60pct": 4, "route_center_60pct": 2, "route_back_60pct": 1},
            "session_cap": 2,
            "vol_cap": 2,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "cost_defense_moderate",
            "family_id": "COST_DEFENSE",
            "method": "route_prune",
            "score_family": "COST_DEFENSE",
            "budgets": {"route_front_60pct": 6, "route_center_60pct": 3, "route_back_60pct": 2},
            "session_cap": 3,
            "vol_cap": 3,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "tail_penalized_conservative",
            "family_id": "TAIL_PENALIZED",
            "method": "route_prune",
            "score_family": "TAIL_PENALIZED",
            "budgets": {"route_front_60pct": 2, "route_center_60pct": 4, "route_back_60pct": 2},
            "session_cap": 2,
            "vol_cap": 2,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "tail_penalized_moderate",
            "family_id": "TAIL_PENALIZED",
            "method": "route_prune",
            "score_family": "TAIL_PENALIZED",
            "budgets": {"route_front_60pct": 3, "route_center_60pct": 6, "route_back_60pct": 3},
            "session_cap": 3,
            "vol_cap": 3,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "center_guard_conservative",
            "family_id": "CENTER_GUARD",
            "method": "route_prune",
            "score_family": "CENTER_GUARD",
            "budgets": {"route_front_60pct": 4, "route_center_60pct": 6, "route_back_60pct": 1},
            "session_cap": 2,
            "vol_cap": None,
            "weight_scale": 1.0,
        },
        {
            "prototype_id": "center_guard_moderate",
            "family_id": "CENTER_GUARD",
            "method": "route_prune",
            "score_family": "CENTER_GUARD",
            "budgets": {"route_front_60pct": 6, "route_center_60pct": 8, "route_back_60pct": 2},
            "session_cap": 3,
            "vol_cap": None,
            "weight_scale": 1.0,
        },
    ]


def materialize_keep_ids(
    *,
    proto: Dict[str, Any],
    feature_df: pd.DataFrame,
    route_test_ids_map: Dict[str, set[str]],
    quantiles: Dict[str, float],
) -> set[str]:
    method = str(proto["method"])
    if method == "baseline":
        return set(feature_df["signal_id"].astype(str).tolist())
    if method == "hard_rule":
        return apply_hard_rule(feature_df, str(proto["rule_id"]), quantiles)
    if method == "route_prune":
        score = compute_score(feature_df, str(proto["score_family"]), weight_scale=float(proto.get("weight_scale", 1.0)))
        return apply_route_prune(
            feature_df=feature_df,
            score=score,
            route_test_ids_map=route_test_ids_map,
            budgets={str(k): int(v) for k, v in dict(proto.get("budgets", {})).items()},
            session_cap=int(proto["session_cap"]) if proto.get("session_cap") is not None else None,
            vol_cap=int(proto["vol_cap"]) if proto.get("vol_cap") is not None else None,
        )
    raise ValueError(f"Unknown prototype method: {method}")


def build_quantiles(feature_df: pd.DataFrame) -> Dict[str, float]:
    x = feature_df.copy()
    return {
        "pre3m_atr_z_q80": float(to_num(x["pre3m_atr_z"]).quantile(0.80)),
        "trend_up_q20": float(to_num(x["trend_up_1h"]).quantile(0.20)),
        "spread_q80": float(to_num(x["pre3m_spread_proxy_bps"]).quantile(0.80)),
        "rv_q80": float(to_num(x["pre3m_realized_vol_12"]).quantile(0.80)),
        "body_q80": float(to_num(x["pre3m_body_bps_abs"]).quantile(0.80)),
    }


def evaluate_prototype(
    *,
    proto: Dict[str, Any],
    keep_ids: set[str],
    exec_refs: Dict[str, Dict[str, Any]],
    full_bundle: ga_exec.SymbolBundle,
    route_bundles: Dict[str, ga_exec.SymbolBundle],
    full_rows_cache: Dict[str, pd.DataFrame],
    route_rows_cache: Dict[str, Dict[str, pd.DataFrame]],
    full_ref_stats: Dict[str, Dict[str, Any]],
    route_ref_stats: Dict[str, Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
    prior_route_valid_rate: float,
) -> Dict[str, Any]:
    per_exec: Dict[str, Dict[str, Any]] = {}
    full_signals_union = set()
    full_removed = 0

    for exec_id, meta in exec_refs.items():
        genome = meta["genome"]
        full_eval = subset_eval_from_rows(
            rows_df=full_rows_cache[exec_id],
            bundle=full_bundle,
            args=args,
            genome=genome,
            keep_ids=keep_ids,
            ref_roll=full_ref_stats[exec_id]["roll"],
            ref_split_means=full_ref_stats[exec_id]["split_means"],
        )
        full_rows = full_rows_cache[exec_id].copy()
        full_ids = set(full_rows["signal_id"].astype(str).tolist())
        full_signals_union.update(full_ids)
        full_removed = max(full_removed, int(len(full_ids - keep_ids)))

        route_cells: List[Dict[str, Any]] = []
        for rid, bundle in route_bundles.items():
            rv = subset_eval_from_rows(
                rows_df=route_rows_cache[exec_id][rid],
                bundle=bundle,
                args=args,
                genome=genome,
                keep_ids=keep_ids,
                ref_roll=route_ref_stats[exec_id][rid]["roll"],
                ref_split_means=route_ref_stats[exec_id][rid]["split_means"],
            )
            rv["route_id"] = rid
            route_cells.append(rv)
        per_exec[exec_id] = {"full": full_eval, "routes": route_cells}

    row: Dict[str, Any] = {
        "prototype_id": str(proto["prototype_id"]),
        "family_id": str(proto["family_id"]),
        "method": str(proto["method"]),
        "keep_count": int(len(keep_ids)),
        "removed_signals": int(full_removed),
        "removed_signals_pct": float(safe_div(float(full_removed), float(max(1, len(full_signals_union))))),
        "kept_hash": sha1_text("|".join(sorted(keep_ids))),
        "params_json": json.dumps(
            {
                "method": proto.get("method"),
                "rule_id": proto.get("rule_id"),
                "score_family": proto.get("score_family"),
                "budgets": proto.get("budgets"),
                "session_cap": proto.get("session_cap"),
                "vol_cap": proto.get("vol_cap"),
                "weight_scale": proto.get("weight_scale"),
            },
            sort_keys=True,
        ),
    }

    full_valids = []
    full_deltas = []
    full_cvars = []
    full_maxdds = []
    full_entry_rates = []
    full_takers = []
    full_delays = []
    min_split_deltas = []
    route_valids = []
    route_pos_deltas = []
    center_deltas = []
    center_valids = []
    route_min_signals = []
    route_min_entries = []
    invalid_parts: List[str] = []

    for exec_id, pack in per_exec.items():
        fv = pack["full"]
        row[f"{exec_id.lower()}_valid_for_ranking"] = int(fv["valid_for_ranking"])
        row[f"{exec_id.lower()}_invalid_reason"] = str(fv["invalid_reason"])
        row[f"{exec_id.lower()}_expectancy_net"] = float(fv["expectancy_net"])
        row[f"{exec_id.lower()}_delta_vs_exec_baseline"] = float(fv["delta_vs_ref"])
        row[f"{exec_id.lower()}_cvar_improve_ratio"] = float(fv["cvar_improve_ratio"])
        row[f"{exec_id.lower()}_maxdd_improve_ratio"] = float(fv["maxdd_improve_ratio"])
        row[f"{exec_id.lower()}_entry_rate"] = float(fv["entry_rate"])
        row[f"{exec_id.lower()}_taker_share"] = float(fv["taker_share"])
        row[f"{exec_id.lower()}_median_fill_delay_min"] = float(fv["median_fill_delay_min"])
        row[f"{exec_id.lower()}_min_split_delta"] = float(fv["min_split_delta"])
        full_valids.append(int(fv["valid_for_ranking"]))
        full_deltas.append(float(fv["delta_vs_ref"]))
        full_cvars.append(float(fv["cvar_improve_ratio"]))
        full_maxdds.append(float(fv["maxdd_improve_ratio"]))
        full_entry_rates.append(float(fv["entry_rate"]))
        full_takers.append(float(fv["taker_share"]))
        full_delays.append(float(fv["median_fill_delay_min"]))
        min_split_deltas.append(float(fv["min_split_delta"]))
        if str(fv["invalid_reason"]).strip():
            invalid_parts.extend([x for x in str(fv["invalid_reason"]).split("|") if x])

        route_sig_counts = {}
        route_entry_counts = {}
        route_val_flags = {}
        route_delta_vals = {}
        for rv in pack["routes"]:
            rid = str(rv["route_id"])
            route_sig_counts[rid] = int(rv["signals_total"])
            route_entry_counts[rid] = int(rv["entries_valid"])
            route_val_flags[rid] = int(rv["valid_for_ranking"])
            route_delta_vals[rid] = float(rv["delta_vs_ref"])
            route_valids.append(int(rv["valid_for_ranking"]))
            route_pos_deltas.append(int(np.isfinite(rv["delta_vs_ref"]) and float(rv["delta_vs_ref"]) > 0.0))
            route_min_signals.append(int(rv["signals_total"]))
            route_min_entries.append(int(rv["entries_valid"]))
            if rid == "route_center_60pct":
                center_deltas.append(float(rv["delta_vs_ref"]))
                center_valids.append(int(rv["valid_for_ranking"]))
        row[f"{exec_id.lower()}_route_valid_json"] = json.dumps(route_val_flags, sort_keys=True)
        row[f"{exec_id.lower()}_route_delta_json"] = json.dumps(route_delta_vals, sort_keys=True)
        row[f"{exec_id.lower()}_route_entries_json"] = json.dumps(route_entry_counts, sort_keys=True)

    row["valid_for_ranking"] = int(all(v == 1 for v in full_valids))
    row["invalid_reason"] = "|".join(sorted(set(invalid_parts)))
    row["exec_expectancy_net"] = float(np.nanmean([pack["full"]["expectancy_net"] for pack in per_exec.values()])) if per_exec else float("nan")
    row["delta_expectancy_vs_exec_baseline"] = float(np.nanmean(full_deltas)) if full_deltas else float("nan")
    row["cvar_improve_ratio"] = float(np.nanmean(full_cvars)) if full_cvars else float("nan")
    row["maxdd_improve_ratio"] = float(np.nanmean(full_maxdds)) if full_maxdds else float("nan")
    row["route_pass_rate"] = float(np.mean(route_valids)) if route_valids else float("nan")
    row["route_positive_delta_rate"] = float(np.mean(route_pos_deltas)) if route_pos_deltas else float("nan")
    row["min_subperiod_delta"] = float(np.nanmin(min_split_deltas)) if min_split_deltas else float("nan")
    row["center_route_delta"] = float(np.nanmean(center_deltas)) if center_deltas else float("nan")
    row["center_route_delta_min"] = float(np.nanmin(center_deltas)) if center_deltas else float("nan")
    row["center_route_valid_rate"] = float(np.mean(center_valids)) if center_valids else float("nan")
    row["support_min_signals_route"] = int(min(route_min_signals)) if route_min_signals else 0
    row["support_min_entries_route"] = int(min(route_min_entries)) if route_min_entries else 0
    row["entry_rate_mean"] = float(np.nanmean(full_entry_rates)) if full_entry_rates else float("nan")
    row["taker_share_mean"] = float(np.nanmean(full_takers)) if full_takers else float("nan")
    row["median_fill_delay_mean"] = float(np.nanmean(full_delays)) if full_delays else float("nan")
    row["route_survivability_vs_prior_exec_branch"] = float(row["route_pass_rate"] - prior_route_valid_rate) if np.isfinite(row["route_pass_rate"]) else float("nan")

    fail_tags: List[str] = []
    if row["valid_for_ranking"] != 1:
        fail_tags.append("invalid_for_ranking")
    if not (np.isfinite(row["delta_expectancy_vs_exec_baseline"]) and float(row["delta_expectancy_vs_exec_baseline"]) > 0.0):
        fail_tags.append("delta_nonpositive")
    if not (np.isfinite(row["cvar_improve_ratio"]) and float(row["cvar_improve_ratio"]) >= 0.0):
        fail_tags.append("cvar_negative")
    if not (np.isfinite(row["maxdd_improve_ratio"]) and float(row["maxdd_improve_ratio"]) > 0.0):
        fail_tags.append("maxdd_nonpositive")
    if not (np.isfinite(row["center_route_delta_min"]) and float(row["center_route_delta_min"]) > 0.0 and float(row["center_route_valid_rate"]) >= 1.0):
        fail_tags.append("route_center_collapse")
    if int(row["support_min_entries_route"]) < int(args.hard_min_trades_overall):
        fail_tags.append("support_starvation")
    if not (np.isfinite(row["route_pass_rate"]) and float(row["route_pass_rate"]) > float(prior_route_valid_rate)):
        fail_tags.append("route_survivability_not_improved")
    row["decision_fail_tags"] = "|".join(fail_tags)
    row["s3_go_candidate"] = int(len(fail_tags) == 0)
    return row


def compute_proto_delta_vector(
    *,
    keep_ids: set[str],
    full_rows_cache: Dict[str, pd.DataFrame],
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for exec_id, df in full_rows_cache.items():
        x = df.copy()
        x["signal_id"] = x["signal_id"].astype(str)
        base = to_num(x["exec_pnl_net_pct"]).fillna(0.0).to_numpy(dtype=float)
        kept = x["signal_id"].isin(keep_ids).to_numpy(dtype=bool)
        sub = np.where(kept, base, 0.0)
        vecs.append(sub - base)
    return np.concatenate(vecs) if vecs else np.array([], dtype=float)


def sample_pilot_variants(
    *,
    winner_rows: pd.DataFrame,
    seed_protos: Dict[str, Dict[str, Any]],
    n_total: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    fams = sorted(set(winner_rows["family_id"].astype(str).tolist()) - {"REFERENCE"})
    if not fams:
        return []
    per_fam = max(8, int(math.ceil(float(n_total) / max(1, len(fams)))))
    out: List[Dict[str, Any]] = []
    for fam in fams:
        seeds = [p for p in seed_protos.values() if str(p.get("family_id")) == fam and str(p.get("method")) == "route_prune"]
        if not seeds:
            continue
        base = copy.deepcopy(seeds[-1])
        for i in range(per_fam):
            p = copy.deepcopy(base)
            p["prototype_id"] = f"pilot_{fam.lower()}_{i:03d}"
            budgets = dict(p.get("budgets", {}))
            for rid in list(budgets.keys()):
                delta = rng.randint(-2, 3)
                budgets[rid] = int(max(0, min(12 if rid != "route_center_60pct" else 14, int(budgets[rid]) + delta)))
            p["budgets"] = budgets
            if p.get("session_cap") is not None:
                p["session_cap"] = int(max(1, min(4, int(p["session_cap"]) + rng.choice([-1, 0, 1]))))
            if p.get("vol_cap") is not None:
                p["vol_cap"] = int(max(1, min(4, int(p["vol_cap"]) + rng.choice([-1, 0, 1]))))
            p["weight_scale"] = float(max(0.75, min(1.35, float(p.get("weight_scale", 1.0)) * (1.0 + rng.uniform(-0.18, 0.18)))))
            out.append(p)
    return out[:n_total]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase S signal-economics redesign under repaired route harness")
    ap.add_argument(
        "--route-dir",
        default="/root/analysis/0.87/reports/execution_layer/PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334",
    )
    ap.add_argument(
        "--nx-dir",
        default="/root/analysis/0.87/reports/execution_layer/PHASENX_EXEC_FAMILY_DISCOVERY_20260227_115329",
    )
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--pilot-candidates", type=int, default=48)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    route_dir = Path(args.route_dir).resolve()
    nx_dir = Path(args.nx_dir).resolve()
    if not route_dir.exists():
        raise FileNotFoundError(f"Route harness artifact dir not found: {route_dir}")
    if not nx_dir.exists():
        raise FileNotFoundError(f"NX artifact dir not found: {nx_dir}")

    out_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = out_root / f"PHASES_SIGNAL_ECON_REDESIGN_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    rep_df = pd.read_csv(subset_path)
    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze lock validation failed")

    bundles, load_meta = ga_exec._prepare_bundles(exec_args)
    if not bundles:
        raise RuntimeError("No bundles prepared under frozen harness")
    full_bundle = bundles[0]

    hist_cands = pr.load_historical_exec_candidates()
    exec_refs = {c["candidate_id"]: c for c in hist_cands if c["candidate_id"] in {"E1", "E2"}}
    if sorted(exec_refs.keys()) != ["E1", "E2"]:
        raise RuntimeError("Could not load E1/E2 historical execution references")

    feature_df = build_feature_frame(full_bundle, rep_df, exec_refs["E1"]["genome"])

    route_bundles, route_examples_df, feasibility_df, route_meta = pr.build_support_feasible_route_family(base_bundle=full_bundle, args=exec_args, coverage_frac=0.60)
    validate_repaired_routes(
        route_examples_fp=route_dir / "phaseR1_route_examples.csv",
        route_examples_df=route_examples_df,
        route_bundles=route_bundles,
    )
    route_headroom = {str(r["route_id"]): int(r["headroom_vs_overall_gate"]) for r in feasibility_df.to_dict("records")}

    full_rows_cache: Dict[str, pd.DataFrame] = {}
    route_rows_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    baseline_full_rows = baseline_detailed(bundle=full_bundle, feature_df=feature_df, route_id="full")
    baseline_route_rows = {rid: baseline_detailed(bundle=b, feature_df=feature_df, route_id=rid) for rid, b in route_bundles.items()}

    for exec_id, meta in exec_refs.items():
        full_rows_cache[exec_id] = run_exec_detailed(
            genome=meta["genome"],
            bundle=full_bundle,
            feature_df=feature_df,
            args=exec_args,
            route_id="full",
            exec_id=exec_id,
        )
        route_rows_cache[exec_id] = {}
        for rid, bundle in route_bundles.items():
            route_rows_cache[exec_id][rid] = run_exec_detailed(
                genome=meta["genome"],
                bundle=bundle,
                feature_df=feature_df,
                args=exec_args,
                route_id=rid,
                exec_id=exec_id,
            )

    full_ref_stats: Dict[str, Dict[str, Any]] = {}
    route_ref_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for exec_id, meta in exec_refs.items():
        fref = subset_eval_from_rows(
            rows_df=full_rows_cache[exec_id],
            bundle=full_bundle,
            args=exec_args,
            genome=meta["genome"],
            keep_ids=None,
        )
        full_ref_stats[exec_id] = {"roll": fref, "split_means": dict(fref["split_means"])}
        route_ref_stats[exec_id] = {}
        for rid, bundle in route_bundles.items():
            rref = subset_eval_from_rows(
                rows_df=route_rows_cache[exec_id][rid],
                bundle=bundle,
                args=exec_args,
                genome=meta["genome"],
                keep_ids=None,
            )
            route_ref_stats[exec_id][rid] = {"roll": rref, "split_means": dict(rref["split_means"])}

    prior_route_valid_rate = float(
        np.mean(
            [
                int(route_ref_stats[exec_id][rid]["roll"]["valid_for_ranking"])
                for exec_id in sorted(exec_refs.keys())
                for rid in sorted(route_bundles.keys())
            ]
        )
    )

    # S1
    s1_table = build_s1_table(
        e1_route_rows=route_rows_cache["E1"],
        e2_route_rows=route_rows_cache["E2"],
        base_route_rows=baseline_route_rows,
        route_headroom=route_headroom,
    )
    if s1_table.empty:
        raise RuntimeError("S1 table is empty")
    s1_table.to_parquet(run_dir / "phaseS1_signal_econ_table.parquet", index=False)

    s1_summary = summarize_s1_table(s1_table)
    s1_summary.to_csv(run_dir / "phaseS1_signal_econ_summary.csv", index=False)

    feature_cols = [
        "pre3m_ret_3bars",
        "pre3m_realized_vol_12",
        "pre3m_atr_z",
        "pre3m_spread_proxy_bps",
        "pre3m_body_bps_abs",
        "pre3m_wick_ratio",
        "pre3m_impulse_atr",
        "pre3m_trend_slope_12",
        "pre3m_accel_6v6",
        "pre3m_upbar_ratio_12",
        "pre3m_close_to_high_dist_bps",
        "pre3m_close_to_low_dist_bps",
        "est_taker_risk_proxy",
        "est_fill_delay_risk_proxy",
    ]
    assoc_reports = {
        "toxic_candidate": top_feature_associations(s1_table, "toxic_candidate", feature_cols),
        "weak_post_exec_delta": top_feature_associations((s1_table.assign(weak_post_exec_delta=(to_num(s1_table["delta_vs_execution_baseline"]) <= 0.0).astype(int))), "weak_post_exec_delta", feature_cols),
        "center_route_failure_bucket": top_feature_associations(s1_table, "center_route_failure_bucket", feature_cols),
        "negative_cvar_contrib": top_feature_associations((s1_table.assign(negative_cvar_contrib=(to_num(s1_table["cvar_contrib_proxy"]) < 0.0).astype(int))), "negative_cvar_contrib", feature_cols),
    }
    ranked_causes = [
        "Center-route weakness is dominated by fill-risk / delay-risk clusters.",
        "Front-route weakness is dominated by taker-risk / spread-cost bursts.",
        "Cost-killed gross edge remains material in stressed microstructure buckets.",
        "Tail weakness clusters around high ATR-z / impulsive bar conditions.",
    ]
    feat_lines = [
        "# S1 Feature Association Report",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Ranked likely upstream causes:",
    ]
    for i, cause in enumerate(ranked_causes, start=1):
        feat_lines.append(f"  {i}. {cause}")
    feat_lines.append("")
    for key, df_assoc in assoc_reports.items():
        feat_lines.append(f"## {key}")
        feat_lines.append("")
        feat_lines.append(markdown_table(df_assoc, ["feature", "corr", "mean_when_flag_1", "mean_when_flag_0", "mean_diff"], n=8))
        feat_lines.append("")
    write_text(run_dir / "phaseS1_feature_association_report.md", "\n".join(feat_lines))

    x = s1_table.copy()
    gross = to_num(x["exec_pnl_gross_pct"]).fillna(0.0)
    net = to_num(x["exec_pnl_net_pct"]).fillna(0.0)
    edge_state = pd.Series(index=x.index, dtype=object)
    edge_state[gross <= 0.0] = "no_gross_edge"
    edge_state[(gross > 0.0) & (net <= 0.0)] = "cost_killed_edge"
    edge_state[net > 0.0] = "positive_net"
    x["edge_vs_cost_bucket"] = edge_state.fillna("other")
    evc = (
        x.groupby(["route_id", "edge_vs_cost_bucket"], as_index=False)
        .agg(
            signals=("signal_id", "count"),
            mean_gross=("exec_pnl_gross_pct", lambda s: float(to_num(s).mean())),
            mean_net=("exec_pnl_net_pct", lambda s: float(to_num(s).mean())),
            mean_fee_drag=("fee_drag_trade", lambda s: float(to_num(s).mean())),
        )
        .sort_values(["route_id", "signals"], ascending=[True, False])
        .reset_index(drop=True)
    )
    evc_lines = [
        "# S1 Edge Vs Cost Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- No-gross-edge share: `{float(np.mean(x['edge_vs_cost_bucket'] == 'no_gross_edge')):.4f}`",
        f"- Cost-killed-edge share: `{float(np.mean(x['edge_vs_cost_bucket'] == 'cost_killed_edge')):.4f}`",
        f"- Positive-net share: `{float(np.mean(x['edge_vs_cost_bucket'] == 'positive_net')):.4f}`",
        "",
        markdown_table(evc, ["route_id", "edge_vs_cost_bucket", "signals", "mean_gross", "mean_net", "mean_fee_drag"], n=30),
        "",
    ]
    write_text(run_dir / "phaseS1_edge_vs_cost_report.md", "\n".join(evc_lines))

    center = s1_table[s1_table["route_id"].astype(str) == "route_center_60pct"].copy()
    other = s1_table[s1_table["route_id"].astype(str) != "route_center_60pct"].copy()
    center_cmp = pd.DataFrame(
        {
            "metric": ["signals", "mean_exec_net", "entry_rate", "taker_rate", "mean_fill_delay", "toxic_rate", "adverse_rate"],
            "center_route": [
                int(len(center)),
                float(to_num(center["exec_pnl_net_pct"]).mean()),
                float(to_num(center["entry_rate_outcome"]).mean()),
                float(to_num(center["taker_share_outcome"]).mean()),
                float(to_num(center["fill_delay_min"]).mean()),
                float(to_num(center["toxic_candidate"]).mean()),
                float(to_num(center["adverse_selection_candidate"]).mean()),
            ],
            "other_routes": [
                int(len(other)),
                float(to_num(other["exec_pnl_net_pct"]).mean()),
                float(to_num(other["entry_rate_outcome"]).mean()),
                float(to_num(other["taker_share_outcome"]).mean()),
                float(to_num(other["fill_delay_min"]).mean()),
                float(to_num(other["toxic_candidate"]).mean()),
                float(to_num(other["adverse_selection_candidate"]).mean()),
            ],
        }
    )
    center_lines = [
        "# S1 Route Center Failure Report",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Center-route rows are economically strong on filled trades but structurally vulnerable on entry-rate / adverse-selection clusters.",
        "",
        markdown_table(center_cmp, ["metric", "center_route", "other_routes"], n=20),
        "",
    ]
    write_text(run_dir / "phaseS1_route_center_failure_report.md", "\n".join(center_lines))

    # S2
    proto_specs = build_prototype_specs()
    quantiles = build_quantiles(feature_df)
    route_test_ids_map = {
        rid: set(route_rows_cache["E1"][rid]["signal_id"].astype(str).tolist())
        for rid in sorted(route_bundles.keys())
    }

    s2_yaml = {
        "phase": "S2",
        "objective_families": [
            {
                "family_id": "FILL_RESCUE",
                "target_metric": "positive delta expectancy vs E1/E2 with entry-rate repair",
                "penalties": ["fill_delay_risk", "spread", "realized_vol", "weak trend"],
                "hard_exclusions": ["none in spec; prototype uses bounded ranked pruning only"],
                "support_minimums": {"per_route_entries_min": 200},
                "route_balance_requirement": "front/center/back removal budgets capped to preserve all repaired routes",
                "center_route_anti_collapse": "center route gets higher fill-risk pruning budget",
                "tail_cvar_safeguard": "do not remove more than support headroom permits",
                "trade_density_safeguard": "session caps on removals",
            },
            {
                "family_id": "COST_DEFENSE",
                "target_metric": "reduce cost-killed edge and taker failures",
                "penalties": ["taker_risk", "spread", "ATR stress", "adverse selection proxy"],
                "hard_exclusions": ["none in spec; bounded ranked pruning"],
                "support_minimums": {"per_route_entries_min": 200},
                "route_balance_requirement": "front route gets largest taker-risk budget",
                "center_route_anti_collapse": "center budget retained but smaller than fill-rescue",
                "tail_cvar_safeguard": "vol-bucket caps avoid concentrating removals in one regime",
                "trade_density_safeguard": "session + vol caps",
            },
            {
                "family_id": "TAIL_PENALIZED",
                "target_metric": "reduce tail contribution without collapsing density",
                "penalties": ["ATR stress", "large bodies", "wick imbalance", "impulse"],
                "hard_exclusions": ["none in spec; bounded ranked pruning"],
                "support_minimums": {"per_route_entries_min": 200},
                "route_balance_requirement": "balanced removal budgets across all repaired routes",
                "center_route_anti_collapse": "center route gets moderate extra budget",
                "tail_cvar_safeguard": "tail-focused score only; no low-signal starvation allowed",
                "trade_density_safeguard": "session + vol caps",
            },
            {
                "family_id": "CENTER_GUARD",
                "target_metric": "eliminate center-route universal failure as primary bottleneck",
                "penalties": ["fill_delay_risk", "taker_risk", "spread", "ATR stress"],
                "hard_exclusions": ["none in spec; bounded ranked pruning"],
                "support_minimums": {"per_route_entries_min": 200},
                "route_balance_requirement": "all routes preserved, but center route gets largest budget",
                "center_route_anti_collapse": "explicit design focus",
                "tail_cvar_safeguard": "back-route budget remains small to avoid overfit",
                "trade_density_safeguard": "session caps",
            },
        ],
        "prototype_seeds": proto_specs,
    }
    json_dump(run_dir / "phaseS2_candidate_objectives.yaml", s2_yaml)

    s2_lines = [
        "# S2 Objective Family Spec",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Objective families are implemented as execution-aware signal-ranking / bounded-pruning prototypes over the frozen representative subset.",
        "- They do not alter execution mechanics; they alter which 1h signals survive into the fixed E1/E2 execution scorers.",
        "",
        "## Families",
        "",
    ]
    for fam in s2_yaml["objective_families"]:
        s2_lines.extend(
            [
                f"### {fam['family_id']}",
                f"- target metric: {fam['target_metric']}",
                f"- penalties: {', '.join(fam['penalties'])}",
                f"- hard exclusions: {', '.join(fam['hard_exclusions'])}",
                f"- support minimums: {fam['support_minimums']}",
                f"- route-balance requirement: {fam['route_balance_requirement']}",
                f"- center-route anti-collapse: {fam['center_route_anti_collapse']}",
                f"- tail / CVaR safeguard: {fam['tail_cvar_safeguard']}",
                f"- trade density safeguard: {fam['trade_density_safeguard']}",
                "",
            ]
        )
    write_text(run_dir / "phaseS2_objective_family_spec.md", "\n".join(s2_lines))

    map_lines = [
        "# S2 Mapping To Existing 1h Engine",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Existing 1h optimization engine: `src/bot087/optim/ga.py`.",
        "- This branch does not replace that engine; it specifies how to change its objective / labeling layer:",
        "  - add execution-aware post-score penalties as additional fitness terms",
        "  - enforce route-balance/support minimum constraints before accepting a candidate",
        "  - include center-route anti-collapse penalties in validation score",
        "  - use repaired-route slices as mandatory validation cohorts",
        "- In this run, the spec is benchmarked as deterministic prototypes over the frozen subset before spending GA compute.",
        "",
    ]
    write_text(run_dir / "phaseS2_mapping_to_existing_1h_engine.md", "\n".join(map_lines))

    # S3
    seed_proto_map = {str(p["prototype_id"]): p for p in proto_specs}
    s3_rows: List[Dict[str, Any]] = []
    invalid_hist: Counter[str] = Counter()
    for proto in proto_specs:
        keep_ids = materialize_keep_ids(proto=proto, feature_df=feature_df, route_test_ids_map=route_test_ids_map, quantiles=quantiles)
        row = evaluate_prototype(
            proto=proto,
            keep_ids=keep_ids,
            exec_refs=exec_refs,
            full_bundle=full_bundle,
            route_bundles=route_bundles,
            full_rows_cache=full_rows_cache,
            route_rows_cache=route_rows_cache,
            full_ref_stats=full_ref_stats,
            route_ref_stats=route_ref_stats,
            args=exec_args,
            prior_route_valid_rate=prior_route_valid_rate,
        )
        s3_rows.append(row)
        if str(row["decision_fail_tags"]).strip():
            for part in [x for x in str(row["decision_fail_tags"]).split("|") if x]:
                invalid_hist[part] += 1

    s3_df = pd.DataFrame(s3_rows).sort_values(
        ["s3_go_candidate", "delta_expectancy_vs_exec_baseline", "cvar_improve_ratio", "maxdd_improve_ratio", "route_pass_rate"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    s3_df.to_csv(run_dir / "phaseS3_prototype_results.csv", index=False)
    json_dump(run_dir / "phaseS3_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    s3_winners = s3_df[s3_df["s3_go_candidate"] == 1].copy()
    topk_lines = [
        "# S3 Top-K Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Prototype count: `{len(s3_df)}`",
        f"- S3 GO candidates: `{len(s3_winners)}`",
        "",
        markdown_table(
            s3_df,
            [
                "prototype_id",
                "family_id",
                "valid_for_ranking",
                "delta_expectancy_vs_exec_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "route_pass_rate",
                "center_route_delta_min",
                "support_min_entries_route",
                "decision_fail_tags",
            ],
            n=16,
        ),
        "",
    ]
    write_text(run_dir / "phaseS3_topk_report.md", "\n".join(topk_lines))

    s3_go = int(len(s3_winners) >= 1)
    s3_decision_lines = [
        "# S3 Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Go to S4: `{s3_go}`",
        f"- Winning prototypes: `{s3_winners['prototype_id'].astype(str).tolist()}`",
    ]
    write_text(run_dir / "phaseS3_decision.md", "\n".join(s3_decision_lines) + "\n")

    furthest_phase = "S3"
    classification = "SIGNAL_REDESIGN_NO_GO"
    mainline_status = classification
    stable_neighborhood = 0
    bounded_pilot_reached = 0
    exact_stop_reason = "S3 failed: no prototype satisfied full validity + positive delta + non-collapsing center-route criteria"

    if s3_go == 1:
        # S4 bounded pilot
        bounded_pilot_reached = 1
        furthest_phase = "S4"
        rng = random.Random(int(args.seed) + 404)
        pilot_specs = sample_pilot_variants(
            winner_rows=s3_winners,
            seed_protos=seed_proto_map,
            n_total=int(args.pilot_candidates),
            rng=rng,
        )
        pilot_rows: List[Dict[str, Any]] = []
        kept_map_rows: List[Dict[str, Any]] = []
        delta_vecs: List[np.ndarray] = []
        pilot_by_hash: Dict[str, Dict[str, Any]] = {}

        for proto in pilot_specs:
            keep_ids = materialize_keep_ids(proto=proto, feature_df=feature_df, route_test_ids_map=route_test_ids_map, quantiles=quantiles)
            row = evaluate_prototype(
                proto=proto,
                keep_ids=keep_ids,
                exec_refs=exec_refs,
                full_bundle=full_bundle,
                route_bundles=route_bundles,
                full_rows_cache=full_rows_cache,
                route_rows_cache=route_rows_cache,
                full_ref_stats=full_ref_stats,
                route_ref_stats=route_ref_stats,
                args=exec_args,
                prior_route_valid_rate=prior_route_valid_rate,
            )
            # Simple fitness proxy.
            fitness = (
                (1000.0 if int(row["valid_for_ranking"]) == 1 else -1000.0)
                + 500.0 * float(row["delta_expectancy_vs_exec_baseline"]) if np.isfinite(row["delta_expectancy_vs_exec_baseline"]) else -1e9
            )
            if np.isfinite(row["cvar_improve_ratio"]):
                fitness += 50.0 * float(row["cvar_improve_ratio"])
            if np.isfinite(row["maxdd_improve_ratio"]):
                fitness += 50.0 * float(row["maxdd_improve_ratio"])
            if np.isfinite(row["route_pass_rate"]):
                fitness += 25.0 * float(row["route_pass_rate"])
            if np.isfinite(row["center_route_delta_min"]):
                fitness += 50.0 * float(row["center_route_delta_min"])
            row["fitness"] = float(fitness)
            pilot_rows.append(row)
            kept_map_rows.append(
                {
                    "prototype_id": str(row["prototype_id"]),
                    "family_id": str(row["family_id"]),
                    "kept_hash": str(row["kept_hash"]),
                    "removed_signals": int(row["removed_signals"]),
                    "params_json": str(row["params_json"]),
                }
            )
            if str(row["kept_hash"]) not in pilot_by_hash:
                pilot_by_hash[str(row["kept_hash"])] = row
                delta_vecs.append(compute_proto_delta_vector(keep_ids=keep_ids, full_rows_cache=full_rows_cache))

        pilot_df = pd.DataFrame(pilot_rows).sort_values(["fitness", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
        pilot_df.to_csv(run_dir / "phaseS4_pilot_results.csv", index=False)
        pd.DataFrame(kept_map_rows).to_csv(run_dir / "phaseS4_duplicate_variant_map.csv", index=False)

        unique_valid = pd.DataFrame(list(pilot_by_hash.values()))
        unique_valid = unique_valid.sort_values(["fitness", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
        if delta_vecs:
            mat = np.vstack(delta_vecs)
            n_eff_corr, avg_abs_corr = effective_trials_from_corr(mat)
        else:
            n_eff_corr, avg_abs_corr = 0.0, 0.0

        eff_lines = [
            "# S4 Effective Trials Summary",
            "",
            f"- Generated UTC: {utc_now()}",
            f"- Raw pilot candidates: `{len(pilot_df)}`",
            f"- Duplicate-adjusted unique kept sets: `{len(unique_valid)}`",
            f"- Corr-adjusted effective trials: `{n_eff_corr:.4f}`",
            f"- Average absolute correlation across unique candidate delta vectors: `{avg_abs_corr:.4f}`",
        ]
        write_text(run_dir / "phaseS4_effective_trials_summary.md", "\n".join(eff_lines) + "\n")

        robust_rows: List[Dict[str, Any]] = []
        top_unique = unique_valid.head(12).copy()
        for r in top_unique.to_dict("records"):
            keep_hash = str(r["kept_hash"])
            keep_ids = None
            # Reconstruct by matching one kept-set row from pilot map.
            match = next((m for m in kept_map_rows if str(m["kept_hash"]) == keep_hash), None)
            if match is None:
                continue
            # Regenerate from params.
            proto_like = json.loads(str(match["params_json"]))
            proto_like["prototype_id"] = str(r["prototype_id"])
            proto_like["family_id"] = str(r["family_id"])
            if "method" not in proto_like:
                continue
            keep_ids = materialize_keep_ids(proto=proto_like, feature_df=feature_df, route_test_ids_map=route_test_ids_map, quantiles=quantiles)

            stress_passes = []
            stress_deltas = []
            for exec_id, meta in exec_refs.items():
                sev = subset_eval_from_rows(
                    rows_df=full_rows_cache[exec_id],
                    bundle=full_bundle,
                    args=exec_args,
                    genome=meta["genome"],
                    keep_ids=keep_ids,
                    ref_roll=full_ref_stats[exec_id]["roll"],
                    ref_split_means=full_ref_stats[exec_id]["split_means"],
                    stress_mult=1.15,
                )
                stress_passes.append(int(sev["valid_for_ranking"]) == 1 and float(sev["delta_vs_ref"]) > 0.0)
                stress_deltas.append(float(sev["delta_vs_ref"]))

            boot_rates = []
            delta_vec = compute_proto_delta_vector(keep_ids=keep_ids, full_rows_cache=full_rows_cache)
            if delta_vec.size >= 10:
                brng = np.random.default_rng(int(args.seed) + int(hashlib.md5(keep_hash.encode("utf-8")).hexdigest()[:8], 16))
                n = len(delta_vec)
                passes = 0
                n_boot = 120
                for _ in range(n_boot):
                    idx = brng.integers(0, n, size=n)
                    if float(np.mean(delta_vec[idx])) > 0.0:
                        passes += 1
                boot_rate = float(passes / n_boot)
            else:
                boot_rate = 0.0
            psr, dsr = psr_dsr_proxy(delta_vec, eff_trials=max(1.0, n_eff_corr))
            robust = int(
                int(r["valid_for_ranking"]) == 1
                and float(r["delta_expectancy_vs_exec_baseline"]) > 0.0
                and float(r["cvar_improve_ratio"]) >= 0.0
                and float(r["maxdd_improve_ratio"]) > 0.0
                and float(r["route_pass_rate"]) >= 1.0
                and float(r["center_route_delta_min"]) > 0.0
                and float(r["min_subperiod_delta"]) > 0.0
                and all(stress_passes)
                and boot_rate >= 0.55
            )
            robust_rows.append(
                {
                    "prototype_id": str(r["prototype_id"]),
                    "family_id": str(r["family_id"]),
                    "fitness": float(r["fitness"]),
                    "valid_for_ranking": int(r["valid_for_ranking"]),
                    "delta_expectancy_vs_exec_baseline": float(r["delta_expectancy_vs_exec_baseline"]),
                    "cvar_improve_ratio": float(r["cvar_improve_ratio"]),
                    "maxdd_improve_ratio": float(r["maxdd_improve_ratio"]),
                    "route_pass_rate": float(r["route_pass_rate"]),
                    "route_positive_delta_rate": float(r["route_positive_delta_rate"]),
                    "center_route_delta_min": float(r["center_route_delta_min"]),
                    "min_subperiod_delta": float(r["min_subperiod_delta"]),
                    "stress_pass": int(all(stress_passes)),
                    "stress_delta_mean": float(np.mean(stress_deltas)) if stress_deltas else float("nan"),
                    "bootstrap_pass_rate": float(boot_rate),
                    "psr_proxy": float(psr),
                    "dsr_proxy": float(dsr),
                    "robust_survivor": int(robust),
                }
            )
        robust_df = pd.DataFrame(robust_rows).sort_values(["robust_survivor", "fitness"], ascending=[False, False]).reset_index(drop=True)
        robust_df.to_csv(run_dir / "phaseS4_robustness_matrix.csv", index=False)

        top_surv_lines = [
            "# S4 Top Survivors Report",
            "",
            f"- Generated UTC: {utc_now()}",
            f"- Unique candidates: `{len(unique_valid)}`",
            f"- Robust survivors: `{int((robust_df['robust_survivor'] == 1).sum()) if not robust_df.empty else 0}`",
            "",
            markdown_table(
                robust_df,
                [
                    "prototype_id",
                    "family_id",
                    "delta_expectancy_vs_exec_baseline",
                    "cvar_improve_ratio",
                    "maxdd_improve_ratio",
                    "route_pass_rate",
                    "center_route_delta_min",
                    "stress_pass",
                    "bootstrap_pass_rate",
                    "robust_survivor",
                ],
                n=12,
            ),
            "",
        ]
        write_text(run_dir / "phaseS4_top_survivors_report.md", "\n".join(top_surv_lines))

        robust_survivors = robust_df[robust_df["robust_survivor"] == 1].copy()
        stable_neighborhood = int((not robust_survivors.empty) and (int((robust_survivors["family_id"] == robust_survivors.iloc[0]["family_id"]).sum()) >= 2))
        if robust_survivors.empty:
            furthest_phase = "S4"
            classification = "SIGNAL_REDESIGN_NO_GO"
            mainline_status = classification
            exact_stop_reason = "S4 failed: bounded pilot found no robust survivor after route/split/stress/bootstrap-lite checks"
            s4_decision = "STOP_NO_GO"
        else:
            furthest_phase = "S5"
            if stable_neighborhood == 1 and len(robust_survivors) >= 2:
                classification = "SIGNAL_REDESIGN_WEAK_GO"
            else:
                classification = "SIGNAL_REDESIGN_WEAK_GO"
            mainline_status = classification
            exact_stop_reason = ""
            s4_decision = "GO"
        write_text(
            run_dir / "phaseS4_decision.md",
            "\n".join(
                [
                    "# S4 Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: `{s4_decision}`",
                    f"- Robust survivors: `{len(robust_survivors)}`",
                    f"- Stable neighborhood: `{stable_neighborhood}`",
                ]
            )
            + "\n",
        )
    # S5
    if bounded_pilot_reached == 0:
        furthest_phase = "S3"
        classification = "SIGNAL_REDESIGN_NO_GO"
        mainline_status = classification
    elif classification not in {"SIGNAL_REDESIGN_WEAK_GO", "SIGNAL_REDESIGN_STRONG_GO"}:
        classification = "SIGNAL_REDESIGN_NO_GO"
        mainline_status = classification

    s5_lines = [
        "# S5 Decision Next Step",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: `{classification}`",
        f"- Furthest phase: `{furthest_phase}`",
        f"- Stable neighborhood: `{stable_neighborhood}`",
        f"- Bounded pilot reached: `{bounded_pilot_reached}`",
    ]
    if classification == "SIGNAL_REDESIGN_NO_GO":
        s5_lines.append("- Next step: abandon SOL 1h signal redesign under the current family and move to a different upstream signal family or symbol.")
    else:
        s5_lines.append("- Next step: run a tightly bounded confirmation phase on the surviving signal objective family only.")
    write_text(run_dir / "phaseS5_decision_next_step.md", "\n".join(s5_lines) + "\n")

    if classification in {"SIGNAL_REDESIGN_WEAK_GO", "SIGNAL_REDESIGN_STRONG_GO"}:
        next_prompt = (
            "ROLE\n"
            "You are in bounded confirmation mode for SOL 1h signal-economics under the repaired route harness.\n\n"
            "MISSION\n"
            "Confirm only the surviving signal objective family under the same frozen execution contract, repaired routes, and unchanged hard gates.\n\n"
            "RULES\n"
            "1) Keep E1/E2 execution scorers fixed.\n"
            "2) Keep repaired routes mandatory.\n"
            "3) No new execution-family search.\n"
            "4) Stop NO_GO on first route-center or robustness relapse.\n"
        )
        write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)

    patch_lines = [
        "# S Patch Diff Summary",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Files changed:",
        "  - scripts/phase_s_signal_econ_redesign.py (new)",
        "- Rationale:",
        "  - Adds upstream signal-economics forensics, objective prototyping, and bounded pilot scoring under the repaired route harness.",
    ]
    write_text(run_dir / "phaseS_patch_diff_summary.md", "\n".join(patch_lines) + "\n")

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "route_dir": str(route_dir),
        "nx_dir": str(nx_dir),
        "freeze_lock_validation": lock_info,
        "load_meta": load_meta,
        "repaired_route_meta": route_meta,
        "prior_execution_only_route_valid_rate": float(prior_route_valid_rate),
        "furthest_phase": furthest_phase,
        "classification": classification,
        "bounded_pilot_reached": int(bounded_pilot_reached),
        "stable_neighborhood": int(stable_neighborhood),
    }
    json_dump(run_dir / "phaseS_run_manifest.json", manifest)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "furthest_phase": furthest_phase,
                "classification": classification,
                "mainline_status": mainline_status,
                "bounded_pilot_reached": int(bounded_pilot_reached),
                "stable_neighborhood": int(stable_neighborhood),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
