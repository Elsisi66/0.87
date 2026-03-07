#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/root/analysis/0.87").resolve()
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution import ga_exec_3m_opt as ga  # noqa: E402


LOCKED = {
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_rank_key(v: Any) -> Tuple[float, ...]:
    if isinstance(v, list):
        return tuple(float(x) for x in v)
    s = str(v)
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, (list, tuple)):
            return tuple(float(x) for x in arr)
    except Exception:
        pass
    return tuple()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def ensure_cols(df: pd.DataFrame, cols: Sequence[str], fill: float = np.nan) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill
    return out


def norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def z_proxy(mean: float, std: float, n: float) -> float:
    if not (np.isfinite(mean) and np.isfinite(std) and np.isfinite(n)):
        return float("nan")
    if std <= 0 or n <= 1:
        return float("nan")
    return float(mean / (std / math.sqrt(n)))


def effective_trials_from_corr(mat: np.ndarray) -> Tuple[float, float]:
    if mat.size == 0 or mat.shape[0] <= 1:
        return float(mat.shape[0]), 0.0
    x = mat.copy()
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


def build_args(
    *,
    signals_csv: Path,
    wf_splits: int,
    train_ratio: float,
    fee_maker: float = 2.0,
    fee_taker: float = 4.0,
    slip_limit: float = 0.5,
    slip_market: float = 2.0,
    mode: str = "tight",
    hard_max_taker_share: float = 0.25,
    hard_max_median_fill_delay_min: float = 45.0,
    hard_max_p95_fill_delay_min: float = 180.0,
) -> Any:
    parser = ga.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = "SOLUSDT"
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = int(wf_splits)
    args.train_ratio = float(train_ratio)
    args.mode = str(mode)
    args.workers = 1
    args.seed = 20260225
    args.pop = 1
    args.gens = 1
    args.fee_bps_maker = float(fee_maker)
    args.fee_bps_taker = float(fee_taker)
    args.slippage_bps_limit = float(slip_limit)
    args.slippage_bps_market = float(slip_market)
    args.execution_config = "configs/execution_configs.yaml"
    args.canonical_fee_model_path = LOCKED["canonical_fee_model"]
    args.canonical_metrics_definition_path = LOCKED["canonical_metrics_definition"]
    args.expected_fee_model_sha256 = LOCKED["expected_fee_sha"]
    args.expected_metrics_definition_sha256 = LOCKED["expected_metrics_sha"]
    args.allow_freeze_hash_mismatch = 0
    args.hard_max_taker_share = float(hard_max_taker_share)
    args.hard_max_median_fill_delay_min = float(hard_max_median_fill_delay_min)
    args.hard_max_p95_fill_delay_min = float(hard_max_p95_fill_delay_min)
    return args


def extract_genome(row: pd.Series) -> Dict[str, Any]:
    g = {}
    for c in row.index:
        if not c.startswith("g_"):
            continue
        k = c[2:]
        v = row[c]
        if pd.isna(v):
            continue
        if isinstance(v, (np.integer, int)):
            g[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            fv = float(v)
            if abs(fv - round(fv)) < 1e-12 and k in {
                "max_fill_delay_min",
                "fallback_to_market",
                "fallback_delay_min",
                "micro_vol_filter",
                "killzone_filter",
                "mss_displacement_gate",
                "time_stop_min",
                "break_even_enabled",
                "trailing_enabled",
                "partial_take_enabled",
                "skip_if_vol_gate",
                "use_signal_quality_gate",
                "cooldown_min",
            }:
                g[k] = int(round(fv))
            else:
                g[k] = fv
        else:
            g[k] = v
    return ga._repair_genome(g, mode="tight", repair_hist=None)


def metric_signature(df: pd.DataFrame) -> pd.Series:
    cols = [
        "overall_entries_valid",
        "overall_entry_rate",
        "overall_exec_expectancy_net",
        "overall_exec_pnl_std",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_exec_p95_fill_delay_min",
    ]
    x = ensure_cols(df, cols)
    for c in cols:
        x[c] = to_num(x[c])
    return (
        x[cols]
        .round(12)
        .apply(lambda r: "|".join("nan" if pd.isna(v) else str(v) for v in r.tolist()), axis=1)
    )


def pick_source_run() -> Tuple[Path, Dict[str, Any], pd.DataFrame]:
    cands = sorted([p for p in REPORTS_ROOT.glob("GA_EXEC_OPT_*") if p.is_dir()], key=lambda p: p.name)
    selected = None
    selected_manifest = None
    selected_df = None
    for p in reversed(cands):
        req = ["genomes.csv", "run_manifest.json", "gen_status.json", "invalid_reason_histogram.json"]
        if not all((p / r).exists() for r in req):
            continue
        man = read_json(p / "run_manifest.json")
        if int(man.get("valid_for_ranking_count", 0)) <= 0:
            continue
        freeze = man.get("freeze_lock", {})
        if int(freeze.get("freeze_lock_pass", 0)) != 1:
            continue
        df = pd.read_csv(p / "genomes.csv")
        selected = p
        selected_manifest = man
        selected_df = df
        break
    if selected is None:
        raise RuntimeError("No eligible GA_EXEC_OPT_* run found with valid_for_ranking > 0 and freeze lock pass")
    return selected, selected_manifest, selected_df


def validate_global_freeze() -> Dict[str, Any]:
    rep = Path(LOCKED["representative_subset_csv"]).resolve()
    fee = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics = Path(LOCKED["canonical_metrics_definition"]).resolve()
    if not rep.exists() or not fee.exists() or not metrics.exists():
        raise FileNotFoundError("Locked setup file missing")
    fee_sha = sha256_file(fee)
    metrics_sha = sha256_file(metrics)
    return {
        "representative_subset_csv": str(rep),
        "canonical_fee_model": str(fee),
        "canonical_metrics_definition": str(metrics),
        "canonical_fee_sha256": fee_sha,
        "canonical_metrics_sha256": metrics_sha,
        "expected_fee_sha256": LOCKED["expected_fee_sha"],
        "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
        "fee_hash_match": int(fee_sha == LOCKED["expected_fee_sha"]),
        "metrics_hash_match": int(metrics_sha == LOCKED["expected_metrics_sha"]),
    }


def shortlist_from_full_run(df: pd.DataFrame, phase_q_dir: Path) -> pd.DataFrame:
    x = df.copy()
    x["rank_key_tuple"] = x.get("rank_key", pd.Series(["[]"] * len(x))).map(parse_rank_key)
    x["valid_for_ranking"] = to_num(x.get("valid_for_ranking", 0)).fillna(0).astype(int)
    x = x[x["valid_for_ranking"] == 1].copy()
    if x.empty:
        raise RuntimeError("No valid_for_ranking candidates in source run")

    x["metric_signature"] = metric_signature(x)
    x = x.drop_duplicates(subset=["genome_hash"]).copy()
    x = x.sort_values("rank_key_tuple", ascending=False).reset_index(drop=True)

    # Top 10 by rank among non-duplicate signatures.
    top = x.drop_duplicates(subset=["metric_signature"], keep="first").head(10).copy()
    top["selection_type"] = "top_rank"
    top["selection_rationale"] = "top10_rank_key_valid_nondedup"

    # Diversity picks from remaining, maximizing distance on behavior/risk features.
    rem = x[~x["genome_hash"].isin(top["genome_hash"])].copy()
    rem = rem.drop_duplicates(subset=["metric_signature"], keep="first").reset_index(drop=True)

    feat_cols = [
        "overall_exec_taker_share",
        "overall_exec_p95_fill_delay_min",
        "overall_entry_rate",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "g_limit_offset_bps",
        "g_max_fill_delay_min",
        "g_cooldown_min",
        "g_mss_displacement_gate",
        "g_micro_vol_filter",
        "g_use_signal_quality_gate",
    ]

    def _featurize(df_in: pd.DataFrame) -> np.ndarray:
        z = ensure_cols(df_in, feat_cols)
        for c in feat_cols:
            z[c] = to_num(z[c])
        arr = z[feat_cols].to_numpy(dtype=float)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            finite = np.isfinite(col)
            fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
            col[~finite] = fill
            mu = float(np.mean(col))
            sd = float(np.std(col))
            if sd <= 1e-12:
                col = col - mu
            else:
                col = (col - mu) / sd
            arr[:, j] = col
        return arr

    selected_div = []
    if not rem.empty:
        arr_top = _featurize(top)
        arr_rem = _featurize(rem)
        # Align scaling between top and rem by recomputing jointly.
        both = pd.concat([top[feat_cols], rem[feat_cols]], ignore_index=True)
        both = ensure_cols(both, feat_cols)
        for c in feat_cols:
            both[c] = to_num(both[c])
        all_arr = both.to_numpy(dtype=float)
        for j in range(all_arr.shape[1]):
            col = all_arr[:, j]
            finite = np.isfinite(col)
            fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
            col[~finite] = fill
            mu = float(np.mean(col))
            sd = float(np.std(col))
            if sd <= 1e-12:
                col = col - mu
            else:
                col = (col - mu) / sd
            all_arr[:, j] = col
        arr_top = all_arr[: len(top)]
        arr_rem = all_arr[len(top) :]

        chosen_idx = set()
        for _ in range(min(5, len(rem))):
            best_i = None
            best_d = -1.0
            for i in range(len(rem)):
                if i in chosen_idx:
                    continue
                v = arr_rem[i]
                base = arr_top if not chosen_idx else np.vstack([arr_top, arr_rem[list(chosen_idx)]])
                d = float(np.min(np.sqrt(np.sum((base - v) ** 2, axis=1)))) if base.size else 0.0
                if d > best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                break
            chosen_idx.add(best_i)
            row = rem.iloc[[best_i]].copy()
            row["selection_type"] = "diversity_pick"
            row["selection_rationale"] = f"max_min_distance_behavioral_features_{best_d:.4f}"
            row["diversity_distance"] = best_d
            selected_div.append(row)

    if selected_div:
        div_df = pd.concat(selected_div, ignore_index=True)
    else:
        div_df = pd.DataFrame(columns=top.columns.tolist() + ["diversity_distance"])

    top["diversity_distance"] = np.nan
    out = pd.concat([top, div_df], ignore_index=True)
    out = out.drop_duplicates(subset=["genome_hash"], keep="first").reset_index(drop=True)
    out["shortlist_index"] = np.arange(1, len(out) + 1)

    keep_cols = [
        "shortlist_index",
        "genome_hash",
        "selection_type",
        "selection_rationale",
        "diversity_distance",
        "valid_for_ranking",
        "overall_entries_valid",
        "overall_entry_rate",
        "overall_exec_expectancy_net",
        "overall_delta_expectancy_exec_minus_baseline",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_exec_taker_share",
        "overall_exec_p95_fill_delay_min",
        "rank_key",
        "g_entry_mode",
        "g_limit_offset_bps",
        "g_max_fill_delay_min",
        "g_mss_displacement_gate",
        "g_micro_vol_filter",
        "g_use_signal_quality_gate",
        "g_cooldown_min",
    ]
    out = ensure_cols(out, keep_cols)
    out[keep_cols].to_csv(phase_q_dir / "phaseQ_shortlist_selected.csv", index=False)
    return out


def eval_genomes(
    *,
    genomes_df: pd.DataFrame,
    args: Any,
    bundles: List[Any],
    route_name: str,
) -> pd.DataFrame:
    rows = []
    for _, r in genomes_df.iterrows():
        gh = str(r["genome_hash"])
        gdict = extract_genome(r)
        met = ga._evaluate_genome(genome=gdict, bundles=bundles, args=args, detailed=False)
        row = {
            "route": route_name,
            "genome_hash": gh,
            "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
            "constraint_pass": int(met.get("constraint_pass", 0)),
            "participation_pass": int(met.get("participation_pass", 0)),
            "realism_pass": int(met.get("realism_pass", 0)),
            "nan_pass": int(met.get("nan_pass", 0)),
            "invalid_reason": str(met.get("invalid_reason", "")),
            "overall_signals_total": float(met.get("overall_signals_total", np.nan)),
            "overall_entries_valid": float(met.get("overall_entries_valid", np.nan)),
            "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
            "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
            "overall_baseline_expectancy_net": float(met.get("overall_baseline_expectancy_net", np.nan)),
            "overall_delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
            "overall_exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
            "overall_baseline_cvar_5": float(met.get("overall_baseline_cvar_5", np.nan)),
            "overall_cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
            "overall_exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
            "overall_baseline_max_drawdown": float(met.get("overall_baseline_max_drawdown", np.nan)),
            "overall_maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
            "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
            "overall_exec_median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
            "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
            "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
            "median_split_expectancy_net": float(met.get("median_split_expectancy_net", np.nan)),
            "std_split_expectancy_net": float(met.get("std_split_expectancy_net", np.nan)),
            "tail_gate_pass_cvar": int(met.get("tail_gate_pass_cvar", 0)),
            "tail_gate_pass_maxdd": int(met.get("tail_gate_pass_maxdd", 0)),
            "eval_time_sec": float(met.get("eval_time_sec", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def phase_q(
    *,
    parent_dir: Path,
    source_run_dir: Path,
    source_manifest: Dict[str, Any],
    source_df: pd.DataFrame,
    freeze_global: Dict[str, Any],
    command_log: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
    q_dir = parent_dir / "phaseQ"
    t0 = time.time()

    shortlist = shortlist_from_full_run(source_df, q_dir)
    shortlist = shortlist.merge(source_df, on="genome_hash", how="left", suffixes=("", "_src"))

    # Build route datasets.
    sig_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    sig = pd.read_csv(sig_fp)
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True, errors="coerce")
    sig = sig[sig["signal_time"].notna()].sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    holdout_n = max(120, int(round(len(sig) * 0.20)))
    holdout = sig.iloc[-holdout_n:].copy().reset_index(drop=True)
    holdout_fp = q_dir / "route1_holdout_signals.csv"
    holdout.to_csv(holdout_fp, index=False)

    # Route2 uses full signals with split perturbation.
    route2_fp = q_dir / "route2_full_signals.csv"
    sig.to_csv(route2_fp, index=False)

    # Validate freeze lock for phase Q mainline by invoking canonical validator.
    args_lock = build_args(signals_csv=sig_fp, wf_splits=5, train_ratio=0.70)
    q_freeze = ga._validate_and_lock_frozen_artifacts(args=args_lock, run_dir=q_dir)

    # Evaluate routes.
    args_r1 = build_args(signals_csv=holdout_fp, wf_splits=3, train_ratio=0.50)
    args_r2 = build_args(signals_csv=route2_fp, wf_splits=7, train_ratio=0.65)

    b1, _ = ga._prepare_bundles(args_r1)
    b2, _ = ga._prepare_bundles(args_r2)

    t_r1 = time.time()
    r1 = eval_genomes(genomes_df=shortlist, args=args_r1, bundles=b1, route_name="route1_holdout")
    r1["route_duration_sec"] = float(time.time() - t_r1)
    r1.to_csv(q_dir / "phaseQ_oos_results_route1.csv", index=False)

    t_r2 = time.time()
    r2 = eval_genomes(genomes_df=shortlist, args=args_r2, bundles=b2, route_name="route2_reslice")
    r2["route_duration_sec"] = float(time.time() - t_r2)
    r2.to_csv(q_dir / "phaseQ_oos_results_route2.csv", index=False)

    # Stability + degradation.
    base_cols = [
        "genome_hash",
        "overall_exec_expectancy_net",
        "overall_delta_expectancy_exec_minus_baseline",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_entries_valid",
    ]
    base = source_df[base_cols].copy().drop_duplicates("genome_hash")

    rr = []
    for route_name, rdf in [("route1_holdout", r1), ("route2_reslice", r2)]:
        m = rdf.merge(base, on="genome_hash", how="left", suffixes=("", "_base"))
        m["delta_expectancy_abs_drop"] = m["overall_exec_expectancy_net"] - m["overall_exec_expectancy_net_base"]
        m["delta_expectancy_pct_drop"] = [
            safe_div(a - b, abs(b)) for a, b in zip(m["overall_exec_expectancy_net"], m["overall_exec_expectancy_net_base"])
        ]
        m["delta_vs_base_abs_drop"] = m["overall_delta_expectancy_exec_minus_baseline"] - m["overall_delta_expectancy_exec_minus_baseline_base"]
        m["delta_vs_base_pct_drop"] = [
            safe_div(a - b, abs(b))
            for a, b in zip(m["overall_delta_expectancy_exec_minus_baseline"], m["overall_delta_expectancy_exec_minus_baseline_base"])
        ]
        m["route_pass"] = (
            (to_num(m["valid_for_ranking"]).fillna(0).astype(int) == 1)
            & (to_num(m["overall_delta_expectancy_exec_minus_baseline"]) > 0.0)
            & (to_num(m["overall_cvar_improve_ratio"]) > 0.0)
            & (to_num(m["overall_maxdd_improve_ratio"]) > 0.0)
            & (to_num(m["overall_entries_valid"]) >= 200)
            & (to_num(m["overall_entry_rate"]) >= 0.70)
        ).astype(int)
        m["route"] = route_name
        rr.append(m)
    degr = pd.concat(rr, ignore_index=True)
    degr.to_csv(q_dir / "phaseQ_degradation_analysis.csv", index=False)

    # Rank-order stability.
    def _route_rank(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
        z = df_in.copy()
        z = z.sort_values(
            ["valid_for_ranking", "overall_exec_expectancy_net", "overall_delta_expectancy_exec_minus_baseline"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        z[col] = np.arange(1, len(z) + 1)
        return z[["genome_hash", col]]

    rk1 = _route_rank(r1, "rank_r1")
    rk2 = _route_rank(r2, "rank_r2")
    stab = rk1.merge(rk2, on="genome_hash", how="outer")
    stab = stab.merge(
        degr.pivot_table(index="genome_hash", columns="route", values="route_pass", aggfunc="max").reset_index(),
        on="genome_hash",
        how="left",
    )
    stab["rank_shift_abs"] = (to_num(stab["rank_r1"]) - to_num(stab["rank_r2"])).abs()
    stab["pass_both_routes"] = (
        (to_num(stab.get("route1_holdout", 0)).fillna(0).astype(int) == 1)
        & (to_num(stab.get("route2_reslice", 0)).fillna(0).astype(int) == 1)
    ).astype(int)

    # sign consistency on delta vs baseline
    d1 = r1[["genome_hash", "overall_delta_expectancy_exec_minus_baseline"]].rename(columns={"overall_delta_expectancy_exec_minus_baseline": "delta_r1"})
    d2 = r2[["genome_hash", "overall_delta_expectancy_exec_minus_baseline"]].rename(columns={"overall_delta_expectancy_exec_minus_baseline": "delta_r2"})
    stab = stab.merge(d1, on="genome_hash", how="left").merge(d2, on="genome_hash", how="left")
    stab["delta_sign_consistent"] = (
        np.sign(to_num(stab["delta_r1"]).fillna(0)) == np.sign(to_num(stab["delta_r2"]).fillna(0))
    ).astype(int)
    stab.to_csv(q_dir / "phaseQ_oos_stability_summary.csv", index=False)

    # Decision logic.
    pass_both = int(stab["pass_both_routes"].sum())
    pass_any = int(((to_num(stab.get("route1_holdout", 0)).fillna(0).astype(int) == 1) | (to_num(stab.get("route2_reslice", 0)).fillna(0).astype(int) == 1)).sum())

    scons = float(stab["delta_sign_consistent"].mean()) if len(stab) else 0.0
    med_delta_drop = float(pd.to_numeric(degr["delta_vs_base_pct_drop"], errors="coerce").median()) if not degr.empty else float("nan")

    if pass_both >= 3 and scons >= 0.60 and (not np.isfinite(med_delta_drop) or med_delta_drop > -0.80):
        q_result = "Q_PASS_STRONG"
    elif pass_any >= 1:
        q_result = "Q_PASS_WEAK"
    else:
        q_result = "Q_FAIL"

    # Survivors for Phase R.
    surv = stab[stab["pass_both_routes"] == 1][["genome_hash", "rank_r1", "rank_r2", "rank_shift_abs"]].copy()
    if surv.empty:
        surv = stab[((to_num(stab.get("route1_holdout", 0)).fillna(0).astype(int) == 1) | (to_num(stab.get("route2_reslice", 0)).fillna(0).astype(int) == 1))][["genome_hash", "rank_r1", "rank_r2", "rank_shift_abs"]].copy()
    if not surv.empty:
        surv["rank_sum"] = to_num(surv["rank_r1"]).fillna(9999) + to_num(surv["rank_r2"]).fillna(9999)
        surv = surv.sort_values(["rank_sum", "rank_shift_abs"], ascending=[True, True]).reset_index(drop=True)

    # Report + manifest.
    man = {
        "generated_utc": utc_now(),
        "phase": "Q",
        "source_run_dir": str(source_run_dir),
        "source_valid_for_ranking_count": int(source_manifest.get("valid_for_ranking_count", 0)),
        "freeze_global": freeze_global,
        "phase_freeze_lock_validation": q_freeze,
        "shortlist_count": int(len(shortlist)),
        "route1_signals": int(len(holdout)),
        "route2_signals": int(len(sig)),
        "pass_both_count": int(pass_both),
        "pass_any_count": int(pass_any),
        "delta_sign_consistency": float(scons),
        "median_delta_vs_base_pct_drop": float(med_delta_drop) if np.isfinite(med_delta_drop) else None,
        "decision": q_result,
        "duration_sec": float(time.time() - t0),
        "command_log": command_log,
        "code_modified": "NO",
    }
    json_dump(q_dir / "phaseQ_run_manifest.json", man)

    lines = []
    lines.append("# Phase Q Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Source run: `{source_run_dir}`")
    lines.append(f"- Shortlist size: {len(shortlist)}")
    lines.append(f"- Route1 (holdout) signals: {len(holdout)}")
    lines.append(f"- Route2 (reslice) signals: {len(sig)}")
    lines.append("")
    lines.append("## OOS Summary")
    lines.append("")
    lines.append(f"- pass_both_routes: {pass_both}")
    lines.append(f"- pass_any_route: {pass_any}")
    lines.append(f"- delta_sign_consistency: {scons:.4f}")
    lines.append(f"- median_delta_vs_base_pct_drop: {med_delta_drop:.6f}" if np.isfinite(med_delta_drop) else "- median_delta_vs_base_pct_drop: nan")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Result: **{q_result}**")
    (q_dir / "phaseQ_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    (q_dir / "decision_phaseQ.md").write_text(
        f"# Decision Phase Q\n\n- Generated UTC: {utc_now()}\n- Decision: **{q_result}**\n- pass_both_routes: {pass_both}\n- pass_any_route: {pass_any}\n",
        encoding="utf-8",
    )

    if q_result == "Q_FAIL":
        # Failure branch artifacts and stop mainline.
        fail_class = "C"
        if pass_any == 0 and pass_both == 0 and scons < 0.5:
            fail_class = "B"
        elif pass_any == 0:
            fail_class = "A"
        elif len(shortlist) > 0 and len(shortlist["genome_hash"].unique()) < 6:
            fail_class = "D"

        root = []
        root.append("# Phase Q Failure Root Cause")
        root.append("")
        root.append(f"- Decision: Q_FAIL")
        root.append(f"- Classification: {fail_class}")
        root.append(f"- pass_any_route: {pass_any}")
        root.append(f"- pass_both_routes: {pass_both}")
        root.append(f"- sign_consistency: {scons:.4f}")
        (q_dir / "phaseQ_failure_root_cause.md").write_text("\n".join(root).strip() + "\n", encoding="utf-8")

        rec = []
        rec.append("# Phase Q Retest Recommendations")
        rec.append("")
        rec.append("- Status: NO_DEPLOY and NO_PHASE_R/S_MAINLINE")
        rec.append("- Next: run sampler/objective redesign focused on OOS sign-retention constraints before additional compute.")
        (q_dir / "phaseQ_retest_recommendations.md").write_text("\n".join(rec).strip() + "\n", encoding="utf-8")
        return q_result, man, shortlist, None

    return q_result, man, shortlist, surv


def build_signals_variants(base_signals: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    out = {}
    b = base_signals.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    out["baseline"] = out_dir / "signals_baseline.csv"
    b.to_csv(out["baseline"], index=False)

    s3 = b.copy()
    s3["signal_time"] = pd.to_datetime(s3["signal_time"], utc=True) + pd.Timedelta(minutes=3)
    out["latency_plus3m"] = out_dir / "signals_latency_plus3m.csv"
    s3.to_csv(out["latency_plus3m"], index=False)

    s6 = b.copy()
    s6["signal_time"] = pd.to_datetime(s6["signal_time"], utc=True) + pd.Timedelta(minutes=6)
    out["latency_plus6m"] = out_dir / "signals_latency_plus6m.csv"
    s6.to_csv(out["latency_plus6m"], index=False)

    alt = b.iloc[::2].copy().reset_index(drop=True)
    out["subset_alt_chrono"] = out_dir / "signals_subset_alt_chrono.csv"
    alt.to_csv(out["subset_alt_chrono"], index=False)

    miss = b.copy().reset_index(drop=True)
    mask = np.ones(len(miss), dtype=bool)
    mask[np.arange(0, len(miss), 20)] = False
    miss = miss[mask].copy().reset_index(drop=True)
    out["missingness_5pct"] = out_dir / "signals_missingness_5pct.csv"
    miss.to_csv(out["missingness_5pct"], index=False)

    return out


def phase_r(
    *,
    parent_dir: Path,
    source_run_dir: Path,
    source_df: pd.DataFrame,
    q_result: str,
    q_manifest: Dict[str, Any],
    shortlist: pd.DataFrame,
    survivors: pd.DataFrame,
    freeze_global: Dict[str, Any],
    command_log: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], Optional[pd.DataFrame]]:
    r_dir = parent_dir / "phaseR"
    t0 = time.time()

    if survivors is None or survivors.empty:
        raise RuntimeError("Phase R invoked without survivors")

    # Choose survivor set size based on Q strength.
    n_surv = 8 if q_result == "Q_PASS_STRONG" else 4
    surv_hashes = survivors["genome_hash"].head(n_surv).tolist()
    pool = source_df[source_df["genome_hash"].isin(surv_hashes)].copy()
    pool = pool.drop_duplicates("genome_hash").copy()
    if pool.empty:
        raise RuntimeError("No survivor genomes found in source run dataframe")

    # Save survivor set.
    surv_cols = [
        "genome_hash",
        "valid_for_ranking",
        "overall_entries_valid",
        "overall_entry_rate",
        "overall_exec_expectancy_net",
        "overall_delta_expectancy_exec_minus_baseline",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_exec_taker_share",
        "overall_exec_p95_fill_delay_min",
        "g_entry_mode",
        "g_limit_offset_bps",
        "g_max_fill_delay_min",
        "g_mss_displacement_gate",
        "g_micro_vol_filter",
        "g_use_signal_quality_gate",
        "g_cooldown_min",
    ]
    ensure_cols(pool, surv_cols)[surv_cols].to_csv(r_dir / "phaseR_survivor_set.csv", index=False)

    # Build baseline signals and variants.
    sig = pd.read_csv(Path(LOCKED["representative_subset_csv"]))
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True, errors="coerce")
    sig = sig[sig["signal_time"].notna()].sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    variants = build_signals_variants(sig, r_dir)

    # Stress scenarios.
    scenarios = [
        {
            "scenario": "baseline_canonical",
            "analysis_only": 0,
            "signals_key": "baseline",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "fee_slip_plus25",
            "analysis_only": 1,
            "signals_key": "baseline",
            "fee_mult": 1.25,
            "slip_mult": 1.25,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "fee_slip_plus50",
            "analysis_only": 1,
            "signals_key": "baseline",
            "fee_mult": 1.50,
            "slip_mult": 1.50,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "latency_plus3m",
            "analysis_only": 1,
            "signals_key": "latency_plus3m",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "latency_plus6m",
            "analysis_only": 1,
            "signals_key": "latency_plus6m",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "diag_taker_cap_tight",
            "analysis_only": 1,
            "signals_key": "baseline",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.10,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "diag_delay_caps_tight",
            "analysis_only": 1,
            "signals_key": "baseline",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 15.0,
            "hard_p95_delay": 60.0,
        },
        {
            "scenario": "subset_alt_chrono",
            "analysis_only": 1,
            "signals_key": "subset_alt_chrono",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
        {
            "scenario": "missingness_5pct",
            "analysis_only": 1,
            "signals_key": "missingness_5pct",
            "fee_mult": 1.0,
            "slip_mult": 1.0,
            "hard_taker_cap": 0.25,
            "hard_med_delay": 45.0,
            "hard_p95_delay": 180.0,
        },
    ]

    # Baseline freeze lock validation for phase R mainline.
    args_lock = build_args(signals_csv=variants["baseline"], wf_splits=5, train_ratio=0.70)
    r_freeze = ga._validate_and_lock_frozen_artifacts(args=args_lock, run_dir=r_dir)

    stress_rows = []
    scen_build_meta = []

    for sc in scenarios:
        sname = sc["scenario"]
        sfp = variants[sc["signals_key"]]
        args_sc = build_args(
            signals_csv=sfp,
            wf_splits=5,
            train_ratio=0.70,
            fee_maker=2.0 * sc["fee_mult"],
            fee_taker=4.0 * sc["fee_mult"],
            slip_limit=0.5 * sc["slip_mult"],
            slip_market=2.0 * sc["slip_mult"],
            hard_max_taker_share=sc["hard_taker_cap"],
            hard_max_median_fill_delay_min=sc["hard_med_delay"],
            hard_max_p95_fill_delay_min=sc["hard_p95_delay"],
        )
        tb = time.time()
        bundles, _meta = ga._prepare_bundles(args_sc)
        scen_build_meta.append({"scenario": sname, "bundle_build_sec": float(time.time() - tb), "signals_file": str(sfp)})
        for _, rr in pool.iterrows():
            gh = str(rr["genome_hash"])
            gdict = extract_genome(rr)
            met = ga._evaluate_genome(genome=gdict, bundles=bundles, args=args_sc, detailed=False)
            stress_rows.append(
                {
                    "scenario": sname,
                    "analysis_only": int(sc["analysis_only"]),
                    "genome_hash": gh,
                    "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                    "constraint_pass": int(met.get("constraint_pass", 0)),
                    "participation_pass": int(met.get("participation_pass", 0)),
                    "realism_pass": int(met.get("realism_pass", 0)),
                    "nan_pass": int(met.get("nan_pass", 0)),
                    "invalid_reason": str(met.get("invalid_reason", "")),
                    "overall_entries_valid": float(met.get("overall_entries_valid", np.nan)),
                    "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
                    "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                    "overall_delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                    "overall_cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
                    "overall_maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
                    "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                    "overall_exec_median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
                    "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                    "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
                    "median_split_expectancy_net": float(met.get("median_split_expectancy_net", np.nan)),
                    "std_split_expectancy_net": float(met.get("std_split_expectancy_net", np.nan)),
                    "tail_gate_pass_cvar": int(met.get("tail_gate_pass_cvar", 0)),
                    "tail_gate_pass_maxdd": int(met.get("tail_gate_pass_maxdd", 0)),
                    "fee_mult": float(sc["fee_mult"]),
                    "slip_mult": float(sc["slip_mult"]),
                    "hard_taker_cap": float(sc["hard_taker_cap"]),
                    "hard_med_delay": float(sc["hard_med_delay"]),
                    "hard_p95_delay": float(sc["hard_p95_delay"]),
                }
            )

    stress = pd.DataFrame(stress_rows)
    stress.to_csv(r_dir / "phaseR_stress_matrix_results.csv", index=False)

    baseline = stress[stress["scenario"] == "baseline_canonical"].copy()
    base = baseline.set_index("genome_hash")

    score_rows = []
    flag_rows = []
    for gh in pool["genome_hash"].astype(str).tolist():
        sub = stress[stress["genome_hash"] == gh].copy()
        if sub.empty:
            continue
        b = base.loc[gh] if gh in base.index else None
        if b is None or isinstance(b, pd.DataFrame):
            continue

        sign_ret = float((to_num(sub["overall_delta_expectancy_exec_minus_baseline"]) > 0).mean())
        risk_ret = float(((to_num(sub["overall_cvar_improve_ratio"]) > 0) & (to_num(sub["overall_maxdd_improve_ratio"]) > 0)).mean())
        part_ret = float((to_num(sub["participation_pass"]).fillna(0).astype(int) == 1).mean())

        fee_sub = sub[sub["scenario"].isin(["baseline_canonical", "fee_slip_plus25", "fee_slip_plus50"])].copy()
        fee_sub = fee_sub.sort_values("fee_mult")
        x = to_num(fee_sub["fee_mult"]).to_numpy(dtype=float) - 1.0
        y = to_num(fee_sub["overall_delta_expectancy_exec_minus_baseline"]).to_numpy(dtype=float)
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = float("nan")

        robust_pass = int(
            (sign_ret >= 0.70)
            and (risk_ret >= 0.70)
            and (part_ret >= 0.55)
            and float(b["overall_delta_expectancy_exec_minus_baseline"]) > 0
            and float(b["overall_cvar_improve_ratio"]) > 0
            and float(b["overall_maxdd_improve_ratio"]) > 0
        )

        score_rows.append(
            {
                "genome_hash": gh,
                "baseline_delta_expectancy": float(b["overall_delta_expectancy_exec_minus_baseline"]),
                "baseline_exec_expectancy": float(b["overall_exec_expectancy_net"]),
                "baseline_cvar_improve": float(b["overall_cvar_improve_ratio"]),
                "baseline_maxdd_improve": float(b["overall_maxdd_improve_ratio"]),
                "baseline_taker_share": float(b["overall_exec_taker_share"]),
                "baseline_p95_delay": float(b["overall_exec_p95_fill_delay_min"]),
                "sign_retention_delta": sign_ret,
                "risk_retention_ratio": risk_ret,
                "participation_retention_ratio": part_ret,
                "fee_degradation_slope": slope,
                "robust_pass": robust_pass,
            }
        )

        # Brittleness flags.
        sc25 = sub[sub["scenario"] == "fee_slip_plus25"]
        sc_taker = sub[sub["scenario"] == "diag_taker_cap_tight"]
        sc_delay = sub[sub["scenario"] == "diag_delay_caps_tight"]

        flip_small_cost = int(
            (not sc25.empty)
            and (float(to_num(sc25.iloc[0:1]["overall_delta_expectancy_exec_minus_baseline"]).iloc[0]) <= 0)
        )
        taker_corner = int(
            float(b["overall_exec_taker_share"]) < 0.02
            and (
                (not sc_taker.empty)
                and int(to_num(sc_taker.iloc[0:1]["valid_for_ranking"]).fillna(0).astype(int).iloc[0]) == 0
            )
        )
        split_instability = int(
            np.isfinite(float(b["min_split_expectancy_net"]))
            and np.isfinite(float(b["median_split_expectancy_net"]))
            and (
                float(b["min_split_expectancy_net"])
                < float(b["median_split_expectancy_net"]) - abs(float(b["median_split_expectancy_net"])) * 2.0
            )
        )
        delay_fragile = int(
            (not sc_delay.empty)
            and int(to_num(sc_delay.iloc[0:1]["realism_pass"]).fillna(0).astype(int).iloc[0]) == 0
        )

        flag_rows.append(
            {
                "genome_hash": gh,
                "flag_flip_negative_small_cost": flip_small_cost,
                "flag_taker_corner_dependency": taker_corner,
                "flag_split_instability": split_instability,
                "flag_delay_cap_fragility": delay_fragile,
                "brittle_any": int(flip_small_cost or taker_corner or split_instability or delay_fragile),
            }
        )

    score = pd.DataFrame(score_rows)
    flags = pd.DataFrame(flag_rows)

    # Add significance controls on survivor family baseline.
    bsv = baseline.copy()
    bsv["metric_signature"] = metric_signature(bsv)
    bsv["metric_signature_group_size"] = bsv.groupby("metric_signature")["genome_hash"].transform("count").astype(int)
    bsv["is_metric_duplicate"] = (bsv["metric_signature_group_size"] > 1).astype(int)

    n_uncorr = int(bsv["metric_signature"].nunique())
    nondup = bsv.drop_duplicates("metric_signature", keep="first").copy()
    mat = nondup[
        [
            "overall_entries_valid",
            "overall_entry_rate",
            "overall_exec_expectancy_net",
            "overall_cvar_improve_ratio",
            "overall_maxdd_improve_ratio",
            "overall_exec_taker_share",
            "overall_exec_p95_fill_delay_min",
        ]
    ].to_numpy(dtype=float)
    n_eff_corr, avg_abs_corr = effective_trials_from_corr(mat)
    penalty = math.sqrt(2.0 * math.log(max(2.0, n_eff_corr if np.isfinite(n_eff_corr) else float(n_uncorr))))

    nondup["z_expectancy_proxy"] = [
        z_proxy(m, s, n)
        for m, s, n in zip(
            to_num(nondup["overall_exec_expectancy_net"]),
            np.maximum(1e-12, np.abs(to_num(nondup["overall_exec_expectancy_net"])) * 0.5 + 1e-4),
            np.maximum(2.0, to_num(nondup["overall_entries_valid"])),
        )
    ]
    nondup["psr_proxy"] = nondup["z_expectancy_proxy"].map(norm_cdf)
    nondup["dsr_proxy"] = nondup["z_expectancy_proxy"].map(lambda z: norm_cdf(z - penalty))

    score = score.merge(flags, on="genome_hash", how="left")
    score = score.merge(
        nondup[["genome_hash", "psr_proxy", "dsr_proxy", "metric_signature_group_size", "is_metric_duplicate"]],
        on="genome_hash",
        how="left",
    )

    score.to_csv(r_dir / "phaseR_robustness_scorecard.csv", index=False)
    flags.to_csv(r_dir / "phaseR_brittleness_flags.csv", index=False)

    sig_lines = []
    sig_lines.append("# Phase R Significance Controls")
    sig_lines.append("")
    sig_lines.append(f"- Generated UTC: {utc_now()}")
    sig_lines.append(f"- survivor_candidates: {len(pool)}")
    sig_lines.append(f"- baseline_nonduplicate_signatures: {n_uncorr}")
    sig_lines.append(f"- effective_trials_uncorrelated: {float(n_uncorr):.6f}")
    sig_lines.append(f"- avg_abs_metric_corr_proxy: {float(avg_abs_corr):.6f}")
    sig_lines.append(f"- effective_trials_corr_adjusted: {float(n_eff_corr):.6f}")
    sig_lines.append("")
    sig_lines.append("PSR/DSR proxies are exported in `phaseR_robustness_scorecard.csv`.")
    sig_lines.append("")
    sig_lines.append("Reality-check control:")
    sig_lines.append("- TODO: bootstrap reality-check benchmark remains unimplemented in this phase.")
    (r_dir / "phaseR_significance_controls.md").write_text("\n".join(sig_lines).strip() + "\n", encoding="utf-8")

    robust_count = int(to_num(score.get("robust_pass", 0)).fillna(0).astype(int).sum()) if not score.empty else 0
    brittle_rate = float(to_num(score.get("brittle_any", 0)).fillna(0).mean()) if not score.empty else float("nan")

    if robust_count >= 3:
        r_result = "R_PASS"
    elif robust_count >= 1:
        r_result = "R_PASS_FRAGILE"
    else:
        r_result = "R_FAIL"

    man = {
        "generated_utc": utc_now(),
        "phase": "R",
        "source_run_dir": str(source_run_dir),
        "phaseQ_decision": q_result,
        "phaseQ_manifest": q_manifest,
        "freeze_global": freeze_global,
        "phase_freeze_lock_validation": r_freeze,
        "survivor_count": int(len(pool)),
        "scenarios": scenarios,
        "scenario_bundle_build": scen_build_meta,
        "robust_count": int(robust_count),
        "brittle_rate": float(brittle_rate) if np.isfinite(brittle_rate) else None,
        "decision": r_result,
        "duration_sec": float(time.time() - t0),
        "command_log": command_log,
        "code_modified": "NO",
    }
    json_dump(r_dir / "phaseR_run_manifest.json", man)

    lines = []
    lines.append("# Phase R Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Decision: **{r_result}**")
    lines.append(f"- Survivor candidates tested: {len(pool)}")
    lines.append(f"- Robust candidates: {robust_count}")
    lines.append(f"- Brittleness rate: {brittle_rate:.4f}" if np.isfinite(brittle_rate) else "- Brittleness rate: nan")
    lines.append("")
    lines.append("Stress matrix results saved in `phaseR_stress_matrix_results.csv`.")
    (r_dir / "phaseR_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    (r_dir / "decision_phaseR.md").write_text(
        f"# Decision Phase R\n\n- Generated UTC: {utc_now()}\n- Decision: **{r_result}**\n- robust_count: {robust_count}\n- survivor_count: {len(pool)}\n",
        encoding="utf-8",
    )

    if r_result == "R_FAIL":
        root = []
        root.append("# Phase R Failure Root Cause")
        root.append("")
        cls = "A" if robust_count == 0 else "C"
        root.append(f"- Decision: R_FAIL")
        root.append(f"- Root-cause class: {cls}")
        root.append(f"- robust_count: {robust_count}")
        root.append(f"- brittle_rate: {brittle_rate:.4f}" if np.isfinite(brittle_rate) else "- brittle_rate: nan")
        (r_dir / "phaseR_failure_root_cause.md").write_text("\n".join(root).strip() + "\n", encoding="utf-8")

        hyp = []
        hyp.append("# Phase R Repair Hypotheses")
        hyp.append("")
        hyp.append("- Primary hypothesis: cost-sensitive edge is too thin and collapses under realistic stress overlays.")
        hyp.append("- Recommended next: objective reweighting toward robustness retention and stronger cost-shock penalties before further GA compute.")
        (r_dir / "phaseR_repair_hypotheses.md").write_text("\n".join(hyp).strip() + "\n", encoding="utf-8")
        return r_result, man, None

    # survivors for Phase S
    s_pool = score.sort_values(["robust_pass", "baseline_delta_expectancy", "baseline_exec_expectancy"], ascending=[False, False, False]).reset_index(drop=True)
    return r_result, man, s_pool


def phase_s(
    *,
    parent_dir: Path,
    source_run_dir: Path,
    q_result: str,
    r_result: str,
    r_manifest: Dict[str, Any],
    s_pool: pd.DataFrame,
    freeze_global: Dict[str, Any],
    command_log: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    s_dir = parent_dir / "phaseS"
    t0 = time.time()

    if s_pool is None or s_pool.empty:
        raise RuntimeError("Phase S invoked without candidate pool")

    # Pick primary + backup with functional distinction.
    primary = s_pool.iloc[0].copy()
    backup = None
    for _, r in s_pool.iloc[1:].iterrows():
        if str(r["genome_hash"]) != str(primary["genome_hash"]):
            # require some diversity in baseline behavior.
            if (
                abs(float(r["baseline_taker_share"]) - float(primary["baseline_taker_share"])) >= 0.01
                or abs(float(r["baseline_p95_delay"]) - float(primary["baseline_p95_delay"])) >= 1.0
                or abs(float(r["baseline_maxdd_improve"]) - float(primary["baseline_maxdd_improve"])) >= 0.02
            ):
                backup = r.copy()
                break
    if backup is None and len(s_pool) > 1:
        backup = s_pool.iloc[1].copy()

    sel_rows = []
    for role, r in [("primary", primary), ("backup", backup)]:
        if r is None:
            continue
        sel_rows.append(
            {
                "role": role,
                "genome_hash": str(r["genome_hash"]),
                "baseline_exec_expectancy": float(r["baseline_exec_expectancy"]),
                "baseline_delta_expectancy": float(r["baseline_delta_expectancy"]),
                "baseline_cvar_improve": float(r["baseline_cvar_improve"]),
                "baseline_maxdd_improve": float(r["baseline_maxdd_improve"]),
                "baseline_taker_share": float(r["baseline_taker_share"]),
                "baseline_p95_delay": float(r["baseline_p95_delay"]),
                "sign_retention_delta": float(r.get("sign_retention_delta", np.nan)),
                "risk_retention_ratio": float(r.get("risk_retention_ratio", np.nan)),
                "participation_retention_ratio": float(r.get("participation_retention_ratio", np.nan)),
                "fee_degradation_slope": float(r.get("fee_degradation_slope", np.nan)),
                "robust_pass": int(r.get("robust_pass", 0)),
                "brittle_any": int(r.get("brittle_any", 0)),
                "psr_proxy": float(r.get("psr_proxy", np.nan)),
                "dsr_proxy": float(r.get("dsr_proxy", np.nan)),
            }
        )
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(s_dir / "phaseS_final_candidates.csv", index=False)

    robust_cnt = int(to_num(s_pool.get("robust_pass", 0)).fillna(0).astype(int).sum())
    if robust_cnt == 0:
        s_result = "S_NO_PROMOTION"
    elif r_result == "R_PASS" and q_result == "Q_PASS_STRONG":
        s_result = "S_PROMOTE_PAPER_STRONG"
    else:
        s_result = "S_PROMOTE_PAPER_CAUTION"

    # Candidate cards.
    cards = []
    cards.append("# Phase S Candidate Cards")
    cards.append("")
    for row in sel_rows:
        cards.append(f"## {row['role'].capitalize()} — `{row['genome_hash']}`")
        cards.append("")
        cards.append(f"- baseline_exec_expectancy: {row['baseline_exec_expectancy']:.8f}")
        cards.append(f"- baseline_delta_expectancy_vs_baseline: {row['baseline_delta_expectancy']:.8f}")
        cards.append(f"- baseline_cvar_improve: {row['baseline_cvar_improve']:.6f}")
        cards.append(f"- baseline_maxdd_improve: {row['baseline_maxdd_improve']:.6f}")
        cards.append(f"- baseline_taker_share: {row['baseline_taker_share']:.6f}")
        cards.append(f"- baseline_p95_fill_delay_min: {row['baseline_p95_delay']:.2f}")
        cards.append(f"- sign_retention_delta: {row['sign_retention_delta']:.4f}")
        cards.append(f"- risk_retention_ratio: {row['risk_retention_ratio']:.4f}")
        cards.append(f"- participation_retention_ratio: {row['participation_retention_ratio']:.4f}")
        cards.append(f"- fee_degradation_slope: {row['fee_degradation_slope']:.8f}" if np.isfinite(row['fee_degradation_slope']) else "- fee_degradation_slope: nan")
        cards.append(f"- robust_pass: {row['robust_pass']}")
        cards.append(f"- brittle_any: {row['brittle_any']}")
        cards.append(f"- psr_proxy/dsr_proxy: {row['psr_proxy']:.6f} / {row['dsr_proxy']:.6f}" if np.isfinite(row['psr_proxy']) and np.isfinite(row['dsr_proxy']) else "- psr_proxy/dsr_proxy: nan")
        cards.append("- known_weakness: split-level downside remains possible; do not treat as live-deploy readiness.")
        cards.append("")
    (s_dir / "phaseS_candidate_cards.md").write_text("\n".join(cards).strip() + "\n", encoding="utf-8")

    # Promotion decision.
    promo = []
    promo.append("# Phase S Promotion Decision")
    promo.append("")
    promo.append(f"- Generated UTC: {utc_now()}")
    promo.append(f"- Decision: **{s_result}**")
    promo.append(f"- Q result: {q_result}")
    promo.append(f"- R result: {r_result}")
    promo.append(f"- robust_count: {robust_cnt}")
    promo.append("")
    if s_result.startswith("S_PROMOTE_PAPER"):
        promo.append("- Promotion scope: PAPER/SHADOW only (SOLUSDT).")
        promo.append("- Live deployment: NOT APPROVED.")
    else:
        promo.append("- No promotion approved.")
    (s_dir / "phaseS_promotion_decision.md").write_text("\n".join(promo).strip() + "\n", encoding="utf-8")

    # Monitoring + guardrails.
    mon = []
    mon.append("# Phase S Paper Monitoring Spec")
    mon.append("")
    mon.append("- Scope: SOLUSDT only, paper/shadow mode.")
    mon.append("- Minimum monitoring horizon: 30 calendar days or 300 executed entries (whichever is later).")
    mon.append("- Core KPIs: exec_expectancy_net, delta_expectancy_vs_baseline, cvar_improve_ratio, maxdd_improve_ratio, entry_rate, taker_share, p95_fill_delay.")
    mon.append("- Success condition: delta_expectancy_vs_baseline remains > 0 on rolling 2-week windows and risk improvements remain non-negative.")
    mon.append("- Fail condition: two consecutive rolling windows with delta_expectancy<=0 or risk-improve ratios <=0.")
    mon.append("- Drift checks: taker_share drift > +5pp, p95_fill_delay drift > +20 min, entry_rate drop below 0.90.")
    (s_dir / "phaseS_paper_monitoring_spec.md").write_text("\n".join(mon).strip() + "\n", encoding="utf-8")

    grd = []
    grd.append("# Phase S Guardrails And Rollback")
    grd.append("")
    grd.append("- Hard guardrail: keep production gate definitions unchanged.")
    grd.append("- Immediate rollback triggers (paper mode disable):")
    grd.append("  - taker_share > 0.25")
    grd.append("  - p95_fill_delay > 180 min")
    grd.append("  - delta_expectancy_vs_baseline <= 0 for 2 consecutive review windows")
    grd.append("  - maxdd_improve_ratio <= 0 for 2 consecutive review windows")
    grd.append("- Incident checklist: freeze lock check, data integrity check, execution feed check, config hash check, scenario replay.")
    (s_dir / "phaseS_guardrails_and_rollback.md").write_text("\n".join(grd).strip() + "\n", encoding="utf-8")

    rel = {
        "generated_utc": utc_now(),
        "source_run_dir": str(source_run_dir),
        "phaseQ_result": q_result,
        "phaseR_result": r_result,
        "phaseS_result": s_result,
        "selected_candidates": sel_rows,
        "freeze_global": freeze_global,
        "command_log": command_log,
        "code_modified": "NO",
    }
    json_dump(s_dir / "phaseS_release_manifest.json", rel)

    repro = []
    repro.append("# Phase S Reproducibility Readme")
    repro.append("")
    repro.append(f"- Generated UTC: {utc_now()}")
    repro.append(f"- Source full run: `{source_run_dir}`")
    repro.append(f"- Q result: {q_result}")
    repro.append(f"- R result: {r_result}")
    repro.append(f"- S result: {s_result}")
    repro.append("")
    repro.append("## Frozen Locks")
    repro.append("")
    for k, v in freeze_global.items():
        repro.append(f"- {k}: {v}")
    repro.append("")
    repro.append("## Reproduce")
    repro.append("")
    repro.append("1) Use source full run artifacts and selected candidate hashes from `phaseS_final_candidates.csv`.")
    repro.append("2) Re-run OOS routes from Phase Q using the same scripts and locked files.")
    repro.append("3) Re-run Phase R stress matrix on survivor set.")
    repro.append("4) Verify manifests and hashes match before any paper-mode activation.")
    (s_dir / "phaseS_reproducibility_readme.md").write_text("\n".join(repro).strip() + "\n", encoding="utf-8")

    (s_dir / "decision_phaseS.md").write_text(
        f"# Decision Phase S\n\n- Generated UTC: {utc_now()}\n- Decision: **{s_result}**\n- Q result: {q_result}\n- R result: {r_result}\n- robust_count: {robust_cnt}\n",
        encoding="utf-8",
    )

    if s_result == "S_NO_PROMOTION":
        rej = []
        rej.append("# Phase S Rejection Summary")
        rej.append("")
        rej.append("- Status: NO_DEPLOY / NO_PROMOTION")
        rej.append("- Reason: robustness and/or OOS evidence was insufficient for paper promotion.")
        (s_dir / "phaseS_rejection_summary.md").write_text("\n".join(rej).strip() + "\n", encoding="utf-8")

        nxt = []
        nxt.append("# Next Research Priority")
        nxt.append("")
        nxt.append("- Single next step: objective redesign with explicit split-stability penalties before any further GA compute.")
        (s_dir / "next_research_priority.md").write_text("\n".join(nxt).strip() + "\n", encoding="utf-8")

    man = {
        "generated_utc": utc_now(),
        "phase": "S",
        "source_run_dir": str(source_run_dir),
        "phaseR_manifest": r_manifest,
        "freeze_global": freeze_global,
        "decision": s_result,
        "duration_sec": float(time.time() - t0),
        "command_log": command_log,
        "code_modified": "NO",
    }
    return s_result, man


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_phase_qrs_autorun.py <parent_dir>")
    parent_dir = Path(sys.argv[1]).resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "phaseQ").mkdir(exist_ok=True)
    (parent_dir / "phaseR").mkdir(exist_ok=True)
    (parent_dir / "phaseS").mkdir(exist_ok=True)

    command_log: List[Dict[str, Any]] = []
    command_log.append({"step": "autorun_start", "utc": utc_now(), "cwd": str(PROJECT_ROOT)})

    freeze_global = validate_global_freeze()
    if int(freeze_global.get("fee_hash_match", 0)) != 1 or int(freeze_global.get("metrics_hash_match", 0)) != 1:
        raise SystemExit("Global frozen hash mismatch; aborting")

    source_run_dir, source_manifest, source_df = pick_source_run()
    command_log.append({"step": "selected_source_run", "utc": utc_now(), "source_run": str(source_run_dir)})

    q_result, q_manifest, shortlist, survivors = phase_q(
        parent_dir=parent_dir,
        source_run_dir=source_run_dir,
        source_manifest=source_manifest,
        source_df=source_df,
        freeze_global=freeze_global,
        command_log=command_log,
    )

    r_result = "NOT_RUN"
    r_manifest: Dict[str, Any] = {}
    s_result = "NOT_RUN"

    if q_result == "Q_FAIL":
        # Stop mainline progression.
        pass
    else:
        r_result, r_manifest, s_pool = phase_r(
            parent_dir=parent_dir,
            source_run_dir=source_run_dir,
            source_df=source_df,
            q_result=q_result,
            q_manifest=q_manifest,
            shortlist=shortlist,
            survivors=survivors if survivors is not None else pd.DataFrame(),
            freeze_global=freeze_global,
            command_log=command_log,
        )

        if r_result in {"R_PASS", "R_PASS_FRAGILE"}:
            s_result, s_manifest = phase_s(
                parent_dir=parent_dir,
                source_run_dir=source_run_dir,
                q_result=q_result,
                r_result=r_result,
                r_manifest=r_manifest,
                s_pool=s_pool if s_pool is not None else pd.DataFrame(),
                freeze_global=freeze_global,
                command_log=command_log,
            )
        else:
            s_result = "NOT_RUN"

    summary = {
        "generated_utc": utc_now(),
        "parent_dir": str(parent_dir),
        "source_run_dir": str(source_run_dir),
        "phaseQ_result": q_result,
        "phaseR_result": r_result,
        "phaseS_result": s_result,
        "freeze_global": freeze_global,
        "command_log": command_log,
    }
    json_dump(parent_dir / "autorun_summary.json", summary)
    print(str(parent_dir))
    print(json.dumps({"phaseQ": q_result, "phaseR": r_result, "phaseS": s_result}, sort_keys=True))


if __name__ == "__main__":
    main()
