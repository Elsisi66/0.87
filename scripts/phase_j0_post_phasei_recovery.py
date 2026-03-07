#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_d123_tail_filter as dmod  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "symbol": "SOLUSDT",
    "phase_i_dir": "/root/analysis/0.87/reports/execution_layer/PHASEI_EXECAWARE_1H_GA_EXPANSION_20260224_012237",
    "phase_ae_dir": "/root/analysis/0.87/reports/execution_layer/PHASEAE_SIGNAL_LABELING_20260223_111116",
    "phase_af_dir": "/root/analysis/0.87/reports/execution_layer/PHASEAF_SIZING_PILOT_20260223_120514",
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "primary_exec_hash": "862c940746de0da984862d95",
    "backup_exec_hash": "992bd371689ba3936f3b4d09",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


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


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def markdown_table(df: pd.DataFrame, cols: Iterable[str], n: int = 20) -> str:
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if n > 0:
        x = x.head(n)
    if x.empty:
        return "_(none)_"
    lines: List[str] = []
    headers = list(x.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in x.itertuples(index=False):
        vals: List[str] = []
        for v in r:
            if isinstance(v, float):
                vals.append(f"{v:.8g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def validate_contract(run_dir: Path, seed: int, subset_csv: Path, fee_path: Path, metrics_path: Path) -> Dict[str, Any]:
    for fp in (subset_csv, fee_path, metrics_path):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")
    fee_sha = sha256_file(fee_path)
    met_sha = sha256_file(metrics_path)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"fee sha mismatch: {fee_sha}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"metrics sha mismatch: {met_sha}")
    sig_raw = pd.read_csv(subset_csv)
    sig = ae.ensure_signals_schema(sig_raw)
    if sig.empty:
        raise RuntimeError("frozen subset is empty")
    lock_args = ae.build_args(signals_csv=subset_csv, seed=int(seed))
    lock_args.allow_freeze_hash_mismatch = 0
    lock = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=run_dir)
    if int(lock.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze_lock_pass != 1")
    return {
        "generated_utc": utc_now(),
        "subset_rows": int(len(sig)),
        "fee_sha256": fee_sha,
        "metrics_sha256": met_sha,
        "freeze_lock_validation": lock,
    }


def _required_phase_i_files(phase_i_dir: Path) -> Dict[str, Path]:
    req = {
        "phaseI2_ga_results": phase_i_dir / "phaseI2_ga_results.csv",
        "phaseI2_duplicate_stats": phase_i_dir / "phaseI2_duplicate_stats.csv",
        "phaseI2_effective_trials_summary": phase_i_dir / "phaseI2_effective_trials_summary.md",
        "phaseI3_route_checks": phase_i_dir / "phaseI3_route_checks.csv",
        "phaseI3_split_stability": phase_i_dir / "phaseI3_split_stability.csv",
        "phaseI3_stress_matrix": phase_i_dir / "phaseI3_stress_matrix.csv",
        "phaseI3_bootstrap_summary": phase_i_dir / "phaseI3_bootstrap_summary.csv",
        "phaseI3_top_survivors_report": phase_i_dir / "phaseI3_top_survivors_report.md",
        "phaseI_frontier_comparison_vs_H": phase_i_dir / "phaseI_frontier_comparison_vs_H.csv",
        "phaseI_decision_next_step": phase_i_dir / "phaseI_decision_next_step.md",
    }
    miss = [k for k, p in req.items() if not p.exists()]
    if miss:
        raise FileNotFoundError(f"Missing Phase I artifacts: {miss}")
    return req


def phase_j01_forensics(phase_i_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    req = _required_phase_i_files(phase_i_dir)
    i2 = pd.read_csv(req["phaseI2_ga_results"])
    i2_dup = pd.read_csv(req["phaseI2_duplicate_stats"])
    route = pd.read_csv(req["phaseI3_route_checks"])
    split = pd.read_csv(req["phaseI3_split_stability"])
    stress = pd.read_csv(req["phaseI3_stress_matrix"])
    boot = pd.read_csv(req["phaseI3_bootstrap_summary"])
    frontier = pd.read_csv(req["phaseI_frontier_comparison_vs_H"])
    decision_text = req["phaseI_decision_next_step"].read_text(encoding="utf-8")

    tested_ids = sorted(set(route["candidate_id"].astype(str)))
    tested = i2[i2["candidate_id"].astype(str).isin(tested_ids)].copy()
    if tested.empty:
        raise RuntimeError("No overlap between I2 and I3 candidate ids")

    route_agg = (
        route.groupby("candidate_id", dropna=False)
        .agg(
            route_pass_rate=("route_pass", "mean"),
            route_fail_count=("route_pass", lambda s: int(np.sum(to_num(s) != 1))),
            min_route_delta=("min_delta_expectancy_vs_base", "min"),
            min_route_cvar=("min_cvar_improve_ratio_vs_base", "min"),
            min_route_dd=("min_maxdd_improve_ratio_vs_base", "min"),
            min_route_entries=("min_entries_valid", "min"),
        )
        .reset_index()
    )
    split_agg = (
        split.groupby("candidate_id", dropna=False)
        .agg(
            min_subperiod_delta=("min_subperiod_delta", "min"),
            min_subperiod_cvar=("min_subperiod_cvar_improve", "min"),
            split_fail_count=("split_stability_pass", lambda s: int(np.sum(to_num(s) != 1))),
        )
        .reset_index()
    )
    stress_agg = (
        stress.groupby("candidate_id", dropna=False)
        .agg(
            stress_pass_rate=("scenario_pass", "mean"),
            stress_fail_count=("scenario_pass", lambda s: int(np.sum(to_num(s) != 1))),
            min_stress_delta=("min_delta_expectancy_vs_base", "min"),
            min_stress_cvar=("min_cvar_improve_ratio_vs_base", "min"),
            min_stress_dd=("min_maxdd_improve_ratio_vs_base", "min"),
            min_kept_entries_pct=("min_filter_kept_entries_pct", "min"),
        )
        .reset_index()
    )
    csum = (
        tested.merge(route_agg, on="candidate_id", how="left")
        .merge(split_agg, on="candidate_id", how="left")
        .merge(stress_agg, on="candidate_id", how="left")
        .merge(boot, on="candidate_id", how="left")
    )
    csum["fail_route"] = (to_num(csum["route_pass_rate"]) < 1.0).astype(int)
    csum["fail_split"] = ((to_num(csum["min_subperiod_delta"]) <= 0.0) | (to_num(csum["min_subperiod_cvar"]) < 0.0)).astype(int)
    csum["fail_stress"] = (to_num(csum["stress_pass_rate"]) < 0.60).astype(int)
    csum["fail_bootstrap"] = (to_num(csum["bootstrap_pass_rate"]) < 0.10).astype(int)
    csum["fail_tail_sign"] = (to_num(csum["cvar_improve_ratio"]) < 0.0).astype(int)
    csum["fail_dd_sign"] = (to_num(csum["maxdd_improve_ratio"]) <= 0.0).astype(int)

    # Failure mode table: scenario/route/split concentration + candidate-level modes.
    rows: List[Dict[str, Any]] = []
    mode_cols = ["fail_route", "fail_split", "fail_stress", "fail_bootstrap", "fail_tail_sign", "fail_dd_sign"]
    n_c = len(csum)
    for m in mode_cols:
        cnt = int(to_num(csum[m]).sum())
        rows.append(
            {
                "scope": "candidate_mode",
                "scope_id": m,
                "fail_count": cnt,
                "fail_pct": safe_div(float(cnt), float(max(1, n_c))),
                "support_n": int(n_c),
            }
        )

    for rid, g in route.groupby("route_id", dropna=False):
        cnt = int(np.sum(to_num(g["route_pass"]) != 1))
        rows.append(
            {
                "scope": "route",
                "scope_id": str(rid),
                "fail_count": cnt,
                "fail_pct": safe_div(float(cnt), float(max(1, len(g)))),
                "support_n": int(len(g)),
            }
        )
    for sid, g in stress.groupby("scenario_id", dropna=False):
        cnt = int(np.sum(to_num(g["scenario_pass"]) != 1))
        rows.append(
            {
                "scope": "stress_scenario",
                "scope_id": str(sid),
                "fail_count": cnt,
                "fail_pct": safe_div(float(cnt), float(max(1, len(g)))),
                "support_n": int(len(g)),
            }
        )
    for rid, g in split.groupby("route_id", dropna=False):
        cnt = int(np.sum(to_num(g["split_stability_pass"]) != 1))
        rows.append(
            {
                "scope": "split_route",
                "scope_id": str(rid),
                "fail_count": cnt,
                "fail_pct": safe_div(float(cnt), float(max(1, len(g)))),
                "support_n": int(len(g)),
            }
        )
    fail_breakdown = pd.DataFrame(rows).sort_values(["scope", "fail_pct", "fail_count"], ascending=[True, False, False]).reset_index(drop=True)
    fail_breakdown.to_csv(out_dir / "phaseJ01_failure_mode_breakdown.csv", index=False)

    # Lucky-point signature catalog.
    # Use robustness-tested candidate summary as the base table (avoids duplicate-column suffix issues).
    sig = csum.copy()
    frontier_i = frontier[frontier["phase"].astype(str) == "I"].copy()
    if "near_winner_neighborhood" in frontier_i.columns:
        sig = sig.merge(frontier_i[["candidate_id", "near_winner_neighborhood"]], on="candidate_id", how="left")
    if "near_winner_neighborhood" not in sig.columns:
        sig["near_winner_neighborhood"] = False
    sig["near_h0313_clone"] = sig["seed_origin"].astype(str).str.contains("H0313|near_H0313|crossover", regex=True, na=False)
    sig["sig_in_sample_outlier"] = (
        (to_num(sig["OJ2"]) >= to_num(sig["OJ2"]).quantile(0.90))
        & (to_num(sig["delta_expectancy_vs_exec_baseline"]) > 0.00070)
    ).astype(int)
    sig["sig_route_fragile"] = (to_num(sig["route_pass_rate"]) < 1.0).astype(int)
    sig["sig_split_fragile"] = ((to_num(sig["min_subperiod_delta"]) <= 0.0) | (to_num(sig["min_subperiod_cvar"]) < 0.0)).astype(int)
    sig["sig_stress_fragile"] = (to_num(sig["stress_pass_rate"]) < 0.60).astype(int)
    sig["sig_bootstrap_weak"] = (to_num(sig["bootstrap_pass_rate"]) < 0.10).astype(int)
    sig["sig_negative_tail_metric"] = (to_num(sig["cvar_improve_ratio"]) < 0.0).astype(int)
    sig["lucky_point_score"] = (
        to_num(sig["sig_in_sample_outlier"]).fillna(0.0)
        + to_num(sig["sig_route_fragile"]).fillna(0.0)
        + to_num(sig["sig_split_fragile"]).fillna(0.0)
        + to_num(sig["sig_stress_fragile"]).fillna(0.0)
        + to_num(sig["sig_bootstrap_weak"]).fillna(0.0)
        + to_num(sig["sig_negative_tail_metric"]).fillna(0.0)
    )
    sig = sig.sort_values(["lucky_point_score", "OJ2"], ascending=[False, False]).reset_index(drop=True)
    sig_cols = [
        "candidate_id",
        "seed_origin",
        "near_h0313_clone",
        "OJ2",
        "delta_expectancy_vs_exec_baseline",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "entries_valid",
        "entry_rate",
        "route_pass_rate",
        "stress_pass_rate",
        "bootstrap_pass_rate",
        "min_subperiod_delta",
        "min_subperiod_cvar",
        "lucky_point_score",
        "sig_in_sample_outlier",
        "sig_route_fragile",
        "sig_split_fragile",
        "sig_stress_fragile",
        "sig_bootstrap_weak",
        "sig_negative_tail_metric",
    ]
    sig[sig_cols].to_csv(out_dir / "phaseJ01_lucky_point_signature_catalog.csv", index=False)

    # Future constraints draft.
    cons = [
        "# Phase J01 Future Robustness Constraints Draft",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Source: Phase I failure-forensics (lucky-point frontier collapse).",
        "",
        "## Do-Not-Chase Signatures",
        "",
        "1. High OJ2 + high delta candidates with route_pass_rate < 1.0.",
        "2. Candidates with min_subperiod_delta <= 0 and/or min_subperiod_cvar < 0 despite positive aggregate metrics.",
        "3. Candidates with bootstrap_pass_rate < 0.10 (weak perturbation confidence).",
        "4. Candidate families with negative cvar_improve_ratio under route2/stress scenarios.",
        "5. High-correlation duplicate clusters with near-identical metric shape (effective_trials collapse).",
        "",
        "## Draft Robustness-First Constraints/Objectives",
        "",
        "- Add route fragility penalty: penalty if route_pass_rate < 1.0.",
        "- Add split stability floor: require min_subperiod_delta > 0 and min_subperiod_cvar >= 0.",
        "- Add perturb confidence floor: bootstrap_pass_rate target >= 0.10 (pilot stage).",
        "- Add duplicate-collapse penalty tied to correlation-adjusted effective trials.",
        "- Downweight high in-sample gains when accompanied by negative tail-risk signs.",
    ]
    write_text(out_dir / "phaseJ01_future_robustness_constraints_draft.md", "\n".join(cons))

    top_causes = fail_breakdown[(fail_breakdown["scope"] == "candidate_mode")].copy()
    route_hotspots = fail_breakdown[(fail_breakdown["scope"] == "route")].copy()
    stress_hotspots = fail_breakdown[(fail_breakdown["scope"] == "stress_scenario")].copy()
    dups = i2_dup.head(1).to_dict(orient="records")[0] if not i2_dup.empty else {}
    rep = [
        "# Phase J01 Phase I Failure Forensics Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Source Phase I dir: `{phase_i_dir}`",
        f"- Robustness-tested candidates: `{len(csum)}`",
        f"- Decision excerpt: `{decision_text.splitlines()[3] if len(decision_text.splitlines()) > 3 else decision_text.splitlines()[0]}`",
        f"- Duplicate ratio among valid (I2): `{float(dups.get('duplicate_ratio_among_valid', float('nan'))):.6f}`",
        f"- Effective trials corr-adjusted (I2): `{float(dups.get('effective_trials_corr_adjusted', float('nan'))):.6f}`",
        f"- Avg abs metric corr (I2): `{float(dups.get('avg_abs_metric_corr', float('nan'))):.6f}`",
        "",
        "## Top Candidate-Level Failure Modes",
        "",
        markdown_table(top_causes, ["scope_id", "fail_count", "fail_pct", "support_n"], n=12),
        "",
        "## Route Failure Concentration",
        "",
        markdown_table(route_hotspots, ["scope_id", "fail_count", "fail_pct", "support_n"], n=12),
        "",
        "## Stress Scenario Failure Concentration",
        "",
        markdown_table(stress_hotspots, ["scope_id", "fail_count", "fail_pct", "support_n"], n=20),
        "",
        "## Lucky-Point Signatures (head)",
        "",
        markdown_table(sig[sig_cols], sig_cols, n=20),
    ]
    write_text(out_dir / "phaseJ01_phaseI_failure_forensics_report.md", "\n".join(rep))
    return fail_breakdown, sig[sig_cols]


def _flat_policy() -> Dict[str, Any]:
    return {
        "policy_id": "baseline_flat",
        "family": "streak_control",
        "params": {"k_streak": 100000, "size_down": 1.0, "size_up": 1.0},
        "weights": {},
        "size_bounds": [0.5, 1.5],
    }


def _streak_stats_from_unweighted(df: pd.DataFrame) -> Dict[str, int]:
    x = af.parse_entry_rows(df)
    valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
    t = x.loc[valid].copy().sort_values(["entry_time_utc", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    pnl = to_num(t["pnl_net_trade_notional_dec"]).to_numpy(dtype=float)
    losses = pnl < 0
    lens: List[int] = []
    i = 0
    while i < len(losses):
        if losses[i]:
            s = i
            while i + 1 < len(losses) and losses[i + 1]:
                i += 1
            lens.append(i - s + 1)
        i += 1
    arr = np.asarray(lens, dtype=int) if lens else np.array([], dtype=int)
    return {
        "max_consecutive_losses": int(arr.max()) if arr.size else 0,
        "streak_ge3_count": int(np.sum(arr >= 3)) if arr.size else 0,
        "streak_ge5_count": int(np.sum(arr >= 5)) if arr.size else 0,
        "streak_ge10_count": int(np.sum(arr >= 10)) if arr.size else 0,
    }


def phase_j02_baseline_rebuild(run_dir: Path, seed: int, subset_csv: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sig_in = ae.ensure_signals_schema(pd.read_csv(subset_csv))
    exec_pair = ae.load_exec_pair(PROJECT_ROOT / "reports" / "execution_layer")
    if exec_pair["E1"]["genome_hash"] != LOCKED["primary_exec_hash"]:
        raise RuntimeError("E1 hash mismatch vs locked primary")
    if exec_pair["E2"]["genome_hash"] != LOCKED["backup_exec_hash"]:
        raise RuntimeError("E2 hash mismatch vs locked backup")

    all_rows: List[pd.DataFrame] = []
    base_rows: List[Dict[str, Any]] = []
    route_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ix, choice_id in enumerate(["E1", "E2"]):
        g = copy.deepcopy(exec_pair[choice_id]["genome"])
        cache_dir = run_dir / "phaseJ02_route_cache" / choice_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        d_by_route = dmod.evaluate_baseline_routes(
            run_dir=cache_dir,
            sig_in=sig_in,
            genome=g,
            seed=int(seed) + 100 * (ix + 1),
        )
        route_cache[choice_id] = d_by_route
        for rid, d in d_by_route.items():
            z = d.copy()
            z["route_id"] = str(rid)
            z["exec_choice_id"] = str(choice_id)
            z["exec_genome_hash"] = str(exec_pair[choice_id]["genome_hash"])
            z["candidate_role"] = "primary" if choice_id == "E1" else "backup"
            all_rows.append(z)

            flat = _flat_policy()
            qstats = af.build_quantile_stats(af.parse_entry_rows(z))
            x = af.parse_entry_rows(z)
            valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
            base_unweighted = {
                "entries_valid": int(valid.sum()),
                "entry_rate": float(valid.sum() / max(1, len(x))),
                "taker_share": float(np.nanmean(to_num(x.loc[valid, "taker_flag"]))) if int(valid.sum()) > 0 else float("nan"),
                "p95_fill_delay_min": float(np.nanquantile(to_num(x.loc[valid, "fill_delay_min"]).dropna(), 0.95))
                if int(valid.sum()) > 0 and to_num(x.loc[valid, "fill_delay_min"]).dropna().size > 0
                else float("nan"),
            }
            met, _ = af.evaluate_sizing_policy_on_dataset(
                df_full=z,
                policy=flat,
                qstats=qstats,
                baseline_unweighted=base_unweighted,
                rng_seed=int(seed) + 17,
            )
            cl = _streak_stats_from_unweighted(z)
            base_rows.append(
                {
                    "exec_choice_id": choice_id,
                    "route_id": rid,
                    "exec_expectancy_net": float(met["exec_expectancy_net_weighted"]),
                    "exec_cvar_5": float(met["cvar_5_weighted"]),
                    "exec_max_drawdown": float(met["max_drawdown_weighted"]),
                    "entries_valid": int(met["entries_valid"]),
                    "entry_rate": float(met["entry_rate"]),
                    "taker_share": float(met["taker_share"]),
                    "p95_fill_delay_min": float(met["p95_fill_delay_min"]),
                    "min_split_expectancy_net": float(met["min_split_expectancy_net_weighted"]),
                    **cl,
                }
            )
    tf = pd.concat(all_rows, axis=0, ignore_index=True)
    base_df = pd.DataFrame(base_rows).sort_values(["exec_choice_id", "route_id"]).reset_index(drop=True)
    tf.to_parquet(run_dir / "phaseJ02_trade_feature_table.parquet", index=False)

    dq = []
    req_cols = [
        "signal_id",
        "signal_time_utc",
        "entry_for_labels",
        "entry_time_utc",
        "pnl_net_trade_notional_dec",
        "split_id",
        "prior_loss_streak_len",
        "prior_rolling_tail_count_20",
        "prior_rolling_loss_rate_5",
        "pre3m_close_to_high_dist_bps",
        "pre3m_realized_vol_12",
        "pre3m_wick_ratio",
        "route_id",
        "exec_choice_id",
    ]
    for c in req_cols:
        dq.append({"column": c, "missing_rate": float(1.0 - np.mean(tf[c].notna())) if c in tf.columns else 1.0})
    dqr = pd.DataFrame(dq).sort_values("missing_rate", ascending=False).reset_index(drop=True)
    rep = [
        "# Phase J02 Data Quality Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Trade-feature rows: `{len(tf)}`",
        f"- Unique signals: `{tf['signal_id'].astype(str).nunique()}`",
        f"- Routes: `{sorted(tf['route_id'].astype(str).unique().tolist())}`",
        f"- Exec choices: `{sorted(tf['exec_choice_id'].astype(str).unique().tolist())}`",
        "",
        "## Baseline by route/choice",
        "",
        markdown_table(
            base_df,
            [
                "exec_choice_id",
                "route_id",
                "exec_expectancy_net",
                "exec_cvar_5",
                "exec_max_drawdown",
                "entries_valid",
                "entry_rate",
                "taker_share",
                "p95_fill_delay_min",
                "max_consecutive_losses",
                "streak_ge5_count",
                "streak_ge10_count",
            ],
            n=20,
        ),
        "",
        "## Missingness",
        "",
        markdown_table(dqr, ["column", "missing_rate"], n=30),
        "",
        "## Join/Leakage checks",
        "",
        "- Timestamp columns normalized via existing validated parsers (`phase_ae_signal_labeling` + `phase_af_ah_sizing_autorun`).",
        "- Prior-state features are inherited from validated AE pipeline and remain pre-entry/prior-only.",
    ]
    write_text(run_dir / "phaseJ02_data_quality_report.md", "\n".join(rep))

    repro = {
        "generated_utc": utc_now(),
        "primary_exec_hash": str(exec_pair["E1"]["genome_hash"]),
        "backup_exec_hash": str(exec_pair["E2"]["genome_hash"]),
        "routes": sorted(tf["route_id"].astype(str).unique().tolist()),
        "trade_feature_rows": int(len(tf)),
        "baseline_rows": int(len(base_df)),
        "baseline_metrics_by_route_choice": base_df.to_dict(orient="records"),
    }
    json_dump(run_dir / "phaseJ02_baseline_reproduction_check.json", repro)
    return tf, {"baseline": base_df, "route_cache": route_cache}


def phase_j03_write_specs(run_dir: Path) -> None:
    spec = [
        "# Phase J03 Soft Sizing Family Spec",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Objective: reduce drawdown/tail/clustering burden without participation starvation.",
        "- Scope: sizing/risk modulation only on top of proven execution winners (E1/E2).",
        "- Hard filters are excluded as primary lever.",
        "",
        "## Families",
        "",
        "A) `size_step_by_risk_score` (piecewise monotonic).",
        "B) `size_linear_by_risk_score` (smooth monotonic).",
        "C) `size_cap_after_loss_streak` (state throttle).",
        "D) `size_cap_after_tail_count` (state throttle).",
        "E) `cooldown-lite + reduced size` (approximated as state-trigger downsize, no skip).",
        "F) `hybrid_stateaware` (risk score + state interactions).",
        "G) Optional extreme-risk tightening reserved for future engine support (not activated here).",
        "",
        "## Design Guards",
        "",
        "- No entry/exit mechanics change.",
        "- No trade filtering by default (size floor > 0).",
        "- Mean-size normalization active.",
        "- Same fee/slippage and hard-gate contract.",
    ]
    write_text(run_dir / "phaseJ03_sizing_family_spec.md", "\n".join(spec))

    bounds = [
        "family_bounds:",
        "  size_step_by_risk_score:",
        "    t1: [0.40, 0.60]",
        "    t2: [0.65, 0.85]",
        "    s_hi: [1.00, 1.15]",
        "    s_mid: [0.85, 1.00]",
        "    s_lo: [0.65, 0.88]",
        "  size_linear_by_risk_score:",
        "    slope: [0.25, 1.10]",
        "  size_cap_after_loss_streak:",
        "    k_streak: [2, 5]",
        "    s_down: [0.70, 0.92]",
        "    s_up: [1.00, 1.12]",
        "  size_cap_after_tail_count:",
        "    k_tail20: [6, 12]",
        "    s_down: [0.70, 0.92]",
        "    s_up: [1.00, 1.12]",
        "  hybrid_stateaware:",
        "    slope: [0.30, 1.00]",
        "    k_streak: [2, 5]",
        "    k_tail20: [6, 12]",
        "    penalty_mult: [0.65, 0.95]",
        "    recovery_bonus: [0.00, 0.08]",
        "global_constraints:",
        "  size_floor: 0.50",
        "  size_cap: 1.50",
        "  mean_size_normalization: true",
        "  hard_gates_unchanged: true",
        "  no_entry_exit_mechanics_change: true",
    ]
    write_text(run_dir / "phaseJ03_param_bounds.yaml", "\n".join(bounds))

    mapping = [
        "# Phase J03 Mapping To Existing Engine",
        "",
        "## Reused modules",
        "- Route generation: `scripts.phase_d123_tail_filter.evaluate_baseline_routes()`",
        "- Feature transforms and policy sizing: `scripts.phase_af_ah_sizing_autorun.py`",
        "- Contract lock: `src.execution.ga_exec_3m_opt._validate_and_lock_frozen_artifacts()`",
        "",
        "## Sizing-only scope",
        "- All policies map to sizing functions already available (`step`, `smooth`, `streak`, `hybrid`, `ae_s1_anchor`, `streak_control`).",
        "- `cooldown-lite` is represented as state-triggered size-down (no skip) because hard skip paths were previously fragile.",
        "",
        "## Out-of-scope in this branch",
        "- Combined 1h+3m GA.",
        "- Hard-gate changes.",
        "- Execution entry/exit mechanic modifications.",
    ]
    write_text(run_dir / "phaseJ03_mapping_to_existing_engine.md", "\n".join(mapping))


def _policy_template(pid: str, family: str, weights: Dict[str, float], params: Dict[str, Any]) -> Dict[str, Any]:
    return {"policy_id": pid, "family": family, "weights": weights, "params": params, "size_bounds": [0.50, 1.50]}


def build_soft_policy_set(qstats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    w_ae = {"w_streak": 1.30, "w_tail20": 1.20, "w_closehigh": 1.20, "w_vol": 0.90, "w_wick": 0.75, "w_int1": 1.00, "w_int2": 1.00}
    w_bal = {"w_streak": 1.10, "w_tail20": 1.00, "w_closehigh": 1.05, "w_vol": 1.00, "w_wick": 0.80, "w_int1": 0.90, "w_int2": 0.90}
    pols: List[Dict[str, Any]] = []
    pols.append(_flat_policy())
    pols.append(af.s1_policy(qstats))
    pols.append(_policy_template("step_mild", "step", w_bal, {"t1": 0.55, "t2": 0.78, "s_hi": 1.06, "s_mid": 0.94, "s_lo": 0.84}))
    pols.append(_policy_template("step_mid", "step", w_ae, {"t1": 0.50, "t2": 0.74, "s_hi": 1.08, "s_mid": 0.90, "s_lo": 0.78}))
    pols.append(_policy_template("step_aggr", "step", w_ae, {"t1": 0.46, "t2": 0.70, "s_hi": 1.10, "s_mid": 0.88, "s_lo": 0.72}))
    pols.append(_policy_template("smooth_mild", "smooth", w_bal, {"slope": 0.35}))
    pols.append(_policy_template("smooth_mid", "smooth", w_ae, {"slope": 0.55}))
    pols.append(_policy_template("smooth_aggr", "smooth", w_ae, {"slope": 0.85}))
    pols.append(_policy_template("streak_cap3", "streak", w_bal, {"k_streak": 3, "k_tail20": 99, "s_up": 1.04, "s_down": 0.82}))
    pols.append(_policy_template("streak_cap4", "streak", w_bal, {"k_streak": 4, "k_tail20": 99, "s_up": 1.05, "s_down": 0.86}))
    pols.append(_policy_template("tail_cap8", "streak", w_bal, {"k_streak": 99, "k_tail20": 8, "s_up": 1.05, "s_down": 0.82}))
    pols.append(_policy_template("tail_cap10", "streak", w_bal, {"k_streak": 99, "k_tail20": 10, "s_up": 1.05, "s_down": 0.86}))
    pols.append(_policy_template("cooldown_lite_proxy1", "streak_control", {}, {"k_streak": 2, "size_down": 0.86, "size_up": 1.03}))
    pols.append(_policy_template("cooldown_lite_proxy2", "streak_control", {}, {"k_streak": 3, "size_down": 0.82, "size_up": 1.04}))
    pols.append(_policy_template("hybrid_v1", "hybrid", w_ae, {"slope": 0.45, "k_streak": 3, "k_tail20": 8, "penalty_mult": 0.82, "recovery_bonus": 0.03}))
    pols.append(_policy_template("hybrid_v2", "hybrid", w_ae, {"slope": 0.60, "k_streak": 3, "k_tail20": 8, "penalty_mult": 0.78, "recovery_bonus": 0.04}))
    pols.append(_policy_template("hybrid_v3", "hybrid", w_bal, {"slope": 0.50, "k_streak": 2, "k_tail20": 7, "penalty_mult": 0.80, "recovery_bonus": 0.05}))
    pols.append(_policy_template("hybrid_v4", "hybrid", w_bal, {"slope": 0.70, "k_streak": 2, "k_tail20": 6, "penalty_mult": 0.74, "recovery_bonus": 0.06}))
    return pols


def _variant_hash(policy: Dict[str, Any]) -> str:
    txt = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def phase_j04_ablation(
    run_dir: Path,
    tf: pd.DataFrame,
    route_cache: Dict[str, Dict[str, pd.DataFrame]],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    qstats = af.build_quantile_stats(af.parse_entry_rows(tf))
    policies = build_soft_policy_set(qstats)

    ds_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    base_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cluster_ref: Dict[Tuple[str, str], Dict[str, int]] = {}
    flat = _flat_policy()
    for cid, by_route in route_cache.items():
        for rid, d in by_route.items():
            key = (str(cid), str(rid))
            ds_map[key] = d.copy()
            x = af.parse_entry_rows(d)
            valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
            bu = {
                "entries_valid": int(valid.sum()),
                "entry_rate": float(valid.sum() / max(1, len(x))),
                "taker_share": float(np.nanmean(to_num(x.loc[valid, "taker_flag"]))) if int(valid.sum()) > 0 else float("nan"),
                "p95_fill_delay_min": float(np.nanquantile(to_num(x.loc[valid, "fill_delay_min"]).dropna(), 0.95))
                if int(valid.sum()) > 0 and to_num(x.loc[valid, "fill_delay_min"]).dropna().size > 0
                else float("nan"),
            }
            m_flat, _ = af.evaluate_sizing_policy_on_dataset(
                df_full=d,
                policy=flat,
                qstats=qstats,
                baseline_unweighted=bu,
                rng_seed=int(seed) + 101,
            )
            base_metrics[key] = m_flat
            cluster_ref[key] = _streak_stats_from_unweighted(d)

    by_dataset_rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []
    for p_ix, pol in enumerate(policies):
        rid_rows: List[Dict[str, Any]] = []
        for d_ix, (key, dset) in enumerate(ds_map.items()):
            cid, rid = key
            bu = {
                "entries_valid": int(base_metrics[key]["entries_valid"]),
                "entry_rate": float(base_metrics[key]["entry_rate"]),
                "taker_share": float(base_metrics[key]["taker_share"]),
                "p95_fill_delay_min": float(base_metrics[key]["p95_fill_delay_min"]),
            }
            met, _ = af.evaluate_sizing_policy_on_dataset(
                df_full=dset,
                policy=pol,
                qstats=qstats,
                baseline_unweighted=bu,
                rng_seed=int(seed) + 1000 + 37 * p_ix + d_ix,
            )
            b = base_metrics[key]
            delta = float(met["exec_expectancy_net_weighted"] - b["exec_expectancy_net_weighted"])
            cvar_imp = safe_div(abs(float(b["cvar_5_weighted"])) - abs(float(met["cvar_5_weighted"])), abs(float(b["cvar_5_weighted"])))
            dd_imp = safe_div(abs(float(b["max_drawdown_weighted"])) - abs(float(met["max_drawdown_weighted"])), abs(float(b["max_drawdown_weighted"])))
            row = {
                "variant_id": str(pol["policy_id"]),
                "variant_hash": _variant_hash(pol),
                "variant_family": str(pol["family"]),
                "exec_choice_id": str(cid),
                "route_id": str(rid),
                "exec_expectancy_net": float(met["exec_expectancy_net_weighted"]),
                "delta_expectancy_vs_baseline": float(delta),
                "cvar_improve_ratio": float(cvar_imp),
                "maxdd_improve_ratio": float(dd_imp),
                "min_split_expectancy_net": float(met["min_split_expectancy_net_weighted"]),
                "entries_valid": int(met["entries_valid"]),
                "entry_rate": float(met["entry_rate"]),
                "taker_share": float(met["taker_share"]),
                "p95_fill_delay_min": float(met["p95_fill_delay_min"]),
                "max_consecutive_losses": int(cluster_ref[key]["max_consecutive_losses"]),
                "streak_ge3_count": int(cluster_ref[key]["streak_ge3_count"]),
                "streak_ge5_count": int(cluster_ref[key]["streak_ge5_count"]),
                "streak_ge10_count": int(cluster_ref[key]["streak_ge10_count"]),
            }
            by_dataset_rows.append(row)
            rid_rows.append(row)

        g = pd.DataFrame(rid_rows)
        min_delta = float(to_num(g["delta_expectancy_vs_baseline"]).min())
        min_cvar = float(to_num(g["cvar_improve_ratio"]).min())
        min_dd = float(to_num(g["maxdd_improve_ratio"]).min())
        no_path = int(np.isfinite(min_delta) and np.isfinite(min_cvar) and np.isfinite(min_dd))
        valid = int((min_delta > 0.0) and (min_cvar >= 0.0) and (min_dd > 0.0) and (no_path == 1))
        reasons: List[str] = []
        if min_delta <= 0.0:
            reasons.append("min_delta_expectancy<=0")
        if min_cvar < 0.0:
            reasons.append("min_cvar_improve<0")
        if min_dd <= 0.0:
            reasons.append("min_maxdd_improve<=0")
        if no_path != 1:
            reasons.append("pathology")
        agg_rows.append(
            {
                "variant_id": str(pol["policy_id"]),
                "variant_hash": _variant_hash(pol),
                "variant_family": str(pol["family"]),
                "valid_for_ranking": int(valid),
                "invalid_reason": ";".join(reasons),
                "exec_expectancy_net": float(to_num(g["exec_expectancy_net"]).mean()),
                "delta_expectancy_vs_baseline": float(to_num(g["delta_expectancy_vs_baseline"]).mean()),
                "cvar_improve_ratio": float(to_num(g["cvar_improve_ratio"]).mean()),
                "maxdd_improve_ratio": float(to_num(g["maxdd_improve_ratio"]).mean()),
                "min_split_expectancy_net": float(to_num(g["min_split_expectancy_net"]).min()),
                "entries_valid": int(to_num(g["entries_valid"]).min()),
                "entry_rate": float(to_num(g["entry_rate"]).min()),
                "taker_share": float(to_num(g["taker_share"]).max()),
                "p95_fill_delay_min": float(to_num(g["p95_fill_delay_min"]).max()),
                "max_consecutive_losses": int(to_num(g["max_consecutive_losses"]).max()),
                "streak_ge3_count": int(to_num(g["streak_ge3_count"]).max()),
                "streak_ge5_count": int(to_num(g["streak_ge5_count"]).max()),
                "streak_ge10_count": int(to_num(g["streak_ge10_count"]).max()),
                "min_delta_expectancy_vs_baseline": float(min_delta),
                "min_cvar_improve_ratio": float(min_cvar),
                "min_maxdd_improve_ratio": float(min_dd),
                "no_pathology": int(no_path),
            }
        )

    by_ds = pd.DataFrame(by_dataset_rows).sort_values(["variant_id", "exec_choice_id", "route_id"]).reset_index(drop=True)
    out = pd.DataFrame(agg_rows).sort_values(
        ["valid_for_ranking", "min_delta_expectancy_vs_baseline", "min_cvar_improve_ratio", "min_maxdd_improve_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out.to_csv(run_dir / "phaseJ04_ablation_results.csv", index=False)
    by_ds.to_csv(run_dir / "phaseJ04_ablation_results_by_dataset.csv", index=False)

    invalid_hist: Dict[str, int] = {}
    for _, r in out.iterrows():
        if int(r["valid_for_ranking"]) == 1:
            continue
        for rs in str(r["invalid_reason"]).split(";"):
            rs = rs.strip()
            if not rs:
                continue
            invalid_hist[rs] = int(invalid_hist.get(rs, 0) + 1)
    json_dump(run_dir / "phaseJ04_invalid_reason_histogram.json", invalid_hist)

    rep = [
        "# Phase J04 Controlled Ablation Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Variants tested: `{len(out)}`",
        f"- Route-choice datasets: `{len(ds_map)}`",
        f"- valid_for_ranking count: `{int((to_num(out['valid_for_ranking']) == 1).sum())}`",
        "",
        "## Top variants",
        "",
        markdown_table(
            out,
            [
                "variant_id",
                "variant_family",
                "valid_for_ranking",
                "min_delta_expectancy_vs_baseline",
                "min_cvar_improve_ratio",
                "min_maxdd_improve_ratio",
                "entries_valid",
                "entry_rate",
                "max_consecutive_losses",
                "streak_ge3_count",
                "streak_ge5_count",
                "streak_ge10_count",
                "invalid_reason",
            ],
            n=24,
        ),
    ]
    write_text(run_dir / "phaseJ04_ablation_report.md", "\n".join(rep))
    return out, invalid_hist


def phase_j05_decision(run_dir: Path, ablation: pd.DataFrame) -> Tuple[str, str]:
    x = ablation.copy()
    if x.empty:
        write_text(
            run_dir / "phaseJ05_decision_next_step.md",
            "\n".join(
                [
                    "# Phase J05 Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Classification: **J0_STOP_INFRA**",
                    "- Reason: empty ablation output",
                ]
            ),
        )
        return "J0_STOP_INFRA", "STOP_INFRA"

    base = x[x["variant_id"] == "baseline_flat"].head(1)
    base_entry = float(base.iloc[0]["entry_rate"]) if not base.empty else float("nan")
    base_entries = int(base.iloc[0]["entries_valid"]) if not base.empty else -1

    cand = x[x["variant_id"] != "baseline_flat"].copy()
    pass_df = cand[to_num(cand["valid_for_ranking"]) == 1].copy()
    healthy = pass_df[
        (to_num(pass_df["entry_rate"]) >= (base_entry - 1e-9 if np.isfinite(base_entry) else 0.0))
        & (to_num(pass_df["entries_valid"]) >= (base_entries if base_entries > 0 else 0))
    ].copy()
    improved = healthy[
        (to_num(healthy["maxdd_improve_ratio"]) > 0.05)
        | (to_num(healthy["cvar_improve_ratio"]) > 0.01)
        | (to_num(healthy["maxdd_improve_ratio"]) > 0.0)
    ].copy()
    improved = improved[to_num(improved["delta_expectancy_vs_baseline"]) >= -0.00002]

    if len(improved) >= 2:
        cls = "J0_GO_SIZING_PILOT"
        main = "CONTINUE_READY_FOR_SIZING_PILOT"
        why = ">=2 soft sizing variants valid_for_ranking with risk improvement and healthy participation."
    elif len(improved) == 1:
        cls = "J0_GO_WEAK_SIZING_PILOT"
        main = "CONTINUE_READY_FOR_SIZING_PILOT"
        why = "exactly one viable soft sizing variant; evidence supports only a very small bounded pilot GA."
    else:
        cls = "J0_NO_GO_ABLATION_ONLY"
        main = "STOP_NO_GO"
        why = "no soft sizing variant met acceptable risk/expectancy trade-off under strict gates."

    # Tradeoff frontier file.
    fr = cand.copy()
    fr["frontier_score"] = (
        8.0 * to_num(fr["delta_expectancy_vs_baseline"]).fillna(-1e9)
        + 1.8 * to_num(fr["cvar_improve_ratio"]).fillna(-1e9)
        + 1.8 * to_num(fr["maxdd_improve_ratio"]).fillna(-1e9)
    )
    fr = fr.sort_values(["valid_for_ranking", "frontier_score"], ascending=[False, False]).reset_index(drop=True)
    fr_cols = [
        "variant_id",
        "variant_family",
        "valid_for_ranking",
        "frontier_score",
        "exec_expectancy_net",
        "delta_expectancy_vs_baseline",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "min_split_expectancy_net",
        "entries_valid",
        "entry_rate",
        "max_consecutive_losses",
        "streak_ge3_count",
        "streak_ge5_count",
        "streak_ge10_count",
        "invalid_reason",
    ]
    fr[fr_cols].to_csv(run_dir / "phaseJ05_tradeoff_frontier.csv", index=False)

    ans_q1 = "yes" if len(improved) >= 1 else "no"
    ans_q2 = "yes" if len(healthy) >= 1 else "no"
    ans_q3 = "yes" if cls in {"J0_GO_SIZING_PILOT", "J0_GO_WEAK_SIZING_PILOT"} else "no"
    next_branch = "bounded sizing pilot GA" if ans_q3 == "yes" else "execution-only monitoring + feature/regime relabeling refresh"

    lines = [
        "# Phase J05 Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: **{cls}**",
        f"- Mainline status: **{main}**",
        f"- Decision rationale: {why}",
        "",
        "## Explicit Answers",
        "",
        f"1. Did soft sizing improve drawdown/clustering without breaking the edge? **{ans_q1}**",
        f"2. Is participation preserved vs baseline? **{ans_q2}**",
        f"3. Is a bounded sizing GA pilot justified? **{ans_q3}**",
        f"4. If no, next highest-EV branch: `{next_branch}`",
        "",
        "## Top trade-off variants",
        "",
        markdown_table(fr[fr_cols], fr_cols, n=15),
    ]
    write_text(run_dir / "phaseJ05_decision_next_step.md", "\n".join(lines))

    if cls in {"J0_GO_SIZING_PILOT", "J0_GO_WEAK_SIZING_PILOT"}:
        prompt = [
            "ROLE",
            "You are in bounded sizing pilot GA mode for SOLUSDT on top of frozen execution winners (E1/E2).",
            "",
            "MISSION",
            "Run a very small sizing-only GA around J0 survivors (no execution mechanics change, no 1h GA).",
            "",
            "CONSTRAINTS",
            "1) Hard gates unchanged and frozen hash lock enforced.",
            "2) Candidate budget <= 96 evaluations total.",
            "3) Seed with top J0 tradeoff frontier variants only.",
            "4) Objective: preserve/raise expectancy while improving DD/CVaR and keeping participation healthy.",
            "5) Include route/split/stress-lite check before any promotion claim.",
            "6) Stop NO_GO if robustness collapses.",
        ]
        write_text(run_dir / "ready_to_launch_sizing_pilot_ga_prompt.txt", "\n".join(prompt))
    return cls, main


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase J0 post-Phase-I NO_GO recovery (no combined GA rerun)")
    ap.add_argument("--seed", type=int, default=20260224)
    ap.add_argument("--phase_i_dir", default=LOCKED["phase_i_dir"])
    ap.add_argument("--subset_csv", default=LOCKED["representative_subset_csv"])
    ap.add_argument("--fee_path", default=LOCKED["canonical_fee_model"])
    ap.add_argument("--metrics_path", default=LOCKED["canonical_metrics_definition"])
    args = ap.parse_args()

    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEJ0_POST_PHASEI_RECOVERY_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "source_phase_i": str(Path(args.phase_i_dir).resolve()),
        "source_phase_ae": LOCKED["phase_ae_dir"],
        "source_phase_af": LOCKED["phase_af_dir"],
        "phases": {},
    }

    try:
        lock_obj = validate_contract(
            run_dir=run_dir,
            seed=int(args.seed),
            subset_csv=Path(args.subset_csv).resolve(),
            fee_path=Path(args.fee_path).resolve(),
            metrics_path=Path(args.metrics_path).resolve(),
        )
        manifest["lock_validation"] = lock_obj
    except Exception as e:
        manifest["classification"] = "J0_STOP_INFRA"
        manifest["mainline_status"] = "STOP_INFRA"
        manifest["reason"] = str(e)
        write_text(run_dir / "phaseJ05_decision_next_step.md", f"# Phase J05 Decision\n\n- Classification: **J0_STOP_INFRA**\n- Reason: {e}")
        json_dump(run_dir / "phaseJ0_run_manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "classification": "J0_STOP_INFRA", "error": str(e)}))
        return

    # J0.1
    try:
        fb, sig = phase_j01_forensics(Path(args.phase_i_dir).resolve(), run_dir)
        manifest["phases"]["J0.1"] = {"classification": "PASS", "failure_rows": int(len(fb)), "signature_rows": int(len(sig))}
    except Exception as e:
        manifest["classification"] = "J0_STOP_INFRA"
        manifest["mainline_status"] = "STOP_INFRA"
        manifest["reason"] = f"J0.1 failure: {e}"
        write_text(run_dir / "phaseJ05_decision_next_step.md", f"# Phase J05 Decision\n\n- Classification: **J0_STOP_INFRA**\n- Reason: J0.1 failure: {e}")
        json_dump(run_dir / "phaseJ0_run_manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "classification": "J0_STOP_INFRA", "error": str(e)}))
        return

    # J0.2
    try:
        tf, j2_obj = phase_j02_baseline_rebuild(run_dir=run_dir, seed=int(args.seed), subset_csv=Path(args.subset_csv).resolve())
        manifest["phases"]["J0.2"] = {"classification": "PASS", "trade_feature_rows": int(len(tf))}
    except Exception as e:
        manifest["classification"] = "J0_STOP_INFRA"
        manifest["mainline_status"] = "STOP_INFRA"
        manifest["reason"] = f"J0.2 failure: {e}"
        write_text(run_dir / "phaseJ05_decision_next_step.md", f"# Phase J05 Decision\n\n- Classification: **J0_STOP_INFRA**\n- Reason: J0.2 failure: {e}")
        json_dump(run_dir / "phaseJ0_run_manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "classification": "J0_STOP_INFRA", "error": str(e)}))
        return

    # J0.3
    phase_j03_write_specs(run_dir)
    manifest["phases"]["J0.3"] = {"classification": "PASS"}

    # J0.4
    try:
        ablation, inv = phase_j04_ablation(run_dir=run_dir, tf=tf, route_cache=j2_obj["route_cache"], seed=int(args.seed))
        manifest["phases"]["J0.4"] = {
            "classification": "PASS",
            "variants_tested": int(len(ablation)),
            "valid_for_ranking": int((to_num(ablation["valid_for_ranking"]) == 1).sum()),
            "invalid_reason_histogram": inv,
        }
    except Exception as e:
        manifest["classification"] = "J0_STOP_INFRA"
        manifest["mainline_status"] = "STOP_INFRA"
        manifest["reason"] = f"J0.4 failure: {e}"
        write_text(run_dir / "phaseJ05_decision_next_step.md", f"# Phase J05 Decision\n\n- Classification: **J0_STOP_INFRA**\n- Reason: J0.4 failure: {e}")
        json_dump(run_dir / "phaseJ0_run_manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "classification": "J0_STOP_INFRA", "error": str(e)}))
        return

    # J0.5
    cls, main = phase_j05_decision(run_dir, ablation)
    manifest["classification"] = cls
    manifest["mainline_status"] = main
    manifest["duration_sec"] = float(time.time() - t0)
    manifest["phases"]["J0.5"] = {"classification": cls}
    json_dump(run_dir / "phaseJ0_run_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "classification": cls, "mainline_status": main}, sort_keys=True))


if __name__ == "__main__":
    main()
