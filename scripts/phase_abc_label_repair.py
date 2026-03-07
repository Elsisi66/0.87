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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "repo_root": "/root/analysis/0.87",
    "symbol": "SOLUSDT",
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "primary_exec_hash": "862c940746de0da984862d95",
    "backup_exec_hash": "992bd371689ba3936f3b4d09",
    "phase_ae_dir": "/root/analysis/0.87/reports/execution_layer/PHASEAE_SIGNAL_LABELING_20260223_111116",
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


def markdown_table(df: pd.DataFrame, cols: Iterable[str]) -> str:
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
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


def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    a = to_num(x)
    b = to_num(y)
    m = a.notna() & b.notna()
    if int(m.sum()) < 2:
        return float("nan")
    ar = a[m].rank(method="average").to_numpy(dtype=float)
    br = b[m].rank(method="average").to_numpy(dtype=float)
    sa = float(np.std(ar, ddof=0))
    sb = float(np.std(br, ddof=0))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def build_s1_risk_features(df: pd.DataFrame, qstats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    z = af.compute_policy_features(df.copy(), qstats=qstats)
    # Use AE-stable feature family with interaction emphasis.
    # f_streak: prior_loss_streak_len
    # f_tail20: prior_rolling_tail_count_20
    # f_closehigh: pre3m_close_to_high_dist_bps
    # f_int1: prior_rolling_loss_rate_5 x pre3m_wick_ratio
    # f_int2: pre3m_close_to_high_dist_bps x pre3m_realized_vol_12
    w = {"f_streak": 0.24, "f_tail20": 0.22, "f_closehigh": 0.22, "f_int1": 0.18, "f_int2": 0.14}
    score = (
        w["f_streak"] * to_num(z["f_streak"]).fillna(0.0)
        + w["f_tail20"] * to_num(z["f_tail20"]).fillna(0.0)
        + w["f_closehigh"] * to_num(z["f_closehigh"]).fillna(0.0)
        + w["f_int1"] * to_num(z["f_int1"]).fillna(0.0)
        + w["f_int2"] * to_num(z["f_int2"]).fillna(0.0)
    )
    z["risk_score_s1"] = score.clip(lower=0.0, upper=1.0)
    return z


def risk_stability_summary(df: pd.DataFrame, score_col: str = "risk_score_s1") -> Dict[str, Any]:
    x = df.copy()
    x = x[(x["entry_for_labels"] == 1) & to_num(x["y_toxic_trade"]).notna()].copy()
    if x.empty:
        return {
            "support": 0,
            "overall_spearman": float("nan"),
            "stable_sign_frac": 0.0,
            "split_eligible": 0,
            "split_positive": 0,
            "score_nan_rate": 1.0,
            "monotonic_violations": -1,
            "event_rate_by_quintile": [],
        }
    s = to_num(x[score_col])
    y = to_num(x["y_toxic_trade"])
    m = s.notna() & y.notna()
    x = x.loc[m].copy()
    s = to_num(x[score_col])
    y = to_num(x["y_toxic_trade"])
    overall_spearman = spearman_corr_no_scipy(s, y) if len(x) >= 10 else float("nan")

    # Quintile monotonicity.
    try:
        bins = pd.qcut(s, q=min(5, max(2, int(s.nunique()))), duplicates="drop")
        qdf = pd.DataFrame({"bin": bins.astype(str), "score": s, "y": y}).groupby("bin", dropna=False).agg(
            support=("y", "size"),
            event_rate=("y", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        ).reset_index()
        rates = qdf["event_rate"].to_numpy(dtype=float)
        monotonic_violations = int(np.sum(np.diff(rates) < -1e-12)) if rates.size >= 2 else 0
        event_rate_by_quintile = qdf.to_dict(orient="records")
    except Exception:
        monotonic_violations = -1
        event_rate_by_quintile = []

    split_positive = 0
    split_eligible = 0
    split_rows: List[Dict[str, Any]] = []
    for sid, g in x.groupby("split_id", dropna=True):
        if len(g) < 30:
            continue
        sc = spearman_corr_no_scipy(to_num(g[score_col]), to_num(g["y_toxic_trade"]))
        if np.isfinite(sc):
            split_eligible += 1
            if sc > 0:
                split_positive += 1
        split_rows.append({"split_id": int(sid) if np.isfinite(sid) else str(sid), "support": int(len(g)), "spearman": sc})
    stable_sign_frac = safe_div(float(split_positive), float(split_eligible)) if split_eligible > 0 else 0.0

    return {
        "support": int(len(x)),
        "overall_spearman": overall_spearman,
        "stable_sign_frac": float(stable_sign_frac),
        "split_eligible": int(split_eligible),
        "split_positive": int(split_positive),
        "score_nan_rate": float(1.0 - (m.sum() / max(1, len(df[(df["entry_for_labels"] == 1)])))),
        "monotonic_violations": int(monotonic_violations),
        "event_rate_by_quintile": event_rate_by_quintile,
        "split_rows": split_rows,
    }


def policy_hash(policy: Dict[str, Any]) -> str:
    txt = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def build_small_policy_set(max_policies: int = 80) -> List[Dict[str, Any]]:
    # Fixed AE-stable risk score weights.
    base_weights = {
        "w_streak": 0.24,
        "w_tail20": 0.22,
        "w_closehigh": 0.22,
        "w_vol": 0.00,
        "w_wick": 0.00,
        "w_int1": 0.18,
        "w_int2": 0.14,
    }
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(p: Dict[str, Any]) -> None:
        h = policy_hash(p)
        if h in seen:
            return
        seen.add(h)
        out.append(p)

    # Core S1-like policies.
    core = [
        {"policy_id": "S1_half_q60", "family": "step", "weights": base_weights, "params": {"t1": 0.00, "t2": 0.60, "s_hi": 1.00, "s_mid": 1.00, "s_lo": 0.50}, "size_bounds": [0.50, 1.50]},
        {"policy_id": "S1_half_q65", "family": "step", "weights": base_weights, "params": {"t1": 0.00, "t2": 0.65, "s_hi": 1.00, "s_mid": 1.00, "s_lo": 0.50}, "size_bounds": [0.50, 1.50]},
        {"policy_id": "S1_half_q70", "family": "step", "weights": base_weights, "params": {"t1": 0.00, "t2": 0.70, "s_hi": 1.00, "s_mid": 1.00, "s_lo": 0.50}, "size_bounds": [0.50, 1.50]},
        {"policy_id": "S1_half_q75", "family": "step", "weights": base_weights, "params": {"t1": 0.00, "t2": 0.75, "s_hi": 1.00, "s_mid": 1.00, "s_lo": 0.50}, "size_bounds": [0.50, 1.50]},
    ]
    for p in core:
        add(p)

    # Grid around threshold + downsize strength.
    thrs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    downs = [0.40, 0.50, 0.60, 0.70]
    ups = [1.00, 1.05, 1.10]
    for t2 in thrs:
        for d in downs:
            for u in ups:
                add(
                    {
                        "policy_id": f"RS_step_t{t2:.2f}_d{d:.2f}_u{u:.2f}",
                        "family": "step",
                        "weights": copy.deepcopy(base_weights),
                        "params": {"t1": 0.00, "t2": float(t2), "s_hi": float(u), "s_mid": float(u), "s_lo": float(d)},
                        "size_bounds": [0.50, 1.50],
                    }
                )

    # Smooth family around same score.
    for slope in [0.20, 0.30, 0.40, 0.50, 0.60]:
        add(
            {
                "policy_id": f"RS_smooth_s{slope:.2f}",
                "family": "smooth",
                "weights": copy.deepcopy(base_weights),
                "params": {"slope": float(slope)},
                "size_bounds": [0.50, 1.50],
            }
        )

    # Hybrid with streak/tail triggers from AE.
    for slope in [0.25, 0.35, 0.45]:
        for ks, kt in [(3, 8), (4, 10), (5, 12)]:
            for penalty in [0.90, 0.82]:
                add(
                    {
                        "policy_id": f"RS_hyb_s{slope:.2f}_k{ks}_{kt}_p{penalty:.2f}",
                        "family": "hybrid",
                        "weights": copy.deepcopy(base_weights),
                        "params": {
                            "slope": float(slope),
                            "k_streak": int(ks),
                            "k_tail20": int(kt),
                            "penalty_mult": float(penalty),
                            "recovery_bonus": 0.04,
                        },
                        "size_bounds": [0.50, 1.50],
                    }
                )

    out = out[: int(max_policies)]
    # Canonicalize via AF helper for family constraints.
    canon: List[Dict[str, Any]] = []
    for p in out:
        canon.append(canonicalize_policy(p))
    return canon


def canonicalize_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(policy)
    fam = str(p.get("family", "step"))
    p.setdefault("weights", {})
    p.setdefault("params", {})
    p["size_bounds"] = [0.50, 1.50]
    if fam == "step":
        t1 = float(np.clip(float(p["params"].get("t1", 0.0)), 0.0, 0.90))
        t2 = float(np.clip(float(p["params"].get("t2", 0.65)), t1 + 0.01, 0.98))
        s_hi = float(np.clip(float(p["params"].get("s_hi", 1.0)), 1.0, 1.5))
        s_mid = float(np.clip(float(p["params"].get("s_mid", 1.0)), 0.5, s_hi))
        s_lo = float(np.clip(float(p["params"].get("s_lo", 0.5)), 0.5, s_mid))
        p["params"] = {"t1": t1, "t2": t2, "s_hi": s_hi, "s_mid": s_mid, "s_lo": s_lo}
    elif fam == "smooth":
        p["params"] = {"slope": float(np.clip(float(p["params"].get("slope", 0.35)), 0.0, 1.5))}
    elif fam == "hybrid":
        p["params"] = {
            "slope": float(np.clip(float(p["params"].get("slope", 0.35)), 0.0, 1.5)),
            "k_streak": int(np.clip(float(p["params"].get("k_streak", 4)), 1, 20)),
            "k_tail20": int(np.clip(float(p["params"].get("k_tail20", 10)), 1, 30)),
            "penalty_mult": float(np.clip(float(p["params"].get("penalty_mult", 0.85)), 0.5, 1.0)),
            "recovery_bonus": float(np.clip(float(p["params"].get("recovery_bonus", 0.04)), 0.0, 0.20)),
        }
    else:
        raise ValueError(f"Unsupported policy family for phaseC: {fam}")
    return p


def flat_policy() -> Dict[str, Any]:
    return {
        "policy_id": "flat_baseline",
        "family": "streak_control",
        "weights": {},
        "params": {"k_streak": 10000, "size_down": 1.0, "size_up": 1.0},
        "size_bounds": [0.50, 1.50],
    }


def base_unweighted(df: pd.DataFrame) -> Dict[str, Any]:
    valid = (df["entry_for_labels"] == 1) & df["pnl_net_trade_notional_dec"].notna()
    entries = int(valid.sum())
    entry_rate = float(entries / max(1, len(df)))
    taker_share = float(np.nanmean(df.loc[valid, "taker_flag"])) if entries > 0 else float("nan")
    d = to_num(df.loc[valid, "fill_delay_min"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(d, 0.95)) if d.size else float("nan")
    return {
        "entries_valid": entries,
        "entry_rate": entry_rate,
        "taker_share": taker_share,
        "p95_fill_delay_min": p95,
    }


def evaluate_phase_c(
    *,
    phase_dir: Path,
    sig_in: pd.DataFrame,
    exec_genome: Dict[str, Any],
    policies: List[Dict[str, Any]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    routes = af.route_signal_sets(sig_in)
    rows: List[Dict[str, Any]] = []
    invalid_reasons: List[Dict[str, str]] = []
    route_ix = 0
    for route_id, sdf in routes.items():
        route_ix += 1
        wf_splits = 5 if route_id == "route1_holdout" else 7
        train_ratio = 0.70 if route_id == "route1_holdout" else 0.65
        met, sig, _split, _args, bundle = ae.evaluate_exact(
            run_dir=phase_dir,
            signals_df=sdf,
            genome=exec_genome,
            seed=int(seed) + route_ix,
            name=f"phaseC_{route_id}_base",
            wf_splits=wf_splits,
            train_ratio=train_ratio,
        )
        pre = ae.build_preentry_features(bundle, exec_genome)
        d = af.parse_entry_rows(ae.build_trade_labels(sig.merge(pre, on=["signal_id", "signal_time"], how="left")))
        qstats = af.build_quantile_stats(d)
        b_unw = base_unweighted(d)
        flat_met, _ = af.evaluate_sizing_policy_on_dataset(
            d,
            policy=flat_policy(),
            qstats=qstats,
            baseline_unweighted=b_unw,
            rng_seed=int(seed) + 123 + route_ix,
        )
        rows.append(
            {
                "route_id": route_id,
                "policy_id": "flat_baseline",
                "policy_family": "flat",
                "policy_hash": "flat",
                "delta_expectancy_vs_flat": 0.0,
                "cvar_improve_ratio_vs_flat": 0.0,
                "maxdd_improve_ratio_vs_flat": 0.0,
                "candidate_budget_norm_pass": 1,
                "candidate_invariance_pass": 1,
                **flat_met,
            }
        )
        for j, p in enumerate(policies):
            pm, _ = af.evaluate_sizing_policy_on_dataset(
                d,
                policy=p,
                qstats=qstats,
                baseline_unweighted=b_unw,
                rng_seed=int(seed) + 1000 * route_ix + j,
            )
            budget_pass = int(np.isfinite(pm["budget_norm_error"]) and abs(float(pm["budget_norm_error"])) <= 0.02)
            inv_pass = int(
                int(pm["invariance_entries_match"]) == 1
                and int(pm["invariance_entry_rate_match"]) == 1
                and int(pm["invariance_taker_share_match"]) == 1
                and int(pm["invariance_fill_delay_match"]) == 1
            )
            row = {
                "route_id": route_id,
                "policy_id": str(p["policy_id"]),
                "policy_family": str(p["family"]),
                "policy_hash": policy_hash(p),
                "delta_expectancy_vs_flat": float(pm["exec_expectancy_net_weighted"] - flat_met["exec_expectancy_net_weighted"]),
                "cvar_improve_ratio_vs_flat": safe_div(abs(float(flat_met["cvar_5_weighted"])) - abs(float(pm["cvar_5_weighted"])), abs(float(flat_met["cvar_5_weighted"]))),
                "maxdd_improve_ratio_vs_flat": safe_div(abs(float(flat_met["max_drawdown_weighted"])) - abs(float(pm["max_drawdown_weighted"])), abs(float(flat_met["max_drawdown_weighted"]))),
                "candidate_budget_norm_pass": int(budget_pass),
                "candidate_invariance_pass": int(inv_pass),
                **pm,
            }
            rows.append(row)
            # Per-route strict reasons for histogram.
            reasons: List[str] = []
            if row["delta_expectancy_vs_flat"] <= 0:
                reasons.append("non_positive_expectancy_delta")
            if row["maxdd_improve_ratio_vs_flat"] <= 0:
                reasons.append("non_positive_maxdd_improvement")
            if row["cvar_improve_ratio_vs_flat"] < 0:
                reasons.append("negative_cvar_improvement")
            if budget_pass != 1:
                reasons.append("budget_norm_fail")
            if inv_pass != 1:
                reasons.append("invariance_fail")
            for r in reasons:
                invalid_reasons.append({"route_id": route_id, "policy_id": str(p["policy_id"]), "reason": r})

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf, pd.DataFrame()
    agg = rdf[rdf["policy_id"] != "flat_baseline"].groupby(["policy_id", "policy_hash", "policy_family"], dropna=False).agg(
        routes_tested=("route_id", "nunique"),
        min_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "min"),
        min_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "min"),
        min_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "min"),
        mean_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "mean"),
        mean_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "mean"),
        mean_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "mean"),
        all_budget_pass=("candidate_budget_norm_pass", "min"),
        all_invariance_pass=("candidate_invariance_pass", "min"),
    ).reset_index()
    agg["strict_pass"] = (
        (to_num(agg["min_delta_expectancy_vs_flat"]) > 0.0)
        & (to_num(agg["min_cvar_improve_ratio_vs_flat"]) >= 0.0)
        & (to_num(agg["min_maxdd_improve_ratio_vs_flat"]) > 0.0)
        & (to_num(agg["all_budget_pass"]) == 1)
        & (to_num(agg["all_invariance_pass"]) == 1)
    ).astype(int)
    agg["rank_score"] = (
        8.0 * to_num(agg["min_delta_expectancy_vs_flat"]).fillna(-1e9)
        + 2.0 * to_num(agg["min_cvar_improve_ratio_vs_flat"]).fillna(-1e9)
        + 2.0 * to_num(agg["min_maxdd_improve_ratio_vs_flat"]).fillna(-1e9)
        + 1.0 * to_num(agg["mean_delta_expectancy_vs_flat"]).fillna(-1e9)
    )
    agg = agg.sort_values(
        ["strict_pass", "rank_score", "min_delta_expectancy_vs_flat", "min_cvar_improve_ratio_vs_flat", "min_maxdd_improve_ratio_vs_flat"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return rdf, agg


def psr_dsr_for_policies(
    *,
    phase_dir: Path,
    sig_in: pd.DataFrame,
    exec_genome: Dict[str, Any],
    policies: List[Dict[str, Any]],
    seed: int,
) -> pd.DataFrame:
    # Use route2 (full-like) as proxy vector source.
    route2 = af.route_signal_sets(sig_in)["route2_reslice"]
    met, sig, _split, _args, bundle = ae.evaluate_exact(
        run_dir=phase_dir,
        signals_df=route2,
        genome=exec_genome,
        seed=int(seed) + 77,
        name="phaseC_psr_route2",
        wf_splits=7,
        train_ratio=0.65,
    )
    pre = ae.build_preentry_features(bundle, exec_genome)
    d = af.parse_entry_rows(ae.build_trade_labels(sig.merge(pre, on=["signal_id", "signal_time"], how="left")))
    qstats = af.build_quantile_stats(d)
    b_unw = base_unweighted(d)

    vectors: Dict[str, np.ndarray] = {}
    for i, p in enumerate(policies):
        _m, size = af.evaluate_sizing_policy_on_dataset(
            d,
            policy=p,
            qstats=qstats,
            baseline_unweighted=b_unw,
            rng_seed=int(seed) + 800 + i,
        )
        valid = (d["entry_for_labels"] == 1) & d["pnl_net_trade_notional_dec"].notna()
        vec = np.zeros(len(d), dtype=float)
        vmask = valid.to_numpy(dtype=bool)
        vec[vmask] = to_num(d.loc[valid, "pnl_net_trade_notional_dec"]).to_numpy(dtype=float) * size.loc[valid].to_numpy(dtype=float)
        vectors[str(p["policy_id"])] = vec

    ids = list(vectors.keys())
    corrs: List[float] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = vectors[ids[i]]
            b = vectors[ids[j]]
            if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))
    avg_corr = float(np.mean(corrs)) if corrs else float("nan")
    eff_uncorr = float(len(ids))
    eff_corr = float(len(ids) / (1.0 + (len(ids) - 1.0) * max(0.0, avg_corr))) if len(ids) else 0.0

    rows: List[Dict[str, Any]] = []
    for pid in ids:
        psr, dsr = af.psr_proxy_from_pnl(vectors[pid], eff_trials=max(1.0, eff_corr))
        rows.append(
            {
                "policy_id": pid,
                "psr_proxy": float(psr),
                "dsr_proxy": float(dsr),
                "effective_trials_uncorrelated": eff_uncorr,
                "effective_trials_corr_adjusted": eff_corr,
                "avg_abs_corr_proxy": avg_corr,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A/B/C: execution-aware label-quality upgrade and small sizing rerun (contract-locked)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--max-policies", type=int, default=72)
    args = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_root = exec_root / f"PHASEABC_LABEL_REPAIR_{utc_tag()}"
    run_root.mkdir(parents=True, exist_ok=False)

    overall = {
        "generated_utc": utc_now(),
        "run_root": str(run_root),
        "seed": int(args.seed),
        "phases": {},
        "mainline_status": "CONTINUE",
    }

    # --------------------
    # Phase A
    # --------------------
    phase_a = run_root / f"phaseA_{utc_tag()}"
    phase_a.mkdir(parents=True, exist_ok=False)
    a_start = time.time()
    contract_status = "PASS"
    contract_reason = "all checks passed"
    try:
        rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
        fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
        met_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
        for fp in (rep_fp, fee_fp, met_fp):
            if not fp.exists():
                raise FileNotFoundError(f"Missing locked file: {fp}")
        fee_sha = sha256_file(fee_fp)
        met_sha = sha256_file(met_fp)
        if fee_sha != LOCKED["expected_fee_sha"] or met_sha != LOCKED["expected_metrics_sha"]:
            contract_status = "CONTRACT_FAIL"
            contract_reason = "canonical hash mismatch"
        sig_raw = pd.read_csv(rep_fp)
        sig = ae.ensure_signals_schema(sig_raw)
        if sig.empty:
            contract_status = "CONTRACT_FAIL"
            contract_reason = "frozen subset empty after schema normalization"
        expected_cols = ["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]
        schema_ok = int(all(c in sig.columns for c in expected_cols))
        if schema_ok != 1:
            contract_status = "INFRA_FAIL"
            contract_reason = "subset schema mismatch"

        lock_args = ae.build_args(signals_csv=rep_fp, seed=int(args.seed))
        lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=phase_a)
        if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
            contract_status = "CONTRACT_FAIL"
            contract_reason = "freeze_lock_pass != 1"

        val_obj = {
            "generated_utc": utc_now(),
            "repo_root": LOCKED["repo_root"],
            "symbol": LOCKED["symbol"],
            "subset_path": str(rep_fp),
            "subset_rows_raw": int(len(sig_raw)),
            "subset_rows_schema_normalized": int(len(sig)),
            "subset_columns": list(sig.columns),
            "schema_expected_columns": expected_cols,
            "schema_ok": int(schema_ok),
            "fee_path": str(fee_fp),
            "metrics_path": str(met_fp),
            "observed_fee_sha256": fee_sha,
            "observed_metrics_sha256": met_sha,
            "expected_fee_sha256": LOCKED["expected_fee_sha"],
            "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
            "fee_hash_match": int(fee_sha == LOCKED["expected_fee_sha"]),
            "metrics_hash_match": int(met_sha == LOCKED["expected_metrics_sha"]),
            "allow_freeze_hash_mismatch": 0,
            "freeze_lock_validation": lock_validation,
            "phase_decision": contract_status,
            "phase_reason": contract_reason,
        }
        json_dump(phase_a / "phaseA_freeze_lock_validation.json", val_obj)
        a_manifest = {
            "generated_utc": utc_now(),
            "phase": "A",
            "duration_sec": float(time.time() - a_start),
            "decision": contract_status,
            "reason": contract_reason,
            "phase_dir": str(phase_a),
        }
        json_dump(phase_a / "phaseA_run_manifest.json", a_manifest)
        write_text(
            phase_a / "phaseA_report.md",
            "\n".join(
                [
                    "# Phase A Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: **{contract_status}**",
                    f"- Reason: {contract_reason}",
                    f"- Subset rows (normalized): `{len(sig)}`",
                    f"- Freeze lock pass: `{int(lock_validation.get('freeze_lock_pass', 0))}`",
                    f"- Fee hash match: `{int(fee_sha == LOCKED['expected_fee_sha'])}`",
                    f"- Metrics hash match: `{int(met_sha == LOCKED['expected_metrics_sha'])}`",
                ]
            ),
        )
        write_text(
            phase_a / "phaseA_decision.md",
            "\n".join(
                [
                    "# Phase A Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: **{contract_status}**",
                    f"- Reason: {contract_reason}",
                ]
            ),
        )
        overall["phases"]["A"] = {"classification": contract_status, "phase_dir": str(phase_a)}
        if contract_status != "PASS":
            overall["mainline_status"] = "STOP_CONTRACT" if contract_status == "CONTRACT_FAIL" else "STOP_INFRA"
            json_dump(run_root / "run_manifest.json", overall)
            print(json.dumps({"furthest_phase": "A", "classification": contract_status, "mainline_status": overall["mainline_status"], "run_root": str(run_root)}, sort_keys=True))
            return
    except Exception as e:
        write_text(
            phase_a / "phaseA_decision.md",
            f"# Phase A Decision\n\n- Generated UTC: {utc_now()}\n- Classification: **INFRA_FAIL**\n- Reason: {e}",
        )
        json_dump(
            phase_a / "phaseA_run_manifest.json",
            {"generated_utc": utc_now(), "phase": "A", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(phase_a)},
        )
        overall["phases"]["A"] = {"classification": "INFRA_FAIL", "phase_dir": str(phase_a), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "A", "classification": "INFRA_FAIL", "mainline_status": "STOP_INFRA", "run_root": str(run_root)}, sort_keys=True))
        return

    # --------------------
    # Phase B
    # --------------------
    phase_b = run_root / f"phaseB_{utc_tag()}"
    phase_b.mkdir(parents=True, exist_ok=False)
    (phase_b / "phaseB_artifacts").mkdir(parents=True, exist_ok=True)
    b_start = time.time()
    b_class = "PASS"
    b_reason = "stable risk score"
    try:
        sig = ae.ensure_signals_schema(pd.read_csv(Path(LOCKED["representative_subset_csv"])))
        exec_pair = ae.load_exec_pair(exec_root)
        if exec_pair["E1"]["genome_hash"] != LOCKED["primary_exec_hash"] or exec_pair["E2"]["genome_hash"] != LOCKED["backup_exec_hash"]:
            b_class = "CONTRACT_FAIL"
            b_reason = "execution hash mismatch"
        # Evaluate E1.
        met1, sig1, split1, _args1, bun1 = ae.evaluate_exact(
            run_dir=phase_b,
            signals_df=sig,
            genome=exec_pair["E1"]["genome"],
            seed=int(args.seed) + 1,
            name="phaseB_e1",
        )
        pre1 = ae.build_preentry_features(bun1, exec_pair["E1"]["genome"])
        d1 = af.parse_entry_rows(ae.build_trade_labels(sig1.merge(pre1, on=["signal_id", "signal_time"], how="left")))
        q1 = af.build_quantile_stats(d1)
        d1f = build_s1_risk_features(d1, qstats=q1)

        # Evaluate E2 for direction confirmation.
        met2, sig2, split2, _args2, bun2 = ae.evaluate_exact(
            run_dir=phase_b,
            signals_df=sig,
            genome=exec_pair["E2"]["genome"],
            seed=int(args.seed) + 2,
            name="phaseB_e2",
        )
        pre2 = ae.build_preentry_features(bun2, exec_pair["E2"]["genome"])
        d2 = af.parse_entry_rows(ae.build_trade_labels(sig2.merge(pre2, on=["signal_id", "signal_time"], how="left")))
        q2 = af.build_quantile_stats(d2)
        d2f = build_s1_risk_features(d2, qstats=q2)

        # Stability summaries.
        s1 = risk_stability_summary(d1f, score_col="risk_score_s1")
        s2 = risk_stability_summary(d2f, score_col="risk_score_s1")
        backup_direction_consistent = int(
            np.isfinite(s2["overall_spearman"]) and np.isfinite(s1["overall_spearman"]) and (s1["overall_spearman"] > 0) and (s2["overall_spearman"] > 0)
        )
        # AE-aligned rule of thumb: stable_sign_frac >= 0.60.
        pass_rule = {
            "stable_sign_frac_e1_min": 0.60,
            "overall_spearman_e1_positive": True,
            "monotonic_violations_e1_max": 2,
            "score_nan_rate_e1_max": 0.00,
            "backup_direction_consistent_required": True,
            "leakage_check": "pre-entry and prior-only features only",
        }
        b_pass = (
            float(s1["stable_sign_frac"]) >= pass_rule["stable_sign_frac_e1_min"]
            and float(s1["overall_spearman"]) > 0.0
            and int(s1["monotonic_violations"]) >= 0
            and int(s1["monotonic_violations"]) <= pass_rule["monotonic_violations_e1_max"]
            and float(s1["score_nan_rate"]) <= pass_rule["score_nan_rate_e1_max"]
            and int(backup_direction_consistent) == 1
        )
        if not b_pass:
            b_class = "NO_GO"
            b_reason = "stability/monotonicity or backup-direction criteria failed"

        # risk_score artifact (entry rows only, no leakage fields from post-entry in features section).
        rs = d1f.copy()
        rs = rs[(rs["entry_for_labels"] == 1)].copy()
        rs["risk_bucket"] = pd.qcut(to_num(rs["risk_score_s1"]), q=min(5, max(2, int(to_num(rs["risk_score_s1"]).nunique()))), duplicates="drop").astype(str)
        keep_cols = [
            "signal_id",
            "signal_time_utc",
            "entry_time_utc",
            "split_id",
            "risk_score_s1",
            "risk_bucket",
            "prior_loss_streak_len",
            "prior_rolling_tail_count_20",
            "pre3m_close_to_high_dist_bps",
            "pre3m_realized_vol_12",
            "pre3m_wick_ratio",
            "prior_rolling_loss_rate_5",
            "f_int1",
            "f_int2",
            "y_toxic_trade",
            "y_cluster_loss",
            "y_tail_loss",
        ]
        rs = rs.loc[:, [c for c in keep_cols if c in rs.columns]].copy()
        rs.to_csv(phase_b / "phaseB_artifacts" / "risk_score.csv", index=False)

        split_stab_obj = {
            "generated_utc": utc_now(),
            "phase": "B",
            "score_name": "risk_score_s1",
            "rule": pass_rule,
            "e1_summary": s1,
            "e2_summary": s2,
            "backup_direction_consistent": int(backup_direction_consistent),
        }
        json_dump(phase_b / "phaseB_artifacts" / "split_stability.json", split_stab_obj)

        b_manifest = {
            "generated_utc": utc_now(),
            "phase": "B",
            "duration_sec": float(time.time() - b_start),
            "decision": b_class,
            "reason": b_reason,
            "phase_dir": str(phase_b),
            "source_phase_ae_dir": LOCKED["phase_ae_dir"],
            "exec_hashes": {"E1": exec_pair["E1"]["genome_hash"], "E2": exec_pair["E2"]["genome_hash"]},
        }
        json_dump(phase_b / "phaseB_run_manifest.json", b_manifest)

        write_text(
            phase_b / "phaseB_report.md",
            "\n".join(
                [
                    "# Phase B Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: **{b_class}**",
                    f"- Reason: {b_reason}",
                    f"- E1 support: `{s1['support']}`",
                    f"- E1 stable_sign_frac: `{float(s1['stable_sign_frac']):.4f}` (rule >= 0.60)",
                    f"- E1 overall spearman(y_toxic): `{float(s1['overall_spearman']):.6f}`",
                    f"- E1 monotonic violations: `{int(s1['monotonic_violations'])}`",
                    f"- E2 overall spearman(y_toxic): `{float(s2['overall_spearman']):.6f}`",
                    f"- Backup direction consistent: `{int(backup_direction_consistent)}`",
                    "",
                    "## Split Spearman (E1)",
                    "",
                    markdown_table(pd.DataFrame(s1.get("split_rows", [])), ["split_id", "support", "spearman"]),
                    "",
                    "## Split Spearman (E2)",
                    "",
                    markdown_table(pd.DataFrame(s2.get("split_rows", [])), ["split_id", "support", "spearman"]),
                    "",
                    "## Leakage Audit",
                    "",
                    "- Feature set uses only pre-entry fields and prior-trade context fields from AE.",
                    "- Excluded post-entry outcome fields from risk score construction.",
                ]
            ),
        )
        write_text(
            phase_b / "phaseB_decision.md",
            "\n".join(
                [
                    "# Phase B Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: **{b_class}**",
                    f"- Reason: {b_reason}",
                ]
            ),
        )
        overall["phases"]["B"] = {"classification": b_class, "phase_dir": str(phase_b)}
        if b_class != "PASS":
            overall["mainline_status"] = "STOP_NO_GO" if b_class == "NO_GO" else ("STOP_CONTRACT" if b_class == "CONTRACT_FAIL" else "STOP_INFRA")
            ng = phase_b / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(
                ng / "phaseB_no_go_diagnosis.md",
                "\n".join(
                    [
                        "# Phase B NO_GO Diagnosis",
                        "",
                        f"- Generated UTC: {utc_now()}",
                        f"- Classification: {b_class}",
                        f"- Reason: {b_reason}",
                        "- Mainline stopped before Phase C.",
                    ]
                ),
            )
            write_text(
                ng / "next_step_prompt.txt",
                "Phase B NO_GO fallback: improve label quality first (feature robustness and split-consistent toxic-trade signals), then rerun a small sizing pilot. Keep hard gates and contract lock unchanged.",
            )
            json_dump(run_root / "run_manifest.json", overall)
            print(json.dumps({"furthest_phase": "B", "classification": b_class, "mainline_status": overall["mainline_status"], "run_root": str(run_root)}, sort_keys=True))
            return
    except Exception as e:
        b_class = "INFRA_FAIL"
        b_reason = str(e)
        write_text(
            phase_b / "phaseB_decision.md",
            f"# Phase B Decision\n\n- Generated UTC: {utc_now()}\n- Classification: **INFRA_FAIL**\n- Reason: {e}",
        )
        json_dump(
            phase_b / "phaseB_run_manifest.json",
            {"generated_utc": utc_now(), "phase": "B", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(phase_b)},
        )
        overall["phases"]["B"] = {"classification": "INFRA_FAIL", "phase_dir": str(phase_b), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "B", "classification": "INFRA_FAIL", "mainline_status": "STOP_INFRA", "run_root": str(run_root)}, sort_keys=True))
        return

    # --------------------
    # Phase C
    # --------------------
    phase_c = run_root / f"phaseC_{utc_tag()}"
    phase_c.mkdir(parents=True, exist_ok=False)
    c_start = time.time()
    c_class = "PASS"
    c_reason = "strict passers found"
    try:
        sig = ae.ensure_signals_schema(pd.read_csv(Path(LOCKED["representative_subset_csv"])))
        exec_pair = ae.load_exec_pair(exec_root)
        policies = build_small_policy_set(max_policies=int(args.max_policies))
        route_df, agg_df = evaluate_phase_c(
            phase_dir=phase_c,
            sig_in=sig,
            exec_genome=copy.deepcopy(exec_pair["E1"]["genome"]),
            policies=policies,
            seed=int(args.seed) + 10,
        )
        route_df.to_csv(phase_c / "phaseC_results_by_route.csv", index=False)
        agg_df.to_csv(phase_c / "phaseC_results.csv", index=False)

        strict_passers = int((agg_df["strict_pass"] == 1).sum()) if not agg_df.empty else 0
        if strict_passers == 0:
            c_class = "NO_GO"
            c_reason = "0 strict passers on required routes"

        # invalid reason histogram.
        invalid_rows: List[str] = []
        if not agg_df.empty:
            for _, r in agg_df.iterrows():
                if int(r["strict_pass"]) == 1:
                    continue
                if float(r["min_delta_expectancy_vs_flat"]) <= 0:
                    invalid_rows.append("non_positive_expectancy_delta")
                if float(r["min_cvar_improve_ratio_vs_flat"]) < 0:
                    invalid_rows.append("negative_cvar_improvement")
                if float(r["min_maxdd_improve_ratio_vs_flat"]) <= 0:
                    invalid_rows.append("non_positive_maxdd_improvement")
                if int(r["all_budget_pass"]) != 1:
                    invalid_rows.append("budget_norm_fail")
                if int(r["all_invariance_pass"]) != 1:
                    invalid_rows.append("invariance_fail")
        hist: Dict[str, int] = {}
        for k in invalid_rows:
            hist[k] = int(hist.get(k, 0) + 1)
        json_dump(phase_c / "invalid_reason_histogram.json", hist)

        # PSR/DSR proxies for top candidates.
        top_ids = agg_df.head(12)["policy_id"].astype(str).tolist() if not agg_df.empty else []
        top_pols = [p for p in policies if str(p["policy_id"]) in set(top_ids)]
        psr_df = psr_dsr_for_policies(
            phase_dir=phase_c,
            sig_in=sig,
            exec_genome=copy.deepcopy(exec_pair["E1"]["genome"]),
            policies=top_pols,
            seed=int(args.seed) + 333,
        ) if top_pols else pd.DataFrame()
        if not psr_df.empty:
            psr_df.to_csv(phase_c / "phaseC_psr_dsr_proxy.csv", index=False)

        c_manifest = {
            "generated_utc": utc_now(),
            "phase": "C",
            "duration_sec": float(time.time() - c_start),
            "decision": c_class,
            "reason": c_reason,
            "phase_dir": str(phase_c),
            "policy_count": int(len(policies)),
            "strict_passers": int(strict_passers),
        }
        json_dump(phase_c / "phaseC_run_manifest.json", c_manifest)

        lines: List[str] = []
        lines.append("# Phase C Report")
        lines.append("")
        lines.append(f"- Generated UTC: {utc_now()}")
        lines.append(f"- Decision: **{c_class}**")
        lines.append(f"- Reason: {c_reason}")
        lines.append(f"- Policies evaluated: `{len(policies)}`")
        lines.append(f"- Strict passers: `{strict_passers}`")
        lines.append("")
        lines.append("## Strict Gate Definition")
        lines.append("")
        lines.append("- min_delta_expectancy_vs_flat > 0 on all required routes")
        lines.append("- min_cvar_improve_ratio_vs_flat >= 0 on all required routes")
        lines.append("- min_maxdd_improve_ratio_vs_flat > 0 on all required routes")
        lines.append("- budget/invariance pass flags required")
        lines.append("")
        lines.append("## Top Candidates")
        lines.append("")
        lines.append(
            markdown_table(
                agg_df.head(20),
                [
                    "policy_id",
                    "policy_family",
                    "strict_pass",
                    "min_delta_expectancy_vs_flat",
                    "min_cvar_improve_ratio_vs_flat",
                    "min_maxdd_improve_ratio_vs_flat",
                    "all_budget_pass",
                    "all_invariance_pass",
                    "rank_score",
                ],
            )
        )
        lines.append("")
        lines.append("## Reality-Check")
        lines.append("")
        lines.append("- Reality-check bootstrap: TODO placeholder (not implemented in this branch).")
        if not psr_df.empty:
            lines.append("")
            lines.append("## PSR/DSR Proxy (Top Subset)")
            lines.append("")
            lines.append(markdown_table(psr_df, ["policy_id", "psr_proxy", "dsr_proxy", "effective_trials_uncorrelated", "effective_trials_corr_adjusted", "avg_abs_corr_proxy"]))
        write_text(phase_c / "phaseC_report.md", "\n".join(lines))
        write_text(
            phase_c / "phaseC_decision.md",
            "\n".join(
                [
                    "# Phase C Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: **{c_class}**",
                    f"- Reason: {c_reason}",
                ]
            ),
        )

        overall["phases"]["C"] = {"classification": c_class, "phase_dir": str(phase_c)}
        if c_class != "PASS":
            overall["mainline_status"] = "STOP_NO_GO"
            ng = phase_c / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(
                ng / "phaseC_no_go_reasoning.md",
                "\n".join(
                    [
                        "# Phase C NO_GO Reasoning",
                        "",
                        f"- Generated UTC: {utc_now()}",
                        f"- Strict passers: {strict_passers}",
                        "- No policy passed strict multi-route criteria under unchanged hard gates.",
                    ]
                ),
            )
            write_text(
                ng / "next_step_prompt.txt",
                "Phase C NO_GO fallback: stop sizing expansion. Revisit upstream execution-aware labels/objective for tail-risk and split-stability, then rerun a bounded pilot only after measurable label-quality gain.",
            )
        else:
            overall["mainline_status"] = "CONTINUE"
            write_text(
                phase_c / "prompt_next.txt",
                "Phase D follow-up: run a medium (not marathon) sizing robustness pilot on the strict-pass policies only, with route perturbation + fee/slippage stress and unchanged hard gates/contract lock.",
            )
    except Exception as e:
        c_class = "INFRA_FAIL"
        c_reason = str(e)
        write_text(
            phase_c / "phaseC_decision.md",
            f"# Phase C Decision\n\n- Generated UTC: {utc_now()}\n- Classification: **INFRA_FAIL**\n- Reason: {e}",
        )
        json_dump(
            phase_c / "phaseC_run_manifest.json",
            {"generated_utc": utc_now(), "phase": "C", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(phase_c)},
        )
        overall["phases"]["C"] = {"classification": "INFRA_FAIL", "phase_dir": str(phase_c), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"

    overall["duration_sec"] = float(time.time() - t0)
    json_dump(run_root / "run_manifest.json", overall)
    furthest = "C" if "C" in overall["phases"] else ("B" if "B" in overall["phases"] else "A")
    cls = overall["phases"][furthest]["classification"]
    print(
        json.dumps(
            {
                "furthest_phase": furthest,
                "classification": cls,
                "mainline_status": overall["mainline_status"],
                "run_root": str(run_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
