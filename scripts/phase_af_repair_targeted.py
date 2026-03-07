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
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "symbol": "SOLUSDT",
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "primary_hash": "862c940746de0da984862d95",
    "backup_hash": "992bd371689ba3936f3b4d09",
    "base_policy_ids": ["P005", "P011", "P030"],
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


def policy_hash(policy: Dict[str, Any]) -> str:
    txt = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def latest_phaseaf_results(exec_root: Path) -> Path:
    dirs = sorted([p for p in exec_root.glob("PHASEAF_SIZING_PILOT_*") if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError("No PHASEAF_SIZING_PILOT_* directories found")
    for d in reversed(dirs):
        fp = d / "phaseAF_pilot_results.csv"
        if fp.exists():
            return fp
    raise FileNotFoundError("No phaseAF_pilot_results.csv found in PHASEAF_SIZING_PILOT_*")


def load_base_policies(af_results_csv: Path, policy_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(af_results_csv)
    out: Dict[str, Dict[str, Any]] = {}
    for pid in policy_ids:
        sub = df[df["policy_id"].astype(str) == str(pid)]
        if sub.empty:
            raise RuntimeError(f"Base policy {pid} missing in {af_results_csv}")
        desc = str(sub.iloc[0]["policy_desc"])
        pol = json.loads(desc)
        pol["policy_id"] = str(pid)
        out[str(pid)] = pol
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _scale_weights(weights: Dict[str, Any], scale: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in weights.items():
        out[k] = _clip(float(v) * float(scale), 0.0, 3.0)
    return out


def canonicalize_policy(policy: Dict[str, Any], pid: str) -> Dict[str, Any]:
    p = copy.deepcopy(policy)
    p["policy_id"] = str(pid)
    p["size_bounds"] = [0.50, 1.50]
    fam = str(p.get("family", "step"))
    p.setdefault("weights", {})
    p.setdefault("params", {})
    if fam == "step":
        t1 = _clip(float(p["params"].get("t1", 0.35)), 0.05, 0.80)
        t2 = _clip(float(p["params"].get("t2", 0.65)), t1 + 0.05, 0.95)
        s_hi = _clip(float(p["params"].get("s_hi", 1.10)), 1.00, 1.50)
        s_mid = _clip(float(p["params"].get("s_mid", 1.00)), 0.70, s_hi)
        s_lo = _clip(float(p["params"].get("s_lo", 0.80)), 0.50, s_mid)
        p["params"] = {"t1": t1, "t2": t2, "s_hi": s_hi, "s_mid": s_mid, "s_lo": s_lo}
    elif fam == "smooth":
        slope = _clip(float(p["params"].get("slope", 0.6)), 0.0, 1.5)
        p["params"] = {"slope": slope}
    elif fam == "streak":
        k_streak = int(_clip(float(p["params"].get("k_streak", 4)), 1, 12))
        k_tail20 = int(_clip(float(p["params"].get("k_tail20", 8)), 2, 20))
        s_up = _clip(float(p["params"].get("s_up", 1.05)), 1.00, 1.50)
        s_down = _clip(float(p["params"].get("s_down", 0.75)), 0.50, s_up)
        p["params"] = {"k_streak": k_streak, "k_tail20": k_tail20, "s_up": s_up, "s_down": s_down}
    elif fam == "hybrid":
        slope = _clip(float(p["params"].get("slope", 0.5)), 0.0, 1.5)
        k_streak = int(_clip(float(p["params"].get("k_streak", 4)), 1, 12))
        k_tail20 = int(_clip(float(p["params"].get("k_tail20", 8)), 2, 20))
        penalty = _clip(float(p["params"].get("penalty_mult", 0.85)), 0.50, 1.00)
        recovery = _clip(float(p["params"].get("recovery_bonus", 0.03)), 0.0, 0.20)
        p["params"] = {
            "slope": slope,
            "k_streak": k_streak,
            "k_tail20": k_tail20,
            "penalty_mult": penalty,
            "recovery_bonus": recovery,
        }
    else:
        raise ValueError(f"Unsupported policy family: {fam}")
    return p


def mutate_policy(base: Dict[str, Any], pid: str, weight_scale: float, param_scale: float, threshold_shift: float) -> Dict[str, Any]:
    p = copy.deepcopy(base)
    p["weights"] = _scale_weights(p.get("weights", {}), weight_scale)
    fam = str(p.get("family", "step"))
    prm = copy.deepcopy(p.get("params", {}))
    if fam == "step":
        prm["t1"] = float(prm.get("t1", 0.35)) * float(param_scale) + float(threshold_shift)
        prm["t2"] = float(prm.get("t2", 0.65)) * float(param_scale) + float(threshold_shift)
        prm["s_hi"] = float(prm.get("s_hi", 1.10)) * float(param_scale)
        prm["s_mid"] = float(prm.get("s_mid", 1.00)) * float(param_scale)
        prm["s_lo"] = float(prm.get("s_lo", 0.80)) * float(param_scale)
    elif fam == "hybrid":
        prm["slope"] = float(prm.get("slope", 0.5)) * float(param_scale)
        prm["k_streak"] = int(round(float(prm.get("k_streak", 4)) * float(param_scale)))
        prm["k_tail20"] = int(round(float(prm.get("k_tail20", 8)) * float(param_scale)))
        prm["penalty_mult"] = float(prm.get("penalty_mult", 0.85)) * (1.0 - 0.25 * (param_scale - 1.0))
        prm["recovery_bonus"] = float(prm.get("recovery_bonus", 0.03)) * float(param_scale)
    elif fam == "smooth":
        prm["slope"] = float(prm.get("slope", 0.5)) * float(param_scale)
    elif fam == "streak":
        prm["k_streak"] = int(round(float(prm.get("k_streak", 4)) * float(param_scale)))
        prm["k_tail20"] = int(round(float(prm.get("k_tail20", 8)) * float(param_scale)))
        prm["s_up"] = float(prm.get("s_up", 1.05)) * float(param_scale)
        prm["s_down"] = float(prm.get("s_down", 0.75)) * float(param_scale)
    p["params"] = prm
    return canonicalize_policy(p, pid=pid)


def build_hybrid_from_step(step_pol: Dict[str, Any], pid: str, slope: float, k_streak: int, k_tail20: int, penalty: float, recovery: float) -> Dict[str, Any]:
    p = {
        "policy_id": pid,
        "family": "hybrid",
        "weights": copy.deepcopy(step_pol.get("weights", {})),
        "params": {
            "slope": float(slope),
            "k_streak": int(k_streak),
            "k_tail20": int(k_tail20),
            "penalty_mult": float(penalty),
            "recovery_bonus": float(recovery),
        },
        "size_bounds": [0.50, 1.50],
    }
    return canonicalize_policy(p, pid=pid)


def generate_policy_neighborhood(base_policies: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(p: Dict[str, Any]) -> None:
        h = policy_hash(p)
        if h in seen:
            return
        seen.add(h)
        out.append(p)

    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    weight_scales = [0.8, 1.0, 1.2]
    shifts = [-0.08, -0.04, 0.00, 0.04, 0.08]

    for pid, base in base_policies.items():
        add(canonicalize_policy(copy.deepcopy(base), pid=f"{pid}_base"))
        fam = str(base.get("family", "step"))
        for ws in weight_scales:
            for ps in scales:
                if fam == "step":
                    for sh in shifts:
                        add(mutate_policy(base, pid=f"{pid}_w{ws:.2f}_p{ps:.2f}_s{sh:+.2f}", weight_scale=ws, param_scale=ps, threshold_shift=sh))
                else:
                    add(mutate_policy(base, pid=f"{pid}_w{ws:.2f}_p{ps:.2f}", weight_scale=ws, param_scale=ps, threshold_shift=0.0))

        if fam == "step":
            for slope in [0.25, 0.40, 0.55, 0.70]:
                for ks, kt in [(3, 8), (4, 10), (5, 12)]:
                    for penalty, rec in [(0.90, 0.02), (0.82, 0.04), (0.75, 0.06)]:
                        add(build_hybrid_from_step(base, pid=f"{pid}_hyb_s{slope:.2f}_k{ks}_{kt}_p{penalty:.2f}_r{rec:.2f}", slope=slope, k_streak=ks, k_tail20=kt, penalty=penalty, recovery=rec))

    return out


def flat_policy() -> Dict[str, Any]:
    return {
        "policy_id": "flat_baseline",
        "family": "streak_control",
        "weights": {},
        "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0},
        "size_bounds": [0.5, 1.5],
    }


def baseline_unweighted_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    valid = (df["entry_for_labels"] == 1) & df["pnl_net_trade_notional_dec"].notna()
    entries = int(valid.sum())
    entry_rate = float(entries / max(1, len(df)))
    taker = float(np.nanmean(df.loc[valid, "taker_flag"])) if entries > 0 else float("nan")
    d = to_num(df.loc[valid, "fill_delay_min"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(d, 0.95)) if d.size else float("nan")
    return {"entries_valid": entries, "entry_rate": entry_rate, "taker_share": taker, "p95_fill_delay_min": p95}


def evaluate_on_routes(
    *,
    run_dir: Path,
    signals_df: pd.DataFrame,
    genome: Dict[str, Any],
    policies: List[Dict[str, Any]],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    route_sigs = af.route_signal_sets(signals_df)
    rows: List[Dict[str, Any]] = []
    route_meta: Dict[str, Any] = {}

    for ix, (route_id, sdf) in enumerate(route_sigs.items()):
        wf_splits = 5 if route_id == "route1_holdout" else 7
        train_ratio = 0.70 if route_id == "route1_holdout" else 0.65
        met, sig, _split, _args, bundle = ae.evaluate_exact(
            run_dir=run_dir,
            signals_df=sdf,
            genome=genome,
            seed=int(seed) + 11 + ix,
            name=f"subphase1_{route_id}",
            wf_splits=wf_splits,
            train_ratio=train_ratio,
        )
        pre = ae.build_preentry_features(bundle, genome)
        d = af.parse_entry_rows(ae.build_trade_labels(sig.merge(pre, on=["signal_id", "signal_time"], how="left")))
        qstats = af.build_quantile_stats(d)
        b_unw = baseline_unweighted_from_df(d)
        flat = flat_policy()
        flat_met, _ = af.evaluate_sizing_policy_on_dataset(
            d,
            policy=flat,
            qstats=qstats,
            baseline_unweighted=b_unw,
            rng_seed=int(seed) + 101 + ix,
        )

        route_meta[route_id] = {
            "signals_total": int(len(d)),
            "entries_valid_flat": int(flat_met["entries_valid"]),
            "entry_rate_flat": float(flat_met["entry_rate"]),
            "exec_expectancy_net_weighted_flat": float(flat_met["exec_expectancy_net_weighted"]),
            "cvar_5_weighted_flat": float(flat_met["cvar_5_weighted"]),
            "max_drawdown_weighted_flat": float(flat_met["max_drawdown_weighted"]),
            "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
        }

        rows.append(
            {
                "route_id": route_id,
                "policy_id": "flat_baseline",
                "policy_hash": "flat",
                "policy_family": "flat",
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
                rng_seed=int(seed) + 1000 * (ix + 1) + j,
            )
            budget_pass = int(np.isfinite(pm["budget_norm_error"]) and abs(float(pm["budget_norm_error"])) <= 0.02)
            invariance_pass = int(
                int(pm["invariance_entries_match"]) == 1
                and int(pm["invariance_entry_rate_match"]) == 1
                and int(pm["invariance_taker_share_match"]) == 1
                and int(pm["invariance_fill_delay_match"]) == 1
            )
            rows.append(
                {
                    "route_id": route_id,
                    "policy_id": str(p["policy_id"]),
                    "policy_hash": policy_hash(p),
                    "policy_family": str(p["family"]),
                    "delta_expectancy_vs_flat": float(pm["exec_expectancy_net_weighted"] - flat_met["exec_expectancy_net_weighted"]),
                    "cvar_improve_ratio_vs_flat": safe_div(
                        abs(float(flat_met["cvar_5_weighted"])) - abs(float(pm["cvar_5_weighted"])),
                        abs(float(flat_met["cvar_5_weighted"])),
                    ),
                    "maxdd_improve_ratio_vs_flat": safe_div(
                        abs(float(flat_met["max_drawdown_weighted"])) - abs(float(pm["max_drawdown_weighted"])),
                        abs(float(flat_met["max_drawdown_weighted"])),
                    ),
                    "candidate_budget_norm_pass": int(budget_pass),
                    "candidate_invariance_pass": int(invariance_pass),
                    **pm,
                }
            )
    return pd.DataFrame(rows), route_meta


def summarize(route_df: pd.DataFrame) -> pd.DataFrame:
    x = route_df[route_df["policy_id"] != "flat_baseline"].copy()
    if x.empty:
        return pd.DataFrame()
    g = x.groupby(["policy_id", "policy_hash", "policy_family"], dropna=False).agg(
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

    g["pass_gate"] = (
        (to_num(g["min_delta_expectancy_vs_flat"]) > 0.0)
        & (to_num(g["min_maxdd_improve_ratio_vs_flat"]) > 0.0)
        & (to_num(g["min_cvar_improve_ratio_vs_flat"]) >= 0.0)
        & (to_num(g["all_budget_pass"]) == 1)
        & (to_num(g["all_invariance_pass"]) == 1)
    ).astype(int)
    g["robust_score"] = (
        8.0 * to_num(g["min_delta_expectancy_vs_flat"]).fillna(-1e9)
        + 2.0 * to_num(g["min_cvar_improve_ratio_vs_flat"]).fillna(-1e9)
        + 2.0 * to_num(g["min_maxdd_improve_ratio_vs_flat"]).fillna(-1e9)
        + 1.0 * to_num(g["mean_delta_expectancy_vs_flat"]).fillna(-1e9)
    )
    g = g.sort_values(
        ["pass_gate", "robust_score", "min_delta_expectancy_vs_flat", "min_cvar_improve_ratio_vs_flat", "min_maxdd_improve_ratio_vs_flat"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return g


def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    if df.empty:
        return "_(none)_"
    y = df.loc[:, [c for c in cols if c in df.columns]].copy()
    lines = []
    lines.append("| " + " | ".join(y.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(y.columns)) + " |")
    for r in y.itertuples(index=False):
        row_vals: List[str] = []
        for v in r:
            if isinstance(v, float):
                row_vals.append(f"{v:.8g}" if np.isfinite(v) else "nan")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sub-phase 1 targeted AF-repair sizing pilot (contract-locked, no GA)")
    ap.add_argument("--seed", type=int, default=20260223)
    args = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    sub1_dir = exec_root / f"PHASEAF_REPAIR_EXT_{utc_tag()}"
    sub1_dir.mkdir(parents=True, exist_ok=False)

    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    met_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, met_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Locked input missing: {fp}")
    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(met_fp)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha} != {LOCKED['expected_fee_sha']}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {met_sha} != {LOCKED['expected_metrics_sha']}")

    sig_in = ae.ensure_signals_schema(pd.read_csv(rep_fp))
    exec_pair = ae.load_exec_pair(exec_root)
    if exec_pair["E1"]["genome_hash"] != LOCKED["primary_hash"]:
        raise RuntimeError("Primary execution hash mismatch")
    if exec_pair["E2"]["genome_hash"] != LOCKED["backup_hash"]:
        raise RuntimeError("Backup execution hash mismatch")

    lock_args = ae.build_args(signals_csv=rep_fp, seed=int(args.seed))
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=sub1_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze_lock_pass != 1")

    af_source_csv = latest_phaseaf_results(exec_root)
    base_policies = load_base_policies(af_source_csv, LOCKED["base_policy_ids"])
    policy_grid = generate_policy_neighborhood(base_policies)
    route_df, route_meta = evaluate_on_routes(
        run_dir=sub1_dir,
        signals_df=sig_in,
        genome=copy.deepcopy(exec_pair["E1"]["genome"]),
        policies=policy_grid,
        seed=int(args.seed),
    )
    summary_df = summarize(route_df)

    route_df.to_csv(sub1_dir / "phaseAF_repair_route_results.csv", index=False)
    summary_df.to_csv(sub1_dir / "phaseAF_repair_summary.csv", index=False)

    pass_count = int((summary_df["pass_gate"] == 1).sum()) if not summary_df.empty else 0
    decision_obj = {
        "generated_utc": utc_now(),
        "phase": "SUBPHASE1_AF_REPAIR",
        "classification": "SUBPHASE1_PASS" if pass_count > 0 else "SUBPHASE1_FAIL_STOP_NO_GO",
        "go": int(pass_count > 0),
        "candidate_count": int(len(summary_df)),
        "pass_count": int(pass_count),
        "strict_cvar_reject_rule": "reject if cvar_improve_ratio_vs_flat < 0 on either route",
        "pass_gate_rule": {
            "delta_expectancy_vs_flat": "> 0 on both routes",
            "maxdd_improve_ratio_vs_flat": "> 0 on both routes",
            "cvar_improve_ratio_vs_flat": ">= 0 on both routes",
            "invariance_and_budget": "all pass",
        },
    }
    json_dump(sub1_dir / "decision.json", decision_obj)

    report_lines: List[str] = []
    report_lines.append("# Sub-phase 1: Targeted AF-Repair Sizing Pilot")
    report_lines.append("")
    report_lines.append(f"- Generated UTC: {utc_now()}")
    report_lines.append(f"- Run dir: `{sub1_dir}`")
    report_lines.append(f"- AF source policies CSV: `{af_source_csv}`")
    report_lines.append(f"- Base policy IDs: `{', '.join(LOCKED['base_policy_ids'])}`")
    report_lines.append(f"- Policy variants evaluated: `{len(summary_df)}`")
    report_lines.append(f"- Strict passers: `{pass_count}`")
    report_lines.append("")
    report_lines.append("## Route Metadata")
    report_lines.append("")
    route_rows = pd.DataFrame(
        [
            {"route_id": rid, **obj}
            for rid, obj in route_meta.items()
        ]
    )
    report_lines.append(markdown_table(route_rows, ["route_id", "signals_total", "entries_valid_flat", "entry_rate_flat", "exec_expectancy_net_weighted_flat", "cvar_5_weighted_flat", "max_drawdown_weighted_flat", "valid_for_ranking"]))
    report_lines.append("")
    report_lines.append("## Top Candidate Summary")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            summary_df.head(20),
            [
                "policy_id",
                "policy_family",
                "routes_tested",
                "pass_gate",
                "min_delta_expectancy_vs_flat",
                "min_cvar_improve_ratio_vs_flat",
                "min_maxdd_improve_ratio_vs_flat",
                "all_budget_pass",
                "all_invariance_pass",
                "robust_score",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Decision")
    report_lines.append("")
    if pass_count > 0:
        report_lines.append("- Sub-phase 1 PASS: at least one policy satisfies strict two-route gates.")
        report_lines.append("- Next: proceed to Sub-phase 2 long-horizon simulation.")
    else:
        report_lines.append("- Sub-phase 1 FAIL: no policy satisfied strict two-route gates.")
        report_lines.append("- Branch action: STOP_NO_GO (per plan).")
    write_text(sub1_dir / "phaseAF_repair_report.md", "\n".join(report_lines))

    manifest = {
        "generated_utc": utc_now(),
        "phase": "SUBPHASE1_AF_REPAIR",
        "run_dir": str(sub1_dir),
        "duration_sec": float(time.time() - t0),
        "lock": {
            "representative_subset_csv": str(rep_fp),
            "canonical_fee_model": str(fee_fp),
            "canonical_metrics_definition": str(met_fp),
            "expected_fee_sha256": LOCKED["expected_fee_sha"],
            "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
            "observed_fee_sha256": fee_sha,
            "observed_metrics_sha256": met_sha,
            "lock_validation": lock_validation,
        },
        "source": {
            "af_results_csv": str(af_source_csv),
            "base_policy_ids": LOCKED["base_policy_ids"],
            "execution_primary_hash": exec_pair["E1"]["genome_hash"],
            "execution_backup_hash": exec_pair["E2"]["genome_hash"],
        },
        "counts": {
            "policy_grid_raw": int(len(policy_grid)),
            "summary_candidates": int(len(summary_df)),
            "strict_passers": int(pass_count),
        },
        "decision_path": "STOP_NO_GO" if pass_count == 0 else "CONTINUE_SUBPHASE2",
    }
    json_dump(sub1_dir / "run_manifest.json", manifest)

    if pass_count == 0:
        nogo_dir = sub1_dir / "no_go_package"
        nogo_dir.mkdir(parents=True, exist_ok=True)
        write_text(
            nogo_dir / "reasoning.md",
            "\n".join(
                [
                    "# Sub-phase 1 NO_GO Reasoning",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Strict pass gate required positive expectancy and maxDD improvement with non-negative CVaR on both routes.",
                    "- No candidate satisfied strict gate set after ±20% neighborhood exploration around P005/P011/P030 and hybrid step-streak extensions.",
                    "- Continue path blocked by design: STOP_NO_GO.",
                ]
            ),
        )
        write_text(
            nogo_dir / "next_step_recommendation.txt",
            "NO_GO fallback: do not escalate compute to long-horizon/execution-cross/stress stack. Re-open upstream label quality and objective definitions for tail-risk prediction before any further sizing search.",
        )

    print(
        json.dumps(
            {
                "phase": "SUBPHASE1_AF_REPAIR",
                "run_dir": str(sub1_dir),
                "classification": decision_obj["classification"],
                "go": decision_obj["go"],
                "candidate_count": decision_obj["candidate_count"],
                "pass_count": decision_obj["pass_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

