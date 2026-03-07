#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    "phase_e_dir": "/root/analysis/0.87/reports/execution_layer/PHASEE_PAPER_CONFIRM_20260223_185139",
    "phase_ae_dir": "/root/analysis/0.87/reports/execution_layer/PHASEAE_SIGNAL_LABELING_20260223_111116",
    "frozen_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
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
        if isinstance(v, (datetime, pd.Timestamp)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
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


def parse_policy_dims(policy_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "policy_id": policy_id,
        "family_hint": "unknown",
        "risk_threshold": float("nan"),
        "streak_depth": float("nan"),
        "cooldown_min": float("nan"),
    }
    if policy_id.startswith("skip_risk_ge_"):
        out["family_hint"] = "risk_skip"
        m = re.search(r"skip_risk_ge_(\d+\.\d+)", policy_id)
        if m:
            out["risk_threshold"] = float(m.group(1))
    elif policy_id.startswith("skip_streak") and "cool" in policy_id:
        out["family_hint"] = "cooldown_skip"
        m1 = re.search(r"skip_streak(\d+)_cool(\d+)m", policy_id)
        if m1:
            out["streak_depth"] = float(m1.group(1))
            out["cooldown_min"] = float(m1.group(2))
    elif policy_id.startswith("skip_streak_ge_"):
        out["family_hint"] = "streak_skip"
        m = re.search(r"skip_streak_ge_(\d+)", policy_id)
        if m:
            out["streak_depth"] = float(m.group(1))
    elif policy_id.startswith("skip_tail20_ge_"):
        out["family_hint"] = "tail_skip"
        m = re.search(r"skip_tail20_ge_(\d+)", policy_id)
        if m:
            out["streak_depth"] = float(m.group(1))
    elif policy_id.startswith("skip_risk") and "_streak" in policy_id:
        out["family_hint"] = "risk_streak_combo"
        m = re.search(r"skip_risk(\d+\.\d+)_streak(\d+)", policy_id)
        if m:
            out["risk_threshold"] = float(m.group(1))
            out["streak_depth"] = float(m.group(2))
    return out


def validate_contract(run_dir: Path, frozen_subset_csv: Path, fee_path: Path, metrics_path: Path, seed: int) -> Dict[str, Any]:
    for fp in (frozen_subset_csv, fee_path, metrics_path):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")
    fee_sha = sha256_file(fee_path)
    met_sha = sha256_file(metrics_path)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {met_sha}")

    sig_raw = pd.read_csv(frozen_subset_csv)
    sig = ae.ensure_signals_schema(sig_raw)
    if sig.empty:
        raise RuntimeError("Frozen subset loaded empty")

    lock_args = ae.build_args(signals_csv=frozen_subset_csv, seed=int(seed))
    lock_args.allow_freeze_hash_mismatch = 0
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=run_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze_lock_pass != 1")
    return {
        "generated_utc": utc_now(),
        "subset_rows": int(len(sig)),
        "fee_sha256": fee_sha,
        "metrics_sha256": met_sha,
        "freeze_lock_validation": lock_validation,
    }


def f1_forensics(phase_e_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    e2 = pd.read_csv(phase_e_dir / "phaseE2" / "route_perturb_matrix.csv")
    e3 = pd.read_csv(phase_e_dir / "phaseE3" / "stress_matrix_results.csv")
    e3_route = pd.read_csv(phase_e_dir / "phaseE3" / "stress_matrix_results_by_route.csv")

    x = e3.copy()
    x["fail_delta"] = (to_num(x["min_delta_expectancy_vs_flat"]) <= 0.0).astype(int)
    x["fail_cvar"] = (to_num(x["min_cvar_improve_ratio_vs_flat"]) < 0.0).astype(int)
    x["fail_dd"] = (to_num(x["min_maxdd_improve_ratio_vs_flat"]) <= 0.0).astype(int)
    x["fail_subperiod_delta"] = (to_num(x["min_subperiod_delta"]) <= 0.0).astype(int)
    x["fail_subperiod_cvar"] = (to_num(x["min_subperiod_cvar_improve"]) < 0.0).astype(int)
    x["fail_participation"] = (to_num(x["min_filter_kept_entries_pct"]) < 0.60).astype(int)
    x["fail_pathology"] = (to_num(x["no_pathology"]) != 1).astype(int)
    x["scenario_fail"] = (to_num(x["scenario_pass"]) != 1).astype(int)

    # Scenario x reason breakdown.
    reason_cols = [
        "fail_delta",
        "fail_cvar",
        "fail_dd",
        "fail_subperiod_delta",
        "fail_subperiod_cvar",
        "fail_participation",
        "fail_pathology",
    ]
    br_rows: List[Dict[str, Any]] = []
    for sid, g in x.groupby("scenario_id", dropna=False):
        n = len(g)
        for rc in reason_cols:
            cnt = int(to_num(g[rc]).sum())
            br_rows.append(
                {
                    "scope": "scenario",
                    "scope_id": str(sid),
                    "reason": rc,
                    "fail_count": cnt,
                    "fail_pct": safe_div(float(cnt), float(max(1, n))),
                    "support_n": int(n),
                }
            )
    for pid, g in x.groupby("policy_id", dropna=False):
        n = len(g)
        for rc in reason_cols:
            cnt = int(to_num(g[rc]).sum())
            br_rows.append(
                {
                    "scope": "policy",
                    "scope_id": str(pid),
                    "reason": rc,
                    "fail_count": cnt,
                    "fail_pct": safe_div(float(cnt), float(max(1, n))),
                    "support_n": int(n),
                }
            )

    # Route-focused weakness in stress by-route matrix.
    rr = e3_route.copy()
    rr["fail_delta_route"] = (to_num(rr["delta_expectancy_vs_flat"]) <= 0.0).astype(int)
    rr["fail_cvar_route"] = (to_num(rr["cvar_improve_ratio_vs_flat"]) < 0.0).astype(int)
    rr["fail_dd_route"] = (to_num(rr["maxdd_improve_ratio_vs_flat"]) <= 0.0).astype(int)
    route_sum = (
        rr.groupby(["route_id"], dropna=False)
        .agg(
            rows=("scenario_id", "size"),
            fail_delta_route=("fail_delta_route", "sum"),
            fail_cvar_route=("fail_cvar_route", "sum"),
            fail_dd_route=("fail_dd_route", "sum"),
            mean_delta=("delta_expectancy_vs_flat", "mean"),
            mean_cvar_imp=("cvar_improve_ratio_vs_flat", "mean"),
            mean_dd_imp=("maxdd_improve_ratio_vs_flat", "mean"),
        )
        .reset_index()
    )
    for _, r in route_sum.iterrows():
        for rc in ["fail_delta_route", "fail_cvar_route", "fail_dd_route"]:
            br_rows.append(
                {
                    "scope": "route",
                    "scope_id": str(r["route_id"]),
                    "reason": rc,
                    "fail_count": int(r[rc]),
                    "fail_pct": safe_div(float(r[rc]), float(max(1, r["rows"]))),
                    "support_n": int(r["rows"]),
                }
            )

    breakdown = pd.DataFrame(br_rows).sort_values(["scope", "scope_id", "reason"]).reset_index(drop=True)
    breakdown.to_csv(out_dir / "phaseF1_failure_mode_breakdown.csv", index=False)

    # Policy dimension attribution table.
    p = x[
        [
            "policy_id",
            "min_delta_expectancy_vs_flat",
            "min_cvar_improve_ratio_vs_flat",
            "min_maxdd_improve_ratio_vs_flat",
            "min_subperiod_delta",
            "min_subperiod_cvar_improve",
            "min_filter_kept_entries_pct",
            "scenario_pass",
            "strict_pass",
        ]
    ].copy()
    pdims = pd.DataFrame([parse_policy_dims(pid) for pid in p["policy_id"].astype(str).unique()])
    pattr = (
        p.groupby("policy_id", dropna=False)
        .agg(
            scenarios=("scenario_pass", "size"),
            scenario_pass_rate=("scenario_pass", "mean"),
            strict_pass_rate=("strict_pass", "mean"),
            mean_min_delta=("min_delta_expectancy_vs_flat", "mean"),
            mean_min_cvar=("min_cvar_improve_ratio_vs_flat", "mean"),
            mean_min_dd=("min_maxdd_improve_ratio_vs_flat", "mean"),
            min_subperiod_delta=("min_subperiod_delta", "min"),
            min_subperiod_cvar=("min_subperiod_cvar_improve", "min"),
            min_kept_entries_pct=("min_filter_kept_entries_pct", "min"),
        )
        .reset_index()
    )
    # Join D3 base metrics for the same policies.
    d3 = pd.read_csv(phase_e_dir / "phaseE1" / "phaseE1_repro_results.csv")
    d3 = d3[
        [
            "policy_id",
            "min_delta_expectancy_vs_flat",
            "min_cvar_improve_ratio_vs_flat",
            "min_maxdd_improve_ratio_vs_flat",
            "strict_pass",
        ]
    ].rename(
        columns={
            "min_delta_expectancy_vs_flat": "base_min_delta",
            "min_cvar_improve_ratio_vs_flat": "base_min_cvar",
            "min_maxdd_improve_ratio_vs_flat": "base_min_dd",
            "strict_pass": "base_strict_pass",
        }
    )
    attrib = pattr.merge(pdims, on="policy_id", how="left").merge(d3, on="policy_id", how="left")
    attrib = attrib.sort_values(["scenario_pass_rate", "mean_min_delta"], ascending=[True, True]).reset_index(drop=True)
    attrib.to_csv(out_dir / "phaseF1_policy_failure_attribution.csv", index=False)

    top_causes = (
        breakdown[breakdown["scope"] == "scenario"]
        .groupby("reason", dropna=False)["fail_count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    lines = []
    lines.append("# Phase F1 Forensics Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append("- Source Phase E run: `{}`".format(phase_e_dir))
    lines.append(f"- E2 rows: `{len(e2)}`")
    lines.append(f"- E3 rows: `{len(e3)}`")
    lines.append(f"- E3 by-route rows: `{len(e3_route)}`")
    lines.append("")
    lines.append("## Top Failure Causes (aggregate)")
    lines.append("")
    lines.append(markdown_table(top_causes, ["reason", "fail_count"]))
    lines.append("")
    lines.append("## Scenario-level breakdown (head)")
    lines.append("")
    lines.append(markdown_table(breakdown[breakdown["scope"] == "scenario"].head(30), ["scope_id", "reason", "fail_count", "fail_pct", "support_n"]))
    lines.append("")
    lines.append("## Policy attribution")
    lines.append("")
    lines.append(
        markdown_table(
            attrib,
            [
                "policy_id",
                "family_hint",
                "risk_threshold",
                "streak_depth",
                "cooldown_min",
                "scenario_pass_rate",
                "mean_min_delta",
                "mean_min_cvar",
                "mean_min_dd",
                "min_subperiod_delta",
                "min_subperiod_cvar",
                "min_kept_entries_pct",
                "base_strict_pass",
            ],
        )
    )
    write_text(out_dir / "phaseF1_forensics_report.md", "\n".join(lines))
    return breakdown, attrib


def build_route_data(seed: int, run_dir: Path, frozen_subset_csv: Path) -> Dict[str, pd.DataFrame]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sig_in = ae.ensure_signals_schema(pd.read_csv(frozen_subset_csv))
    exec_pair = ae.load_exec_pair(PROJECT_ROOT / "reports" / "execution_layer")
    if exec_pair["E1"]["genome_hash"] != LOCKED["primary_exec_hash"]:
        raise RuntimeError("Primary hash mismatch")
    return dmod.evaluate_baseline_routes(
        run_dir=run_dir,
        sig_in=sig_in,
        genome=copy.deepcopy(exec_pair["E1"]["genome"]),
        seed=int(seed),
    )


def make_soft_variants() -> List[Dict[str, Any]]:
    return [
        {"variant_id": "baseline_flat", "family": "flat", "params": {}},
        {"variant_id": "soft_lin_mild", "family": "risk_linear", "params": {"a0": 1.06, "a1": 0.20, "floor": 0.78, "cap": 1.18}},
        {"variant_id": "soft_lin_mid", "family": "risk_linear", "params": {"a0": 1.08, "a1": 0.30, "floor": 0.72, "cap": 1.20}},
        {"variant_id": "soft_lin_aggr", "family": "risk_linear", "params": {"a0": 1.10, "a1": 0.40, "floor": 0.65, "cap": 1.24}},
        {"variant_id": "soft_step_mild", "family": "risk_step", "params": {"t1": 0.55, "t2": 0.75, "s_low": 1.06, "s_mid": 0.92, "s_high": 0.82}},
        {"variant_id": "soft_step_mid", "family": "risk_step", "params": {"t1": 0.50, "t2": 0.72, "s_low": 1.08, "s_mid": 0.90, "s_high": 0.76}},
        {"variant_id": "state_streak_cap3", "family": "state_streak_cap", "params": {"base_a0": 1.08, "base_a1": 0.28, "streak_k": 3, "streak_cap": 0.82, "floor": 0.72, "cap": 1.18}},
        {"variant_id": "state_tail_cap8", "family": "state_tail_cap", "params": {"base_a0": 1.08, "base_a1": 0.30, "tail_k": 8, "tail_cap": 0.80, "floor": 0.72, "cap": 1.18}},
        {
            "variant_id": "hybrid_stateaware_v1",
            "family": "hybrid_stateaware",
            "params": {
                "a0": 1.09,
                "a1": 0.32,
                "streak_k": 3,
                "tail_k": 8,
                "state_cap": 0.82,
                "recovery_bonus": 0.04,
                "floor": 0.72,
                "cap": 1.18,
            },
        },
        {
            "variant_id": "hybrid_stateaware_v2",
            "family": "hybrid_stateaware",
            "params": {
                "a0": 1.10,
                "a1": 0.38,
                "streak_k": 2,
                "tail_k": 7,
                "state_cap": 0.78,
                "recovery_bonus": 0.05,
                "floor": 0.68,
                "cap": 1.20,
            },
        },
    ]


def base_risk_score(df_feat: pd.DataFrame) -> pd.Series:
    # AE-guided non-leaky blend: streak/tail + close-high + interaction.
    s = (
        0.24 * to_num(df_feat["f_streak"]).fillna(0.0)
        + 0.22 * to_num(df_feat["f_tail20"]).fillna(0.0)
        + 0.22 * to_num(df_feat["f_closehigh"]).fillna(0.0)
        + 0.18 * to_num(df_feat["f_int1"]).fillna(0.0)
        + 0.14 * to_num(df_feat["f_int2"]).fillna(0.0)
    )
    return s.clip(lower=0.0, upper=1.0)


def sized_metrics_from_df(df: pd.DataFrame, size_mult: pd.Series) -> Dict[str, Any]:
    x = df.copy()
    valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
    n = len(x)
    pnl = np.zeros(n, dtype=float)
    pnl[valid.to_numpy(dtype=bool)] = (to_num(x.loc[valid, "pnl_net_trade_notional_dec"]) * size_mult.loc[valid]).to_numpy(dtype=float)
    exp = float(np.mean(pnl)) if n else float("nan")
    k5 = max(1, int(math.ceil(0.05 * max(1, len(pnl)))))
    cvar = float(np.mean(np.sort(pnl)[:k5])) if len(pnl) else float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    dd = cum - np.maximum.accumulate(cum) if len(cum) else np.array([], dtype=float)
    mdd = float(np.nanmin(dd)) if dd.size else float("nan")

    entries = int(valid.sum())
    entry_rate = float(entries / max(1, n))
    taker = float(np.nanmean(to_num(x.loc[valid, "taker_flag"]))) if entries > 0 else float("nan")
    delays = to_num(x.loc[valid, "fill_delay_min"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(delays, 0.95)) if delays.size else float("nan")

    z = x.copy()
    z["pnl_eval"] = 0.0
    z.loc[valid, "pnl_eval"] = pnl[valid.to_numpy(dtype=bool)]
    split_exp = z.groupby("split_id", dropna=True)["pnl_eval"].mean()
    min_split = float(split_exp.min()) if not split_exp.empty else float("nan")

    # unweighted clustering (count-based; expected invariant for sizing-only).
    t = x.loc[valid].copy().sort_values(["entry_time_utc", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    un = to_num(t["pnl_net_trade_notional_dec"]).to_numpy(dtype=float)
    losses = un < 0
    streaks: List[int] = []
    i = 0
    while i < len(losses):
        if losses[i]:
            s = i
            while i + 1 < len(losses) and losses[i + 1]:
                i += 1
            streaks.append(i - s + 1)
        i += 1
    arr = np.asarray(streaks, dtype=int) if streaks else np.array([], dtype=int)
    max_streak = int(arr.max()) if arr.size else 0
    ge5 = int(np.sum(arr >= 5)) if arr.size else 0

    return {
        "exec_expectancy_net": float(exp),
        "exec_cvar_5": float(cvar),
        "exec_max_drawdown": float(mdd),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "taker_share": float(taker),
        "p95_fill_delay_min": float(p95),
        "min_split_expectancy_net": float(min_split),
        "max_consecutive_losses": int(max_streak),
        "streak_ge5_count": int(ge5),
    }


def apply_soft_variant(df_feat: pd.DataFrame, variant: Dict[str, Any]) -> pd.Series:
    x = df_feat.copy()
    valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
    s = pd.Series(np.ones(len(x), dtype=float), index=x.index)
    fam = str(variant["family"])
    p = variant["params"]
    r = base_risk_score(x)
    if fam == "flat":
        s[valid] = 1.0
    elif fam == "risk_linear":
        raw = to_num(p["a0"]) - to_num(p["a1"]) * r
        s[valid] = raw.loc[valid]
    elif fam == "risk_step":
        t1, t2 = float(p["t1"]), float(p["t2"])
        s_low, s_mid, s_high = float(p["s_low"]), float(p["s_mid"]), float(p["s_high"])
        raw = np.where(r <= t1, s_low, np.where(r <= t2, s_mid, s_high))
        s[valid] = raw[valid]
    elif fam == "state_streak_cap":
        raw = float(p["base_a0"]) - float(p["base_a1"]) * r
        state = to_num(x["prior_loss_streak_len"]).fillna(0.0) >= float(p["streak_k"])
        raw = np.where(state, np.minimum(raw, float(p["streak_cap"])), raw)
        s[valid] = raw[valid]
    elif fam == "state_tail_cap":
        raw = float(p["base_a0"]) - float(p["base_a1"]) * r
        state = to_num(x["prior_rolling_tail_count_20"]).fillna(0.0) >= float(p["tail_k"])
        raw = np.where(state, np.minimum(raw, float(p["tail_cap"])), raw)
        s[valid] = raw[valid]
    elif fam == "hybrid_stateaware":
        raw = float(p["a0"]) - float(p["a1"]) * r
        state = (to_num(x["prior_loss_streak_len"]).fillna(0.0) >= float(p["streak_k"])) | (
            to_num(x["prior_rolling_tail_count_20"]).fillna(0.0) >= float(p["tail_k"])
        )
        raw = np.where(state, np.minimum(raw, float(p["state_cap"])), raw)
        rec = (to_num(x["prior_loss_streak_len"]).fillna(0.0) <= 1.0) & (to_num(x["prior_rolling_loss_rate_5"]).fillna(0.0) <= 0.35)
        raw = np.where(rec, raw + float(p["recovery_bonus"]), raw)
        s[valid] = raw[valid]
    else:
        s[valid] = 1.0

    floor = float(p.get("floor", 0.60))
    cap = float(p.get("cap", 1.25))
    s = s.clip(lower=floor, upper=cap)
    if int(valid.sum()) > 0:
        mean_s = float(np.nanmean(to_num(s.loc[valid])))
        if np.isfinite(mean_s) and mean_s > 1e-12:
            s.loc[valid] = s.loc[valid] / mean_s
    s = s.clip(lower=floor, upper=cap)
    if int(valid.sum()) > 0:
        mean_s2 = float(np.nanmean(to_num(s.loc[valid])))
        if np.isfinite(mean_s2) and mean_s2 > 1e-12:
            s.loc[valid] = s.loc[valid] / mean_s2
    s = s.clip(lower=floor, upper=cap)
    return s.astype(float)


def f3_ablation(route_data: Dict[str, pd.DataFrame], out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int], str, str]:
    variants = make_soft_variants()
    rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []

    # route-local feature prep with shared quantiles.
    all_entries = pd.concat([d.copy() for d in route_data.values()], axis=0, ignore_index=True)
    qstats = af.build_quantile_stats(all_entries)
    route_feat: Dict[str, pd.DataFrame] = {}
    route_base: Dict[str, Dict[str, Any]] = {}
    for rid, d in route_data.items():
        df = d.copy()
        df = af.compute_policy_features(df, qstats=qstats)
        df = dmod.build_s1_risk_features(df, qstats=qstats)
        route_feat[rid] = df
        route_base[rid] = sized_metrics_from_df(df, pd.Series(np.ones(len(df), dtype=float), index=df.index))

    for var in variants:
        rid_rows: List[Dict[str, Any]] = []
        for rid, df in route_feat.items():
            size = apply_soft_variant(df, var)
            met = sized_metrics_from_df(df, size)
            base = route_base[rid]
            delta = float(met["exec_expectancy_net"] - base["exec_expectancy_net"])
            cvar_imp = safe_div(abs(float(base["exec_cvar_5"])) - abs(float(met["exec_cvar_5"])), abs(float(base["exec_cvar_5"])))
            dd_imp = safe_div(abs(float(base["exec_max_drawdown"])) - abs(float(met["exec_max_drawdown"])), abs(float(base["exec_max_drawdown"])))
            valid = (df["entry_for_labels"] == 1) & to_num(df["pnl_net_trade_notional_dec"]).notna()
            kept_entries_pct = float(np.mean(size.loc[valid] > 0.0)) if int(valid.sum()) > 0 else float("nan")
            row = {
                "variant_id": str(var["variant_id"]),
                "variant_family": str(var["family"]),
                "route_id": str(rid),
                "exec_expectancy_net": float(met["exec_expectancy_net"]),
                "delta_expectancy_vs_baseline": delta,
                "cvar_improve_ratio": float(cvar_imp),
                "maxdd_improve_ratio": float(dd_imp),
                "max_consecutive_losses": int(met["max_consecutive_losses"]),
                "streak_ge5_count": int(met["streak_ge5_count"]),
                "entries_valid": int(met["entries_valid"]),
                "entry_rate": float(met["entry_rate"]),
                "taker_share": float(met["taker_share"]),
                "p95_fill_delay_min": float(met["p95_fill_delay_min"]),
                "min_split_expectancy_net": float(met["min_split_expectancy_net"]),
                "size_mean_valid": float(np.nanmean(to_num(size.loc[valid]))) if int(valid.sum()) > 0 else float("nan"),
                "size_p10_valid": float(np.nanquantile(to_num(size.loc[valid]), 0.10)) if int(valid.sum()) > 0 else float("nan"),
                "size_p90_valid": float(np.nanquantile(to_num(size.loc[valid]), 0.90)) if int(valid.sum()) > 0 else float("nan"),
                "kept_entries_pct_proxy": kept_entries_pct,
            }
            route_rows.append(row)
            rid_rows.append(row)

        g = pd.DataFrame(rid_rows)
        min_delta = float(to_num(g["delta_expectancy_vs_baseline"]).min())
        min_cvar = float(to_num(g["cvar_improve_ratio"]).min())
        min_dd = float(to_num(g["maxdd_improve_ratio"]).min())
        min_entries = int(to_num(g["entries_valid"]).min())
        min_entry_rate = float(to_num(g["entry_rate"]).min())
        min_kept = float(to_num(g["kept_entries_pct_proxy"]).min())
        no_pathology = int(
            np.isfinite(min_delta)
            and np.isfinite(min_cvar)
            and np.isfinite(min_dd)
            and np.isfinite(min_entry_rate)
            and min_entries > 0
        )
        # valid_for_ranking proxy under unchanged strict multi-route gates.
        valid_for_ranking = int((min_delta > 0.0) and (min_cvar >= 0.0) and (min_dd > 0.0) and (no_pathology == 1))
        invalid_reason = ""
        if valid_for_ranking != 1:
            reasons: List[str] = []
            if min_delta <= 0.0:
                reasons.append("min_delta_expectancy<=0")
            if min_cvar < 0.0:
                reasons.append("min_cvar_improve<0")
            if min_dd <= 0.0:
                reasons.append("min_maxdd_improve<=0")
            if no_pathology != 1:
                reasons.append("pathology")
            invalid_reason = ";".join(reasons)

        rows.append(
            {
                "variant_id": str(var["variant_id"]),
                "variant_family": str(var["family"]),
                "valid_for_ranking": int(valid_for_ranking),
                "invalid_reason": invalid_reason,
                "exec_expectancy_net": float(to_num(g["exec_expectancy_net"]).mean()),
                "delta_expectancy_vs_baseline": float(to_num(g["delta_expectancy_vs_baseline"]).mean()),
                "cvar_improve_ratio": float(to_num(g["cvar_improve_ratio"]).mean()),
                "maxdd_improve_ratio": float(to_num(g["maxdd_improve_ratio"]).mean()),
                "max_consecutive_losses": int(to_num(g["max_consecutive_losses"]).max()),
                "streak_ge5_count": int(to_num(g["streak_ge5_count"]).max()),
                "entries_valid": int(to_num(g["entries_valid"]).min()),
                "entry_rate": float(to_num(g["entry_rate"]).min()),
                "min_delta_expectancy_vs_baseline": min_delta,
                "min_cvar_improve_ratio": min_cvar,
                "min_maxdd_improve_ratio": min_dd,
                "min_kept_entries_pct_proxy": min_kept,
                "no_pathology": int(no_pathology),
            }
        )

    vdf = pd.DataFrame(rows).sort_values(
        ["valid_for_ranking", "min_delta_expectancy_vs_baseline", "min_cvar_improve_ratio", "min_maxdd_improve_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    rdf = pd.DataFrame(route_rows).sort_values(["variant_id", "route_id"]).reset_index(drop=True)
    vdf.to_csv(out_dir / "phaseF3_ablation_results.csv", index=False)
    rdf.to_csv(out_dir / "phaseF3_ablation_results_by_route.csv", index=False)

    invalid_hist: Dict[str, int] = {}
    for _, r in vdf.iterrows():
        if int(r["valid_for_ranking"]) == 1:
            continue
        for rs in str(r["invalid_reason"]).split(";"):
            rs = rs.strip()
            if not rs:
                continue
            invalid_hist[rs] = int(invalid_hist.get(rs, 0) + 1)
    json_dump(out_dir / "phaseF3_invalid_reason_histogram.json", invalid_hist)

    # GO/NO_GO decision.
    pass_df = vdf[vdf["valid_for_ranking"] == 1].copy()
    if pass_df.empty:
        cls = "STOP_NO_GO"
        reason = "no variant reached valid_for_ranking under strict multi-route gates"
    else:
        # Require risk-side benefit not starvation artifact and no severe expectancy collapse.
        good = pass_df[
            (to_num(pass_df["cvar_improve_ratio"]) > 0.0)
            | (to_num(pass_df["maxdd_improve_ratio"]) > 0.02)
        ].copy()
        good = good[to_num(good["min_kept_entries_pct_proxy"]) >= 0.60]
        if good.empty:
            cls = "STOP_NO_GO"
            reason = "valid variants exist but no non-trivial robustness benefit beyond baseline"
        else:
            cls = "CONTINUE_READY_FOR_PILOT"
            reason = ">=1 soft state-aware variant is valid and shows robustness-oriented benefit without starvation"
    return vdf, invalid_hist, cls, reason


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase F NO_GO recovery + family pivot (no GA)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--phase_e_dir", default=LOCKED["phase_e_dir"])
    ap.add_argument("--phase_ae_dir", default=LOCKED["phase_ae_dir"])
    ap.add_argument("--frozen_subset_csv", default=LOCKED["frozen_subset_csv"])
    ap.add_argument("--fee_path", default=LOCKED["canonical_fee_model"])
    ap.add_argument("--metrics_path", default=LOCKED["canonical_metrics_definition"])
    args = ap.parse_args()

    root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = root / f"PHASEF_NO_GO_RECOVERY_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    try:
        lock_obj = validate_contract(
            run_dir=run_dir,
            frozen_subset_csv=Path(args.frozen_subset_csv).resolve(),
            fee_path=Path(args.fee_path).resolve(),
            metrics_path=Path(args.metrics_path).resolve(),
            seed=int(args.seed),
        )
    except Exception as e:
        write_text(run_dir / "phaseF_infra_or_contract_fail.md", f"Validation failure: {e}")
        final = {
            "furthest_phase": "F0",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": str(e),
        }
        json_dump(run_dir / "phaseF_run_manifest.json", final)
        print(json.dumps({"status": "STOP_INFRA", "run_dir": str(run_dir), "error": str(e)}))
        return
    json_dump(run_dir / "phaseF_freeze_lock_validation.json", lock_obj)

    phase_e_dir = Path(args.phase_e_dir).resolve()
    phase_ae_dir = Path(args.phase_ae_dir).resolve()
    if not phase_e_dir.exists() or not phase_ae_dir.exists():
        write_text(run_dir / "phaseF_infra_or_contract_fail.md", "Missing phaseE or phaseAE source directory.")
        final = {
            "furthest_phase": "F0",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": "source directories missing",
        }
        json_dump(run_dir / "phaseF_run_manifest.json", final)
        print(json.dumps({"status": "STOP_INFRA", "run_dir": str(run_dir), "error": "source directories missing"}))
        return

    # F1
    try:
        f1_breakdown, f1_attrib = f1_forensics(phase_e_dir, run_dir)
    except Exception as e:
        write_text(run_dir / "phaseF1_forensics_report.md", f"# Phase F1 Forensics Report\n\n- FAIL: {e}")
        final = {
            "furthest_phase": "F1",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": str(e),
        }
        json_dump(run_dir / "phaseF_run_manifest.json", final)
        print(json.dumps({"status": "STOP_INFRA", "run_dir": str(run_dir), "error": str(e)}))
        return

    # F2
    write_text(
        run_dir / "phaseF2_next_family_spec.md",
        "\n".join(
            [
                "# Phase F2 Next Family Spec",
                "",
                f"- Generated UTC: {utc_now()}",
                "- Family objective: replace brittle hard skip/cooldown gating with soft, state-aware risk sizing.",
                "- Primary evidence basis:",
                "  - Phase E: hard-filter family failed due to subperiod expectancy fragility, not global base-route pass failure.",
                "  - Phase AE/AF: soft risk sizing proxies improved DD/tail metrics while preserving participation/invariance.",
                "",
                "## Proposed Control Families",
                "",
                "1. `risk_linear`: continuous down-scaling as AE risk score rises.",
                "2. `risk_step`: interpretable piecewise sizing on risk score bins.",
                "3. `state_streak_cap`: apply temporary sizing cap during elevated prior loss streak state.",
                "4. `state_tail_cap`: apply sizing cap during elevated rolling tail-count state.",
                "5. `hybrid_stateaware`: risk-linear base + state cap + controlled recovery bonus.",
                "",
                "## Why this family",
                "",
                "- Avoids hard trade starvation and route/subperiod over-fragility from binary skip gates.",
                "- Directly targets AE predictors (`prior_loss_streak_len`, `prior_rolling_tail_count_20`, `pre3m_close_to_high_dist_bps`, interaction terms).",
                "- Preserves existing execution mechanics and hard gates; only notional scaling changes.",
            ]
        ),
    )
    write_text(
        run_dir / "phaseF2_param_bounds.yaml",
        "\n".join(
            [
                "family_bounds:",
                "  risk_linear:",
                "    a0: [1.04, 1.12]",
                "    a1: [0.18, 0.45]",
                "    floor: [0.65, 0.80]",
                "    cap: [1.15, 1.25]",
                "  risk_step:",
                "    t1: [0.45, 0.60]",
                "    t2: [0.68, 0.80]",
                "    s_low: [1.03, 1.10]",
                "    s_mid: [0.86, 0.94]",
                "    s_high: [0.70, 0.85]",
                "  state_streak_cap:",
                "    streak_k: [2, 4]",
                "    streak_cap: [0.78, 0.86]",
                "    base_a0: [1.05, 1.12]",
                "    base_a1: [0.22, 0.35]",
                "  state_tail_cap:",
                "    tail_k: [7, 10]",
                "    tail_cap: [0.76, 0.84]",
                "    base_a0: [1.05, 1.12]",
                "    base_a1: [0.24, 0.36]",
                "  hybrid_stateaware:",
                "    a0: [1.07, 1.12]",
                "    a1: [0.28, 0.42]",
                "    streak_k: [2, 4]",
                "    tail_k: [7, 10]",
                "    state_cap: [0.76, 0.85]",
                "    recovery_bonus: [0.02, 0.06]",
                "global_constraints:",
                "  mean_size_normalization: true",
                "  size_floor_hard: 0.60",
                "  size_cap_hard: 1.25",
                "  no_entry_exit_mechanics_change: true",
                "  hard_gates_unchanged: true",
            ]
        ),
    )
    write_text(
        run_dir / "phaseF2_mapping_to_existing_engines.md",
        "\n".join(
            [
                "# Phase F2 Mapping to Existing Engines",
                "",
                "## Inputs and route construction",
                "- Route data generation: `scripts.phase_d123_tail_filter.evaluate_baseline_routes()`",
                "- Uses locked E1 execution genome and frozen representative subset.",
                "",
                "## Feature construction",
                "- Base feature transforms: `scripts.phase_af_ah_sizing_autorun.compute_policy_features()`",
                "- AE-aligned risk score helper: `scripts.phase_d123_tail_filter.build_s1_risk_features()`",
                "",
                "## Metric and gate evaluation",
                "- Route-level weighted metrics computed with the same signal-level pnl vector convention used in AF/AG.",
                "- Strict valid_for_ranking proxy kept unchanged from prior branch gate semantics:",
                "  - min_delta_expectancy_vs_baseline > 0",
                "  - min_cvar_improve_ratio >= 0",
                "  - min_maxdd_improve_ratio > 0",
                "",
                "## Scope guard",
                "- No hard-gate changes.",
                "- No execution entry/exit mechanic changes.",
                "- No GA in this phase (direct 6-12 variant ablation only).",
            ]
        ),
    )

    # F3
    try:
        route_data = build_route_data(seed=int(args.seed), run_dir=run_dir / "phaseF3_route_cache", frozen_subset_csv=Path(args.frozen_subset_csv).resolve())
        vdf, invalid_hist, cls, reason = f3_ablation(route_data, run_dir)
    except Exception as e:
        write_text(run_dir / "phaseF3_ablation_report.md", f"# Phase F3 Ablation Report\n\n- FAIL: {e}")
        final = {
            "furthest_phase": "F3",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": str(e),
        }
        json_dump(run_dir / "phaseF_run_manifest.json", final)
        print(json.dumps({"status": "STOP_INFRA", "run_dir": str(run_dir), "error": str(e)}))
        return

    lines = []
    lines.append("# Phase F3 Ablation Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Decision class: **{cls}**")
    lines.append(f"- Reason: {reason}")
    lines.append(f"- Variants tested (incl baseline): `{len(vdf)}`")
    lines.append(f"- valid_for_ranking count: `{int((to_num(vdf['valid_for_ranking']) == 1).sum())}`")
    lines.append("")
    lines.append("## Top variants")
    lines.append("")
    lines.append(
        markdown_table(
            vdf.head(12),
            [
                "variant_id",
                "variant_family",
                "valid_for_ranking",
                "min_delta_expectancy_vs_baseline",
                "min_cvar_improve_ratio",
                "min_maxdd_improve_ratio",
                "entries_valid",
                "entry_rate",
                "invalid_reason",
            ],
        )
    )
    write_text(run_dir / "phaseF3_ablation_report.md", "\n".join(lines))

    # Final decision
    if cls == "CONTINUE_READY_FOR_PILOT":
        decision_class = "CONTINUE_READY_FOR_PILOT"
        mainline = "CONTINUE_READY_FOR_PILOT"
        decision_text = "GO: next-family controlled pilot is justified (no full GA yet)."
        prompt_txt = "\n".join(
            [
                "ROLE",
                "You are in Phase G controlled-pilot mode for the soft state-aware sizing family.",
                "",
                "MISSION",
                "Run a small, contract-locked pilot (no full GA) around the top Phase F soft variants only.",
                "",
                "RULES",
                "1) Hard gates unchanged.",
                "2) No execution mechanics changes.",
                "3) Candidate budget <= 80.",
                "4) Include route perturb + stress mini-check in-pilot.",
                "5) Stop with NO_GO if no robust strict passers.",
            ]
        )
        write_text(run_dir / "ready_to_launch_phaseG_pilot_prompt.txt", prompt_txt)
    else:
        decision_class = "STOP_NO_GO"
        mainline = "STOP_NO_GO"
        decision_text = "NO_GO: no credible soft-family variant met strict multi-route validity with robust risk benefit."

    write_text(
        run_dir / "phaseF_decision_next_step.md",
        "\n".join(
            [
                "# Phase F Decision",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- Classification: **{decision_class}**",
                f"- Mainline status: **{mainline}**",
                f"- Decision rationale: {decision_text}",
                f"- Source Phase E: `{phase_e_dir}`",
                f"- Source Phase AE: `{phase_ae_dir}`",
            ]
        ),
    )

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "source_phase_e": str(phase_e_dir),
        "source_phase_ae": str(phase_ae_dir),
        "classification": decision_class,
        "mainline_status": mainline,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "valid_for_ranking_count": int((to_num(vdf["valid_for_ranking"]) == 1).sum()),
    }
    json_dump(run_dir / "phaseF_run_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "classification": decision_class, "mainline_status": mainline}, sort_keys=True))


if __name__ == "__main__":
    main()
