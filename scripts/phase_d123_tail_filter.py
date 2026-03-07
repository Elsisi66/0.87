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
    "repo_root": "/root/analysis/0.87",
    "symbol": "SOLUSDT",
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


def policy_hash(obj: Dict[str, Any]) -> str:
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def load_latest_phasec(exec_root: Path) -> Tuple[Path, Path, Path]:
    roots = sorted([p for p in exec_root.glob("PHASEABC_LABEL_REPAIR_*") if p.is_dir()], key=lambda p: p.name)
    for r in reversed(roots):
        phase_cs = sorted([p for p in r.glob("phaseC_*") if p.is_dir()], key=lambda p: p.name)
        if not phase_cs:
            continue
        c = phase_cs[-1]
        summary = c / "phaseC_results.csv"
        by_route = c / "phaseC_results_by_route.csv"
        if summary.exists() and by_route.exists():
            return r, c, by_route
    raise FileNotFoundError("No prior PHASEABC_LABEL_REPAIR with phaseC results found")


def build_s1_risk_features(df: pd.DataFrame, qstats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    z = af.compute_policy_features(df.copy(), qstats=qstats)
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


def evaluate_baseline_routes(
    *,
    run_dir: Path,
    sig_in: pd.DataFrame,
    genome: Dict[str, Any],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    routes = af.route_signal_sets(sig_in)
    out: Dict[str, pd.DataFrame] = {}
    for ix, (route_id, sdf) in enumerate(routes.items()):
        wf_splits = 5 if route_id == "route1_holdout" else 7
        train_ratio = 0.70 if route_id == "route1_holdout" else 0.65
        _met, sig, _split, _args, bundle = ae.evaluate_exact(
            run_dir=run_dir,
            signals_df=sdf,
            genome=genome,
            seed=int(seed) + ix + 1,
            name=f"d1_{route_id}_base",
            wf_splits=wf_splits,
            train_ratio=train_ratio,
        )
        pre = ae.build_preentry_features(bundle, genome)
        d = af.parse_entry_rows(ae.build_trade_labels(sig.merge(pre, on=["signal_id", "signal_time"], how="left")))
        d = d.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
        # Prior-only operational helper for cooldown-like policies.
        prev_t = pd.to_datetime(d["signal_time_utc"], utc=True, errors="coerce").shift(1)
        d["prior_signal_gap_min"] = (pd.to_datetime(d["signal_time_utc"], utc=True, errors="coerce") - prev_t).dt.total_seconds() / 60.0
        d["prior_signal_gap_min"] = to_num(d["prior_signal_gap_min"]).fillna(1e9)
        q = af.build_quantile_stats(d)
        d = build_s1_risk_features(d, qstats=q)
        out[route_id] = d
    return out


def phase_d1_attribution(route_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    tail_examples: List[Dict[str, Any]] = []
    route_summary: List[Dict[str, Any]] = []

    for route_id, d in route_data.items():
        x = d[(d["entry_for_labels"] == 1) & d["pnl_net_trade_notional_dec"].notna()].copy()
        if x.empty:
            continue
        pnl = to_num(x["pnl_net_trade_notional_dec"])
        tail_cut10 = float(np.nanquantile(pnl, 0.10))
        cvar5_cut = float(np.nanquantile(pnl, 0.05))
        x["is_tail10"] = (pnl <= tail_cut10).astype(int)
        x["is_cvar5"] = (pnl <= cvar5_cut).astype(int)
        try:
            x["risk_decile"] = pd.qcut(to_num(x["risk_score_s1"]), q=min(10, max(2, int(to_num(x["risk_score_s1"]).nunique()))), duplicates="drop").astype(str)
        except Exception:
            x["risk_decile"] = "all"

        total_tail_abs = float(np.nansum(np.abs(pnl[pnl <= tail_cut10])))
        total_cvar5_abs = float(np.nansum(np.abs(pnl[pnl <= cvar5_cut])))

        g = x.groupby("risk_decile", dropna=False).agg(
            support=("signal_id", "size"),
            tail10_count=("is_tail10", "sum"),
            cvar5_count=("is_cvar5", "sum"),
            mean_pnl=("pnl_net_trade_notional_dec", "mean"),
            tail10_abs_loss=("pnl_net_trade_notional_dec", lambda s: float(np.nansum(np.abs(to_num(s)[to_num(s) <= tail_cut10])))),
            cvar5_abs_loss=("pnl_net_trade_notional_dec", lambda s: float(np.nansum(np.abs(to_num(s)[to_num(s) <= cvar5_cut])))),
        ).reset_index()
        g["route_id"] = route_id
        g["tail10_rate"] = to_num(g["tail10_count"]) / np.maximum(1.0, to_num(g["support"]))
        g["cvar5_rate"] = to_num(g["cvar5_count"]) / np.maximum(1.0, to_num(g["support"]))
        g["tail10_loss_share"] = [safe_div(float(v), total_tail_abs) for v in to_num(g["tail10_abs_loss"]).to_list()]
        g["cvar5_loss_share"] = [safe_div(float(v), total_cvar5_abs) for v in to_num(g["cvar5_abs_loss"]).to_list()]
        rows.extend(g.to_dict(orient="records"))

        x_tail = x.sort_values("pnl_net_trade_notional_dec", ascending=True).head(10).copy()
        for _, r in x_tail.iterrows():
            tail_examples.append(
                {
                    "route_id": route_id,
                    "signal_id": str(r["signal_id"]),
                    "signal_time_utc": str(r["signal_time_utc"]),
                    "entry_time_utc": str(r.get("entry_time_utc", "")),
                    "split_id": float(r.get("split_id", np.nan)),
                    "pnl_net_trade_notional_dec": float(r["pnl_net_trade_notional_dec"]),
                    "risk_score_s1": float(r.get("risk_score_s1", np.nan)),
                    "risk_decile": str(r.get("risk_decile", "unknown")),
                    "prior_loss_streak_len": float(r.get("prior_loss_streak_len", np.nan)),
                    "prior_rolling_tail_count_20": float(r.get("prior_rolling_tail_count_20", np.nan)),
                    "exit_reason": str(r.get("exit_reason", "")),
                    "sl_hit_flag": int(r.get("sl_hit_flag", 0)),
                }
            )

        route_summary.append(
            {
                "route_id": route_id,
                "entries": int(len(x)),
                "tail_cut10": tail_cut10,
                "cvar5_cut": cvar5_cut,
                "mean_pnl": float(np.nanmean(pnl)),
                "cvar5": float(np.nanmean(np.sort(pnl.to_numpy(dtype=float))[: max(1, int(math.ceil(0.05 * len(pnl))))])),
            }
        )

    attr = pd.DataFrame(rows)
    tail_ex = pd.DataFrame(tail_examples)
    rs = pd.DataFrame(route_summary)
    if rs.empty:
        raise RuntimeError("No route entries available for D1 attribution")
    worst_route = str(rs.sort_values("cvar5", ascending=True).iloc[0]["route_id"])
    w = attr[attr["route_id"] == worst_route].copy()
    top_share = float(w.sort_values("cvar5_loss_share", ascending=False).head(2)["cvar5_loss_share"].sum()) if not w.empty else float("nan")
    summary = {
        "worst_route": worst_route,
        "worst_route_cvar5": float(rs.set_index("route_id").loc[worst_route, "cvar5"]),
        "worst_route_top2_decile_cvar5_loss_share": top_share,
        "route_rows": rs.to_dict(orient="records"),
    }
    return attr, summary, tail_ex


def phase_d2_tail_label(route_data: Dict[str, pd.DataFrame], x_pct: float = 0.10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    per_route: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    for route_id, d in route_data.items():
        x = d[(d["entry_for_labels"] == 1) & d["pnl_net_trade_notional_dec"].notna()].copy()
        if x.empty:
            continue
        cut = float(np.nanquantile(to_num(x["pnl_net_trade_notional_dec"]), float(x_pct)))
        x["y_tail_loss_route"] = (to_num(x["pnl_net_trade_notional_dec"]) <= cut).astype(int)
        for _, r in x.iterrows():
            labels.append(
                {
                    "route_id": route_id,
                    "signal_id": str(r["signal_id"]),
                    "signal_time_utc": str(r["signal_time_utc"]),
                    "split_id": float(r.get("split_id", np.nan)),
                    "risk_score_s1": float(r.get("risk_score_s1", np.nan)),
                    "y_tail_loss_route": int(r["y_tail_loss_route"]),
                    "pnl_net_trade_notional_dec": float(r["pnl_net_trade_notional_dec"]),
                    "prior_loss_streak_len": float(r.get("prior_loss_streak_len", np.nan)),
                    "prior_rolling_tail_count_20": float(r.get("prior_rolling_tail_count_20", np.nan)),
                    "pre3m_close_to_high_dist_bps": float(r.get("pre3m_close_to_high_dist_bps", np.nan)),
                    "pre3m_realized_vol_12": float(r.get("pre3m_realized_vol_12", np.nan)),
                    "pre3m_wick_ratio": float(r.get("pre3m_wick_ratio", np.nan)),
                    "prior_rolling_loss_rate_5": float(r.get("prior_rolling_loss_rate_5", np.nan)),
                }
            )

        sc = spearman_corr_no_scipy(to_num(x["risk_score_s1"]), to_num(x["y_tail_loss_route"]))
        split_pos = 0
        split_eligible = 0
        for sid, g in x.groupby("split_id", dropna=True):
            if len(g) < 20:
                continue
            ss = spearman_corr_no_scipy(to_num(g["risk_score_s1"]), to_num(g["y_tail_loss_route"]))
            if np.isfinite(ss):
                split_eligible += 1
                if ss > 0:
                    split_pos += 1
            split_rows.append(
                {
                    "route_id": route_id,
                    "split_id": int(sid) if np.isfinite(sid) else str(sid),
                    "support": int(len(g)),
                    "spearman_tail": float(ss),
                }
            )
        per_route.append(
            {
                "route_id": route_id,
                "support": int(len(x)),
                "tail_rate": float(np.mean(x["y_tail_loss_route"])),
                "overall_spearman_tail": float(sc),
                "stable_sign_frac": safe_div(float(split_pos), float(split_eligible)) if split_eligible > 0 else 0.0,
                "split_eligible": int(split_eligible),
                "split_positive": int(split_pos),
                "score_nan_rate": float(1.0 - np.mean(to_num(x["risk_score_s1"]).notna())),
            }
        )

    ldf = pd.DataFrame(labels)
    pr = pd.DataFrame(per_route)
    srows = pd.DataFrame(split_rows)
    if ldf.empty or pr.empty:
        raise RuntimeError("Tail label build produced empty outputs")

    # Stability comparable to Phase B: stable_sign_frac >= 0.60 and positive overall spearman.
    comb_sc = spearman_corr_no_scipy(to_num(ldf["risk_score_s1"]), to_num(ldf["y_tail_loss_route"]))
    combined_split_pos = int((to_num(srows["spearman_tail"]) > 0).sum()) if not srows.empty else 0
    combined_split_eligible = int(to_num(srows["spearman_tail"]).notna().sum()) if not srows.empty else 0
    combined_stable_sign_frac = safe_div(float(combined_split_pos), float(combined_split_eligible)) if combined_split_eligible > 0 else 0.0
    summary = {
        "label_tail_fraction": float(x_pct),
        "combined_support": int(len(ldf)),
        "combined_overall_spearman_tail": float(comb_sc),
        "combined_stable_sign_frac": float(combined_stable_sign_frac),
        "combined_split_eligible": int(combined_split_eligible),
        "combined_split_positive": int(combined_split_pos),
        "per_route": pr.to_dict(orient="records"),
        "split_rows": srows.to_dict(orient="records"),
        "acceptance_rule": {
            "stable_sign_frac_min": 0.60,
            "overall_spearman_positive": True,
            "score_nan_rate_max": 0.00,
        },
    }
    return ldf, summary


def _base_metrics_from_policy_eval_df(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "exec_expectancy_net": float(df["exec_expectancy_net"]),
        "exec_cvar_5": float(df["exec_cvar_5"]),
        "exec_max_drawdown": float(df["exec_max_drawdown"]),
        "entries_valid": int(df["entries_valid"]),
        "entry_rate": float(df["entry_rate"]),
        "taker_share": float(df["taker_share"]),
        "p95_fill_delay_min": float(df["p95_fill_delay_min"]),
    }


def eval_filter_metrics(d: pd.DataFrame, keep_mask: pd.Series) -> Dict[str, Any]:
    x = d.copy().reset_index(drop=True)
    valid_orig = (x["entry_for_labels"] == 1) & x["pnl_net_trade_notional_dec"].notna()
    keep = keep_mask.reindex(x.index).fillna(True).astype(bool)
    eval_valid = valid_orig & keep

    n = len(x)
    pnl = np.zeros(n, dtype=float)
    pnl[eval_valid.to_numpy(dtype=bool)] = to_num(x.loc[eval_valid, "pnl_net_trade_notional_dec"]).to_numpy(dtype=float)
    exp = float(np.mean(pnl)) if n else float("nan")
    k5 = max(1, int(math.ceil(0.05 * len(pnl)))) if len(pnl) else 1
    cvar = float(np.mean(np.sort(pnl)[:k5])) if len(pnl) else float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum) if cum.size else np.array([], dtype=float)
    dd = cum - peak if cum.size else np.array([], dtype=float)
    mdd = float(np.nanmin(dd)) if dd.size else float("nan")

    entries = int(eval_valid.sum())
    entry_rate = float(entries / max(1, n))
    taker = float(np.nanmean(to_num(x.loc[eval_valid, "taker_flag"]))) if entries > 0 else float("nan")
    delays = to_num(x.loc[eval_valid, "fill_delay_min"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(delays, 0.95)) if delays.size else float("nan")

    # Split min expectancy with signal-level averaging.
    z = x.copy()
    z["pnl_eval"] = 0.0
    z.loc[eval_valid, "pnl_eval"] = to_num(z.loc[eval_valid, "pnl_net_trade_notional_dec"])
    split_exp = z.groupby("split_id", dropna=True)["pnl_eval"].mean()
    min_split = float(split_exp.min()) if not split_exp.empty else float("nan")

    return {
        "exec_expectancy_net": float(exp),
        "exec_cvar_5": float(cvar),
        "exec_max_drawdown": float(mdd),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "taker_share": float(taker),
        "p95_fill_delay_min": float(p95),
        "min_split_expectancy_net": float(min_split),
    }


def build_filter_policies(max_policies: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(p: Dict[str, Any]) -> None:
        h = policy_hash(p)
        if h in seen:
            return
        seen.add(h)
        out.append(p)

    add({"policy_id": "flat_baseline", "type": "flat"})

    for t in [0.70, 0.75, 0.80, 0.85, 0.90]:
        add({"policy_id": f"skip_risk_ge_{t:.2f}", "type": "risk_skip", "risk_threshold": t})
    for k in [3, 4, 5, 6]:
        add({"policy_id": f"skip_streak_ge_{k}", "type": "streak_skip", "k_streak": k})
    for n in [6, 7, 8, 9]:
        add({"policy_id": f"skip_tail20_ge_{n}", "type": "tail_skip", "k_tail20": n})
    for k in [3, 4, 5]:
        for cool in [60, 120, 240]:
            add({"policy_id": f"skip_streak{k}_cool{cool}m", "type": "cooldown_skip", "k_streak": k, "cooldown_min": cool})
    for t in [0.75, 0.80, 0.85]:
        for k in [3, 4, 5]:
            add({"policy_id": f"skip_risk{t:.2f}_streak{k}", "type": "risk_streak_combo", "risk_threshold": t, "k_streak": k})

    return out[: max(1, int(max_policies))]


def apply_filter_policy(d: pd.DataFrame, pol: Dict[str, Any]) -> pd.Series:
    x = d.copy()
    typ = str(pol.get("type", "flat"))
    keep = pd.Series(np.ones(len(x), dtype=bool), index=x.index)
    if typ == "flat":
        return keep
    if typ == "risk_skip":
        thr = float(pol["risk_threshold"])
        keep = ~(to_num(x["risk_score_s1"]) >= thr)
    elif typ == "streak_skip":
        k = float(pol["k_streak"])
        keep = ~(to_num(x["prior_loss_streak_len"]) >= k)
    elif typ == "tail_skip":
        k = float(pol["k_tail20"])
        keep = ~(to_num(x["prior_rolling_tail_count_20"]) >= k)
    elif typ == "cooldown_skip":
        k = float(pol["k_streak"])
        cd = float(pol["cooldown_min"])
        keep = ~((to_num(x["prior_loss_streak_len"]) >= k) & (to_num(x["prior_signal_gap_min"]) <= cd))
    elif typ == "risk_streak_combo":
        thr = float(pol["risk_threshold"])
        k = float(pol["k_streak"])
        keep = ~((to_num(x["risk_score_s1"]) >= thr) & (to_num(x["prior_loss_streak_len"]) >= k))
    else:
        keep = pd.Series(np.ones(len(x), dtype=bool), index=x.index)
    keep = keep.fillna(True)
    # do not touch non-entry rows; they remain non-entry in metric eval anyway.
    return keep.astype(bool)


def phase_d3_filter_pilot(route_data: Dict[str, pd.DataFrame], max_policies: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    policies = build_filter_policies(max_policies=max_policies)
    rows: List[Dict[str, Any]] = []
    vectors: Dict[str, np.ndarray] = {}
    invalid_hist: Dict[str, int] = {}

    for route_id, d in route_data.items():
        flat = eval_filter_metrics(d, keep_mask=pd.Series(np.ones(len(d), dtype=bool), index=d.index))
        for pol in policies:
            keep = apply_filter_policy(d, pol)
            met = eval_filter_metrics(d, keep_mask=keep)
            row = {
                "route_id": route_id,
                "policy_id": str(pol["policy_id"]),
                "policy_type": str(pol.get("type", "flat")),
                "policy_hash": policy_hash(pol),
                "filter_kept_entries_pct": safe_div(float(met["entries_valid"]), float(flat["entries_valid"])) if flat["entries_valid"] > 0 else float("nan"),
                "delta_expectancy_vs_flat": float(met["exec_expectancy_net"] - flat["exec_expectancy_net"]),
                "cvar_improve_ratio_vs_flat": safe_div(abs(float(flat["exec_cvar_5"])) - abs(float(met["exec_cvar_5"])), abs(float(flat["exec_cvar_5"]))),
                "maxdd_improve_ratio_vs_flat": safe_div(abs(float(flat["exec_max_drawdown"])) - abs(float(met["exec_max_drawdown"])), abs(float(flat["exec_max_drawdown"]))),
                **met,
            }
            rows.append(row)

        # Save per-route vector for PSR/DSR proxies later.
        for pol in policies:
            if pol["policy_id"] == "flat_baseline":
                continue
            keep = apply_filter_policy(d, pol)
            valid_orig = (d["entry_for_labels"] == 1) & d["pnl_net_trade_notional_dec"].notna()
            valid = valid_orig & keep
            vec = np.zeros(len(d), dtype=float)
            vec[valid.to_numpy(dtype=bool)] = to_num(d.loc[valid, "pnl_net_trade_notional_dec"]).to_numpy(dtype=float)
            key = f"{pol['policy_id']}::{route_id}"
            vectors[key] = vec

    rdf = pd.DataFrame(rows)
    agg = rdf[rdf["policy_id"] != "flat_baseline"].groupby(["policy_id", "policy_type", "policy_hash"], dropna=False).agg(
        routes_tested=("route_id", "nunique"),
        min_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "min"),
        min_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "min"),
        min_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "min"),
        mean_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "mean"),
        mean_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "mean"),
        mean_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "mean"),
        min_entries_valid=("entries_valid", "min"),
        min_entry_rate=("entry_rate", "min"),
        min_filter_kept_entries_pct=("filter_kept_entries_pct", "min"),
    ).reset_index()
    agg["no_pathology"] = (
        to_num(agg["min_delta_expectancy_vs_flat"]).notna()
        & to_num(agg["min_cvar_improve_ratio_vs_flat"]).notna()
        & to_num(agg["min_maxdd_improve_ratio_vs_flat"]).notna()
        & (to_num(agg["min_entries_valid"]) > 0)
        & (to_num(agg["min_entry_rate"]) > 0)
    ).astype(int)
    agg["strict_pass"] = (
        (to_num(agg["min_delta_expectancy_vs_flat"]) > 0.0)
        & (to_num(agg["min_cvar_improve_ratio_vs_flat"]) >= 0.0)
        & (to_num(agg["min_maxdd_improve_ratio_vs_flat"]) > 0.0)
        & (to_num(agg["no_pathology"]) == 1)
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

    # invalid histogram from aggregate policy-level failures.
    for _, r in agg.iterrows():
        if int(r["strict_pass"]) == 1:
            continue
        if float(r["min_delta_expectancy_vs_flat"]) <= 0:
            invalid_hist["non_positive_expectancy_delta"] = int(invalid_hist.get("non_positive_expectancy_delta", 0) + 1)
        if float(r["min_cvar_improve_ratio_vs_flat"]) < 0:
            invalid_hist["negative_cvar_improvement"] = int(invalid_hist.get("negative_cvar_improvement", 0) + 1)
        if float(r["min_maxdd_improve_ratio_vs_flat"]) <= 0:
            invalid_hist["non_positive_maxdd_improvement"] = int(invalid_hist.get("non_positive_maxdd_improvement", 0) + 1)
        if int(r["no_pathology"]) != 1:
            invalid_hist["metric_pathology"] = int(invalid_hist.get("metric_pathology", 0) + 1)

    return rdf, agg, invalid_hist


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase D1-D3 tail attribution/label/filter branch (contract-locked, no GA marathon)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--max-filter-policies", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_root = exec_root / f"PHASED123_TAIL_FILTER_{utc_tag()}"
    run_root.mkdir(parents=True, exist_ok=False)
    overall: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "run_root": str(run_root),
        "seed": int(args.seed),
        "mainline_status": "CONTINUE",
        "phases": {},
    }

    # Shared lock checks.
    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    met_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, met_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")
    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(met_fp)
    if fee_sha != LOCKED["expected_fee_sha"] or met_sha != LOCKED["expected_metrics_sha"]:
        overall["mainline_status"] = "STOP_CONTRACT"
        write_text(run_root / "contract_fail.md", f"Contract hash mismatch. fee={fee_sha}, metrics={met_sha}")
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "D0", "classification": "CONTRACT_FAIL", "mainline_status": "STOP_CONTRACT", "run_root": str(run_root)}, sort_keys=True))
        return

    sig_in = ae.ensure_signals_schema(pd.read_csv(rep_fp))
    source_root, source_phasec, source_phasec_route = load_latest_phasec(exec_root)
    source_c_summary = source_phasec / "phaseC_results.csv"
    source_c_byroute = source_phasec / "phaseC_results_by_route.csv"
    csum = pd.read_csv(source_c_summary)
    cby = pd.read_csv(source_c_byroute)
    exec_pair = ae.load_exec_pair(exec_root)
    if exec_pair["E1"]["genome_hash"] != LOCKED["primary_exec_hash"] or exec_pair["E2"]["genome_hash"] != LOCKED["backup_exec_hash"]:
        overall["mainline_status"] = "STOP_CONTRACT"
        write_text(run_root / "contract_fail.md", "Execution hash mismatch vs locked contract.")
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "D0", "classification": "CONTRACT_FAIL", "mainline_status": "STOP_CONTRACT", "run_root": str(run_root)}, sort_keys=True))
        return

    # --------------------
    # D1
    # --------------------
    d1_dir = run_root / f"phaseD1_{utc_tag()}"
    d1_dir.mkdir(parents=True, exist_ok=False)
    d1_start = time.time()
    try:
        lock_args = ae.build_args(signals_csv=rep_fp, seed=int(args.seed))
        lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=d1_dir)
        if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
            raise RuntimeError("freeze lock validation failed in D1")

        route_data = evaluate_baseline_routes(
            run_dir=d1_dir,
            sig_in=sig_in,
            genome=copy.deepcopy(exec_pair["E1"]["genome"]),
            seed=int(args.seed),
        )
        attr_df, d1_summary, tail_examples = phase_d1_attribution(route_data)
        attr_df.to_csv(d1_dir / "tail_attribution_by_route.csv", index=False)
        tail_examples.to_csv(d1_dir / "tail_examples_worst_route.csv", index=False)

        d1_pass = int(
            not attr_df.empty
            and not tail_examples.empty
            and isinstance(d1_summary.get("worst_route"), str)
        )
        d1_class = "PASS" if d1_pass == 1 else "INFRA_FAIL"
        d1_reason = "clear worst-route tail driver identified" if d1_pass == 1 else "insufficient attribution evidence"

        write_text(
            d1_dir / "phaseD1_report.md",
            "\n".join(
                [
                    "# Phase D1 Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: **{d1_class}**",
                    f"- Reason: {d1_reason}",
                    f"- Source Phase C root: `{source_root}`",
                    f"- Source phaseC dir: `{source_phasec}`",
                    f"- Worst route by CVaR: `{d1_summary.get('worst_route', 'n/a')}`",
                    f"- Worst route CVaR5: `{float(d1_summary.get('worst_route_cvar5', float('nan'))):.8f}`",
                    f"- Top-2 risk-decile CVaR loss share (worst route): `{float(d1_summary.get('worst_route_top2_decile_cvar5_loss_share', float('nan'))):.6f}`",
                    "",
                    "## Route-level summary from D1",
                    "",
                    markdown_table(pd.DataFrame(d1_summary.get("route_rows", [])), ["route_id", "entries", "mean_pnl", "cvar5", "tail_cut10", "cvar5_cut"]),
                    "",
                    "## Tail attribution by route/decile (head)",
                    "",
                    markdown_table(attr_df.head(20), ["route_id", "risk_decile", "support", "tail10_count", "tail10_rate", "cvar5_count", "cvar5_rate", "tail10_loss_share", "cvar5_loss_share"]),
                ]
            ),
        )
        json_dump(
            d1_dir / "phaseD1_run_manifest.json",
            {
                "generated_utc": utc_now(),
                "phase": "D1",
                "duration_sec": float(time.time() - d1_start),
                "decision": d1_class,
                "reason": d1_reason,
                "phase_dir": str(d1_dir),
                "source_phasec_dir": str(source_phasec),
                "freeze_lock_validation": lock_validation,
            },
        )
        overall["phases"]["D1"] = {"classification": d1_class, "phase_dir": str(d1_dir)}
        if d1_class != "PASS":
            overall["mainline_status"] = "STOP_INFRA"
            json_dump(run_root / "run_manifest.json", overall)
            print(json.dumps({"furthest_phase": "D1", "classification": d1_class, "mainline_status": overall["mainline_status"], "run_root": str(run_root)}, sort_keys=True))
            return
    except Exception as e:
        overall["phases"]["D1"] = {"classification": "INFRA_FAIL", "phase_dir": str(d1_dir), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"
        write_text(d1_dir / "phaseD1_report.md", f"# Phase D1 Report\n\n- Generated UTC: {utc_now()}\n- Decision: **INFRA_FAIL**\n- Reason: {e}")
        json_dump(d1_dir / "phaseD1_run_manifest.json", {"generated_utc": utc_now(), "phase": "D1", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(d1_dir)})
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "D1", "classification": "INFRA_FAIL", "mainline_status": "STOP_INFRA", "run_root": str(run_root)}, sort_keys=True))
        return

    # --------------------
    # D2
    # --------------------
    d2_dir = run_root / f"phaseD2_{utc_tag()}"
    d2_dir.mkdir(parents=True, exist_ok=False)
    d2_start = time.time()
    try:
        tail_labels, stab = phase_d2_tail_label(route_data, x_pct=0.10)
        tail_labels.to_csv(d2_dir / "tail_label.csv", index=False)
        json_dump(d2_dir / "split_stability_tail.json", stab)

        # D2 PASS/FAIL rule
        d2_pass = (
            float(stab["combined_stable_sign_frac"]) >= 0.60
            and float(stab["combined_overall_spearman_tail"]) > 0.0
            and all(float(r.get("score_nan_rate", 1.0)) <= 0.0 for r in stab.get("per_route", []))
        )
        d2_class = "PASS" if d2_pass else "NO_GO"
        d2_reason = "tail label stable and non-leaky" if d2_pass else "tail label stability below threshold or pathology"

        write_text(
            d2_dir / "phaseD2_report.md",
            "\n".join(
                [
                    "# Phase D2 Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: **{d2_class}**",
                    f"- Reason: {d2_reason}",
                    f"- Tail label fraction (worst-X%): `{float(stab.get('label_tail_fraction', float('nan'))):.2f}`",
                    f"- Combined support: `{int(stab.get('combined_support', 0))}`",
                    f"- Combined Spearman(risk_score, y_tail): `{float(stab.get('combined_overall_spearman_tail', float('nan'))):.6f}`",
                    f"- Combined stable_sign_frac: `{float(stab.get('combined_stable_sign_frac', float('nan'))):.4f}`",
                    "",
                    "## Per-route tail stability",
                    "",
                    markdown_table(pd.DataFrame(stab.get("per_route", [])), ["route_id", "support", "tail_rate", "overall_spearman_tail", "stable_sign_frac", "split_eligible", "split_positive", "score_nan_rate"]),
                    "",
                    "## Leakage check",
                    "",
                    "- y_tail_loss_route defined from realized route outcomes only (label), not used as pre-entry feature.",
                    "- Risk score uses pre-entry and prior-only features from AE family.",
                ]
            ),
        )
        json_dump(
            d2_dir / "phaseD2_run_manifest.json",
            {
                "generated_utc": utc_now(),
                "phase": "D2",
                "duration_sec": float(time.time() - d2_start),
                "decision": d2_class,
                "reason": d2_reason,
                "phase_dir": str(d2_dir),
            },
        )
        overall["phases"]["D2"] = {"classification": d2_class, "phase_dir": str(d2_dir)}
        if d2_class != "PASS":
            overall["mainline_status"] = "STOP_NO_GO"
            ng = d2_dir / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(
                ng / "phaseD2_no_go_diagnosis.md",
                "\n".join(
                    [
                        "# Phase D2 NO_GO Diagnosis",
                        "",
                        f"- Generated UTC: {utc_now()}",
                        f"- Reason: {d2_reason}",
                        "- Mainline stopped before D3.",
                    ]
                ),
            )
            write_text(
                ng / "next_step_prompt.txt",
                "Phase D2 NO_GO fallback: improve tail-label stability (split consistency and support quality), then rerun small filter pilot under unchanged hard gates.",
            )
            json_dump(run_root / "run_manifest.json", overall)
            print(json.dumps({"furthest_phase": "D2", "classification": d2_class, "mainline_status": overall["mainline_status"], "run_root": str(run_root)}, sort_keys=True))
            return
    except Exception as e:
        overall["phases"]["D2"] = {"classification": "INFRA_FAIL", "phase_dir": str(d2_dir), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"
        write_text(d2_dir / "phaseD2_report.md", f"# Phase D2 Report\n\n- Generated UTC: {utc_now()}\n- Decision: **INFRA_FAIL**\n- Reason: {e}")
        json_dump(d2_dir / "phaseD2_run_manifest.json", {"generated_utc": utc_now(), "phase": "D2", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(d2_dir)})
        json_dump(run_root / "run_manifest.json", overall)
        print(json.dumps({"furthest_phase": "D2", "classification": "INFRA_FAIL", "mainline_status": "STOP_INFRA", "run_root": str(run_root)}, sort_keys=True))
        return

    # --------------------
    # D3
    # --------------------
    d3_dir = run_root / f"phaseD3_{utc_tag()}"
    d3_dir.mkdir(parents=True, exist_ok=False)
    d3_start = time.time()
    try:
        route_df, agg_df, invalid_hist = phase_d3_filter_pilot(route_data, max_policies=int(args.max_filter_policies))
        route_df.to_csv(d3_dir / "phaseD3_results_by_route.csv", index=False)
        agg_df.to_csv(d3_dir / "phaseD3_results.csv", index=False)
        json_dump(d3_dir / "invalid_reason_histogram.json", invalid_hist)

        passers = int((agg_df["strict_pass"] == 1).sum()) if not agg_df.empty else 0
        d3_class = "PASS" if passers >= 1 else "NO_GO"
        d3_reason = ">=1 strict passer across required routes" if passers >= 1 else "0 strict passers across required routes"

        # PSR/DSR proxy for top policy subset.
        top = agg_df.head(10).copy() if not agg_df.empty else pd.DataFrame()
        psr_rows: List[Dict[str, Any]] = []
        if not top.empty:
            # crude proxy with effective-trials from policy correlation on min-delta vector ranks.
            eff_unc = float(len(top))
            eff_corr = float(max(1.0, len(top) / 2.0))
            for _, r in top.iterrows():
                # proxy by treating per-route deltas as tiny vector.
                rr = route_df[route_df["policy_id"] == str(r["policy_id"])].copy()
                vec = to_num(rr["delta_expectancy_vs_flat"]).fillna(0.0).to_numpy(dtype=float)
                psr, dsr = af.psr_proxy_from_pnl(vec, eff_trials=eff_corr)
                psr_rows.append(
                    {
                        "policy_id": str(r["policy_id"]),
                        "psr_proxy": float(psr),
                        "dsr_proxy": float(dsr),
                        "effective_trials_uncorrelated": eff_unc,
                        "effective_trials_corr_adjusted": eff_corr,
                    }
                )
        psr_df = pd.DataFrame(psr_rows)
        if not psr_df.empty:
            psr_df.to_csv(d3_dir / "phaseD3_psr_dsr_proxy.csv", index=False)

        write_text(
            d3_dir / "phaseD3_report.md",
            "\n".join(
                [
                    "# Phase D3 Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: **{d3_class}**",
                    f"- Reason: {d3_reason}",
                    f"- Policies evaluated (excluding flat): `{len(agg_df)}`",
                    f"- Strict passers: `{passers}`",
                    "",
                    "## Strict gate (unchanged from Phase C style)",
                    "",
                    "- min_delta_expectancy_vs_flat > 0",
                    "- min_cvar_improve_ratio_vs_flat >= 0",
                    "- min_maxdd_improve_ratio_vs_flat > 0",
                    "- no metric pathology",
                    "",
                    "## Top filter policies",
                    "",
                    markdown_table(
                        agg_df.head(20),
                        [
                            "policy_id",
                            "policy_type",
                            "strict_pass",
                            "min_delta_expectancy_vs_flat",
                            "min_cvar_improve_ratio_vs_flat",
                            "min_maxdd_improve_ratio_vs_flat",
                            "min_entries_valid",
                            "min_entry_rate",
                            "min_filter_kept_entries_pct",
                            "rank_score",
                        ],
                    ),
                    "",
                    "## Reality-check",
                    "",
                    "- Reality-check bootstrap: TODO placeholder (not implemented in this branch).",
                ]
            ),
        )
        json_dump(
            d3_dir / "phaseD3_run_manifest.json",
            {
                "generated_utc": utc_now(),
                "phase": "D3",
                "duration_sec": float(time.time() - d3_start),
                "decision": d3_class,
                "reason": d3_reason,
                "phase_dir": str(d3_dir),
                "strict_passers": int(passers),
                "policy_budget": int(args.max_filter_policies),
            },
        )
        write_text(
            d3_dir / "phaseD3_decision.md",
            "\n".join(
                [
                    "# Phase D3 Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: **{d3_class}**",
                    f"- Reason: {d3_reason}",
                ]
            ),
        )
        overall["phases"]["D3"] = {"classification": d3_class, "phase_dir": str(d3_dir)}
        if d3_class != "PASS":
            overall["mainline_status"] = "STOP_NO_GO"
            ng = d3_dir / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(
                ng / "phaseD3_no_go_reasoning.md",
                "\n".join(
                    [
                        "# Phase D3 NO_GO Reasoning",
                        "",
                        f"- Generated UTC: {utc_now()}",
                        f"- Strict passers: {passers}",
                        "- All tested filter/cooldown policies failed strict multi-route gates.",
                    ]
                ),
            )
            write_text(
                ng / "next_step_prompt.txt",
                "Phase D3 NO_GO fallback: stop filter expansion. Rework tail labels and route-specific objective weighting before any further policy search; keep hard gates and contract lock unchanged.",
            )
        else:
            overall["mainline_status"] = "CONTINUE"
            write_text(
                d3_dir / "prompt_next.txt",
                "Phase D4 follow-up: run a bounded robustness confirmation on strict-pass filter policies (route perturbation + stress) before any promotion decision.",
            )
    except Exception as e:
        overall["phases"]["D3"] = {"classification": "INFRA_FAIL", "phase_dir": str(d3_dir), "reason": str(e)}
        overall["mainline_status"] = "STOP_INFRA"
        write_text(d3_dir / "phaseD3_report.md", f"# Phase D3 Report\n\n- Generated UTC: {utc_now()}\n- Decision: **INFRA_FAIL**\n- Reason: {e}")
        json_dump(d3_dir / "phaseD3_run_manifest.json", {"generated_utc": utc_now(), "phase": "D3", "decision": "INFRA_FAIL", "reason": str(e), "phase_dir": str(d3_dir)})

    overall["duration_sec"] = float(time.time() - t0)
    json_dump(run_root / "run_manifest.json", overall)
    furthest = "D3" if "D3" in overall["phases"] else ("D2" if "D2" in overall["phases"] else "D1")
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
