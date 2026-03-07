#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
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


def markdown_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    headers = list(x.columns)
    lines = []
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


def parse_entry_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time_utc"] = pd.to_datetime(x.get("signal_time_utc", x.get("signal_time", pd.NaT)), utc=True, errors="coerce")
    x["entry_time_utc"] = pd.to_datetime(x.get("entry_time_utc", x.get("exec_entry_time", pd.NaT)), utc=True, errors="coerce")
    x["entry_taken"] = to_num(x.get("entry_taken", x.get("exec_filled", 0))).fillna(0).astype(int)
    x["entry_for_labels"] = to_num(x.get("entry_for_labels", 0)).fillna(0).astype(int)
    x["split_id"] = to_num(x.get("split_id", np.nan))
    x["pnl_net_trade_notional_dec"] = to_num(x.get("pnl_net_trade_notional_dec", x.get("exec_pnl_net_pct", np.nan)))
    x["pnl_gross_trade_notional_dec"] = to_num(x.get("pnl_gross_trade_notional_dec", x.get("exec_pnl_gross_pct", np.nan)))
    x["fee_drag_trade"] = to_num(x.get("fee_drag_trade", x["pnl_gross_trade_notional_dec"] - x["pnl_net_trade_notional_dec"]))
    x["fill_delay_min"] = to_num(x.get("fill_delay_min", x.get("exec_fill_delay_min", np.nan)))
    x["taker_flag"] = to_num(x.get("taker_flag", (x.get("exec_fill_liquidity_type", "").astype(str).str.lower() == "taker").astype(int))).fillna(0).astype(int)
    return x.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)


def build_quantile_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    cols = [
        "pre3m_close_to_high_dist_bps",
        "pre3m_spread_proxy_bps",
        "pre3m_realized_vol_12",
        "pre3m_wick_ratio",
        "pre3m_atr_z",
        "trend_up_1h",
        "prior_loss_streak_len",
        "prior_rolling_tail_count_20",
        "prior_rolling_loss_rate_5",
    ]
    e = df[df["entry_for_labels"] == 1].copy()
    for c in cols:
        v = to_num(e[c]).dropna()
        if len(v) == 0:
            out[c] = {"q10": float("nan"), "q20": float("nan"), "q50": float("nan"), "q80": float("nan"), "q90": float("nan")}
            continue
        out[c] = {
            "q10": float(np.nanquantile(v, 0.10)),
            "q20": float(np.nanquantile(v, 0.20)),
            "q50": float(np.nanquantile(v, 0.50)),
            "q80": float(np.nanquantile(v, 0.80)),
            "q90": float(np.nanquantile(v, 0.90)),
        }
    return out


def normalize_feature(x: pd.Series, q10: float, q90: float) -> pd.Series:
    v = to_num(x)
    if not np.isfinite(q10) or not np.isfinite(q90) or abs(q90 - q10) <= 1e-12:
        return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
    out = (v - q10) / (q90 - q10)
    return out.clip(lower=0.0, upper=1.0).fillna(0.0)


def compute_policy_features(df: pd.DataFrame, qstats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    x = df.copy()
    x["f_streak"] = normalize_feature(x["prior_loss_streak_len"], qstats["prior_loss_streak_len"]["q10"], qstats["prior_loss_streak_len"]["q90"])
    x["f_tail20"] = normalize_feature(x["prior_rolling_tail_count_20"], qstats["prior_rolling_tail_count_20"]["q10"], qstats["prior_rolling_tail_count_20"]["q90"])
    x["f_closehigh"] = normalize_feature(x["pre3m_close_to_high_dist_bps"], qstats["pre3m_close_to_high_dist_bps"]["q10"], qstats["pre3m_close_to_high_dist_bps"]["q90"])
    x["f_vol"] = normalize_feature(x["pre3m_realized_vol_12"], qstats["pre3m_realized_vol_12"]["q10"], qstats["pre3m_realized_vol_12"]["q90"])
    x["f_wick"] = normalize_feature(x["pre3m_wick_ratio"], qstats["pre3m_wick_ratio"]["q10"], qstats["pre3m_wick_ratio"]["q90"])
    x["f_lossrate5"] = to_num(x["prior_rolling_loss_rate_5"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    x["f_int1"] = x["f_lossrate5"] * x["f_wick"]
    x["f_int2"] = x["f_closehigh"] * x["f_vol"]
    return x


def policy_hash(policy: Dict[str, Any]) -> str:
    txt = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def random_policy(rng: random.Random, pid: str) -> Dict[str, Any]:
    family = rng.choices(["step", "smooth", "streak", "hybrid"], weights=[0.30, 0.30, 0.20, 0.20], k=1)[0]
    weights = {
        "w_streak": rng.uniform(0.4, 1.8),
        "w_tail20": rng.uniform(0.4, 1.8),
        "w_closehigh": rng.uniform(0.3, 1.5),
        "w_vol": rng.uniform(0.3, 1.5),
        "w_wick": rng.uniform(0.2, 1.0),
        "w_int1": rng.uniform(0.0, 1.2),
        "w_int2": rng.uniform(0.0, 1.2),
    }
    if family == "step":
        t1 = rng.uniform(0.20, 0.45)
        t2 = rng.uniform(t1 + 0.10, 0.85)
        s_hi = rng.uniform(1.00, 1.30)
        s_mid = rng.uniform(0.80, min(1.05, s_hi))
        s_lo = rng.uniform(0.50, min(0.90, s_mid))
        params = {"t1": t1, "t2": t2, "s_hi": s_hi, "s_mid": s_mid, "s_lo": s_lo}
    elif family == "smooth":
        params = {"slope": rng.uniform(0.40, 1.20)}
    elif family == "streak":
        params = {
            "k_streak": rng.randint(2, 8),
            "k_tail20": rng.randint(4, 12),
            "s_up": rng.uniform(1.00, 1.25),
            "s_down": rng.uniform(0.50, 0.90),
        }
    else:
        params = {
            "slope": rng.uniform(0.30, 1.00),
            "k_streak": rng.randint(2, 8),
            "k_tail20": rng.randint(4, 12),
            "penalty_mult": rng.uniform(0.60, 0.95),
            "recovery_bonus": rng.uniform(0.00, 0.10),
        }
    return {
        "policy_id": pid,
        "family": family,
        "weights": weights,
        "params": params,
        "size_bounds": [0.50, 1.50],
    }


def s1_policy(qstats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    # Phase AE anchor: risk_score_half_size.
    return {
        "policy_id": "S1_anchor",
        "family": "ae_s1_anchor",
        "weights": {},
        "params": {
            "atr_z_q80": qstats["pre3m_atr_z"]["q80"],
            "vol_q80": qstats["pre3m_realized_vol_12"]["q80"],
            "spread_q80": qstats["pre3m_spread_proxy_bps"]["q80"],
            "trend_q20": qstats["trend_up_1h"]["q20"],
            "threshold": 2,
            "size_triggered": 0.50,
            "size_else": 1.00,
        },
        "size_bounds": [0.50, 1.50],
    }


def streak_control_policy() -> Dict[str, Any]:
    return {
        "policy_id": "streak_only_control",
        "family": "streak_control",
        "weights": {},
        "params": {"k_streak": 3, "size_down": 0.50, "size_up": 1.00},
        "size_bounds": [0.50, 1.50],
    }


def compute_risk_score(df: pd.DataFrame, policy: Dict[str, Any]) -> pd.Series:
    w = policy.get("weights", {})
    feats = {
        "w_streak": df["f_streak"],
        "w_tail20": df["f_tail20"],
        "w_closehigh": df["f_closehigh"],
        "w_vol": df["f_vol"],
        "w_wick": df["f_wick"],
        "w_int1": df["f_int1"],
        "w_int2": df["f_int2"],
    }
    num = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    den = 0.0
    for k, v in feats.items():
        wk = float(w.get(k, 0.0))
        if wk <= 0:
            continue
        num = num + wk * to_num(v).fillna(0.0)
        den += wk
    if den <= 1e-12:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return (num / den).clip(lower=0.0, upper=1.0)


def apply_policy_sizes(
    entries_df: pd.DataFrame,
    policy: Dict[str, Any],
    qstats: Dict[str, Dict[str, float]],
    noise_sigma: float = 0.0,
    rng: Optional[random.Random] = None,
    bounds_override: Optional[Tuple[float, float]] = None,
) -> pd.Series:
    x = entries_df.copy()
    fam = str(policy["family"])
    lb, ub = (float(policy["size_bounds"][0]), float(policy["size_bounds"][1]))
    if bounds_override is not None:
        lb, ub = float(bounds_override[0]), float(bounds_override[1])

    if fam == "ae_s1_anchor":
        p = policy["params"]
        f1 = (to_num(x["pre3m_atr_z"]) >= float(p["atr_z_q80"])).astype(int)
        f2 = (to_num(x["pre3m_realized_vol_12"]) >= float(p["vol_q80"])).astype(int)
        f3 = (to_num(x["pre3m_spread_proxy_bps"]) >= float(p["spread_q80"])).astype(int)
        f4 = (to_num(x["trend_up_1h"]) <= float(p["trend_q20"])).astype(int)
        score = f1 + f2 + f3 + f4
        s = pd.Series(
            np.where(score >= int(p["threshold"]), float(p["size_triggered"]), float(p["size_else"])),
            index=x.index,
            dtype=float,
        )
    elif fam == "streak_control":
        p = policy["params"]
        trig = (to_num(x["prior_loss_streak_len"]) >= float(p["k_streak"])).astype(int)
        s = pd.Series(np.where(trig == 1, float(p["size_down"]), float(p["size_up"])), index=x.index, dtype=float)
    else:
        score = compute_risk_score(x, policy)
        if noise_sigma > 0 and rng is not None:
            noise = np.asarray([rng.gauss(0.0, noise_sigma) for _ in range(len(score))], dtype=float)
            score = (score + noise).clip(lower=0.0, upper=1.0)
        p = policy["params"]
        if fam == "step":
            s = pd.Series(np.where(score <= float(p["t1"]), float(p["s_hi"]), np.where(score <= float(p["t2"]), float(p["s_mid"]), float(p["s_lo"]))), index=x.index, dtype=float)
        elif fam == "smooth":
            slope = float(p["slope"])
            s = pd.Series(1.0 + slope * (0.5 - score), index=x.index, dtype=float)
        elif fam == "streak":
            trig = (
                (to_num(x["prior_loss_streak_len"]) >= float(p["k_streak"]))
                | (to_num(x["prior_rolling_tail_count_20"]) >= float(p["k_tail20"]))
            ).astype(int)
            s = pd.Series(np.where(trig == 1, float(p["s_down"]), float(p["s_up"])), index=x.index, dtype=float)
        else:  # hybrid
            slope = float(p["slope"])
            base = 1.0 + slope * (0.5 - score)
            trig = (
                (to_num(x["prior_loss_streak_len"]) >= float(p["k_streak"]))
                | (to_num(x["prior_rolling_tail_count_20"]) >= float(p["k_tail20"]))
            ).astype(int)
            cool = (
                (to_num(x["prior_loss_streak_len"]) <= 1.0)
                & (to_num(x["prior_rolling_loss_rate_5"]).fillna(0.0) <= 0.35)
            ).astype(int)
            s = pd.Series(base, index=x.index, dtype=float)
            s = s * np.where(trig == 1, float(p["penalty_mult"]), 1.0)
            s = s + np.where(cool == 1, float(p["recovery_bonus"]), 0.0)

    s = s.clip(lower=lb, upper=ub)
    m = float(np.nanmean(s)) if len(s) else 1.0
    if np.isfinite(m) and m > 1e-12:
        s = s / m
    s = s.clip(lower=lb, upper=ub)
    m2 = float(np.nanmean(s)) if len(s) else 1.0
    if np.isfinite(m2) and m2 > 1e-12:
        s = s / m2
    s = s.clip(lower=lb, upper=ub)
    return s.astype(float)


def evaluate_sizing_policy_on_dataset(
    df_full: pd.DataFrame,
    policy: Dict[str, Any],
    qstats: Dict[str, Dict[str, float]],
    baseline_unweighted: Dict[str, Any],
    noise_sigma: float = 0.0,
    bounds_override: Optional[Tuple[float, float]] = None,
    rng_seed: int = 0,
    override_policy: Optional[Dict[str, Any]] = None,
    fee_mult_override: Optional[float] = None,
) -> Tuple[Dict[str, Any], pd.Series]:
    x = parse_entry_rows(df_full)
    valid = (x["entry_for_labels"] == 1) & x["pnl_net_trade_notional_dec"].notna()
    e = x.loc[valid].copy()
    e = compute_policy_features(e, qstats)
    pol = copy.deepcopy(override_policy if override_policy is not None else policy)
    rng = random.Random(int(rng_seed))
    sizes = apply_policy_sizes(e, pol, qstats=qstats, noise_sigma=float(noise_sigma), rng=rng, bounds_override=bounds_override)

    size_all = pd.Series(np.ones(len(x), dtype=float), index=x.index)
    size_all.loc[e.index] = sizes
    x["size_mult"] = size_all
    x["size_mult_entry"] = np.where(valid, x["size_mult"], np.nan)

    net = x["pnl_net_trade_notional_dec"].copy()
    gross = x["pnl_gross_trade_notional_dec"].copy()
    if fee_mult_override is not None:
        fee_drag = to_num(x["fee_drag_trade"]).fillna(0.0)
        gross_v = to_num(gross).fillna(np.nan)
        net = gross_v - float(fee_mult_override) * fee_drag

    x["pnl_weighted"] = np.where(valid, to_num(net) * x["size_mult"], np.nan)
    x["pnl_weighted_gross"] = np.where(valid, to_num(gross) * x["size_mult"], np.nan)

    n = len(x)
    pnl_sig = np.zeros(n, dtype=float)
    pnl_sig[valid.to_numpy(dtype=bool)] = x.loc[valid, "pnl_weighted"].to_numpy(dtype=float)
    exp = float(np.mean(pnl_sig)) if n else float("nan")
    k5 = max(1, int(math.ceil(0.05 * len(pnl_sig)))) if len(pnl_sig) else 1
    cvar = float(np.mean(np.sort(pnl_sig)[:k5])) if len(pnl_sig) else float("nan")
    cum = np.cumsum(np.nan_to_num(pnl_sig, nan=0.0))
    peak = np.maximum.accumulate(cum) if cum.size else np.array([], dtype=float)
    dd = cum - peak if cum.size else np.array([], dtype=float)
    mdd = float(np.nanmin(dd)) if dd.size else float("nan")
    pnl_sum = float(np.nansum(np.where(valid, x["pnl_weighted"], 0.0)))
    pnl_std = float(np.nanstd(pnl_sig, ddof=0)) if len(pnl_sig) else float("nan")

    splits = x["split_id"]
    split_vals = []
    for sid, g in x.groupby(splits, dropna=True):
        vec = np.zeros(len(g), dtype=float)
        gv = (g["entry_for_labels"] == 1) & g["pnl_weighted"].notna()
        vec[gv.to_numpy(dtype=bool)] = g.loc[gv, "pnl_weighted"].to_numpy(dtype=float)
        split_vals.append(float(np.mean(vec)))
    split_arr = np.asarray(split_vals, dtype=float) if split_vals else np.array([], dtype=float)
    min_split = float(np.nanmin(split_arr)) if split_arr.size else float("nan")
    med_split = float(np.nanmedian(split_arr)) if split_arr.size else float("nan")
    std_split = float(np.nanstd(split_arr, ddof=0)) if split_arr.size else float("nan")

    entries = int(valid.sum())
    entry_rate = float(entries / max(1, n))
    taker_share = float(np.nanmean(x.loc[valid, "taker_flag"])) if entries > 0 else float("nan")
    d = to_num(x.loc[valid, "fill_delay_min"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(d, 0.95)) if d.size else float("nan")

    # Unweighted clustering counts (invariant) and weighted burdens.
    t = x.loc[valid].copy().sort_values(["entry_time_utc", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    pnl_u = to_num(t["pnl_net_trade_notional_dec"]).to_numpy(dtype=float)
    pnl_w = to_num(t["pnl_weighted"]).to_numpy(dtype=float)
    loss_u = pnl_u < 0
    streaks: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(loss_u):
        if loss_u[i]:
            s = i
            while i + 1 < len(loss_u) and loss_u[i + 1]:
                i += 1
            streaks.append((s, i, i - s + 1))
        i += 1
    arr_len = np.asarray([z[2] for z in streaks], dtype=int) if streaks else np.array([], dtype=int)
    max_streak = int(arr_len.max()) if arr_len.size else 0
    ge5 = int(np.sum(arr_len >= 5)) if arr_len.size else 0
    ge10 = int(np.sum(arr_len >= 10)) if arr_len.size else 0

    total_weighted_loss_abs = float(np.nansum(np.abs(pnl_w[pnl_w < 0])))
    run_loss_abs = 0.0
    for s, eidx, ln in streaks:
        if ln >= 3:
            seg = pnl_w[s : eidx + 1]
            run_loss_abs += float(np.nansum(np.abs(seg[seg < 0])))
    weighted_loss_run_burden = safe_div(run_loss_abs, total_weighted_loss_abs)

    if len(pnl_w):
        k_dec = max(1, int(math.ceil(0.10 * len(pnl_w))))
        idx = np.argsort(pnl_w)[:k_dec]
        worst = pnl_w[idx]
        weighted_tail_loss_share = safe_div(float(np.nansum(np.abs(worst[worst < 0]))), total_weighted_loss_abs)
    else:
        weighted_tail_loss_share = float("nan")

    s_valid = to_num(x.loc[valid, "size_mult_entry"]).dropna().to_numpy(dtype=float)
    size_mean = float(np.nanmean(s_valid)) if s_valid.size else float("nan")
    size_std = float(np.nanstd(s_valid, ddof=0)) if s_valid.size else float("nan")
    size_p10 = float(np.nanquantile(s_valid, 0.10)) if s_valid.size else float("nan")
    size_p50 = float(np.nanquantile(s_valid, 0.50)) if s_valid.size else float("nan")
    size_p90 = float(np.nanquantile(s_valid, 0.90)) if s_valid.size else float("nan")
    budget_norm_error = float(size_mean - 1.0) if np.isfinite(size_mean) else float("nan")

    if s_valid.size:
        top_n = max(1, int(math.ceil(0.10 * s_valid.size)))
        top_share = safe_div(float(np.sum(np.sort(s_valid)[-top_n:])), float(np.sum(s_valid)))
    else:
        top_share = float("nan")

    inv_entries = int(entries == int(baseline_unweighted["entries_valid"]))
    inv_entry_rate = int(abs(entry_rate - float(baseline_unweighted["entry_rate"])) <= 1e-12)
    inv_taker = int(abs(taker_share - float(baseline_unweighted["taker_share"])) <= 1e-12 if np.isfinite(taker_share) and np.isfinite(float(baseline_unweighted["taker_share"])) else 0)
    inv_delay = int(abs(p95 - float(baseline_unweighted["p95_fill_delay_min"])) <= 1e-12 if np.isfinite(p95) and np.isfinite(float(baseline_unweighted["p95_fill_delay_min"])) else 0)

    out = {
        "exec_expectancy_net_weighted": float(exp),
        "cvar_5_weighted": float(cvar),
        "max_drawdown_weighted": float(mdd),
        "total_pnl_net_sum_weighted": float(pnl_sum),
        "pnl_std_weighted": float(pnl_std),
        "min_split_expectancy_net_weighted": float(min_split),
        "median_split_expectancy_net_weighted": float(med_split),
        "std_split_expectancy_net_weighted": float(std_split),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "taker_share": float(taker_share),
        "p95_fill_delay_min": float(p95),
        "max_consecutive_losses": int(max_streak),
        "streak_ge5_count": int(ge5),
        "streak_ge10_count": int(ge10),
        "weighted_loss_run_burden": float(weighted_loss_run_burden),
        "weighted_tail_loss_share": float(weighted_tail_loss_share),
        "size_mean": float(size_mean),
        "size_std": float(size_std),
        "size_p10": float(size_p10),
        "size_p50": float(size_p50),
        "size_p90": float(size_p90),
        "budget_norm_error": float(budget_norm_error),
        "effective_exposure_concentration_top_decile_share": float(top_share),
        "invariance_entries_match": int(inv_entries),
        "invariance_entry_rate_match": int(inv_entry_rate),
        "invariance_taker_share_match": int(inv_taker),
        "invariance_fill_delay_match": int(inv_delay),
    }
    return out, x["size_mult"]


def rank_policy_row(row: Dict[str, Any]) -> float:
    delta_exp = float(row.get("delta_expectancy_vs_flatsize_baseline", np.nan))
    cvar_imp = float(row.get("cvar_improve_ratio_vs_flat", np.nan))
    dd_imp = float(row.get("maxdd_improve_ratio_vs_flat", np.nan))
    run_red = float(row.get("weighted_loss_run_burden_reduction_vs_flat", np.nan))
    if not np.isfinite(delta_exp):
        delta_exp = -1e9
    if not np.isfinite(cvar_imp):
        cvar_imp = -1e9
    if not np.isfinite(dd_imp):
        dd_imp = -1e9
    if not np.isfinite(run_red):
        run_red = -1e9
    return float(8.0 * delta_exp + 1.5 * cvar_imp + 1.5 * dd_imp + 0.5 * run_red)


def evaluate_policy_grid(
    phase_dir: Path,
    dataset: pd.DataFrame,
    policies: List[Dict[str, Any]],
    qstats: Dict[str, Dict[str, float]],
    baseline_row: Dict[str, Any],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    rows = []
    size_map: Dict[str, pd.Series] = {}
    base = copy.deepcopy(baseline_row)
    base.update(
        {
            "policy_id": "flat_baseline",
            "policy_hash": "flat",
            "policy_family": "flat",
            "policy_desc": "flat_size_1.0",
            "candidate_budget_norm_pass": 1,
            "candidate_invariance_pass": 1,
            "delta_expectancy_vs_flatsize_baseline": 0.0,
            "cvar_improve_ratio_vs_flat": 0.0,
            "maxdd_improve_ratio_vs_flat": 0.0,
            "weighted_loss_run_burden_reduction_vs_flat": 0.0,
            "score_rank_key": 0.0,
        }
    )
    rows.append(base)
    size_map["flat_baseline"] = pd.Series(np.ones(len(dataset), dtype=float), index=dataset.index)

    for i, pol in enumerate(policies):
        met, size_series = evaluate_sizing_policy_on_dataset(
            dataset,
            policy=pol,
            qstats=qstats,
            baseline_unweighted=baseline_row,
            rng_seed=int(seed + i),
        )
        row = {
            "policy_id": str(pol["policy_id"]),
            "policy_hash": policy_hash(pol),
            "policy_family": str(pol["family"]),
            "policy_desc": json.dumps(pol, sort_keys=True),
        }
        row.update(met)
        row["delta_expectancy_vs_flatsize_baseline"] = float(row["exec_expectancy_net_weighted"] - baseline_row["exec_expectancy_net_weighted"])
        row["cvar_improve_ratio_vs_flat"] = safe_div(abs(baseline_row["cvar_5_weighted"]) - abs(row["cvar_5_weighted"]), abs(baseline_row["cvar_5_weighted"]))
        row["maxdd_improve_ratio_vs_flat"] = safe_div(abs(baseline_row["max_drawdown_weighted"]) - abs(row["max_drawdown_weighted"]), abs(baseline_row["max_drawdown_weighted"]))
        row["weighted_loss_run_burden_reduction_vs_flat"] = safe_div(
            float(baseline_row["weighted_loss_run_burden"] - row["weighted_loss_run_burden"]),
            float(max(1e-12, baseline_row["weighted_loss_run_burden"])),
        )
        budget_pass = int(np.isfinite(row["budget_norm_error"]) and abs(float(row["budget_norm_error"])) <= 0.02)
        invariance_pass = int(
            row["invariance_entries_match"] == 1
            and row["invariance_entry_rate_match"] == 1
            and row["invariance_taker_share_match"] == 1
            and row["invariance_fill_delay_match"] == 1
        )
        row["candidate_budget_norm_pass"] = int(budget_pass)
        row["candidate_invariance_pass"] = int(invariance_pass)
        row["score_rank_key"] = rank_policy_row(row)
        rows.append(row)
        size_map[str(pol["policy_id"])] = size_series
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["score_rank_key", "delta_expectancy_vs_flatsize_baseline", "cvar_improve_ratio_vs_flat", "maxdd_improve_ratio_vs_flat"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out, size_map


def route_signal_sets(sig_in: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    x = sig_in.copy().reset_index(drop=True)
    holdout_n = max(120, int(round(len(x) * 0.20)))
    route1 = x.iloc[-holdout_n:].copy().reset_index(drop=True)
    route2 = x.copy().reset_index(drop=True)
    return {"route1_holdout": route1, "route2_reslice": route2}


def choose_phaseaf_classification(result_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    c = result_df[result_df["policy_id"] != "flat_baseline"].copy()
    if c.empty:
        return "E_AF_NO_GO_INFRA_OR_REPRO_FAIL", c
    c["within_expectancy_tol"] = (to_num(c["delta_expectancy_vs_flatsize_baseline"]) >= -0.00002).astype(int)
    c["risk_pair_improve"] = (
        (to_num(c["cvar_improve_ratio_vs_flat"]) >= 0.01)
        & (to_num(c["maxdd_improve_ratio_vs_flat"]) >= 0.01)
    ).astype(int)
    c["split_ok"] = (
        to_num(c["min_split_expectancy_net_weighted"]).notna()
        & (to_num(c["min_split_expectancy_net_weighted"]) >= (float(result_df.loc[result_df["policy_id"] == "flat_baseline", "min_split_expectancy_net_weighted"].iloc[0]) - 0.0002))
    ).astype(int)
    c["af_candidate_pass"] = (
        (to_num(c["candidate_budget_norm_pass"]) == 1)
        & (to_num(c["candidate_invariance_pass"]) == 1)
        & (
            (to_num(c["delta_expectancy_vs_flatsize_baseline"]) > 0)
            | ((c["within_expectancy_tol"] == 1) & (c["risk_pair_improve"] == 1))
        )
        & (c["split_ok"] == 1)
    ).astype(int)
    pass_df = c[c["af_candidate_pass"] == 1].copy()
    if pass_df.empty:
        # Distinguish cosmetic risk vs expectancy damage.
        risk_any = c[(to_num(c["cvar_improve_ratio_vs_flat"]) > 0) | (to_num(c["maxdd_improve_ratio_vs_flat"]) > 0)]
        if risk_any.empty:
            return "C_AF_NO_GO_ONLY_COSMETIC_RISK", c
        worst_exp = float(to_num(c["delta_expectancy_vs_flatsize_baseline"]).max())
        if worst_exp < -0.00002:
            return "D_AF_NO_GO_EXPECTANCY_DAMAGE", c
        return "C_AF_NO_GO_ONLY_COSMETIC_RISK", c
    strong = pass_df[
        (to_num(pass_df["delta_expectancy_vs_flatsize_baseline"]) > 0)
        & (to_num(pass_df["cvar_improve_ratio_vs_flat"]) >= 0.05)
        & (to_num(pass_df["maxdd_improve_ratio_vs_flat"]) >= 0.05)
    ]
    if len(strong) >= 1:
        return "A_AF_GO_STRONG_SIZING_EDGE", c
    return "B_AF_GO_WEAK_BUT_PROMISING", c


def psr_proxy_from_pnl(pnl_sig: np.ndarray, eff_trials: float = 1.0) -> Tuple[float, float]:
    x = np.asarray(pnl_sig, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd <= 1e-12:
        psr = 1.0 if mu > 0 else 0.0
    else:
        z = (mu / sd) * math.sqrt(float(x.size))
        psr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    dsr = float(psr / max(1.0, math.sqrt(max(1.0, eff_trials))))
    return float(psr), float(dsr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase AF/AG/AH regime-aware sizing optimization (SOLUSDT, contract-locked, no mechanics changes)")
    ap.add_argument("--seed", type=int, default=20260307)
    ap.add_argument("--pilot-candidates", type=int, default=64)
    args = ap.parse_args()

    root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    af_dir = root / f"PHASEAF_SIZING_PILOT_{utc_tag()}"
    af_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    # Locked checks.
    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    met_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, met_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")
    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(met_fp)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha} != {LOCKED['expected_fee_sha']}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {met_sha} != {LOCKED['expected_metrics_sha']}")

    # Load phase AE reference and E1/E2 genomes.
    phase_ae_dir = Path(LOCKED["phase_ae_dir"]).resolve()
    if not phase_ae_dir.exists():
        raise FileNotFoundError(f"Phase AE dir missing: {phase_ae_dir}")
    ae_labels_fp = phase_ae_dir / "phaseAE_labels_dataset.parquet"
    if not ae_labels_fp.exists():
        ae_labels_fp = phase_ae_dir / "phaseAE_labels_dataset.csv"
    if not ae_labels_fp.exists():
        raise FileNotFoundError("Phase AE labels dataset not found")

    sig_in = ae.ensure_signals_schema(pd.read_csv(rep_fp))
    exec_pair = ae.load_exec_pair(root)
    if exec_pair["E1"]["genome_hash"] != LOCKED["primary_hash"]:
        raise RuntimeError("Primary hash mismatch")
    if exec_pair["E2"]["genome_hash"] != LOCKED["backup_hash"]:
        raise RuntimeError("Backup hash mismatch")

    # Contract freeze lock.
    lock_args = ae.build_args(signals_csv=rep_fp, seed=int(args.seed))
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=af_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze lock validation failed")

    # AF1 reproduction.
    met1, sig1, split1, _args1, bundle1 = ae.evaluate_exact(
        run_dir=af_dir,
        signals_df=sig_in,
        genome=exec_pair["E1"]["genome"],
        seed=int(args.seed),
        name="af_base_e1",
    )
    met2, sig2, split2, _args2, bundle2 = ae.evaluate_exact(
        run_dir=af_dir,
        signals_df=sig_in,
        genome=exec_pair["E2"]["genome"],
        seed=int(args.seed) + 1,
        name="af_base_e2",
    )
    pre_feat1 = ae.build_preentry_features(bundle1, exec_pair["E1"]["genome"])
    d1 = ae.build_trade_labels(sig1.merge(pre_feat1, on=["signal_id", "signal_time"], how="left"))
    d1 = parse_entry_rows(d1)
    pre_feat2 = ae.build_preentry_features(bundle2, exec_pair["E2"]["genome"])
    d2 = ae.build_trade_labels(sig2.merge(pre_feat2, on=["signal_id", "signal_time"], how="left"))
    d2 = parse_entry_rows(d2)

    qstats = build_quantile_stats(d1)

    # Flat baseline metrics.
    base_flat_metrics, _ = evaluate_sizing_policy_on_dataset(
        d1,
        policy={"policy_id": "flat", "family": "streak_control", "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0}, "weights": {}, "size_bounds": [0.5, 1.5]},
        qstats=qstats,
        baseline_unweighted={
            "entries_valid": int((d1["entry_for_labels"] == 1).sum()),
            "entry_rate": float(np.mean(d1["entry_for_labels"] == 1)),
            "taker_share": float(np.nanmean(d1.loc[d1["entry_for_labels"] == 1, "taker_flag"])),
            "p95_fill_delay_min": float(np.nanquantile(to_num(d1.loc[d1["entry_for_labels"] == 1, "fill_delay_min"]).dropna().to_numpy(dtype=float), 0.95)),
        },
        rng_seed=int(args.seed),
    )
    base_unweighted = {
        "entries_valid": int(base_flat_metrics["entries_valid"]),
        "entry_rate": float(base_flat_metrics["entry_rate"]),
        "taker_share": float(base_flat_metrics["taker_share"]),
        "p95_fill_delay_min": float(base_flat_metrics["p95_fill_delay_min"]),
    }

    # AE S1 anchor recreation.
    s1 = s1_policy(qstats=qstats)
    s1_met, _ = evaluate_sizing_policy_on_dataset(d1, s1, qstats=qstats, baseline_unweighted=base_unweighted, rng_seed=int(args.seed) + 3)

    repro_obj = {
        "generated_utc": utc_now(),
        "phase": "AF1",
        "run_dir": str(af_dir),
        "frozen_contract": {
            "representative_subset_csv": str(rep_fp),
            "canonical_fee_model": str(fee_fp),
            "canonical_metrics_definition": str(met_fp),
            "fee_sha256": fee_sha,
            "metrics_sha256": met_sha,
            "fee_hash_match": int(fee_sha == LOCKED["expected_fee_sha"]),
            "metrics_hash_match": int(met_sha == LOCKED["expected_metrics_sha"]),
        },
        "freeze_lock_validation": lock_validation,
        "baseline_candidates": {
            "E1": {"hash": exec_pair["E1"]["genome_hash"], "valid_for_ranking": int(met1.get("valid_for_ranking", 0)), "exec_expectancy_net": float(met1.get("overall_exec_expectancy_net", np.nan))},
            "E2": {"hash": exec_pair["E2"]["genome_hash"], "valid_for_ranking": int(met2.get("valid_for_ranking", 0)), "exec_expectancy_net": float(met2.get("overall_exec_expectancy_net", np.nan))},
        },
        "flat_weighted_metrics_e1": base_flat_metrics,
        "ae_s1_anchor_metrics_e1": s1_met,
        "ae_s1_delta_expectancy_vs_flat": float(s1_met["exec_expectancy_net_weighted"] - base_flat_metrics["exec_expectancy_net_weighted"]),
        "ae_s1_cvar_improve_vs_flat": safe_div(abs(base_flat_metrics["cvar_5_weighted"]) - abs(s1_met["cvar_5_weighted"]), abs(base_flat_metrics["cvar_5_weighted"])),
        "ae_s1_maxdd_improve_vs_flat": safe_div(abs(base_flat_metrics["max_drawdown_weighted"]) - abs(s1_met["max_drawdown_weighted"]), abs(base_flat_metrics["max_drawdown_weighted"])),
    }
    json_dump(af_dir / "phaseAF_reproduction_check.json", repro_obj)

    # AF2 sizing policy space.
    pol_space = {
        "phase": "AF2",
        "symbol": LOCKED["symbol"],
        "size_bounds_default": [0.50, 1.50],
        "normalization": "two-pass mean-normalization on taken entries; report budget_norm_error",
        "features": [
            "prior_loss_streak_len",
            "prior_rolling_tail_count_20",
            "pre3m_close_to_high_dist_bps",
            "pre3m_realized_vol_12",
            "pre3m_wick_ratio",
            "interaction(prior_rolling_loss_rate_5, pre3m_wick_ratio)",
            "interaction(pre3m_close_to_high_dist_bps, pre3m_realized_vol_12)",
        ],
        "families": {
            "step": {"risk->size": "piecewise non-increasing"},
            "smooth": {"risk->size": "linear clipped monotonic"},
            "streak": {"risk->size": "triggered downscale by prior streak/tail count"},
            "hybrid": {"risk->size": "smooth base + streak penalty + optional recovery bonus"},
            "ae_s1_anchor": {"risk->size": "AE anchor 4-flag score threshold"},
        },
        "constraints": {
            "monotonic_non_increasing_with_risk": True,
            "no_upsize_highest_risk_bucket": True,
            "entries_invariance_required": True,
            "budget_norm_error_abs_max": 0.02,
            "hard_gates_unchanged": True,
        },
    }
    yaml_lines = ["phase: AF2", "symbol: SOLUSDT", "size_bounds_default: [0.5, 1.5]", "families:"]
    for k, v in pol_space["families"].items():
        yaml_lines.append(f"  {k}:")
        yaml_lines.append(f"    risk_to_size: \"{v['risk->size']}\"")
    yaml_lines.append("features:")
    for f in pol_space["features"]:
        yaml_lines.append(f"  - \"{f}\"")
    yaml_lines.append("constraints:")
    for k, v in pol_space["constraints"].items():
        yaml_lines.append(f"  {k}: {str(v).lower() if isinstance(v, bool) else v}")
    write_text(af_dir / "phaseAF_sizing_policy_space.yaml", "\n".join(yaml_lines))

    # AF3 pilot search.
    rng = random.Random(int(args.seed))
    pols: List[Dict[str, Any]] = [s1, streak_control_policy()]
    seen = {policy_hash(p) for p in pols}
    for i in range(int(args.pilot_candidates)):
        p = random_policy(rng, pid=f"P{i:03d}")
        h = policy_hash(p)
        if h in seen:
            continue
        seen.add(h)
        pols.append(p)

    af_results, size_map = evaluate_policy_grid(af_dir, d1, pols, qstats=qstats, baseline_row=base_flat_metrics, seed=int(args.seed))

    # Backup confirmation top5 non-flat.
    top5 = af_results[af_results["policy_id"] != "flat_baseline"].head(5).copy()
    b_rows = []
    for _, r in top5.iterrows():
        pid = str(r["policy_id"])
        pol = next((p for p in pols if str(p["policy_id"]) == pid), None)
        if pol is None:
            continue
        bm2 = {
            "entries_valid": int((d2["entry_for_labels"] == 1).sum()),
            "entry_rate": float(np.mean(d2["entry_for_labels"] == 1)),
            "taker_share": float(np.nanmean(d2.loc[d2["entry_for_labels"] == 1, "taker_flag"])),
            "p95_fill_delay_min": float(np.nanquantile(to_num(d2.loc[d2["entry_for_labels"] == 1, "fill_delay_min"]).dropna().to_numpy(dtype=float), 0.95)),
        }
        flat2, _ = evaluate_sizing_policy_on_dataset(
            d2,
            policy={"policy_id": "flat2", "family": "streak_control", "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0}, "weights": {}, "size_bounds": [0.5, 1.5]},
            qstats=qstats,
            baseline_unweighted=bm2,
            rng_seed=int(args.seed) + 17,
        )
        m2, _ = evaluate_sizing_policy_on_dataset(d2, pol, qstats=qstats, baseline_unweighted=bm2, rng_seed=int(args.seed) + 29)
        b_rows.append(
            {
                "policy_id": pid,
                "backup_delta_expectancy_vs_flat": float(m2["exec_expectancy_net_weighted"] - flat2["exec_expectancy_net_weighted"]),
                "backup_cvar_improve_vs_flat": safe_div(abs(flat2["cvar_5_weighted"]) - abs(m2["cvar_5_weighted"]), abs(flat2["cvar_5_weighted"])),
                "backup_maxdd_improve_vs_flat": safe_div(abs(flat2["max_drawdown_weighted"]) - abs(m2["max_drawdown_weighted"]), abs(flat2["max_drawdown_weighted"])),
            }
        )
    bdf = pd.DataFrame(b_rows)
    af_results = af_results.merge(bdf, on="policy_id", how="left")
    af_results.to_csv(af_dir / "phaseAF_pilot_results.csv", index=False)

    # AF4 baseline/s1/pilot + component ablations.
    best_nonflat = af_results[af_results["policy_id"] != "flat_baseline"].head(1)
    best_pid = str(best_nonflat.iloc[0]["policy_id"]) if not best_nonflat.empty else "S1_anchor"
    focus = af_results[af_results["policy_id"].isin(["flat_baseline", "S1_anchor", "streak_only_control", best_pid])].copy()
    focus.to_csv(af_dir / "phaseAF_baseline_vs_s1_vs_pilot.csv", index=False)

    ab_rows = []
    best_pol = next((p for p in pols if str(p["policy_id"]) == best_pid), None)
    if best_pol is not None:
        variants = []
        variants.append(("best_full", copy.deepcopy(best_pol)))
        if best_pol.get("weights"):
            p1 = copy.deepcopy(best_pol)
            for k in ("w_streak", "w_tail20", "w_int1"):
                if k in p1["weights"]:
                    p1["weights"][k] = 0.0
            variants.append(("ablate_streak_terms", p1))
            p2 = copy.deepcopy(best_pol)
            for k in ("w_closehigh", "w_vol", "w_wick"):
                if k in p2["weights"]:
                    p2["weights"][k] = 0.0
            variants.append(("ablate_microstructure_terms", p2))
            p3 = copy.deepcopy(best_pol)
            for k in ("w_int1", "w_int2"):
                if k in p3["weights"]:
                    p3["weights"][k] = 0.0
            variants.append(("ablate_interactions", p3))
        p4 = copy.deepcopy(best_pol)
        if p4["family"] in {"smooth", "hybrid"} and "slope" in p4["params"]:
            p4["params"]["slope"] = 0.0
        elif p4["family"] == "step":
            p4["params"]["s_hi"] = 1.0
            p4["params"]["s_mid"] = 1.0
            p4["params"]["s_lo"] = 1.0
        elif p4["family"] in {"streak", "streak_control"}:
            p4["params"]["s_up"] = 1.0 if "s_up" in p4["params"] else p4["params"].get("size_up", 1.0)
            if "s_down" in p4["params"]:
                p4["params"]["s_down"] = 1.0
            if "size_down" in p4["params"]:
                p4["params"]["size_down"] = 1.0
        variants.append(("ablate_flatten_slope", p4))

        for vid, vp in variants:
            m, _ = evaluate_sizing_policy_on_dataset(d1, vp, qstats=qstats, baseline_unweighted=base_unweighted, rng_seed=int(args.seed) + 101)
            ab_rows.append(
                {
                    "ablation_id": vid,
                    "policy_id": best_pid,
                    "exec_expectancy_net_weighted": float(m["exec_expectancy_net_weighted"]),
                    "delta_expectancy_vs_flat": float(m["exec_expectancy_net_weighted"] - base_flat_metrics["exec_expectancy_net_weighted"]),
                    "cvar_improve_ratio_vs_flat": safe_div(abs(base_flat_metrics["cvar_5_weighted"]) - abs(m["cvar_5_weighted"]), abs(base_flat_metrics["cvar_5_weighted"])),
                    "maxdd_improve_ratio_vs_flat": safe_div(abs(base_flat_metrics["max_drawdown_weighted"]) - abs(m["max_drawdown_weighted"]), abs(base_flat_metrics["max_drawdown_weighted"])),
                    "weighted_loss_run_burden": float(m["weighted_loss_run_burden"]),
                    "entry_rate": float(m["entry_rate"]),
                    "budget_norm_error": float(m["budget_norm_error"]),
                }
            )
    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(af_dir / "phaseAF_component_ablations.csv", index=False)

    # AF5 decision.
    af_class, af_aug = choose_phaseaf_classification(af_results)
    af_decision = {
        "generated_utc": utc_now(),
        "phase": "AF",
        "classification": af_class,
        "go": int(af_class.startswith("A_") or af_class.startswith("B_")),
        "policy_count_evaluated": int(len(af_results) - 1),
        "top_policy_id": best_pid,
    }
    json_dump(af_dir / "phaseAF_decision.json", af_decision)

    rep_lines = []
    rep_lines.append("# Phase AF Pilot Report")
    rep_lines.append("")
    rep_lines.append(f"- Generated UTC: {utc_now()}")
    rep_lines.append(f"- Classification: **{af_class}**")
    rep_lines.append(f"- Policies evaluated (excl. flat): `{len(af_results)-1}`")
    rep_lines.append("")
    rep_lines.append("## Top Policies")
    rep_lines.append("")
    rep_lines.append(
        markdown_table(
            af_results.head(10),
            [
                "policy_id",
                "policy_family",
                "delta_expectancy_vs_flatsize_baseline",
                "cvar_improve_ratio_vs_flat",
                "maxdd_improve_ratio_vs_flat",
                "weighted_loss_run_burden_reduction_vs_flat",
                "budget_norm_error",
                "candidate_budget_norm_pass",
                "candidate_invariance_pass",
                "backup_delta_expectancy_vs_flat",
            ],
        )
    )
    rep_lines.append("")
    rep_lines.append("## Baseline / S1 / Best")
    rep_lines.append("")
    rep_lines.append(
        markdown_table(
            focus,
            [
                "policy_id",
                "exec_expectancy_net_weighted",
                "delta_expectancy_vs_flatsize_baseline",
                "cvar_improve_ratio_vs_flat",
                "maxdd_improve_ratio_vs_flat",
                "max_consecutive_losses",
                "streak_ge5_count",
                "weighted_loss_run_burden",
                "entry_rate",
                "budget_norm_error",
            ],
        )
    )
    write_text(af_dir / "phaseAF_pilot_report.md", "\n".join(rep_lines))

    next_lines = [
        "# Decision Next Step",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Phase AF classification: **{af_class}**",
    ]
    if af_decision["go"] == 1:
        next_lines.append("- Mainline: continue to Phase AG OOS/stress confirmation.")
        write_text(
            af_dir / "ready_to_launch_phaseAG_prompt.txt",
            "Phase AG sizing OOS/stress confirmation (SOLUSDT, contract-locked): evaluate top 1-3 AF sizing policies vs flat-size baseline on route1 holdout and route2 reslice, then run fee/slippage and sizing-perturbation stress matrix. Keep entries/execution mechanics unchanged, enforce budget normalization/invariance checks, and classify AG_PASS_ROBUST / AG_PASS_WEAK / AG_FAIL.",
        )
    else:
        next_lines.append("- Mainline: stop with AF NO_GO.")
        write_text(
            af_dir / "ready_to_launch_phaseAG_prompt.txt",
            "AF no-go fallback: perform deeper regime labeling and feature quality audit before any further sizing optimization; do not run AG/AH.",
        )
    write_text(af_dir / "decision_next_step.md", "\n".join(next_lines))

    if af_decision["go"] != 1:
        print(
            json.dumps(
                {
                    "furthest_phase": "AF",
                    "classification": af_class,
                    "mainline_status": "STOP_NO_GO",
                    "af_dir": str(af_dir),
                },
                sort_keys=True,
            )
        )
        return

    # =========================
    # Phase AG
    # =========================
    ag_dir = root / f"PHASEAG_SIZING_OOS_STRESS_{utc_tag()}"
    ag_dir.mkdir(parents=True, exist_ok=False)

    # Select top policies for AG.
    af_cands = af_aug.copy()
    af_cands = af_cands.sort_values(
        ["af_candidate_pass", "score_rank_key", "delta_expectancy_vs_flatsize_baseline"],
        ascending=[False, False, False],
    )
    top_ids = af_cands["policy_id"].head(3).astype(str).tolist()
    top_policies = [p for p in pols if str(p["policy_id"]) in top_ids]
    if not top_policies:
        top_policies = [s1]
        top_ids = ["S1_anchor"]

    # AG1 route-based OOS.
    routes = route_signal_sets(sig_in)
    route_rows = []
    for rid, sdf in routes.items():
        wf_splits = 5 if rid == "route1_holdout" else 7
        train_ratio = 0.70 if rid == "route1_holdout" else 0.65
        met_r, sig_r, split_r, _args_r, bun_r = ae.evaluate_exact(
            run_dir=ag_dir,
            signals_df=sdf,
            genome=exec_pair["E1"]["genome"],
            seed=int(args.seed) + (11 if rid == "route1_holdout" else 19),
            name=f"ag_{rid}_base",
            wf_splits=wf_splits,
            train_ratio=train_ratio,
        )
        pre_r = ae.build_preentry_features(bun_r, exec_pair["E1"]["genome"])
        d_r = parse_entry_rows(ae.build_trade_labels(sig_r.merge(pre_r, on=["signal_id", "signal_time"], how="left")))
        q_r = build_quantile_stats(d_r)
        bm_r = {
            "entries_valid": int((d_r["entry_for_labels"] == 1).sum()),
            "entry_rate": float(np.mean(d_r["entry_for_labels"] == 1)),
            "taker_share": float(np.nanmean(d_r.loc[d_r["entry_for_labels"] == 1, "taker_flag"])),
            "p95_fill_delay_min": float(np.nanquantile(to_num(d_r.loc[d_r["entry_for_labels"] == 1, "fill_delay_min"]).dropna().to_numpy(dtype=float), 0.95)),
        }
        flat_r, _ = evaluate_sizing_policy_on_dataset(
            d_r,
            policy={"policy_id": "flat", "family": "streak_control", "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0}, "weights": {}, "size_bounds": [0.5, 1.5]},
            qstats=q_r,
            baseline_unweighted=bm_r,
            rng_seed=int(args.seed) + 31,
        )
        route_rows.append(
            {
                "route_id": rid,
                "policy_id": "flat_baseline",
                "delta_expectancy_vs_flat": 0.0,
                "cvar_improve_ratio_vs_flat": 0.0,
                "maxdd_improve_ratio_vs_flat": 0.0,
                **flat_r,
            }
        )
        for p in top_policies:
            mr, _ = evaluate_sizing_policy_on_dataset(d_r, p, qstats=q_r, baseline_unweighted=bm_r, rng_seed=int(args.seed) + 37)
            route_rows.append(
                {
                    "route_id": rid,
                    "policy_id": str(p["policy_id"]),
                    "delta_expectancy_vs_flat": float(mr["exec_expectancy_net_weighted"] - flat_r["exec_expectancy_net_weighted"]),
                    "cvar_improve_ratio_vs_flat": safe_div(abs(flat_r["cvar_5_weighted"]) - abs(mr["cvar_5_weighted"]), abs(flat_r["cvar_5_weighted"])),
                    "maxdd_improve_ratio_vs_flat": safe_div(abs(flat_r["max_drawdown_weighted"]) - abs(mr["max_drawdown_weighted"]), abs(flat_r["max_drawdown_weighted"])),
                    **mr,
                }
            )
    route_df = pd.DataFrame(route_rows)
    route_df.to_csv(ag_dir / "phaseAG_oos_routes_results.csv", index=False)

    # AG2 stress matrix (full-route data on E1).
    stress_rows = []
    d_full = d1.copy()
    q_full = qstats
    bm_full = base_unweighted
    flat_full, _ = evaluate_sizing_policy_on_dataset(
        d_full,
        policy={"policy_id": "flat", "family": "streak_control", "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0}, "weights": {}, "size_bounds": [0.5, 1.5]},
        qstats=q_full,
        baseline_unweighted=bm_full,
        rng_seed=int(args.seed) + 41,
    )
    scenarios = [
        ("baseline", {"fee_mult_override": None, "noise_sigma": 0.0, "bounds_override": None, "policy_mod": None}),
        ("fee_mult_1p25", {"fee_mult_override": 1.25, "noise_sigma": 0.0, "bounds_override": None, "policy_mod": None}),
        ("fee_mult_1p50", {"fee_mult_override": 1.50, "noise_sigma": 0.0, "bounds_override": None, "policy_mod": None}),
        ("score_noise_0p05", {"fee_mult_override": None, "noise_sigma": 0.05, "bounds_override": None, "policy_mod": None}),
        ("cap_tight_0p6_1p4", {"fee_mult_override": None, "noise_sigma": 0.0, "bounds_override": (0.60, 1.40), "policy_mod": None}),
        ("slope_plus10pct", {"fee_mult_override": None, "noise_sigma": 0.0, "bounds_override": None, "policy_mod": "slope_plus"}),
        ("slope_minus10pct", {"fee_mult_override": None, "noise_sigma": 0.0, "bounds_override": None, "policy_mod": "slope_minus"}),
    ]

    def policy_with_mod(pol: Dict[str, Any], mod: Optional[str]) -> Dict[str, Any]:
        p = copy.deepcopy(pol)
        if mod is None:
            return p
        fam = str(p["family"])
        if fam in {"smooth", "hybrid"} and "slope" in p["params"]:
            if mod == "slope_plus":
                p["params"]["slope"] = float(p["params"]["slope"]) * 1.10
            if mod == "slope_minus":
                p["params"]["slope"] = float(p["params"]["slope"]) * 0.90
        elif fam == "step":
            if mod == "slope_plus":
                p["params"]["s_hi"] = min(1.50, float(p["params"]["s_hi"]) + 0.05)
                p["params"]["s_lo"] = max(0.50, float(p["params"]["s_lo"]) - 0.05)
            if mod == "slope_minus":
                p["params"]["s_hi"] = max(1.00, float(p["params"]["s_hi"]) - 0.05)
                p["params"]["s_lo"] = min(1.00, float(p["params"]["s_lo"]) + 0.05)
        return p

    for p in top_policies:
        for sc_id, sc in scenarios:
            flat_s, _ = evaluate_sizing_policy_on_dataset(
                d_full,
                policy={"policy_id": "flat", "family": "streak_control", "params": {"k_streak": 10_000, "size_down": 1.0, "size_up": 1.0}, "weights": {}, "size_bounds": [0.5, 1.5]},
                qstats=q_full,
                baseline_unweighted=bm_full,
                rng_seed=int(args.seed) + 43,
                fee_mult_override=sc["fee_mult_override"],
            )
            pol_mod = policy_with_mod(p, sc["policy_mod"])
            mr, _ = evaluate_sizing_policy_on_dataset(
                d_full,
                policy=pol_mod,
                qstats=q_full,
                baseline_unweighted=bm_full,
                rng_seed=int(args.seed) + 47,
                noise_sigma=sc["noise_sigma"],
                bounds_override=sc["bounds_override"],
                fee_mult_override=sc["fee_mult_override"],
            )
            stress_rows.append(
                {
                    "policy_id": str(p["policy_id"]),
                    "scenario_id": sc_id,
                    "delta_expectancy_vs_flat": float(mr["exec_expectancy_net_weighted"] - flat_s["exec_expectancy_net_weighted"]),
                    "cvar_improve_ratio_vs_flat": safe_div(abs(flat_s["cvar_5_weighted"]) - abs(mr["cvar_5_weighted"]), abs(flat_s["cvar_5_weighted"])),
                    "maxdd_improve_ratio_vs_flat": safe_div(abs(flat_s["max_drawdown_weighted"]) - abs(mr["max_drawdown_weighted"]), abs(flat_s["max_drawdown_weighted"])),
                    **mr,
                }
            )
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(ag_dir / "phaseAG_stress_matrix_results.csv", index=False)

    # AG3 anti-overfit controls.
    uniq_policies = {policy_hash(p): p for p in top_policies}
    series_map: Dict[str, np.ndarray] = {}
    for p in top_policies:
        m, s = evaluate_sizing_policy_on_dataset(d1, p, qstats=qstats, baseline_unweighted=base_unweighted, rng_seed=int(args.seed) + 61)
        valid = (d1["entry_for_labels"] == 1) & d1["pnl_net_trade_notional_dec"].notna()
        vec = np.zeros(len(d1), dtype=float)
        vmask = valid.to_numpy(dtype=bool)
        vec[vmask] = to_num(d1.loc[valid, "pnl_net_trade_notional_dec"]).to_numpy(dtype=float) * s.loc[valid].to_numpy(dtype=float)
        series_map[str(p["policy_id"])] = vec
    pol_ids = list(series_map.keys())
    corr_vals = []
    for i in range(len(pol_ids)):
        for j in range(i + 1, len(pol_ids)):
            a = series_map[pol_ids[i]]
            b = series_map[pol_ids[j]]
            if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(c):
                corr_vals.append(c)
    avg_corr = float(np.mean(corr_vals)) if corr_vals else float("nan")
    eff_uncorr = float(len(pol_ids))
    eff_corr = float(len(pol_ids) / (1.0 + (len(pol_ids) - 1.0) * max(0.0, avg_corr))) if len(pol_ids) else 0.0

    sig_lines = []
    sig_lines.append("# Phase AG Effective Trials Summary")
    sig_lines.append("")
    sig_lines.append(f"- Generated UTC: {utc_now()}")
    sig_lines.append(f"- Policy count (post duplicate collapse by hash): `{len(uniq_policies)}`")
    sig_lines.append(f"- Average pairwise return correlation: `{avg_corr:.6f}`")
    sig_lines.append(f"- Effective trials uncorrelated proxy: `{eff_uncorr:.4f}`")
    sig_lines.append(f"- Effective trials corr-adjusted proxy: `{eff_corr:.4f}`")
    sig_lines.append("- PSR/DSR are proxy metrics on weighted per-signal return vectors.")
    sig_lines.append("- Reality-check bootstrap: TODO (not implemented in this phase).")
    sig_rows = []
    for pid in pol_ids:
        psr, dsr = psr_proxy_from_pnl(series_map[pid], eff_trials=max(1.0, eff_corr))
        sig_rows.append({"policy_id": pid, "psr_proxy": psr, "dsr_proxy": dsr})
    sdf = pd.DataFrame(sig_rows)
    if not sdf.empty:
        sig_lines.append("")
        sig_lines.append(markdown_table(sdf, ["policy_id", "psr_proxy", "dsr_proxy"]))
    write_text(ag_dir / "phaseAG_effective_trials_summary.md", "\n".join(sig_lines))

    # AG4 decision.
    ag_eval = []
    for pid in top_ids:
        rr = route_df[route_df["policy_id"] == pid].copy()
        ss = stress_df[stress_df["policy_id"] == pid].copy()
        if rr.empty or ss.empty:
            continue
        route_pass = int(
            np.sum(
                (to_num(rr["delta_expectancy_vs_flat"]) >= -0.00002)
                & (to_num(rr["cvar_improve_ratio_vs_flat"]) >= 0.0)
                & (to_num(rr["maxdd_improve_ratio_vs_flat"]) >= 0.0)
            )
        )
        stress_sign = float(
            np.mean(
                (to_num(ss["delta_expectancy_vs_flat"]) >= -0.00002)
                & (to_num(ss["cvar_improve_ratio_vs_flat"]) >= 0.0)
                & (to_num(ss["maxdd_improve_ratio_vs_flat"]) >= 0.0)
            )
        )
        ag_eval.append({"policy_id": pid, "route_pass_count": route_pass, "route_total": int(len(rr)), "stress_retention": stress_sign})
    ag_df = pd.DataFrame(ag_eval)
    if ag_df.empty:
        ag_class = "D_AG_FAIL_NO_OOS"
    else:
        robust = ag_df[(ag_df["route_pass_count"] >= 2) & (ag_df["stress_retention"] >= 0.70)]
        weak = ag_df[(ag_df["route_pass_count"] >= 1) & (ag_df["stress_retention"] >= 0.50)]
        if not robust.empty:
            ag_class = "A_AG_PASS_ROBUST"
        elif not weak.empty:
            ag_class = "B_AG_PASS_WEAK"
        else:
            ag_class = "C_AG_FAIL_BRITTLE"

    ag_report_lines = []
    ag_report_lines.append("# Phase AG Robustness Report")
    ag_report_lines.append("")
    ag_report_lines.append(f"- Generated UTC: {utc_now()}")
    ag_report_lines.append(f"- Classification: **{ag_class}**")
    ag_report_lines.append("")
    ag_report_lines.append("## OOS Route Results")
    ag_report_lines.append("")
    ag_report_lines.append(markdown_table(route_df, ["route_id", "policy_id", "delta_expectancy_vs_flat", "cvar_improve_ratio_vs_flat", "maxdd_improve_ratio_vs_flat", "min_split_expectancy_net_weighted"]))
    ag_report_lines.append("")
    ag_report_lines.append("## Stress Matrix")
    ag_report_lines.append("")
    ag_report_lines.append(markdown_table(stress_df, ["policy_id", "scenario_id", "delta_expectancy_vs_flat", "cvar_improve_ratio_vs_flat", "maxdd_improve_ratio_vs_flat", "budget_norm_error"]))
    ag_report_lines.append("")
    ag_report_lines.append("## Policy Robustness Summary")
    ag_report_lines.append("")
    ag_report_lines.append(markdown_table(ag_df, ["policy_id", "route_pass_count", "route_total", "stress_retention"]))
    write_text(ag_dir / "phaseAG_robustness_report.md", "\n".join(ag_report_lines))

    ag_dec = {"generated_utc": utc_now(), "phase": "AG", "classification": ag_class, "go": int(ag_class.startswith("A_") or ag_class.startswith("B_"))}
    json_dump(ag_dir / "phaseAG_decision.json", ag_dec)

    if ag_dec["go"] != 1:
        write_text(
            ag_dir / "ready_to_launch_phaseAH_prompt.txt",
            "AG fail/no-oos fallback: stop promotion, run targeted robustness repair on sizing policy stability (route/sensitivity) before any paper deployment.",
        )
        print(
            json.dumps(
                {
                    "furthest_phase": "AG",
                    "classification": ag_class,
                    "mainline_status": "STOP_NO_GO",
                    "af_dir": str(af_dir),
                    "ag_dir": str(ag_dir),
                },
                sort_keys=True,
            )
        )
        return

    write_text(
        ag_dir / "ready_to_launch_phaseAH_prompt.txt",
        "Phase AH sizing promotion package: prepare paper/shadow deployment policy cards for top AG survivors, monitoring spec, and rollback triggers on top of locked execution candidate hashes.",
    )

    # =========================
    # Phase AH
    # =========================
    ah_dir = root / f"PHASEAH_SIZING_PROMOTION_{utc_tag()}"
    ah_dir.mkdir(parents=True, exist_ok=False)

    # Select primary/backup policy by AG robustness then AF score.
    merged = af_results.merge(ag_df, on="policy_id", how="inner")
    merged = merged.sort_values(
        ["route_pass_count", "stress_retention", "score_rank_key", "delta_expectancy_vs_flatsize_baseline"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    primary = merged.iloc[0] if not merged.empty else None
    backup = merged.iloc[1] if len(merged) > 1 else None

    if primary is None:
        ah_class = "C_AH_NO_PROMOTION"
    else:
        if ag_class.startswith("A_"):
            ah_class = "A_AH_PROMOTE_PAPER"
        elif ag_class.startswith("B_"):
            ah_class = "B_AH_PROMOTE_PAPER_CAUTION"
        else:
            ah_class = "C_AH_NO_PROMOTION"

    # Build promotion report.
    pr_lines = []
    pr_lines.append("# Phase AH Promotion Report")
    pr_lines.append("")
    pr_lines.append(f"- Generated UTC: {utc_now()}")
    pr_lines.append(f"- Classification: **{ah_class}**")
    pr_lines.append(f"- Base execution primary hash: `{exec_pair['E1']['genome_hash']}`")
    pr_lines.append(f"- Base execution backup hash: `{exec_pair['E2']['genome_hash']}`")
    pr_lines.append("")
    if primary is not None:
        pr_lines.append("## Primary Sizing Policy")
        pr_lines.append("")
        pr_lines.append(f"- policy_id: `{primary['policy_id']}`")
        pr_lines.append(f"- family: `{primary['policy_family']}`")
        pr_lines.append(f"- delta_expectancy_vs_flat: `{float(primary['delta_expectancy_vs_flatsize_baseline']):.8f}`")
        pr_lines.append(f"- cvar_improve_ratio_vs_flat: `{float(primary['cvar_improve_ratio_vs_flat']):.6f}`")
        pr_lines.append(f"- maxdd_improve_ratio_vs_flat: `{float(primary['maxdd_improve_ratio_vs_flat']):.6f}`")
        pr_lines.append(f"- weighted_loss_run_burden_reduction: `{float(primary['weighted_loss_run_burden_reduction_vs_flat']):.6f}`")
        pr_lines.append(f"- route_pass_count/stress_retention: `{int(primary['route_pass_count'])}/{int(primary['route_total'])}` / `{float(primary['stress_retention']):.4f}`")
        pr_lines.append("")
    if backup is not None:
        pr_lines.append("## Backup Sizing Policy")
        pr_lines.append("")
        pr_lines.append(f"- policy_id: `{backup['policy_id']}`")
        pr_lines.append(f"- family: `{backup['policy_family']}`")
        pr_lines.append(f"- delta_expectancy_vs_flat: `{float(backup['delta_expectancy_vs_flatsize_baseline']):.8f}`")
        pr_lines.append(f"- cvar_improve_ratio_vs_flat: `{float(backup['cvar_improve_ratio_vs_flat']):.6f}`")
        pr_lines.append(f"- maxdd_improve_ratio_vs_flat: `{float(backup['maxdd_improve_ratio_vs_flat']):.6f}`")
        pr_lines.append(f"- route_pass_count/stress_retention: `{int(backup['route_pass_count'])}/{int(backup['route_total'])}` / `{float(backup['stress_retention']):.4f}`")
        pr_lines.append("")
    write_text(ah_dir / "phaseAH_promotion_report.md", "\n".join(pr_lines))

    monitor_yaml = []
    monitor_yaml.append("phase: AH2")
    monitor_yaml.append("mode: paper_shadow")
    monitor_yaml.append("symbol: SOLUSDT")
    monitor_yaml.append("base_execution_hashes:")
    monitor_yaml.append(f"  primary: {exec_pair['E1']['genome_hash']}")
    monitor_yaml.append(f"  backup: {exec_pair['E2']['genome_hash']}")
    monitor_yaml.append("monitor_fields:")
    monitor_yaml.append("  - realized_weighted_expectancy_delta_vs_flat")
    monitor_yaml.append("  - realized_drawdown_delta_vs_flat")
    monitor_yaml.append("  - weighted_tail_loss_share")
    monitor_yaml.append("  - weighted_loss_run_burden")
    monitor_yaml.append("  - score_distribution_drift")
    monitor_yaml.append("  - feature_drift")
    monitor_yaml.append("  - size_distribution_drift")
    monitor_yaml.append("  - policy_activation_rate")
    monitor_yaml.append("  - budget_norm_error")
    write_text(ah_dir / "phaseAH_paper_monitor_spec.yaml", "\n".join(monitor_yaml))

    rollback_yaml = []
    rollback_yaml.append("phase: AH3")
    rollback_yaml.append("rollback_triggers:")
    rollback_yaml.append("  hard:")
    rollback_yaml.append("    - condition: realized_weighted_expectancy_delta_vs_flat < -0.00005 for 2 consecutive review windows")
    rollback_yaml.append("    - condition: realized_drawdown_delta_vs_flat < -0.02 in any review window")
    rollback_yaml.append("    - condition: budget_norm_error_abs > 0.03")
    rollback_yaml.append("  caution:")
    rollback_yaml.append("    - condition: stress_retention_proxy drops below 0.5")
    rollback_yaml.append("    - condition: policy_activation_rate outside [0.05, 0.95] for 3 windows")
    write_text(ah_dir / "phaseAH_rollback_triggers.yaml", "\n".join(rollback_yaml))

    ah_dec = {"generated_utc": utc_now(), "phase": "AH", "classification": ah_class, "promotion": int(ah_class.startswith("A_") or ah_class.startswith("B_"))}
    json_dump(ah_dir / "phaseAH_decision.json", ah_dec)

    if ah_dec["promotion"] == 1:
        write_text(
            ah_dir / "ready_to_launch_paper_shadow_prompt.txt",
            "Launch SOLUSDT paper/shadow sizing overlay on top of locked E1 execution with backup policy fallback. Keep execution mechanics unchanged, monitor weighted delta expectancy/drawdown/tail burden vs flat shadow, and enforce rollback triggers from phaseAH_rollback_triggers.yaml.",
        )

    # Final summary marker for tooling.
    print(
        json.dumps(
            {
                "furthest_phase": "AH",
                "af_classification": af_class,
                "ag_classification": ag_class,
                "ah_classification": ah_class,
                "mainline_status": "PROMOTE_PAPER" if ah_class.startswith("A_") else ("PROMOTE_PAPER_CAUTION" if ah_class.startswith("B_") else "STOP_NO_GO"),
                "af_dir": str(af_dir),
                "ag_dir": str(ag_dir),
                "ah_dir": str(ah_dir),
                "duration_sec": float(time.time() - t0),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
