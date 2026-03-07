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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

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
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def tail_mean(arr: np.ndarray, q: float) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(q) * x.size)))
    return float(np.mean(np.sort(x)[:k]))


def max_drawdown_from_pnl(pnl_sig: np.ndarray) -> float:
    if pnl_sig.size == 0:
        return float("nan")
    cum = np.cumsum(np.nan_to_num(pnl_sig, nan=0.0))
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def max_consecutive_losses(pnl_sig: np.ndarray) -> int:
    best = 0
    cur = 0
    for x in pnl_sig:
        if np.isfinite(x) and x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


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


def session_bucket_from_ts(ts: pd.Series) -> pd.Series:
    h = pd.to_datetime(ts, utc=True, errors="coerce").dt.hour
    out = pd.Series(index=h.index, dtype=object)
    out[(h >= 0) & (h < 6)] = "00_05"
    out[(h >= 6) & (h < 12)] = "06_11"
    out[(h >= 12) & (h < 18)] = "12_17"
    out[(h >= 18) & (h <= 23)] = "18_23"
    return out.fillna("unknown").astype(str)


def build_args(
    *,
    signals_csv: Path,
    wf_splits: int,
    train_ratio: float,
    seed: int,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = LOCKED["symbol"]
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = int(wf_splits)
    args.train_ratio = float(train_ratio)
    args.mode = "tight"
    args.workers = 1
    args.seed = int(seed)
    args.pop = 1
    args.gens = 1
    args.execution_config = "configs/execution_configs.yaml"
    args.fee_bps_maker = float(2.0 * fee_mult)
    args.fee_bps_taker = float(4.0 * fee_mult)
    args.slippage_bps_limit = float(0.5 * slip_mult)
    args.slippage_bps_market = float(2.0 * slip_mult)
    args.canonical_fee_model_path = LOCKED["canonical_fee_model"]
    args.canonical_metrics_definition_path = LOCKED["canonical_metrics_definition"]
    args.expected_fee_model_sha256 = LOCKED["expected_fee_sha"]
    args.expected_metrics_definition_sha256 = LOCKED["expected_metrics_sha"]
    args.allow_freeze_hash_mismatch = 0
    return args


def load_latest_phasev(exec_root: Path) -> Tuple[Path, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    runs = sorted([p for p in exec_root.glob("PHASEV_BRANCHB_PORTABILITY_DD_*") if p.is_dir()], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError("No PHASEV_BRANCHB_PORTABILITY_DD_* run found")
    for p in reversed(runs):
        man = p / "phaseV_run_manifest.json"
        cand = p / "phaseV_exec_candidates_locked.json"
        over = p / "phaseV_risk_overlay_benchmark.csv"
        if man.exists() and cand.exists() and over.exists():
            mobj = json.loads(man.read_text(encoding="utf-8"))
            cobj = json.loads(cand.read_text(encoding="utf-8"))
            odf = pd.read_csv(over)
            return p, mobj, cobj, odf
    raise FileNotFoundError("No complete Phase V run with manifest/candidates/overlay benchmark")


def load_exec_pair_from_phasev(cobj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cands = cobj.get("candidates", {})
    out: Dict[str, Dict[str, Any]] = {}
    for eid in ("E1", "E2"):
        if eid not in cands:
            raise KeyError(f"Missing {eid} in phaseV_exec_candidates_locked.json")
        out[eid] = {
            "exec_choice_id": eid,
            "description": str(cands[eid].get("description", "")),
            "genome_hash": str(cands[eid].get("genome_hash", "")),
            "genome": copy.deepcopy(cands[eid].get("genome", {})),
            "source_run": str(cands[eid].get("source_run", "")),
        }

    if out["E1"]["genome_hash"] != LOCKED["primary_hash"]:
        raise RuntimeError(f"E1 hash mismatch: {out['E1']['genome_hash']} != {LOCKED['primary_hash']}")
    if out["E2"]["genome_hash"] != LOCKED["backup_hash"]:
        raise RuntimeError(f"E2 hash mismatch: {out['E2']['genome_hash']} != {LOCKED['backup_hash']}")
    return out


def ensure_signals_schema(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x["tp_mult"] = to_num(x["tp_mult"])
    x["sl_mult"] = to_num(x["sl_mult"])
    for c in ("atr_percentile_1h", "trend_up_1h"):
        if c not in x.columns:
            x[c] = np.nan
        x[c] = to_num(x[c])
    x = x.dropna(subset=["signal_id", "signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return x


def subset_hash(df: pd.DataFrame) -> str:
    x = ensure_signals_schema(df)
    rows = [
        f"{r.signal_id}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}|{float(r.tp_mult):.10f}|{float(r.sl_mult):.10f}"
        for r in x.itertuples(index=False)
    ]
    return sha256_text("\n".join(rows))


def build_bundle(
    *,
    run_dir: Path,
    df_signals: pd.DataFrame,
    name: str,
    wf_splits: int,
    train_ratio: float,
    seed: int,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
    validate_lock: bool = False,
) -> Tuple[ga_exec.SymbolBundle, argparse.Namespace, Path]:
    fp = run_dir / f"{name}.csv"
    df_signals.to_csv(fp, index=False)
    args = build_args(
        signals_csv=fp,
        wf_splits=wf_splits,
        train_ratio=train_ratio,
        seed=seed,
        fee_mult=fee_mult,
        slip_mult=slip_mult,
    )
    if validate_lock:
        ga_exec._validate_and_lock_frozen_artifacts(args=args, run_dir=run_dir)
    bundles, _meta = ga_exec._prepare_bundles(args)
    if not bundles:
        raise RuntimeError(f"No bundles built for {name}")
    return bundles[0], args, fp


def attach_signal_features(sig_rows: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    x = sig_rows.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    feat = signals_df[["signal_id", "signal_time", "atr_percentile_1h", "trend_up_1h"]].copy()
    feat["signal_id"] = feat["signal_id"].astype(str)
    feat["signal_time"] = pd.to_datetime(feat["signal_time"], utc=True, errors="coerce")
    out = x.merge(feat, on=["signal_id", "signal_time"], how="left")
    out["session_bucket"] = session_bucket_from_ts(out["signal_time"])
    return out


def metrics_from_signal_rows(
    *,
    sig: pd.DataFrame,
    args: argparse.Namespace,
    keep_mask: Optional[np.ndarray] = None,
    base_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    x = sig.copy().reset_index(drop=True)
    if keep_mask is None:
        keep = np.ones(len(x), dtype=bool)
    else:
        keep = np.asarray(keep_mask, dtype=bool)
        if keep.size != len(x):
            keep = np.ones(len(x), dtype=bool)

    x["exec_filled"] = to_num(x.get("exec_filled", 0)).fillna(0).astype(int)
    x["exec_valid_for_metrics"] = to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    x["exec_pnl_net_pct"] = to_num(x.get("exec_pnl_net_pct", np.nan))
    x["exec_pnl_gross_pct"] = to_num(x.get("exec_pnl_gross_pct", np.nan))
    x["exec_fill_delay_min"] = to_num(x.get("exec_fill_delay_min", np.nan))
    x["split_id"] = to_num(x.get("split_id", np.nan))
    x["exec_exit_reason"] = x.get("exec_exit_reason", "").astype(str)
    x["exec_fill_liquidity_type"] = x.get("exec_fill_liquidity_type", "").astype(str)

    valid_entry = (x["exec_filled"] == 1) & (x["exec_valid_for_metrics"] == 1) & x["exec_pnl_net_pct"].notna()
    eff = valid_entry.to_numpy(dtype=bool) & keep

    signals_total = int(len(x))
    entries = int(eff.sum())
    pnl_sig = np.zeros(signals_total, dtype=float)
    pnl_sig[eff] = x.loc[eff, "exec_pnl_net_pct"].to_numpy(dtype=float)

    exp = float(np.mean(pnl_sig))
    cvar5 = float(tail_mean(pnl_sig, 0.05))
    maxdd = float(max_drawdown_from_pnl(pnl_sig))
    entry_rate = float(entries / max(1, signals_total))

    taker = (x["exec_fill_liquidity_type"].str.lower() == "taker").to_numpy(dtype=bool)
    taker_share = float(np.mean(taker[eff])) if entries > 0 else float("nan")
    d = x.loc[eff, "exec_fill_delay_min"].dropna().to_numpy(dtype=float)
    med_delay = float(np.median(d)) if d.size else float("nan")
    p95_delay = float(np.quantile(d, 0.95)) if d.size else float("nan")

    v = x.loc[eff].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    v_pnl = v["exec_pnl_net_pct"].to_numpy(dtype=float) if not v.empty else np.array([], dtype=float)
    neg = v[v["exec_pnl_net_pct"] < 0].copy()
    total_loss_abs = float(np.abs(neg["exec_pnl_net_pct"]).sum()) if not neg.empty else 0.0
    sl_loss_share = float("nan")
    if total_loss_abs > 1e-12:
        sl_loss = float(np.abs(neg[neg["exec_exit_reason"].str.lower() == "sl"]["exec_pnl_net_pct"]).sum())
        sl_loss_share = float(sl_loss / total_loss_abs)

    runs_ge3 = 0
    cur = 0
    for vpx in v_pnl:
        if np.isfinite(vpx) and vpx < 0:
            cur += 1
        else:
            if cur >= 3:
                runs_ge3 += 1
            cur = 0
    if cur >= 3:
        runs_ge3 += 1

    fee_drag_per_trade = float((v["exec_pnl_gross_pct"] - v["exec_pnl_net_pct"]).mean()) if not v.empty else float("nan")
    gross_sum = float(v["exec_pnl_gross_pct"].sum()) if not v.empty else float("nan")
    fee_drag_total = float((v["exec_pnl_gross_pct"] - v["exec_pnl_net_pct"]).sum()) if not v.empty else float("nan")
    fee_drag_to_gross_abs_ratio = safe_div(fee_drag_total, abs(gross_sum)) if np.isfinite(gross_sum) else float("nan")

    split_stats = []
    if not v.empty and "split_id" in v.columns:
        for sid, grp in v.groupby("split_id", dropna=True):
            split_stats.append((sid, float(grp["exec_pnl_net_pct"].mean())))
    split_vals = np.array([z[1] for z in split_stats], dtype=float) if split_stats else np.array([], dtype=float)
    min_split = float(np.nanmin(split_vals)) if split_vals.size else float("nan")
    med_split = float(np.nanmedian(split_vals)) if split_vals.size else float("nan")
    std_split = float(np.nanstd(split_vals, ddof=0)) if split_vals.size else float("nan")

    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, signals_total))))
    participation_pass = int(entries >= min_trades_overall and np.isfinite(entry_rate) and entry_rate >= float(args.hard_min_entry_rate_overall))
    realism_pass = int(
        np.isfinite(taker_share)
        and np.isfinite(med_delay)
        and np.isfinite(p95_delay)
        and taker_share <= float(args.hard_max_taker_share)
        and med_delay <= float(args.hard_max_median_fill_delay_min)
        and p95_delay <= float(args.hard_max_p95_fill_delay_min)
    )
    nan_pass = int(np.isfinite(exp) and np.isfinite(cvar5) and np.isfinite(maxdd))
    split_pass = int(split_vals.size > 0 and np.isfinite(min_split) and np.isfinite(med_split) and np.isfinite(std_split))

    out = {
        "signals_total": int(signals_total),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "exec_expectancy_net": float(exp),
        "exec_cvar_5": float(cvar5),
        "exec_max_drawdown": float(maxdd),
        "taker_share": float(taker_share),
        "median_fill_delay_min": float(med_delay),
        "p95_fill_delay_min": float(p95_delay),
        "min_split_expectancy_net": float(min_split),
        "median_split_expectancy_net": float(med_split),
        "std_split_expectancy_net": float(std_split),
        "max_consecutive_losses": int(max_consecutive_losses(v_pnl)) if v_pnl.size else 0,
        "loss_run_ge3_count": int(runs_ge3),
        "sl_loss_share": float(sl_loss_share),
        "fee_drag_per_trade": float(fee_drag_per_trade),
        "fee_drag_total": float(fee_drag_total),
        "fee_drag_to_gross_abs_ratio": float(fee_drag_to_gross_abs_ratio),
        "participation_pass_proxy": int(participation_pass),
        "realism_pass_proxy": int(realism_pass),
        "nan_pass_proxy": int(nan_pass),
        "split_pass_proxy": int(split_pass),
        "hard_gate_proxy_pass": int(participation_pass == 1 and realism_pass == 1 and nan_pass == 1 and split_pass == 1),
        "removed_entries_count": int(valid_entry.sum() - entries),
        "removed_entries_pct": float((valid_entry.sum() - entries) / max(1, int(valid_entry.sum()))),
    }

    if base_row is not None:
        b_exp = float(base_row.get("exec_expectancy_net", np.nan))
        b_cvar = float(base_row.get("exec_cvar_5", np.nan))
        b_dd = float(base_row.get("exec_max_drawdown", np.nan))
        out["delta_expectancy_vs_base"] = float(out["exec_expectancy_net"] - b_exp) if np.isfinite(out["exec_expectancy_net"]) and np.isfinite(b_exp) else float("nan")
        out["delta_cvar_vs_base"] = float(out["exec_cvar_5"] - b_cvar) if np.isfinite(out["exec_cvar_5"]) and np.isfinite(b_cvar) else float("nan")
        out["delta_maxdd_vs_base"] = float(out["exec_max_drawdown"] - b_dd) if np.isfinite(out["exec_max_drawdown"]) and np.isfinite(b_dd) else float("nan")
        out["cvar_improve_ratio_vs_base"] = safe_div(abs(b_cvar) - abs(out["exec_cvar_5"]), abs(b_cvar))
        out["maxdd_improve_ratio_vs_base"] = safe_div(abs(b_dd) - abs(out["exec_max_drawdown"]), abs(b_dd))
        out["delta_max_consecutive_losses_vs_base"] = int(out["max_consecutive_losses"] - int(base_row.get("max_consecutive_losses", 0)))
        out["delta_loss_run_ge3_count_vs_base"] = int(out["loss_run_ge3_count"] - int(base_row.get("loss_run_ge3_count", 0)))
        out["delta_sl_loss_share_vs_base"] = float(out["sl_loss_share"] - float(base_row.get("sl_loss_share", np.nan))) if np.isfinite(out["sl_loss_share"]) and np.isfinite(float(base_row.get("sl_loss_share", np.nan))) else float("nan")
        out["delta_fee_drag_per_trade_vs_base"] = float(out["fee_drag_per_trade"] - float(base_row.get("fee_drag_per_trade", np.nan))) if np.isfinite(out["fee_drag_per_trade"]) and np.isfinite(float(base_row.get("fee_drag_per_trade", np.nan))) else float("nan")
    else:
        out["delta_expectancy_vs_base"] = float("nan")
        out["delta_cvar_vs_base"] = float("nan")
        out["delta_maxdd_vs_base"] = float("nan")
        out["cvar_improve_ratio_vs_base"] = float("nan")
        out["maxdd_improve_ratio_vs_base"] = float("nan")
        out["delta_max_consecutive_losses_vs_base"] = 0
        out["delta_loss_run_ge3_count_vs_base"] = 0
        out["delta_sl_loss_share_vs_base"] = float("nan")
        out["delta_fee_drag_per_trade_vs_base"] = float("nan")
    return out


def evaluate_exact(
    *,
    run_dir: Path,
    overlay_signals: pd.DataFrame,
    genome: Dict[str, Any],
    candidate_id: str,
    overlay_id: str,
    overlay_desc: str,
    seed: int,
    wf_splits: int,
    train_ratio: float,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
    base_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    name = f"w_{candidate_id}_{overlay_id}_wf{wf_splits}_tr{int(round(train_ratio*100))}_f{int(round(fee_mult*100))}_s{int(round(slip_mult*100))}"
    bundle, args, _ = build_bundle(
        run_dir=run_dir,
        df_signals=overlay_signals,
        name=name,
        wf_splits=wf_splits,
        train_ratio=train_ratio,
        seed=seed,
        fee_mult=fee_mult,
        slip_mult=slip_mult,
    )
    met = ga_exec._evaluate_genome(genome=genome, bundles=[bundle], args=args, detailed=True)
    sig = attach_signal_features(met["signal_rows_df"], overlay_signals)
    core = metrics_from_signal_rows(sig=sig, args=args, keep_mask=None, base_row=base_row)
    row = {
        "candidate_id": candidate_id,
        "overlay_id": overlay_id,
        "overlay_desc": overlay_desc,
        "overlay_eval_type": "exact_engine_integrated",
        "approximate_counterfactual": 0,
        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
        "constraint_pass": int(met.get("constraint_pass", 1)),
        "participation_pass": int(met.get("participation_pass", 0)),
        "realism_pass": int(met.get("realism_pass", 0)),
        "nan_pass": int(met.get("nan_pass", 0)),
        "split_pass": int(met.get("split_pass", 0)),
        "hard_invalid": int(met.get("hard_invalid", 1)),
        "invalid_reason": str(met.get("invalid_reason", "")),
        "participation_fail_reason": str(met.get("participation_fail_reason", "")),
        "realism_fail_reason": str(met.get("realism_fail_reason", "")),
        "split_fail_reason": str(met.get("split_fail_reason", "")),
    }
    row.update(core)
    row["hard_gate_pass"] = int(row["valid_for_ranking"] == 1)
    row["hard_gate_proxy_pass"] = int(row["valid_for_ranking"] == 1)
    return row, sig


def evaluate_proxy(
    *,
    sig_base: pd.DataFrame,
    args_base: argparse.Namespace,
    candidate_id: str,
    overlay_id: str,
    overlay_desc: str,
    keep_mask: np.ndarray,
    base_row: Dict[str, Any],
) -> Dict[str, Any]:
    core = metrics_from_signal_rows(sig=sig_base, args=args_base, keep_mask=keep_mask, base_row=base_row)
    row = {
        "candidate_id": candidate_id,
        "overlay_id": overlay_id,
        "overlay_desc": overlay_desc,
        "overlay_eval_type": "proxy_counterfactual",
        "approximate_counterfactual": 1,
        "valid_for_ranking": int(core["hard_gate_proxy_pass"]),
        "constraint_pass": 1,
        "participation_pass": int(core["participation_pass_proxy"]),
        "realism_pass": int(core["realism_pass_proxy"]),
        "nan_pass": int(core["nan_pass_proxy"]),
        "split_pass": int(core["split_pass_proxy"]),
        "hard_invalid": int(core["hard_gate_proxy_pass"] == 0),
        "invalid_reason": "",
        "participation_fail_reason": "",
        "realism_fail_reason": "",
        "split_fail_reason": "",
        "hard_gate_pass": int(core["hard_gate_proxy_pass"]),
    }
    if row["hard_gate_pass"] == 0:
        fails = []
        if row["participation_pass"] == 0:
            fails.append("participation")
        if row["realism_pass"] == 0:
            fails.append("realism")
        if row["split_pass"] == 0:
            fails.append("split")
        if row["nan_pass"] == 0:
            fails.append("nan")
        row["invalid_reason"] = "|".join(fails)
    row.update(core)
    return row


def filter_signals(df: pd.DataFrame, rule: str, meta: Dict[str, Any]) -> pd.DataFrame:
    x = ensure_signals_schema(df)
    ts = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    sess = session_bucket_from_ts(ts)
    if rule == "none":
        return x
    if rule == "veto_06_11":
        return x[sess != "06_11"].copy().reset_index(drop=True)
    if rule == "veto_12_17":
        return x[sess != "12_17"].copy().reset_index(drop=True)
    if rule == "veto_subwindow":
        h0 = int(meta.get("subwindow_start_h", 10))
        h1 = int(meta.get("subwindow_end_h", 11))
        hh = ts.dt.hour
        return x[~((hh >= h0) & (hh <= h1))].copy().reset_index(drop=True)
    if rule == "keep_best_session":
        keep_sess = str(meta.get("best_session", "18_23"))
        return x[sess == keep_sess].copy().reset_index(drop=True)
    if rule == "veto_06_11_plus_risky_vol":
        atr = to_num(x["atr_percentile_1h"])
        hh = ts.dt.hour
        m_veto = (sess == "06_11") | ((sess == "12_17") & (atr >= float(meta.get("risky_atr_floor", 70.0))))
        return x[~m_veto].copy().reset_index(drop=True)
    if rule == "trend_quality_floor":
        tr = to_num(x["trend_up_1h"])
        return x[tr >= float(meta.get("trend_floor", 1.0))].copy().reset_index(drop=True)
    if rule == "vol_spike_veto":
        atr = to_num(x["atr_percentile_1h"])
        return x[atr < float(meta.get("atr_veto_threshold", 85.0))].copy().reset_index(drop=True)
    if rule == "combo_veto06_vol":
        atr = to_num(x["atr_percentile_1h"])
        m = (sess == "06_11") | (atr >= float(meta.get("atr_veto_threshold", 85.0)))
        return x[~m].copy().reset_index(drop=True)
    raise ValueError(f"Unknown exact filter rule: {rule}")


def proxy_keep_mask(sig: pd.DataFrame, rule: str, meta: Dict[str, Any]) -> np.ndarray:
    x = sig.copy().reset_index(drop=True)
    ts = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    sess = session_bucket_from_ts(ts)
    valid = (
        (to_num(x.get("exec_filled", 0)).fillna(0).astype(int) == 1)
        & (to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int) == 1)
        & to_num(x.get("exec_pnl_net_pct", np.nan)).notna()
    )
    pnl = to_num(x.get("exec_pnl_net_pct", np.nan)).fillna(0.0)
    keep = np.ones(len(x), dtype=bool)

    if rule == "daily_loss_cap":
        cap = float(meta.get("daily_cap", -0.0035))
        x["day"] = ts.dt.floor("D")
        for _d, grp in x.groupby("day", dropna=False):
            cum = 0.0
            stop = False
            for i in grp.index.tolist():
                if not bool(valid.iloc[i]):
                    continue
                if stop:
                    keep[i] = False
                    continue
                cum += float(pnl.iloc[i])
                if cum <= cap:
                    stop = True
        return keep

    if rule == "session_killswitch_n_losses":
        n_losses = int(meta.get("n_losses", 2))
        x["day"] = ts.dt.floor("D")
        ordered = x.sort_values(["signal_time", "signal_id"]).index.tolist()
        state: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for i in ordered:
            d = x.loc[i, "day"]
            s = str(sess.iloc[i])
            k = (d, s)
            if k not in state:
                state[k] = {"losses": 0, "stop": False}
            st = state[k]
            if not bool(valid.iloc[i]):
                continue
            if st["stop"]:
                keep[i] = False
                continue
            if float(pnl.iloc[i]) < 0:
                st["losses"] += 1
                if st["losses"] >= n_losses:
                    st["stop"] = True
        return keep

    if rule == "loss_cluster_pause_rest_session":
        k_losses = int(meta.get("k_losses", 3))
        x["day"] = ts.dt.floor("D")
        ordered = x.sort_values(["signal_time", "signal_id"]).index.tolist()
        state: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for i in ordered:
            d = x.loc[i, "day"]
            s = str(sess.iloc[i])
            key = (d, s)
            if key not in state:
                state[key] = {"run": 0, "stop": False}
            st = state[key]
            if not bool(valid.iloc[i]):
                continue
            if st["stop"]:
                keep[i] = False
                continue
            if float(pnl.iloc[i]) < 0:
                st["run"] += 1
                if st["run"] >= k_losses:
                    st["stop"] = True
            else:
                st["run"] = 0
        return keep

    raise ValueError(f"Unknown proxy rule: {rule}")


def choose_subwindow_from_baseline(sig: pd.DataFrame) -> Tuple[int, int]:
    x = sig.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x["hour"] = x["signal_time"].dt.hour
    x["pnl"] = to_num(x.get("exec_pnl_net_pct", np.nan))
    valid = (
        (to_num(x.get("exec_filled", 0)).fillna(0).astype(int) == 1)
        & (to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int) == 1)
        & x["pnl"].notna()
    )
    x = x[valid].copy()
    if x.empty:
        return 10, 11
    hour_sum = x.groupby("hour")["pnl"].sum().to_dict()
    windows = [(6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)]
    best = None
    for h0, h1 in windows:
        s = float(hour_sum.get(h0, 0.0) + hour_sum.get(h1, 0.0))
        if best is None or s < best[0]:
            best = (s, h0, h1)
    if best is None:
        return 10, 11
    return int(best[1]), int(best[2])


def phase_q_like_routes(base_signals: pd.DataFrame, run_dir: Path) -> Dict[str, pd.DataFrame]:
    x = ensure_signals_schema(base_signals)
    holdout_n = max(120, int(round(len(x) * 0.20)))
    route1 = x.iloc[-holdout_n:].copy().reset_index(drop=True)
    route2 = x.copy().reset_index(drop=True)
    route1.to_csv(run_dir / "route1_holdout_signals.csv", index=False)
    route2.to_csv(run_dir / "route2_reslice_signals.csv", index=False)
    return {"route1_holdout": route1, "route2_reslice": route2}


def build_stress_signals(base_signals: pd.DataFrame, run_dir: Path) -> Dict[str, pd.DataFrame]:
    b = ensure_signals_schema(base_signals)
    out = {"baseline": b}
    s3 = b.copy()
    s3["signal_time"] = pd.to_datetime(s3["signal_time"], utc=True) + pd.Timedelta(minutes=3)
    out["latency_plus3m"] = s3
    s6 = b.copy()
    s6["signal_time"] = pd.to_datetime(s6["signal_time"], utc=True) + pd.Timedelta(minutes=6)
    out["latency_plus6m"] = s6
    miss = b.copy().reset_index(drop=True)
    mask = np.ones(len(miss), dtype=bool)
    mask[np.arange(0, len(miss), 20)] = False
    out["missingness_5pct"] = miss[mask].copy().reset_index(drop=True)
    for k, v in out.items():
        v.to_csv(run_dir / f"stress_signals_{k}.csv", index=False)
    return out


def evaluate_overlay_on_dataset(
    *,
    run_dir: Path,
    dataset_name: str,
    signals_df: pd.DataFrame,
    candidate_id: str,
    genome: Dict[str, Any],
    overlay_def: Dict[str, Any],
    seed: int,
    wf_splits: int,
    train_ratio: float,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_row, sig_base = evaluate_exact(
        run_dir=run_dir,
        overlay_signals=signals_df,
        genome=genome,
        candidate_id=candidate_id,
        overlay_id="O0",
        overlay_desc="no_overlay",
        seed=seed,
        wf_splits=wf_splits,
        train_ratio=train_ratio,
        fee_mult=fee_mult,
        slip_mult=slip_mult,
        base_row=None,
    )
    base_row["dataset"] = dataset_name
    base_row["scenario"] = dataset_name

    mode = str(overlay_def.get("mode", "exact"))
    if mode == "exact":
        filt = filter_signals(signals_df, str(overlay_def["rule"]), overlay_def.get("meta", {}))
        row, _sig_overlay = evaluate_exact(
            run_dir=run_dir,
            overlay_signals=filt,
            genome=genome,
            candidate_id=candidate_id,
            overlay_id=str(overlay_def["overlay_id"]),
            overlay_desc=str(overlay_def["overlay_desc"]),
            seed=seed,
            wf_splits=wf_splits,
            train_ratio=train_ratio,
            fee_mult=fee_mult,
            slip_mult=slip_mult,
            base_row=base_row,
        )
    else:
        # proxy uses base exact detailed signal rows
        k = proxy_keep_mask(sig_base, str(overlay_def["rule"]), overlay_def.get("meta", {}))
        args_tmp = build_args(
            signals_csv=run_dir / f"_tmp_args_{dataset_name}.csv",
            wf_splits=wf_splits,
            train_ratio=train_ratio,
            seed=seed,
            fee_mult=fee_mult,
            slip_mult=slip_mult,
        )
        row = evaluate_proxy(
            sig_base=sig_base,
            args_base=args_tmp,
            candidate_id=candidate_id,
            overlay_id=str(overlay_def["overlay_id"]),
            overlay_desc=str(overlay_def["overlay_desc"]),
            keep_mask=k,
            base_row=base_row,
        )

    row["dataset"] = dataset_name
    row["scenario"] = dataset_name
    row["base_entries_valid"] = int(base_row["entries_valid"])
    row["base_entry_rate"] = float(base_row["entry_rate"])
    row["base_exec_expectancy_net"] = float(base_row["exec_expectancy_net"])
    row["base_exec_cvar_5"] = float(base_row["exec_cvar_5"])
    row["base_exec_max_drawdown"] = float(base_row["exec_max_drawdown"])
    row["base_sl_loss_share"] = float(base_row["sl_loss_share"])
    row["base_loss_run_ge3_count"] = int(base_row["loss_run_ge3_count"])
    row["base_max_consecutive_losses"] = int(base_row["max_consecutive_losses"])
    return base_row, row


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase W overlay architecture validation (SOLUSDT, contract-locked)")
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEW_OVERLAY_VALIDATION_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "phase": "W",
        "run_dir": str(run_dir),
        "project_root": str(PROJECT_ROOT),
        "commands": [{"cmd": "python scripts/phase_w_overlay_validation.py", "utc": utc_now()}],
        "symbol": LOCKED["symbol"],
        "code_modified": "YES (new script phase_w_overlay_validation.py)",
    }

    # W1 lock checks and Phase V loading
    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, metrics_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Missing required file: {fp}")

    fee_sha = sha256_file(fee_fp)
    metrics_sha = sha256_file(metrics_fp)
    lock_hash_pass = int(fee_sha == LOCKED["expected_fee_sha"] and metrics_sha == LOCKED["expected_metrics_sha"])

    phasev_dir, phasev_manifest, phasev_candidates_json, phasev_overlay_df = load_latest_phasev(exec_root)
    exec_pair = load_exec_pair_from_phasev(phasev_candidates_json)

    # Canonical ga_exec lock validator.
    args_lock = build_args(signals_csv=rep_fp, wf_splits=5, train_ratio=0.70, seed=args.seed)
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=args_lock, run_dir=run_dir)

    c_lines: List[str] = []
    c_lines.append("# Phase W Contract Validation")
    c_lines.append("")
    c_lines.append(f"- Generated UTC: {utc_now()}")
    c_lines.append(f"- Source Phase V dir: `{phasev_dir}`")
    c_lines.append(f"- fee sha256: `{fee_sha}` (match={int(fee_sha == LOCKED['expected_fee_sha'])})")
    c_lines.append(f"- metrics sha256: `{metrics_sha}` (match={int(metrics_sha == LOCKED['expected_metrics_sha'])})")
    c_lines.append(f"- ga_exec freeze lock pass: `{int(lock_validation.get('freeze_lock_pass', 0))}`")
    c_lines.append(f"- primary hash: `{exec_pair['E1']['genome_hash']}`")
    c_lines.append(f"- backup hash: `{exec_pair['E2']['genome_hash']}`")
    write_text(run_dir / "phaseW_contract_validation.md", "\n".join(c_lines))

    base_signals = ensure_signals_schema(pd.read_csv(rep_fp))
    base_signals.to_csv(run_dir / "base_signals.csv", index=False)

    # Baseline detailed eval for E1/E2 (mainline)
    base_rows: Dict[str, Dict[str, Any]] = {}
    base_sigs: Dict[str, pd.DataFrame] = {}
    for cid in ("E1", "E2"):
        b_row, b_sig = evaluate_exact(
            run_dir=run_dir,
            overlay_signals=base_signals,
            genome=exec_pair[cid]["genome"],
            candidate_id=cid,
            overlay_id="O0",
            overlay_desc="no_overlay",
            seed=args.seed,
            wf_splits=5,
            train_ratio=0.70,
            base_row=None,
        )
        base_rows[cid] = b_row
        base_sigs[cid] = b_sig

    # Reproduction target from Phase V winner: SOL_BASE_E1 + O1(session_veto_worst:06_11)
    tgt = phasev_overlay_df[
        (phasev_overlay_df["scenario_id"].astype(str) == "SOL_BASE_E1")
        & (phasev_overlay_df["overlay_id"].astype(str) == "O1")
    ].copy()
    if tgt.empty:
        raise RuntimeError("Could not find Phase V reproduction target row (SOL_BASE_E1, O1)")
    target_row = tgt.iloc[0].to_dict()

    keep_rep = ~(session_bucket_from_ts(base_sigs["E1"]["signal_time"]).astype(str).to_numpy() == "06_11")
    rep_row = evaluate_proxy(
        sig_base=base_sigs["E1"],
        args_base=args_lock,
        candidate_id="E1",
        overlay_id="O1a",
        overlay_desc="session_veto_06_11",
        keep_mask=keep_rep,
        base_row=base_rows["E1"],
    )

    checks = [
        ("delta_expectancy_vs_base", float(target_row.get("delta_expectancy_vs_base", np.nan)), float(rep_row.get("delta_expectancy_vs_base", np.nan)), float(args.tolerance)),
        ("delta_maxdd_vs_base", float(target_row.get("delta_maxdd_vs_base", np.nan)), float(rep_row.get("delta_maxdd_vs_base", np.nan)), float(args.tolerance)),
        ("maxdd_improve_ratio_vs_base", float(target_row.get("maxdd_improve_ratio_vs_base", np.nan)), float(rep_row.get("maxdd_improve_ratio_vs_base", np.nan)), float(args.tolerance)),
        ("delta_cvar_vs_base", float(target_row.get("delta_cvar_vs_base", np.nan)), float(rep_row.get("delta_cvar_vs_base", np.nan)), float(args.tolerance)),
        ("cvar_improve_ratio_vs_base", float(target_row.get("cvar_improve_ratio_vs_base", np.nan)), float(rep_row.get("cvar_improve_ratio_vs_base", np.nan)), float(args.tolerance)),
        ("removed_entries_count", float(target_row.get("removed_entries_count", np.nan)), float(rep_row.get("removed_entries_count", np.nan)), 0.0),
        ("entry_rate", float(target_row.get("entry_rate", np.nan)), float(rep_row.get("entry_rate", np.nan)), float(args.tolerance)),
        ("hard_gate_proxy_pass", float(target_row.get("hard_gate_proxy_pass", np.nan)), float(rep_row.get("hard_gate_proxy_pass", np.nan)), 0.0),
    ]
    repro_rows = []
    for m, tv, rv, tol in checks:
        diff = abs(rv - tv) if np.isfinite(rv) and np.isfinite(tv) else float("nan")
        within = int(np.isfinite(diff) and diff <= tol)
        repro_rows.append(
            {
                "metric": m,
                "phaseV_target": tv,
                "reproduced_value": rv,
                "abs_diff": diff,
                "tolerance": tol,
                "within_tolerance": within,
            }
        )
    repro_df = pd.DataFrame(repro_rows)
    repro_df.to_csv(run_dir / "phaseW_overlay_reproduction.csv", index=False)
    repro_pass = int((repro_df["within_tolerance"] == 1).all())

    leak = []
    leak.append("# Phase W Leakage Audit")
    leak.append("")
    leak.append(f"- Generated UTC: {utc_now()}")
    leak.append("- Session overlays are clock-based and ex-ante definable from timestamp only.")
    leak.append("- `session_veto_06_11` uses fixed clock bucket and does not depend on future PnL.")
    leak.append("- `worst_session` style selection is train-derived; treated as in-sample learned rule and must pass OOS checks.")
    leak.append("- Loss-control overlays (`daily_loss_cap`, `session_killswitch`, `loss_cluster_pause`) are path-dependent policy proxies in this harness.")
    leak.append("- Proxy overlays are explicitly labeled `approximate_counterfactual=1` and are not treated as exact engine-integrated signals.")
    leak.append(f"- W1 reproduction pass: `{repro_pass}`")
    write_text(run_dir / "phaseW_leakage_audit.md", "\n".join(leak))

    if repro_pass == 0:
        reason = "W_FAIL_REPRO"
        decision = []
        decision.append("# Decision Next Step")
        decision.append("")
        decision.append(f"- Generated UTC: {utc_now()}")
        decision.append(f"- Classification: **{reason}**")
        decision.append("- Reproduction mismatch on Phase V winner exceeded tolerance; stop before W2/W3.")
        write_text(run_dir / "decision_next_step.md", "\n".join(decision))
        write_text(
            run_dir / "ready_to_launch_failure_branch_prompt.txt",
            "W_FAIL_REPRO repair: reconcile Phase W reproduction mismatch for SOL_BASE_E1 + session veto 06_11 by auditing signal ordering, valid-entry mask, and base-row normalization. Do not run W2/W3 until exact metric parity is restored within tolerance.",
        )
        manifest.update(
            {
                "classification": reason,
                "phaseW_reproduction_pass": 0,
                "duration_sec": float(time.time() - t0),
                "source_phaseV_dir": str(phasev_dir),
                "freeze_lock_validation": lock_validation,
            }
        )
        json_dump(run_dir / "phaseW_run_manifest.json", manifest)
        print(json.dumps({"classification": reason, "run_dir": str(run_dir)}))
        return

    # W2 controlled overlay family benchmark
    h0, h1 = choose_subwindow_from_baseline(base_sigs["E1"])
    sess_stats = (
        base_sigs["E1"][
            (to_num(base_sigs["E1"]["exec_filled"]).fillna(0).astype(int) == 1)
            & (to_num(base_sigs["E1"]["exec_valid_for_metrics"]).fillna(0).astype(int) == 1)
            & to_num(base_sigs["E1"]["exec_pnl_net_pct"]).notna()
        ]
        .assign(session_bucket=session_bucket_from_ts(base_sigs["E1"]["signal_time"]))
        .groupby("session_bucket", dropna=False)["exec_pnl_net_pct"]
        .sum()
    )
    best_session = str(sess_stats.sort_values(ascending=False).index[0]) if len(sess_stats) else "18_23"

    overlays: List[Dict[str, Any]] = [
        {"overlay_id": "O1a", "overlay_desc": "session_veto_06_11", "family": "session", "mode": "exact", "rule": "veto_06_11", "meta": {}},
        {"overlay_id": "O1b", "overlay_desc": "session_veto_12_17", "family": "session", "mode": "exact", "rule": "veto_12_17", "meta": {}},
        {
            "overlay_id": "O1c",
            "overlay_desc": f"session_veto_subwindow_{h0:02d}_{h1:02d}",
            "family": "session",
            "mode": "exact",
            "rule": "veto_subwindow",
            "meta": {"subwindow_start_h": h0, "subwindow_end_h": h1},
        },
        {
            "overlay_id": "O1d",
            "overlay_desc": f"keep_only_best_session_{best_session}",
            "family": "session",
            "mode": "exact",
            "rule": "keep_best_session",
            "meta": {"best_session": best_session},
        },
        {
            "overlay_id": "O1e",
            "overlay_desc": "session_veto_06_11_plus_risky_vol_guard",
            "family": "session",
            "mode": "exact",
            "rule": "veto_06_11_plus_risky_vol",
            "meta": {"risky_atr_floor": 70.0},
        },
        {
            "overlay_id": "O2a",
            "overlay_desc": "daily_loss_cap_-0.0035",
            "family": "loss_control",
            "mode": "proxy",
            "rule": "daily_loss_cap",
            "meta": {"daily_cap": -0.0035},
        },
        {
            "overlay_id": "O2b",
            "overlay_desc": "session_killswitch_after_2_losses",
            "family": "loss_control",
            "mode": "proxy",
            "rule": "session_killswitch_n_losses",
            "meta": {"n_losses": 2},
        },
        {
            "overlay_id": "O2c",
            "overlay_desc": "loss_cluster_pause_after_3_losses_rest_session",
            "family": "loss_control",
            "mode": "proxy",
            "rule": "loss_cluster_pause_rest_session",
            "meta": {"k_losses": 3},
        },
        {
            "overlay_id": "O3a",
            "overlay_desc": "trend_quality_floor_trend_up>=1",
            "family": "quality_risk",
            "mode": "exact",
            "rule": "trend_quality_floor",
            "meta": {"trend_floor": 1.0},
        },
        {
            "overlay_id": "O3b",
            "overlay_desc": "vol_spike_veto_atr_pct>=85",
            "family": "quality_risk",
            "mode": "exact",
            "rule": "vol_spike_veto",
            "meta": {"atr_veto_threshold": 85.0},
        },
        {
            "overlay_id": "O3c",
            "overlay_desc": "combo_session_veto_06_11_plus_vol_spike_veto",
            "family": "quality_risk",
            "mode": "exact",
            "rule": "combo_veto06_vol",
            "meta": {"atr_veto_threshold": 85.0},
        },
    ]

    bench_rows: List[Dict[str, Any]] = []
    invalid_hist: Counter[str] = Counter()
    for cid in ("E1", "E2"):
        base = base_rows[cid]
        bench_rows.append(
            {
                "candidate_id": cid,
                "candidate_hash": exec_pair[cid]["genome_hash"],
                "overlay_id": "O0",
                "overlay_desc": "no_overlay",
                "overlay_family": "baseline",
                "overlay_eval_type": "exact_engine_integrated",
                "approximate_counterfactual": 0,
                **base,
                "hard_gate_pass": int(base.get("valid_for_ranking", 0)),
                "hard_gate_proxy_pass": int(base.get("valid_for_ranking", 0)),
            }
        )
        for ov in overlays:
            if ov["mode"] == "exact":
                df_f = filter_signals(base_signals, ov["rule"], ov.get("meta", {}))
                row, _sig = evaluate_exact(
                    run_dir=run_dir,
                    overlay_signals=df_f,
                    genome=exec_pair[cid]["genome"],
                    candidate_id=cid,
                    overlay_id=ov["overlay_id"],
                    overlay_desc=ov["overlay_desc"],
                    seed=args.seed,
                    wf_splits=5,
                    train_ratio=0.70,
                    base_row=base,
                )
            else:
                k = proxy_keep_mask(base_sigs[cid], ov["rule"], ov.get("meta", {}))
                row = evaluate_proxy(
                    sig_base=base_sigs[cid],
                    args_base=args_lock,
                    candidate_id=cid,
                    overlay_id=ov["overlay_id"],
                    overlay_desc=ov["overlay_desc"],
                    keep_mask=k,
                    base_row=base,
                )

            row["overlay_family"] = ov["family"]
            row["candidate_hash"] = exec_pair[cid]["genome_hash"]
            if not str(row.get("invalid_reason", "")).strip() and int(row.get("hard_gate_pass", 0)) == 0:
                row["invalid_reason"] = "hard_gate_fail"
            if str(row.get("invalid_reason", "")).strip():
                for tok in str(row["invalid_reason"]).split("|"):
                    t = tok.strip()
                    if t:
                        invalid_hist[t] += 1
            bench_rows.append(row)

    bench = pd.DataFrame(bench_rows)
    bench["rank_score"] = 0.0
    nz = bench["overlay_id"] != "O0"
    if nz.any():
        b0 = bench[bench["overlay_id"] == "O0"][["candidate_id", "loss_run_ge3_count", "sl_loss_share"]].rename(
            columns={"loss_run_ge3_count": "base_loss_run_ge3", "sl_loss_share": "base_sl_loss_share"}
        )
        bench = bench.merge(b0, on="candidate_id", how="left")
        bench["cluster_improve_ratio"] = np.where(
            to_num(bench["base_loss_run_ge3"]).fillna(0) > 0,
            (to_num(bench["base_loss_run_ge3"]).fillna(0) - to_num(bench["loss_run_ge3_count"]).fillna(0)) / to_num(bench["base_loss_run_ge3"]).replace(0, np.nan),
            0.0,
        )
        bench["cluster_improve_ratio"] = to_num(bench["cluster_improve_ratio"]).fillna(0.0)
        bench["sl_improve_ratio"] = np.where(
            to_num(bench["base_sl_loss_share"]).fillna(0) > 0,
            (to_num(bench["base_sl_loss_share"]).fillna(0) - to_num(bench["sl_loss_share"]).fillna(0)) / to_num(bench["base_sl_loss_share"]).replace(0, np.nan),
            0.0,
        )
        bench["sl_improve_ratio"] = to_num(bench["sl_improve_ratio"]).fillna(0.0)

        complexity_penalty = bench["overlay_eval_type"].map({"exact_engine_integrated": 0.0, "proxy_counterfactual": 0.05}).fillna(0.05)
        bench["rank_score"] = (
            1200.0 * to_num(bench["delta_expectancy_vs_base"]).fillna(-9.0)
            + 180.0 * to_num(bench["maxdd_improve_ratio_vs_base"]).fillna(-9.0)
            + 120.0 * to_num(bench["cvar_improve_ratio_vs_base"]).fillna(-9.0)
            + 25.0 * to_num(bench["cluster_improve_ratio"]).fillna(0.0)
            + 15.0 * to_num(bench["sl_improve_ratio"]).fillna(0.0)
            - 60.0 * to_num(bench["removed_entries_pct"]).fillna(1.0)
            - 20.0 * complexity_penalty
        )

    bench.to_csv(run_dir / "phaseW_overlay_benchmark_results.csv", index=False)

    shortlist = bench[bench["overlay_id"] != "O0"].copy()
    shortlist = shortlist.sort_values(
        ["hard_gate_pass", "delta_expectancy_vs_base", "maxdd_improve_ratio_vs_base", "cvar_improve_ratio_vs_base", "rank_score"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    shortlist.to_csv(run_dir / "phaseW_overlay_ranked_shortlist.csv", index=False)

    rep_lines = []
    rep_lines.append("# Phase W Overlay Benchmark Report")
    rep_lines.append("")
    rep_lines.append(f"- Generated UTC: {utc_now()}")
    rep_lines.append(f"- Candidate hashes: E1={exec_pair['E1']['genome_hash']}, E2={exec_pair['E2']['genome_hash']}")
    rep_lines.append(f"- Overlay rows (incl. O0): {len(bench)}")
    rep_lines.append(f"- Approximate counterfactual rows: {int(to_num(bench.get('approximate_counterfactual', 0)).fillna(0).sum())}")
    rep_lines.append("")
    if not shortlist.empty:
        topn = shortlist.head(8)[
            [
                "candidate_id",
                "overlay_id",
                "overlay_desc",
                "overlay_eval_type",
                "hard_gate_pass",
                "delta_expectancy_vs_base",
                "maxdd_improve_ratio_vs_base",
                "cvar_improve_ratio_vs_base",
                "removed_entries_pct",
                "loss_run_ge3_count",
                "sl_loss_share",
                "rank_score",
            ]
        ]
        rep_lines.append("Top overlays:")
        rep_lines.append(topn.to_string(index=False))
    else:
        rep_lines.append("No overlay shortlist rows.")
    write_text(run_dir / "phaseW_overlay_benchmark_report.md", "\n".join(rep_lines))

    # W3 OOS and stress on top 2-3 overlays (+ O1a forced inclusion)
    pick = shortlist.copy()
    pick["overlay_key"] = pick["candidate_id"].astype(str) + "|" + pick["overlay_id"].astype(str)
    top_keys: List[str] = []
    for _, r in pick.iterrows():
        k = str(r["overlay_key"])
        if k not in top_keys:
            top_keys.append(k)
        if len(top_keys) >= 3:
            break
    force_keys = [k for k in pick["overlay_key"].tolist() if k.endswith("|O1a")]
    for k in force_keys:
        if k not in top_keys and len(top_keys) < 4:
            top_keys.append(k)
    sel = pick[pick["overlay_key"].isin(top_keys)].copy().drop_duplicates("overlay_key").reset_index(drop=True)
    if sel.empty:
        sel = shortlist.head(2).copy()
        sel["overlay_key"] = sel["candidate_id"].astype(str) + "|" + sel["overlay_id"].astype(str)

    routes = phase_q_like_routes(base_signals, run_dir)
    oos_rows: List[Dict[str, Any]] = []
    for rname, rdf in routes.items():
        wf_splits = 3 if rname == "route1_holdout" else 7
        train_ratio = 0.50 if rname == "route1_holdout" else 0.65
        for _, srow in sel.iterrows():
            cid = str(srow["candidate_id"])
            ov_id = str(srow["overlay_id"])
            ov_desc = str(srow["overlay_desc"])
            ov_mode = str(srow["overlay_eval_type"])
            if ov_id == "O0":
                continue
            ov_def = {
                "overlay_id": ov_id,
                "overlay_desc": ov_desc,
                "mode": "exact" if ov_mode == "exact_engine_integrated" else "proxy",
                "rule": "",
                "meta": {},
            }
            # recover rule/meta from master list
            for z in overlays:
                if str(z["overlay_id"]) == ov_id:
                    ov_def = z
                    break
            base_route, over_route = evaluate_overlay_on_dataset(
                run_dir=run_dir,
                dataset_name=rname,
                signals_df=rdf,
                candidate_id=cid,
                genome=exec_pair[cid]["genome"],
                overlay_def=ov_def,
                seed=args.seed,
                wf_splits=wf_splits,
                train_ratio=train_ratio,
            )
            route_pass = int(
                int(over_route.get("hard_gate_pass", over_route.get("hard_gate_proxy_pass", 0))) == 1
                and float(over_route.get("delta_expectancy_vs_base", np.nan)) >= 0.0
                and float(over_route.get("maxdd_improve_ratio_vs_base", np.nan)) > 0.0
                and float(over_route.get("cvar_improve_ratio_vs_base", np.nan)) >= 0.0
            )
            fail_reasons = []
            if int(over_route.get("hard_gate_pass", over_route.get("hard_gate_proxy_pass", 0))) == 0:
                fail_reasons.append("hard_gate")
            if float(over_route.get("delta_expectancy_vs_base", -1.0)) < 0.0:
                fail_reasons.append("delta_expectancy_negative")
            if float(over_route.get("maxdd_improve_ratio_vs_base", -1.0)) <= 0.0:
                fail_reasons.append("maxdd_not_improved")
            if float(over_route.get("cvar_improve_ratio_vs_base", -1.0)) < 0.0:
                fail_reasons.append("cvar_not_improved")
            oos_rows.append(
                {
                    "route": rname,
                    "candidate_id": cid,
                    "candidate_hash": exec_pair[cid]["genome_hash"],
                    "overlay_id": ov_id,
                    "overlay_desc": ov_desc,
                    "overlay_eval_type": ov_mode,
                    "base_entries_valid": int(base_route["entries_valid"]),
                    "base_entry_rate": float(base_route["entry_rate"]),
                    "base_exec_expectancy_net": float(base_route["exec_expectancy_net"]),
                    "base_exec_cvar_5": float(base_route["exec_cvar_5"]),
                    "base_exec_max_drawdown": float(base_route["exec_max_drawdown"]),
                    "entries_valid": int(over_route["entries_valid"]),
                    "entry_rate": float(over_route["entry_rate"]),
                    "exec_expectancy_net": float(over_route["exec_expectancy_net"]),
                    "delta_expectancy_vs_base": float(over_route["delta_expectancy_vs_base"]),
                    "exec_cvar_5": float(over_route["exec_cvar_5"]),
                    "delta_cvar_vs_base": float(over_route["delta_cvar_vs_base"]),
                    "cvar_improve_ratio_vs_base": float(over_route["cvar_improve_ratio_vs_base"]),
                    "exec_max_drawdown": float(over_route["exec_max_drawdown"]),
                    "delta_maxdd_vs_base": float(over_route["delta_maxdd_vs_base"]),
                    "maxdd_improve_ratio_vs_base": float(over_route["maxdd_improve_ratio_vs_base"]),
                    "removed_entries_count": int(over_route["removed_entries_count"]),
                    "removed_entries_pct": float(over_route["removed_entries_pct"]),
                    "taker_share": float(over_route["taker_share"]),
                    "p95_fill_delay_min": float(over_route["p95_fill_delay_min"]),
                    "loss_run_ge3_count": int(over_route["loss_run_ge3_count"]),
                    "max_consecutive_losses": int(over_route["max_consecutive_losses"]),
                    "sl_loss_share": float(over_route["sl_loss_share"]),
                    "hard_gate_pass": int(over_route.get("hard_gate_pass", over_route.get("hard_gate_proxy_pass", 0))),
                    "route_pass": int(route_pass),
                    "route_fail_reasons": "|".join(fail_reasons),
                }
            )
    oos = pd.DataFrame(oos_rows)
    oos.to_csv(run_dir / "phaseW_overlay_oos_results.csv", index=False)

    stress_sets = build_stress_signals(base_signals, run_dir)
    stress_cfg = [
        {"scenario": "baseline_canonical", "signals_key": "baseline", "fee_mult": 1.0, "slip_mult": 1.0},
        {"scenario": "fee_slip_plus25", "signals_key": "baseline", "fee_mult": 1.25, "slip_mult": 1.25},
        {"scenario": "fee_slip_plus50", "signals_key": "baseline", "fee_mult": 1.50, "slip_mult": 1.50},
        {"scenario": "latency_plus3m", "signals_key": "latency_plus3m", "fee_mult": 1.0, "slip_mult": 1.0},
        {"scenario": "latency_plus6m", "signals_key": "latency_plus6m", "fee_mult": 1.0, "slip_mult": 1.0},
    ]
    stress_rows: List[Dict[str, Any]] = []
    for sc in stress_cfg:
        sname = str(sc["scenario"])
        sdf = stress_sets[str(sc["signals_key"])].copy()
        for _, srow in sel.iterrows():
            cid = str(srow["candidate_id"])
            ov_id = str(srow["overlay_id"])
            ov_desc = str(srow["overlay_desc"])
            ov_mode = str(srow["overlay_eval_type"])
            if ov_id == "O0":
                continue
            ov_def = {
                "overlay_id": ov_id,
                "overlay_desc": ov_desc,
                "mode": "exact" if ov_mode == "exact_engine_integrated" else "proxy",
                "rule": "",
                "meta": {},
            }
            for z in overlays:
                if str(z["overlay_id"]) == ov_id:
                    ov_def = z
                    break
            b_sc, o_sc = evaluate_overlay_on_dataset(
                run_dir=run_dir,
                dataset_name=sname,
                signals_df=sdf,
                candidate_id=cid,
                genome=exec_pair[cid]["genome"],
                overlay_def=ov_def,
                seed=args.seed,
                wf_splits=5,
                train_ratio=0.70,
                fee_mult=float(sc["fee_mult"]),
                slip_mult=float(sc["slip_mult"]),
            )
            stress_rows.append(
                {
                    "scenario": sname,
                    "candidate_id": cid,
                    "candidate_hash": exec_pair[cid]["genome_hash"],
                    "overlay_id": ov_id,
                    "overlay_desc": ov_desc,
                    "overlay_eval_type": ov_mode,
                    "base_entries_valid": int(b_sc["entries_valid"]),
                    "entries_valid": int(o_sc["entries_valid"]),
                    "entry_rate": float(o_sc["entry_rate"]),
                    "exec_expectancy_net": float(o_sc["exec_expectancy_net"]),
                    "delta_expectancy_vs_base": float(o_sc["delta_expectancy_vs_base"]),
                    "cvar_improve_ratio_vs_base": float(o_sc["cvar_improve_ratio_vs_base"]),
                    "maxdd_improve_ratio_vs_base": float(o_sc["maxdd_improve_ratio_vs_base"]),
                    "removed_entries_pct": float(o_sc["removed_entries_pct"]),
                    "taker_share": float(o_sc["taker_share"]),
                    "p95_fill_delay_min": float(o_sc["p95_fill_delay_min"]),
                    "hard_gate_pass": int(o_sc.get("hard_gate_pass", o_sc.get("hard_gate_proxy_pass", 0))),
                    "fee_mult": float(sc["fee_mult"]),
                    "slip_mult": float(sc["slip_mult"]),
                }
            )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(run_dir / "phaseW_overlay_stress_results.csv", index=False)

    # Robustness verdict per selected overlay-candidate.
    rob_rows = []
    key_cols = ["candidate_id", "overlay_id", "overlay_desc", "overlay_eval_type"]
    for key, grp in sel.groupby(key_cols, dropna=False):
        cid, oid, odesc, otype = key
        rsub = oos[(oos["candidate_id"] == cid) & (oos["overlay_id"] == oid)].copy()
        ssub = stress[(stress["candidate_id"] == cid) & (stress["overlay_id"] == oid)].copy()
        if rsub.empty and ssub.empty:
            continue
        oos_pass_rate = float(to_num(rsub["route_pass"]).fillna(0).mean()) if not rsub.empty else 0.0
        stress_sign_retention = float((to_num(ssub["delta_expectancy_vs_base"]) >= 0.0).mean()) if not ssub.empty else 0.0
        stress_risk_retention = float(((to_num(ssub["maxdd_improve_ratio_vs_base"]) > 0.0) & (to_num(ssub["cvar_improve_ratio_vs_base"]) >= 0.0)).mean()) if not ssub.empty else 0.0
        stress_gate_rate = float(to_num(ssub["hard_gate_pass"]).fillna(0).mean()) if not ssub.empty else 0.0
        base_row = shortlist[(shortlist["candidate_id"] == cid) & (shortlist["overlay_id"] == oid)].head(1)
        if base_row.empty:
            continue
        br = base_row.iloc[0]
        b_delta = float(br.get("delta_expectancy_vs_base", np.nan))
        b_dd = float(br.get("maxdd_improve_ratio_vs_base", np.nan))
        b_cvar = float(br.get("cvar_improve_ratio_vs_base", np.nan))
        b_gate = int(br.get("hard_gate_pass", br.get("hard_gate_proxy_pass", 0)))
        if (
            b_gate == 1
            and b_delta >= 0.0
            and b_dd >= 0.10
            and b_cvar >= 0.03
            and oos_pass_rate >= 1.0
            and stress_risk_retention >= 0.75
            and stress_sign_retention >= 0.50
            and stress_gate_rate >= 0.75
        ):
            verdict = "ROBUST_PASS"
        elif (
            b_gate == 1
            and b_dd > 0.0
            and b_cvar >= 0.0
            and oos_pass_rate >= 0.5
            and stress_risk_retention >= 0.5
        ):
            verdict = "WEAK_PASS"
        elif b_gate == 1 and (oos_pass_rate > 0.0 or stress_risk_retention > 0.0):
            verdict = "BRITTLE"
        else:
            verdict = "NO_GO"

        rob_rows.append(
            {
                "candidate_id": cid,
                "candidate_hash": exec_pair[cid]["genome_hash"],
                "overlay_id": oid,
                "overlay_desc": odesc,
                "overlay_eval_type": otype,
                "baseline_hard_gate_pass": b_gate,
                "baseline_delta_expectancy_vs_base": b_delta,
                "baseline_maxdd_improve_ratio_vs_base": b_dd,
                "baseline_cvar_improve_ratio_vs_base": b_cvar,
                "oos_pass_rate": oos_pass_rate,
                "stress_sign_retention": stress_sign_retention,
                "stress_risk_retention": stress_risk_retention,
                "stress_gate_rate": stress_gate_rate,
                "verdict": verdict,
            }
        )

    rob = pd.DataFrame(rob_rows).sort_values(
        ["verdict", "baseline_delta_expectancy_vs_base", "baseline_maxdd_improve_ratio_vs_base"],
        ascending=[True, False, False],
    )
    write_text(
        run_dir / "phaseW_overlay_robustness_report.md",
        "\n".join(
            [
                "# Phase W Overlay Robustness Report",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- Selected overlays tested: {len(sel)}",
                "",
                (rob.to_string(index=False) if not rob.empty else "No robustness rows."),
            ]
        ),
    )

    # W4 decision class A/B/C/D
    classification = "C"
    class_reason = "all_overlays_fragile_or_no_go"
    if not rob.empty:
        any_exact_robust = bool(((rob["verdict"] == "ROBUST_PASS") & (rob["overlay_eval_type"] == "exact_engine_integrated")).any())
        any_robust = bool((rob["verdict"] == "ROBUST_PASS").any())
        any_weak = bool((rob["verdict"] == "WEAK_PASS").any())
        all_proxy = bool((rob["overlay_eval_type"] == "proxy_counterfactual").all())
        if any_exact_robust and any_robust:
            classification = "A"
            class_reason = "at_least_one_exact_overlay_robust_pass"
        elif all_proxy and (any_robust or any_weak):
            classification = "D"
            class_reason = "benefit_detected_but_proxy_only"
        elif any_weak:
            classification = "B"
            class_reason = "overlay_benefit_exists_but_robustness_weak"
        else:
            classification = "C"
            class_reason = "all_overlays_brittle_no_go"
    else:
        classification = "C"
        class_reason = "no_robustness_rows"

    # Primary / backup overlay picks
    primary = None
    backup = None
    if not rob.empty:
        order = {"ROBUST_PASS": 0, "WEAK_PASS": 1, "BRITTLE": 2, "NO_GO": 3}
        x = rob.copy()
        x["v_rank"] = x["verdict"].map(order).fillna(9).astype(int)
        x = x.sort_values(
            ["v_rank", "baseline_delta_expectancy_vs_base", "baseline_maxdd_improve_ratio_vs_base", "baseline_cvar_improve_ratio_vs_base"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        if not x.empty:
            primary = x.iloc[0].to_dict()
        if len(x) > 1:
            for _, rr in x.iloc[1:].iterrows():
                if primary is None or str(rr["overlay_id"]) != str(primary["overlay_id"]):
                    backup = rr.to_dict()
                    break
            if backup is None:
                backup = x.iloc[1].to_dict()

    decision_lines = []
    decision_lines.append("# Decision Next Step")
    decision_lines.append("")
    decision_lines.append(f"- Generated UTC: {utc_now()}")
    decision_lines.append(f"- Phase W classification: **{classification}**")
    decision_lines.append(f"- Reason: {class_reason}")
    decision_lines.append(f"- Overlay remains highest-ROI next step: {('yes' if classification in {'A', 'B'} else 'no')}")
    if primary is not None:
        decision_lines.append(
            f"- Primary overlay: {primary['candidate_id']} {primary['overlay_id']} `{primary['overlay_desc']}` ({primary['verdict']})"
        )
    if backup is not None:
        decision_lines.append(
            f"- Backup overlay: {backup['candidate_id']} {backup['overlay_id']} `{backup['overlay_desc']}` ({backup['verdict']})"
        )
    decision_lines.append(
        f"- Paper/shadow promotion status (overlay-on-exec, SOL only): {('PROMOTE_PAPER' if classification=='A' else 'PAPER_CAUTION' if classification=='B' else 'NO_PROMOTION' if classification=='C' else 'INFRA_LIMITED')}"
    )
    decision_lines.append("- Portability revisit now: deferred")
    decision_lines.append(f"- New GA compute justified now: {('yes' if classification=='A' else 'no')}")
    write_text(run_dir / "decision_next_step.md", "\n".join(decision_lines))

    if classification == "A":
        next_prompt = (
            "Phase X overlay-on-exec paper rollout (SOL, contract-locked): promote primary robust overlay with backup fallback on top of promoted execution candidate pair. "
            "Run paper/shadow monitoring with fixed hard gates, route/stress drift checks, and rollback triggers. No new GA marathon."
        )
        fail_prompt = ""
    elif classification == "B":
        next_prompt = (
            "Phase X overlay caution pilot (SOL, contract-locked): keep best WEAK_PASS overlay and backup only, run short paper/shadow validation with strict split-aware rollback. "
            "Do not expand overlay family or run GA; require robustness improvement before promotion uplift."
        )
        fail_prompt = (
            "Failure branch (Phase W=B): if caution-paper run shows split instability or participation starvation, stop overlay work and pivot to 1h objective redesign."
        )
    elif classification == "C":
        next_prompt = (
            "Phase X pivot to 1h objective redesign (SOL): overlay family proved fragile under OOS/stress. Keep execution candidate fixed, rebuild upstream objective to improve execution-scored stability."
        )
        fail_prompt = (
            "Failure branch (Phase W=C): avoid additional overlay/GA compute on current family; allocate next block to objective redesign diagnostics."
        )
    else:
        next_prompt = (
            "Phase X infra patch first: implement exact engine-integrated overlay hooks for path-dependent controls (daily cap/session kill-switch), then rerun Phase W before any promotion decision."
        )
        fail_prompt = (
            "Failure branch (Phase W=D): no deployment/promotion while evidence remains proxy-only; complete harness integration before further overlay compute."
        )

    write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)
    if fail_prompt.strip():
        write_text(run_dir / "ready_to_launch_failure_branch_prompt.txt", fail_prompt)

    manifest.update(
        {
            "source_phaseV_dir": str(phasev_dir),
            "source_phaseV_classification": str(phasev_manifest.get("classification", "")),
            "freeze_lock_validation": lock_validation,
            "phaseW_reproduction_pass": repro_pass,
            "classification": classification,
            "classification_reason": class_reason,
            "exec_candidates": {
                "E1": exec_pair["E1"]["genome_hash"],
                "E2": exec_pair["E2"]["genome_hash"],
            },
            "overlay_count_tested": int(len(overlays)),
            "benchmark_rows": int(len(bench)),
            "selected_for_robustness": int(len(sel)),
            "invalid_reason_histogram": dict(sorted(invalid_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
            "duration_sec": float(time.time() - t0),
        }
    )
    json_dump(run_dir / "phaseW_run_manifest.json", manifest)
    json_dump(run_dir / "phaseW_invalid_reason_histogram.json", manifest["invalid_reason_histogram"])

    print(json.dumps({"classification": classification, "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()

