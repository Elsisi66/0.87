#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import exit_sweep  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _json_dump(path: Path, obj: Any) -> None:
    def _default(x: Any) -> Any:
        if isinstance(x, pd.DataFrame):
            return {
                "_type": "DataFrame",
                "shape": [int(x.shape[0]), int(x.shape[1])],
                "columns": [str(c) for c in list(x.columns)],
            }
        if isinstance(x, pd.Series):
            return {
                "_type": "Series",
                "shape": [int(x.shape[0])],
                "name": str(x.name),
            }
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        return str(x)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def _sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cfg_key(cfg: exit_sweep.ExitCfg) -> Tuple[Any, ...]:
    return (
        float(cfg.tp_mult),
        float(cfg.sl_mult),
        int(cfg.time_stop_min),
        int(cfg.break_even_enabled),
        float(cfg.break_even_trigger_r),
        float(cfg.break_even_offset_bps),
        int(cfg.partial_take_enabled),
        float(cfg.partial_take_r),
        float(cfg.partial_take_pct),
    )


def _cfg_hash(cfg: exit_sweep.ExitCfg) -> str:
    return _sha256_text(json.dumps(asdict(cfg), sort_keys=True))[:24]


def _stratum_key(cfg: exit_sweep.ExitCfg) -> Tuple[Any, ...]:
    return (
        float(cfg.tp_mult),
        float(cfg.sl_mult),
        int(cfg.time_stop_min),
        int(cfg.break_even_enabled),
        int(cfg.partial_take_enabled),
    )


def _pick_deterministic(items: List[Any], k: int, seed: int) -> List[Any]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    xs = list(items)
    if not xs:
        return []
    rot = int(seed) % len(xs)
    xs = xs[rot:] + xs[:rot]
    idx = np.linspace(0, len(xs) - 1, num=k, dtype=int).tolist()
    out: List[Any] = []
    used: set[int] = set()
    for i in idx:
        ii = int(i)
        if ii not in used:
            used.add(ii)
            out.append(xs[ii])
    j = 0
    while len(out) < k and j < len(xs):
        if j not in used:
            used.add(j)
            out.append(xs[j])
        j += 1
    return out[:k]


def _sample_configs_stratified(
    cfgs: List[exit_sweep.ExitCfg],
    cap: int,
    seed: int,
    label: str,
) -> Tuple[List[exit_sweep.ExitCfg], Dict[str, Any]]:
    if cap <= 0 or len(cfgs) <= cap:
        counts = {}
        for c in cfgs:
            k = str(_stratum_key(c))
            counts[k] = int(counts.get(k, 0) + 1)
        meta = {
            "label": str(label),
            "method": "stratified_deterministic_no_cap",
            "sampling_seed": int(seed),
            "cap": int(cap),
            "full_config_count": int(len(cfgs)),
            "sampled_config_count": int(len(cfgs)),
            "stratum_count_before": int(len(counts)),
            "stratum_count_after": int(len(counts)),
            "stratum_counts_before": counts,
            "stratum_counts_after": counts,
        }
        return list(cfgs), meta

    groups: Dict[Tuple[Any, ...], List[exit_sweep.ExitCfg]] = {}
    for c in cfgs:
        groups.setdefault(_stratum_key(c), []).append(c)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=_cfg_key)
    keys = sorted(groups.keys())

    before_counts = {str(k): int(len(v)) for k, v in groups.items()}
    selected_counts: Dict[Tuple[Any, ...], int] = {k: 0 for k in keys}

    if cap < len(keys):
        pick_keys = _pick_deterministic(list(keys), cap, seed)
        selected_counts = {k: 0 for k in keys}
        for k in pick_keys:
            selected_counts[k] = 1
    else:
        for k in keys:
            selected_counts[k] = 1
        remain = int(cap - len(keys))
        residual = {k: max(0, len(groups[k]) - selected_counts[k]) for k in keys}
        total_res = int(sum(residual.values()))
        if remain > 0 and total_res > 0:
            quotas: Dict[Tuple[Any, ...], int] = {}
            fracs: List[Tuple[float, Tuple[Any, ...]]] = []
            used = 0
            for k in keys:
                raw = float(remain * residual[k] / total_res) if total_res > 0 else 0.0
                q = int(math.floor(raw))
                q = max(0, min(q, residual[k]))
                quotas[k] = q
                used += q
                fracs.append((raw - q, k))
            left = int(remain - used)
            for _, k in sorted(fracs, key=lambda x: (-x[0], x[1])):
                if left <= 0:
                    break
                if quotas[k] < residual[k]:
                    quotas[k] += 1
                    left -= 1
            for k, q in quotas.items():
                selected_counts[k] = int(min(len(groups[k]), selected_counts[k] + q))

    out: List[exit_sweep.ExitCfg] = []
    for k in keys:
        q = int(max(0, min(selected_counts[k], len(groups[k]))))
        if q <= 0:
            continue
        salt = abs(hash((label, k, seed))) % 1000003
        out.extend(_pick_deterministic(groups[k], q, seed=salt))

    if len(out) > cap:
        out = sorted(out, key=_cfg_key)
        out = _pick_deterministic(out, cap, seed=seed)
    out = sorted(out, key=_cfg_key)
    after_counts: Dict[str, int] = {}
    for c in out:
        kk = str(_stratum_key(c))
        after_counts[kk] = int(after_counts.get(kk, 0) + 1)

    meta = {
        "label": str(label),
        "method": "stratified_deterministic_proportional_within_strata",
        "sampling_seed": int(seed),
        "cap": int(cap),
        "full_config_count": int(len(cfgs)),
        "sampled_config_count": int(len(out)),
        "stratum_count_before": int(len(before_counts)),
        "stratum_count_after": int(len(after_counts)),
        "stratum_counts_before": before_counts,
        "stratum_counts_after": after_counts,
        "stratify_fields": ["tp_mult", "sl_mult", "time_stop_min", "break_even_enabled", "partial_take_enabled"],
    }
    return out, meta


def _build_full_grid(args: argparse.Namespace) -> List[exit_sweep.ExitCfg]:
    tp = exit_sweep._parse_floats(args.tp_mults)
    sl = exit_sweep._parse_floats(args.sl_mults)
    ts = exit_sweep._parse_ints(args.time_stops_min)
    be = exit_sweep._parse_bools(args.break_even_enabled)
    be_trig = exit_sweep._parse_floats(args.break_even_trigger_r)
    be_off = exit_sweep._parse_floats(args.break_even_offset_bps)
    pt_en = exit_sweep._parse_bools(args.partial_take_enabled)
    pt_r = exit_sweep._parse_floats(args.partial_take_r)
    pt_pct = exit_sweep._parse_floats(args.partial_take_pct)

    out: List[exit_sweep.ExitCfg] = []
    for x in itertools.product(tp, sl, ts, be, be_trig, be_off, pt_en, pt_r, pt_pct):
        cfg = exit_sweep.ExitCfg(
            tp_mult=float(x[0]),
            sl_mult=float(x[1]),
            time_stop_min=int(x[2]),
            break_even_enabled=int(x[3]),
            break_even_trigger_r=float(x[4]),
            break_even_offset_bps=float(x[5]),
            partial_take_enabled=int(x[6]),
            partial_take_r=float(x[7]),
            partial_take_pct=float(x[8]),
        )
        if cfg.partial_take_enabled == 1 and cfg.partial_take_r >= cfg.tp_mult:
            continue
        out.append(cfg)
    return sorted(out, key=_cfg_key)


def _stable_sort_for_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = [
        "pass_all",
        "pass_stability",
        "delta_expectancy_best_exit_minus_baseline_exit",
        "delta_maxdd_best_exit_minus_baseline_exit",
        "delta_cvar5_best_exit_minus_baseline_exit",
    ]
    return (
        df.sort_values(cols, ascending=[False, False, False, False, False], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )


def _build_eval_args(args: argparse.Namespace, fee_model: Dict[str, Any]) -> argparse.Namespace:
    fee_src = fee_model
    if not all(k in fee_src for k in ["fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market"]):
        for k in ["tight_pipeline_fee_model", "ga_pipeline_fee_model"]:
            cand = fee_model.get(k)
            if isinstance(cand, dict) and all(x in cand for x in ["fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market"]):
                fee_src = cand
                break
    ev = ga_exec.build_arg_parser().parse_args([])
    ev.symbol = "SOLUSDT"
    ev.symbols = "SOLUSDT"
    ev.rank = int(args.rank)
    ev.signals_dir = str(args.signals_dir)
    ev.signals_csv = str(args.signals_csv)
    ev.signal_order = str(args.signal_order)
    ev.max_signals = int(args.max_signals)
    ev.train_ratio = float(args.train_ratio)
    ev.wf_splits = int(args.wf_splits)
    ev.mode = str(args.mode)
    ev.force_no_skip = 1
    ev.timeframe = str(args.timeframe)
    ev.pre_buffer_hours = float(args.pre_buffer_hours)
    ev.exec_horizon_hours = float(args.exec_horizon_hours)
    ev.cache_dir = str(args.cache_dir)
    ev.max_fetch_retries = int(args.max_fetch_retries)
    ev.retry_base_sleep = float(args.retry_base_sleep)
    ev.retry_max_sleep = float(args.retry_max_sleep)
    ev.fetch_pause_sec = float(args.fetch_pause_sec)
    ev.execution_config = str(args.execution_config)
    ev.fee_bps_maker = float(fee_src["fee_bps_maker"])
    ev.fee_bps_taker = float(fee_src["fee_bps_taker"])
    ev.slippage_bps_limit = float(fee_src["slippage_bps_limit"])
    ev.slippage_bps_market = float(fee_src["slippage_bps_market"])
    ev.workers = int(args.workers)

    # Exit sweep is entry-fixed baseline; relax entry realism anti-cheat gates.
    ev.hard_min_trades_overall = 0
    ev.hard_min_trade_frac_overall = 0.0
    ev.hard_min_trades_symbol = 0
    ev.hard_min_trade_frac_symbol = 0.0
    ev.hard_min_entry_rate_symbol = 0.0
    ev.hard_min_entry_rate_overall = 0.0
    ev.hard_max_taker_share = 1.0
    ev.hard_max_median_fill_delay_min = 1e9
    ev.hard_max_p95_fill_delay_min = 1e9
    ev.hard_max_missing_slice_rate = float(args.max_missing_slice_rate)
    return ev


def _signal_subset_hash(df: pd.DataFrame) -> str:
    x = df.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.sort_values("signal_time").reset_index(drop=True)
    lines: List[str] = []
    for _, r in x.iterrows():
        t = pd.to_datetime(r["signal_time"], utc=True, errors="coerce")
        lines.append(f"{r.get('signal_id','')}|{t.isoformat() if pd.notna(t) else ''}")
    return _sha256_text("\n".join(lines))


def _freeze_splits(signals_df: pd.DataFrame, train_ratio: float, wf_splits: int) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    splits = ga_exec._build_walkforward_splits(n=len(signals_df), train_ratio=float(train_ratio), n_splits=int(wf_splits))
    rows = signals_df.sort_values("signal_time").reset_index(drop=True)
    details: List[Dict[str, Any]] = []
    for sp in splits:
        i0 = int(sp["test_start"])
        i1 = int(sp["test_end"])
        d: Dict[str, Any] = {k: int(v) for k, v in sp.items()}
        d["test_count"] = int(max(0, i1 - i0))
        if i0 < len(rows):
            d["test_start_signal_id"] = str(rows.iloc[i0].get("signal_id", ""))
            d["test_start_time"] = pd.to_datetime(rows.iloc[i0]["signal_time"], utc=True).isoformat()
        else:
            d["test_start_signal_id"] = ""
            d["test_start_time"] = ""
        if i1 - 1 < len(rows) and i1 - 1 >= 0:
            d["test_end_signal_id"] = str(rows.iloc[i1 - 1].get("signal_id", ""))
            d["test_end_time"] = pd.to_datetime(rows.iloc[i1 - 1]["signal_time"], utc=True).isoformat()
        else:
            d["test_end_signal_id"] = ""
            d["test_end_time"] = ""
        details.append(d)
    payload = {
        "generated_utc": _utc_now(),
        "n_signals": int(len(rows)),
        "train_ratio": float(train_ratio),
        "wf_splits": int(wf_splits),
        "splits": details,
    }
    return [{k: int(v) for k, v in sp.items()} for sp in splits], payload


def _build_bundle(
    *,
    symbol: str,
    signals_df: pd.DataFrame,
    signal_csv: Path,
    eval_args: argparse.Namespace,
    split_def: List[Dict[str, int]],
) -> ga_exec.SymbolBundle:
    all_cfg = ga_exec._load_execution_config(_resolve(eval_args.execution_config))
    s_cfg = ga_exec._symbol_exec_config(all_cfg, symbol)
    cons = s_cfg.get("tight_constraints") if str(eval_args.mode).lower() == "tight" else s_cfg.get("constraints")
    if not isinstance(cons, dict):
        cons = {}
    constraints = {
        "min_entry_rate": float(
            cons.get(
                "min_entry_rate",
                eval_args.tight_min_entry_rate_default if str(eval_args.mode).lower() == "tight" else eval_args.min_entry_rate_default,
            )
        ),
        "max_taker_share": float(
            cons.get(
                "max_taker_share",
                eval_args.tight_max_taker_share_default if str(eval_args.mode).lower() == "tight" else eval_args.max_taker_share_default,
            )
        ),
        "max_fill_delay_min": float(
            cons.get(
                "max_fill_delay_min",
                eval_args.tight_max_fill_delay_default if str(eval_args.mode).lower() == "tight" else eval_args.max_fill_delay_default,
            )
        ),
        "min_median_entry_improvement_bps": float(cons.get("min_median_entry_improvement_bps", 0.0)),
    }
    bundle = ga_exec._build_bundle_for_symbol(
        symbol=symbol,
        signals_df=signals_df,
        signal_csv=signal_csv,
        constraints=constraints,
        args=eval_args,
    )
    bundle.splits = [dict(s) for s in split_def]
    return bundle


def _split_baseline_stats(ref_eval: Dict[str, Any]) -> Tuple[float, float, float]:
    sdf = ref_eval.get("split_rows_df")
    if not isinstance(sdf, pd.DataFrame) or sdf.empty:
        return float("nan"), float("nan"), float("nan")
    base = pd.to_numeric(sdf.get("baseline_mean_expectancy_net", np.nan), errors="coerce")
    return (
        float(base.min()) if base.notna().any() else float("nan"),
        float(base.median()) if base.notna().any() else float("nan"),
        float(base.std(ddof=0)) if base.notna().any() else float("nan"),
    )


def _evaluate_cfgs(
    *,
    cfgs: List[exit_sweep.ExitCfg],
    bundles: List[ga_exec.SymbolBundle],
    eval_args: argparse.Namespace,
    base_split_stats: Optional[Tuple[float, float, float]],
    expectancy_epsilon: float,
    maxdd_worse_tol: float,
    cvar_worse_tol: float,
    max_missing_slice_rate: float,
) -> Tuple[pd.DataFrame, Tuple[float, float, float], Dict[str, Any]]:
    if not cfgs:
        return pd.DataFrame(), (float("nan"), float("nan"), float("nan")), {}

    genomes = [exit_sweep._cfg_to_genome(c, mode=str(eval_args.mode)) for c in cfgs]
    ref_eval: Dict[str, Any] = {}
    if base_split_stats is None:
        ref_eval = ga_exec._evaluate_genome(genome=genomes[0], bundles=bundles, args=eval_args, detailed=True)
        base_split_stats = _split_baseline_stats(ref_eval)

    eval_cache: Dict[str, Dict[str, Any]] = {}
    records, pop_log = ga_exec._evaluate_population(
        population=genomes,
        eval_cache=eval_cache,
        bundles=bundles,
        args=eval_args,
    )

    rows: List[Dict[str, Any]] = []
    for cfg, rec in zip(cfgs, records):
        m = rec.get("metrics", {})
        d_exp = float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan))
        d_cvar = float(m.get("overall_exec_cvar_5", np.nan) - m.get("overall_baseline_cvar_5", np.nan))
        d_maxdd = float(m.get("overall_exec_max_drawdown", np.nan) - m.get("overall_baseline_max_drawdown", np.nan))
        s_min, s_med, s_std, s_all = exit_sweep._stability_flags(
            exec_min=float(m.get("min_split_expectancy_net", np.nan)),
            exec_med=float(m.get("median_split_expectancy_net", np.nan)),
            exec_std=float(m.get("std_split_expectancy_net", np.nan)),
            base_min=float(base_split_stats[0]),
            base_med=float(base_split_stats[1]),
            base_std=float(base_split_stats[2]),
        )
        p_exp = int(np.isfinite(d_exp) and d_exp >= float(expectancy_epsilon))
        p_cvar = int(np.isfinite(d_cvar) and d_cvar >= -float(cvar_worse_tol))
        p_maxdd = int(np.isfinite(d_maxdd) and d_maxdd >= -float(maxdd_worse_tol))
        p_data = int(
            np.isfinite(float(m.get("overall_missing_slice_rate", np.nan)))
            and float(m.get("overall_missing_slice_rate", np.nan)) <= float(max_missing_slice_rate)
        )
        p_part = int(np.isfinite(float(m.get("overall_entry_rate", np.nan))) and float(m.get("overall_entry_rate", np.nan)) > 0.0)
        p_all = int(p_exp and p_cvar and p_maxdd and s_all and p_data and p_part)

        rows.append(
            {
                "cfg_hash": _cfg_hash(cfg),
                "tp_mult": float(cfg.tp_mult),
                "sl_mult": float(cfg.sl_mult),
                "time_stop_min": int(cfg.time_stop_min),
                "break_even_enabled": int(cfg.break_even_enabled),
                "break_even_trigger_r": float(cfg.break_even_trigger_r),
                "break_even_offset_bps": float(cfg.break_even_offset_bps),
                "partial_take_enabled": int(cfg.partial_take_enabled),
                "partial_take_r": float(cfg.partial_take_r),
                "partial_take_pct": float(cfg.partial_take_pct),
                "signals_total": int(m.get("overall_signals_total", 0)),
                "trades_total": int(m.get("overall_entries_valid", 0)),
                "baseline_expectancy_net": float(m.get("overall_baseline_expectancy_net", np.nan)),
                "best_exit_expectancy_net": float(m.get("overall_exec_expectancy_net", np.nan)),
                "delta_expectancy_best_exit_minus_baseline_exit": float(d_exp),
                "baseline_cvar_5": float(m.get("overall_baseline_cvar_5", np.nan)),
                "best_exit_cvar_5": float(m.get("overall_exec_cvar_5", np.nan)),
                "delta_cvar5_best_exit_minus_baseline_exit": float(d_cvar),
                "baseline_max_drawdown": float(m.get("overall_baseline_max_drawdown", np.nan)),
                "best_exit_max_drawdown": float(m.get("overall_exec_max_drawdown", np.nan)),
                "delta_maxdd_best_exit_minus_baseline_exit": float(d_maxdd),
                "entry_rate": float(m.get("overall_entry_rate", np.nan)),
                "taker_share": float(m.get("overall_exec_taker_share", np.nan)),
                "median_fill_delay_min": float(m.get("overall_exec_median_fill_delay_min", np.nan)),
                "p95_fill_delay_min": float(m.get("overall_exec_p95_fill_delay_min", np.nan)),
                "median_entry_improvement_bps": float(m.get("overall_exec_median_entry_improvement_bps", np.nan)),
                "missing_slice_rate": float(m.get("overall_missing_slice_rate", np.nan)),
                "split_min_expectancy_best_exit": float(m.get("min_split_expectancy_net", np.nan)),
                "split_median_expectancy_best_exit": float(m.get("median_split_expectancy_net", np.nan)),
                "split_std_expectancy_best_exit": float(m.get("std_split_expectancy_net", np.nan)),
                "split_min_expectancy_baseline_exit": float(base_split_stats[0]),
                "split_median_expectancy_baseline_exit": float(base_split_stats[1]),
                "split_std_expectancy_baseline_exit": float(base_split_stats[2]),
                "pass_expectancy": int(p_exp),
                "pass_cvar_not_worse": int(p_cvar),
                "pass_maxdd_not_worse": int(p_maxdd),
                "pass_stability_min": int(s_min),
                "pass_stability_median": int(s_med),
                "pass_stability_std": int(s_std),
                "pass_stability": int(s_all),
                "pass_data_quality": int(p_data),
                "pass_participation": int(p_part),
                "pass_all": int(p_all),
                "invalid_reason": str(m.get("invalid_reason", "")),
            }
        )

    df = pd.DataFrame(rows)
    df = _stable_sort_for_shortlist(df)
    return df, base_split_stats, {"population_eval": pop_log, "reference_eval": ref_eval}


def _fail_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["reason", "count"])
    out: Dict[str, int] = {}
    for c in [
        "pass_expectancy",
        "pass_cvar_not_worse",
        "pass_maxdd_not_worse",
        "pass_stability",
        "pass_data_quality",
        "pass_participation",
        "pass_all",
    ]:
        if c in df.columns:
            out[f"fail_{c}"] = int((pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) == 0).sum())
    reason_counts: Dict[str, int] = {}
    for blob in df.get("invalid_reason", pd.Series(dtype=str)).fillna("").astype(str):
        for p in [x.strip() for x in blob.split("|") if x.strip()]:
            reason_counts[p] = int(reason_counts.get(p, 0) + 1)
    for k, v in sorted(reason_counts.items()):
        out[f"invalid_{k}"] = int(v)
    o = pd.DataFrame([{"reason": k, "count": int(v)} for k, v in out.items()])
    return o.sort_values(["count", "reason"], ascending=[False, True]).reset_index(drop=True)


def _param_effects(df: pd.DataFrame) -> pd.DataFrame:
    params = [
        "tp_mult",
        "sl_mult",
        "time_stop_min",
        "break_even_enabled",
        "break_even_trigger_r",
        "break_even_offset_bps",
        "partial_take_enabled",
        "partial_take_r",
        "partial_take_pct",
    ]
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=["param", "value", "n", "pass_all_rate", "median_delta_expectancy", "median_delta_maxdd", "median_delta_cvar5"])
    for p in params:
        if p not in df.columns:
            continue
        g = df.groupby(p, dropna=False)
        for val, s in g:
            rows.append(
                {
                    "param": p,
                    "value": val,
                    "n": int(len(s)),
                    "pass_all_rate": float(pd.to_numeric(s["pass_all"], errors="coerce").fillna(0).mean()),
                    "median_delta_expectancy": float(pd.to_numeric(s["delta_expectancy_best_exit_minus_baseline_exit"], errors="coerce").median()),
                    "median_delta_maxdd": float(pd.to_numeric(s["delta_maxdd_best_exit_minus_baseline_exit"], errors="coerce").median()),
                    "median_delta_cvar5": float(pd.to_numeric(s["delta_cvar5_best_exit_minus_baseline_exit"], errors="coerce").median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["param", "n", "value"], ascending=[True, False, True]).reset_index(drop=True)


def _gate_aware_top50(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    p_all = pd.to_numeric(df.get("pass_all", 0), errors="coerce").fillna(0).astype(int) == 1
    if p_all.any():
        base = df[p_all].copy()
    else:
        base = df[
            (pd.to_numeric(df.get("pass_data_quality", 0), errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(df.get("pass_participation", 0), errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(df.get("pass_stability", 0), errors="coerce").fillna(0).astype(int) == 1)
        ].copy()
        if base.empty:
            base = df.copy()
    return _stable_sort_for_shortlist(base).head(50).copy().reset_index(drop=True)


def _neighbors(v: float, all_vals: Sequence[float]) -> List[float]:
    xs = sorted({float(x) for x in all_vals})
    if not xs:
        return []
    try:
        i = xs.index(float(v))
    except ValueError:
        i = int(np.argmin([abs(x - float(v)) for x in xs]))
    idx = [max(0, i - 1), i, min(len(xs) - 1, i + 1)]
    out = sorted({float(xs[j]) for j in idx})
    return out


def _build_refine_grid(
    *,
    top50: pd.DataFrame,
    coarse_effects: pd.DataFrame,
    defaults: Dict[str, List[Any]],
    cap: int,
    seed: int,
) -> Tuple[List[exit_sweep.ExitCfg], Dict[str, Any]]:
    if top50.empty:
        full = _build_cfg_grid_from_lists(defaults)
        sampled, smeta = _sample_configs_stratified(full, cap=cap, seed=seed, label="refine")
        return sampled, {"strategy": "fallback_full_defaults", "sampling": smeta}

    def top_values(col: str, k: int, fallback: List[Any]) -> List[Any]:
        if col not in top50.columns:
            return list(fallback)
        vc = (
            top50.groupby(col)["delta_expectancy_best_exit_minus_baseline_exit"]
            .agg(["count", "median"])
            .reset_index()
            .sort_values(["count", "median", col], ascending=[False, False, True])
        )
        vals = vc[col].tolist()[:k]
        if not vals:
            vals = list(fallback)
        return vals

    best_row = top50.iloc[0].to_dict()

    tp_vals = sorted(
        {
            *[float(x) for x in top_values("tp_mult", 3, defaults["tp_mult"])],
            *[float(x) for x in _neighbors(float(best_row.get("tp_mult", defaults["tp_mult"][0])), defaults["tp_mult"])],
        }
    )
    sl_vals = sorted(
        {
            *[float(x) for x in top_values("sl_mult", 3, defaults["sl_mult"])],
            *[float(x) for x in _neighbors(float(best_row.get("sl_mult", defaults["sl_mult"][0])), defaults["sl_mult"])],
        }
    )
    ts_vals = sorted(
        {
            *[int(x) for x in top_values("time_stop_min", 3, defaults["time_stop_min"])],
            *[int(x) for x in _neighbors(float(best_row.get("time_stop_min", defaults["time_stop_min"][0])), defaults["time_stop_min"])],
        }
    )
    be_en_vals = [0, 1]

    be_trig_vals = sorted({float(x) for x in top_values("break_even_trigger_r", 2, defaults["break_even_trigger_r"])})
    be_off_vals = sorted({float(x) for x in top_values("break_even_offset_bps", 2, defaults["break_even_offset_bps"])})

    pt_effect = coarse_effects[coarse_effects["param"] == "partial_take_enabled"].copy()
    use_pt = False
    if not pt_effect.empty:
        m0 = float(pt_effect[pt_effect["value"] == 0]["median_delta_expectancy"].median()) if (pt_effect["value"] == 0).any() else float("-inf")
        m1 = float(pt_effect[pt_effect["value"] == 1]["median_delta_expectancy"].median()) if (pt_effect["value"] == 1).any() else float("-inf")
        use_pt = bool(np.isfinite(m1) and np.isfinite(m0) and m1 >= m0)
    pt_en_vals = [0, 1] if use_pt else [0]
    pt_r_vals = sorted({float(x) for x in top_values("partial_take_r", 2, defaults["partial_take_r"])})
    pt_pct_vals = sorted({float(x) for x in top_values("partial_take_pct", 2, defaults["partial_take_pct"])})

    grid_lists = {
        "tp_mult": tp_vals if tp_vals else defaults["tp_mult"],
        "sl_mult": sl_vals if sl_vals else defaults["sl_mult"],
        "time_stop_min": ts_vals if ts_vals else defaults["time_stop_min"],
        "break_even_enabled": be_en_vals,
        "break_even_trigger_r": be_trig_vals if be_trig_vals else defaults["break_even_trigger_r"],
        "break_even_offset_bps": be_off_vals if be_off_vals else defaults["break_even_offset_bps"],
        "partial_take_enabled": pt_en_vals,
        "partial_take_r": pt_r_vals if pt_r_vals else defaults["partial_take_r"],
        "partial_take_pct": pt_pct_vals if pt_pct_vals else defaults["partial_take_pct"],
    }
    full = _build_cfg_grid_from_lists(grid_lists)
    sampled, smeta = _sample_configs_stratified(full, cap=cap, seed=seed, label="refine")
    meta = {
        "strategy": "top50_support_plus_local_neighbors",
        "grid_lists": grid_lists,
        "sampling": smeta,
    }
    return sampled, meta


def _build_cfg_grid_from_lists(vals: Dict[str, List[Any]]) -> List[exit_sweep.ExitCfg]:
    out: List[exit_sweep.ExitCfg] = []
    for x in itertools.product(
        vals["tp_mult"],
        vals["sl_mult"],
        vals["time_stop_min"],
        vals["break_even_enabled"],
        vals["break_even_trigger_r"],
        vals["break_even_offset_bps"],
        vals["partial_take_enabled"],
        vals["partial_take_r"],
        vals["partial_take_pct"],
    ):
        cfg = exit_sweep.ExitCfg(
            tp_mult=float(x[0]),
            sl_mult=float(x[1]),
            time_stop_min=int(x[2]),
            break_even_enabled=int(x[3]),
            break_even_trigger_r=float(x[4]),
            break_even_offset_bps=float(x[5]),
            partial_take_enabled=int(x[6]),
            partial_take_r=float(x[7]),
            partial_take_pct=float(x[8]),
        )
        if cfg.partial_take_enabled == 1 and cfg.partial_take_r >= cfg.tp_mult:
            continue
        out.append(cfg)
    return sorted(out, key=_cfg_key)


def _normalize_exit_reason(x: Any) -> str:
    s = str(x).strip().lower()
    if not s:
        return "unknown"
    if s == "window_end":
        return "timeout"
    return s


def _hold_minutes(entry_col: pd.Series, exit_col: pd.Series) -> pd.Series:
    et = pd.to_datetime(entry_col, utc=True, errors="coerce")
    xt = pd.to_datetime(exit_col, utc=True, errors="coerce")
    return ((xt - et).dt.total_seconds() / 60.0).astype(float)


def _trade_diagnostics_from_signal_rows(
    signal_df: pd.DataFrame,
    best_cfg: exit_sweep.ExitCfg,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = signal_df.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")

    b = pd.DataFrame(
        {
            "symbol": x.get("symbol"),
            "signal_id": x.get("signal_id"),
            "signal_time": x.get("signal_time"),
            "split_id": pd.to_numeric(x.get("split_id", np.nan), errors="coerce"),
            "signal_tp_mult": pd.to_numeric(x.get("signal_tp_mult", np.nan), errors="coerce"),
            "signal_sl_mult": pd.to_numeric(x.get("signal_sl_mult", np.nan), errors="coerce"),
            "filled": pd.to_numeric(x.get("baseline_filled", 0), errors="coerce"),
            "valid_for_metrics": pd.to_numeric(x.get("baseline_valid_for_metrics", 0), errors="coerce"),
            "sl_hit": pd.to_numeric(x.get("baseline_sl_hit", 0), errors="coerce"),
            "tp_hit": pd.to_numeric(x.get("baseline_tp_hit", 0), errors="coerce"),
            "entry_time": x.get("baseline_entry_time"),
            "exit_time": x.get("baseline_exit_time"),
            "entry_price": pd.to_numeric(x.get("baseline_entry_price", np.nan), errors="coerce"),
            "exit_price": pd.to_numeric(x.get("baseline_exit_price", np.nan), errors="coerce"),
            "exit_reason": x.get("baseline_exit_reason"),
            "pnl_net_pct": pd.to_numeric(x.get("baseline_pnl_net_pct", np.nan), errors="coerce"),
            "mae_pct": pd.to_numeric(x.get("baseline_mae_pct", np.nan), errors="coerce"),
            "mfe_pct": pd.to_numeric(x.get("baseline_mfe_pct", np.nan), errors="coerce"),
        }
    )
    b["exit_reason"] = b["exit_reason"].map(_normalize_exit_reason)
    b["hold_minutes"] = _hold_minutes(b["entry_time"], b["exit_time"])
    b["risk_pct"] = (1.0 - pd.to_numeric(b["signal_sl_mult"], errors="coerce")).clip(lower=1e-8)
    b["mode"] = "baseline_exit"

    e = pd.DataFrame(
        {
            "symbol": x.get("symbol"),
            "signal_id": x.get("signal_id"),
            "signal_time": x.get("signal_time"),
            "split_id": pd.to_numeric(x.get("split_id", np.nan), errors="coerce"),
            "signal_tp_mult": pd.to_numeric(x.get("signal_tp_mult", np.nan), errors="coerce"),
            "signal_sl_mult": pd.to_numeric(x.get("signal_sl_mult", np.nan), errors="coerce"),
            "filled": pd.to_numeric(x.get("exec_filled", 0), errors="coerce"),
            "valid_for_metrics": pd.to_numeric(x.get("exec_valid_for_metrics", 0), errors="coerce"),
            "sl_hit": pd.to_numeric(x.get("exec_sl_hit", 0), errors="coerce"),
            "tp_hit": pd.to_numeric(x.get("exec_tp_hit", 0), errors="coerce"),
            "entry_time": x.get("exec_entry_time"),
            "exit_time": x.get("exec_exit_time"),
            "entry_price": pd.to_numeric(x.get("exec_entry_price", np.nan), errors="coerce"),
            "exit_price": pd.to_numeric(x.get("exec_exit_price", np.nan), errors="coerce"),
            "exit_reason": x.get("exec_exit_reason"),
            "pnl_net_pct": pd.to_numeric(x.get("exec_pnl_net_pct", np.nan), errors="coerce"),
            "mae_pct": pd.to_numeric(x.get("exec_mae_pct", np.nan), errors="coerce"),
            "mfe_pct": pd.to_numeric(x.get("exec_mfe_pct", np.nan), errors="coerce"),
            "entry_type": x.get("exec_entry_type"),
            "entry_improvement_bps": pd.to_numeric(x.get("entry_improvement_bps", np.nan), errors="coerce"),
        }
    )
    e["exit_reason"] = e["exit_reason"].map(_normalize_exit_reason)
    e["hold_minutes"] = _hold_minutes(e["entry_time"], e["exit_time"])
    base_r = (1.0 - pd.to_numeric(e["signal_sl_mult"], errors="coerce")).clip(lower=1e-8)
    e["risk_pct"] = (base_r * float(best_cfg.sl_mult)).clip(lower=1e-8)
    e["mode"] = "best_exit"
    return b, e


def _mfe_mae_quantiles(df: pd.DataFrame, mode: str) -> List[Dict[str, Any]]:
    m = (
        (pd.to_numeric(df.get("filled", 0), errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df.get("valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1)
    )
    rows: List[Dict[str, Any]] = []
    for metric in ["mae_pct", "mfe_pct"]:
        s = pd.to_numeric(df.loc[m, metric], errors="coerce").dropna()
        if s.empty:
            rows.append({"mode": mode, "metric": metric, "q05": np.nan, "q25": np.nan, "q50": np.nan, "q75": np.nan, "q95": np.nan})
            continue
        rows.append(
            {
                "mode": mode,
                "metric": metric,
                "q05": float(s.quantile(0.05)),
                "q25": float(s.quantile(0.25)),
                "q50": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
                "q95": float(s.quantile(0.95)),
            }
        )
    return rows


def _reach_rate(df: pd.DataFrame, r_thr: float) -> float:
    m = (
        (pd.to_numeric(df.get("filled", 0), errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df.get("valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1)
    )
    if not bool(m.any()):
        return float("nan")
    mfe = pd.to_numeric(df.loc[m, "mfe_pct"], errors="coerce")
    risk = pd.to_numeric(df.loc[m, "risk_pct"], errors="coerce").clip(lower=1e-8)
    ok = (mfe >= float(r_thr) * risk).fillna(False)
    return float(ok.mean()) if len(ok) else float("nan")


def _timeout_rate(df: pd.DataFrame) -> float:
    m = (
        (pd.to_numeric(df.get("filled", 0), errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df.get("valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1)
    )
    if not bool(m.any()):
        return float("nan")
    r = df.loc[m, "exit_reason"].astype(str).str.lower().str.strip()
    return float((r == "timeout").mean())


def _median_hold_win_loss(df: pd.DataFrame) -> Tuple[float, float]:
    m = (
        (pd.to_numeric(df.get("filled", 0), errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df.get("valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1)
    )
    if not bool(m.any()):
        return float("nan"), float("nan")
    pnl = pd.to_numeric(df.loc[m, "pnl_net_pct"], errors="coerce")
    hold = pd.to_numeric(df.loc[m, "hold_minutes"], errors="coerce")
    win = hold[pnl > 0].dropna()
    loss = hold[pnl < 0].dropna()
    return float(win.median()) if not win.empty else float("nan"), float(loss.median()) if not loss.empty else float("nan")


def _overall_row(best_eval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": "overall",
        "symbols": 1,
        "signals_total": int(best_eval.get("overall_signals_total", 0)),
        "trades_total": int(best_eval.get("overall_entries_valid", 0)),
        "baseline_mean_expectancy_net": float(best_eval.get("overall_baseline_expectancy_net", np.nan)),
        "best_exit_mean_expectancy_net": float(best_eval.get("overall_exec_expectancy_net", np.nan)),
        "delta_expectancy_best_exit_minus_baseline_exit": float(best_eval.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "baseline_pnl_net_sum": float(best_eval.get("overall_baseline_pnl_net_sum", np.nan)),
        "best_exit_pnl_net_sum": float(best_eval.get("overall_exec_pnl_net_sum", np.nan)),
        "baseline_pnl_std": float(best_eval.get("overall_baseline_pnl_std", np.nan)),
        "best_exit_pnl_std": float(best_eval.get("overall_exec_pnl_std", np.nan)),
        "baseline_worst_decile_mean": float(best_eval.get("overall_baseline_worst_decile_mean", np.nan)),
        "best_exit_worst_decile_mean": float(best_eval.get("overall_exec_worst_decile_mean", np.nan)),
        "baseline_cvar_5": float(best_eval.get("overall_baseline_cvar_5", np.nan)),
        "best_exit_cvar_5": float(best_eval.get("overall_exec_cvar_5", np.nan)),
        "delta_cvar5_best_exit_minus_baseline_exit": float(
            best_eval.get("overall_exec_cvar_5", np.nan) - best_eval.get("overall_baseline_cvar_5", np.nan)
        ),
        "baseline_max_drawdown": float(best_eval.get("overall_baseline_max_drawdown", np.nan)),
        "best_exit_max_drawdown": float(best_eval.get("overall_exec_max_drawdown", np.nan)),
        "delta_maxdd_best_exit_minus_baseline_exit": float(
            best_eval.get("overall_exec_max_drawdown", np.nan) - best_eval.get("overall_baseline_max_drawdown", np.nan)
        ),
        "entry_rate": float(best_eval.get("overall_entry_rate", np.nan)),
        "taker_share": float(best_eval.get("overall_exec_taker_share", np.nan)),
        "median_fill_delay_min": float(best_eval.get("overall_exec_median_fill_delay_min", np.nan)),
        "p95_fill_delay_min": float(best_eval.get("overall_exec_p95_fill_delay_min", np.nan)),
        "median_entry_improvement_bps": float(best_eval.get("overall_exec_median_entry_improvement_bps", np.nan)),
        "missing_slice_rate": float(best_eval.get("overall_missing_slice_rate", np.nan)),
        "min_split_expectancy_best_exit": float(best_eval.get("min_split_expectancy_net", np.nan)),
        "median_split_expectancy_best_exit": float(best_eval.get("median_split_expectancy_net", np.nan)),
        "std_split_expectancy_best_exit": float(best_eval.get("std_split_expectancy_net", np.nan)),
    }


def _decision_pass_flags(ov: Dict[str, Any], args: argparse.Namespace, base_stats: Tuple[float, float, float]) -> Dict[str, int]:
    delta_exp = float(ov.get("delta_expectancy_best_exit_minus_baseline_exit", np.nan))
    delta_maxdd = float(ov.get("delta_maxdd_best_exit_minus_baseline_exit", np.nan))
    delta_cvar = float(ov.get("delta_cvar5_best_exit_minus_baseline_exit", np.nan))
    min_ok, med_ok, std_ok, stab_ok = exit_sweep._stability_flags(
        exec_min=float(ov.get("min_split_expectancy_best_exit", np.nan)),
        exec_med=float(ov.get("median_split_expectancy_best_exit", np.nan)),
        exec_std=float(ov.get("std_split_expectancy_best_exit", np.nan)),
        base_min=float(base_stats[0]),
        base_med=float(base_stats[1]),
        base_std=float(base_stats[2]),
    )
    p_exp = int(np.isfinite(delta_exp) and delta_exp >= float(args.expectancy_epsilon))
    p_maxdd = int(np.isfinite(delta_maxdd) and delta_maxdd >= -float(args.maxdd_worse_tol))
    p_cvar = int(np.isfinite(delta_cvar) and delta_cvar >= -float(args.cvar_worse_tol))
    p_data = int(np.isfinite(float(ov.get("missing_slice_rate", np.nan))) and float(ov.get("missing_slice_rate", np.nan)) <= float(args.max_missing_slice_rate))
    p_part = int(np.isfinite(float(ov.get("entry_rate", np.nan))) and float(ov.get("entry_rate", np.nan)) > 0.0)
    p_all = int(p_exp and p_maxdd and p_cvar and stab_ok and p_data and p_part)
    return {
        "pass_expectancy": p_exp,
        "pass_maxdd_not_worse": p_maxdd,
        "pass_cvar_not_worse": p_cvar,
        "pass_stability_min": int(min_ok),
        "pass_stability_median": int(med_ok),
        "pass_stability_std": int(std_ok),
        "pass_stability": int(stab_ok),
        "pass_data_quality": int(p_data),
        "pass_participation": int(p_part),
        "pass_all": int(p_all),
    }


def _write_git_status(path: Path) -> None:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--short"], check=False, capture_output=True, text=True)
        path.write_text(out.stdout, encoding="utf-8")
    except Exception as e:
        path.write_text(f"git status failed: {e}\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise SystemExit("HARD scope lock: symbol must be SOLUSDT")

    run_root = _resolve(args.outdir) / f"PHASEC_SOL_{_utc_tag()}"
    coarse_dir = run_root / "coarse"
    diag_dir = run_root / "diagnostics"
    refine_dir = run_root / "refine"
    for d in [run_root, coarse_dir, diag_dir, refine_dir, run_root / "config_snapshot"]:
        d.mkdir(parents=True, exist_ok=True)

    phase_a_dir = _resolve(args.phase_a_contract_dir)
    metrics_src = phase_a_dir / "metrics_definition.md"
    fee_src = phase_a_dir / "fee_model.json"
    if not metrics_src.exists() or not fee_src.exists():
        raise SystemExit(f"Phase A contract artifacts missing in {phase_a_dir}")
    shutil.copy2(metrics_src, run_root / "metrics_definition.md")
    shutil.copy2(fee_src, run_root / "fee_model.json")
    fee_model = json.loads((run_root / "fee_model.json").read_text(encoding="utf-8"))

    eval_args = _build_eval_args(args=args, fee_model=fee_model)
    if str(args.signals_csv).strip():
        eval_args.signals_csv = str(args.signals_csv)
    else:
        eval_args.signals_csv = str(_resolve(f"data/signals/{symbol}_signals_1h.csv"))
    signals_df, signals_src = ga_exec._load_signals_for_symbol(symbol=symbol, args=eval_args)

    subset_csv = run_root / "signal_subset.csv"
    signals_df.to_csv(subset_csv, index=False)
    subset_hash = _signal_subset_hash(signals_df)
    (run_root / "signal_subset_hash.txt").write_text(subset_hash + "\n", encoding="utf-8")

    split_def, split_payload = _freeze_splits(signals_df=signals_df, train_ratio=float(args.train_ratio), wf_splits=int(args.wf_splits))
    split_json = run_root / "wf_split_definition.json"
    _json_dump(split_json, split_payload)
    split_hash = _sha256_file(split_json)

    bundle = _build_bundle(
        symbol=symbol,
        signals_df=signals_df,
        signal_csv=subset_csv,
        eval_args=eval_args,
        split_def=split_def,
    )
    bundles = [bundle]

    full_grid = _build_full_grid(args)
    coarse_cfgs, coarse_sampling_meta = _sample_configs_stratified(
        cfgs=full_grid,
        cap=int(args.coarse_max_configs),
        seed=int(args.coarse_seed),
        label="coarse",
    )
    _json_dump(coarse_dir / "sampling_meta.json", coarse_sampling_meta)

    coarse_df, base_split_stats, coarse_eval_meta = _evaluate_cfgs(
        cfgs=coarse_cfgs,
        bundles=bundles,
        eval_args=eval_args,
        base_split_stats=None,
        expectancy_epsilon=float(args.expectancy_epsilon),
        maxdd_worse_tol=float(args.maxdd_worse_tol),
        cvar_worse_tol=float(args.cvar_worse_tol),
        max_missing_slice_rate=float(args.max_missing_slice_rate),
    )
    coarse_results_csv = coarse_dir / "exit_sweep_results_coarse.csv"
    coarse_topk_csv = coarse_dir / "exit_sweep_topk_coarse.csv"
    coarse_df.to_csv(coarse_results_csv, index=False)
    _stable_sort_for_shortlist(coarse_df).head(int(args.topk)).to_csv(coarse_topk_csv, index=False)

    coarse_top50 = _gate_aware_top50(coarse_df)
    coarse_top50_csv = diag_dir / "coarse_top50.csv"
    coarse_top50.to_csv(coarse_top50_csv, index=False)
    coarse_fail_csv = diag_dir / "coarse_fail_reason_counts.csv"
    _fail_reason_counts(coarse_df).to_csv(coarse_fail_csv, index=False)
    coarse_param_csv = diag_dir / "coarse_param_effects.csv"
    coarse_effects = _param_effects(coarse_df)
    coarse_effects.to_csv(coarse_param_csv, index=False)

    diag_md_lines: List[str] = []
    diag_md_lines.append("# Coarse Diagnostics (SOLUSDT)")
    diag_md_lines.append("")
    diag_md_lines.append(f"- Generated UTC: {_utc_now()}")
    diag_md_lines.append(f"- Full grid size: {len(full_grid)}")
    diag_md_lines.append(f"- Coarse sampled configs: {len(coarse_cfgs)}")
    diag_md_lines.append(f"- Sampling method: {coarse_sampling_meta.get('method','')}")
    diag_md_lines.append(f"- Sampling seed: {int(args.coarse_seed)}")
    diag_md_lines.append("")
    for c in [
        "pass_expectancy",
        "pass_cvar_not_worse",
        "pass_maxdd_not_worse",
        "pass_stability",
        "pass_data_quality",
        "pass_participation",
        "pass_all",
    ]:
        if c in coarse_df.columns and len(coarse_df) > 0:
            diag_md_lines.append(f"- {c}: {int(pd.to_numeric(coarse_df[c], errors='coerce').fillna(0).astype(int).sum())}/{len(coarse_df)}")
    if not coarse_df.empty:
        diag_md_lines.append("")
        diag_md_lines.append("## Delta Distributions")
        for c in [
            "delta_expectancy_best_exit_minus_baseline_exit",
            "delta_cvar5_best_exit_minus_baseline_exit",
            "delta_maxdd_best_exit_minus_baseline_exit",
        ]:
            s = pd.to_numeric(coarse_df[c], errors="coerce")
            diag_md_lines.append(f"- {c}: median={float(s.median()):.6f}, p10={float(s.quantile(0.10)):.6f}, p90={float(s.quantile(0.90)):.6f}")
    (diag_dir / "coarse_diagnostics.md").write_text("\n".join(diag_md_lines).strip() + "\n", encoding="utf-8")

    default_lists = {
        "tp_mult": exit_sweep._parse_floats(args.tp_mults),
        "sl_mult": exit_sweep._parse_floats(args.sl_mults),
        "time_stop_min": [int(x) for x in exit_sweep._parse_ints(args.time_stops_min)],
        "break_even_enabled": [int(x) for x in exit_sweep._parse_bools(args.break_even_enabled)],
        "break_even_trigger_r": exit_sweep._parse_floats(args.break_even_trigger_r),
        "break_even_offset_bps": exit_sweep._parse_floats(args.break_even_offset_bps),
        "partial_take_enabled": [int(x) for x in exit_sweep._parse_bools(args.partial_take_enabled)],
        "partial_take_r": exit_sweep._parse_floats(args.partial_take_r),
        "partial_take_pct": exit_sweep._parse_floats(args.partial_take_pct),
    }
    refine_cfgs, refine_meta = _build_refine_grid(
        top50=coarse_top50,
        coarse_effects=coarse_effects,
        defaults=default_lists,
        cap=int(args.refine_max_configs),
        seed=int(args.refine_seed),
    )
    _json_dump(refine_dir / "sampling_meta.json", refine_meta)

    refine_df, _, refine_eval_meta = _evaluate_cfgs(
        cfgs=refine_cfgs,
        bundles=bundles,
        eval_args=eval_args,
        base_split_stats=base_split_stats,
        expectancy_epsilon=float(args.expectancy_epsilon),
        maxdd_worse_tol=float(args.maxdd_worse_tol),
        cvar_worse_tol=float(args.cvar_worse_tol),
        max_missing_slice_rate=float(args.max_missing_slice_rate),
    )
    refine_results_csv = refine_dir / "exit_sweep_results_refine.csv"
    refine_topk_csv = refine_dir / "exit_sweep_topk_refine.csv"
    refine_df.to_csv(refine_results_csv, index=False)
    _stable_sort_for_shortlist(refine_df).head(int(args.topk)).to_csv(refine_topk_csv, index=False)

    ranked_refine = _stable_sort_for_shortlist(refine_df)
    if ranked_refine.empty:
        raise SystemExit("Refine results are empty")
    best_row = ranked_refine.iloc[0].to_dict()
    best_cfg = exit_sweep.ExitCfg(
        tp_mult=float(best_row["tp_mult"]),
        sl_mult=float(best_row["sl_mult"]),
        time_stop_min=int(best_row["time_stop_min"]),
        break_even_enabled=int(best_row["break_even_enabled"]),
        break_even_trigger_r=float(best_row["break_even_trigger_r"]),
        break_even_offset_bps=float(best_row["break_even_offset_bps"]),
        partial_take_enabled=int(best_row["partial_take_enabled"]),
        partial_take_r=float(best_row["partial_take_r"]),
        partial_take_pct=float(best_row["partial_take_pct"]),
    )
    best_genome = exit_sweep._cfg_to_genome(best_cfg, mode=str(eval_args.mode))
    best_eval = ga_exec._evaluate_genome(genome=best_genome, bundles=bundles, args=eval_args, detailed=True)

    split_df = best_eval.get("split_rows_df", pd.DataFrame()).copy()
    if not split_df.empty:
        split_df = split_df.rename(
            columns={
                "exec_mean_expectancy_net": "best_exit_mean_expectancy_net",
                "delta_expectancy_exec_minus_baseline": "delta_expectancy_best_exit_minus_baseline_exit",
                "exec_cvar_5": "best_exit_cvar_5",
                "exec_max_drawdown": "best_exit_max_drawdown",
                "exec_entry_rate": "entry_rate",
                "exec_taker_share": "taker_share",
                "exec_median_fill_delay_min": "median_fill_delay_min",
                "exec_p95_fill_delay_min": "p95_fill_delay_min",
                "exec_median_entry_improvement_bps": "median_entry_improvement_bps",
            }
        )
        if "best_exit_cvar_5" in split_df.columns and "baseline_cvar_5" in split_df.columns:
            split_df["delta_cvar5_best_exit_minus_baseline_exit"] = (
                pd.to_numeric(split_df["best_exit_cvar_5"], errors="coerce")
                - pd.to_numeric(split_df["baseline_cvar_5"], errors="coerce")
            )
        if "best_exit_max_drawdown" in split_df.columns and "baseline_max_drawdown" in split_df.columns:
            split_df["delta_maxdd_best_exit_minus_baseline_exit"] = (
                pd.to_numeric(split_df["best_exit_max_drawdown"], errors="coerce")
                - pd.to_numeric(split_df["baseline_max_drawdown"], errors="coerce")
            )
        if "exec_entries_valid" in split_df.columns:
            split_df["trades_total"] = pd.to_numeric(split_df["exec_entries_valid"], errors="coerce")
        elif "signals_total" in split_df.columns and "entry_rate" in split_df.columns:
            split_df["trades_total"] = (
                pd.to_numeric(split_df["signals_total"], errors="coerce")
                * pd.to_numeric(split_df["entry_rate"], errors="coerce")
            ).round()
    split_csv = run_root / "walkforward_results_by_split.csv"
    split_df.to_csv(split_csv, index=False)

    sym_df = best_eval.get("symbol_rows_df", pd.DataFrame()).copy()
    if not sym_df.empty:
        sym_df = sym_df.rename(
            columns={
                "exec_mean_expectancy_net": "best_exit_mean_expectancy_net",
                "delta_expectancy_exec_minus_baseline": "delta_expectancy_best_exit_minus_baseline_exit",
                "exec_cvar_5": "best_exit_cvar_5",
                "exec_max_drawdown": "best_exit_max_drawdown",
                "exec_entry_rate": "entry_rate",
                "exec_taker_share": "taker_share",
                "exec_median_fill_delay_min": "median_fill_delay_min",
                "exec_p95_fill_delay_min": "p95_fill_delay_min",
                "exec_median_entry_improvement_bps": "median_entry_improvement_bps",
            }
        )
        if "best_exit_cvar_5" in sym_df.columns and "baseline_cvar_5" in sym_df.columns:
            sym_df["delta_cvar5_best_exit_minus_baseline_exit"] = (
                pd.to_numeric(sym_df["best_exit_cvar_5"], errors="coerce")
                - pd.to_numeric(sym_df["baseline_cvar_5"], errors="coerce")
            )
        if "best_exit_max_drawdown" in sym_df.columns and "baseline_max_drawdown" in sym_df.columns:
            sym_df["delta_maxdd_best_exit_minus_baseline_exit"] = (
                pd.to_numeric(sym_df["best_exit_max_drawdown"], errors="coerce")
                - pd.to_numeric(sym_df["baseline_max_drawdown"], errors="coerce")
            )
    sym_csv = run_root / "risk_rollup_by_symbol.csv"
    sym_df.to_csv(sym_csv, index=False)

    ov_row = _overall_row(best_eval)
    ov_flags = _decision_pass_flags(ov_row, args=args, base_stats=base_split_stats)
    ov_row.update(ov_flags)
    ov_df = pd.DataFrame([ov_row])
    ov_csv = run_root / "risk_rollup_overall.csv"
    ov_df.to_csv(ov_csv, index=False)

    signal_rows = best_eval.get("signal_rows_df", pd.DataFrame()).copy()
    bdiag, ediag = _trade_diagnostics_from_signal_rows(signal_rows, best_cfg=best_cfg)
    bdiag_csv = run_root / "trade_diagnostics_baseline.csv"
    ediag_csv = run_root / "trade_diagnostics_best.csv"
    bdiag.to_csv(bdiag_csv, index=False)
    ediag.to_csv(ediag_csv, index=False)

    mfe_rows = _mfe_mae_quantiles(bdiag, "baseline_exit") + _mfe_mae_quantiles(ediag, "best_exit")
    mfe_df = pd.DataFrame(mfe_rows)
    mfe_csv = run_root / "mfe_mae_summary.csv"
    mfe_df.to_csv(mfe_csv, index=False)

    bdist = bdiag["exit_reason"].fillna("unknown").astype(str).value_counts(dropna=False).rename_axis("exit_reason").reset_index(name="baseline_count")
    edist = ediag["exit_reason"].fillna("unknown").astype(str).value_counts(dropna=False).rename_axis("exit_reason").reset_index(name="best_count")
    ex = bdist.merge(edist, on="exit_reason", how="outer").fillna(0)
    ex["baseline_count"] = pd.to_numeric(ex["baseline_count"], errors="coerce").fillna(0).astype(int)
    ex["best_count"] = pd.to_numeric(ex["best_count"], errors="coerce").fillna(0).astype(int)
    btot = max(1, int(ex["baseline_count"].sum()))
    etot = max(1, int(ex["best_count"].sum()))
    ex["baseline_pct"] = ex["baseline_count"] / btot
    ex["best_pct"] = ex["best_count"] / etot
    ex["delta_pct_best_minus_baseline"] = ex["best_pct"] - ex["baseline_pct"]
    ex = ex.sort_values("delta_pct_best_minus_baseline", ascending=False).reset_index(drop=True)
    ex_csv = run_root / "exit_reason_distribution.csv"
    ex.to_csv(ex_csv, index=False)

    b_r05 = _reach_rate(bdiag, 0.5)
    b_r10 = _reach_rate(bdiag, 1.0)
    b_r15 = _reach_rate(bdiag, 1.5)
    e_r05 = _reach_rate(ediag, 0.5)
    e_r10 = _reach_rate(ediag, 1.0)
    e_r15 = _reach_rate(ediag, 1.5)

    b_timeout = _timeout_rate(bdiag)
    e_timeout = _timeout_rate(ediag)
    b_win_hold, b_loss_hold = _median_hold_win_loss(bdiag)
    e_win_hold, e_loss_hold = _median_hold_win_loss(ediag)

    miss = signal_rows.copy()
    miss["missing_slice_flag"] = pd.to_numeric(miss.get("missing_slice_flag", 0), errors="coerce").fillna(0).astype(int)
    miss = miss[miss["missing_slice_flag"] == 1].copy()
    miss_reason = (
        miss.get("exec_skip_reason", pd.Series(dtype=str)).fillna("").astype(str).replace("", "unknown").value_counts().rename_axis("reason").reset_index(name="count")
    )
    miss_reason_csv = run_root / "missing_slice_reasons.csv"
    miss_reason.to_csv(miss_reason_csv, index=False)

    baseline_vs_best = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "signals_total_test": int(ov_row["signals_total"]),
                "trades_total_test": int(ov_row["trades_total"]),
                "baseline_expectancy_net": float(ov_row["baseline_mean_expectancy_net"]),
                "best_exit_expectancy_net": float(ov_row["best_exit_mean_expectancy_net"]),
                "delta_expectancy_best_exit_minus_baseline_exit": float(ov_row["delta_expectancy_best_exit_minus_baseline_exit"]),
                "baseline_cvar_5": float(ov_row["baseline_cvar_5"]),
                "best_exit_cvar_5": float(ov_row["best_exit_cvar_5"]),
                "delta_cvar5_best_exit_minus_baseline_exit": float(ov_row["delta_cvar5_best_exit_minus_baseline_exit"]),
                "baseline_max_drawdown": float(ov_row["baseline_max_drawdown"]),
                "best_exit_max_drawdown": float(ov_row["best_exit_max_drawdown"]),
                "delta_maxdd_best_exit_minus_baseline_exit": float(ov_row["delta_maxdd_best_exit_minus_baseline_exit"]),
                "pass_expectancy": int(ov_flags["pass_expectancy"]),
                "pass_maxdd_not_worse": int(ov_flags["pass_maxdd_not_worse"]),
                "pass_cvar_not_worse": int(ov_flags["pass_cvar_not_worse"]),
                "pass_stability": int(ov_flags["pass_stability"]),
                "pass_data_quality": int(ov_flags["pass_data_quality"]),
                "pass_participation": int(ov_flags["pass_participation"]),
                "pass_all": int(ov_flags["pass_all"]),
            }
        ]
    )
    summary_csv = run_root / "baseline_vs_best_summary.csv"
    baseline_vs_best.to_csv(summary_csv, index=False)

    split_trades = pd.to_numeric(split_df.get("trades_total", np.nan), errors="coerce").dropna()
    min_split_trades = int(split_trades.min()) if not split_trades.empty else 0
    med_split_trades = float(split_trades.median()) if not split_trades.empty else float("nan")
    low_support = int((split_trades < 30).any()) if not split_trades.empty else 1

    # Repro checks.
    coarse_cfgs_check, _ = _sample_configs_stratified(
        cfgs=full_grid,
        cap=int(args.coarse_max_configs),
        seed=int(args.coarse_seed),
        label="coarse",
    )
    same_coarse_sample = int([_cfg_key(x) for x in coarse_cfgs_check] == [_cfg_key(x) for x in coarse_cfgs])
    best_eval_recheck = ga_exec._evaluate_genome(genome=best_genome, bundles=bundles, args=eval_args, detailed=False)
    repro_delta = abs(
        float(best_eval_recheck.get("overall_exec_expectancy_net", np.nan))
        - float(best_eval.get("overall_exec_expectancy_net", np.nan))
    )
    repro_pass = int(np.isfinite(repro_delta) and repro_delta <= 1e-12 and same_coarse_sample == 1)

    diag_sum_lines: List[str] = []
    diag_sum_lines.append("# Diagnostics Summary (SOLUSDT Phase C)")
    diag_sum_lines.append("")
    diag_sum_lines.append(f"- Generated UTC: {_utc_now()}")
    diag_sum_lines.append(f"- Signals total test: {int(ov_row['signals_total'])}")
    diag_sum_lines.append(f"- Trades total test: {int(ov_row['trades_total'])}")
    diag_sum_lines.append(f"- Min split trades: {min_split_trades}")
    diag_sum_lines.append(f"- Median split trades: {med_split_trades:.2f}" if np.isfinite(med_split_trades) else "- Median split trades: nan")
    if low_support == 1:
        diag_sum_lines.append("- LOW-SUPPORT warning: at least one split has <30 trades.")
    diag_sum_lines.append("")
    diag_sum_lines.append("## Reach Before Exit")
    diag_sum_lines.append("")
    diag_sum_lines.append(f"- Baseline reach +0.5R/+1.0R/+1.5R: {b_r05:.4f}, {b_r10:.4f}, {b_r15:.4f}")
    diag_sum_lines.append(f"- Best reach +0.5R/+1.0R/+1.5R: {e_r05:.4f}, {e_r10:.4f}, {e_r15:.4f}")
    diag_sum_lines.append("")
    diag_sum_lines.append("## Timeout And Hold")
    diag_sum_lines.append("")
    diag_sum_lines.append(f"- Timeout rate baseline/best: {b_timeout:.4f} / {e_timeout:.4f}")
    diag_sum_lines.append(f"- Median hold (winners) baseline/best: {b_win_hold:.2f} / {e_win_hold:.2f} minutes")
    diag_sum_lines.append(f"- Median hold (losers) baseline/best: {b_loss_hold:.2f} / {e_loss_hold:.2f} minutes")
    diag_sum_lines.append("")
    diag_sum_lines.append("## Reproducibility")
    diag_sum_lines.append("")
    diag_sum_lines.append(f"- Coarse stratified sample deterministic check: {same_coarse_sample}")
    diag_sum_lines.append(f"- Best config re-eval expectancy abs delta: {repro_delta:.12f}")
    diag_sum_lines.append(f"- Repro check pass: {repro_pass}")
    diag_sum_lines.append("")
    diag_sum_lines.append("## Data Quality")
    diag_sum_lines.append("")
    diag_sum_lines.append(f"- Missing slice rate: {float(ov_row['missing_slice_rate']):.6f}")
    if not miss_reason.empty:
        top_miss = miss_reason.head(5)
        for _, r in top_miss.iterrows():
            diag_sum_lines.append(f"- missing_reason `{r['reason']}`: {int(r['count'])}")
    else:
        diag_sum_lines.append("- missing_reason: none")
    (run_root / "diagnostics_summary.md").write_text("\n".join(diag_sum_lines).strip() + "\n", encoding="utf-8")

    # Decision.
    decision_pass = int(ov_flags["pass_all"])
    proposal: List[str] = []
    if decision_pass == 0:
        if int(ov_flags["pass_expectancy"]) == 0:
            if float(best_cfg.tp_mult) > 1.2:
                proposal.append("TP looks too greedy in current optimum; test lower TP bands around 0.8-1.1.")
            if float(best_cfg.time_stop_min) >= 1440:
                proposal.append("Time stop appears too long; test shorter windows (360-1440 min) to reduce time decay.")
            if int(best_cfg.break_even_enabled) == 1:
                proposal.append("Break-even may be clipping recovery; test BE off and higher trigger levels.")
            if int(best_cfg.partial_take_enabled) == 1:
                proposal.append("Partial take may be dragging expectancy; test partial-off or higher partial_take_r.")
        if int(ov_flags["pass_stability"]) == 0:
            proposal.append("Stability failed; narrow refine grid around top stable cluster instead of expectancy-only peak.")
        if int(ov_flags["pass_maxdd_not_worse"]) == 0:
            proposal.append("MaxDD worsened; test tighter sl_mult neighborhood and shorter time_stop.")
        if not proposal:
            proposal.append("Run another refine pass with tighter gate-aware shortlist and reduced high-variance configs.")

    dec_lines: List[str] = []
    dec_lines.append("# Phase C SOL Decision")
    dec_lines.append("")
    dec_lines.append(f"- Generated UTC: {_utc_now()}")
    dec_lines.append(f"- Symbol: {symbol}")
    dec_lines.append(f"- Phase A contract dir: `{phase_a_dir}`")
    dec_lines.append(f"- Signal subset hash: `{subset_hash}`")
    dec_lines.append(f"- WF split definition hash: `{split_hash}`")
    dec_lines.append("")
    dec_lines.append("## Best Refined Config")
    dec_lines.append("")
    for k, v in asdict(best_cfg).items():
        dec_lines.append(f"- {k}: {v}")
    dec_lines.append(f"- cfg_hash: `{_cfg_hash(best_cfg)}`")
    dec_lines.append("")
    dec_lines.append("## Support")
    dec_lines.append("")
    dec_lines.append(f"- signals_total_test: {int(ov_row['signals_total'])}")
    dec_lines.append(f"- trades_total_test: {int(ov_row['trades_total'])}")
    dec_lines.append(f"- min_split_trades: {min_split_trades}")
    dec_lines.append(f"- median_split_trades: {med_split_trades:.2f}" if np.isfinite(med_split_trades) else "- median_split_trades: nan")
    if low_support == 1:
        dec_lines.append("- LOW-SUPPORT warning: at least one split has <30 trades.")
    dec_lines.append("")
    dec_lines.append("## Rubric")
    dec_lines.append("")
    dec_lines.append(
        f"- pass_expectancy (delta_expectancy_best_exit_minus_baseline_exit >= {float(args.expectancy_epsilon):.6f}): {int(ov_flags['pass_expectancy'])}"
    )
    dec_lines.append(
        f"- pass_maxdd_not_worse (delta_maxdd_best_exit_minus_baseline_exit >= -{float(args.maxdd_worse_tol):.3f}): {int(ov_flags['pass_maxdd_not_worse'])}"
    )
    dec_lines.append(
        f"- pass_cvar_not_worse (delta_cvar5_best_exit_minus_baseline_exit >= -{float(args.cvar_worse_tol):.6f}): {int(ov_flags['pass_cvar_not_worse'])}"
    )
    dec_lines.append(f"- pass_stability: {int(ov_flags['pass_stability'])}")
    dec_lines.append(f"- pass_data_quality: {int(ov_flags['pass_data_quality'])}")
    dec_lines.append(f"- pass_participation: {int(ov_flags['pass_participation'])}")
    dec_lines.append(f"- Decision: **{'PASS' if decision_pass == 1 else 'FAIL'}**")
    dec_lines.append("")
    dec_lines.append("## Deltas")
    dec_lines.append("")
    dec_lines.append(f"- delta_expectancy_best_exit_minus_baseline_exit: {float(ov_row['delta_expectancy_best_exit_minus_baseline_exit']):.6f}")
    dec_lines.append(f"- delta_maxdd_best_exit_minus_baseline_exit: {float(ov_row['delta_maxdd_best_exit_minus_baseline_exit']):.6f}")
    dec_lines.append(f"- delta_cvar5_best_exit_minus_baseline_exit: {float(ov_row['delta_cvar5_best_exit_minus_baseline_exit']):.6f}")
    if decision_pass == 0:
        dec_lines.append("")
        dec_lines.append("## Next Refinement Proposal")
        dec_lines.append("")
        for p in proposal:
            dec_lines.append(f"- {p}")
    (run_root / "decision.md").write_text("\n".join(dec_lines).strip() + "\n", encoding="utf-8")

    repro_lines = [
        "# Repro",
        "",
        "```bash",
        "python3 scripts/phase_c_sol_runner.py \\",
        f"  --symbol {symbol} \\",
        f"  --phase-a-contract-dir {phase_a_dir} \\",
        f"  --signals-csv {signals_src} \\",
        f"  --max-signals {int(args.max_signals)} \\",
        f"  --wf-splits {int(args.wf_splits)} \\",
        f"  --coarse-max-configs {int(args.coarse_max_configs)} \\",
        f"  --refine-max-configs {int(args.refine_max_configs)} \\",
        f"  --coarse-seed {int(args.coarse_seed)} \\",
        f"  --refine-seed {int(args.refine_seed)} \\",
        f"  --workers {int(args.workers)}",
        "```",
    ]
    (run_root / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: C (SOL-only decision-grade rerun: coarse -> refine + diagnostics)",
        f"Timestamp UTC: {_utc_now()}",
        f"Status: {'PASS' if decision_pass == 1 else 'FAIL'}",
        f"Symbol: {symbol}",
        f"Signals used: {len(signals_df)}",
        f"Coarse configs evaluated: {len(coarse_df)}",
        f"Refine configs evaluated: {len(refine_df)}",
        f"delta_expectancy_best_exit_minus_baseline_exit: {float(ov_row['delta_expectancy_best_exit_minus_baseline_exit']):.6f}",
        f"delta_maxdd_best_exit_minus_baseline_exit: {float(ov_row['delta_maxdd_best_exit_minus_baseline_exit']):.6f}",
        f"delta_cvar5_best_exit_minus_baseline_exit: {float(ov_row['delta_cvar5_best_exit_minus_baseline_exit']):.6f}",
        f"Repro check pass: {repro_pass}",
        "No Phase D executed.",
    ]
    (run_root / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    # Manifest and snapshots.
    snap = run_root / "config_snapshot"
    ecfg = _resolve(args.execution_config)
    if ecfg.exists():
        shutil.copy2(ecfg, snap / ecfg.name)
    shutil.copy2(signals_src, snap / Path(signals_src).name)
    shutil.copy2(metrics_src, snap / metrics_src.name)
    shutil.copy2(fee_src, snap / fee_src.name)
    _write_git_status(run_root / "git_status.txt")

    manifest = {
        "generated_utc": _utc_now(),
        "symbol": symbol,
        "phase_a_contract": {
            "dir": str(phase_a_dir),
            "metrics_definition_path": str(metrics_src),
            "metrics_definition_sha256": _sha256_file(metrics_src),
            "fee_model_path": str(fee_src),
            "fee_model_sha256": _sha256_file(fee_src),
        },
        "signal_source_path": str(signals_src),
        "signal_subset_path": str(subset_csv),
        "signal_subset_hash": str(subset_hash),
        "split_definition_path": str(split_json),
        "split_definition_sha256": str(split_hash),
        "coarse_seed": int(args.coarse_seed),
        "refine_seed": int(args.refine_seed),
        "coarse_run_dir": str(coarse_dir),
        "diagnostics_dir": str(diag_dir),
        "refine_run_dir": str(refine_dir),
        "final_selected_cfg_hash": _cfg_hash(best_cfg),
        "final_selected_cfg": asdict(best_cfg),
        "sampling": {
            "coarse": coarse_sampling_meta,
            "refine": refine_meta,
        },
        "eval_meta": {
            "coarse": coarse_eval_meta,
            "refine": refine_eval_meta,
        },
        "repro_check": {
            "same_coarse_sample": int(same_coarse_sample),
            "best_rerun_expectancy_abs_delta": float(repro_delta),
            "pass": int(repro_pass),
        },
    }
    _json_dump(run_root / "run_manifest.json", manifest)

    print(str(run_root))
    print(str(coarse_results_csv))
    print(str(refine_results_csv))
    print(str(run_root / "decision.md"))
    return run_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOLUSDT-only Phase C decision-grade rerun (coarse -> refine + diagnostics).")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--signals-dir", default="data/signals")
    ap.add_argument("--signals-csv", default="")
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--max-signals", type=int, default=2000)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--wf-splits", type=int, default=5)
    ap.add_argument("--mode", choices=["tight", "normal"], default="tight")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")
    ap.add_argument("--outdir", default="reports/execution_layer")

    ap.add_argument("--tp-mults", default="0.7,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--sl-mults", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--time-stops-min", default="360,720,1440,2160,2880")
    ap.add_argument("--break-even-enabled", default="0,1")
    ap.add_argument("--break-even-trigger-r", default="0.5,0.75,1.0")
    ap.add_argument("--break-even-offset-bps", default="0,2,5")
    ap.add_argument("--partial-take-enabled", default="0,1")
    ap.add_argument("--partial-take-r", default="0.6,0.8,1.0")
    ap.add_argument("--partial-take-pct", default="0.25,0.5")

    ap.add_argument("--coarse-max-configs", type=int, default=1200)
    ap.add_argument("--refine-max-configs", type=int, default=1500)
    ap.add_argument("--coarse-seed", type=int, default=42)
    ap.add_argument("--refine-seed", type=int, default=314159)
    ap.add_argument("--topk", type=int, default=50)

    ap.add_argument("--expectancy-epsilon", type=float, default=5e-5)
    ap.add_argument("--maxdd-worse-tol", type=float, default=0.02)
    ap.add_argument("--cvar-worse-tol", type=float, default=1e-4)
    ap.add_argument("--max-missing-slice-rate", type=float, default=0.02)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
