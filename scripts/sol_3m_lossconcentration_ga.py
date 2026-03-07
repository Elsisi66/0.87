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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1 as subset1  # noqa: E402


RUN_PREFIX = "SOL_3M_LOSSCONC_GA"
SYMBOL = "SOLUSDT"
BASELINE_STRATEGY_ID = "M1_ENTRY_ONLY_PASSIVE_BASELINE"
MEANINGFUL_WINNER_THRESHOLD = 0.0020


@dataclass
class CandidateEval:
    cfg: Dict[str, Any]
    genome_hash: str
    rows_df: pd.DataFrame
    metrics: Dict[str, Any]
    loss: Dict[str, Any]
    chrono: Dict[str, Any]
    compare_vs_baseline: Dict[str, Any]
    gates: Dict[str, Any]
    objective: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = to_num(pd.Series([x])).iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (tuple, set)):
            return list(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 20) -> str:
    if df.empty:
        return "_(none)_"
    use = [c for c in cols if c in df.columns]
    if not use:
        return "_(none)_"
    x = df.loc[:, use].head(n).copy()
    out = ["| " + " | ".join(x.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def find_latest_complete(root: Path, pattern: str, required: Iterable[str]) -> Optional[Path]:
    cands = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    return None


def parse_semicolon_records(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def sha256_text(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def genome_hash(cfg: Dict[str, Any]) -> str:
    return sha256_text(json.dumps(cfg, sort_keys=True, separators=(",", ":")))[:24]


def is_stop_reason(reason: Any) -> bool:
    r = str(reason).strip().lower()
    return ("sl" in r) or ("stop" in r)


def normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    mode = str(out.get("entry_mode", "limit")).strip().lower()
    if mode not in {"limit", "market"}:
        mode = "limit"
    out["entry_mode"] = mode
    out["limit_offset_bps"] = float(np.clip(float(out.get("limit_offset_bps", 0.75)), 0.0, 3.0))
    out["fallback_to_market"] = int(1 if int(out.get("fallback_to_market", 1)) == 1 else 0)
    out["fallback_delay_min"] = float(np.clip(float(out.get("fallback_delay_min", 6.0)), 0.0, 60.0))
    out["max_fill_delay_min"] = float(np.clip(float(out.get("max_fill_delay_min", 24.0)), 0.0, 60.0))
    if mode == "market":
        out["limit_offset_bps"] = 0.0
        out["fallback_to_market"] = 0
        out["fallback_delay_min"] = 0.0
        out["max_fill_delay_min"] = 0.0
    else:
        if out["fallback_to_market"] == 0:
            out["fallback_delay_min"] = 0.0
        if out["fallback_delay_min"] > out["max_fill_delay_min"]:
            out["fallback_delay_min"] = float(out["max_fill_delay_min"])
        if out["max_fill_delay_min"] < 3.0:
            out["max_fill_delay_min"] = 3.0
    if not str(out.get("candidate_id", "")).strip():
        out["candidate_id"] = f"GA_{genome_hash(out)}"
    if not str(out.get("label", "")).strip():
        out["label"] = "GA candidate"
    return out


def random_cfg(rng: random.Random) -> Dict[str, Any]:
    mode = "limit" if rng.random() < 0.88 else "market"
    if mode == "market":
        cfg = {
            "candidate_id": "",
            "label": "GA random",
            "entry_mode": "market",
            "limit_offset_bps": 0.0,
            "fallback_to_market": 0,
            "fallback_delay_min": 0.0,
            "max_fill_delay_min": 0.0,
        }
        return normalize_cfg(cfg)
    max_fill = rng.uniform(6.0, 48.0)
    fallback_to_market = 1 if rng.random() < 0.92 else 0
    fallback_delay = rng.uniform(0.0, max_fill) if fallback_to_market == 1 else 0.0
    cfg = {
        "candidate_id": "",
        "label": "GA random",
        "entry_mode": "limit",
        "limit_offset_bps": rng.uniform(0.15, 2.25),
        "fallback_to_market": fallback_to_market,
        "fallback_delay_min": fallback_delay,
        "max_fill_delay_min": max_fill,
    }
    return normalize_cfg(cfg)


def mutate_cfg(parent: Dict[str, Any], rng: random.Random, mutation_rate: float) -> Dict[str, Any]:
    c = dict(parent)
    if rng.random() < mutation_rate * 0.25:
        c["entry_mode"] = "market" if str(c.get("entry_mode", "limit")) == "limit" else "limit"
    if str(c.get("entry_mode", "limit")) == "limit":
        if rng.random() < mutation_rate:
            c["limit_offset_bps"] = float(c.get("limit_offset_bps", 0.75)) + rng.gauss(0.0, 0.35)
        if rng.random() < mutation_rate:
            c["max_fill_delay_min"] = float(c.get("max_fill_delay_min", 24.0)) + rng.gauss(0.0, 6.0)
        if rng.random() < mutation_rate:
            c["fallback_to_market"] = 1 if int(c.get("fallback_to_market", 1)) == 0 else 0
        if rng.random() < mutation_rate:
            c["fallback_delay_min"] = float(c.get("fallback_delay_min", 6.0)) + rng.gauss(0.0, 4.0)
    c["candidate_id"] = ""
    c["label"] = "GA mutated"
    return normalize_cfg(c)


def crossover_cfg(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ["entry_mode", "limit_offset_bps", "fallback_to_market", "fallback_delay_min", "max_fill_delay_min"]:
        if rng.random() < 0.5:
            out[k] = a.get(k)
        else:
            out[k] = b.get(k)
    out["candidate_id"] = ""
    out["label"] = "GA crossover"
    return normalize_cfg(out)


def subset_by_signal_ids(df: pd.DataFrame, signal_ids: Sequence[str]) -> pd.DataFrame:
    keep = set(str(x) for x in signal_ids)
    return df[df["signal_id"].astype(str).isin(keep)].copy().reset_index(drop=True)


def build_subbundle(base_bundle: modela.ga_exec.SymbolBundle, start: int, end: int, exec_args: argparse.Namespace) -> modela.ga_exec.SymbolBundle:
    contexts = list(base_bundle.contexts[int(start):int(end)])
    splits = modela.ga_exec._build_walkforward_splits(  # pylint: disable=protected-access
        n=len(contexts),
        train_ratio=float(exec_args.train_ratio),
        n_splits=int(exec_args.wf_splits),
    )
    return modela.ga_exec.SymbolBundle(
        symbol=base_bundle.symbol,
        signals_csv=base_bundle.signals_csv,
        contexts=contexts,
        splits=splits,
        constraints=base_bundle.constraints,
    )


def hold_minutes_from_times(entry: Any, exit_: Any) -> float:
    et = pd.to_datetime(entry, utc=True, errors="coerce")
    xt = pd.to_datetime(exit_, utc=True, errors="coerce")
    if pd.isna(et) or pd.isna(xt):
        return float("nan")
    return float((xt - et).total_seconds() / 60.0)


def compare_candidate_vs_baseline(candidate_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, float]:
    cand = modela.ga_exec._rollup_mode(candidate_df, "exec")  # pylint: disable=protected-access
    base = modela.ga_exec._rollup_mode(baseline_df, "exec")  # pylint: disable=protected-access
    baseline_dd = float(base["max_drawdown"])
    candidate_dd = float(cand["max_drawdown"])
    baseline_cvar = float(base["cvar_5"])
    candidate_cvar = float(cand["cvar_5"])
    maxdd_improve_ratio = (
        float((abs(baseline_dd) - abs(candidate_dd)) / abs(baseline_dd))
        if np.isfinite(baseline_dd) and abs(baseline_dd) > 1e-12 and np.isfinite(candidate_dd)
        else float("nan")
    )
    cvar_improve_ratio = (
        float((abs(baseline_cvar) - abs(candidate_cvar)) / abs(baseline_cvar))
        if np.isfinite(baseline_cvar) and abs(baseline_cvar) > 1e-12 and np.isfinite(candidate_cvar)
        else float("nan")
    )
    return {
        "baseline_expectancy_net": float(base["mean_expectancy_net"]),
        "candidate_expectancy_net": float(cand["mean_expectancy_net"]),
        "delta_expectancy": float(cand["mean_expectancy_net"] - base["mean_expectancy_net"]),
        "baseline_cvar_5": baseline_cvar,
        "candidate_cvar_5": candidate_cvar,
        "delta_cvar_5": float(candidate_cvar - baseline_cvar),
        "baseline_max_drawdown": baseline_dd,
        "candidate_max_drawdown": candidate_dd,
        "delta_max_drawdown": float(candidate_dd - baseline_dd),
        "maxdd_improve_ratio": float(maxdd_improve_ratio),
        "cvar_improve_ratio": float(cvar_improve_ratio),
        "baseline_entries_valid": int(base["entries_valid"]),
        "candidate_entries_valid": int(cand["entries_valid"]),
        "candidate_entry_rate": float(cand["entry_rate"]),
        "baseline_entry_rate": float(base["entry_rate"]),
        "candidate_taker_share": float(cand["taker_share"]),
        "baseline_taker_share": float(base["taker_share"]),
        "candidate_median_fill_delay_min": float(cand["median_fill_delay_min"]),
        "baseline_median_fill_delay_min": float(base["median_fill_delay_min"]),
        "candidate_p95_fill_delay_min": float(cand["p95_fill_delay_min"]),
        "baseline_p95_fill_delay_min": float(base["p95_fill_delay_min"]),
    }


def compute_loss_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    filled = to_num(df.get("exec_filled", 0)).fillna(0).astype(int)
    valid = to_num(df.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    pnl = to_num(df.get("exec_pnl_net_pct", np.nan))
    reason = df.get("exec_exit_reason", pd.Series(dtype=object)).fillna("").astype(str)
    hold = pd.Series(
        [hold_minutes_from_times(e, x) for e, x in zip(df.get("exec_entry_time", ""), df.get("exec_exit_time", ""))],
        index=df.index,
        dtype=float,
    )
    m = (filled == 1) & (valid == 1) & pnl.notna()
    pnl_v = pnl.loc[m].astype(float)
    reason_v = reason.loc[m]
    hold_v = hold.loc[m]
    n = int(len(pnl_v))
    if n <= 0:
        return {
            "trade_count": 0,
            "instant_loser_count": 0,
            "fast_loser_count": 0,
            "meaningful_winner_count": 0,
            "instant_loser_rate": float("nan"),
            "fast_loser_rate": float("nan"),
            "bottom_decile_pnl_share": float("nan"),
            "worst_10_trades_sum": float("nan"),
            "worst_25_trades_sum": float("nan"),
        }

    stop_mask = reason_v.map(is_stop_reason)
    instant_mask = stop_mask & (hold_v <= 60.0)
    fast_mask = stop_mask & (hold_v > 60.0) & (hold_v <= 240.0)
    meaningful_mask = pnl_v >= float(MEANINGFUL_WINNER_THRESHOLD)

    sorted_pnl = np.sort(pnl_v.to_numpy(dtype=float))
    worst_10 = float(np.sum(sorted_pnl[: min(10, n)]))
    worst_25 = float(np.sum(sorted_pnl[: min(25, n)]))
    k = max(1, int(math.ceil(0.10 * n)))
    worst_decile = sorted_pnl[:k]
    total_neg = float(abs(np.sum(pnl_v[pnl_v < 0.0])))
    decile_neg = float(abs(np.sum(worst_decile[worst_decile < 0.0])))
    bottom_decile_share = float(decile_neg / total_neg) if total_neg > 1e-12 else 0.0

    return {
        "trade_count": int(n),
        "instant_loser_count": int(instant_mask.sum()),
        "fast_loser_count": int(fast_mask.sum()),
        "meaningful_winner_count": int(meaningful_mask.sum()),
        "instant_loser_rate": float(instant_mask.sum() / max(1, n)),
        "fast_loser_rate": float(fast_mask.sum() / max(1, n)),
        "bottom_decile_pnl_share": float(bottom_decile_share),
        "worst_10_trades_sum": float(worst_10),
        "worst_25_trades_sum": float(worst_25),
    }


def compute_chronology_stats(df: pd.DataFrame) -> Dict[str, Any]:
    sig_t = pd.to_datetime(df.get("signal_time", ""), utc=True, errors="coerce")
    ent_t = pd.to_datetime(df.get("exec_entry_time", ""), utc=True, errors="coerce")
    ext_t = pd.to_datetime(df.get("exec_exit_time", ""), utc=True, errors="coerce")
    filled = to_num(df.get("exec_filled", 0)).fillna(0).astype(int) == 1
    valid = to_num(df.get("exec_valid_for_metrics", 0)).fillna(0).astype(int) == 1
    m = filled & valid & ent_t.notna() & ext_t.notna()
    entry_parent = ent_t.dt.floor("1h")
    exit_parent = ext_t.dt.floor("1h")
    same_parent_exit = int((m & (entry_parent == exit_parent)).sum())
    # Keep this as a diagnostic: it means both TP and SL were touched in one exit bar,
    # not that entry and exit occurred in the same parent bar.
    same_bar_touch = int(to_num(df.get("exec_same_bar_hit", 0)).fillna(0).astype(int).sum())
    invalid_stop = int(to_num(df.get("exec_invalid_stop_geometry", 0)).fillna(0).astype(int).sum())
    invalid_tp = int(to_num(df.get("exec_invalid_tp_geometry", 0)).fillna(0).astype(int).sum())
    lookahead = int(to_num(df.get("lookahead_violation", 0)).fillna(0).astype(int).sum())
    # `signal_time` is a decision timestamp; entry on exactly that timestamp can be a
    # boundary-aligned next-bar open. Pre-signal entries are true chronology violations.
    entry_on_signal = int(((ent_t.notna()) & (sig_t.notna()) & (ent_t < sig_t)).sum())
    exit_before_entry = int(((ent_t.notna()) & (ext_t.notna()) & (ext_t <= ent_t)).sum())
    parity_clean = int(
        same_parent_exit == 0
        and exit_before_entry == 0
        and entry_on_signal == 0
        and invalid_stop == 0
        and invalid_tp == 0
        and lookahead == 0
    )
    return {
        "parity_clean": int(parity_clean),
        "same_bar_exit_count": int(same_parent_exit),
        "same_bar_touch_count": int(same_bar_touch),
        "exit_before_entry_count": int(exit_before_entry),
        "entry_on_signal_count": int(entry_on_signal),
        "invalid_stop_geometry_count": int(invalid_stop),
        "invalid_tp_geometry_count": int(invalid_tp),
        "lookahead_violation_count": int(lookahead),
    }


def relative_reduction(base: float, cand: float) -> float:
    if not np.isfinite(base) or base <= 1e-12:
        return 0.0
    if not np.isfinite(cand):
        return float("nan")
    return float((base - cand) / base)


def compute_stage_gates(
    *,
    compare: Dict[str, Any],
    baseline_loss: Dict[str, Any],
    candidate_loss: Dict[str, Any],
    chrono: Dict[str, Any],
    retention_floor: float,
    min_trades_abs: int,
    min_trades_frac: float,
    winner_retention_floor: float,
    instant_reduction_rel: float,
    tail_reduction_rel: float,
) -> Dict[str, Any]:
    baseline_trade_count = int(baseline_loss.get("trade_count", 0))
    candidate_trade_count = int(candidate_loss.get("trade_count", 0))
    retention = float(candidate_trade_count / max(1, baseline_trade_count))
    min_trades_req = int(max(int(min_trades_abs), int(math.ceil(float(min_trades_frac) * max(1, baseline_trade_count)))))
    baseline_winners = int(baseline_loss.get("meaningful_winner_count", 0))
    candidate_winners = int(candidate_loss.get("meaningful_winner_count", 0))
    winner_retention = float(candidate_winners / max(1, baseline_winners))
    maxdd_improve_ratio = finite_float(compare.get("maxdd_improve_ratio", np.nan))
    instant_rel_red = relative_reduction(
        float(baseline_loss.get("instant_loser_rate", np.nan)),
        float(candidate_loss.get("instant_loser_rate", np.nan)),
    )
    tail_rel_red = relative_reduction(
        float(baseline_loss.get("bottom_decile_pnl_share", np.nan)),
        float(candidate_loss.get("bottom_decile_pnl_share", np.nan)),
    )
    improve_target_pass = int(
        (np.isfinite(instant_rel_red) and instant_rel_red >= float(instant_reduction_rel))
        or (np.isfinite(tail_rel_red) and tail_rel_red >= float(tail_reduction_rel))
    )

    g0 = int(
        int(chrono.get("parity_clean", 0)) == 1
        and int(chrono.get("same_bar_exit_count", 0)) == 0
        and int(chrono.get("exit_before_entry_count", 0)) == 0
    )
    g1 = int(retention >= float(retention_floor) and candidate_trade_count >= int(min_trades_req))
    g2 = int(winner_retention >= float(winner_retention_floor))
    g3 = int(np.isfinite(maxdd_improve_ratio) and maxdd_improve_ratio >= 0.0)
    pass_core = int(g0 == 1 and g1 == 1 and g2 == 1 and g3 == 1 and improve_target_pass == 1)
    participation_penalty = max(0.0, float(retention_floor) - retention)
    objective = (
        420.0 * float(compare.get("delta_expectancy", np.nan) if np.isfinite(compare.get("delta_expectancy", np.nan)) else -1e9)
        + 110.0 * float(compare.get("delta_cvar_5", np.nan) if np.isfinite(compare.get("delta_cvar_5", np.nan)) else -1e9)
        + 80.0 * float(maxdd_improve_ratio if np.isfinite(maxdd_improve_ratio) else -1e9)
        + 2.0 * float(instant_rel_red if np.isfinite(instant_rel_red) else -1e9)
        + 2.0 * float(tail_rel_red if np.isfinite(tail_rel_red) else -1e9)
        - 5.0 * float(participation_penalty)
    )
    return {
        "gate_g0_chronology": int(g0),
        "gate_g1_participation": int(g1),
        "gate_g2_winner_preservation": int(g2),
        "gate_g3_risk_sanity": int(g3),
        "gate_improve_target": int(improve_target_pass),
        "gate_pass_core": int(pass_core),
        "baseline_trade_count": int(baseline_trade_count),
        "candidate_trade_count": int(candidate_trade_count),
        "min_trade_count_required": int(min_trades_req),
        "retention": float(retention),
        "baseline_winner_count": int(baseline_winners),
        "candidate_winner_count": int(candidate_winners),
        "winner_retention": float(winner_retention),
        "maxdd_improve_ratio": float(maxdd_improve_ratio),
        "instant_loser_rel_reduction": float(instant_rel_red),
        "bottom_decile_rel_reduction": float(tail_rel_red),
        "participation_penalty": float(participation_penalty),
        "objective_core": float(objective),
    }


def evaluate_cfg_on_bundle(
    *,
    bundle: modela.ga_exec.SymbolBundle,
    baseline_1h_df: pd.DataFrame,
    cfg: Dict[str, Any],
    one_h: modela.OneHMarket,
    exec_args: argparse.Namespace,
) -> Dict[str, Any]:
    return modela.evaluate_model_a_variant(
        bundle=bundle,
        baseline_df=baseline_1h_df,
        cfg=cfg,
        one_h=one_h,
        args=exec_args,
    )


def build_route_family(
    *,
    base_bundle: modela.ga_exec.SymbolBundle,
    exec_args: argparse.Namespace,
    fee: modela.phasec_bt.FeeModel,
) -> Tuple[Dict[str, modela.ga_exec.SymbolBundle], Dict[str, pd.DataFrame], Dict[str, Any]]:
    route_bundles, route_examples_df, route_feas_df, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=base_bundle,
        args=exec_args,
        coverage_frac=0.60,
    )
    baseline_1h_by_route: Dict[str, pd.DataFrame] = {}
    for rid, rb in route_bundles.items():
        baseline_1h_by_route[rid] = modela.build_1h_reference_rows(
            bundle=rb,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )
    meta = {
        "route_meta": dict(route_meta),
        "route_examples_count": int(len(route_examples_df)),
        "route_feasibility_count": int(len(route_feas_df)),
        "route_ids": sorted([str(x) for x in route_bundles.keys()]),
    }
    return route_bundles, baseline_1h_by_route, meta


def route_confirm_for_candidate(
    *,
    cfg: Dict[str, Any],
    one_h: modela.OneHMarket,
    exec_args: argparse.Namespace,
    route_bundles: Dict[str, modela.ga_exec.SymbolBundle],
    route_baseline_1h: Dict[str, pd.DataFrame],
    route_baseline_exec_rows: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    confirm_count = 0
    total = 0
    for rid, rb in route_bundles.items():
        total += 1
        cand_eval = evaluate_cfg_on_bundle(
            bundle=rb,
            baseline_1h_df=route_baseline_1h[rid],
            cfg=cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        cand_rows = cand_eval["signal_rows_df"].copy()
        base_rows = route_baseline_exec_rows[rid]
        cmp = compare_candidate_vs_baseline(cand_rows, base_rows)
        valid = int(cand_eval["metrics"]["valid_for_ranking"])
        confirm_flag = int(
            valid == 1
            and np.isfinite(cmp["delta_expectancy"])
            and cmp["delta_expectancy"] > 0.0
            and np.isfinite(cmp["delta_cvar_5"])
            and cmp["delta_cvar_5"] >= 0.0
            and np.isfinite(cmp["maxdd_improve_ratio"])
            and cmp["maxdd_improve_ratio"] >= 0.0
        )
        confirm_count += int(confirm_flag)
        rows.append(
            {
                "route_id": str(rid),
                "candidate_valid_for_ranking": int(valid),
                "delta_expectancy_vs_baseline": float(cmp["delta_expectancy"]),
                "delta_cvar_vs_baseline": float(cmp["delta_cvar_5"]),
                "maxdd_improve_ratio": float(cmp["maxdd_improve_ratio"]),
                "confirm_flag": int(confirm_flag),
            }
        )
    route_pass_rate = float(confirm_count / max(1, total))
    return {
        "route_total": int(total),
        "route_confirm_count": int(confirm_count),
        "route_pass_rate": float(route_pass_rate),
        "route_support_pass": int(total > 0 and confirm_count == total),
        "route_detail_json": json.dumps(rows, sort_keys=True),
    }


def choose_parents(pool: List[CandidateEval], rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not pool:
        raise RuntimeError("Parent pool is empty")
    top = pool[: max(2, min(8, len(pool)))]
    p1 = rng.choice(top)
    p2 = rng.choice(top)
    return dict(p1.cfg), dict(p2.cfg)


def baseline_hash_payload(
    *,
    baseline_cfg: Dict[str, Any],
    params_payload: Dict[str, Any],
    posture_dir: Path,
    freeze_dir: Path,
    foundation_dir: Path,
) -> str:
    payload = {
        "baseline_cfg": baseline_cfg,
        "params_payload": params_payload,
        "posture_dir": str(posture_dir),
        "freeze_dir": str(freeze_dir),
        "foundation_dir": str(foundation_dir),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOL-only 3m loss-concentration GA under repaired frozen posture")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--posture-dir", default="")
    ap.add_argument("--strict-confirm-dir", default="")
    ap.add_argument("--seed", type=int, default=20260306)
    ap.add_argument("--pop-size", type=int, default=18)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--elite-k", type=int, default=6)
    ap.add_argument("--mutation-rate", type=float, default=0.35)
    ap.add_argument("--top-confirm", type=int, default=12)
    ap.add_argument("--retention-floor", type=float, default=0.90)
    ap.add_argument("--min-trades-abs", type=int, default=1000)
    ap.add_argument("--min-trades-frac", type=float, default=0.80)
    ap.add_argument("--winner-retention-floor", type=float, default=0.98)
    ap.add_argument("--instant-reduction-rel", type=float, default=0.05)
    ap.add_argument("--tail-reduction-rel", type=float, default=0.15)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(int(args.seed))

    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    posture_dir = (
        Path(args.posture_dir).resolve()
        if str(args.posture_dir).strip()
        else find_latest_complete(
            exec_root,
            "REPAIRED_BRANCH_3M_POSTURE_FREEZE_*",
            [
                "repaired_active_3m_subset.csv",
                "repaired_active_3m_params",
                "repaired_3m_posture_table.csv",
                "repaired_3m_posture_manifest.json",
            ],
        )
    )
    if posture_dir is None:
        raise FileNotFoundError("Missing completed REPAIRED_BRANCH_3M_POSTURE_FREEZE_* directory")

    strict_confirm_dir = (
        Path(args.strict_confirm_dir).resolve()
        if str(args.strict_confirm_dir).strip()
        else find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_*",
            [
                "repaired_subset1_confirm_by_symbol.csv",
                "repaired_subset1_confirm_manifest.json",
            ],
        )
    )
    if strict_confirm_dir is None:
        raise FileNotFoundError("Missing completed REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_* directory")

    posture_manifest = json.loads((posture_dir / "repaired_3m_posture_manifest.json").read_text(encoding="utf-8"))
    strict_manifest = json.loads((strict_confirm_dir / "repaired_subset1_confirm_manifest.json").read_text(encoding="utf-8"))

    freeze_dir = Path(
        posture_manifest.get("source_artifacts", {}).get("frozen_repaired_universe_dir", strict_manifest.get("freeze_dir", ""))
    ).resolve()
    if not freeze_dir.exists():
        raise FileNotFoundError(f"Freeze directory from manifest does not exist: {freeze_dir}")
    foundation_dir = Path(strict_manifest.get("foundation_dir", "")).resolve()
    if not foundation_dir.exists():
        raise FileNotFoundError(f"Foundation directory from strict confirm manifest does not exist: {foundation_dir}")

    posture_active_df = pd.read_csv(posture_dir / "repaired_active_3m_subset.csv")
    posture_table_df = pd.read_csv(posture_dir / "repaired_3m_posture_table.csv")
    posture_active_df["symbol"] = posture_active_df["symbol"].astype(str).str.upper()
    posture_table_df["symbol"] = posture_table_df["symbol"].astype(str).str.upper()
    sol_active = posture_active_df[posture_active_df["symbol"] == SYMBOL].copy()
    if sol_active.empty:
        raise RuntimeError("SOLUSDT is not present in repaired active subset")
    if int(len(posture_active_df)) != 1:
        raise RuntimeError(f"Active subset is not SOL-only: {posture_active_df['symbol'].tolist()}")
    sol_winner_id = str(sol_active.iloc[0]["winner_config_id"]).strip()
    if sol_winner_id != BASELINE_STRATEGY_ID:
        raise RuntimeError(f"Baseline strategy mismatch. Expected {BASELINE_STRATEGY_ID}, got {sol_winner_id}")

    freeze_df = pd.read_csv(freeze_dir / "repaired_best_by_symbol.csv")
    freeze_df["symbol"] = freeze_df["symbol"].astype(str).str.upper()
    sol_row_df = freeze_df[(freeze_df["symbol"] == SYMBOL) & (freeze_df["side"].astype(str).str.lower() == "long")].copy()
    if sol_row_df.empty:
        raise RuntimeError("SOLUSDT long row missing in repaired_best_by_symbol.csv")
    sol_row = sol_row_df.iloc[0]
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    params_payload = subset1.parse_params_from_row(sol_row, selected_params_dir)

    run_dir = ensure_dir((PROJECT_ROOT / args.outdir).resolve() / f"{RUN_PREFIX}_{utc_tag()}")
    inputs_dir = ensure_dir(run_dir / "_inputs")
    cache_dir = ensure_dir(run_dir / "_window_cache")
    signal_dir = ensure_dir(run_dir / "_signal_inputs")

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_df = subset1.build_signal_table_for_row(
        row=sol_row,
        params=params_payload,
        df_cache=df_cache,
        raw_cache=raw_cache,
    )
    if signal_df.empty:
        raise RuntimeError("No rebuilt repaired 1h signals for SOLUSDT")
    signal_df.to_csv(inputs_dir / "sol_signal_timeline.csv", index=False)
    signal_df.to_csv(inputs_dir / "universe_signal_timeline.csv", index=False)

    symbol_windows = subset1.build_window_pool_for_symbol(
        symbol=SYMBOL,
        signal_df=signal_df,
        foundation_state=foundation_state,
        cache_dir=cache_dir,
    )
    if symbol_windows.empty:
        raise RuntimeError("No usable 3m windows for SOLUSDT")

    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=signal_df.copy(),
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(args.seed),
    )
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    if int(contract_validation.get("wrapper_uses_1h_exit_owner", 0)) != 1:
        raise RuntimeError("Contract validation failed: 1h exit ownership is not locked")

    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=signal_df.copy(),
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    n_total = int(len(bundle.contexts))
    if n_total < 500:
        raise RuntimeError(f"Insufficient contexts for SOLUSDT GA ({n_total})")
    train_end = int(math.floor(0.60 * n_total))
    holdout_start = int(math.floor(0.80 * n_total))
    train_bundle = build_subbundle(bundle, 0, train_end, exec_args)
    holdout_signal_ids = [str(ctx.signal_id) for ctx in bundle.contexts[holdout_start:]]
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )
    one_h = modela.load_1h_market(SYMBOL)

    baseline_1h_full = modela.build_1h_reference_rows(bundle=bundle, fee=fee, exec_horizon_hours=float(exec_args.exec_horizon_hours))
    baseline_1h_train = modela.build_1h_reference_rows(bundle=train_bundle, fee=fee, exec_horizon_hours=float(exec_args.exec_horizon_hours))
    variants = phase_v.sanitize_variants()
    variant_map = {str(v["candidate_id"]): dict(v) for v in variants}
    if BASELINE_STRATEGY_ID not in variant_map:
        raise RuntimeError(f"Baseline variant {BASELINE_STRATEGY_ID} missing from sanitized variants")
    baseline_cfg = normalize_cfg(dict(variant_map[BASELINE_STRATEGY_ID]))
    baseline_cfg["candidate_id"] = BASELINE_STRATEGY_ID
    baseline_cfg["label"] = "Frozen baseline"

    baseline_train_eval = evaluate_cfg_on_bundle(
        bundle=train_bundle,
        baseline_1h_df=baseline_1h_train,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )
    baseline_full_eval = evaluate_cfg_on_bundle(
        bundle=bundle,
        baseline_1h_df=baseline_1h_full,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )

    baseline_train_rows = baseline_train_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_full_rows = baseline_full_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_train_loss = compute_loss_metrics(baseline_train_rows)
    baseline_full_loss = compute_loss_metrics(baseline_full_rows)
    baseline_train_chrono = compute_chronology_stats(baseline_train_rows)
    baseline_full_chrono = compute_chronology_stats(baseline_full_rows)
    baseline_train_cmp = compare_candidate_vs_baseline(baseline_train_rows, baseline_train_rows)
    baseline_full_cmp = compare_candidate_vs_baseline(baseline_full_rows, baseline_full_rows)

    route_bundles, route_baseline_1h, route_meta = build_route_family(
        base_bundle=bundle,
        exec_args=exec_args,
        fee=fee,
    )
    route_baseline_exec_rows: Dict[str, pd.DataFrame] = {}
    for rid, rb in route_bundles.items():
        route_baseline_eval = evaluate_cfg_on_bundle(
            bundle=rb,
            baseline_1h_df=route_baseline_1h[rid],
            cfg=baseline_cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        route_baseline_exec_rows[rid] = route_baseline_eval["signal_rows_df"].copy().reset_index(drop=True)

    screen_cache: Dict[str, CandidateEval] = {}
    full_cache: Dict[str, Dict[str, Any]] = {}
    screen_rows: List[Dict[str, Any]] = []

    def screen_candidate(cfg_in: Dict[str, Any], generation: int) -> CandidateEval:
        cfg = normalize_cfg(cfg_in)
        h = genome_hash(cfg)
        if h in screen_cache:
            ev = screen_cache[h]
            row_cached = {
                "stage": "screen_train60",
                "generation": int(generation),
                "candidate_id": str(cfg["candidate_id"]),
                "genome_hash": str(h),
                "entry_mode": str(cfg["entry_mode"]),
                "limit_offset_bps": float(cfg["limit_offset_bps"]),
                "fallback_to_market": int(cfg["fallback_to_market"]),
                "fallback_delay_min": float(cfg["fallback_delay_min"]),
                "max_fill_delay_min": float(cfg["max_fill_delay_min"]),
                "screen_pass": int(ev.gates["gate_pass_core"]),
                "screen_objective": float(ev.objective),
                "gate_g0_chronology": int(ev.gates["gate_g0_chronology"]),
                "gate_g1_participation": int(ev.gates["gate_g1_participation"]),
                "gate_g2_winner_preservation": int(ev.gates["gate_g2_winner_preservation"]),
                "gate_g3_risk_sanity": int(ev.gates["gate_g3_risk_sanity"]),
                "gate_improve_target": int(ev.gates["gate_improve_target"]),
                "delta_expectancy_vs_baseline": float(ev.compare_vs_baseline["delta_expectancy"]),
                "delta_cvar_vs_baseline": float(ev.compare_vs_baseline["delta_cvar_5"]),
                "maxdd_improve_ratio": float(ev.compare_vs_baseline["maxdd_improve_ratio"]),
                "retention": float(ev.gates["retention"]),
                "trade_count": int(ev.loss["trade_count"]),
                "instant_loser_rate": float(ev.loss["instant_loser_rate"]),
                "fast_loser_rate": float(ev.loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(ev.loss["bottom_decile_pnl_share"]),
                "instant_loser_rel_reduction": float(ev.gates["instant_loser_rel_reduction"]),
                "bottom_decile_rel_reduction": float(ev.gates["bottom_decile_rel_reduction"]),
                "same_bar_exit_count": int(ev.chrono["same_bar_exit_count"]),
                "exit_before_entry_count": int(ev.chrono["exit_before_entry_count"]),
                "winner_retention": float(ev.gates["winner_retention"]),
            }
            screen_rows.append(row_cached)
            return ev

        eval_pack = evaluate_cfg_on_bundle(
            bundle=train_bundle,
            baseline_1h_df=baseline_1h_train,
            cfg=cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        rows_df = eval_pack["signal_rows_df"].copy().reset_index(drop=True)
        cmp = compare_candidate_vs_baseline(rows_df, baseline_train_rows)
        loss = compute_loss_metrics(rows_df)
        chrono = compute_chronology_stats(rows_df)
        gates = compute_stage_gates(
            compare=cmp,
            baseline_loss=baseline_train_loss,
            candidate_loss=loss,
            chrono=chrono,
            retention_floor=float(args.retention_floor),
            # Train-stage screen uses relative participation floor only.
            # Absolute 1000-trade gate is enforced at full confirmation stage.
            min_trades_abs=0,
            min_trades_frac=float(args.min_trades_frac),
            winner_retention_floor=float(args.winner_retention_floor),
            instant_reduction_rel=float(args.instant_reduction_rel),
            tail_reduction_rel=float(args.tail_reduction_rel),
        )
        ev = CandidateEval(
            cfg=cfg,
            genome_hash=h,
            rows_df=rows_df,
            metrics=dict(eval_pack["metrics"]),
            loss=loss,
            chrono=chrono,
            compare_vs_baseline=cmp,
            gates=gates,
            objective=float(gates["objective_core"]),
        )
        screen_cache[h] = ev
        screen_rows.append(
            {
                "stage": "screen_train60",
                "generation": int(generation),
                "candidate_id": str(cfg["candidate_id"]),
                "genome_hash": str(h),
                "entry_mode": str(cfg["entry_mode"]),
                "limit_offset_bps": float(cfg["limit_offset_bps"]),
                "fallback_to_market": int(cfg["fallback_to_market"]),
                "fallback_delay_min": float(cfg["fallback_delay_min"]),
                "max_fill_delay_min": float(cfg["max_fill_delay_min"]),
                "screen_pass": int(gates["gate_pass_core"]),
                "screen_objective": float(ev.objective),
                "gate_g0_chronology": int(gates["gate_g0_chronology"]),
                "gate_g1_participation": int(gates["gate_g1_participation"]),
                "gate_g2_winner_preservation": int(gates["gate_g2_winner_preservation"]),
                "gate_g3_risk_sanity": int(gates["gate_g3_risk_sanity"]),
                "gate_improve_target": int(gates["gate_improve_target"]),
                "delta_expectancy_vs_baseline": float(cmp["delta_expectancy"]),
                "delta_cvar_vs_baseline": float(cmp["delta_cvar_5"]),
                "maxdd_improve_ratio": float(cmp["maxdd_improve_ratio"]),
                "retention": float(gates["retention"]),
                "trade_count": int(loss["trade_count"]),
                "instant_loser_rate": float(loss["instant_loser_rate"]),
                "fast_loser_rate": float(loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(loss["bottom_decile_pnl_share"]),
                "worst_10_trades_sum": float(loss["worst_10_trades_sum"]),
                "worst_25_trades_sum": float(loss["worst_25_trades_sum"]),
                "instant_loser_rel_reduction": float(gates["instant_loser_rel_reduction"]),
                "bottom_decile_rel_reduction": float(gates["bottom_decile_rel_reduction"]),
                "same_bar_exit_count": int(chrono["same_bar_exit_count"]),
                "exit_before_entry_count": int(chrono["exit_before_entry_count"]),
                "parity_clean": int(chrono["parity_clean"]),
                "winner_retention": float(gates["winner_retention"]),
                "meaningful_winner_count": int(loss["meaningful_winner_count"]),
                "candidate_valid_for_ranking": int(eval_pack["metrics"]["valid_for_ranking"]),
                "candidate_min_split_delta_vs_1href": float(eval_pack["metrics"]["min_split_delta"]),
            }
        )
        return ev

    population: List[Dict[str, Any]] = [normalize_cfg(baseline_cfg)]
    while len(population) < int(args.pop_size):
        population.append(random_cfg(rng))

    for gen in range(int(args.generations)):
        evals: List[CandidateEval] = [screen_candidate(cfg, generation=gen) for cfg in population]
        evals_sorted = sorted(
            evals,
            key=lambda e: (int(e.gates["gate_pass_core"]), float(e.objective), float(e.compare_vs_baseline["delta_expectancy"])),
            reverse=True,
        )
        elites = evals_sorted[: max(1, min(int(args.elite_k), len(evals_sorted)))]
        next_pop: List[Dict[str, Any]] = [copy.deepcopy(e.cfg) for e in elites]
        while len(next_pop) < int(args.pop_size):
            p1, p2 = choose_parents(evals_sorted, rng)
            child = crossover_cfg(p1, p2, rng)
            if rng.random() < 0.90:
                child = mutate_cfg(child, rng, mutation_rate=float(args.mutation_rate))
            next_pop.append(child)
        population = [normalize_cfg(x) for x in next_pop]

    screen_df = pd.DataFrame(screen_rows)
    if screen_df.empty:
        raise RuntimeError("No screened candidates were produced")
    screen_df = screen_df.sort_values(
        ["screen_pass", "screen_objective", "delta_expectancy_vs_baseline", "generation"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    screen_df.to_csv(run_dir / "sol_3m_ga_candidates_screen.csv", index=False)

    # Build confirmation set from unique passing genomes plus baseline.
    pass_hashes = screen_df[screen_df["screen_pass"] == 1]["genome_hash"].astype(str).drop_duplicates().tolist()
    if not pass_hashes:
        pass_hashes = screen_df["genome_hash"].astype(str).drop_duplicates().head(int(args.top_confirm)).tolist()
    if baseline_cfg["candidate_id"] not in screen_df["candidate_id"].astype(str).tolist():
        screen_candidate(baseline_cfg, generation=int(args.generations))
    top_hashes = pass_hashes[: int(args.top_confirm)]
    baseline_hash = genome_hash(baseline_cfg)
    if baseline_hash not in top_hashes:
        top_hashes = [baseline_hash] + top_hashes
    unique_top_hashes: List[str] = []
    seen: set[str] = set()
    for h in top_hashes:
        if h in seen:
            continue
        seen.add(h)
        unique_top_hashes.append(h)
    top_hashes = unique_top_hashes

    confirm_rows: List[Dict[str, Any]] = []
    for h in top_hashes:
        if h not in screen_cache:
            continue
        cfg = dict(screen_cache[h].cfg)
        if h in full_cache:
            full_eval_pack = full_cache[h]
        else:
            full_eval = evaluate_cfg_on_bundle(
                bundle=bundle,
                baseline_1h_df=baseline_1h_full,
                cfg=cfg,
                one_h=one_h,
                exec_args=exec_args,
            )
            full_rows = full_eval["signal_rows_df"].copy().reset_index(drop=True)
            full_cmp = compare_candidate_vs_baseline(full_rows, baseline_full_rows)
            full_loss = compute_loss_metrics(full_rows)
            full_chrono = compute_chronology_stats(full_rows)
            full_gates = compute_stage_gates(
                compare=full_cmp,
                baseline_loss=baseline_full_loss,
                candidate_loss=full_loss,
                chrono=full_chrono,
                retention_floor=float(args.retention_floor),
                min_trades_abs=int(args.min_trades_abs),
                min_trades_frac=float(args.min_trades_frac),
                winner_retention_floor=float(args.winner_retention_floor),
                instant_reduction_rel=float(args.instant_reduction_rel),
                tail_reduction_rel=float(args.tail_reduction_rel),
            )
            holdout_cand = subset_by_signal_ids(full_rows, holdout_signal_ids)
            holdout_base = subset_by_signal_ids(baseline_full_rows, holdout_signal_ids)
            holdout_cmp = compare_candidate_vs_baseline(holdout_cand, holdout_base) if (not holdout_cand.empty and not holdout_base.empty) else {
                "delta_expectancy": float("nan"),
                "delta_cvar_5": float("nan"),
            }
            holdout_pass = int(
                np.isfinite(holdout_cmp.get("delta_expectancy", np.nan))
                and float(holdout_cmp["delta_expectancy"]) > 0.0
                and np.isfinite(holdout_cmp.get("delta_cvar_5", np.nan))
                and float(holdout_cmp["delta_cvar_5"]) >= 0.0
            )
            route_pack = route_confirm_for_candidate(
                cfg=cfg,
                one_h=one_h,
                exec_args=exec_args,
                route_bundles=route_bundles,
                route_baseline_1h=route_baseline_1h,
                route_baseline_exec_rows=route_baseline_exec_rows,
            )
            split_pass = int(
                int(full_eval["metrics"]["valid_for_ranking"]) == 1
                and np.isfinite(float(full_eval["metrics"]["min_split_delta"]))
                and float(full_eval["metrics"]["min_split_delta"]) > 0.0
            )
            g4 = int(split_pass == 1 and int(route_pack["route_support_pass"]) == 1 and holdout_pass == 1)
            confirm_pass = int(int(full_gates["gate_pass_core"]) == 1 and g4 == 1)
            final_objective = (
                float(full_gates["objective_core"])
                + 1.25 * float(holdout_cmp.get("delta_expectancy", np.nan) if np.isfinite(holdout_cmp.get("delta_expectancy", np.nan)) else -1e9)
                + 0.5 * float(route_pack["route_pass_rate"])
            )
            full_eval_pack = {
                "cfg": cfg,
                "rows_df": full_rows,
                "metrics": dict(full_eval["metrics"]),
                "compare": full_cmp,
                "loss": full_loss,
                "chrono": full_chrono,
                "gates_core": full_gates,
                "split_pass": int(split_pass),
                "holdout_cmp": holdout_cmp,
                "holdout_trade_count": int(len(holdout_cand)),
                "holdout_pass": int(holdout_pass),
                "route_pack": route_pack,
                "gate_g4_robustness": int(g4),
                "confirm_pass": int(confirm_pass),
                "final_objective": float(final_objective),
            }
            full_cache[h] = full_eval_pack

        cfg = dict(full_eval_pack["cfg"])
        c = full_eval_pack["compare"]
        l = full_eval_pack["loss"]
        g = full_eval_pack["gates_core"]
        ch = full_eval_pack["chrono"]
        rp = full_eval_pack["route_pack"]
        hm = full_eval_pack["holdout_cmp"]
        confirm_rows.append(
            {
                "stage": "confirm_full",
                "candidate_id": str(cfg["candidate_id"]),
                "genome_hash": str(h),
                "entry_mode": str(cfg["entry_mode"]),
                "limit_offset_bps": float(cfg["limit_offset_bps"]),
                "fallback_to_market": int(cfg["fallback_to_market"]),
                "fallback_delay_min": float(cfg["fallback_delay_min"]),
                "max_fill_delay_min": float(cfg["max_fill_delay_min"]),
                "confirm_pass": int(full_eval_pack["confirm_pass"]),
                "final_objective": float(full_eval_pack["final_objective"]),
                "gate_g0_chronology": int(g["gate_g0_chronology"]),
                "gate_g1_participation": int(g["gate_g1_participation"]),
                "gate_g2_winner_preservation": int(g["gate_g2_winner_preservation"]),
                "gate_g3_risk_sanity": int(g["gate_g3_risk_sanity"]),
                "gate_improve_target": int(g["gate_improve_target"]),
                "gate_g4_robustness": int(full_eval_pack["gate_g4_robustness"]),
                "split_valid_for_ranking": int(full_eval_pack["metrics"]["valid_for_ranking"]),
                "best_min_subperiod_delta": float(full_eval_pack["metrics"]["min_split_delta"]),
                "split_support_pass": int(full_eval_pack["split_pass"]),
                "route_total": int(rp["route_total"]),
                "route_confirm_count": int(rp["route_confirm_count"]),
                "route_pass_rate": float(rp["route_pass_rate"]),
                "route_support_pass": int(rp["route_support_pass"]),
                "holdout_trade_count": int(full_eval_pack["holdout_trade_count"]),
                "holdout_delta_expectancy_vs_baseline": float(hm.get("delta_expectancy", np.nan)),
                "holdout_delta_cvar_vs_baseline": float(hm.get("delta_cvar_5", np.nan)),
                "holdout_pass": int(full_eval_pack["holdout_pass"]),
                "delta_expectancy_vs_baseline": float(c["delta_expectancy"]),
                "delta_cvar_vs_baseline": float(c["delta_cvar_5"]),
                "delta_maxdd_vs_baseline": float(c["delta_max_drawdown"]),
                "maxdd_improve_ratio": float(c["maxdd_improve_ratio"]),
                "cvar_improve_ratio": float(c["cvar_improve_ratio"]),
                "baseline_expectancy_net": float(c["baseline_expectancy_net"]),
                "candidate_expectancy_net": float(c["candidate_expectancy_net"]),
                "baseline_cvar_5": float(c["baseline_cvar_5"]),
                "candidate_cvar_5": float(c["candidate_cvar_5"]),
                "baseline_max_drawdown": float(c["baseline_max_drawdown"]),
                "candidate_max_drawdown": float(c["candidate_max_drawdown"]),
                "baseline_trade_count": int(g["baseline_trade_count"]),
                "trade_count": int(l["trade_count"]),
                "retention": float(g["retention"]),
                "min_trade_count_required": int(g["min_trade_count_required"]),
                "baseline_winner_count": int(g["baseline_winner_count"]),
                "winner_count": int(l["meaningful_winner_count"]),
                "winner_retention": float(g["winner_retention"]),
                "instant_loser_rate": float(l["instant_loser_rate"]),
                "fast_loser_rate": float(l["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(l["bottom_decile_pnl_share"]),
                "worst_10_trades_sum": float(l["worst_10_trades_sum"]),
                "worst_25_trades_sum": float(l["worst_25_trades_sum"]),
                "instant_loser_rel_reduction": float(g["instant_loser_rel_reduction"]),
                "bottom_decile_rel_reduction": float(g["bottom_decile_rel_reduction"]),
                "same_bar_exit_count": int(ch["same_bar_exit_count"]),
                "exit_before_entry_count": int(ch["exit_before_entry_count"]),
                "entry_on_signal_count": int(ch["entry_on_signal_count"]),
                "parity_clean": int(ch["parity_clean"]),
                "route_detail_json": str(rp["route_detail_json"]),
            }
        )

    confirm_df = pd.DataFrame(confirm_rows)
    if confirm_df.empty:
        raise RuntimeError("No confirmation rows were produced")
    confirm_df = confirm_df.sort_values(
        ["confirm_pass", "final_objective", "delta_expectancy_vs_baseline", "route_pass_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    confirm_df.to_csv(run_dir / "sol_3m_ga_candidates_confirm.csv", index=False)

    approved_df = confirm_df[confirm_df["confirm_pass"] == 1].copy()
    decision = "APPROVED_EXEC_REPAIR" if not approved_df.empty else "NO_REPAIR_APPROVED"
    best_row = approved_df.iloc[0] if not approved_df.empty else confirm_df.iloc[0]

    baseline_payload = {
        "candidate_id": BASELINE_STRATEGY_ID,
        "cfg": baseline_cfg,
        "train_metrics": {
            "compare_vs_self": baseline_train_cmp,
            "loss": baseline_train_loss,
            "chrono": baseline_train_chrono,
            "trade_count": int(baseline_train_loss["trade_count"]),
        },
        "full_metrics": {
            "compare_vs_self": baseline_full_cmp,
            "loss": baseline_full_loss,
            "chrono": baseline_full_chrono,
            "trade_count": int(baseline_full_loss["trade_count"]),
        },
    }

    manifest = {
        "generated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "decision": decision,
        "symbol": SYMBOL,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "seed": int(args.seed),
        "ga_params": {
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "elite_k": int(args.elite_k),
            "mutation_rate": float(args.mutation_rate),
            "top_confirm": int(args.top_confirm),
        },
        "search_space": {
            "entry_mode": ["limit", "market"],
            "limit_offset_bps": [0.0, 3.0],
            "fallback_to_market": [0, 1],
            "fallback_delay_min": [0.0, 60.0],
            "max_fill_delay_min": [0.0, 60.0],
            "feature_skip_mask_enabled": 0,
        },
        "gates": {
            "g0_chronology_clean": "parity_clean==1 and same_bar_exit_count==0 and exit_before_entry_count==0",
            "g1_participation": f"retention>={float(args.retention_floor):.4f} and trade_count>=max({int(args.min_trades_abs)}, {float(args.min_trades_frac):.4f}*baseline_trade_count)",
            "g2_winner_preservation": f"winner_retention>={float(args.winner_retention_floor):.4f}",
            "g3_risk_sanity": "maxdd_improve_ratio>=0",
            "g4_robustness": "split_valid_for_ranking==1 and best_min_subperiod_delta>0 and route_pass_rate==1.0 and holdout_delta_expectancy>0 and holdout_delta_cvar>=0",
            "loss_target": f"instant_loser_rel_reduction>={float(args.instant_reduction_rel):.4f} OR bottom_decile_rel_reduction>={float(args.tail_reduction_rel):.4f}",
        },
        "windows": {
            "n_total_signals": int(n_total),
            "train_ratio": 0.60,
            "train_end_idx": int(train_end),
            "holdout_ratio": 0.20,
            "holdout_start_idx": int(holdout_start),
            "holdout_count": int(len(holdout_signal_ids)),
        },
        "input_paths": {
            "posture_dir": str(posture_dir),
            "strict_confirm_dir": str(strict_confirm_dir),
            "freeze_dir": str(freeze_dir),
            "foundation_dir": str(foundation_dir),
            "posture_active_subset_csv": str(posture_dir / "repaired_active_3m_subset.csv"),
            "posture_table_csv": str(posture_dir / "repaired_3m_posture_table.csv"),
            "selected_params_dir": str(selected_params_dir),
            "repaired_best_by_symbol_csv": str(freeze_dir / "repaired_best_by_symbol.csv"),
        },
        "baseline_hash": baseline_hash_payload(
            baseline_cfg=baseline_cfg,
            params_payload=params_payload,
            posture_dir=posture_dir,
            freeze_dir=freeze_dir,
            foundation_dir=foundation_dir,
        ),
        "build_meta": build_meta,
        "route_meta": route_meta,
        "contract_validation": contract_validation,
        "baseline_payload": baseline_payload,
        "outputs": {
            "screen_csv": str(run_dir / "sol_3m_ga_candidates_screen.csv"),
            "confirm_csv": str(run_dir / "sol_3m_ga_candidates_confirm.csv"),
            "report_md": str(run_dir / "sol_3m_ga_report.md"),
            "manifest_json": str(run_dir / "ga_manifest.json"),
        },
    }
    json_dump(run_dir / "ga_manifest.json", manifest)

    if decision == "APPROVED_EXEC_REPAIR":
        best_payload = {
            "generated_utc": utc_now_iso(),
            "decision": decision,
            "symbol": SYMBOL,
            "baseline_strategy_id": BASELINE_STRATEGY_ID,
            "candidate": {
                "candidate_id": str(best_row["candidate_id"]),
                "genome_hash": str(best_row["genome_hash"]),
                "entry_mode": str(best_row["entry_mode"]),
                "limit_offset_bps": float(best_row["limit_offset_bps"]),
                "fallback_to_market": int(best_row["fallback_to_market"]),
                "fallback_delay_min": float(best_row["fallback_delay_min"]),
                "max_fill_delay_min": float(best_row["max_fill_delay_min"]),
            },
            "metrics": {
                "delta_expectancy_vs_baseline": float(best_row["delta_expectancy_vs_baseline"]),
                "delta_cvar_vs_baseline": float(best_row["delta_cvar_vs_baseline"]),
                "maxdd_improve_ratio": float(best_row["maxdd_improve_ratio"]),
                "retention": float(best_row["retention"]),
                "winner_retention": float(best_row["winner_retention"]),
                "instant_loser_rel_reduction": float(best_row["instant_loser_rel_reduction"]),
                "bottom_decile_rel_reduction": float(best_row["bottom_decile_rel_reduction"]),
                "route_pass_rate": float(best_row["route_pass_rate"]),
                "holdout_delta_expectancy_vs_baseline": float(best_row["holdout_delta_expectancy_vs_baseline"]),
                "holdout_delta_cvar_vs_baseline": float(best_row["holdout_delta_cvar_vs_baseline"]),
            },
            "provenance": {
                "run_dir": str(run_dir),
                "ga_manifest": str(run_dir / "ga_manifest.json"),
                "confirm_csv": str(run_dir / "sol_3m_ga_candidates_confirm.csv"),
            },
        }
        json_dump(run_dir / "sol_3m_ga_best_candidate.json", best_payload)

    # Report
    screen_top = screen_df.sort_values(["screen_pass", "screen_objective"], ascending=[False, False]).head(10).copy()
    confirm_top = confirm_df.head(10).copy()
    baseline_table = pd.DataFrame(
        [
            {
                "scope": "train60",
                "trade_count": int(baseline_train_loss["trade_count"]),
                "instant_loser_rate": float(baseline_train_loss["instant_loser_rate"]),
                "fast_loser_rate": float(baseline_train_loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(baseline_train_loss["bottom_decile_pnl_share"]),
                "expectancy_net": float(modela.ga_exec._rollup_mode(baseline_train_rows, "exec")["mean_expectancy_net"]),  # pylint: disable=protected-access
                "cvar_5": float(modela.ga_exec._rollup_mode(baseline_train_rows, "exec")["cvar_5"]),  # pylint: disable=protected-access
                "max_drawdown": float(modela.ga_exec._rollup_mode(baseline_train_rows, "exec")["max_drawdown"]),  # pylint: disable=protected-access
                "same_bar_exit_count": int(baseline_train_chrono["same_bar_exit_count"]),
                "exit_before_entry_count": int(baseline_train_chrono["exit_before_entry_count"]),
            },
            {
                "scope": "full",
                "trade_count": int(baseline_full_loss["trade_count"]),
                "instant_loser_rate": float(baseline_full_loss["instant_loser_rate"]),
                "fast_loser_rate": float(baseline_full_loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(baseline_full_loss["bottom_decile_pnl_share"]),
                "expectancy_net": float(modela.ga_exec._rollup_mode(baseline_full_rows, "exec")["mean_expectancy_net"]),  # pylint: disable=protected-access
                "cvar_5": float(modela.ga_exec._rollup_mode(baseline_full_rows, "exec")["cvar_5"]),  # pylint: disable=protected-access
                "max_drawdown": float(modela.ga_exec._rollup_mode(baseline_full_rows, "exec")["max_drawdown"]),  # pylint: disable=protected-access
                "same_bar_exit_count": int(baseline_full_chrono["same_bar_exit_count"]),
                "exit_before_entry_count": int(baseline_full_chrono["exit_before_entry_count"]),
            },
        ]
    )

    report_lines: List[str] = [
        "# SOL 3m Loss-Concentration GA",
        "",
        f"- Generated UTC: `{utc_now_iso()}`",
        f"- Run dir: `{run_dir}`",
        f"- Decision: `{decision}`",
        f"- Symbol: `{SYMBOL}`",
        f"- Baseline strategy id lock: `{BASELINE_STRATEGY_ID}`",
        "",
        "## A) Baseline Metrics",
        "",
        markdown_table(
            baseline_table,
            [
                "scope",
                "trade_count",
                "expectancy_net",
                "cvar_5",
                "max_drawdown",
                "instant_loser_rate",
                "fast_loser_rate",
                "bottom_decile_pnl_share",
                "same_bar_exit_count",
                "exit_before_entry_count",
            ],
            n=5,
        ),
        "",
        "## B) Best Candidate Metrics + Deltas",
        "",
        markdown_table(
            pd.DataFrame([best_row.to_dict()]),
            [
                "candidate_id",
                "entry_mode",
                "limit_offset_bps",
                "fallback_to_market",
                "fallback_delay_min",
                "max_fill_delay_min",
                "delta_expectancy_vs_baseline",
                "delta_cvar_vs_baseline",
                "maxdd_improve_ratio",
                "retention",
                "winner_retention",
                "instant_loser_rel_reduction",
                "bottom_decile_rel_reduction",
                "route_pass_rate",
                "holdout_delta_expectancy_vs_baseline",
                "holdout_delta_cvar_vs_baseline",
                "confirm_pass",
            ],
            n=1,
        ),
        "",
        "## C) Gate Checklist (G0–G4)",
        "",
        markdown_table(
            confirm_top,
            [
                "candidate_id",
                "confirm_pass",
                "gate_g0_chronology",
                "gate_g1_participation",
                "gate_g2_winner_preservation",
                "gate_g3_risk_sanity",
                "gate_improve_target",
                "gate_g4_robustness",
            ],
            n=10,
        ),
        "",
        "## D) Split / Route / Holdout Evidence",
        "",
        markdown_table(
            confirm_top,
            [
                "candidate_id",
                "split_valid_for_ranking",
                "best_min_subperiod_delta",
                "route_confirm_count",
                "route_total",
                "route_pass_rate",
                "holdout_trade_count",
                "holdout_delta_expectancy_vs_baseline",
                "holdout_delta_cvar_vs_baseline",
                "holdout_pass",
            ],
            n=10,
        ),
        "",
        "## E) Proven vs Assumed",
        "",
        "- Proven: SOL-only input came from repaired posture freeze active subset and locked winner config id.",
        "- Proven: 1h signal layer/params remained frozen from repaired universe artifacts.",
        "- Proven: evaluation path reused `phase_a` entry-only wrapper with 1h-owned exits and chronology protections.",
        "- Proven: route family check required 3/3 confirmations for pass.",
        "- Assumed: no extra execution skip-mask features were needed in this pass; only existing 3m knob family was searched.",
        "",
        "## F) Recommended Next Step",
        "",
    ]
    if decision == "APPROVED_EXEC_REPAIR":
        report_lines.extend(
            [
                "- Run operational op-check on the approved SOL candidate, then run it in paper SHADOW mode side-by-side with current baseline before promotion.",
            ]
        )
    else:
        report_lines.extend(
            [
                "- Stop this GA branch; either redesign the allowed entry lever family or keep current SOL baseline as active execution posture.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Top Screened Candidates",
            "",
            markdown_table(
                screen_top,
                [
                    "generation",
                    "candidate_id",
                    "entry_mode",
                    "limit_offset_bps",
                    "fallback_to_market",
                    "fallback_delay_min",
                    "max_fill_delay_min",
                    "screen_pass",
                    "screen_objective",
                    "instant_loser_rel_reduction",
                    "bottom_decile_rel_reduction",
                ],
                n=10,
            ),
        ]
    )
    (run_dir / "sol_3m_ga_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
