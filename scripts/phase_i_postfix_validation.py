#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
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

from scripts import phase_e2_sol_representative as e2  # noqa: E402
from scripts import phase_g_sol_pathology_rehab as g  # noqa: E402
from scripts import phase_i_sol_signal_fork as p1  # noqa: E402


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    x = df.head(max_rows).copy()
    cols = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in x.iterrows():
        vals: List[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("nan" if not np.isfinite(v) else f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _latest_dir(root: Path, prefix: str) -> Path:
    cands = sorted([p for p in root.glob(f"{prefix}_*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"No dirs found for prefix `{prefix}` under {root}")
    return cands[-1]


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for t in str(raw).split(","):
        s = str(t).strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for t in str(raw).split(","):
        s = str(t).strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _parse_vol_sets(raw: str) -> List[Tuple[str, ...]]:
    # Format: "high;mid;low,mid;high,mid;high,low,mid"
    out: List[Tuple[str, ...]] = []
    for block in str(raw).split(";"):
        tok = str(block).strip()
        if not tok:
            continue
        parts = [str(x).strip().lower() for x in tok.split(",") if str(x).strip()]
        if not parts:
            continue
        uniq = tuple(sorted(set(parts)))
        out.append(uniq)
    # de-dup preserve order
    seen = set()
    uniq_out: List[Tuple[str, ...]] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq_out.append(t)
    return uniq_out


def _vol_key(buckets: Sequence[str]) -> str:
    return "_".join(sorted(str(x) for x in buckets))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_signal_ids(ids: Iterable[str]) -> str:
    return _sha256_text("|".join(str(x) for x in ids))


@dataclass(frozen=True)
class EvalConfig:
    name: str
    stage: str
    is_baseline: bool
    trend_min_threshold: float
    trend_gate_on: bool
    allowed_vol_buckets: Tuple[str, ...]
    vol_gate_on: bool
    cooldown_hours: int
    delay_1h_bars: int
    source_note: str


@dataclass
class EvalArtifacts:
    row: Dict[str, Any]
    trades_full: pd.DataFrame
    trades_valid: pd.DataFrame
    per_signal_step_vec: np.ndarray
    signal_ids_hash: str
    trade_vec_hash: str


def _evaluate_one(
    *,
    cfg: EvalConfig,
    ctx: Dict[str, Any],
    fee: Any,
    initial_equity: float,
    risk_per_trade: float,
    exec_horizon_hours: float,
) -> EvalArtifacts:
    rep_feat = ctx["rep_feat"]
    rep_subset = ctx["rep_subset"]
    split_definition = ctx["split_definition"]
    entries_base = ctx["entries_base"]
    rep_signal_ids = rep_subset["signal_id"].astype(str).tolist()

    if cfg.is_baseline:
        cand = p1.CandidateConfig(
            name=cfg.name,
            is_baseline=True,
            apply_trend_gate=False,
            trend_min=float(cfg.trend_min_threshold),
            apply_vol_gate=False,
            allowed_vol_buckets=tuple(),
            cooldown_hours=0,
            delay_bars=0,
        )
    else:
        cand = p1.CandidateConfig(
            name=cfg.name,
            is_baseline=False,
            apply_trend_gate=bool(cfg.trend_gate_on),
            trend_min=float(cfg.trend_min_threshold),
            apply_vol_gate=bool(cfg.vol_gate_on),
            allowed_vol_buckets=tuple(sorted(str(x) for x in cfg.allowed_vol_buckets)),
            cooldown_hours=int(cfg.cooldown_hours),
            delay_bars=int(cfg.delay_1h_bars),
        )

    ent, stage_trace = p1._apply_candidate_with_trace(entries_base, rep_feat, cand, symbol="SOLUSDT")
    signal_ids = ent["signal_id"].astype(str).tolist()
    signal_ids_hash = _hash_signal_ids(signal_ids)

    trades = e2._simulate_1h_from_entries(
        entries_df=ent[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
        symbol="SOLUSDT",
        fee=fee,
        exec_horizon_hours=float(exec_horizon_hours),
    )
    trades = trades.merge(
        rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "vol_bucket", "trend_bucket", "regime_bucket"]],
        on=["signal_id", "signal_time"],
        how="left",
    )
    valid = p1._valid_trades(trades)
    ret_net = pd.to_numeric(valid["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    ret_gross = pd.to_numeric(valid["pnl_gross_pct"], errors="coerce").dropna().to_numpy(dtype=float)

    eq_fix, m_fix = g._compute_fixed_size_equity_curve(
        trades,
        signals_total=int(len(ent)),
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
    )
    step_ret = p1._fixed_equity_step_returns(eq_fix, initial_equity=initial_equity)
    geom_legacy = p1._geom_mean_return(step_ret)
    geom_clean, ruin_event = p1._geom_mean_return_with_ruin_flag(step_ret)

    expectancy_net = float(np.mean(ret_net)) if ret_net.size else float("nan")
    expectancy_gross = float(np.mean(ret_gross)) if ret_gross.size else float("nan")
    cvar5_trade, cvar_tail_n, trade_vec_hash = p1._compute_cvar5_trade_notional(ret_net)
    pos = ret_net[ret_net > 0]
    neg = ret_net[ret_net < 0]
    pf_trade = float(np.sum(pos) / abs(np.sum(neg))) if neg.size and abs(np.sum(neg)) > 1e-12 else (float("inf") if pos.size else float("nan"))
    win_rate_trade = float((ret_net > 0).sum() / max(1, ret_net.size))

    support_ok, min_split_trades, _ = p1._support_stats(
        valid,
        rep_subset,
        split_definition,
        min_split_trades_req=int(args_global.min_split_trades),
    )

    total_return_fixed = _safe_float(m_fix.get("total_return_fixed"))
    maxdd_fixed = _safe_float(m_fix.get("max_drawdown_pct_fixed"))
    fatal_gate_fixed = int(
        (np.isfinite(total_return_fixed) and total_return_fixed <= float(args_global.fatal_total_return))
        or (np.isfinite(maxdd_fixed) and maxdd_fixed <= float(args_global.fatal_max_dd))
    )

    gate_expectancy = int(np.isfinite(expectancy_net) and expectancy_net > 0.0)
    gate_total_return = int(np.isfinite(total_return_fixed) and total_return_fixed > 0.0)
    gate_maxdd = int(np.isfinite(maxdd_fixed) and maxdd_fixed > float(args_global.deploy_max_dd_floor))
    gate_cvar = int(np.isfinite(cvar5_trade) and cvar5_trade > float(args_global.deploy_cvar5_floor))
    gate_pf = int(np.isfinite(pf_trade) and pf_trade >= float(args_global.deploy_pf_floor))
    gate_support = int(support_ok == 1)
    fixed_pass = int(gate_expectancy and gate_total_return and gate_maxdd and gate_cvar and gate_pf and gate_support)

    adverse = p1._adverse_loss_share(valid)
    worst_regime = p1._dominant_worst_regime_bucket(valid)
    per_signal_step_vec = p1._per_signal_fixed_step(eq_fix, rep_signal_ids=rep_signal_ids, initial_equity=initial_equity)

    avg_total_cost_bps = float(pd.to_numeric(valid.get("total_cost_bps"), errors="coerce").dropna().mean()) if not valid.empty else float("nan")
    avg_fee_bps = float(
        (pd.to_numeric(valid.get("entry_fee_bps"), errors="coerce").fillna(0.0) + pd.to_numeric(valid.get("exit_fee_bps"), errors="coerce").fillna(0.0)).mean()
    ) if not valid.empty else float("nan")
    avg_slippage_bps = float(
        (pd.to_numeric(valid.get("entry_slippage_bps"), errors="coerce").fillna(0.0) + pd.to_numeric(valid.get("exit_slippage_bps"), errors="coerce").fillna(0.0)).mean()
    ) if not valid.empty else float("nan")
    cost_drag = float(expectancy_gross - expectancy_net) if np.isfinite(expectancy_gross) and np.isfinite(expectancy_net) else float("nan")

    row = {
        "variant": cfg.name,
        "stage": cfg.stage,
        "source_note": cfg.source_note,
        "is_baseline": int(cfg.is_baseline),
        "trend_min_threshold": float(cfg.trend_min_threshold),
        "trend_alignment_gate_on": int(cfg.trend_gate_on),
        "volatility_gate_on": int(cfg.vol_gate_on),
        "allowed_vol_buckets": ",".join(cfg.allowed_vol_buckets),
        "cooldown_hours": int(cfg.cooldown_hours),
        "delay_1h_bars": int(cfg.delay_1h_bars),
        "signals_total": int(len(ent)),
        "trades_total": int(len(valid)),
        "signals_before_all": int(stage_trace.get("signals_before_all", 0)),
        "signals_after_trend_gate": int(stage_trace.get("signals_after_trend_gate", 0)),
        "signals_after_vol_gate": int(stage_trace.get("signals_after_vol_gate", 0)),
        "signals_after_cooldown": int(stage_trace.get("signals_after_cooldown", 0)),
        "signals_after_delay": int(stage_trace.get("signals_after_delay", 0)),
        "signals_after_dropna": int(stage_trace.get("signals_after_dropna", 0)),
        "removed_by_trend_gate": int(stage_trace.get("removed_by_trend_gate", 0)),
        "removed_by_vol_gate": int(stage_trace.get("removed_by_vol_gate", 0)),
        "removed_by_cooldown": int(stage_trace.get("removed_by_cooldown", 0)),
        "removed_by_delay_or_rebuild": int(stage_trace.get("removed_by_delay_or_rebuild", 0)),
        "removed_by_dropna": int(stage_trace.get("removed_by_dropna", 0)),
        "expectancy_net_trade_notional_dec": float(expectancy_net),
        "expectancy_gross_trade_notional_dec": float(expectancy_gross),
        "cvar_5_trade_notional_dec": float(cvar5_trade),
        "profit_factor_trade": float(pf_trade),
        "win_rate_trade": float(win_rate_trade),
        "trade_return_vector_sha256": str(trade_vec_hash),
        "signal_ids_sha256": str(signal_ids_hash),
        "trade_return_count": int(ret_net.size),
        "cvar_tail_count": int(cvar_tail_n),
        "total_return_fixed_equity_dec": float(total_return_fixed),
        "max_drawdown_pct_fixed_equity_dec": float(maxdd_fixed),
        "geometric_equity_step_return_fixed": float(geom_legacy),
        "geometric_equity_step_return_fixed_clean": float(geom_clean),
        "ruin_event_fixed": int(ruin_event),
        "support_ok": int(support_ok),
        "min_split_trades": int(min_split_trades),
        "fatal_gate_fixed": int(fatal_gate_fixed),
        "fixed_absolute_practical_pass": int(fixed_pass),
        "gate_expectancy_trade_gt_0": int(gate_expectancy),
        "gate_total_return_fixed_gt_0": int(gate_total_return),
        "gate_maxdd_fixed_gt_floor": int(gate_maxdd),
        "gate_cvar5_trade_gt_floor": int(gate_cvar),
        "gate_pf_trade_ge_floor": int(gate_pf),
        "gate_support_ok": int(gate_support),
        "variant_adverse_loss_share": float(adverse),
        "dominant_worst_regime_bucket": str(worst_regime),
        "avg_total_cost_bps": float(avg_total_cost_bps),
        "avg_fee_bps": float(avg_fee_bps),
        "avg_slippage_bps": float(avg_slippage_bps),
        "cost_drag_expectancy_dec": float(cost_drag),
    }
    return EvalArtifacts(
        row=row,
        trades_full=trades,
        trades_valid=valid,
        per_signal_step_vec=per_signal_step_vec,
        signal_ids_hash=signal_ids_hash,
        trade_vec_hash=trade_vec_hash,
    )


def _knob_effect_yes(df: pd.DataFrame, value_col: str) -> bool:
    x = df.copy()
    if x.empty:
        return False
    uniq_values = x[value_col].dropna().astype(str).unique().tolist()
    if len(uniq_values) <= 1:
        return False
    probe_cols = [
        "signals_total",
        "trades_total",
        "removed_by_trend_gate",
        "removed_by_vol_gate",
        "removed_by_cooldown",
        "expectancy_net_trade_notional_dec",
        "cvar_5_trade_notional_dec",
        "total_return_fixed_equity_dec",
        "max_drawdown_pct_fixed_equity_dec",
        "trade_return_vector_sha256",
    ]
    for c in probe_cols:
        if c not in x.columns:
            continue
        if x[c].astype(str).nunique(dropna=False) > 1:
            return True
    return False


def _rank_for_path_quality(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in [
        "fixed_absolute_practical_pass",
        "support_ok",
        "ruin_event_fixed",
        "geometric_equity_step_return_fixed_clean",
        "max_drawdown_pct_fixed_equity_dec",
        "cvar_5_trade_notional_dec",
        "variant_adverse_loss_share",
        "expectancy_net_trade_notional_dec",
        "profit_factor_trade",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values(
        [
            "fixed_absolute_practical_pass",
            "support_ok",
            "ruin_event_fixed",
            "geometric_equity_step_return_fixed_clean",
            "max_drawdown_pct_fixed_equity_dec",
            "cvar_5_trade_notional_dec",
            "variant_adverse_loss_share",
            "expectancy_net_trade_notional_dec",
            "profit_factor_trade",
        ],
        ascending=[False, False, True, False, False, False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    x["path_rank"] = np.arange(1, len(x) + 1, dtype=int)
    return x


def _dedup_candidates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy().reset_index(drop=True)
    x["duplicate_key"] = x["signal_ids_sha256"].astype(str) + "::" + x["trade_return_vector_sha256"].astype(str)
    groups = x.groupby("duplicate_key", dropna=False)
    map_rows: List[Dict[str, Any]] = []
    canonical_indices: List[int] = []
    gid = 0
    for key, gdf in groups:
        gid += 1
        gdf = gdf.sort_values(["is_baseline", "stage", "variant"], ascending=[False, True, True]).reset_index()
        can_idx = int(gdf.iloc[0]["index"])
        canonical_indices.append(can_idx)
        can = x.loc[can_idx]
        can_params = {
            "trend_min_threshold": _safe_float(can["trend_min_threshold"]),
            "allowed_vol_buckets": str(can["allowed_vol_buckets"]),
            "cooldown_hours": int(_safe_float(can["cooldown_hours"], 0)),
            "delay_1h_bars": int(_safe_float(can["delay_1h_bars"], 0)),
            "stage": str(can["stage"]),
        }
        for _, rr in gdf.iterrows():
            idx = int(rr["index"])
            if idx == can_idx:
                continue
            dup = x.loc[idx]
            dup_params = {
                "trend_min_threshold": _safe_float(dup["trend_min_threshold"]),
                "allowed_vol_buckets": str(dup["allowed_vol_buckets"]),
                "cooldown_hours": int(_safe_float(dup["cooldown_hours"], 0)),
                "delay_1h_bars": int(_safe_float(dup["delay_1h_bars"], 0)),
                "stage": str(dup["stage"]),
            }
            diffs = [k for k in ["trend_min_threshold", "allowed_vol_buckets", "cooldown_hours", "delay_1h_bars"] if str(can_params[k]) != str(dup_params[k])]
            map_rows.append(
                {
                    "duplicate_group_id": int(gid),
                    "duplicate_key": str(key),
                    "canonical_variant": str(can["variant"]),
                    "duplicate_variant": str(dup["variant"]),
                    "canonical_stage": str(can["stage"]),
                    "duplicate_stage": str(dup["stage"]),
                    "same_signal_ids": int(str(can["signal_ids_sha256"]) == str(dup["signal_ids_sha256"])),
                    "same_trade_return_vector": int(str(can["trade_return_vector_sha256"]) == str(dup["trade_return_vector_sha256"])),
                    "canonical_params_json": json.dumps(can_params, sort_keys=True),
                    "duplicate_params_json": json.dumps(dup_params, sort_keys=True),
                    "differing_params": ",".join(diffs),
                }
            )
    nondup = x.loc[sorted(set(canonical_indices))].copy().reset_index(drop=True)
    dup_map = pd.DataFrame(map_rows)
    return nondup, dup_map


def _loss_regime_concentration(valid: pd.DataFrame) -> Tuple[str, float, float]:
    if valid.empty:
        return "none", float("nan"), float("nan")
    x = valid.copy()
    x["pnl_net_pct"] = pd.to_numeric(x["pnl_net_pct"], errors="coerce")
    lose = x[x["pnl_net_pct"] < 0].copy()
    if lose.empty:
        return "none", 0.0, 0.0
    cnt = lose["regime_bucket"].astype(str).value_counts(dropna=False)
    dom = str(cnt.index[0]) if len(cnt) else "unknown"
    share_cnt = float(cnt.iloc[0] / max(1, int(cnt.sum()))) if len(cnt) else float("nan")
    abs_loss = lose.assign(abs_loss=np.abs(pd.to_numeric(lose["pnl_net_pct"], errors="coerce")))
    sum_by = abs_loss.groupby("regime_bucket", dropna=False)["abs_loss"].sum()
    share_abs = float(sum_by.max() / max(1e-12, float(sum_by.sum()))) if not sum_by.empty else float("nan")
    return dom, share_cnt, share_abs


def _copy_snapshot(run_dir: Path, files: Sequence[Path]) -> None:
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for fp in files:
        p = Path(fp)
        if p.exists():
            shutil.copy2(p, snap / p.name)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase I post-fix validation and focused research search (SOL, contract-locked).")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--phaseh-dir", default="")
    ap.add_argument("--phaseg-dir", default="")

    ap.add_argument("--expected-rep-subset-hash", default=p1.EXPECTED_REPRESENTATIVE_SUBSET_SHA256)
    ap.add_argument("--expected-selected-model-set-hash", default=p1.EXPECTED_SELECTED_MODEL_SET_SHA256)

    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--initial-equity", type=float, default=1.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)

    ap.add_argument("--trend-values", default="0.30,0.50,0.60,0.75")
    ap.add_argument("--vol-oat-values", default="high;mid;low,mid;high,mid;high,low,mid")
    ap.add_argument("--cooldown-values", default="0,2,4,6")
    ap.add_argument("--delay-values", default="0,1,2")

    ap.add_argument("--anchor-trend", type=float, default=0.50)
    ap.add_argument("--anchor-vol-buckets", default="high,low,mid")
    ap.add_argument("--anchor-cooldown", type=int, default=4)
    ap.add_argument("--anchor-delay", type=int, default=1)

    ap.add_argument("--focus-top-n", type=int, default=3)
    ap.add_argument("--max-cross-combos", type=int, default=36)

    ap.add_argument("--min-split-trades", type=int, default=40)
    ap.add_argument("--min-bucket-support", type=int, default=30)

    ap.add_argument("--fatal-max-dd", type=float, default=-0.95)
    ap.add_argument("--fatal-total-return", type=float, default=-0.95)
    ap.add_argument("--deploy-max-dd-floor", type=float, default=-0.35)
    ap.add_argument("--deploy-cvar5-floor", type=float, default=-0.0015)
    ap.add_argument("--deploy-pf-floor", type=float, default=1.05)
    return ap


def run(args: argparse.Namespace) -> Path:
    if str(args.symbol).strip().upper() != "SOLUSDT":
        raise RuntimeError("This validation workflow is scoped to SOLUSDT.")

    out_root = _resolve(args.outdir)
    phaseh_dir = _resolve(args.phaseh_dir) if args.phaseh_dir else _latest_dir(out_root, "PHASEH_SOL_FREEZE_FORK")
    phaseh_manifest = json.loads((phaseh_dir / "phaseH_run_manifest.json").read_text(encoding="utf-8"))
    phaseg_dir = _resolve(args.phaseg_dir) if args.phaseg_dir else _resolve(str(phaseh_manifest.get("phaseg_source_dir", "")))

    ctx = p1._prepare_rep_context(
        phaseg_dir=phaseg_dir,
        symbol="SOLUSDT",
        expected_rep_hash=str(args.expected_rep_subset_hash),
        expected_selected_model_hash=str(args.expected_selected_model_set_hash),
    )
    fee = g.phasec_bt._load_fee_model(_resolve(str(ctx["contract"]["fee_model_path"])))
    initial_equity = float(ctx["contract"].get("initial_equity", args.initial_equity))
    risk_per_trade = float(ctx["contract"].get("risk_per_trade", args.risk_per_trade))

    run_dir = out_root / f"PHASEI_SOL_POSTFIX_VALIDATION_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    _copy_snapshot(
        run_dir,
        [
            ctx["e2_dir"] / "run_manifest.json",
            ctx["e2_dir"] / "accounting_contract.json",
            ctx["e2_dir"] / "pass_fail_gates.json",
            ctx["e2_dir"] / "representative_subset_signals.csv",
            ctx["e2_dir"] / "trades_v2r_1h_reference_control.csv",
            ctx["e2_dir"] / "trades_v3r_exec_3m_phasec_control.csv",
            ctx["e2_dir"] / "trades_v4r_exec_3m_phasec_best.csv",
            _resolve(str(ctx["contract"]["fee_model_path"])),
            _resolve(str(ctx["contract"]["metrics_definition_path"])),
            _resolve(str(ctx["signal_source_path"])),
            phaseg_dir / "phaseG_report.md",
            phaseh_dir / "phaseH_sol_signal_fork_spec.md",
        ],
    )

    # ----------------------------
    # STEP 1: OAT sensitivity
    # ----------------------------
    trend_vals = _parse_float_list(args.trend_values)
    vol_sets = _parse_vol_sets(args.vol_oat_values)
    cooldown_vals = _parse_int_list(args.cooldown_values)
    delay_vals = _parse_int_list(args.delay_values)
    anchor_vol = tuple(sorted(set(_parse_vol_sets(args.anchor_vol_buckets.replace(";", ","))[0] if ";" in args.anchor_vol_buckets else [x.strip().lower() for x in args.anchor_vol_buckets.split(",") if x.strip()])))
    if not anchor_vol:
        anchor_vol = ("high", "low", "mid")

    eval_rows: List[Dict[str, Any]] = []
    eval_artifacts: Dict[str, EvalArtifacts] = {}

    def _run_cfg(cfg: EvalConfig) -> None:
        art = _evaluate_one(
            cfg=cfg,
            ctx=ctx,
            fee=fee,
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
            exec_horizon_hours=float(args.exec_horizon_hours),
        )
        eval_rows.append(art.row)
        eval_artifacts[cfg.name] = art

    # Baseline contract-locked reference.
    _run_cfg(
        EvalConfig(
            name="baseline_contract_locked",
            stage="baseline",
            is_baseline=True,
            trend_min_threshold=float(args.anchor_trend),
            trend_gate_on=False,
            allowed_vol_buckets=tuple(),
            vol_gate_on=False,
            cooldown_hours=0,
            delay_1h_bars=0,
            source_note="contract_locked_baseline",
        )
    )

    # OAT: trend
    for t in trend_vals:
        _run_cfg(
            EvalConfig(
                name=f"oat_trend_t{t:.2f}",
                stage="oat_trend",
                is_baseline=False,
                trend_min_threshold=float(t),
                trend_gate_on=True,
                allowed_vol_buckets=tuple(sorted(anchor_vol)),
                vol_gate_on=True,
                cooldown_hours=int(args.anchor_cooldown),
                delay_1h_bars=int(args.anchor_delay),
                source_note="oat_hold_vol_cooldown_delay",
            )
        )

    # OAT: vol buckets
    for vb in vol_sets:
        _run_cfg(
            EvalConfig(
                name=f"oat_vol_{_vol_key(vb)}",
                stage="oat_vol",
                is_baseline=False,
                trend_min_threshold=float(args.anchor_trend),
                trend_gate_on=True,
                allowed_vol_buckets=tuple(sorted(vb)),
                vol_gate_on=True,
                cooldown_hours=int(args.anchor_cooldown),
                delay_1h_bars=int(args.anchor_delay),
                source_note="oat_hold_trend_cooldown_delay",
            )
        )

    # OAT: cooldown
    for cd in cooldown_vals:
        _run_cfg(
            EvalConfig(
                name=f"oat_cooldown_cd{int(cd)}h",
                stage="oat_cooldown",
                is_baseline=False,
                trend_min_threshold=float(args.anchor_trend),
                trend_gate_on=True,
                allowed_vol_buckets=tuple(sorted(anchor_vol)),
                vol_gate_on=True,
                cooldown_hours=int(cd),
                delay_1h_bars=int(args.anchor_delay),
                source_note="oat_hold_trend_vol_delay",
            )
        )

    # OAT: delay
    for d in delay_vals:
        _run_cfg(
            EvalConfig(
                name=f"oat_delay_d{int(d)}",
                stage="oat_delay",
                is_baseline=False,
                trend_min_threshold=float(args.anchor_trend),
                trend_gate_on=True,
                allowed_vol_buckets=tuple(sorted(anchor_vol)),
                vol_gate_on=True,
                cooldown_hours=int(args.anchor_cooldown),
                delay_1h_bars=int(d),
                source_note="oat_hold_trend_vol_cooldown",
            )
        )

    eval_df = pd.DataFrame(eval_rows)
    oat_df = eval_df[eval_df["stage"].astype(str).str.startswith("oat_") | (eval_df["stage"] == "baseline")].copy()
    oat_df.to_csv(run_dir / "sensitivity_oat_results.csv", index=False)

    # Knob effect flags.
    effect_trend = _knob_effect_yes(oat_df[oat_df["stage"] == "oat_trend"], "trend_min_threshold")
    effect_vol = _knob_effect_yes(oat_df[oat_df["stage"] == "oat_vol"], "allowed_vol_buckets")
    effect_cd = _knob_effect_yes(oat_df[oat_df["stage"] == "oat_cooldown"], "cooldown_hours")
    effect_delay = _knob_effect_yes(oat_df[oat_df["stage"] == "oat_delay"], "delay_1h_bars")
    effect_exists = bool(effect_trend or effect_vol or effect_cd or effect_delay)

    # ----------------------------
    # STEP 3: reduced cross (only if effect exists)
    # ----------------------------
    cross_df = pd.DataFrame()
    if effect_exists:
        ranked_oat = _rank_for_path_quality(oat_df[oat_df["stage"] != "baseline"].copy())
        top_oat = ranked_oat.head(int(args.focus_top_n)).copy()

        # keep at most 2 levels per knob for small focused cross.
        trend_levels = list(dict.fromkeys(pd.to_numeric(top_oat["trend_min_threshold"], errors="coerce").dropna().tolist()))[:2]
        vol_levels = list(dict.fromkeys(top_oat["allowed_vol_buckets"].astype(str).tolist()))[:2]
        cd_levels = list(dict.fromkeys(pd.to_numeric(top_oat["cooldown_hours"], errors="coerce").dropna().astype(int).tolist()))[:2]
        d_levels = list(dict.fromkeys(pd.to_numeric(top_oat["delay_1h_bars"], errors="coerce").dropna().astype(int).tolist()))[:2]

        if not trend_levels:
            trend_levels = [float(args.anchor_trend)]
        if not vol_levels:
            vol_levels = [",".join(sorted(anchor_vol))]
        if not cd_levels:
            cd_levels = [int(args.anchor_cooldown)]
        if not d_levels:
            d_levels = [int(args.anchor_delay)]

        cross_cfgs: List[EvalConfig] = []
        for t, vtxt, cd, d in itertools.product(trend_levels, vol_levels, cd_levels, d_levels):
            vb = tuple(sorted([z.strip().lower() for z in str(vtxt).split(",") if z.strip()]))
            name = f"cross_t{float(t):.2f}_v{_vol_key(vb)}_cd{int(cd)}h_d{int(d)}"
            cross_cfgs.append(
                EvalConfig(
                    name=name,
                    stage="cross",
                    is_baseline=False,
                    trend_min_threshold=float(t),
                    trend_gate_on=True,
                    allowed_vol_buckets=vb,
                    vol_gate_on=True,
                    cooldown_hours=int(cd),
                    delay_1h_bars=int(d),
                    source_note="reduced_cross_from_oat_top",
                )
            )

        if len(cross_cfgs) > int(args.max_cross_combos):
            cross_cfgs = cross_cfgs[: int(args.max_cross_combos)]

        for cfg in cross_cfgs:
            _run_cfg(cfg)

        eval_df = pd.DataFrame(eval_rows)
        cross_df = eval_df[eval_df["stage"] == "cross"].copy()
        cross_df.to_csv(run_dir / "reduced_cross_results.csv", index=False)

    # ----------------------------
    # STEP 2: duplicate pruning on all tested variants
    # ----------------------------
    eval_df = pd.DataFrame(eval_rows)
    nondup_df, dup_map = _dedup_candidates(eval_df)
    dup_map.to_csv(run_dir / "duplicate_variant_map.csv", index=False)

    # Effective trials among non-duplicate non-baseline variants.
    nondup_nonbase = nondup_df[nondup_df["is_baseline"] == 0].copy()
    step_cols: List[np.ndarray] = []
    for v in nondup_nonbase["variant"].astype(str).tolist():
        art = eval_artifacts.get(v)
        if art is None:
            continue
        step_cols.append(np.asarray(art.per_signal_step_vec, dtype=float))
    if step_cols:
        mat = np.column_stack(step_cols)
    else:
        mat = np.zeros((len(ctx["rep_subset"]), 1), dtype=float)
    n_eff_nondup, avg_corr_nondup = p1._estimate_effective_trials(mat)

    # Non-duplicate shortlist top 5 (plus baseline row).
    ranked_nondup = _rank_for_path_quality(nondup_df.copy())
    top5_nondup = ranked_nondup[ranked_nondup["is_baseline"] == 0].head(5).copy()
    baseline_row = ranked_nondup[ranked_nondup["is_baseline"] == 1].head(1).copy()
    shortlist = pd.concat([baseline_row, top5_nondup], ignore_index=True)
    shortlist.to_csv(run_dir / "nonduplicate_shortlist.csv", index=False)

    # ----------------------------
    # STEP 4: edge vs cost decomposition
    # ----------------------------
    edge_rows: List[Dict[str, Any]] = []
    if baseline_row.empty:
        raise RuntimeError("Missing baseline row in non-duplicate set.")
    base = baseline_row.iloc[0]
    base_exp_gross = _safe_float(base["expectancy_gross_trade_notional_dec"])
    base_exp_net = _safe_float(base["expectancy_net_trade_notional_dec"])
    base_adverse = _safe_float(base["variant_adverse_loss_share"])

    for _, r in shortlist.iterrows():
        v = str(r["variant"])
        art = eval_artifacts[v]
        valid = art.trades_valid.copy()
        dom_reg, dom_share_cnt, dom_share_abs = _loss_regime_concentration(valid)
        exp_g = _safe_float(r["expectancy_gross_trade_notional_dec"])
        exp_n = _safe_float(r["expectancy_net_trade_notional_dec"])
        edge_rows.append(
            {
                "variant": v,
                "is_baseline": int(r["is_baseline"]),
                "stage": str(r["stage"]),
                "signals_total": int(r["signals_total"]),
                "trades_total": int(r["trades_total"]),
                "entry_rate_trades_per_signal": float(_safe_float(r["trades_total"]) / max(1, int(_safe_float(r["signals_total"], 0)))),
                "expectancy_gross_trade_notional_dec": float(exp_g),
                "expectancy_net_trade_notional_dec": float(exp_n),
                "cost_drag_expectancy_dec": float(_safe_float(r["cost_drag_expectancy_dec"])),
                "avg_total_cost_bps": float(_safe_float(r["avg_total_cost_bps"])),
                "avg_fee_bps": float(_safe_float(r["avg_fee_bps"])),
                "avg_slippage_bps": float(_safe_float(r["avg_slippage_bps"])),
                "profit_factor_trade": float(_safe_float(r["profit_factor_trade"])),
                "cvar_5_trade_notional_dec": float(_safe_float(r["cvar_5_trade_notional_dec"])),
                "total_return_fixed_equity_dec": float(_safe_float(r["total_return_fixed_equity_dec"])),
                "max_drawdown_pct_fixed_equity_dec": float(_safe_float(r["max_drawdown_pct_fixed_equity_dec"])),
                "geometric_equity_step_return_fixed_clean": float(_safe_float(r["geometric_equity_step_return_fixed_clean"])),
                "ruin_event_fixed": int(_safe_float(r["ruin_event_fixed"], 0)),
                "dominant_worst_regime_bucket": str(dom_reg),
                "loss_concentration_share_count": float(dom_share_cnt),
                "loss_concentration_share_abs": float(dom_share_abs),
                "adverse_loss_share": float(_safe_float(r["variant_adverse_loss_share"])),
                "adverse_loss_share_delta_vs_baseline": float(_safe_float(r["variant_adverse_loss_share"]) - base_adverse),
                "delta_expectancy_gross_vs_baseline": float(exp_g - base_exp_gross) if np.isfinite(exp_g) and np.isfinite(base_exp_gross) else float("nan"),
                "delta_expectancy_net_vs_baseline": float(exp_n - base_exp_net) if np.isfinite(exp_n) and np.isfinite(base_exp_net) else float("nan"),
            }
        )

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(run_dir / "edge_vs_cost_decomposition.csv", index=False)

    # Edge-vs-fee verdict.
    nonbase_edge = edge_df[edge_df["is_baseline"] == 0].copy()
    hidden_gross = int(((pd.to_numeric(nonbase_edge["expectancy_gross_trade_notional_dec"], errors="coerce") > 0) & (pd.to_numeric(nonbase_edge["expectancy_net_trade_notional_dec"], errors="coerce") <= 0)).any()) if not nonbase_edge.empty else 0
    no_signal_edge = int((pd.to_numeric(nonbase_edge["expectancy_gross_trade_notional_dec"], errors="coerce") <= 0).all()) if not nonbase_edge.empty else 1
    if no_signal_edge == 1:
        edge_fee_verdict = "no_signal_edge_even_before_costs"
    elif hidden_gross == 1:
        edge_fee_verdict = "some_gross_edge_killed_by_costs"
    else:
        edge_fee_verdict = "mixed_or_weak_gross_edge_not_cost_dominated"

    # Final recommendation.
    any_fixed_pass_nondup = int((pd.to_numeric(nondup_nonbase["fixed_absolute_practical_pass"], errors="coerce") == 1).any()) if not nondup_nonbase.empty else 0
    if any_fixed_pass_nondup == 1:
        recommendation = "continue_signal_fork"
    else:
        recommendation = "pivot_to_execution_exit_optimization"

    # Report.
    setup_checks = {
        "symbol_match": int(str(ctx["contract"].get("symbol", "")).upper() == "SOLUSDT"),
        "contract_id_match": int(str(ctx["contract"].get("contract_id", "")) == p1.EXPECTED_CONTRACT_ID),
        "fee_hash_match": int(str(ctx["fee_hash"]) == g.EXPECTED_PHASEA_FEE_HASH),
        "metrics_hash_match": int(str(ctx["metrics_hash"]) == g.EXPECTED_PHASEA_METRICS_HASH),
        "subset_hash_match": int(str(ctx["rep_subset_hash"]) == str(args.expected_rep_subset_hash)),
        "selected_model_set_hash_match": int(str(ctx["selected_model_set_hash"]) == str(args.expected_selected_model_set_hash)),
        "split_integrity": int(len(ctx["split_definition"]) >= 1),
    }

    report_lines = [
        "# Phase I SOL Post-Fix Validation Report",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Run dir: `{run_dir}`",
        f"- Source Phase H dir: `{phaseh_dir}`",
        f"- Source Phase G dir: `{phaseg_dir}`",
        "",
        "## Frozen Setup",
        "",
        f"- representative_subset_sha256: `{ctx['rep_subset_hash']}`",
        f"- fee_model_sha256: `{ctx['fee_hash']}`",
        f"- metrics_definition_sha256: `{ctx['metrics_hash']}`",
        f"- selected_model_set_sha256: `{ctx['selected_model_set_hash']}`",
        f"- setup_checks: {json.dumps(setup_checks, sort_keys=True)}",
        "",
        "## Step 1: Knob Sensitivity (OAT)",
        "",
        f"- trend threshold effect detected: {int(effect_trend)}",
        f"- volatility bucket effect detected: {int(effect_vol)}",
        f"- cooldown effect detected: {int(effect_cd)}",
        f"- delay effect detected: {int(effect_delay)}",
        f"- overall sensitivity exists: {int(effect_exists)}",
        f"- OAT rows: {int(len(oat_df))}",
        "",
        _markdown_table(
            _rank_for_path_quality(oat_df[oat_df["stage"] != "baseline"].copy())[
                [
                    "variant",
                    "stage",
                    "signals_total",
                    "trades_total",
                    "expectancy_net_trade_notional_dec",
                    "cvar_5_trade_notional_dec",
                    "geometric_equity_step_return_fixed_clean",
                    "total_return_fixed_equity_dec",
                    "max_drawdown_pct_fixed_equity_dec",
                    "support_ok",
                    "fixed_absolute_practical_pass",
                ]
            ]
        ),
        "",
        "## Step 2: Duplicate Pruning",
        "",
        f"- tested_variants_total: {int(len(eval_df))}",
        f"- duplicate_rows_detected: {int(len(dup_map))}",
        f"- nonduplicate_variants_total: {int(len(nondup_df))}",
        f"- effective_trials_after_pruning: {float(n_eff_nondup):.6f}",
        f"- avg_pairwise_step_return_corr_nondup: {float(avg_corr_nondup):.6f}" if np.isfinite(avg_corr_nondup) else "- avg_pairwise_step_return_corr_nondup: n/a",
        "",
        "## Step 3: Reduced Cross Search",
        "",
        f"- reduced_cross_run: {int(effect_exists)}",
        f"- reduced_cross_rows: {int(len(cross_df))}",
        "Stop rule:",
        "- no fixed-size absolute passers => stop search for this run.",
        "",
        "## Step 4: Edge vs Cost Decomposition",
        "",
        f"- edge_vs_fee_verdict: **{edge_fee_verdict}**",
        _markdown_table(
            edge_df[
                [
                    "variant",
                    "is_baseline",
                    "expectancy_gross_trade_notional_dec",
                    "expectancy_net_trade_notional_dec",
                    "cost_drag_expectancy_dec",
                    "avg_total_cost_bps",
                    "trades_total",
                    "adverse_loss_share",
                    "dominant_worst_regime_bucket",
                ]
            ]
        ),
        "",
        "## Best Non-Duplicate Candidates",
        "",
        _markdown_table(
            top5_nondup[
                [
                    "variant",
                    "stage",
                    "signals_total",
                    "trades_total",
                    "geometric_equity_step_return_fixed_clean",
                    "total_return_fixed_equity_dec",
                    "max_drawdown_pct_fixed_equity_dec",
                    "cvar_5_trade_notional_dec",
                    "fixed_absolute_practical_pass",
                ]
            ]
        ),
        "",
        "## Decision",
        "",
        f"- fixed_size_passers_nonduplicate: {any_fixed_pass_nondup}",
        f"- recommendation: **{recommendation}**",
        f"- deployment_status: **{'NO_DEPLOY' if any_fixed_pass_nondup == 0 else 'REVIEW_FOR_PAPER'}**",
    ]
    (run_dir / "phaseI_sol_postfix_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": "SOLUSDT",
        "phaseh_source_dir": str(phaseh_dir),
        "phaseg_source_dir": str(phaseg_dir),
        "representative_subset_sha256": str(ctx["rep_subset_hash"]),
        "fee_model_sha256": str(ctx["fee_hash"]),
        "metrics_definition_sha256": str(ctx["metrics_hash"]),
        "selected_model_set_sha256": str(ctx["selected_model_set_hash"]),
        "setup_checks": setup_checks,
        "oat_rows": int(len(oat_df)),
        "cross_rows": int(len(cross_df)),
        "duplicates": int(len(dup_map)),
        "nonduplicates": int(len(nondup_df)),
        "effective_trials_nondup": float(n_eff_nondup),
        "avg_pairwise_corr_nondup": float(avg_corr_nondup) if np.isfinite(avg_corr_nondup) else None,
        "edge_vs_fee_verdict": edge_fee_verdict,
        "recommendation": recommendation,
    }
    (run_dir / "phaseI_postfix_run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return run_dir


args_global: argparse.Namespace


def main() -> None:
    global args_global
    args_global = build_arg_parser().parse_args()
    out = run(args_global)
    print(str(out))


if __name__ == "__main__":
    main()
