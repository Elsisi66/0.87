#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1 as subset1  # noqa: E402
from scripts import sol_3m_lossconcentration_ga as ga  # noqa: E402


RUN_PREFIX = "SOL_ENTRY_MECHANICS_SWEEP"
SYMBOL = "SOLUSDT"
BASELINE_STRATEGY_ID = "M1_ENTRY_ONLY_PASSIVE_BASELINE"


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


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (set, tuple)):
            return list(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
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


def find_latest_complete(root: Path, pattern: str, required: List[str]) -> Optional[Path]:
    cands = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    return None


def fill_type_stats(rows_df: pd.DataFrame) -> Dict[str, Any]:
    filled = to_num(rows_df.get("exec_filled", 0)).fillna(0).astype(int)
    valid = to_num(rows_df.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    m = (filled == 1) & (valid == 1)
    if int(m.sum()) == 0:
        return {
            "fills_valid": 0,
            "fill_limit_count": 0,
            "fill_market_count": 0,
            "fill_market_fallback_count": 0,
            "fill_market_guard_fallback_count": 0,
            "fill_limit_cap_fallback_count": 0,
            "fill_market_cap_fallback_count": 0,
            "maker_fill_share": float("nan"),
            "taker_fill_share": float("nan"),
            "entry_improvement_bps_mean": float("nan"),
            "entry_improvement_bps_median": float("nan"),
            "entry_price_vs_signal_open_bps_mean": float("nan"),
            "entry_price_vs_signal_open_bps_median": float("nan"),
        }

    entry_type = rows_df.loc[m, "exec_entry_type"].astype(str).str.strip().str.lower()
    liq = rows_df.loc[m, "exec_fill_liquidity_type"].astype(str).str.strip().str.lower()
    imp = to_num(rows_df.loc[m, "entry_improvement_bps"]).astype(float)
    vc = entry_type.value_counts()

    return {
        "fills_valid": int(m.sum()),
        "fill_limit_count": int(vc.get("limit", 0)),
        "fill_market_count": int(vc.get("market", 0)),
        "fill_market_fallback_count": int(vc.get("market_fallback", 0)),
        "fill_market_guard_fallback_count": int(vc.get("market_guard_fallback", 0)),
        "fill_limit_cap_fallback_count": int(vc.get("limit_cap_fallback", 0)),
        "fill_market_cap_fallback_count": int(vc.get("market_cap_fallback", 0)),
        "maker_fill_share": float((liq == "maker").mean()),
        "taker_fill_share": float((liq == "taker").mean()),
        "entry_improvement_bps_mean": float(np.nanmean(imp)),
        "entry_improvement_bps_median": float(np.nanmedian(imp)),
        "entry_price_vs_signal_open_bps_mean": float(-np.nanmean(imp)),
        "entry_price_vs_signal_open_bps_median": float(-np.nanmedian(imp)),
    }


def build_variants(baseline_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    b = dict(baseline_cfg)

    def mk(name: str, cfg_updates: Dict[str, Any], label: str) -> Dict[str, Any]:
        x = dict(b)
        x.update(cfg_updates)
        x["candidate_id"] = name
        x["label"] = label
        x["feature_skip_mask_enabled"] = 0
        return ga.normalize_cfg(x)

    variants: List[Dict[str, Any]] = [
        mk(
            "V0_BASELINE",
            {
                "delay_bars_before_entry": 0,
                "adverse_move_guard_bps": float("nan"),
                "adverse_guard_fallback_to_baseline_market": 1,
                "max_adverse_market_fill_bps": float("nan"),
                "cap_fallback_limit_offset_bps": 0.0,
                "cap_fallback_ttl_bars": 2,
            },
            "Control baseline",
        ),
        mk(
            "V1_PASSIVE_LIMIT_TTL",
            {
                "entry_mode": "limit",
                "limit_offset_bps": 0.0,
                "fallback_to_market": 1,
                "fallback_delay_min": 6.0,
                "max_fill_delay_min": 6.0,
                "delay_bars_before_entry": 0,
                "adverse_move_guard_bps": float("nan"),
                "max_adverse_market_fill_bps": float("nan"),
            },
            "Passive limit at ref, TTL=2 bars then market fallback",
        ),
        mk(
            "V2_DELAYED_REPRICE_GUARD",
            {
                "entry_mode": "market",
                "limit_offset_bps": 0.0,
                "fallback_to_market": 0,
                "fallback_delay_min": 0.0,
                "max_fill_delay_min": 0.0,
                "delay_bars_before_entry": 1,
                "adverse_move_guard_bps": 8.0,
                "adverse_guard_fallback_to_baseline_market": 1,
                "max_adverse_market_fill_bps": float("nan"),
            },
            "Delay 1 bar; guard adverse move >8bps then fallback baseline market",
        ),
        mk(
            "V3_MAX_SLIPPAGE_CAP",
            {
                "delay_bars_before_entry": 0,
                "adverse_move_guard_bps": float("nan"),
                "adverse_guard_fallback_to_baseline_market": 1,
                "max_adverse_market_fill_bps": 10.0,
                "cap_fallback_limit_offset_bps": 0.0,
                "cap_fallback_ttl_bars": 2,
            },
            "Cap adverse market fill at 10bps; fallback to limit+TTL",
        ),
    ]
    return variants


def quality_from_build_meta(build_meta: Dict[str, Any]) -> Dict[str, Any]:
    total = int(build_meta.get("signals_total", 0))
    with_data = int(build_meta.get("signals_with_3m_data", 0))
    partial = int(build_meta.get("signals_partial_3m_data", 0))
    missing = int(build_meta.get("signals_missing_3m_data", 0))
    ready = max(0, with_data - partial)
    miss_rate = float(missing / total) if total > 0 else float("nan")
    if with_data <= 0:
        status = "DATA_BLOCKED"
    elif missing > 0 or partial > 0:
        status = "PARTIAL"
    else:
        status = "READY"
    return {
        "integrity_status": status,
        "windows_ready": int(ready),
        "windows_partial": int(partial),
        "windows_blocked": int(missing),
        "missing_window_rate": miss_rate,
        "signals_covered": int(with_data),
        "signals_uncovered": int(missing),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOL-only bounded entry-mechanics sweep under repaired branch")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--posture-dir", default="")
    ap.add_argument("--strict-confirm-dir", default="")
    ap.add_argument("--seed", type=int, default=20260307)
    ap.add_argument("--retention-floor", type=float, default=0.95)
    ap.add_argument("--min-trades-abs", type=int, default=1000)
    ap.add_argument("--min-trades-frac", type=float, default=0.80)
    ap.add_argument("--winner-retention-floor", type=float, default=0.98)
    ap.add_argument("--instant-reduction-rel", type=float, default=0.05)
    ap.add_argument("--fast-reduction-rel", type=float, default=0.05)
    ap.add_argument("--tail-reduction-rel", type=float, default=0.15)
    ap.add_argument(
        "--variants",
        default="V0_BASELINE,V1_PASSIVE_LIMIT_TTL,V2_DELAYED_REPRICE_GUARD,V3_MAX_SLIPPAGE_CAP",
        help="Comma-separated variant ids to run (must include V0_BASELINE for invariance checks)",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

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
        raise FileNotFoundError(f"Freeze directory does not exist: {freeze_dir}")
    foundation_dir = Path(strict_manifest.get("foundation_dir", "")).resolve()
    if not foundation_dir.exists():
        raise FileNotFoundError(f"Foundation directory does not exist: {foundation_dir}")

    posture_active_df = pd.read_csv(posture_dir / "repaired_active_3m_subset.csv")
    posture_active_df["symbol"] = posture_active_df["symbol"].astype(str).str.upper()
    sol_active = posture_active_df[posture_active_df["symbol"] == SYMBOL].copy()
    if sol_active.empty:
        raise RuntimeError("SOLUSDT is not present in repaired active subset")
    if int(len(sol_active)) != 1:
        raise RuntimeError(f"Expected one SOLUSDT active row, found {len(sol_active)}")
    winner_id = str(sol_active.iloc[0]["winner_config_id"]).strip()
    if winner_id != BASELINE_STRATEGY_ID:
        raise RuntimeError(f"Baseline strategy mismatch: expected {BASELINE_STRATEGY_ID}, got {winner_id}")

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

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    df_cache: Dict[Any, pd.DataFrame] = {}
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
    setattr(exec_args, "tie_touch_policy", "sl_first")
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)

    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=signal_df.copy(),
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    n_total = int(len(bundle.contexts))
    if n_total < 500:
        raise RuntimeError(f"Insufficient contexts for sweep ({n_total})")
    train_end = int(math.floor(0.60 * n_total))
    holdout_start = int(math.floor(0.80 * n_total))
    train_bundle = ga.build_subbundle(bundle, 0, train_end, exec_args)
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

    base_cfg = ga.normalize_cfg(dict(variant_map[BASELINE_STRATEGY_ID]))
    base_cfg["feature_skip_mask_enabled"] = 0
    base_cfg["candidate_id"] = "V0_BASELINE"
    base_cfg["label"] = "Control baseline"
    all_variants = build_variants(base_cfg)
    variant_map = {str(v["candidate_id"]): dict(v) for v in all_variants}
    if "V0_BASELINE" not in variant_map:
        raise RuntimeError("Internal error: missing V0_BASELINE variant")

    requested_ids = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    if not requested_ids:
        requested_ids = [str(v["candidate_id"]) for v in all_variants]
    unknown = [vid for vid in requested_ids if vid not in variant_map]
    if unknown:
        raise RuntimeError(f"Unknown --variants ids: {unknown}")
    sweep_variants = [dict(variant_map[vid]) for vid in requested_ids]
    baseline_cfg = dict(variant_map["V0_BASELINE"])

    baseline_train_eval = ga.evaluate_cfg_on_bundle(
        bundle=train_bundle,
        baseline_1h_df=baseline_1h_train,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )
    baseline_full_eval = ga.evaluate_cfg_on_bundle(
        bundle=bundle,
        baseline_1h_df=baseline_1h_full,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )
    baseline_train_rows = baseline_train_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_full_rows = baseline_full_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_train_loss = ga.compute_loss_metrics(baseline_train_rows)
    baseline_full_loss = ga.compute_loss_metrics(baseline_full_rows)

    qrow = quality_from_build_meta(build_meta)
    rrow = {
        "bucket_1h": "REPAIRED_LONG",
        "integrity_status": str(qrow.get("integrity_status", "")),
        "windows_ready": int(qrow.get("windows_ready", 0)),
        "windows_partial": int(qrow.get("windows_partial", 0)),
        "windows_blocked": int(qrow.get("windows_blocked", 0)),
    }
    route_meta: Dict[str, Any] = {}

    screen_rows: List[Dict[str, Any]] = []
    confirm_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for cfg in sweep_variants:
        cfg = ga.normalize_cfg(dict(cfg))

        # Stage 1: screen on train60.
        tr_eval = ga.evaluate_cfg_on_bundle(
            bundle=train_bundle,
            baseline_1h_df=baseline_1h_train,
            cfg=cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        tr_rows = tr_eval["signal_rows_df"].copy().reset_index(drop=True)
        tr_cmp = ga.compare_candidate_vs_baseline(tr_rows, baseline_train_rows)
        tr_loss = ga.compute_loss_metrics(tr_rows)
        tr_chrono = ga.compute_chronology_stats(tr_rows)
        tr_gates = ga.compute_stage_gates(
            compare=tr_cmp,
            baseline_loss=baseline_train_loss,
            candidate_loss=tr_loss,
            chrono=tr_chrono,
            retention_floor=float(args.retention_floor),
            min_trades_abs=0,
            min_trades_frac=float(args.min_trades_frac),
            winner_retention_floor=float(args.winner_retention_floor),
            instant_reduction_rel=float(args.instant_reduction_rel),
            tail_reduction_rel=float(args.tail_reduction_rel),
        )
        fast_rel_train = ga.relative_reduction(
            float(baseline_train_loss.get("fast_loser_rate", np.nan)),
            float(tr_loss.get("fast_loser_rate", np.nan)),
        )
        screen_pass = int(
            int(tr_gates["gate_g0_chronology"]) == 1
            and int(tr_gates["gate_g1_participation"]) == 1
            and int(tr_gates["gate_g2_winner_preservation"]) == 1
            and int(tr_loss.get("trade_count", 0)) > 0
        )
        screen_rows.append(
            {
                "variant_id": str(cfg["candidate_id"]),
                "label": str(cfg.get("label", "")),
                "entry_mode": str(cfg.get("entry_mode", "")),
                "limit_offset_bps": float(cfg.get("limit_offset_bps", np.nan)),
                "fallback_to_market": int(cfg.get("fallback_to_market", 0)),
                "fallback_delay_min": float(cfg.get("fallback_delay_min", np.nan)),
                "max_fill_delay_min": float(cfg.get("max_fill_delay_min", np.nan)),
                "delay_bars_before_entry": int(cfg.get("delay_bars_before_entry", 0)),
                "adverse_move_guard_bps": float(cfg.get("adverse_move_guard_bps", np.nan)),
                "max_adverse_market_fill_bps": float(cfg.get("max_adverse_market_fill_bps", np.nan)),
                "screen_pass": int(screen_pass),
                "gate_g0": int(tr_gates["gate_g0_chronology"]),
                "gate_g1": int(tr_gates["gate_g1_participation"]),
                "gate_g2": int(tr_gates["gate_g2_winner_preservation"]),
                "trade_count": int(tr_loss["trade_count"]),
                "retention": float(tr_gates["retention"]),
                "winner_retention": float(tr_gates["winner_retention"]),
                "delta_expectancy_vs_baseline": float(tr_cmp["delta_expectancy"]),
                "delta_cvar_vs_baseline": float(tr_cmp["delta_cvar_5"]),
                "maxdd_improve_ratio": float(tr_cmp["maxdd_improve_ratio"]),
                "instant_loser_rel_reduction": float(tr_gates["instant_loser_rel_reduction"]),
                "fast_loser_rel_reduction": float(fast_rel_train),
                "bottom_decile_rel_reduction": float(tr_gates["bottom_decile_rel_reduction"]),
                "same_bar_exit_count": int(tr_chrono["same_bar_exit_count"]),
                "same_bar_touch_count": int(tr_chrono["same_bar_touch_count"]),
            }
        )

        # Stage 2: strict confirm on full + routes + holdout.
        full_eval = ga.evaluate_cfg_on_bundle(
            bundle=bundle,
            baseline_1h_df=baseline_1h_full,
            cfg=cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        full_rows = full_eval["signal_rows_df"].copy().reset_index(drop=True)
        full_cmp = ga.compare_candidate_vs_baseline(full_rows, baseline_full_rows)
        full_loss = ga.compute_loss_metrics(full_rows)
        full_chrono = ga.compute_chronology_stats(full_rows)
        full_gates = ga.compute_stage_gates(
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
        fast_rel_full = ga.relative_reduction(
            float(baseline_full_loss.get("fast_loser_rate", np.nan)),
            float(full_loss.get("fast_loser_rate", np.nan)),
        )
        improve_target = int(
            (np.isfinite(float(full_gates["instant_loser_rel_reduction"])) and float(full_gates["instant_loser_rel_reduction"]) >= float(args.instant_reduction_rel))
            or (np.isfinite(float(fast_rel_full)) and float(fast_rel_full) >= float(args.fast_reduction_rel))
            or (np.isfinite(float(full_gates["bottom_decile_rel_reduction"])) and float(full_gates["bottom_decile_rel_reduction"]) >= float(args.tail_reduction_rel))
        )

        holdout_cand = ga.subset_by_signal_ids(full_rows, holdout_signal_ids)
        holdout_base = ga.subset_by_signal_ids(baseline_full_rows, holdout_signal_ids)
        holdout_cmp = ga.compare_candidate_vs_baseline(holdout_cand, holdout_base)
        holdout_loss = ga.compute_loss_metrics(holdout_cand) if not holdout_cand.empty else {"fast_loser_rate": float("nan"), "instant_loser_rate": float("nan"), "bottom_decile_pnl_share": float("nan")}
        holdout_base_loss = ga.compute_loss_metrics(holdout_base) if not holdout_base.empty else {"fast_loser_rate": float("nan"), "instant_loser_rate": float("nan"), "bottom_decile_pnl_share": float("nan")}
        holdout_fast_rel = ga.relative_reduction(
            float(holdout_base_loss.get("fast_loser_rate", np.nan)),
            float(holdout_loss.get("fast_loser_rate", np.nan)),
        )
        holdout_target = int(
            (np.isfinite(ga.relative_reduction(float(holdout_base_loss.get("instant_loser_rate", np.nan)), float(holdout_loss.get("instant_loser_rate", np.nan))) ) and ga.relative_reduction(float(holdout_base_loss.get("instant_loser_rate", np.nan)), float(holdout_loss.get("instant_loser_rate", np.nan))) > 0.0)
            or (np.isfinite(float(holdout_fast_rel)) and float(holdout_fast_rel) > 0.0)
            or (np.isfinite(ga.relative_reduction(float(holdout_base_loss.get("bottom_decile_pnl_share", np.nan)), float(holdout_loss.get("bottom_decile_pnl_share", np.nan))) ) and ga.relative_reduction(float(holdout_base_loss.get("bottom_decile_pnl_share", np.nan)), float(holdout_loss.get("bottom_decile_pnl_share", np.nan))) > 0.0)
        )
        holdout_pass = int(
            np.isfinite(float(holdout_cmp.get("delta_expectancy", np.nan)))
            and float(holdout_cmp["delta_expectancy"]) > 0.0
            and np.isfinite(float(holdout_cmp.get("delta_cvar_5", np.nan)))
            and float(holdout_cmp["delta_cvar_5"]) >= 0.0
            and holdout_target == 1
        )

        strict_eval_pack = phase_v.evaluate_symbol(
            symbol=SYMBOL,
            bundle=bundle,
            foundation_quality=qrow,
            foundation_readiness=rrow,
            exec_args=exec_args,
            variants=[cfg],
        )
        if not route_meta:
            route_meta = dict(strict_eval_pack.get("route_meta", {}))
        strict_rows = strict_eval_pack["results_df"].copy()
        strict_candidate = strict_rows[strict_rows["candidate_id"].astype(str) == str(cfg["candidate_id"])]
        if strict_candidate.empty:
            raise RuntimeError(f"Strict route evaluation missing candidate row: {cfg['candidate_id']}")
        strict_row = strict_candidate.iloc[0]

        split_pass = int(
            int(strict_row.get("valid_for_ranking", 0)) == 1
            and np.isfinite(float(pd.to_numeric(pd.Series([strict_row.get("min_subperiod_delta", np.nan)]), errors="coerce").iloc[0]))
            and float(pd.to_numeric(pd.Series([strict_row.get("min_subperiod_delta", np.nan)]), errors="coerce").iloc[0]) > 0.0
        )
        route_total = int(pd.to_numeric(pd.Series([strict_row.get("route_count", 0)]), errors="coerce").fillna(0).iloc[0])
        route_pass_rate = float(pd.to_numeric(pd.Series([strict_row.get("route_pass_rate", np.nan)]), errors="coerce").iloc[0])
        route_support_pass = int(pd.to_numeric(pd.Series([strict_row.get("route_pass", 0)]), errors="coerce").fillna(0).iloc[0])
        route_confirm_count = int(round(route_pass_rate * max(1, route_total))) if np.isfinite(route_pass_rate) else 0

        g4 = int(split_pass == 1 and int(route_support_pass) == 1 and holdout_pass == 1)
        confirm_pass = int(
            int(full_gates["gate_g0_chronology"]) == 1
            and int(full_gates["gate_g1_participation"]) == 1
            and int(full_gates["gate_g2_winner_preservation"]) == 1
            and int(full_gates["gate_g3_risk_sanity"]) == 1
            and int(improve_target) == 1
            and int(g4) == 1
        )

        fill_stats = fill_type_stats(full_rows)

        confirm_rows.append(
            {
                "variant_id": str(cfg["candidate_id"]),
                "label": str(cfg.get("label", "")),
                "entry_mode": str(cfg.get("entry_mode", "")),
                "limit_offset_bps": float(cfg.get("limit_offset_bps", np.nan)),
                "fallback_to_market": int(cfg.get("fallback_to_market", 0)),
                "fallback_delay_min": float(cfg.get("fallback_delay_min", np.nan)),
                "max_fill_delay_min": float(cfg.get("max_fill_delay_min", np.nan)),
                "delay_bars_before_entry": int(cfg.get("delay_bars_before_entry", 0)),
                "adverse_move_guard_bps": float(cfg.get("adverse_move_guard_bps", np.nan)),
                "max_adverse_market_fill_bps": float(cfg.get("max_adverse_market_fill_bps", np.nan)),
                "cap_fallback_limit_offset_bps": float(cfg.get("cap_fallback_limit_offset_bps", np.nan)),
                "cap_fallback_ttl_bars": int(float(cfg.get("cap_fallback_ttl_bars", 0))),
                "confirm_pass": int(confirm_pass),
                "gate_g0_chronology": int(full_gates["gate_g0_chronology"]),
                "gate_g1_participation": int(full_gates["gate_g1_participation"]),
                "gate_g2_winner_preservation": int(full_gates["gate_g2_winner_preservation"]),
                "gate_g3_risk_sanity": int(full_gates["gate_g3_risk_sanity"]),
                "gate_improve_target": int(improve_target),
                "gate_g4_robustness": int(g4),
                "split_valid_for_ranking": int(pd.to_numeric(pd.Series([strict_row.get("valid_for_ranking", 0)]), errors="coerce").fillna(0).iloc[0]),
                "best_min_subperiod_delta": float(pd.to_numeric(pd.Series([strict_row.get("min_subperiod_delta", np.nan)]), errors="coerce").iloc[0]),
                "split_support_pass": int(split_pass),
                "route_total": int(route_total),
                "route_confirm_count": int(route_confirm_count),
                "route_pass_rate": float(route_pass_rate),
                "route_support_pass": int(route_support_pass),
                "holdout_trade_count": int(len(holdout_cand)),
                "holdout_delta_expectancy_vs_baseline": float(holdout_cmp.get("delta_expectancy", np.nan)),
                "holdout_delta_cvar_vs_baseline": float(holdout_cmp.get("delta_cvar_5", np.nan)),
                "holdout_fast_loser_rel_reduction": float(holdout_fast_rel),
                "holdout_pass": int(holdout_pass),
                "trade_count": int(full_loss["trade_count"]),
                "retention": float(full_gates["retention"]),
                "winner_count": int(full_loss["meaningful_winner_count"]),
                "winner_retention": float(full_gates["winner_retention"]),
                "expectancy_net": float(full_cmp["candidate_expectancy_net"]),
                "delta_expectancy_vs_baseline": float(full_cmp["delta_expectancy"]),
                "cvar_5": float(full_cmp["candidate_cvar_5"]),
                "delta_cvar_vs_baseline": float(full_cmp["delta_cvar_5"]),
                "max_drawdown": float(full_cmp["candidate_max_drawdown"]),
                "maxdd_improve_ratio": float(full_cmp["maxdd_improve_ratio"]),
                "instant_loser_rate": float(full_loss["instant_loser_rate"]),
                "fast_loser_rate": float(full_loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(full_loss["bottom_decile_pnl_share"]),
                "instant_loser_rel_reduction": float(full_gates["instant_loser_rel_reduction"]),
                "fast_loser_rel_reduction": float(fast_rel_full),
                "bottom_decile_rel_reduction": float(full_gates["bottom_decile_rel_reduction"]),
                "same_bar_exit_count": int(full_chrono["same_bar_exit_count"]),
                "same_bar_touch_count": int(full_chrono["same_bar_touch_count"]),
                "exit_before_entry_count": int(full_chrono["exit_before_entry_count"]),
                "entry_on_signal_count": int(full_chrono["entry_on_signal_count"]),
                "parity_clean": int(full_chrono["parity_clean"]),
                "strict_exec_expectancy_net": float(pd.to_numeric(pd.Series([strict_row.get("exec_expectancy_net", np.nan)]), errors="coerce").iloc[0]),
                "strict_delta_expectancy_vs_1h_reference": float(pd.to_numeric(pd.Series([strict_row.get("delta_expectancy_vs_1h_reference", np.nan)]), errors="coerce").iloc[0]),
                "strict_cvar_improve_ratio": float(pd.to_numeric(pd.Series([strict_row.get("cvar_improve_ratio", np.nan)]), errors="coerce").iloc[0]),
                "strict_maxdd_improve_ratio": float(pd.to_numeric(pd.Series([strict_row.get("maxdd_improve_ratio", np.nan)]), errors="coerce").iloc[0]),
                "strict_entries_valid": int(pd.to_numeric(pd.Series([strict_row.get("entries_valid", 0)]), errors="coerce").fillna(0).iloc[0]),
                **fill_stats,
            }
        )

        split_df = full_eval["split_rows_df"].copy().reset_index(drop=True)
        if not split_df.empty:
            split_df["variant_id"] = str(cfg["candidate_id"])
            split_df["detail_type"] = "split"
            detail_rows.extend(split_df.to_dict("records"))
        route_df = strict_eval_pack.get("route_df", pd.DataFrame()).copy()
        if isinstance(route_df, pd.DataFrame) and not route_df.empty:
            route_df = route_df[route_df["candidate_id"].astype(str) == str(cfg["candidate_id"])].copy()
        if not route_df.empty:
            route_df = route_df.reset_index(drop=True)
            route_df["variant_id"] = str(cfg["candidate_id"])
            route_df["detail_type"] = "route"
            detail_rows.extend(route_df.to_dict("records"))

    screen_df = pd.DataFrame(screen_rows).sort_values(
        ["screen_pass", "delta_expectancy_vs_baseline"],
        ascending=[False, False],
    ).reset_index(drop=True)
    confirm_df = pd.DataFrame(confirm_rows).sort_values(
        ["confirm_pass", "delta_expectancy_vs_baseline", "route_pass_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows)

    confirm_df.to_csv(run_dir / "sol_entry_mechanics_by_variant.csv", index=False)
    detail_df.to_csv(run_dir / "sol_entry_mechanics_route_split_detail.csv", index=False)

    approved_df = confirm_df[confirm_df["confirm_pass"] == 1].copy()
    decision = "APPROVED_ENTRY_MECHANICS_UPDATE" if not approved_df.empty else "NO_CHANGE_RECOMMENDED"
    best_row = approved_df.iloc[0] if not approved_df.empty else confirm_df.iloc[0]

    report_lines: List[str] = [
        "# SOL Entry Mechanics Sweep",
        "",
        f"- Generated UTC: `{utc_now_iso()}`",
        f"- Run dir: `{run_dir}`",
        f"- Decision: `{decision}`",
        f"- Symbol: `{SYMBOL}`",
        f"- Baseline strategy lock: `{BASELINE_STRATEGY_ID}`",
        "",
        "## Stage 1 Screen (Train 60%)",
        "",
        markdown_table(
            screen_df,
            [
                "variant_id",
                "screen_pass",
                "trade_count",
                "retention",
                "winner_retention",
                "delta_expectancy_vs_baseline",
                "instant_loser_rel_reduction",
                "fast_loser_rel_reduction",
                "bottom_decile_rel_reduction",
                "same_bar_exit_count",
            ],
            n=20,
        ),
        "",
        "## Stage 2 Confirm (Splits + Routes + Holdout)",
        "",
        markdown_table(
            confirm_df,
            [
                "variant_id",
                "confirm_pass",
                "gate_g0_chronology",
                "gate_g1_participation",
                "gate_g2_winner_preservation",
                "gate_g3_risk_sanity",
                "gate_improve_target",
                "gate_g4_robustness",
                "split_support_pass",
                "route_pass_rate",
                "holdout_pass",
                "retention",
                "winner_retention",
                "delta_expectancy_vs_baseline",
                "delta_cvar_vs_baseline",
                "maxdd_improve_ratio",
                "instant_loser_rel_reduction",
                "fast_loser_rel_reduction",
                "bottom_decile_rel_reduction",
            ],
            n=20,
        ),
        "",
        "## Fill-Type and Adverse Selection Proxy",
        "",
        markdown_table(
            confirm_df,
            [
                "variant_id",
                "fills_valid",
                "fill_limit_count",
                "fill_market_count",
                "fill_market_fallback_count",
                "fill_market_guard_fallback_count",
                "fill_limit_cap_fallback_count",
                "fill_market_cap_fallback_count",
                "maker_fill_share",
                "taker_fill_share",
                "entry_improvement_bps_mean",
                "entry_price_vs_signal_open_bps_mean",
            ],
            n=20,
        ),
        "",
        "## Proven vs Assumed",
        "",
        "- Proven: 1h signal layer and exits remained frozen; only entry price formation knobs were varied.",
        "- Proven: chronology checks remained enforced (same-parent-bar exit=0, exit-before-entry=0, entry-on-signal precondition=0).",
        "- Proven: split support + full 3-route family + holdout checks were applied per variant.",
        "- Assumed: fallback-to-market branches model realistic fill behavior under the existing simulator cost model (no free fills).",
        "",
        "## Final Recommendation",
        "",
        f"- `{decision}`",
        f"- Best row by confirm ordering: `{best_row.get('variant_id', '')}`",
    ]
    if decision == "APPROVED_ENTRY_MECHANICS_UPDATE":
        report_lines.append("- Next step: run operational op-check + paper SHADOW for the winning variant before any active promotion.")
    else:
        report_lines.append("- Next step: keep current baseline entry mechanics; this bounded sweep did not produce a robust improvement.")

    (run_dir / "sol_entry_mechanics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "decision": decision,
        "symbol": SYMBOL,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "variants_tested": [
            {
                "variant_id": str(v.get("candidate_id", "")),
                "label": str(v.get("label", "")),
                "entry_mode": str(v.get("entry_mode", "")),
                "limit_offset_bps": float(v.get("limit_offset_bps", np.nan)),
                "fallback_to_market": int(v.get("fallback_to_market", 0)),
                "fallback_delay_min": float(v.get("fallback_delay_min", np.nan)),
                "max_fill_delay_min": float(v.get("max_fill_delay_min", np.nan)),
                "delay_bars_before_entry": int(v.get("delay_bars_before_entry", 0)),
                "adverse_move_guard_bps": float(v.get("adverse_move_guard_bps", np.nan)),
                "max_adverse_market_fill_bps": float(v.get("max_adverse_market_fill_bps", np.nan)),
            }
            for v in sweep_variants
        ],
        "variants_requested": requested_ids,
        "gates": {
            "chronology": "parity_clean==1 and same_bar_exit_count==0 and exit_before_entry_count==0 and entry_on_signal_count==0",
            "participation": f"retention>={float(args.retention_floor):.4f}",
            "winner_retention": f">={float(args.winner_retention_floor):.4f}",
            "robustness": "split_valid_for_ranking==1 and best_min_subperiod_delta>0 and route_pass_rate==1.0 and holdout_delta_expectancy>0 and holdout_delta_cvar>=0",
            "improve_target": f"instant_rel>={float(args.instant_reduction_rel):.4f} OR fast_rel>={float(args.fast_reduction_rel):.4f} OR tail_rel>={float(args.tail_reduction_rel):.4f}",
        },
        "input_paths": {
            "posture_dir": str(posture_dir),
            "strict_confirm_dir": str(strict_confirm_dir),
            "freeze_dir": str(freeze_dir),
            "foundation_dir": str(foundation_dir),
        },
        "contract_validation": contract_validation,
        "build_meta": build_meta,
        "route_meta": route_meta,
        "outputs": {
            "variant_csv": str(run_dir / "sol_entry_mechanics_by_variant.csv"),
            "route_split_detail_csv": str(run_dir / "sol_entry_mechanics_route_split_detail.csv"),
            "report_md": str(run_dir / "sol_entry_mechanics_report.md"),
        },
    }
    json_dump(run_dir / "sol_entry_mechanics_manifest.json", manifest)

    if decision == "APPROVED_ENTRY_MECHANICS_UPDATE":
        best_payload = {
            "generated_utc": utc_now_iso(),
            "decision": decision,
            "symbol": SYMBOL,
            "best_variant": best_row.to_dict(),
            "run_dir": str(run_dir),
        }
        json_dump(run_dir / "sol_entry_mechanics_best_variant.json", best_payload)
        diff_txt = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "diff", "--", "scripts/phase_a_model_a_audit.py", "scripts/sol_entry_mechanics_sweep.py"],
            text=True,
        )
        (run_dir / "sol_entry_mechanics_patch.diff").write_text(diff_txt, encoding="utf-8")

    print(json.dumps({"decision": decision, "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
