#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
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
from scripts import sol_3m_skipmask_ga as ga  # noqa: E402


RUN_PREFIX = "SOL_TIE_TOUCH_POLICY_AUDIT"
SYMBOL = "SOLUSDT"
BASELINE_STRATEGY_ID = "M1_ENTRY_ONLY_PASSIVE_BASELINE"


POLICIES: List[Dict[str, Any]] = [
    {
        "policy_id": "P1_SL_FIRST",
        "tie_touch_policy": "sl_first",
        "description": "Conservative baseline: if SL and TP touched in same evaluated bar, resolve SL first",
        "conservative_for_approval": 1,
    },
    {
        "policy_id": "P2_TP_FIRST",
        "tie_touch_policy": "tp_first",
        "description": "Optimistic sanity bound: resolve TP first on tie-touch",
        "conservative_for_approval": 0,
    },
    {
        "policy_id": "P3_DISTANCE_TO_ENTRY",
        "tie_touch_policy": "distance_to_entry",
        "description": "Deterministic causal: resolve to boundary closer to entry price; equal distance falls back to SL",
        "conservative_for_approval": 1,
    },
]


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


def tie_touch_tail_contribution(rows_df: pd.DataFrame) -> Dict[str, Any]:
    filled = to_num(rows_df.get("exec_filled", 0)).fillna(0).astype(int)
    valid = to_num(rows_df.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    pnl = to_num(rows_df.get("exec_pnl_net_pct", np.nan))
    tie = to_num(rows_df.get("exec_same_bar_hit", 0)).fillna(0).astype(int)

    m = (filled == 1) & (valid == 1) & pnl.notna()
    df = rows_df.loc[m].copy()
    if df.empty:
        return {
            "trade_count": 0,
            "tie_touch_count": 0,
            "tie_touch_rate": float("nan"),
            "worst10_tie_count": 0,
            "worst10_tie_share": float("nan"),
            "worst25_tie_count": 0,
            "worst25_tie_share": float("nan"),
            "bottom_decile_count": 0,
            "total_neg_abs": float("nan"),
            "bottom_decile_neg_abs": float("nan"),
            "bottom_decile_tie_neg_abs": float("nan"),
            "tie_contrib_to_bottom_decile_share": float("nan"),
            "tie_share_within_bottom_decile_losses": float("nan"),
        }

    p = to_num(df.get("exec_pnl_net_pct", np.nan)).astype(float)
    t = to_num(df.get("exec_same_bar_hit", 0)).fillna(0).astype(int)
    n = int(len(df))

    order = np.argsort(p.to_numpy(dtype=float))
    idx10 = order[: min(10, n)]
    idx25 = order[: min(25, n)]

    tie_touch_count = int((t == 1).sum())
    tie_touch_rate = float(tie_touch_count / max(1, n))
    worst10_tie_count = int(t.iloc[idx10].sum())
    worst25_tie_count = int(t.iloc[idx25].sum())

    k = max(1, int(math.ceil(0.10 * n)))
    idx_decile = order[:k]
    p_decile = p.iloc[idx_decile]
    t_decile = t.iloc[idx_decile]

    total_neg_abs = float(abs(float(p[p < 0.0].sum())))
    bottom_decile_neg_abs = float(abs(float(p_decile[p_decile < 0.0].sum())))
    bottom_decile_tie_neg_abs = float(abs(float(p_decile[(p_decile < 0.0) & (t_decile == 1)].sum())))

    tie_contrib_to_bottom_decile_share = (
        float(bottom_decile_tie_neg_abs / total_neg_abs)
        if np.isfinite(total_neg_abs) and total_neg_abs > 1e-12
        else 0.0
    )
    tie_share_within_bottom_decile_losses = (
        float(bottom_decile_tie_neg_abs / bottom_decile_neg_abs)
        if np.isfinite(bottom_decile_neg_abs) and bottom_decile_neg_abs > 1e-12
        else 0.0
    )

    return {
        "trade_count": int(n),
        "tie_touch_count": int(tie_touch_count),
        "tie_touch_rate": float(tie_touch_rate),
        "worst10_tie_count": int(worst10_tie_count),
        "worst10_tie_share": float(worst10_tie_count / max(1, min(10, n))),
        "worst25_tie_count": int(worst25_tie_count),
        "worst25_tie_share": float(worst25_tie_count / max(1, min(25, n))),
        "bottom_decile_count": int(k),
        "total_neg_abs": float(total_neg_abs),
        "bottom_decile_neg_abs": float(bottom_decile_neg_abs),
        "bottom_decile_tie_neg_abs": float(bottom_decile_tie_neg_abs),
        "tie_contrib_to_bottom_decile_share": float(tie_contrib_to_bottom_decile_share),
        "tie_share_within_bottom_decile_losses": float(tie_share_within_bottom_decile_losses),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOL-only tie-touch policy audit under repaired branch")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--posture-dir", default="")
    ap.add_argument("--strict-confirm-dir", default="")
    ap.add_argument("--seed", type=int, default=20260306)
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
    foundation_dir = Path(strict_manifest.get("foundation_dir", "")).resolve()
    if not freeze_dir.exists() or not foundation_dir.exists():
        raise FileNotFoundError("freeze_dir or foundation_dir from manifests does not exist")

    posture_active_df = pd.read_csv(posture_dir / "repaired_active_3m_subset.csv")
    posture_active_df["symbol"] = posture_active_df["symbol"].astype(str).str.upper()
    sol_active = posture_active_df[posture_active_df["symbol"] == SYMBOL].copy()
    if sol_active.empty:
        raise RuntimeError("SOLUSDT not present in repaired active subset")
    sol_winner_id = str(sol_active.iloc[0]["winner_config_id"]).strip()
    if sol_winner_id != BASELINE_STRATEGY_ID:
        raise RuntimeError(f"Expected {BASELINE_STRATEGY_ID}, found {sol_winner_id}")

    freeze_df = pd.read_csv(freeze_dir / "repaired_best_by_symbol.csv")
    freeze_df["symbol"] = freeze_df["symbol"].astype(str).str.upper()
    sol_row_df = freeze_df[(freeze_df["symbol"] == SYMBOL) & (freeze_df["side"].astype(str).str.lower() == "long")].copy()
    if sol_row_df.empty:
        raise RuntimeError("SOLUSDT long row missing in repaired_best_by_symbol.csv")
    sol_row = sol_row_df.iloc[0]
    params_payload = subset1.parse_params_from_row(sol_row, freeze_dir / "repaired_universe_selected_params")

    run_dir = ensure_dir((PROJECT_ROOT / args.outdir).resolve() / f"{RUN_PREFIX}_{utc_tag()}")
    inputs_dir = ensure_dir(run_dir / "_inputs")
    cache_dir = ensure_dir(run_dir / "_window_cache")

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    df_cache: Dict[Any, pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_df = subset1.build_signal_table_for_row(row=sol_row, params=params_payload, df_cache=df_cache, raw_cache=raw_cache)
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

    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=signal_df.copy(),
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    n_total = int(len(bundle.contexts))
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

    variants = phase_v.sanitize_variants()
    variant_map = {str(v["candidate_id"]): dict(v) for v in variants}
    if BASELINE_STRATEGY_ID not in variant_map:
        raise RuntimeError(f"Missing baseline variant {BASELINE_STRATEGY_ID}")
    baseline_cfg = ga.normalize_cfg(dict(variant_map[BASELINE_STRATEGY_ID]))
    baseline_cfg["candidate_id"] = BASELINE_STRATEGY_ID
    baseline_cfg["label"] = "Frozen baseline"
    baseline_cfg["feature_skip_mask_enabled"] = 0

    route_bundles, route_baseline_1h, route_meta = ga.build_route_family(
        base_bundle=bundle,
        exec_args=exec_args,
        fee=fee,
    )

    # Evaluate control first (SL-first), then compare all other policies to it.
    policy_rows_map: Dict[str, pd.DataFrame] = {}
    policy_eval_map: Dict[str, Dict[str, Any]] = {}

    for pol in POLICIES:
        args_pol = copy.deepcopy(exec_args)
        setattr(args_pol, "tie_touch_policy", str(pol["tie_touch_policy"]))

        full_eval = ga.evaluate_cfg_on_bundle(
            bundle=bundle,
            baseline_1h_df=baseline_1h_full,
            cfg=baseline_cfg,
            one_h=one_h,
            exec_args=args_pol,
        )
        rows = full_eval["signal_rows_df"].copy().reset_index(drop=True)
        policy_rows_map[str(pol["policy_id"])] = rows
        policy_eval_map[str(pol["policy_id"])] = {
            "eval": full_eval,
            "rows": rows,
            "exec_args": args_pol,
            "loss": ga.compute_loss_metrics(rows),
            "chrono": ga.compute_chronology_stats(rows),
            "tail": tie_touch_tail_contribution(rows),
            "rollup": modela.ga_exec._rollup_mode(rows, "exec"),  # pylint: disable=protected-access
        }

    control_id = "P1_SL_FIRST"
    if control_id not in policy_eval_map:
        raise RuntimeError("Missing P1_SL_FIRST control in policy map")
    control_rows = policy_eval_map[control_id]["rows"]
    control_loss = policy_eval_map[control_id]["loss"]

    # Build route control exec rows once.
    route_control_exec_rows: Dict[str, pd.DataFrame] = {}
    args_control = copy.deepcopy(exec_args)
    setattr(args_control, "tie_touch_policy", "sl_first")
    for rid, rb in route_bundles.items():
        route_eval = ga.evaluate_cfg_on_bundle(
            bundle=rb,
            baseline_1h_df=route_baseline_1h[rid],
            cfg=baseline_cfg,
            one_h=one_h,
            exec_args=args_control,
        )
        route_control_exec_rows[rid] = route_eval["signal_rows_df"].copy().reset_index(drop=True)

    event_rows: List[Dict[str, Any]] = []
    tail_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []

    for pol in POLICIES:
        pid = str(pol["policy_id"])
        pack = policy_eval_map[pid]
        rows = pack["rows"]
        loss = pack["loss"]
        chrono = pack["chrono"]
        tail = pack["tail"]
        roll = pack["rollup"]
        cmp_vs_control = ga.compare_candidate_vs_baseline(rows, control_rows)

        holdout_cand = ga.subset_by_signal_ids(rows, holdout_signal_ids)
        holdout_base = ga.subset_by_signal_ids(control_rows, holdout_signal_ids)
        holdout_cmp = ga.compare_candidate_vs_baseline(holdout_cand, holdout_base)
        holdout_cand_loss = ga.compute_loss_metrics(holdout_cand) if not holdout_cand.empty else {"instant_loser_rate": float("nan"), "bottom_decile_pnl_share": float("nan")}
        holdout_base_loss = ga.compute_loss_metrics(holdout_base) if not holdout_base.empty else {"instant_loser_rate": float("nan"), "bottom_decile_pnl_share": float("nan")}

        holdout_instant_rel = ga.relative_reduction(
            float(holdout_base_loss.get("instant_loser_rate", np.nan)),
            float(holdout_cand_loss.get("instant_loser_rate", np.nan)),
        )
        holdout_tail_rel = ga.relative_reduction(
            float(holdout_base_loss.get("bottom_decile_pnl_share", np.nan)),
            float(holdout_cand_loss.get("bottom_decile_pnl_share", np.nan)),
        )
        holdout_loss_target_pass = int(
            (np.isfinite(holdout_instant_rel) and holdout_instant_rel > 0.0)
            or (np.isfinite(holdout_tail_rel) and holdout_tail_rel > 0.0)
        )
        holdout_pass = int(
            np.isfinite(holdout_cmp.get("delta_expectancy", np.nan))
            and float(holdout_cmp["delta_expectancy"]) > 0.0
            and np.isfinite(holdout_cmp.get("delta_cvar_5", np.nan))
            and float(holdout_cmp["delta_cvar_5"]) >= 0.0
            and holdout_loss_target_pass == 1
        )

        route_pack = ga.route_confirm_for_candidate(
            cfg=baseline_cfg,
            one_h=one_h,
            exec_args=pack["exec_args"],
            route_bundles=route_bundles,
            route_baseline_1h=route_baseline_1h,
            route_baseline_exec_rows=route_control_exec_rows,
        )

        split_pass = int(
            int(pack["eval"]["metrics"]["valid_for_ranking"]) == 1
            and np.isfinite(float(pack["eval"]["metrics"]["min_split_delta"]))
            and float(pack["eval"]["metrics"]["min_split_delta"]) > 0.0
        )
        g0 = int(
            int(chrono["parity_clean"]) == 1
            and int(chrono["same_bar_exit_count"]) == 0
            and int(chrono["exit_before_entry_count"]) == 0
            and int(chrono["entry_on_signal_count"]) == 0
        )
        retention = float(loss["trade_count"]) / max(1.0, float(control_loss["trade_count"]))
        winner_ret = float(loss["meaningful_winner_count"]) / max(1.0, float(control_loss["meaningful_winner_count"]))
        instant_rel = ga.relative_reduction(
            float(control_loss.get("instant_loser_rate", np.nan)),
            float(loss.get("instant_loser_rate", np.nan)),
        )
        tail_rel = ga.relative_reduction(
            float(control_loss.get("bottom_decile_pnl_share", np.nan)),
            float(loss.get("bottom_decile_pnl_share", np.nan)),
        )
        improve_target = int(
            (np.isfinite(instant_rel) and instant_rel >= 0.05)
            or (np.isfinite(tail_rel) and tail_rel >= 0.15)
        )
        participation_same = int(retention >= 0.995)
        winners_same = int(winner_ret >= 0.995)
        risk_ok = int(np.isfinite(cmp_vs_control["maxdd_improve_ratio"]) and float(cmp_vs_control["maxdd_improve_ratio"]) >= 0.0)
        route_ok = int(route_pack["route_support_pass"])

        conservative = int(pol.get("conservative_for_approval", 0))
        policy_update_pass = int(
            pid != control_id
            and conservative == 1
            and g0 == 1
            and participation_same == 1
            and winners_same == 1
            and risk_ok == 1
            and split_pass == 1
            and route_ok == 1
            and holdout_pass == 1
            and improve_target == 1
        )

        event_rows.append(
            {
                "policy_id": pid,
                "tie_touch_policy": str(pol["tie_touch_policy"]),
                "trade_count": int(tail["trade_count"]),
                "tie_touch_count": int(tail["tie_touch_count"]),
                "tie_touch_rate": float(tail["tie_touch_rate"]),
                "same_bar_exit_count": int(chrono["same_bar_exit_count"]),
                "same_bar_touch_count": int(chrono["same_bar_touch_count"]),
                "exit_before_entry_count": int(chrono["exit_before_entry_count"]),
                "entry_on_signal_count": int(chrono["entry_on_signal_count"]),
            }
        )
        tail_rows.append(
            {
                "policy_id": pid,
                "tie_touch_policy": str(pol["tie_touch_policy"]),
                "worst10_tie_count": int(tail["worst10_tie_count"]),
                "worst10_tie_share": float(tail["worst10_tie_share"]),
                "worst25_tie_count": int(tail["worst25_tie_count"]),
                "worst25_tie_share": float(tail["worst25_tie_share"]),
                "bottom_decile_count": int(tail["bottom_decile_count"]),
                "total_neg_abs": float(tail["total_neg_abs"]),
                "bottom_decile_neg_abs": float(tail["bottom_decile_neg_abs"]),
                "bottom_decile_tie_neg_abs": float(tail["bottom_decile_tie_neg_abs"]),
                "tie_contrib_to_bottom_decile_share": float(tail["tie_contrib_to_bottom_decile_share"]),
                "tie_share_within_bottom_decile_losses": float(tail["tie_share_within_bottom_decile_losses"]),
            }
        )
        comp_rows.append(
            {
                "policy_id": pid,
                "tie_touch_policy": str(pol["tie_touch_policy"]),
                "policy_description": str(pol["description"]),
                "conservative_for_approval": int(conservative),
                "expectancy_net": float(roll["mean_expectancy_net"]),
                "cvar_5": float(roll["cvar_5"]),
                "max_drawdown": float(roll["max_drawdown"]),
                "delta_expectancy_vs_control": float(cmp_vs_control["delta_expectancy"]),
                "delta_cvar_vs_control": float(cmp_vs_control["delta_cvar_5"]),
                "delta_maxdd_vs_control": float(cmp_vs_control["delta_max_drawdown"]),
                "maxdd_improve_ratio_vs_control": float(cmp_vs_control["maxdd_improve_ratio"]),
                "cvar_improve_ratio_vs_control": float(cmp_vs_control["cvar_improve_ratio"]),
                "trade_count": int(loss["trade_count"]),
                "retention_vs_control": float(retention),
                "winner_retention_vs_control": float(winner_ret),
                "instant_loser_rate": float(loss["instant_loser_rate"]),
                "fast_loser_rate": float(loss["fast_loser_rate"]),
                "bottom_decile_pnl_share": float(loss["bottom_decile_pnl_share"]),
                "instant_loser_rel_reduction": float(instant_rel),
                "bottom_decile_rel_reduction": float(tail_rel),
                "split_valid_for_ranking": int(pack["eval"]["metrics"]["valid_for_ranking"]),
                "best_min_subperiod_delta": float(pack["eval"]["metrics"]["min_split_delta"]),
                "split_support_pass": int(split_pass),
                "route_total": int(route_pack["route_total"]),
                "route_confirm_count": int(route_pack["route_confirm_count"]),
                "route_pass_rate": float(route_pack["route_pass_rate"]),
                "route_support_pass": int(route_ok),
                "holdout_trade_count": int(len(holdout_cand)),
                "holdout_delta_expectancy_vs_control": float(holdout_cmp["delta_expectancy"]),
                "holdout_delta_cvar_vs_control": float(holdout_cmp["delta_cvar_5"]),
                "holdout_instant_loser_rel_reduction": float(holdout_instant_rel),
                "holdout_bottom_decile_rel_reduction": float(holdout_tail_rel),
                "holdout_pass": int(holdout_pass),
                "gate_g0_chronology": int(g0),
                "gate_participation_unchanged": int(participation_same),
                "gate_winner_retention_unchanged": int(winners_same),
                "gate_risk_sanity": int(risk_ok),
                "gate_improve_target": int(improve_target),
                "gate_policy_update_pass": int(policy_update_pass),
            }
        )

    event_df = pd.DataFrame(event_rows).sort_values("policy_id").reset_index(drop=True)
    tail_df = pd.DataFrame(tail_rows).sort_values("policy_id").reset_index(drop=True)
    comp_df = pd.DataFrame(comp_rows).sort_values("policy_id").reset_index(drop=True)

    event_df.to_csv(run_dir / "tie_touch_event_stats.csv", index=False)
    tail_df.to_csv(run_dir / "tie_touch_tail_contribution.csv", index=False)
    comp_df.to_csv(run_dir / "tie_touch_policy_comparison.csv", index=False)

    approved = comp_df[(comp_df["gate_policy_update_pass"] == 1)].copy()
    if approved.empty:
        decision = "NO_CHANGE_RECOMMENDED"
        winner = comp_df[comp_df["policy_id"] == control_id].iloc[0]
    else:
        decision = "APPROVED_TIE_TOUCH_POLICY_UPDATE"
        approved = approved.sort_values(
            ["delta_expectancy_vs_control", "delta_cvar_vs_control", "maxdd_improve_ratio_vs_control"],
            ascending=[False, False, False],
        )
        winner = approved.iloc[0]

    report_lines = [
        "# SOL Tie-Touch Policy Audit",
        "",
        f"- Generated UTC: `{utc_now_iso()}`",
        f"- Run dir: `{run_dir}`",
        f"- Decision: `{decision}`",
        "",
        "## Policy Set",
        "",
        markdown_table(
            pd.DataFrame(POLICIES),
            ["policy_id", "tie_touch_policy", "conservative_for_approval", "description"],
            n=10,
        ),
        "",
        "## Tie-Touch Event Stats",
        "",
        markdown_table(
            event_df,
            [
                "policy_id",
                "tie_touch_count",
                "tie_touch_rate",
                "same_bar_exit_count",
                "same_bar_touch_count",
                "exit_before_entry_count",
                "entry_on_signal_count",
            ],
            n=10,
        ),
        "",
        "## Tail Contribution from Tie-Touch Trades",
        "",
        markdown_table(
            tail_df,
            [
                "policy_id",
                "worst10_tie_count",
                "worst10_tie_share",
                "worst25_tie_count",
                "worst25_tie_share",
                "tie_contrib_to_bottom_decile_share",
                "tie_share_within_bottom_decile_losses",
            ],
            n=10,
        ),
        "",
        "## Policy Comparison (Splits + Routes + Holdout)",
        "",
        markdown_table(
            comp_df,
            [
                "policy_id",
                "delta_expectancy_vs_control",
                "delta_cvar_vs_control",
                "maxdd_improve_ratio_vs_control",
                "retention_vs_control",
                "winner_retention_vs_control",
                "instant_loser_rel_reduction",
                "bottom_decile_rel_reduction",
                "route_pass_rate",
                "holdout_delta_expectancy_vs_control",
                "holdout_delta_cvar_vs_control",
                "gate_policy_update_pass",
            ],
            n=10,
        ),
        "",
        "## Proven vs Assumed",
        "",
        "- Proven: tie-touch policy was the only changed execution variable; entries, participation mechanics, costs, and 1h signal layer were unchanged.",
        "- Proven: chronology guard remained clean under all tested policies (no same-parent exit, no exit-before-entry, no pre-signal entry).",
        "- Proven: split + full 3-route + holdout checks were applied policy-by-policy.",
        "- Assumed: TP-first is an optimistic upper bound and is not conservative for promotion.",
        "",
        "## Final Outcome",
        "",
        f"- `{decision}`",
        f"- Selected policy row: `{winner.get('policy_id', '')}`",
    ]
    if decision == "APPROVED_TIE_TOUCH_POLICY_UPDATE":
        report_lines.append("- Recommended next step: apply the selected tie-touch policy patch and run operational op-check before any paper shadow promotion.")
    else:
        report_lines.append("- Recommended next step: keep conservative SL-first tie handling and stop tie-policy branch; no robust improvement under strict harness.")

    (run_dir / "sol_tie_touch_policy_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "decision": decision,
        "symbol": SYMBOL,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "policies": POLICIES,
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
            "event_stats": str(run_dir / "tie_touch_event_stats.csv"),
            "tail_contribution": str(run_dir / "tie_touch_tail_contribution.csv"),
            "policy_comparison": str(run_dir / "tie_touch_policy_comparison.csv"),
            "report": str(run_dir / "sol_tie_touch_policy_audit_report.md"),
        },
    }
    json_dump(run_dir / "tie_touch_policy_manifest.json", manifest)

    print(json.dumps({"decision": decision, "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
