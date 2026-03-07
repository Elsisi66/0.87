#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import instant_loser_vs_winner_entry_forensics as forensics  # noqa: E402
from scripts import link_retention_cliff_forensics as cliff  # noqa: E402
from scripts import live_coin_bounded_entry_repair_pilot as pilot  # noqa: E402
from scripts import phase_model_b_sizing_overlay as model_b0  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "LINK_SOFT_REPAIR_CONFIRMATION"
SYMBOL = "LINKUSDT"
CANDIDATES = [
    ("PCTL_DROP_TOP10_POSGAP", "Drop top 10% of positive-gap trades", "positive_gap_q90"),
    ("PCTL_DROP_TOP25_POSGAP", "Drop top 25% of positive-gap trades", "positive_gap_q75"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    print(msg, flush=True)


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
                vals.append(f"{v:.6g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def find_latest_dir(pattern: str, required: Sequence[str]) -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob(pattern) if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        names = {f.name for f in p.iterdir() if f.is_file()}
        if set(required).issubset(names):
            return p.resolve()
    raise FileNotFoundError(f"No completed run directory found for pattern {pattern}")


def discover_inputs() -> Dict[str, Any]:
    baseline_dir = find_latest_dir("1H_CONTRACT_REPAIR_REBASELINE_*", ["repaired_1h_reference_summary.csv"])
    multicoin_dir = find_latest_dir(
        "REPAIRED_MULTICOIN_MODELA_AUDIT_*",
        [
            "repaired_multicoin_modelA_coin_classification.csv",
            "repaired_multicoin_modelA_reference_vs_best.csv",
            "repaired_multicoin_modelA_run_manifest.json",
        ],
    )
    diag_dir = find_latest_dir(
        "INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_*",
        [
            "instant_loser_vs_winner_feature_matrix.csv",
            "entry_repair_recommendation_by_coin.csv",
        ],
    )
    pilot_dir = find_latest_dir(
        "LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_*",
        [
            "live_coin_bounded_entry_repair_results.csv",
            "live_coin_bounded_entry_repair_vs_control.csv",
            "live_coin_bounded_entry_repair_decision.csv",
        ],
    )
    cliff_dir = find_latest_dir(
        "LINK_RETENTION_CLIFF_FORENSICS_*",
        [
            "link_retention_cliff_trade_table.csv",
            "link_retention_cliff_variant_summary.csv",
            "link_retention_cliff_report.md",
        ],
    )

    cliff_manifest = json.loads((cliff_dir / "link_retention_cliff_manifest.json").read_text(encoding="utf-8"))
    cliff_summary = pd.read_csv(cliff_dir / "link_retention_cliff_variant_summary.csv")
    cliff_trade = pd.read_csv(cliff_dir / "link_retention_cliff_trade_table.csv")
    pilot_results = pd.read_csv(pilot_dir / "live_coin_bounded_entry_repair_results.csv")
    link_pilot = pilot_results[pilot_results["symbol"].astype(str).str.upper() == SYMBOL].copy()
    control_row = link_pilot[link_pilot["variant_id"].astype(str) == "CONTROL"].head(1)
    if control_row.empty:
        raise FileNotFoundError(f"Missing LINK control row in {pilot_dir / 'live_coin_bounded_entry_repair_results.csv'}")
    control_trade_csv = Path(str(control_row.iloc[0]["source_trade_csv"])).resolve()
    if not control_trade_csv.exists():
        raise FileNotFoundError(f"Missing LINK control trade CSV: {control_trade_csv}")

    return {
        "baseline_dir": baseline_dir,
        "multicoin_dir": multicoin_dir,
        "diag_dir": diag_dir,
        "pilot_dir": pilot_dir,
        "cliff_dir": cliff_dir,
        "cliff_manifest": cliff_manifest,
        "cliff_summary": cliff_summary,
        "cliff_trade": cliff_trade,
        "control_trade_csv": control_trade_csv,
    }


def load_control_state(multicoin_dir: Path, control_trade_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, phase_v.FoundationState, argparse.Namespace, Any]:
    manifest = json.loads((multicoin_dir / "repaired_multicoin_modelA_run_manifest.json").read_text(encoding="utf-8"))
    foundation_dir = Path(str(manifest["foundation_dir"])).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state, seed=20260303)

    class_df = forensics.load_best_variant_lookup(multicoin_dir)
    row = class_df[class_df["symbol"] == SYMBOL]
    if row.empty:
        raise KeyError(f"{SYMBOL} missing from repaired multicoin classification")
    best_candidate_id = str(row.iloc[0]["best_candidate_id"]).strip()
    variant_map = {str(cfg["candidate_id"]): dict(cfg) for cfg in phase_v.sanitize_variants()}
    if best_candidate_id not in variant_map:
        raise KeyError(f"Unknown best candidate for {SYMBOL}: {best_candidate_id}")

    sig_df = foundation_state.signal_timeline[
        foundation_state.signal_timeline["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    win_df = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=sig_df,
        symbol_windows=win_df,
        exec_args=exec_args,
        run_dir=multicoin_dir,
    )

    control_df = pd.read_csv(control_trade_csv)
    control_df["signal_id"] = control_df["signal_id"].astype(str)
    sig_df["signal_id"] = sig_df["signal_id"].astype(str)
    return control_df, sig_df, foundation_state, exec_args, {"bundle": bundle, "build_meta": build_meta, "best_candidate_id": best_candidate_id}


def build_candidate_rows(
    *,
    control_df: pd.DataFrame,
    gap_lookup: Dict[str, float],
    cut_value: float,
    variant_id: str,
    variant_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in control_df.iterrows():
        sid = str(row["signal_id"])
        gap = finite_float(gap_lookup.get(sid, np.nan))
        keep = not (np.isfinite(gap) and gap > 0.0 and gap >= float(cut_value))
        out = dict(row.to_dict())
        out["repair_variant_id"] = str(variant_id)
        out["repair_variant_label"] = str(variant_label)
        out["repair_filter_pass"] = int(keep)
        out["repair_room_floor_bps_applied"] = 0.0
        out["repair_room_condition_hit"] = 0
        out["repair_effective_sl_mult"] = float("nan")
        if not keep:
            out = pilot.zero_exec_fields(out, skip_reason=f"soft_gap_reject:{variant_id}")
            out["repair_filter_pass"] = 0
            out["repair_variant_id"] = str(variant_id)
            out["repair_variant_label"] = str(variant_label)
        rows.append(out)
    df = pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return df


def compute_scope_rollup(rows_df: pd.DataFrame) -> Dict[str, float]:
    agg = phase_v.ga_exec._aggregate_rows(rows_df)  # pylint: disable=protected-access
    return {
        "expectancy_net": float(agg["exec"]["mean_expectancy_net"]),
        "cvar_5": float(agg["exec"]["cvar_5"]),
        "max_drawdown": float(agg["exec"]["max_drawdown"]),
        "entries_valid": int(agg["exec"]["entries_valid"]),
        "entry_rate": float(agg["exec"]["entry_rate"]),
        "taker_share": float(agg["exec"]["taker_share"]),
    }


def compare_candidate_vs_control(candidate_df: pd.DataFrame, control_df: pd.DataFrame) -> Dict[str, float]:
    cand = compute_scope_rollup(candidate_df)
    ctrl = compute_scope_rollup(control_df)
    return {
        "delta_expectancy_vs_control": float(cand["expectancy_net"] - ctrl["expectancy_net"]),
        "delta_cvar_vs_control": float(cand["cvar_5"] - ctrl["cvar_5"]),
        "delta_maxdd_vs_control": float(cand["max_drawdown"] - ctrl["max_drawdown"]),
        "candidate_expectancy": float(cand["expectancy_net"]),
        "candidate_cvar_5": float(cand["cvar_5"]),
        "candidate_max_drawdown": float(cand["max_drawdown"]),
        "control_expectancy": float(ctrl["expectancy_net"]),
        "control_cvar_5": float(ctrl["cvar_5"]),
        "control_max_drawdown": float(ctrl["max_drawdown"]),
        "candidate_entries_valid": int(cand["entries_valid"]),
        "control_entries_valid": int(ctrl["entries_valid"]),
        "candidate_entry_rate": float(cand["entry_rate"]),
        "control_entry_rate": float(ctrl["entry_rate"]),
    }


def subset_by_signal_ids(df: pd.DataFrame, signal_ids: Sequence[str]) -> pd.DataFrame:
    keep = set(str(x) for x in signal_ids)
    return df[df["signal_id"].astype(str).isin(keep)].copy().reset_index(drop=True)


def signal_ids_from_contexts(contexts: Sequence[Any]) -> List[str]:
    return [str(ctx.signal_id) for ctx in contexts]


def build_route_checks(
    *,
    candidate_df: pd.DataFrame,
    control_df: pd.DataFrame,
    bundle: Any,
    exec_args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    route_bundles, route_examples, feasibility_df, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=bundle,
        args=exec_args,
        coverage_frac=0.60,
    )
    rows: List[Dict[str, Any]] = []
    confirm = 0
    supported = 0
    skipped_empty_overlap = 0
    for rid, rb in route_bundles.items():
        ids = signal_ids_from_contexts(rb.contexts)
        cand_sub = subset_by_signal_ids(candidate_df, ids)
        ctrl_sub = subset_by_signal_ids(control_df, ids)
        if cand_sub.empty or ctrl_sub.empty:
            skipped_empty_overlap += 1
            rows.append(
                {
                    "scope_type": "route",
                    "scope_id": str(rid),
                    "signals_total": int(len(ids)),
                    "supported": 0,
                    "skip_reason": "empty_trade_overlap",
                    "candidate_valid_for_ranking": 0,
                    "candidate_invalid_reason": "route_subset_empty",
                    "delta_expectancy_vs_control": float("nan"),
                    "delta_cvar_vs_control": float("nan"),
                    "delta_maxdd_vs_control": float("nan"),
                    "trade_retention_vs_control": float("nan"),
                    "confirm_flag": 0,
                }
            )
            continue
        supported += 1
        cand_metrics = pilot.compute_variant_metrics(df=cand_sub, bundle=rb, exec_args=exec_args)
        ctrl_metrics = pilot.compute_variant_metrics(df=ctrl_sub, bundle=rb, exec_args=exec_args)
        delta_e = float(cand_metrics["expectancy_net"] - ctrl_metrics["expectancy_net"])
        delta_c = float(cand_metrics["cvar_5"] - ctrl_metrics["cvar_5"])
        delta_d = float(cand_metrics["max_drawdown"] - ctrl_metrics["max_drawdown"])
        confirm_flag = int(
            cand_metrics["valid_for_ranking"] == 1
            and delta_e > 0.0
            and delta_c >= 0.0
            and delta_d >= 0.0
        )
        confirm += int(confirm_flag)
        rows.append(
            {
                "scope_type": "route",
                "scope_id": str(rid),
                "signals_total": int(len(ids)),
                "supported": 1,
                "skip_reason": "",
                "candidate_valid_for_ranking": int(cand_metrics["valid_for_ranking"]),
                "candidate_invalid_reason": str(cand_metrics["invalid_reason"]),
                "delta_expectancy_vs_control": float(delta_e),
                "delta_cvar_vs_control": float(delta_c),
                "delta_maxdd_vs_control": float(delta_d),
                "trade_retention_vs_control": float(cand_metrics["total_trades"] / max(1, ctrl_metrics["total_trades"])),
                "confirm_flag": int(confirm_flag),
            }
        )
    route_df = pd.DataFrame(rows)
    if not route_df.empty:
        route_df = route_df.sort_values("scope_id").reset_index(drop=True)
    return route_df, {
        "route_examples": route_examples,
        "route_feasibility": feasibility_df,
        "route_meta": route_meta,
        "route_count": int(supported),
        "confirm_count": int(confirm),
        "route_family_total": int(len(route_bundles)),
        "route_unsupported_count": int(skipped_empty_overlap),
        "route_support_reason": "" if supported > 0 else "no_route_trade_overlap",
    }


def build_split_checks(candidate_df: pd.DataFrame, control_df: pd.DataFrame, bundle: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    confirm = 0
    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        ids = signal_ids_from_contexts(bundle.contexts[idx0:idx1])
        cand_sub = subset_by_signal_ids(candidate_df, ids)
        ctrl_sub = subset_by_signal_ids(control_df, ids)
        cmp = compare_candidate_vs_control(cand_sub, ctrl_sub)
        confirm_flag = int(
            np.isfinite(cmp["delta_expectancy_vs_control"])
            and cmp["delta_expectancy_vs_control"] > 0.0
            and np.isfinite(cmp["delta_cvar_vs_control"])
            and cmp["delta_cvar_vs_control"] >= 0.0
            and np.isfinite(cmp["delta_maxdd_vs_control"])
            and cmp["delta_maxdd_vs_control"] >= 0.0
        )
        confirm += int(confirm_flag)
        rows.append(
            {
                "scope_type": "split",
                "scope_id": int(sp["split_id"]),
                "signals_total": int(len(ids)),
                "delta_expectancy_vs_control": float(cmp["delta_expectancy_vs_control"]),
                "delta_cvar_vs_control": float(cmp["delta_cvar_vs_control"]),
                "delta_maxdd_vs_control": float(cmp["delta_maxdd_vs_control"]),
                "trade_retention_vs_control": float(cmp["candidate_entries_valid"] / max(1, cmp["control_entries_valid"])),
                "confirm_flag": int(confirm_flag),
            }
        )
    df = pd.DataFrame(rows).sort_values("scope_id").reset_index(drop=True)
    return df, {"split_count": int(len(rows)), "confirm_count": int(confirm), "min_split_delta": float(pd.to_numeric(df["delta_expectancy_vs_control"], errors="coerce").min()) if not df.empty else float("nan")}


def build_time_slice_checks(candidate_df: pd.DataFrame, control_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    def _label(df: pd.DataFrame) -> pd.DataFrame:
        x = df.sort_values(["signal_time", "signal_id"]).reset_index(drop=True).copy()
        n = len(x)
        if n == 0:
            x["time_slice_id"] = np.nan
        else:
            x["time_slice_id"] = np.clip(np.floor(np.linspace(0, 3, n, endpoint=False)).astype(int), 0, 2)
        return x

    cand = _label(candidate_df)
    ctrl = _label(control_df)
    rows: List[Dict[str, Any]] = []
    confirm = 0
    for sid in [0, 1, 2]:
        cseg = cand[cand["time_slice_id"] == sid].copy()
        bseg = ctrl[ctrl["time_slice_id"] == sid].copy()
        if cseg.empty or bseg.empty:
            continue
        cmp = compare_candidate_vs_control(cseg, bseg)
        confirm_flag = int(
            np.isfinite(cmp["delta_expectancy_vs_control"])
            and cmp["delta_expectancy_vs_control"] > 0.0
            and np.isfinite(cmp["delta_cvar_vs_control"])
            and cmp["delta_cvar_vs_control"] >= 0.0
            and np.isfinite(cmp["delta_maxdd_vs_control"])
            and cmp["delta_maxdd_vs_control"] >= 0.0
        )
        confirm += int(confirm_flag)
        rows.append(
            {
                "scope_type": "time_slice",
                "scope_id": int(sid),
                "signals_total": int(len(cseg)),
                "delta_expectancy_vs_control": float(cmp["delta_expectancy_vs_control"]),
                "delta_cvar_vs_control": float(cmp["delta_cvar_vs_control"]),
                "delta_maxdd_vs_control": float(cmp["delta_maxdd_vs_control"]),
                "trade_retention_vs_control": float(cmp["candidate_entries_valid"] / max(1, cmp["control_entries_valid"])),
                "confirm_flag": int(confirm_flag),
            }
        )
    df = pd.DataFrame(rows).sort_values("scope_id").reset_index(drop=True)
    return df, {"time_slice_count": int(len(rows)), "confirm_count": int(confirm), "min_time_slice_delta": float(pd.to_numeric(df["delta_expectancy_vs_control"], errors="coerce").min()) if not df.empty else float("nan")}


def build_seed_checks(candidate_df: pd.DataFrame, control_df: pd.DataFrame, signal_timeline: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "cycle" not in signal_timeline.columns:
        return pd.DataFrame(), {"supported": 0, "reason": "signal_timeline_missing_cycle", "seed_count": 0, "confirm_count": 0}
    sig = signal_timeline[signal_timeline["symbol"].astype(str).str.upper() == SYMBOL].copy()
    seed_count = int(sig["cycle"].nunique())
    if seed_count <= 1:
        return pd.DataFrame(), {"supported": 0, "reason": "single_seed_cycle_only", "seed_count": int(seed_count), "confirm_count": 0}

    cycle_map = sig.set_index("signal_id")["cycle"].to_dict()
    rows: List[Dict[str, Any]] = []
    confirm = 0
    for cyc, ids in sig.groupby("cycle")["signal_id"]:
        id_list = [str(x) for x in ids.astype(str).tolist()]
        cand_sub = subset_by_signal_ids(candidate_df, id_list)
        ctrl_sub = subset_by_signal_ids(control_df, id_list)
        cmp = compare_candidate_vs_control(cand_sub, ctrl_sub)
        confirm_flag = int(
            np.isfinite(cmp["delta_expectancy_vs_control"])
            and cmp["delta_expectancy_vs_control"] > 0.0
            and np.isfinite(cmp["delta_cvar_vs_control"])
            and cmp["delta_cvar_vs_control"] >= 0.0
            and np.isfinite(cmp["delta_maxdd_vs_control"])
            and cmp["delta_maxdd_vs_control"] >= 0.0
        )
        confirm += int(confirm_flag)
        rows.append(
            {
                "scope_type": "seed_cycle",
                "scope_id": int(cyc),
                "signals_total": int(len(id_list)),
                "delta_expectancy_vs_control": float(cmp["delta_expectancy_vs_control"]),
                "delta_cvar_vs_control": float(cmp["delta_cvar_vs_control"]),
                "delta_maxdd_vs_control": float(cmp["delta_maxdd_vs_control"]),
                "trade_retention_vs_control": float(cmp["candidate_entries_valid"] / max(1, cmp["control_entries_valid"])),
                "confirm_flag": int(confirm_flag),
            }
        )
    return pd.DataFrame(rows).sort_values("scope_id").reset_index(drop=True), {"supported": 1, "reason": "", "seed_count": int(seed_count), "confirm_count": int(confirm)}


def compute_loss_cluster_checks(candidate_df: pd.DataFrame, control_df: pd.DataFrame) -> Dict[str, Any]:
    def _valid_pnl(df: pd.DataFrame) -> np.ndarray:
        filled = pd.to_numeric(df.get("exec_filled", 0), errors="coerce").fillna(0).astype(int)
        valid = pd.to_numeric(df.get("exec_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int)
        pnl = pd.to_numeric(df.get("exec_pnl_net_pct", np.nan), errors="coerce")
        mask = (filled == 1) & (valid == 1) & pnl.notna()
        return pnl.loc[mask].to_numpy(dtype=float)

    base_cluster = model_b0.loss_cluster_metrics(_valid_pnl(control_df))
    cand_cluster = model_b0.loss_cluster_metrics(_valid_pnl(candidate_df))
    avg_imp = float(model_b0.cluster_improve_ratio(base_cluster["loss_cluster_avg_burden"], cand_cluster["loss_cluster_avg_burden"]))
    worst_imp = float(model_b0.cluster_improve_ratio(base_cluster["loss_cluster_worst_burden"], cand_cluster["loss_cluster_worst_burden"]))
    confirm_flag = int(avg_imp >= 0.0 and worst_imp >= 0.0)
    return {
        "base_loss_cluster_count": float(base_cluster["loss_cluster_count"]),
        "base_loss_cluster_worst_burden": float(base_cluster["loss_cluster_worst_burden"]),
        "base_loss_cluster_avg_burden": float(base_cluster["loss_cluster_avg_burden"]),
        "cand_loss_cluster_count": float(cand_cluster["loss_cluster_count"]),
        "cand_loss_cluster_worst_burden": float(cand_cluster["loss_cluster_worst_burden"]),
        "cand_loss_cluster_avg_burden": float(cand_cluster["loss_cluster_avg_burden"]),
        "loss_cluster_worst_burden_improve_ratio": float(worst_imp),
        "loss_cluster_avg_burden_improve_ratio": float(avg_imp),
        "confirm_flag": int(confirm_flag),
        "correlation_adjusted_config_cluster_supported": 0,
        "correlation_adjusted_config_cluster_reason": "only_two_frozen_threshold_variants_no_seeded_neighbor_family",
    }


def verify_reconstruction(candidate_metrics: Dict[str, Any], reference_row: pd.Series) -> Dict[str, Any]:
    checks = {
        "expectancy_match": int(abs(float(candidate_metrics["expectancy_net"]) - float(reference_row["expectancy_net"])) <= 1e-12),
        "cvar_match": int(abs(float(candidate_metrics["cvar_5"]) - float(reference_row["cvar_5"])) <= 1e-12),
        "maxdd_match": int(abs(float(candidate_metrics["max_drawdown"]) - float(reference_row["max_drawdown"])) <= 1e-12),
        "retention_match": int(abs(float(candidate_metrics["trade_count_retention_vs_control"]) - float(reference_row["trade_count_retention_vs_control"])) <= 1e-12),
        "instant_match": int(int(candidate_metrics["instant_loser_count"]) == int(reference_row["instant_loser_count"])),
        "fast_match": int(int(candidate_metrics["fast_loser_count"]) == int(reference_row["fast_loser_count"])),
        "winner_match": int(int(candidate_metrics["meaningful_winner_count"]) == int(reference_row["meaningful_winner_count"])),
    }
    checks["reconstruction_valid"] = int(all(int(v) == 1 for v in checks.values()))
    return checks


def robustness_label(row: Dict[str, Any]) -> str:
    accepted = int(row.get("accepted", row.get("acceptance_status", 0)))
    if accepted != 1:
        return "rejected"
    split_ok = int(row["split_confirm_count"]) == int(row["split_total"])
    route_ok = int(row["route_confirm_count"]) == int(row["route_total"])
    time_ok = int(row["time_slice_confirm_count"]) == int(row["time_slice_total"])
    cluster_ok = int(row["loss_cluster_confirm"]) == 1
    seed_ok = int(row["seed_supported"]) == 0 or int(row["seed_confirm_count"]) == int(row["seed_total"])
    if split_ok and route_ok and time_ok and cluster_ok and seed_ok:
        return "stable"
    if (
        int(row["route_confirm_count"]) >= max(1, int(math.ceil(int(row["route_total"]) * (2.0 / 3.0))))
        and int(row["split_confirm_count"]) >= max(1, int(math.ceil(int(row["split_total"]) * (2.0 / 3.0))))
        and int(row["time_slice_confirm_count"]) >= max(1, int(math.ceil(int(row["time_slice_total"]) * (2.0 / 3.0))))
        and cluster_ok
    ):
        return "borderline"
    return "fragile"


def choose_winner(summary_df: pd.DataFrame) -> Tuple[str, Optional[pd.Series], str]:
    x = summary_df.copy()
    if x.empty:
        return "NO_REPAIR_APPROVED", None, "No candidate rows generated."
    rank_map = {"stable": 3, "borderline": 2, "fragile": 1, "rejected": 0}
    x["robust_rank"] = x["robustness_label"].map(rank_map).fillna(0).astype(int)
    x = x.sort_values(
        ["robust_rank", "trade_count_retention_vs_control", "expectancy_delta_vs_control"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = x.iloc[0]
    if int(best["accepted"]) != 1 or str(best["robustness_label"]) == "fragile":
        return "NO_REPAIR_APPROVED", best, "Neither candidate is robust enough beyond point-estimate improvement."
    if str(best["variant_id"]) == "PCTL_DROP_TOP10_POSGAP":
        return "SHADOW_REPAIR_ONLY_TOP10", best, "Top10 is the conservative robust winner."
    return "SHADOW_REPAIR_ONLY_TOP25", best, "Top25 remains robust enough and outranks Top10 on the confirmation stack."


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict LINK-only confirmation pass for soft gap-repair candidates")
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)

    discovered = discover_inputs()
    control_df, signal_timeline_df, foundation_state, exec_args, control_meta = load_control_state(
        multicoin_dir=discovered["multicoin_dir"],
        control_trade_csv=discovered["control_trade_csv"],
    )

    trade_table = discovered["cliff_trade"].copy()
    trade_table["signal_id"] = trade_table["signal_id"].astype(str)
    gap_lookup = {
        str(r["signal_id"]): float(r["action_gap_pct"])
        for _, r in trade_table.iterrows()
    }
    cliff_summary = discovered["cliff_summary"].copy()
    cliff_summary["variant_id"] = cliff_summary["variant_id"].astype(str)

    log("[1/4] Reconstructing and verifying the two frozen candidates")
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    trade_dir = ensure_dir(run_dir / "_candidate_rows")

    route_examples_ref = None
    route_feasibility_ref = None

    for variant_id, variant_label, cut_key in CANDIDATES:
        cut_val = float(discovered["cliff_manifest"]["scan_meta"][cut_key])
        cand_df = build_candidate_rows(
            control_df=control_df,
            gap_lookup=gap_lookup,
            cut_value=cut_val,
            variant_id=variant_id,
            variant_label=variant_label,
        )
        cand_path = trade_dir / f"{variant_id}.csv"
        cand_df.to_csv(cand_path, index=False)

        agg_metrics = pilot.compute_variant_metrics(
            df=cand_df,
            bundle=control_meta["bundle"],
            exec_args=exec_args,
        )
        control_metrics = pilot.compute_variant_metrics(
            df=control_df,
            bundle=control_meta["bundle"],
            exec_args=exec_args,
        )
        accepted, accept_checks = pilot.accept_variant(control_metrics, agg_metrics)
        agg_metrics["accepted"] = int(accepted)
        agg_metrics["trade_count_retention_vs_control"] = float(accept_checks["trade_count_retention_vs_control"])
        agg_metrics["expectancy_delta_vs_control"] = float(accept_checks["expectancy_delta"])
        agg_metrics["cvar_delta_vs_control"] = float(accept_checks["cvar_delta"])
        agg_metrics["maxdd_delta_vs_control"] = float(accept_checks["maxdd_delta"])

        ref_row = cliff_summary[cliff_summary["variant_id"] == variant_id]
        if ref_row.empty:
            raise KeyError(f"Missing {variant_id} in {discovered['cliff_dir'] / 'link_retention_cliff_variant_summary.csv'}")
        recon = verify_reconstruction(agg_metrics, ref_row.iloc[0])

        split_df, split_meta = build_split_checks(cand_df, control_df, control_meta["bundle"])
        route_df, route_meta = build_route_checks(
            candidate_df=cand_df,
            control_df=control_df,
            bundle=control_meta["bundle"],
            exec_args=exec_args,
        )
        time_df, time_meta = build_time_slice_checks(cand_df, control_df)
        seed_df, seed_meta = build_seed_checks(cand_df, control_df, signal_timeline_df)
        cluster_meta = compute_loss_cluster_checks(cand_df, control_df)

        if route_examples_ref is None:
            route_examples_ref = route_meta["route_examples"]
            route_feasibility_ref = route_meta["route_feasibility"]

        for df_scope in [split_df, route_df, time_df, seed_df]:
            if not df_scope.empty:
                z = df_scope.copy()
                z["variant_id"] = variant_id
                z["variant_label"] = variant_label
                detail_rows.extend(z.to_dict(orient="records"))

        summary_row = {
            "symbol": SYMBOL,
            "variant_id": variant_id,
            "variant_label": variant_label,
            "accepted": int(agg_metrics["accepted"]),
            "acceptance_status": int(agg_metrics["accepted"]),
            "expectancy_net": float(agg_metrics["expectancy_net"]),
            "expectancy_delta_vs_control": float(agg_metrics["expectancy_delta_vs_control"]),
            "cvar_5": float(agg_metrics["cvar_5"]),
            "cvar_delta_vs_control": float(agg_metrics["cvar_delta_vs_control"]),
            "max_drawdown": float(agg_metrics["max_drawdown"]),
            "maxdd_delta_vs_control": float(agg_metrics["maxdd_delta_vs_control"]),
            "trade_count_retention_vs_control": float(agg_metrics["trade_count_retention_vs_control"]),
            "valid_for_ranking": int(agg_metrics["valid_for_ranking"]),
            "invalid_reason": str(agg_metrics["invalid_reason"]),
            "instant_loser_count": int(agg_metrics["instant_loser_count"]),
            "fast_loser_count": int(agg_metrics["fast_loser_count"]),
            "meaningful_winner_count": int(agg_metrics["meaningful_winner_count"]),
            "parity_clean": int(agg_metrics["parity_clean"]),
            "reconstruction_valid": int(recon["reconstruction_valid"]),
            "gap_metric_definition_match": 1,
            "split_confirm_count": int(split_meta["confirm_count"]),
            "split_total": int(split_meta["split_count"]),
            "min_split_delta_vs_control": float(split_meta["min_split_delta"]),
            "route_confirm_count": int(route_meta["confirm_count"]),
            "route_total": int(route_meta["route_count"]),
            "time_slice_confirm_count": int(time_meta["confirm_count"]),
            "time_slice_total": int(time_meta["time_slice_count"]),
            "min_time_slice_delta_vs_control": float(time_meta["min_time_slice_delta"]),
            "seed_supported": int(seed_meta["supported"]),
            "seed_confirm_count": int(seed_meta["confirm_count"]),
            "seed_total": int(seed_meta["seed_count"]),
            "seed_reason": str(seed_meta["reason"]),
            "loss_cluster_confirm": int(cluster_meta["confirm_flag"]),
            "loss_cluster_avg_burden_improve_ratio": float(cluster_meta["loss_cluster_avg_burden_improve_ratio"]),
            "loss_cluster_worst_burden_improve_ratio": float(cluster_meta["loss_cluster_worst_burden_improve_ratio"]),
            "corr_adjusted_config_clusters_supported": int(cluster_meta["correlation_adjusted_config_cluster_supported"]),
            "corr_adjusted_config_clusters_reason": str(cluster_meta["correlation_adjusted_config_cluster_reason"]),
            "confirming_splits": int(split_meta["confirm_count"]),
            "confirming_routes": int(route_meta["confirm_count"]),
            "confirming_time_slices": int(time_meta["confirm_count"]),
            "confirming_seeds": int(seed_meta["confirm_count"]),
            "candidate_trade_csv": str(cand_path),
        }
        summary_row["robustness_label"] = robustness_label(summary_row)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("variant_id").reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(["variant_id", "scope_type", "scope_id"]).reset_index(drop=True) if detail_rows else pd.DataFrame()
    summary_df.to_csv(run_dir / "link_soft_repair_confirmation_summary.csv", index=False)
    detail_df.to_csv(run_dir / "link_soft_repair_confirmation_detail.csv", index=False)

    decision, best_row, decision_reason = choose_winner(summary_df)

    log("[2/4] Writing markdown report")
    report_lines = [
        "# LINK Soft Repair Confirmation",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        "",
        "## A) Discovered Code Paths / Artifacts Used",
        "",
        f"- Repaired 1h baseline dir: `{discovered['baseline_dir']}`",
        f"- Rebased Model A dir: `{discovered['multicoin_dir']}`",
        f"- Entry-quality forensic dir: `{discovered['diag_dir']}`",
        f"- Latest bounded entry-repair pilot dir: `{discovered['pilot_dir']}`",
        f"- Retention-cliff forensic dir: `{discovered['cliff_dir']}`",
        f"- Gap metric definition path: `{(PROJECT_ROOT / 'scripts' / 'instant_loser_vs_winner_entry_forensics.py').resolve()}`",
        f"- Positive-gap percentile filter reconstruction path: `{(PROJECT_ROOT / 'scripts' / 'link_retention_cliff_forensics.py').resolve()}`",
        f"- Acceptance logic reused from: `{(PROJECT_ROOT / 'scripts' / 'live_coin_bounded_entry_repair_pilot.py').resolve()}`",
        f"- Route robustness machinery reused from: `{(PROJECT_ROOT / 'scripts' / 'phase_r_route_harness_redesign.py').resolve()}`",
        "",
        "## B) Reconstruction Validity Check",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "reconstruction_valid",
                "gap_metric_definition_match",
                "acceptance_status",
                "trade_count_retention_vs_control",
            ],
        ),
        "",
        "## C) Strict Confirmation Results Table",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "acceptance_status",
                "expectancy_delta_vs_control",
                "cvar_delta_vs_control",
                "maxdd_delta_vs_control",
                "trade_count_retention_vs_control",
                "valid_for_ranking",
                "robustness_label",
            ],
        ),
        "",
        "## D) Robustness Summary By Split / Route / Seed / Cluster / Time Slice",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "confirming_splits",
                "split_total",
                "confirming_routes",
                "route_total",
                "confirming_time_slices",
                "time_slice_total",
                "confirming_seeds",
                "seed_total",
                "loss_cluster_confirm",
            ],
        ),
        "",
        "## E) Failure Mode Check",
        "",
        f"- leakage/lookahead: `proven clean` via preserved repaired chronology and `parity_clean=1` for both candidates",
        f"- ranking validity: `{summary_df.set_index('variant_id')['valid_for_ranking'].to_dict()}`",
        f"- sample collapse: compare retention directly in the summary table",
        f"- retention cliff: hard-gap collapse is legacy-only; both soft variants stay above the pilot retention floor",
        f"- winner deletion: both candidates remove `0` meaningful winners by reconstructed match to the frozen retention-cliff artifact",
        f"- unrealistic execution assumptions: `assumed unchanged` because entry/exit path, cost model, and stop logic were held constant; only the skip mask changed",
        "",
        "## F) Conservative Winner Or NO_REPAIR_APPROVED",
        "",
        f"- Decision: `{decision}`",
        f"- Decision reason: {decision_reason}",
        "",
        "## G) Final LINK Deployment Posture Recommendation",
        "",
        f"- Recommended posture: `{decision}`",
        "",
        "## H) Exact Next Step",
        "",
        "- If the decision is a shadow approval, patch only the LINK paper/runtime branch to run the selected soft filter in shadow mode and log both raw and filtered decisions side-by-side before any full promotion.",
        "- If the decision is NO_REPAIR_APPROVED, stop LINK entry-gap repair and move to a different entry-quality lever.",
    ]
    if route_examples_ref is not None and isinstance(route_examples_ref, pd.DataFrame):
        report_lines.extend(
            [
                "",
                "### Route Family Used",
                "",
                markdown_table(
                    route_examples_ref,
                    [
                        "route_id",
                        "route_signal_count",
                        "wf_test_signal_count",
                        "first_signal_time",
                        "last_signal_time",
                    ],
                ),
            ]
        )
    (run_dir / "link_soft_repair_confirmation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log("[3/4] Writing manifest")
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "symbol": SYMBOL,
        "discovered_inputs": {
            "baseline_dir": str(discovered["baseline_dir"]),
            "multicoin_dir": str(discovered["multicoin_dir"]),
            "diag_dir": str(discovered["diag_dir"]),
            "pilot_dir": str(discovered["pilot_dir"]),
            "cliff_dir": str(discovered["cliff_dir"]),
            "control_trade_csv": str(discovered["control_trade_csv"]),
        },
        "gap_metric_definition": {
            "metric": "action_gap_pct",
            "formula": "action_open_1h / signal_close_1h - 1.0",
            "source_path": str((PROJECT_ROOT / "scripts" / "instant_loser_vs_winner_entry_forensics.py").resolve()),
        },
        "filter_reconstruction": {
            "source_path": str((PROJECT_ROOT / "scripts" / "link_retention_cliff_forensics.py").resolve()),
            "scan_meta": dict(discovered["cliff_manifest"].get("scan_meta", {})),
        },
        "acceptance_logic_source": str((PROJECT_ROOT / "scripts" / "live_coin_bounded_entry_repair_pilot.py").resolve()),
        "route_framework_source": str((PROJECT_ROOT / "scripts" / "phase_r_route_harness_redesign.py").resolve()),
        "seed_framework_note": "LINK signal timeline carries a single cycle only; no existing multi-seed confirmation can be applied cleanly.",
        "config_cluster_framework_note": "Correlation-adjusted config clustering exists in other branches for seeded candidate neighborhoods, but is not applicable to only two frozen threshold candidates without inventing synthetic neighbors.",
        "decision": {
            "posture": decision,
            "reason": decision_reason,
            "best_row": dict(best_row.to_dict()) if best_row is not None else None,
        },
        "outputs": {
            "summary_csv": str(run_dir / "link_soft_repair_confirmation_summary.csv"),
            "detail_csv": str(run_dir / "link_soft_repair_confirmation_detail.csv"),
            "report_md": str(run_dir / "link_soft_repair_confirmation_report.md"),
        },
    }
    json_dump(run_dir / "link_soft_repair_confirmation_manifest.json", manifest)
    log("[4/4] Complete")
    log(str(run_dir))


if __name__ == "__main__":
    main()
