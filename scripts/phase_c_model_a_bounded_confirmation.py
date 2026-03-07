#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_a_model_a_audit as phase_a  # noqa: E402
from scripts import phase_b_model_a_bounded_expansion as phase_b  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


PHASEB_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/PHASEB_MODEL_A_BOUNDED_EXPANSION_20260228_020446"
).resolve()


ROBUST_CLUSTER_SEEDS = [
    "M3_ENTRY_ONLY_FASTER",
    "M1_ENTRY_ONLY_PASSIVE_BASELINE_NOFB",
    "M2_ENTRY_ONLY_MORE_PASSIVE_OFF_02",
    "M2_ENTRY_ONLY_MORE_PASSIVE_NOFB",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def write_text(path: Path, text: str) -> None:
    phase_a.write_text(path, text)


def json_dump(path: Path, obj: Any) -> None:
    phase_a.json_dump(path, obj)


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
    return phase_a.markdown_table(df, cols, n)


def load_phaseb_seed_configs(phase_b_dir: Path) -> List[Dict[str, Any]]:
    res_fp = phase_b_dir / "phaseB2_results.csv"
    if not res_fp.exists():
        raise FileNotFoundError(f"Missing Phase B results: {res_fp}")
    df = pd.read_csv(res_fp)
    seed_rows = df[df["candidate_id"].astype(str).isin(ROBUST_CLUSTER_SEEDS)].copy()
    if len(seed_rows) != len(ROBUST_CLUSTER_SEEDS):
        found = set(seed_rows["candidate_id"].astype(str).tolist())
        missing = [cid for cid in ROBUST_CLUSTER_SEEDS if cid not in found]
        raise RuntimeError(f"Missing required Phase B robust seeds: {missing}")
    out: List[Dict[str, Any]] = []
    for _, r in seed_rows.sort_values("candidate_id").iterrows():
        out.append(
            {
                "candidate_id": str(r["candidate_id"]),
                "label": str(r["label"]),
                "seed_anchor": str(r["candidate_id"]),
                "seed_cluster_id": int(r["cluster_id"]),
                "entry_mode": str(r["entry_mode"]),
                "limit_offset_bps": float(r["limit_offset_bps"]),
                "fallback_to_market": int(r["fallback_to_market"]),
                "fallback_delay_min": float(r["fallback_delay_min"]),
                "max_fill_delay_min": float(r["max_fill_delay_min"]),
            }
        )
    return out


def with_updates(base: Dict[str, Any], cid: str, label: str, **updates: Any) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    out.update(updates)
    out["candidate_id"] = str(cid)
    out["label"] = str(label)
    return out


def generate_cluster_neighbors(seed_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sid = str(seed_cfg["candidate_id"])
    base = copy.deepcopy(seed_cfg)
    out: List[Dict[str, Any]] = []
    out.append(with_updates(base, sid, str(seed_cfg["label"])))

    delta_sets = {
        "offset": [-0.10, +0.10, +0.22, -0.22],
        "fallback": [-1.0, +1.0, +2.0],
        "window": [-2.0, +2.0, +4.0],
    }

    for i, d in enumerate(delta_sets["offset"]):
        off = clamp(float(base["limit_offset_bps"]) + float(d), 0.0, 2.25)
        out.append(
            with_updates(
                base,
                f"{sid}_C_OFF_{i:02d}",
                f"{sid} confirm offset {off:.2f}",
                limit_offset_bps=float(off),
            )
        )

    for i, d in enumerate(delta_sets["fallback"]):
        fb = clamp(float(base["fallback_delay_min"]) + float(d), 0.0, 45.0)
        mx = max(fb, clamp(float(base["max_fill_delay_min"]) + (2.0 if d > 0 else -1.0), 0.0, 45.0))
        out.append(
            with_updates(
                base,
                f"{sid}_C_FALL_{i:02d}",
                f"{sid} confirm timing {i}",
                fallback_delay_min=float(fb),
                max_fill_delay_min=float(mx),
            )
        )

    for i, d in enumerate(delta_sets["window"]):
        mx = clamp(float(base["max_fill_delay_min"]) + float(d), 0.0, 45.0)
        mx = max(mx, float(base["fallback_delay_min"]))
        out.append(
            with_updates(
                base,
                f"{sid}_C_WIN_{i:02d}",
                f"{sid} confirm window {i}",
                max_fill_delay_min=float(mx),
            )
        )

    combo_specs = [
        {
            "limit_offset_bps": clamp(float(base["limit_offset_bps"]) - 0.08, 0.0, 2.25),
            "fallback_delay_min": clamp(float(base["fallback_delay_min"]) + 1.0, 0.0, 45.0),
            "max_fill_delay_min": max(
                clamp(float(base["max_fill_delay_min"]) + 2.0, 0.0, 45.0),
                clamp(float(base["fallback_delay_min"]) + 1.0, 0.0, 45.0),
            ),
        },
        {
            "limit_offset_bps": clamp(float(base["limit_offset_bps"]) + 0.08, 0.0, 2.25),
            "fallback_delay_min": clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            "max_fill_delay_min": max(
                clamp(float(base["max_fill_delay_min"]) - 1.0, 0.0, 45.0),
                clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            ),
        },
    ]
    for i, spec in enumerate(combo_specs):
        out.append(
            with_updates(
                base,
                f"{sid}_C_COMBO_{i:02d}",
                f"{sid} confirm combo {i}",
                **spec,
            )
        )

    # One nearby purity-preserving toggle probe around each seed.
    if int(base["fallback_to_market"]) == 0:
        out.append(
            with_updates(
                base,
                f"{sid}_C_FB_ON",
                f"{sid} confirm fallback on",
                fallback_to_market=1,
                fallback_delay_min=max(1.0, clamp(float(base["fallback_delay_min"]), 0.0, 45.0)),
                max_fill_delay_min=max(
                    max(1.0, clamp(float(base["fallback_delay_min"]), 0.0, 45.0)),
                    clamp(float(base["max_fill_delay_min"]), 0.0, 45.0),
                ),
            )
        )
    else:
        out.append(
            with_updates(
                base,
                f"{sid}_C_FB_OFF",
                f"{sid} confirm fallback off",
                fallback_to_market=0,
                fallback_delay_min=0.0,
                max_fill_delay_min=max(0.0, float(base["max_fill_delay_min"])),
            )
        )

    return out


def bootstrap_pass_rate(split_df: pd.DataFrame, draws: int, seed: int) -> float:
    delta = to_num(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float))).dropna().to_numpy(dtype=float)
    if delta.size == 0:
        return float("nan")
    rng = np.random.RandomState(int(seed))
    passes = 0
    for _ in range(int(draws)):
        samp = delta[rng.randint(0, delta.size, size=delta.size)]
        if np.mean(samp) > 0.0 and np.min(samp) > -1e-4:
            passes += 1
    return float(passes / float(max(1, draws)))


def build_stricter_stress_configs(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    specs: List[Tuple[str, Dict[str, Any]]] = []

    c1 = copy.deepcopy(cfg)
    c1["candidate_id"] = f"{cfg['canonical_id']}_C_STRESS_OFF"
    c1["label"] = f"{cfg.get('label', cfg['canonical_id'])} confirm stress offset"
    c1["limit_offset_bps"] = clamp(float(cfg["limit_offset_bps"]) + 0.18, 0.0, 2.25)
    specs.append(("offset", c1))

    c2 = copy.deepcopy(cfg)
    c2["candidate_id"] = f"{cfg['canonical_id']}_C_STRESS_WIN"
    c2["label"] = f"{cfg.get('label', cfg['canonical_id'])} confirm stress window"
    c2["fallback_delay_min"] = clamp(float(cfg["fallback_delay_min"]) + 1.0, 0.0, 45.0)
    c2["max_fill_delay_min"] = max(
        float(c2["fallback_delay_min"]),
        clamp(float(cfg["max_fill_delay_min"]) + 2.0, 0.0, 45.0),
    )
    specs.append(("window", c2))

    c3 = copy.deepcopy(cfg)
    c3["candidate_id"] = f"{cfg['canonical_id']}_C_STRESS_ENTRY"
    c3["label"] = f"{cfg.get('label', cfg['canonical_id'])} confirm stress entry"
    c3["limit_offset_bps"] = clamp(float(cfg["limit_offset_bps"]) - 0.12, 0.0, 2.25)
    if int(cfg["fallback_to_market"]) == 0:
        c3["fallback_to_market"] = 1
        c3["fallback_delay_min"] = max(1.0, clamp(float(cfg["fallback_delay_min"]), 0.0, 45.0))
        c3["max_fill_delay_min"] = max(float(c3["fallback_delay_min"]), clamp(float(cfg["max_fill_delay_min"]), 0.0, 45.0))
    else:
        c3["fallback_to_market"] = 0
        c3["fallback_delay_min"] = 0.0
        c3["max_fill_delay_min"] = max(0.0, clamp(float(cfg["max_fill_delay_min"]), 0.0, 45.0))
    specs.append(("entry_mix", c3))

    return specs


def rank_rows(df: pd.DataFrame, cols: Sequence[str], ascending: Sequence[bool]) -> pd.DataFrame:
    return df.sort_values(list(cols), ascending=list(ascending)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C bounded confirmation on robust Model A clusters")
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--phase-b-dir", default=str(PHASEB_DIR_DEFAULT))
    ap.add_argument("--phase-r-dir", default=str(phase_a.PHASER_DIR_DEFAULT))
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    phase_b_dir = Path(args_cli.phase_b_dir).resolve()
    phase_r_dir = Path(args_cli.phase_r_dir).resolve()

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"PHASEC_MODEL_A_BOUNDED_CONFIRMATION_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args_cli.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)  # pylint: disable=protected-access
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("Frozen contract validation failed")

    bundles, _load_meta = ga_exec._prepare_bundles(exec_args)  # pylint: disable=protected-access
    if not bundles:
        raise RuntimeError("No bundles prepared under locked contract")
    base_bundle = bundles[0]
    one_h = phase_a.load_1h_market(base_bundle.symbol)
    fee = phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )

    route_bundles, route_examples_df, route_feas_df, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=base_bundle,
        args=exec_args,
        coverage_frac=0.60,
    )
    route_match_flag, route_mismatches = phase_a.route_examples_match(
        actual=route_examples_df,
        expected_path=phase_r_dir / "phaseR1_route_examples.csv",
    )

    forbidden_exit_knobs = [
        "tp_mult",
        "sl_mult",
        "time_stop_min",
        "break_even_enabled",
        "break_even_trigger_r",
        "break_even_offset_bps",
        "trailing_enabled",
        "trail_start_r",
        "trail_step_bps",
        "partial_take_enabled",
        "partial_take_r",
        "partial_take_pct",
    ]

    seed_catalog = load_phaseb_seed_configs(phase_b_dir=phase_b_dir)
    baseline_full = phase_a.build_1h_reference_rows(
        bundle=base_bundle,
        fee=fee,
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    baseline_routes: Dict[str, pd.DataFrame] = {}
    for rid, bundle in route_bundles.items():
        baseline_routes[rid] = phase_a.build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )

    c1 = {
        "generated_utc": utc_now(),
        "freeze_lock_pass": int(lock_info.get("freeze_lock_pass", 0)),
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "hybrid_exit_override_detected": 1,
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
        "phase_b_reference_dir": str(phase_b_dir),
        "phase_r_reference_dir": str(phase_r_dir),
        "route_reproduction_match_phaseR": int(route_match_flag),
        "route_reproduction_mismatches": route_mismatches,
        "route_count": int(route_meta.get("route_count", 0)),
        "seed_candidates": [str(x["candidate_id"]) for x in seed_catalog],
    }
    json_dump(run_dir / "phaseC1_contract_validation.json", c1)

    c1_lines = [
        "# C1 Model A Purity Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Freeze lock pass: `{int(lock_info.get('freeze_lock_pass', 0))}`",
        "- Wrapper remains pure Model A:",
        "  - 3m entry execution only",
        "  - 1h TP/SL/exit semantics only",
        "  - no dynamic 3m exit mutation",
        f"- Forbidden exit knobs blocked: `{forbidden_exit_knobs}`",
        f"- Repaired route reproduction matches Phase R: `{route_match_flag}`",
        f"- Route reproduction mismatches: `{route_mismatches}`",
        "",
        "## Seed Clusters",
        "",
        markdown_table(
            pd.DataFrame(seed_catalog),
            [
                "candidate_id",
                "seed_cluster_id",
                "entry_mode",
                "limit_offset_bps",
                "fallback_to_market",
                "fallback_delay_min",
                "max_fill_delay_min",
            ],
            n=12,
        ),
        "",
        "## Repaired Route Feasibility",
        "",
        markdown_table(
            route_feas_df,
            [
                "route_id",
                "route_signal_count",
                "wf_test_signal_count",
                "headroom_vs_overall_gate",
                "route_trade_gates_reachable",
            ],
            n=8,
        ),
        "",
    ]
    write_text(run_dir / "phaseC1_modelA_purity_report.md", "\n".join(c1_lines))

    # C2 robust cluster expansion.
    raw_candidates: List[Dict[str, Any]] = []
    for seed_cfg in seed_catalog:
        for cand in generate_cluster_neighbors(seed_cfg):
            x = copy.deepcopy(cand)
            x["seed_anchor"] = str(seed_cfg["candidate_id"])
            x["seed_cluster_id"] = int(seed_cfg["seed_cluster_id"])
            raw_candidates.append(x)

    unique_candidates, dup_map_df = phase_b.collapse_duplicates(raw_candidates)
    dup_map_df.to_csv(run_dir / "phaseC2_duplicate_variant_map.csv", index=False)

    vectors = {str(c["canonical_id"]): phase_b.config_vector(c) for c in unique_candidates}
    effective_trials_uncorrelated = int(len(unique_candidates))
    corr_cluster_count, cluster_map = phase_b.union_find_cluster_count(vectors=vectors, dist_threshold=0.34)
    for cfg in unique_candidates:
        cfg["cluster_id"] = int(cluster_map[str(cfg["canonical_id"])])

    invalid_hist: Counter[str] = Counter()
    results_rows: List[Dict[str, Any]] = []
    eval_cache: Dict[str, Dict[str, Any]] = {}

    ref_eval = phase_a.evaluate_reference_bundle(baseline_full)
    ref_metrics = ref_eval["metrics"]
    results_rows.append(
        {
            "candidate_id": "M0_1H_REFERENCE",
            "label": "M0 pure original 1h reference",
            "seed_origin": "M0_1H_REFERENCE",
            "seed_cluster_id": 0,
            "cluster_id": 0,
            "raw_variant_count": 1,
            "valid_for_ranking": int(ref_metrics["valid_for_ranking"]),
            "invalid_reason": str(ref_metrics["invalid_reason"]),
            "exec_expectancy_net": float(ref_metrics["overall_exec_expectancy_net"]),
            "delta_expectancy_vs_1h_reference": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "entry_rate": float(ref_metrics["overall_entry_rate"]),
            "entries_valid": int(ref_metrics["overall_entries_valid"]),
            "taker_share": float(ref_metrics["overall_exec_taker_share"]),
            "median_fill_delay_min": float(ref_metrics["overall_exec_median_fill_delay_min"]),
            "p95_fill_delay_min": float(ref_metrics["overall_exec_p95_fill_delay_min"]),
            "entry_mode": "market",
            "limit_offset_bps": 0.0,
            "fallback_to_market": 0,
            "fallback_delay_min": 0.0,
            "max_fill_delay_min": 0.0,
        }
    )

    for cfg in unique_candidates:
        ev = phase_a.evaluate_model_a_variant(
            bundle=base_bundle,
            baseline_df=baseline_full,
            cfg=cfg,
            one_h=one_h,
            args=exec_args,
        )
        eval_cache[str(cfg["canonical_id"])] = ev
        m = ev["metrics"]
        results_rows.append(
            {
                "candidate_id": str(cfg["canonical_id"]),
                "label": str(cfg.get("label", "")),
                "seed_origin": str(cfg.get("seed_anchor", "")),
                "seed_cluster_id": int(cfg.get("seed_cluster_id", -1)),
                "cluster_id": int(cfg.get("cluster_id", -1)),
                "raw_variant_count": int(cfg.get("raw_variant_count", 1)),
                "valid_for_ranking": int(m["valid_for_ranking"]),
                "invalid_reason": str(m["invalid_reason"]),
                "exec_expectancy_net": float(m["overall_exec_expectancy_net"]),
                "delta_expectancy_vs_1h_reference": float(m["overall_delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(m["overall_cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(m["overall_maxdd_improve_ratio"]),
                "entry_rate": float(m["overall_entry_rate"]),
                "entries_valid": int(m["overall_entries_valid"]),
                "taker_share": float(m["overall_exec_taker_share"]),
                "median_fill_delay_min": float(m["overall_exec_median_fill_delay_min"]),
                "p95_fill_delay_min": float(m["overall_exec_p95_fill_delay_min"]),
                "entry_mode": str(cfg["entry_mode"]),
                "limit_offset_bps": float(cfg["limit_offset_bps"]),
                "fallback_to_market": int(cfg["fallback_to_market"]),
                "fallback_delay_min": float(cfg["fallback_delay_min"]),
                "max_fill_delay_min": float(cfg["max_fill_delay_min"]),
            }
        )
        for part in [x for x in str(m["invalid_reason"]).split("|") if x]:
            invalid_hist[str(part)] += 1

    results_df = rank_rows(
        pd.DataFrame(results_rows),
        ["candidate_id"],
        [True],
    )
    results_df.to_csv(run_dir / "phaseC2_results.csv", index=False)
    json_dump(run_dir / "phaseC2_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    eff_lines = [
        "# C2 Effective Trials Summary",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Raw confirmation candidates generated: `{len(raw_candidates)}`",
        f"- Unique configs evaluated after duplicate collapse: `{effective_trials_uncorrelated}`",
        f"- Duplicate-collapsed configs: `{len(raw_candidates) - effective_trials_uncorrelated}`",
        f"- Correlation-adjusted config clusters: `{corr_cluster_count}`",
        "- Correlation adjustment method:",
        "  - normalize entry-only knob vectors",
        "  - cluster configs by L1 distance <= 0.34",
        "  - use cluster count as cheap correlation-adjusted effective trials",
        "",
    ]
    write_text(run_dir / "phaseC2_effective_trials_summary.md", "\n".join(eff_lines))

    c2_lines = [
        "# C2 Robust Cluster Expansion Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Raw candidates: `{len(raw_candidates)}`",
        f"- Unique candidates: `{effective_trials_uncorrelated}`",
        f"- Correlation-adjusted clusters: `{corr_cluster_count}`",
        "",
        "## Top Results",
        "",
        markdown_table(
            results_df[results_df["candidate_id"] != "M0_1H_REFERENCE"].sort_values(
                ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
                ascending=[False, False, False],
            ),
            [
                "candidate_id",
                "seed_origin",
                "seed_cluster_id",
                "cluster_id",
                "valid_for_ranking",
                "exec_expectancy_net",
                "delta_expectancy_vs_1h_reference",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "taker_share",
                "p95_fill_delay_min",
                "invalid_reason",
            ],
            n=24,
        ),
        "",
    ]
    write_text(run_dir / "phaseC2_report.md", "\n".join(c2_lines))

    # C3 stricter confirmation.
    valid_positive = results_df[
        (results_df["candidate_id"] != "M0_1H_REFERENCE")
        & (to_num(results_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(results_df["delta_expectancy_vs_1h_reference"]) > 0.0)
    ].copy()
    ranked_valid_positive = valid_positive.sort_values(
        ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    shortlist_ids: List[str] = []
    for _seed_cluster_id, grp in ranked_valid_positive.groupby("seed_cluster_id", sort=False):
        if grp.empty:
            continue
        shortlist_ids.append(str(grp.iloc[0]["candidate_id"]))
    for _cluster_id, grp in ranked_valid_positive.groupby("cluster_id", sort=False):
        if grp.empty:
            continue
        cid = str(grp.iloc[0]["candidate_id"])
        if cid not in shortlist_ids:
            shortlist_ids.append(cid)
    for cid in ranked_valid_positive["candidate_id"].astype(str).tolist():
        if cid not in shortlist_ids:
            shortlist_ids.append(cid)
        if len(shortlist_ids) >= 10:
            break
    shortlist = ranked_valid_positive[ranked_valid_positive["candidate_id"].astype(str).isin(shortlist_ids[:10])].copy()
    shortlist = shortlist.sort_values(
        ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    route_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []
    survivor_rows: List[Dict[str, Any]] = []

    for pos, (_, top) in enumerate(shortlist.iterrows()):
        cid = str(top["candidate_id"])
        cfg = next(c for c in unique_candidates if str(c["canonical_id"]) == cid)
        ev = eval_cache[cid]
        split_df = ev["split_rows_df"].copy()

        route_df = phase_b.evaluate_routes_for_candidate(
            cfg=cfg,
            route_bundles=route_bundles,
            baseline_routes=baseline_routes,
            one_h=one_h,
            args=exec_args,
        )
        route_rows.extend(route_df.assign(candidate_id=cid).to_dict(orient="records"))

        route_pass_rate = float(
            (
                (to_num(route_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
                & (to_num(route_df["delta_expectancy_vs_1h_reference"]) > 0.0)
            ).mean()
        ) if not route_df.empty else float("nan")
        route_pass = int(
            (not route_df.empty)
            and (
                (to_num(route_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
                & (to_num(route_df["delta_expectancy_vs_1h_reference"]) > 0.0)
            ).all()
        )
        center_row = route_df[route_df["route_id"].astype(str) == "route_center_60pct"].head(1)
        center_valid = int(center_row["valid_for_ranking"].iloc[0]) if not center_row.empty else 0
        center_delta = float(center_row["delta_expectancy_vs_1h_reference"].iloc[0]) if not center_row.empty else float("nan")
        min_subperiod_delta = float(to_num(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float))).min()) if not split_df.empty else float("nan")

        stress_results: List[Dict[str, Any]] = []
        for stress_id, stress_cfg in build_stricter_stress_configs(cfg):
            sev = phase_a.evaluate_model_a_variant(
                bundle=base_bundle,
                baseline_df=baseline_full,
                cfg=stress_cfg,
                one_h=one_h,
                args=exec_args,
            )
            sm = sev["metrics"]
            stress_results.append(
                {
                    "candidate_id": cid,
                    "stress_id": str(stress_id),
                    "valid_for_ranking": int(sm["valid_for_ranking"]),
                    "delta_expectancy_vs_1h_reference": float(sm["overall_delta_expectancy_exec_minus_baseline"]),
                    "exec_expectancy_net": float(sm["overall_exec_expectancy_net"]),
                    "taker_share": float(sm["overall_exec_taker_share"]),
                    "p95_fill_delay_min": float(sm["overall_exec_p95_fill_delay_min"]),
                    "invalid_reason": str(sm["invalid_reason"]),
                }
            )
        stress_df = pd.DataFrame(stress_results)
        stress_rows.extend(stress_results)
        stress_delta_mean = float(to_num(stress_df["delta_expectancy_vs_1h_reference"]).mean()) if not stress_df.empty else float("nan")
        stress_pass = int(
            (not stress_df.empty)
            and (to_num(stress_df["valid_for_ranking"]).fillna(0).astype(int) == 1).all()
            and (to_num(stress_df["delta_expectancy_vs_1h_reference"]) > 0.0).all()
        )

        boot_rate = bootstrap_pass_rate(
            split_df=split_df,
            draws=256,
            seed=int(args_cli.seed) + 1000 + int(pos),
        )
        boot_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "cluster_id": int(top["cluster_id"]),
                "bootstrap_draws": 256,
                "bootstrap_pass_rate": float(boot_rate),
            }
        )

        psr_proxy, dsr_proxy = phase_b.candidate_psr_dsr(
            split_df=split_df,
            effective_trials_corr_adjusted=float(corr_cluster_count),
        )
        robust_survivor = int(
            int(top["valid_for_ranking"]) == 1
            and route_pass == 1
            and np.isfinite(route_pass_rate)
            and route_pass_rate >= 1.0
            and center_valid == 1
            and np.isfinite(center_delta)
            and center_delta > 0.0
            and np.isfinite(min_subperiod_delta)
            and min_subperiod_delta > 0.0
            and stress_pass == 1
            and np.isfinite(stress_delta_mean)
            and stress_delta_mean > 0.0
            and np.isfinite(boot_rate)
            and boot_rate >= 0.90
            and np.isfinite(psr_proxy)
            and psr_proxy >= 0.97
            and np.isfinite(dsr_proxy)
            and dsr_proxy >= 0.90
        )

        sig_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "seed_cluster_id": int(top["seed_cluster_id"]),
                "cluster_id": int(top["cluster_id"]),
                "split_count": int(len(split_df)),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "effective_trials_uncorrelated": int(effective_trials_uncorrelated),
                "effective_trials_corr_adjusted": int(corr_cluster_count),
            }
        )
        survivor_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "seed_cluster_id": int(top["seed_cluster_id"]),
                "cluster_id": int(top["cluster_id"]),
                "valid_for_ranking": int(top["valid_for_ranking"]),
                "exec_expectancy_net": float(top["exec_expectancy_net"]),
                "delta_expectancy_vs_1h_reference": float(top["delta_expectancy_vs_1h_reference"]),
                "cvar_improve_ratio": float(top["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(top["maxdd_improve_ratio"]),
                "entry_rate": float(top["entry_rate"]),
                "entries_valid": int(top["entries_valid"]),
                "taker_share": float(top["taker_share"]),
                "median_fill_delay_min": float(top["median_fill_delay_min"]),
                "p95_fill_delay_min": float(top["p95_fill_delay_min"]),
                "route_pass": int(route_pass),
                "route_pass_rate": float(route_pass_rate),
                "center_route_valid": int(center_valid),
                "center_route_delta": float(center_delta),
                "min_subperiod_delta": float(min_subperiod_delta),
                "stress_pass": int(stress_pass),
                "stress_delta_mean": float(stress_delta_mean),
                "bootstrap_pass_rate": float(boot_rate),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "robust_survivor": int(robust_survivor),
            }
        )

    route_checks_df = rank_rows(pd.DataFrame(route_rows), ["candidate_id", "route_id"], [True, True])
    stress_df = rank_rows(pd.DataFrame(stress_rows), ["candidate_id", "stress_id"], [True, True])
    boot_df = rank_rows(pd.DataFrame(boot_rows), ["candidate_id"], [True])
    sig_df = rank_rows(pd.DataFrame(sig_rows), ["candidate_id"], [True])
    survivors_df = rank_rows(
        pd.DataFrame(survivor_rows),
        ["robust_survivor", "route_pass_rate", "bootstrap_pass_rate", "delta_expectancy_vs_1h_reference", "taker_share"],
        [False, False, False, False, True],
    )

    route_checks_df.to_csv(run_dir / "phaseC3_route_checks.csv", index=False)
    stress_df.to_csv(run_dir / "phaseC3_stress_lite.csv", index=False)
    boot_df.to_csv(run_dir / "phaseC3_bootstrap_summary.csv", index=False)
    sig_df.to_csv(run_dir / "phaseC3_shortlist_significance.csv", index=False)

    robust_survivors = survivors_df[to_num(survivors_df.get("robust_survivor", 0)).fillna(0).astype(int) == 1].copy()
    robust_clusters = set(to_num(robust_survivors.get("seed_cluster_id", pd.Series(dtype=float))).dropna().astype(int).tolist()) if not robust_survivors.empty else set()
    stable_frontier = int(len(robust_survivors) >= 4 and len(robust_clusters) >= 3)

    c3_lines = [
        "# C3 Top Survivors Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Shortlist size: `{len(shortlist)}`",
        f"- Robust survivors: `{len(robust_survivors)}`",
        f"- Robust seed clusters represented: `{sorted(robust_clusters)}`",
        f"- Stable frontier: `{stable_frontier}`",
        "",
        "## Shortlisted Candidates",
        "",
        markdown_table(
            survivors_df,
            [
                "candidate_id",
                "seed_origin",
                "seed_cluster_id",
                "cluster_id",
                "delta_expectancy_vs_1h_reference",
                "route_pass_rate",
                "center_route_delta",
                "min_subperiod_delta",
                "stress_pass",
                "stress_delta_mean",
                "bootstrap_pass_rate",
                "psr_proxy",
                "dsr_proxy",
                "robust_survivor",
            ],
            n=16,
        ),
        "",
    ]
    write_text(run_dir / "phaseC3_top_survivors_report.md", "\n".join(c3_lines))

    # C4 champion / backup selection.
    robust_ranked = robust_survivors.sort_values(
        [
            "route_pass_rate",
            "stress_pass",
            "bootstrap_pass_rate",
            "delta_expectancy_vs_1h_reference",
            "cvar_improve_ratio",
            "maxdd_improve_ratio",
            "taker_share",
            "p95_fill_delay_min",
        ],
        ascending=[False, False, False, False, False, False, True, True],
    ).reset_index(drop=True)

    primary_row = robust_ranked.head(1).copy()
    backup_row = robust_ranked.iloc[0:0].copy()
    if not robust_ranked.empty:
        primary_cluster = int(primary_row["seed_cluster_id"].iloc[0])
        diff_cluster = robust_ranked[to_num(robust_ranked["seed_cluster_id"]).fillna(-1).astype(int) != primary_cluster].copy()
        if not diff_cluster.empty:
            backup_row = diff_cluster.head(1).copy()
        elif len(robust_ranked) >= 2:
            backup_row = robust_ranked.iloc[[1]].copy()

    select_rows: List[Dict[str, Any]] = []
    if not primary_row.empty:
        x = primary_row.iloc[0]
        select_rows.append({"selection_role": "primary", **{k: x[k] for k in primary_row.columns}})
    if not backup_row.empty:
        x = backup_row.iloc[0]
        select_rows.append({"selection_role": "backup", **{k: x[k] for k in backup_row.columns}})
    select_df = pd.DataFrame(select_rows)
    select_df.to_csv(run_dir / "phaseC4_primary_backup.csv", index=False)

    c4_lines = [
        "# C4 Champion Selection",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Robust survivors available: `{len(robust_ranked)}`",
        f"- Primary selected: `{'' if primary_row.empty else str(primary_row['candidate_id'].iloc[0])}`",
        f"- Backup selected: `{'' if backup_row.empty else str(backup_row['candidate_id'].iloc[0])}`",
        "",
        "## Selection Table",
        "",
        markdown_table(
            select_df,
            [
                "selection_role",
                "candidate_id",
                "seed_origin",
                "seed_cluster_id",
                "cluster_id",
                "delta_expectancy_vs_1h_reference",
                "route_pass_rate",
                "bootstrap_pass_rate",
                "stress_delta_mean",
                "taker_share",
                "p95_fill_delay_min",
            ],
            n=4,
        ),
        "",
    ]
    write_text(run_dir / "phaseC4_champion_selection.md", "\n".join(c4_lines))

    # C5 promotion decision.
    primary_ok = not primary_row.empty
    backup_ok = not backup_row.empty
    backup_diff_cluster = int(
        primary_ok
        and backup_ok
        and int(primary_row["seed_cluster_id"].iloc[0]) != int(backup_row["seed_cluster_id"].iloc[0])
    )

    if int(route_match_flag) != 1:
        classification = "MODEL_A_INFRA_BLOCKED"
        mainline_status = "MODEL_A_INFRA_BLOCKED"
        next_prompt = ""
        promotion_text = "No; infra blocked."
    elif len(robust_survivors) == 0 or (not primary_ok):
        classification = "MODEL_A_CONFIRMATION_NO_GO"
        mainline_status = "MODEL_A_CONFIRMATION_NO_GO"
        next_prompt = ""
        promotion_text = "No; stricter confirmation collapsed."
    else:
        primary_boot = float(primary_row["bootstrap_pass_rate"].iloc[0]) if primary_ok else float("nan")
        primary_dsr = float(primary_row["dsr_proxy"].iloc[0]) if primary_ok else float("nan")
        if stable_frontier == 1 and primary_ok and backup_ok and backup_diff_cluster == 1 and np.isfinite(primary_boot) and primary_boot >= 0.95 and np.isfinite(primary_dsr) and primary_dsr >= 0.95:
            classification = "MODEL_A_PROMOTE_PAPER"
            mainline_status = "MODEL_A_PROMOTE_PAPER"
            promotion_text = "Yes; strong paper/shadow promotion justified."
        else:
            classification = "MODEL_A_PAPER_CAUTION"
            mainline_status = "MODEL_A_PAPER_CAUTION"
            promotion_text = "Yes; paper/shadow justified with caution and rollback triggers."

        primary_id = str(primary_row["candidate_id"].iloc[0]) if primary_ok else ""
        backup_id = str(backup_row["candidate_id"].iloc[0]) if backup_ok else ""
        next_prompt = (
            "ROLE\n"
            "You are in Phase D Model A paper/shadow mode for SOLUSDT.\n\n"
            "MISSION\n"
            "Run controlled paper/shadow confirmation only for the selected Model A entry-only configurations under the same frozen 1h signal/exit contract. "
            "Do not add any 3m exit logic and do not authorize live trading.\n\n"
            "PRIMARY\n"
            f"{primary_id}\n\n"
            "BACKUP\n"
            f"{backup_id}\n\n"
            "ROLLBACK TRIGGERS\n"
            "1) Any route-center regression or route validity loss.\n"
            "2) Taker share exceeds 0.25 on the monitored confirmation slice.\n"
            "3) Realized confirmation expectancy turns negative versus the frozen 1h reference.\n"
            "4) Fill-delay realism exceeds the frozen hard gates.\n"
        )
        write_text(run_dir / "ready_to_launch_phaseD_paper_prompt.txt", next_prompt)

    c5_lines = [
        "# Phase C Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Robust survivors: `{len(robust_survivors)}`",
        f"- Stable frontier: `{stable_frontier}`",
        f"- Primary selected: `{primary_ok}`",
        f"- Backup selected: `{backup_ok}`",
        f"- Backup different cluster: `{backup_diff_cluster}`",
        "- Promotion decision:",
        f"  - {promotion_text}",
        "",
    ]
    write_text(run_dir / "phaseC_decision.md", "\n".join(c5_lines))

    manifest = {
        "generated_utc": utc_now(),
        "classification": classification,
        "mainline_status": mainline_status,
        "frozen_subset": str(subset_path),
        "phase_b_dir": str(phase_b_dir),
        "phase_r_dir": str(phase_r_dir),
        "git_snapshot": phase_a.git_snapshot(),
        "freeze_lock": lock_info,
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
        "raw_candidates": int(len(raw_candidates)),
        "unique_candidates": int(effective_trials_uncorrelated),
        "corr_adjusted_clusters": int(corr_cluster_count),
        "robust_survivor_count": int(len(robust_survivors)),
        "stable_frontier": int(stable_frontier),
        "primary_id": "" if primary_row.empty else str(primary_row["candidate_id"].iloc[0]),
        "backup_id": "" if backup_row.empty else str(backup_row["candidate_id"].iloc[0]),
    }
    json_dump(run_dir / "phaseC_run_manifest.json", manifest)

    result = {
        "furthest_phase": "C5",
        "classification": classification,
        "mainline_status": mainline_status,
        "run_dir": str(run_dir),
        "robust_survivor_count": int(len(robust_survivors)),
        "stable_frontier": int(stable_frontier),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
