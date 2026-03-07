#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as pu  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    if x.empty:
        return "_(none)_"
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


def bundle_test_count(bundle: ga_exec.SymbolBundle) -> int:
    return int(sum(int(sp["test_end"]) - int(sp["test_start"]) for sp in bundle.splits))


def min_contexts_for_overall_gate(args: argparse.Namespace) -> Tuple[int, int]:
    target = int(args.hard_min_trades_overall)
    for n in range(1, 20000):
        splits = ga_exec._build_walkforward_splits(n=n, train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))
        test_n = int(sum(int(sp["test_end"]) - int(sp["test_start"]) for sp in splits))
        if test_n >= target:
            return int(n), int(test_n)
    return -1, -1


def _bundle_from_context_slice(
    *,
    base_bundle: ga_exec.SymbolBundle,
    route_id: str,
    contexts: List[ga_exec.SignalContext],
    args: argparse.Namespace,
) -> ga_exec.SymbolBundle:
    route_ctx = sorted(contexts, key=lambda z: (pd.to_datetime(z.signal_time, utc=True), str(z.signal_id)))
    splits = ga_exec._build_walkforward_splits(n=len(route_ctx), train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))
    return ga_exec.SymbolBundle(
        symbol=base_bundle.symbol,
        signals_csv=base_bundle.signals_csv,
        contexts=route_ctx,
        splits=splits,
        constraints=dict(base_bundle.constraints),
    )


def build_support_feasible_route_family(
    *,
    base_bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    coverage_frac: float = 0.60,
) -> Tuple[Dict[str, ga_exec.SymbolBundle], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    n_total = int(len(base_bundle.contexts))
    min_ctx, min_test = min_contexts_for_overall_gate(args)
    if n_total < min_ctx:
        raise RuntimeError(
            f"Representative subset too small for support-feasible route family: contexts={n_total}, required_min={min_ctx}"
        )

    window_n = max(int(min_ctx), int(math.ceil(float(coverage_frac) * n_total)))
    window_n = min(window_n, n_total)

    candidate_windows = [
        ("route_front_60pct", 0, window_n),
        ("route_center_60pct", max(0, int(math.floor((n_total - window_n) / 2))), max(0, int(math.floor((n_total - window_n) / 2))) + window_n),
        ("route_back_60pct", max(0, n_total - window_n), n_total),
    ]

    seen_keys = set()
    bundles: Dict[str, ga_exec.SymbolBundle] = {}
    example_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []

    for rid, start, end in candidate_windows:
        ctx = base_bundle.contexts[int(start):int(end)]
        if len(ctx) < min_ctx:
            continue
        key = tuple(str(c.signal_id) for c in ctx)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        bundle = _bundle_from_context_slice(base_bundle=base_bundle, route_id=rid, contexts=list(ctx), args=args)
        test_n = bundle_test_count(bundle)
        min_trades_symbol = max(
            int(args.hard_min_trades_symbol),
            int(math.ceil(float(args.hard_min_trade_frac_symbol) * max(1, test_n))),
        )
        min_trades_overall = max(
            int(args.hard_min_trades_overall),
            int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, test_n))),
        )

        first_ctx = bundle.contexts[0]
        last_ctx = bundle.contexts[-1]
        bundles[rid] = bundle
        example_rows.append(
            {
                "route_id": rid,
                "route_start_idx": int(start),
                "route_end_idx_exclusive": int(end),
                "route_signal_count": int(len(bundle.contexts)),
                "wf_test_signal_count": int(test_n),
                "first_signal_id": str(first_ctx.signal_id),
                "first_signal_time": str(pd.to_datetime(first_ctx.signal_time, utc=True)),
                "last_signal_id": str(last_ctx.signal_id),
                "last_signal_time": str(pd.to_datetime(last_ctx.signal_time, utc=True)),
            }
        )
        valid_rows.append(
            {
                "route_id": rid,
                "route_signal_count": int(len(bundle.contexts)),
                "wf_test_signal_count": int(test_n),
                "hard_min_trades_symbol": int(min_trades_symbol),
                "hard_min_trades_overall": int(min_trades_overall),
                "headroom_vs_symbol_gate": int(test_n - min_trades_symbol),
                "headroom_vs_overall_gate": int(test_n - min_trades_overall),
                "symbol_trade_gate_reachable": int(test_n >= min_trades_symbol),
                "overall_trade_gate_reachable": int(test_n >= min_trades_overall),
                "route_trade_gates_reachable": int((test_n >= min_trades_symbol) and (test_n >= min_trades_overall)),
            }
        )

    example_df = pd.DataFrame(example_rows).sort_values("route_id").reset_index(drop=True)
    valid_df = pd.DataFrame(valid_rows).sort_values("route_id").reset_index(drop=True)
    meta = {
        "min_contexts_for_overall_gate": int(min_ctx),
        "min_test_signals_for_overall_gate": int(min_test),
        "coverage_frac": float(coverage_frac),
        "window_n": int(window_n),
        "route_count": int(len(bundles)),
    }
    return bundles, example_df, valid_df, meta


def build_legacy_route_support_table(
    *,
    base_bundle: ga_exec.SymbolBundle,
    rep_subset: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    legacy_sets = af.route_signal_sets(rep_subset)
    ctx_by_id = {str(c.signal_id): c for c in base_bundle.contexts}
    rows: List[Dict[str, Any]] = []
    for rid, rdf in legacy_sets.items():
        ids = rdf["signal_id"].astype(str).tolist() if "signal_id" in rdf.columns else []
        ctx = [ctx_by_id[sid] for sid in ids if sid in ctx_by_id]
        route_bundle = _bundle_from_context_slice(base_bundle=base_bundle, route_id=rid, contexts=ctx, args=args)
        route_test_total = bundle_test_count(route_bundle)
        min_trades_symbol = max(
            int(args.hard_min_trades_symbol),
            int(math.ceil(float(args.hard_min_trade_frac_symbol) * max(1, route_test_total))),
        )
        min_trades_overall = max(
            int(args.hard_min_trades_overall),
            int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, route_test_total))),
        )
        cumulative = 0
        for sp in route_bundle.splits:
            test_count = int(sp["test_end"]) - int(sp["test_start"])
            cumulative += test_count
            rows.append(
                {
                    "route_id": rid,
                    "symbol": str(route_bundle.symbol),
                    "route_signal_count": int(len(route_bundle.contexts)),
                    "split_id": int(sp["split_id"]),
                    "split_test_count": int(test_count),
                    "cumulative_test_count": int(cumulative),
                    "route_upper_bound_entries": int(route_test_total),
                    "hard_min_trades_symbol": int(min_trades_symbol),
                    "hard_min_trades_overall": int(min_trades_overall),
                    "symbol_trade_gate_reachable": int(route_test_total >= min_trades_symbol),
                    "overall_trade_gate_reachable": int(route_test_total >= min_trades_overall),
                    "route_trade_gates_reachable": int((route_test_total >= min_trades_symbol) and (route_test_total >= min_trades_overall)),
                }
            )
    return pd.DataFrame(rows).sort_values(["route_id", "split_id"]).reset_index(drop=True)


def baseline_rows_for_bundle(bundle: ga_exec.SymbolBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        split_signal_rows: List[Dict[str, Any]] = []
        for ctx in bundle.contexts[idx0:idx1]:
            row = {
                "signal_id": str(ctx.signal_id),
                "signal_time": str(pd.to_datetime(ctx.signal_time, utc=True)),
                "baseline_filled": int(ctx.baseline_filled),
                "baseline_valid_for_metrics": int(ctx.baseline_valid_for_metrics),
                "baseline_sl_hit": int(ctx.baseline_sl_hit),
                "baseline_pnl_net_pct": float(ctx.baseline_pnl_net_pct),
                "baseline_fill_liquidity_type": str(ctx.baseline_fill_liq),
                "baseline_fill_delay_min": float(ctx.baseline_fill_delay_min),
                "split_id": int(sp["split_id"]),
            }
            split_signal_rows.append(row)
            overall_rows.append(row)
        df_split = pd.DataFrame(split_signal_rows)
        b = ga_exec._rollup_mode(df_split, "baseline")
        split_rows.append(
            {
                "split_id": int(sp["split_id"]),
                "signals_total": int(len(df_split)),
                "entries_valid": int(b["entries_valid"]),
                "baseline_mean_expectancy_net": float(b["mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": 0.0,
                "cvar_improve_ratio": 0.0,
                "maxdd_improve_ratio": 0.0,
            }
        )

    return pd.DataFrame(overall_rows), pd.DataFrame(split_rows)


def evaluate_baseline_bundle(bundle: ga_exec.SymbolBundle) -> Dict[str, Any]:
    signal_df, split_df = baseline_rows_for_bundle(bundle)
    b = ga_exec._rollup_mode(signal_df, "baseline")
    return {
        "signal_df": signal_df,
        "split_df": split_df,
        "metrics": {
            "valid_for_ranking": 1,
            "invalid_reason": "",
            "overall_exec_expectancy_net": float(b["mean_expectancy_net"]),
            "overall_delta_expectancy_exec_minus_baseline": 0.0,
            "overall_cvar_improve_ratio": 0.0,
            "overall_maxdd_improve_ratio": 0.0,
            "overall_entries_valid": int(b["entries_valid"]),
            "overall_entry_rate": float(b["entry_rate"]),
            "overall_exec_taker_share": float(b["taker_share"]),
            "overall_exec_median_fill_delay_min": float(b["median_fill_delay_min"]),
            "overall_exec_p95_fill_delay_min": float(b["p95_fill_delay_min"]),
            "min_split_expectancy_net": 0.0,
        },
    }


def load_historical_exec_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    choices, _meta = pu.load_exec_choices(nx.REPORTS_ROOT)
    by_id = {c.exec_choice_id: c for c in choices}
    for cid in ["E1", "E2"]:
        ch = by_id.get(cid)
        if ch is None or not isinstance(ch.genome, dict):
            continue
        g = ga_exec._repair_genome(copy.deepcopy(ch.genome), mode="tight", repair_hist=None)
        out.append(
            {
                "candidate_id": cid,
                "candidate_type": "historical_exec",
                "label": f"{cid}_historical",
                "genome": g,
            }
        )
    return out


def load_best_nx_candidates(nx_dir: Path, seed: int) -> List[Dict[str, Any]]:
    nx3_fp = nx_dir / "phaseNX3_ablation_results.csv"
    nx_manifest_fp = nx_dir / "run_manifest.json"
    if not nx3_fp.exists():
        return []

    nx3_df = pd.read_csv(nx3_fp)
    if nx3_df.empty:
        return []

    n_per_family = 12
    if nx_manifest_fp.exists():
        try:
            man = json.loads(nx_manifest_fp.read_text(encoding="utf-8"))
            n_ab = int(man.get("compute_budgets_used", {}).get("nx3_ablation_variants", 36))
            n_per_family = max(3, int(n_ab // 3))
        except Exception:
            n_per_family = 12

    base_genome = nx.extract_base_genome()
    ab_variants = nx.build_ablation_variants(base_genome=base_genome, seed=int(seed) + 17, n_per_family=int(n_per_family))
    by_id = {v.variant_id: v for v in ab_variants}

    out: List[Dict[str, Any]] = []
    for fam in [nx.FAMILY_A, nx.FAMILY_B, nx.FAMILY_C]:
        fam_df = nx3_df[nx3_df["family_id"].astype(str) == str(fam)].copy()
        fam_valid = fam_df[to_num(fam_df.get("valid_for_ranking", 0)).fillna(0).astype(int) == 1].copy()
        if fam_valid.empty:
            continue
        best = fam_valid.sort_values(
            ["overall_delta_expectancy_exec_minus_baseline", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"],
            ascending=[False, False, False],
        ).iloc[0]
        vid = str(best["variant_id"])
        if vid not in by_id:
            continue
        out.append(
            {
                "candidate_id": vid,
                "candidate_type": "nx_variant",
                "label": vid,
                "variant": by_id[vid],
            }
        )
    return out


def eval_genome_candidate(
    *,
    genome: Dict[str, Any],
    bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    detailed: bool,
) -> Dict[str, Any]:
    return ga_exec._evaluate_genome(copy.deepcopy(genome), [bundle], args, detailed)


def eval_family_candidate(
    *,
    variant: nx.FamilyVariant,
    bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    detailed: bool,
) -> Dict[str, Any]:
    ev = nx.evaluate_family_variant(variant=variant, bundles=[bundle], args=args, detailed=detailed)
    return {
        **dict(ev.metrics),
        "split_rows_df": ev.split_df.copy(),
        "signal_rows_df": ev.signal_df.copy(),
    }


def evaluate_candidate_with_routes(
    *,
    candidate: Dict[str, Any],
    full_bundle: ga_exec.SymbolBundle,
    route_bundles: Dict[str, ga_exec.SymbolBundle],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[str]]:
    invalid_parts: List[str] = []
    if candidate["candidate_type"] == "historical_exec":
        full = eval_genome_candidate(genome=candidate["genome"], bundle=full_bundle, args=args, detailed=True)
    else:
        full = eval_family_candidate(variant=candidate["variant"], bundle=full_bundle, args=args, detailed=True)

    m = dict(full)
    split_df = m.get("split_rows_df", pd.DataFrame())
    if not isinstance(split_df, pd.DataFrame):
        split_df = pd.DataFrame()
    min_subperiod_delta = float(to_num(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float))).min()) if not split_df.empty else float("nan")

    route_test_counts: Dict[str, int] = {}
    route_entries_valid: Dict[str, int] = {}
    route_valid_flags: Dict[str, int] = {}
    route_delta: Dict[str, float] = {}

    route_ok = True
    for rid, bundle in route_bundles.items():
        if candidate["candidate_type"] == "historical_exec":
            rm = eval_genome_candidate(genome=candidate["genome"], bundle=bundle, args=args, detailed=False)
        else:
            rm = eval_family_candidate(variant=candidate["variant"], bundle=bundle, args=args, detailed=False)
        route_test_counts[rid] = int(bundle_test_count(bundle))
        route_entries_valid[rid] = int(rm.get("overall_entries_valid", 0))
        route_valid_flags[rid] = int(rm.get("valid_for_ranking", 0))
        route_delta[rid] = float(rm.get("overall_delta_expectancy_exec_minus_baseline", np.nan))
        if not (int(rm.get("valid_for_ranking", 0)) == 1 and float(rm.get("overall_delta_expectancy_exec_minus_baseline", np.nan)) > 0.0):
            route_ok = False

    reason_blob = str(m.get("invalid_reason", "")).strip()
    if reason_blob:
        invalid_parts.extend([x.strip() for x in reason_blob.split("|") if x.strip()])

    row = {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_type": str(candidate["candidate_type"]),
        "valid_for_ranking": int(m.get("valid_for_ranking", 0)),
        "invalid_reason": reason_blob,
        "expectancy_net": float(m.get("overall_exec_expectancy_net", np.nan)),
        "delta_expectancy_vs_baseline": float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "cvar_improve_ratio": float(m.get("overall_cvar_improve_ratio", np.nan)),
        "maxdd_improve_ratio": float(m.get("overall_maxdd_improve_ratio", np.nan)),
        "route_pass": int(route_ok),
        "min_subperiod_delta": float(min_subperiod_delta),
        "route_min_test_signals": int(min(route_test_counts.values())) if route_test_counts else 0,
        "route_test_counts_json": json.dumps(route_test_counts, sort_keys=True),
        "route_entries_valid_json": json.dumps(route_entries_valid, sort_keys=True),
        "route_valid_flags_json": json.dumps(route_valid_flags, sort_keys=True),
        "route_delta_json": json.dumps(route_delta, sort_keys=True),
    }
    return row, invalid_parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase R route harness redesign and bounded revalidation")
    ap.add_argument(
        "--nx-dir",
        default="/root/analysis/0.87/reports/execution_layer/PHASENX_EXEC_FAMILY_DISCOVERY_20260227_115329",
    )
    ap.add_argument(
        "--ny-dir",
        default="/root/analysis/0.87/reports/execution_layer/PHASENY_NX_POSTMORTEM_FEASIBILITY_20260227_123357",
    )
    ap.add_argument("--seed", type=int, default=20260225)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    nx_dir = Path(args.nx_dir).resolve()
    ny_dir = Path(args.ny_dir).resolve()
    if not nx_dir.exists():
        raise FileNotFoundError(f"NX dir not found: {nx_dir}")
    if not ny_dir.exists():
        raise FileNotFoundError(f"NY dir not found: {ny_dir}")

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"PHASER_ROUTE_HARNESS_REDESIGN_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze lock validation failed under locked contract")

    rep_df = pd.read_csv(subset_path)
    bundles, load_meta = ga_exec._prepare_bundles(exec_args)
    if not bundles:
        raise RuntimeError("No bundles prepared under frozen harness")
    base_bundle = bundles[0]

    # R0: forensics on legacy route harness.
    legacy_support_df = build_legacy_route_support_table(base_bundle=base_bundle, rep_subset=rep_df, args=exec_args)
    legacy_support_df.to_csv(run_dir / "phaseR0_route_support_table.csv", index=False)
    legacy_route_summary = (
        legacy_support_df.groupby("route_id", as_index=False)
        .agg(
            route_signal_count=("route_signal_count", "max"),
            route_upper_bound_entries=("route_upper_bound_entries", "max"),
            min_split_test_count=("split_test_count", "min"),
            max_split_test_count=("split_test_count", "max"),
            route_trade_gates_reachable=("route_trade_gates_reachable", "max"),
        )
        .sort_values("route_id")
        .reset_index(drop=True)
    )
    failing_legacy = legacy_route_summary[legacy_route_summary["route_trade_gates_reachable"] != 1]["route_id"].astype(str).tolist()
    legacy_route1 = legacy_route_summary[legacy_route_summary["route_id"] == "route1_holdout"].head(1)
    route1_entries = int(legacy_route1["route_upper_bound_entries"].iloc[0]) if not legacy_route1.empty else -1

    r0_lines = [
        "# R0 Route Harness Forensics",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- NY evidence source: `{ny_dir}`",
        f"- Legacy route constructor source: `scripts/phase_nx_exec_family_discovery.py::build_route_bundles`",
        "- Legacy route definitions are imported from `scripts/phase_af_ah_sizing_autorun.py::route_signal_sets`.",
        "- `route1_holdout` is the last `max(120, round(20% * N_subset))` signals; with `N=1200`, this becomes `240` route signals.",
        f"- Under walkforward (`train_ratio={exec_args.train_ratio}`, `wf_splits={exec_args.wf_splits}`), `route1_holdout` yields only `{route1_entries}` scored test signals.",
        f"- Hard gate requires `overall>={int(exec_args.hard_min_trades_overall)}`; therefore `route1_holdout` is ex-ante infeasible.",
        f"- Failing legacy routes: `{failing_legacy}`",
        "",
        "## Legacy Route Support Summary",
        "",
        markdown_table(
            legacy_route_summary,
            ["route_id", "route_signal_count", "route_upper_bound_entries", "min_split_test_count", "max_split_test_count", "route_trade_gates_reachable"],
            n=12,
        ),
        "",
        "Exact split-level support counts are stored in `phaseR0_route_support_table.csv`.",
        "",
    ]
    write_text(run_dir / "phaseR0_route_forensics_report.md", "\n".join(r0_lines))

    # R1: support-feasible redesign.
    repaired_routes, route_examples_df, feasibility_df, route_meta = build_support_feasible_route_family(base_bundle=base_bundle, args=exec_args, coverage_frac=0.60)
    route_examples_df.to_csv(run_dir / "phaseR1_route_examples.csv", index=False)
    feasibility_df.to_csv(run_dir / "phaseR1_feasibility_validation.csv", index=False)

    repair_failed = False
    repair_blockers: List[str] = []
    if int(route_meta["route_count"]) < 2:
        repair_failed = True
        repair_blockers.append(f"insufficient_unique_routes:{int(route_meta['route_count'])}")
    if not feasibility_df.empty and int((feasibility_df["route_trade_gates_reachable"] == 1).all()) != 1:
        repair_failed = True
        repair_blockers.append("some_repaired_routes_below_support_gate")

    r1_lines = [
        "# R1 Route Redesign Spec",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Objective: keep route-based robustness meaningful while guaranteeing every route used in pass/fail logic is support-feasible under unchanged hard gates.",
        "- Redesign rule:",
        f"  - Compute the smallest route size whose walkforward test windows can score at least `{int(exec_args.hard_min_trades_overall)}` trades.",
        f"  - Under current settings, this minimum is `{int(route_meta['min_contexts_for_overall_gate'])}` contexts (producing `{int(route_meta['min_test_signals_for_overall_gate'])}` test signals).",
        f"  - Build deterministic chronological windows of size `max(min_required, ceil({route_meta['coverage_frac']:.2f} * N_subset))`.",
        f"  - Current run uses window size `{int(route_meta['window_n'])}` over `N_subset={len(base_bundle.contexts)}`.",
        "- Route family used for pass/fail logic:",
        "  - `route_front_60pct`: earliest support-feasible 60% window.",
        "  - `route_center_60pct`: centered support-feasible 60% window.",
        "  - `route_back_60pct`: latest support-feasible 60% window.",
        "- Integrity properties:",
        "  - deterministic on chronological order only (no label/outcome lookahead)",
        "  - each route re-runs its own walkforward split construction",
        "  - routes are deduplicated by exact signal-id membership",
        f"- Repair status: `{'failed' if repair_failed else 'ready'}`",
        f"- Repair blockers: `{repair_blockers}`",
        "",
        "## Repaired Route Feasibility",
        "",
        markdown_table(
            feasibility_df,
            ["route_id", "route_signal_count", "wf_test_signal_count", "hard_min_trades_overall", "headroom_vs_overall_gate", "route_trade_gates_reachable"],
            n=12,
        ),
        "",
    ]
    write_text(run_dir / "phaseR1_route_redesign_spec.md", "\n".join(r1_lines))

    # Optional patch summary for the new harness runner.
    patch_lines = [
        "# R Patch Diff Summary",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Files changed:",
        "  - scripts/phase_r_route_harness_redesign.py (new)",
        "- Rationale:",
        "  - Adds a support-feasible route harness and bounded revalidation pack without changing hard gates or the frozen contract.",
        "  - Replaces the infeasible legacy holdout route with deterministic front/center/back support-feasible windows.",
    ]
    write_text(run_dir / "phaseR_patch_diff_summary.md", "\n".join(patch_lines) + "\n")

    # R2: bounded revalidation pack.
    invalid_hist: Counter[str] = Counter()
    reval_rows: List[Dict[str, Any]] = []

    baseline_full = evaluate_baseline_bundle(base_bundle)
    baseline_route_test_counts = {rid: int(bundle_test_count(b)) for rid, b in repaired_routes.items()}
    baseline_route_entries = {}
    for rid, bundle in repaired_routes.items():
        b_eval = evaluate_baseline_bundle(bundle)
        baseline_route_entries[rid] = int(b_eval["metrics"]["overall_entries_valid"])

    reval_rows.append(
        {
            "candidate_id": "BASELINE_FROZEN",
            "candidate_type": "baseline_reference",
            "valid_for_ranking": 1,
            "invalid_reason": "",
            "expectancy_net": float(baseline_full["metrics"]["overall_exec_expectancy_net"]),
            "delta_expectancy_vs_baseline": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "route_pass": int(not repair_failed),
            "min_subperiod_delta": 0.0,
            "route_min_test_signals": int(min(baseline_route_test_counts.values())) if baseline_route_test_counts else 0,
            "route_test_counts_json": json.dumps(baseline_route_test_counts, sort_keys=True),
            "route_entries_valid_json": json.dumps(baseline_route_entries, sort_keys=True),
            "route_valid_flags_json": json.dumps({rid: 1 for rid in repaired_routes}, sort_keys=True),
            "route_delta_json": json.dumps({rid: 0.0 for rid in repaired_routes}, sort_keys=True),
        }
    )

    candidates = load_historical_exec_candidates()
    candidates.extend(load_best_nx_candidates(nx_dir=nx_dir, seed=int(args.seed)))

    for cand in candidates:
        row, invalid_parts = evaluate_candidate_with_routes(candidate=cand, full_bundle=base_bundle, route_bundles=repaired_routes, args=exec_args)
        reval_rows.append(row)
        for part in invalid_parts:
            invalid_hist[str(part)] += 1

    reval_df = pd.DataFrame(reval_rows).sort_values(["candidate_type", "delta_expectancy_vs_baseline"], ascending=[True, False]).reset_index(drop=True)
    reval_df.to_csv(run_dir / "phaseR2_revalidation_results.csv", index=False)
    json_dump(run_dir / "phaseR2_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    non_base = reval_df[reval_df["candidate_type"] != "baseline_reference"].copy()
    survivors = non_base[
        (to_num(non_base["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(non_base["route_pass"]).fillna(0).astype(int) == 1)
        & (to_num(non_base["delta_expectancy_vs_baseline"]) > 0.0)
        & (to_num(non_base["cvar_improve_ratio"]) >= 0.0)
        & (to_num(non_base["maxdd_improve_ratio"]) >= 0.0)
    ].copy()

    r2_lines = [
        "# R2 Revalidation Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Candidate count (incl baseline): `{len(reval_df)}`",
        f"- Repaired route count: `{len(repaired_routes)}`",
        f"- Route-feasible routes: `{int((feasibility_df['route_trade_gates_reachable'] == 1).sum())}` / `{len(feasibility_df)}`",
        f"- Surviving non-baseline candidates (valid + route-pass + positive delta): `{len(survivors)}`",
        "",
        "## Revalidation Results",
        "",
        markdown_table(
            reval_df,
            [
                "candidate_id",
                "candidate_type",
                "valid_for_ranking",
                "expectancy_net",
                "delta_expectancy_vs_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "route_pass",
                "min_subperiod_delta",
                "invalid_reason",
            ],
            n=20,
        ),
        "",
    ]
    write_text(run_dir / "phaseR2_revalidation_report.md", "\n".join(r2_lines))

    # R3: decision.
    if repair_failed:
        classification = "HARNESS_REPAIR_FAILED"
        next_step = f"Stop and fix route design blockers: {repair_blockers}"
    else:
        best_survivor_net = float(to_num(survivors["expectancy_net"]).max()) if not survivors.empty else float("nan")
        if survivors.empty or (np.isfinite(best_survivor_net) and best_survivor_net < 0.0):
            classification = "HARNESS_FIXED_EXEC_STILL_NO_GO"
            next_step = "Move upstream to signal economics redesign; execution-only improvements remain net negative under the fixed harness."
        else:
            classification = "HARNESS_FIXED_EXEC_REMAINS_CANDIDATE"
            next_step = "Run a bounded confirmation pass on the repaired route harness using the surviving candidate set only."

    r3_lines = [
        "# R3 Decision Next Step",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: `{classification}`",
        f"- Repaired routes support-feasible: `{int((not repair_failed) and (len(feasibility_df) > 0) and (feasibility_df['route_trade_gates_reachable'] == 1).all())}`",
        f"- Non-baseline survivors: `{len(survivors)}`",
        "- Single best next step:",
        f"  - {next_step}",
    ]
    write_text(run_dir / "phaseR3_decision_next_step.md", "\n".join(r3_lines) + "\n")

    if classification == "HARNESS_FIXED_EXEC_REMAINS_CANDIDATE":
        survivor_ids = survivors["candidate_id"].astype(str).tolist()
        prompt = (
            "ROLE\n"
            "You are in bounded confirmation mode on the repaired SOL route harness.\n\n"
            "MISSION\n"
            "Re-run only the surviving execution candidates on the support-feasible front/center/back route family under the same frozen contract and unchanged hard gates.\n\n"
            "SURVIVOR SET\n"
            f"{json.dumps(survivor_ids)}\n\n"
            "RULES\n"
            "1) Keep fee/metrics/subset lock unchanged.\n"
            "2) Keep walkforward ON and use the repaired support-feasible routes only.\n"
            "3) No GA expansion and no new execution-family invention.\n"
            "4) Stop NO_GO on first robustness or economics collapse.\n"
        )
        write_text(run_dir / "ready_to_launch_next_prompt.txt", prompt)

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "nx_dir": str(nx_dir),
        "ny_dir": str(ny_dir),
        "freeze_lock_validation": lock_info,
        "load_meta": load_meta,
        "legacy_failing_routes": failing_legacy,
        "repaired_route_meta": route_meta,
        "repaired_route_ids": sorted(repaired_routes.keys()),
        "classification": classification,
    }
    json_dump(run_dir / "phaseR_run_manifest.json", manifest)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "classification": classification,
                "repair_failed": repair_failed,
                "repaired_route_count": int(len(repaired_routes)),
                "survivor_count": int(len(survivors)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

