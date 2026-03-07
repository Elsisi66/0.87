#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
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


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 12) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    if x.empty:
        return "_(none)_"
    lines = ["| " + " | ".join(x.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _rollup_expectancy_from_signal_df(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    n = int(len(df))
    if n <= 0:
        return {
            "signals_total": 0.0,
            "entries_valid": 0.0,
            "entry_rate": float("nan"),
            "expectancy_net": float("nan"),
            "expectancy_gross": float("nan"),
            "pnl_net_sum": float("nan"),
            "pnl_gross_sum": float("nan"),
            "fee_drag_per_signal": float("nan"),
            "fee_drag_per_trade": float("nan"),
            "taker_share": float("nan"),
        }

    fill_col = f"{prefix}_filled"
    valid_col = f"{prefix}_valid_for_metrics"
    pnl_net_col = f"{prefix}_pnl_net_pct"
    pnl_gross_col = f"{prefix}_pnl_gross_pct"
    liq_col = "baseline_fill_liq" if prefix == "baseline" else "exec_fill_liquidity_type"

    filled = to_num(df.get(fill_col, 0)).fillna(0).astype(int)
    valid = to_num(df.get(valid_col, 0)).fillna(0).astype(int)
    mask = (filled == 1) & (valid == 1)
    entries = int(mask.sum())

    net_raw = to_num(df.get(pnl_net_col, np.nan))
    gross_raw = to_num(df.get(pnl_gross_col, np.nan))

    net_sig = np.zeros(n, dtype=float)
    gross_sig = np.zeros(n, dtype=float)

    good_net = mask & net_raw.notna()
    good_gross = mask & gross_raw.notna()

    if int(good_net.sum()) > 0:
        net_sig[good_net.to_numpy(dtype=bool)] = net_raw[good_net].to_numpy(dtype=float)
    if int(good_gross.sum()) > 0:
        gross_sig[good_gross.to_numpy(dtype=bool)] = gross_raw[good_gross].to_numpy(dtype=float)

    liq = df.get(liq_col, pd.Series([""] * n)).astype(str).str.lower()
    taker_share = float(((liq == "taker") & mask).sum() / entries) if entries > 0 else float("nan")

    pnl_net_sum = float(np.sum(net_sig))
    pnl_gross_sum = float(np.sum(gross_sig))
    exp_net = float(np.mean(net_sig))
    exp_gross = float(np.mean(gross_sig))
    fee_drag_per_signal = float(exp_gross - exp_net)
    fee_drag_per_trade = float((pnl_gross_sum - pnl_net_sum) / entries) if entries > 0 else float("nan")

    return {
        "signals_total": float(n),
        "entries_valid": float(entries),
        "entry_rate": float(entries / max(1, n)),
        "expectancy_net": float(exp_net),
        "expectancy_gross": float(exp_gross),
        "pnl_net_sum": float(pnl_net_sum),
        "pnl_gross_sum": float(pnl_gross_sum),
        "fee_drag_per_signal": float(fee_drag_per_signal),
        "fee_drag_per_trade": float(fee_drag_per_trade),
        "taker_share": float(taker_share),
    }


def _min_route_size_for_overall_gate(args: argparse.Namespace) -> Tuple[int, int]:
    for n in range(1, 20000):
        splits = ga_exec._build_walkforward_splits(n=n, train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))
        test_n = int(sum(int(s["test_end"]) - int(s["test_start"]) for s in splits))
        if test_n >= int(args.hard_min_trades_overall):
            return int(n), int(test_n)
    return -1, -1


def main() -> None:
    ap = argparse.ArgumentParser(description="NY postmortem feasibility audit for NX route/economics upper bounds")
    ap.add_argument(
        "--nx-dir",
        default="/root/analysis/0.87/reports/execution_layer/PHASENX_EXEC_FAMILY_DISCOVERY_20260227_115329",
    )
    ap.add_argument("--seed", type=int, default=20260225)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    nx_dir = Path(args.nx_dir).resolve()
    if not nx_dir.exists():
        raise FileNotFoundError(f"NX artifact dir not found: {nx_dir}")

    out_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = out_root / f"PHASENY_NX_POSTMORTEM_FEASIBILITY_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    fee_path = Path(nx.LOCKED["canonical_fee_model"]).resolve()
    metrics_path = Path(nx.LOCKED["canonical_metrics_definition"]).resolve()
    expected_fee_sha = str(nx.LOCKED["expected_fee_sha"])
    expected_metrics_sha = str(nx.LOCKED["expected_metrics_sha"])

    contract_obj: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "allow_freeze_hash_mismatch": int(nx.LOCKED["allow_freeze_hash_mismatch"]),
        "paths": {
            "representative_subset_csv": str(subset_path),
            "canonical_fee_model": str(fee_path),
            "canonical_metrics_definition": str(metrics_path),
            "nx_dir": str(nx_dir),
        },
        "expected_hashes": {
            "fee_sha256": expected_fee_sha,
            "metrics_sha256": expected_metrics_sha,
        },
    }

    fee_sha = nx.sha256_file(fee_path)
    metrics_sha = nx.sha256_file(metrics_path)
    subset_sha = nx.sha256_file(subset_path)
    contract_obj["observed_hashes"] = {
        "canonical_fee_model_sha256": fee_sha,
        "canonical_metrics_definition_sha256": metrics_sha,
        "representative_subset_csv_sha256": subset_sha,
    }
    contract_obj["fee_hash_match_expected"] = int(fee_sha == expected_fee_sha)
    contract_obj["metrics_hash_match_expected"] = int(metrics_sha == expected_metrics_sha)

    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
    contract_obj["ga_exec_freeze_lock_validation"] = lock_info
    contract_obj["contract_lock_pass"] = int(
        contract_obj["fee_hash_match_expected"] == 1
        and contract_obj["metrics_hash_match_expected"] == 1
        and int(lock_info.get("freeze_lock_pass", 0)) == 1
    )
    contract_obj["mismatch_fields"] = []
    if contract_obj["fee_hash_match_expected"] != 1:
        contract_obj["mismatch_fields"].append("fee_hash")
    if contract_obj["metrics_hash_match_expected"] != 1:
        contract_obj["mismatch_fields"].append("metrics_hash")
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        contract_obj["mismatch_fields"].append("freeze_lock_pass")
    json_dump(run_dir / "phaseNY_contract_validation.json", contract_obj)

    if int(contract_obj["contract_lock_pass"]) != 1:
        decision_txt = "\n".join(
            [
                "# NY2 Decision",
                "",
                f"- Generated UTC: {utc_now()}",
                "- Classification: `ROUTE_INFEASIBLE`",
                "- Reason: Contract lock failed; upper-bound audit cannot proceed under frozen harness.",
                f"- Mismatch fields: `{contract_obj['mismatch_fields']}`",
            ]
        )
        write_text(run_dir / "phaseNY2_decision_next_step.md", decision_txt + "\n")
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "route_feasibility": "infeasible",
                    "economic_feasibility": "infeasible",
                    "classification": "ROUTE_INFEASIBLE",
                    "reason": "contract_lock_failed",
                },
                sort_keys=True,
            )
        )
        return

    rep_df = pd.read_csv(subset_path)
    bundles, load_meta = ga_exec._prepare_bundles(exec_args)
    if not bundles:
        raise RuntimeError("No symbol bundles prepared under frozen harness")
    base_bundle = bundles[0]

    route_sets = af.route_signal_sets(rep_df)
    ctx_ids = {str(c.signal_id) for c in base_bundle.contexts}
    min_route_n_for_200, min_route_test_for_200 = _min_route_size_for_overall_gate(exec_args)

    ny0_rows: List[Dict[str, Any]] = []
    for rid, rdf in route_sets.items():
        route_ids = rdf["signal_id"].astype(str).tolist() if "signal_id" in rdf.columns else []
        n_subset = int(len(route_ids))
        n_context = int(sum(1 for sid in route_ids if sid in ctx_ids))
        splits = ga_exec._build_walkforward_splits(
            n=n_context,
            train_ratio=float(exec_args.train_ratio),
            n_splits=int(exec_args.wf_splits),
        )
        n_test = int(sum(int(sp["test_end"]) - int(sp["test_start"]) for sp in splits))
        ub_entries = int(n_test)  # entry_rate=1.0 upper bound on scored (test) signals

        min_trades_overall = max(
            int(exec_args.hard_min_trades_overall),
            int(math.ceil(float(exec_args.hard_min_trade_frac_overall) * max(1, n_test))),
        )
        min_trades_symbol = max(
            int(exec_args.hard_min_trades_symbol),
            int(math.ceil(float(exec_args.hard_min_trade_frac_symbol) * max(1, n_test))),
        )

        pass_symbol = int(ub_entries >= min_trades_symbol)
        pass_overall = int(ub_entries >= min_trades_overall)
        pass_both = int(pass_symbol == 1 and pass_overall == 1)

        if rid == "route1_holdout":
            desc = "last max(120, round(20% of representative subset)) signals"
        elif rid == "route2_reslice":
            desc = "full representative subset"
        else:
            desc = "custom route from af.route_signal_sets()"

        ny0_rows.append(
            {
                "route_id": str(rid),
                "route_definition": desc,
                "subset_signal_count": int(n_subset),
                "context_overlap_count": int(n_context),
                "wf_split_count": int(len(splits)),
                "wf_test_signal_count": int(n_test),
                "upper_bound_entries_entry_rate_1": int(ub_entries),
                "hard_min_trades_symbol": int(min_trades_symbol),
                "hard_min_trades_overall": int(min_trades_overall),
                "headroom_vs_symbol_gate": int(ub_entries - min_trades_symbol),
                "headroom_vs_overall_gate": int(ub_entries - min_trades_overall),
                "symbol_trade_gate_reachable": int(pass_symbol),
                "overall_trade_gate_reachable": int(pass_overall),
                "route_trade_gates_reachable": int(pass_both),
            }
        )

    ny0_df = pd.DataFrame(ny0_rows).sort_values("route_id").reset_index(drop=True)
    ny0_df.to_csv(run_dir / "phaseNY0_route_support_upper_bound.csv", index=False)

    route_feasible = int((to_num(ny0_df["route_trade_gates_reachable"]).fillna(0).astype(int) == 1).all())
    failing_routes = ny0_df[ny0_df["route_trade_gates_reachable"] != 1]["route_id"].astype(str).tolist()

    ny0_lines = [
        "# NY0 Route Support Upper Bound",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Freeze lock pass: `{int(contract_obj['contract_lock_pass'])}`",
        f"- Representative subset rows: `{len(rep_df)}`",
        f"- Base bundle contexts loaded: `{len(base_bundle.contexts)}`",
        f"- Walkforward config: `train_ratio={exec_args.train_ratio}`, `wf_splits={exec_args.wf_splits}`",
        f"- Hard trade gates: `overall>={int(exec_args.hard_min_trades_overall)}`, `symbol>={int(exec_args.hard_min_trades_symbol)}`",
        f"- Minimum route contexts needed for `overall>=200` under this walkforward setting: `{min_route_n_for_200}` (yields `{min_route_test_for_200}` test signals)",
        f"- Route feasibility verdict: `{'feasible' if route_feasible == 1 else 'infeasible'}`",
        f"- Failing routes: `{failing_routes}`",
        "",
        "## Exact Route Definition Used In NX3",
        "",
        "- `route1_holdout`: last `max(120, round(0.20 * N_subset))` signals.",
        "- `route2_reslice`: full representative subset.",
        "",
        "## Route Upper-Bound Table",
        "",
        markdown_table(
            ny0_df,
            [
                "route_id",
                "subset_signal_count",
                "wf_test_signal_count",
                "upper_bound_entries_entry_rate_1",
                "hard_min_trades_overall",
                "hard_min_trades_symbol",
                "overall_trade_gate_reachable",
                "symbol_trade_gate_reachable",
                "route_trade_gates_reachable",
            ],
            n=20,
        ),
        "",
    ]
    write_text(run_dir / "phaseNY0_route_support_report.md", "\n".join(ny0_lines))

    nx3_fp = nx_dir / "phaseNX3_ablation_results.csv"
    if not nx3_fp.exists():
        raise FileNotFoundError(f"Missing NX3 ablation results: {nx3_fp}")
    nx3_df = pd.read_csv(nx3_fp)

    nx_manifest_fp = nx_dir / "run_manifest.json"
    n_per_family = 12
    if nx_manifest_fp.exists():
        try:
            man = json.loads(nx_manifest_fp.read_text(encoding="utf-8"))
            n_ab = int(man.get("compute_budgets_used", {}).get("nx3_ablation_variants", 36))
            n_per_family = max(3, int(n_ab // 3))
        except Exception:
            n_per_family = 12

    base_genome = nx.extract_base_genome()
    ab_variants = nx.build_ablation_variants(base_genome=base_genome, seed=int(args.seed) + 17, n_per_family=int(n_per_family))
    variant_map = {v.variant_id: v for v in ab_variants}

    chosen_ids: List[str] = []
    valid_df = nx3_df[to_num(nx3_df.get("valid_for_ranking", 0)).fillna(0).astype(int) == 1].copy()
    if not valid_df.empty:
        best_overall_id = str(
            valid_df.sort_values(
                ["overall_delta_expectancy_exec_minus_baseline", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"],
                ascending=[False, False, False],
            ).iloc[0]["variant_id"]
        )
        chosen_ids.append(best_overall_id)

    families_without_valid: List[str] = []
    for fam in sorted(nx3_df["family_id"].astype(str).unique().tolist()):
        fam_df = nx3_df[nx3_df["family_id"].astype(str) == fam].copy()
        fam_valid = fam_df[to_num(fam_df.get("valid_for_ranking", 0)).fillna(0).astype(int) == 1].copy()
        if fam_valid.empty:
            families_without_valid.append(str(fam))
            continue
        best_fam_id = str(
            fam_valid.sort_values(
                ["overall_delta_expectancy_exec_minus_baseline", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"],
                ascending=[False, False, False],
            ).iloc[0]["variant_id"]
        )
        chosen_ids.append(best_fam_id)

    chosen_ids = list(dict.fromkeys(chosen_ids))  # preserve order, unique
    chosen_vars = [variant_map[cid] for cid in chosen_ids if cid in variant_map]
    if not chosen_vars:
        raise RuntimeError("No evaluable NX variants resolved for NY1 economics audit")

    econ_rows: List[Dict[str, Any]] = []
    baseline_added = False

    for v in chosen_vars:
        ev = nx.evaluate_family_variant(variant=v, bundles=[base_bundle], args=exec_args, detailed=True)
        m = ev.metrics
        sig = ev.signal_df.copy()
        if sig.empty:
            continue

        b_roll = _rollup_expectancy_from_signal_df(sig, "baseline")
        e_roll = _rollup_expectancy_from_signal_df(sig, "exec")

        if not baseline_added:
            req_lift = max(0.0, -float(b_roll["expectancy_net"]))
            req_cost_frac = float(req_lift / b_roll["fee_drag_per_signal"]) if np.isfinite(b_roll["fee_drag_per_signal"]) and b_roll["fee_drag_per_signal"] > 0 else float("nan")
            econ_rows.append(
                {
                    "record_type": "baseline",
                    "family_id": "BASELINE",
                    "variant_id": "BASELINE_FROZEN",
                    "valid_for_ranking": int(m.get("valid_for_ranking", 0)),
                    "invalid_reason": "",
                    "signals_total": int(b_roll["signals_total"]),
                    "entries_valid": int(b_roll["entries_valid"]),
                    "entry_rate": float(b_roll["entry_rate"]),
                    "taker_share": float(b_roll["taker_share"]),
                    "expectancy_gross": float(b_roll["expectancy_gross"]),
                    "expectancy_net": float(b_roll["expectancy_net"]),
                    "fee_drag_per_signal": float(b_roll["fee_drag_per_signal"]),
                    "fee_drag_per_trade": float(b_roll["fee_drag_per_trade"]),
                    "delta_vs_baseline_net": 0.0,
                    "required_gross_edge_lift_to_net_zero": float(req_lift),
                    "required_gross_edge_lift_to_net_zero_bps": float(req_lift * 1e4),
                    "required_cost_reduction_frac": float(req_cost_frac),
                    "required_cost_reduction_pct": float(req_cost_frac * 100.0) if np.isfinite(req_cost_frac) else float("nan"),
                    "net_can_cross_zero_if_only_cost_reduced": int(np.isfinite(req_cost_frac) and req_cost_frac <= 1.0),
                }
            )
            baseline_added = True

        req_lift = max(0.0, -float(e_roll["expectancy_net"]))
        req_cost_frac = float(req_lift / e_roll["fee_drag_per_signal"]) if np.isfinite(e_roll["fee_drag_per_signal"]) and e_roll["fee_drag_per_signal"] > 0 else float("nan")
        nx3_match = nx3_df[nx3_df["variant_id"].astype(str) == str(v.variant_id)].head(1)
        nx3_valid = int(nx3_match["valid_for_ranking"].iloc[0]) if not nx3_match.empty else int(m.get("valid_for_ranking", 0))
        nx3_reason = str(nx3_match["invalid_reason"].iloc[0]) if (not nx3_match.empty and pd.notna(nx3_match["invalid_reason"].iloc[0])) else str(m.get("invalid_reason", ""))

        econ_rows.append(
            {
                "record_type": "variant",
                "family_id": str(v.family_id),
                "variant_id": str(v.variant_id),
                "valid_for_ranking": int(nx3_valid),
                "invalid_reason": nx3_reason,
                "signals_total": int(e_roll["signals_total"]),
                "entries_valid": int(e_roll["entries_valid"]),
                "entry_rate": float(e_roll["entry_rate"]),
                "taker_share": float(e_roll["taker_share"]),
                "expectancy_gross": float(e_roll["expectancy_gross"]),
                "expectancy_net": float(e_roll["expectancy_net"]),
                "fee_drag_per_signal": float(e_roll["fee_drag_per_signal"]),
                "fee_drag_per_trade": float(e_roll["fee_drag_per_trade"]),
                "delta_vs_baseline_net": float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                "required_gross_edge_lift_to_net_zero": float(req_lift),
                "required_gross_edge_lift_to_net_zero_bps": float(req_lift * 1e4),
                "required_cost_reduction_frac": float(req_cost_frac),
                "required_cost_reduction_pct": float(req_cost_frac * 100.0) if np.isfinite(req_cost_frac) else float("nan"),
                "net_can_cross_zero_if_only_cost_reduced": int(np.isfinite(req_cost_frac) and req_cost_frac <= 1.0),
            }
        )

    econ_df = pd.DataFrame(econ_rows)
    econ_df = econ_df.sort_values(["record_type", "expectancy_net"], ascending=[True, False]).reset_index(drop=True)
    econ_df.to_csv(run_dir / "phaseNY1_economics_upper_bound.csv", index=False)

    variants_only = econ_df[econ_df["record_type"] == "variant"].copy()
    best_variant_net = float(to_num(variants_only["expectancy_net"]).max()) if not variants_only.empty else float("nan")
    best_variant_row = variants_only.sort_values("expectancy_net", ascending=False).head(1) if not variants_only.empty else pd.DataFrame()
    best_variant_id = str(best_variant_row["variant_id"].iloc[0]) if not best_variant_row.empty else ""
    best_variant_req_cost_pct = float(best_variant_row["required_cost_reduction_pct"].iloc[0]) if not best_variant_row.empty else float("nan")

    economic_feasible = int(np.isfinite(best_variant_net) and best_variant_net >= 0.0)
    near_zero = int(np.isfinite(best_variant_net) and (best_variant_net < 0.0) and (best_variant_net >= -0.00010))

    ny1_lines = [
        "# NY1 Economics Upper Bound",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- NX input dir: `{nx_dir}`",
        f"- Best variant by net expectancy: `{best_variant_id}` ({best_variant_net:.10f})",
        f"- Economic feasibility verdict: `{'feasible' if economic_feasible == 1 else 'infeasible'}`",
        f"- Edge near zero band hit (abs(net)<=1 bp/signal and net<0): `{near_zero}`",
        (
            f"- Best-variant required cost reduction to reach net>=0: `{best_variant_req_cost_pct:.3f}%`"
            if np.isfinite(best_variant_req_cost_pct)
            else "- Best-variant required cost reduction to reach net>=0: `nan`"
        ),
        "",
        (
            f"- Families with zero valid_for_ranking variants in NX3 (excluded from detailed re-eval): `{families_without_valid}`"
            if families_without_valid
            else "- Families with zero valid_for_ranking variants in NX3: `[]`"
        ),
        "",
        "## Economics Table",
        "",
        markdown_table(
            econ_df,
            [
                "record_type",
                "family_id",
                "variant_id",
                "valid_for_ranking",
                "expectancy_gross",
                "expectancy_net",
                "fee_drag_per_signal",
                "required_gross_edge_lift_to_net_zero_bps",
                "required_cost_reduction_pct",
                "invalid_reason",
            ],
            n=20,
        ),
        "",
        "Gross/Net fields are computed from detailed signal rows (`*_pnl_gross_pct`, `*_pnl_net_pct`) using the same expectancy convention as the evaluator (zeros for non-filled/non-valid entries).",
        "",
    ]
    write_text(run_dir / "phaseNY1_economics_report.md", "\n".join(ny1_lines))

    if route_feasible != 1:
        classification = "ROUTE_INFEASIBLE"
        next_step = (
            "Redesign route construction/harness so every route evaluated under walkforward has at least 200 scored test signals "
            "(or remove undersized routes from route-pass criteria); then rerun NX3 with unchanged hard gates."
        )
    else:
        if economic_feasible == 1:
            classification = "ROUTE_FEASIBLE_EDGE_NEAR_ZERO"
            next_step = "Execution family v2 is justified with strict robustness-first search around near-zero frontier."
        elif near_zero == 1:
            classification = "ROUTE_FEASIBLE_EDGE_NEAR_ZERO"
            next_step = "Execution family v2 is justified with tight bounded search near the zero-net boundary."
        else:
            classification = "ROUTE_FEASIBLE_BUT_EDGE_NEGATIVE"
            next_step = "Shift effort to signal-layer edge improvements; execution-only changes are unlikely to produce net-positive results."

    decision_lines = [
        "# NY2 Decision Next Step",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Route feasibility: `{'feasible' if route_feasible == 1 else 'infeasible'}`",
        f"- Economic feasibility: `{'feasible' if economic_feasible == 1 else 'infeasible'}`",
        f"- Classification: `{classification}`",
        f"- Best variant net expectancy: `{best_variant_net:.10f}`",
        (
            f"- Best variant required cost reduction to net>=0: `{best_variant_req_cost_pct:.3f}%`"
            if np.isfinite(best_variant_req_cost_pct)
            else "- Best variant required cost reduction to net>=0: `nan`"
        ),
        "- Single best next step:",
        f"  - {next_step}",
    ]
    write_text(run_dir / "phaseNY2_decision_next_step.md", "\n".join(decision_lines) + "\n")

    if classification == "ROUTE_FEASIBLE_EDGE_NEAR_ZERO":
        next_prompt = (
            "ROLE\n"
            "You are in NX-v2 execution-family mode focused on near-zero edge refinement under the frozen contract.\n\n"
            "MISSION\n"
            "Run a bounded execution-family v2 search with unchanged hard gates, duplicate collapse, and full robustness checks, "
            "targeting net expectancy >= 0 without altering the signal layer.\n\n"
            "RULES\n"
            "1) Keep canonical freeze lock and all hard gates unchanged.\n"
            "2) Keep walkforward ON and require route/split/stress/bootstrap robustness.\n"
            "3) Keep no 1h+3m combined GA and no live/paper promotion decisions.\n"
            "4) Stop NO_GO on first robustness collapse.\n"
        )
        write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "nx_input_dir": str(nx_dir),
        "contract_validation_file": str(run_dir / "phaseNY_contract_validation.json"),
        "phase_files": [
            "phaseNY0_route_support_upper_bound.csv",
            "phaseNY0_route_support_report.md",
            "phaseNY1_economics_upper_bound.csv",
            "phaseNY1_economics_report.md",
            "phaseNY2_decision_next_step.md",
        ],
        "route_feasibility": "feasible" if route_feasible == 1 else "infeasible",
        "economic_feasibility": "feasible" if economic_feasible == 1 else "infeasible",
        "classification": classification,
        "best_variant_net_expectancy": float(best_variant_net) if np.isfinite(best_variant_net) else float("nan"),
        "best_variant_required_cost_reduction_pct": float(best_variant_req_cost_pct) if np.isfinite(best_variant_req_cost_pct) else float("nan"),
        "load_meta": load_meta,
    }
    json_dump(run_dir / "phaseNY_run_manifest.json", manifest)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "route_feasibility": manifest["route_feasibility"],
                "economic_feasibility": manifest["economic_feasibility"],
                "classification": classification,
                "best_variant_net_expectancy": manifest["best_variant_net_expectancy"],
                "best_variant_required_cost_reduction_pct": manifest["best_variant_required_cost_reduction_pct"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
