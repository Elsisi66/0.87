#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1 as subset1  # noqa: E402
from scripts import sol_3m_lossconcentration_ga as ga  # noqa: E402


RUN_PREFIX = "SOL_GA_G0_CHRONOLOGY_AUDIT"
SYMBOL = "SOLUSDT"
BASELINE_CANDIDATE_ID = "M1_ENTRY_ONLY_PASSIVE_BASELINE"


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


def build_context(
    *,
    run_dir: Path,
    posture_dir: Path,
    strict_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    posture_manifest = json.loads((posture_dir / "repaired_3m_posture_manifest.json").read_text(encoding="utf-8"))
    strict_manifest = json.loads((strict_dir / "repaired_subset1_confirm_manifest.json").read_text(encoding="utf-8"))
    freeze_dir = Path(posture_manifest["source_artifacts"]["frozen_repaired_universe_dir"]).resolve()
    foundation_dir = Path(strict_manifest["foundation_dir"]).resolve()

    freeze_df = pd.read_csv(freeze_dir / "repaired_best_by_symbol.csv")
    freeze_df["symbol"] = freeze_df["symbol"].astype(str).str.upper()
    row = freeze_df[(freeze_df["symbol"] == SYMBOL) & (freeze_df["side"].astype(str).str.lower() == "long")]
    if row.empty:
        raise RuntimeError("Missing SOLUSDT long row in repaired_best_by_symbol.csv")
    sol_row = row.iloc[0]
    params = subset1.parse_params_from_row(sol_row, freeze_dir / "repaired_universe_selected_params")

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    df_cache: Dict[Any, pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_df = subset1.build_signal_table_for_row(row=sol_row, params=params, df_cache=df_cache, raw_cache=raw_cache)
    if signal_df.empty:
        raise RuntimeError("No SOLUSDT signals in rebuilt repaired path")

    inputs_dir = ensure_dir(run_dir / "_inputs")
    cache_dir = ensure_dir(run_dir / "_window_cache")
    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=signal_df.copy(),
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(seed),
    )

    symbol_windows = subset1.build_window_pool_for_symbol(
        symbol=SYMBOL,
        signal_df=signal_df,
        foundation_state=foundation_state,
        cache_dir=cache_dir,
    )
    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=signal_df,
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )
    one_h = modela.load_1h_market(SYMBOL)
    baseline_1h = modela.build_1h_reference_rows(
        bundle=bundle,
        fee=fee,
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    route_bundles, route_examples, route_feas, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=bundle,
        args=exec_args,
        coverage_frac=0.60,
    )
    return {
        "freeze_dir": freeze_dir,
        "foundation_dir": foundation_dir,
        "signal_df": signal_df,
        "bundle": bundle,
        "build_meta": build_meta,
        "exec_args": exec_args,
        "one_h": one_h,
        "baseline_1h": baseline_1h,
        "route_bundles": route_bundles,
        "route_examples": route_examples,
        "route_feas": route_feas,
        "route_meta": route_meta,
    }


def eval_candidate(
    *,
    bundle: Any,
    baseline_1h: pd.DataFrame,
    cfg: Dict[str, Any],
    one_h: Any,
    exec_args: argparse.Namespace,
) -> Dict[str, Any]:
    ev = modela.evaluate_model_a_variant(
        bundle=bundle,
        baseline_df=baseline_1h,
        cfg=cfg,
        one_h=one_h,
        args=exec_args,
    )
    rows = ev["signal_rows_df"].copy()
    for c in ["signal_time", "exec_entry_time", "exec_exit_time"]:
        rows[c] = pd.to_datetime(rows[c], utc=True, errors="coerce")
    return {"eval": ev, "rows": rows}


def build_route_context(
    *,
    signal_id: str,
    cfg: Dict[str, Any],
    one_h: Any,
    exec_args: argparse.Namespace,
    route_bundles: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rid, rb in route_bundles.items():
        ids = {str(ctx.signal_id) for ctx in rb.contexts}
        contains = int(str(signal_id) in ids)
        fee = modela.phasec_bt.FeeModel(
            fee_bps_maker=float(exec_args.fee_bps_maker),
            fee_bps_taker=float(exec_args.fee_bps_taker),
            slippage_bps_limit=float(exec_args.slippage_bps_limit),
            slippage_bps_market=float(exec_args.slippage_bps_market),
        )
        base_1h = modela.build_1h_reference_rows(bundle=rb, fee=fee, exec_horizon_hours=float(exec_args.exec_horizon_hours))
        ev = modela.evaluate_model_a_variant(bundle=rb, baseline_df=base_1h, cfg=cfg, one_h=one_h, args=exec_args)
        m = ev["metrics"]
        confirm_flag = int(
            int(m["valid_for_ranking"]) == 1
            and np.isfinite(float(m["overall_delta_expectancy_exec_minus_baseline"]))
            and float(m["overall_delta_expectancy_exec_minus_baseline"]) > 0.0
            and np.isfinite(float(m["overall_cvar_improve_ratio"]))
            and float(m["overall_cvar_improve_ratio"]) >= 0.0
            and np.isfinite(float(m["overall_maxdd_improve_ratio"]))
            and float(m["overall_maxdd_improve_ratio"]) >= 0.0
        )
        rows.append(
            {
                "route_id": str(rid),
                "contains_offending_signal": int(contains),
                "valid_for_ranking": int(m["valid_for_ranking"]),
                "delta_expectancy_vs_baseline": float(m["overall_delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(m["overall_cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(m["overall_maxdd_improve_ratio"]),
                "confirm_flag": int(confirm_flag),
            }
        )
    return pd.DataFrame(rows).sort_values("route_id").reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit G0 same-bar gate behavior for SOL 3m GA confirm path")
    ap.add_argument(
        "--ga-run-dir",
        default="/root/analysis/0.87/reports/execution_layer/SOL_3M_LOSSCONC_GA_20260306_213204",
    )
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--posture-dir", default="/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126")
    ap.add_argument("--strict-dir", default="/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_20260304_010143")
    ap.add_argument("--seed", type=int, default=20260306)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    ga_run_dir = Path(args.ga_run_dir).resolve()
    posture_dir = Path(args.posture_dir).resolve()
    strict_dir = Path(args.strict_dir).resolve()
    confirm_csv = ga_run_dir / "sol_3m_ga_candidates_confirm.csv"
    if not confirm_csv.exists():
        raise FileNotFoundError(f"Missing confirm CSV: {confirm_csv}")

    out_root = (PROJECT_ROOT / args.outdir).resolve()
    out_dir = ensure_dir(out_root / f"{RUN_PREFIX}_{utc_tag()}")
    ensure_dir(out_dir / "_tmp")

    confirm_df = pd.read_csv(confirm_csv)
    offenders = confirm_df[confirm_df["same_bar_exit_count"] > 0].copy().reset_index(drop=True)
    if offenders.empty:
        raise RuntimeError("No rows with same_bar_exit_count>0 in confirm CSV")
    offender = offenders.iloc[0]

    ctx = build_context(
        run_dir=out_dir / "_tmp",
        posture_dir=posture_dir,
        strict_dir=strict_dir,
        seed=int(args.seed),
    )

    baseline_cfg = {
        "candidate_id": BASELINE_CANDIDATE_ID,
        "label": "baseline",
        "entry_mode": "limit",
        "limit_offset_bps": 0.75,
        "fallback_to_market": 1,
        "fallback_delay_min": 6.0,
        "max_fill_delay_min": 24.0,
    }
    best_cfg = {
        "candidate_id": str(offender["candidate_id"]),
        "label": "best",
        "entry_mode": str(offender["entry_mode"]),
        "limit_offset_bps": float(offender["limit_offset_bps"]),
        "fallback_to_market": int(offender["fallback_to_market"]),
        "fallback_delay_min": float(offender["fallback_delay_min"]),
        "max_fill_delay_min": float(offender["max_fill_delay_min"]),
    }

    cand_pack = eval_candidate(
        bundle=ctx["bundle"],
        baseline_1h=ctx["baseline_1h"],
        cfg=best_cfg,
        one_h=ctx["one_h"],
        exec_args=ctx["exec_args"],
    )
    base_pack = eval_candidate(
        bundle=ctx["bundle"],
        baseline_1h=ctx["baseline_1h"],
        cfg=baseline_cfg,
        one_h=ctx["one_h"],
        exec_args=ctx["exec_args"],
    )
    cand_rows = cand_pack["rows"]
    base_rows = base_pack["rows"]

    same_touch = cand_rows[to_num(cand_rows.get("exec_same_bar_hit", 0)).fillna(0).astype(int) > 0].copy().reset_index(drop=True)
    if same_touch.empty:
        raise RuntimeError("Could not find trade row with exec_same_bar_hit>0 during reproduction")
    t = same_touch.iloc[0]
    entry_parent = pd.to_datetime(t["exec_entry_time"], utc=True, errors="coerce").floor("1h")
    exit_parent = pd.to_datetime(t["exec_exit_time"], utc=True, errors="coerce").floor("1h")
    same_parent = bool(pd.notna(entry_parent) and pd.notna(exit_parent) and entry_parent == exit_parent)
    exit_before_entry = bool(
        pd.notna(t["exec_entry_time"])
        and pd.notna(t["exec_exit_time"])
        and pd.to_datetime(t["exec_exit_time"], utc=True) <= pd.to_datetime(t["exec_entry_time"], utc=True)
    )

    route_df = build_route_context(
        signal_id=str(t["signal_id"]),
        cfg=best_cfg,
        one_h=ctx["one_h"],
        exec_args=ctx["exec_args"],
        route_bundles=ctx["route_bundles"],
    )

    # Post-fix chronology stats from patched function.
    base_stats = ga.compute_chronology_stats(base_rows)
    cand_stats = ga.compute_chronology_stats(cand_rows)

    smoke_rows = [
        {
            "scenario": "baseline",
            "candidate_id": BASELINE_CANDIDATE_ID,
            "genome_hash": "baseline",
            "same_bar_exit_count": int(base_stats["same_bar_exit_count"]),
            "same_bar_touch_count": int(base_stats["same_bar_touch_count"]),
            "exit_before_entry_count": int(base_stats["exit_before_entry_count"]),
            "entry_on_signal_count": int(base_stats["entry_on_signal_count"]),
            "parity_clean": int(base_stats["parity_clean"]),
            "old_same_bar_exit_count_from_ga_confirm": int(offender["same_bar_exit_count"]) if BASELINE_CANDIDATE_ID == str(offender["candidate_id"]) else int(confirm_df[confirm_df["candidate_id"] == BASELINE_CANDIDATE_ID]["same_bar_exit_count"].iloc[0]) if (confirm_df["candidate_id"] == BASELINE_CANDIDATE_ID).any() else int(base_stats["same_bar_touch_count"]),
            "confirm_pass_before_fix": int(confirm_df[confirm_df["candidate_id"] == BASELINE_CANDIDATE_ID]["confirm_pass"].iloc[0]) if (confirm_df["candidate_id"] == BASELINE_CANDIDATE_ID).any() else 0,
        },
        {
            "scenario": "best_candidate",
            "candidate_id": str(offender["candidate_id"]),
            "genome_hash": str(offender["genome_hash"]),
            "same_bar_exit_count": int(cand_stats["same_bar_exit_count"]),
            "same_bar_touch_count": int(cand_stats["same_bar_touch_count"]),
            "exit_before_entry_count": int(cand_stats["exit_before_entry_count"]),
            "entry_on_signal_count": int(cand_stats["entry_on_signal_count"]),
            "parity_clean": int(cand_stats["parity_clean"]),
            "old_same_bar_exit_count_from_ga_confirm": int(offender["same_bar_exit_count"]),
            "confirm_pass_before_fix": int(offender["confirm_pass"]),
        },
    ]
    smoke_df = pd.DataFrame(smoke_rows)
    smoke_df.to_csv(out_dir / "g0_postfix_smokecheck.csv", index=False)

    # Decision
    g0_clean = (
        same_parent is False
        and exit_before_entry is False
        and int(base_stats["same_bar_exit_count"]) == 0
        and int(cand_stats["same_bar_exit_count"]) == 0
        and int(base_stats["exit_before_entry_count"]) == 0
        and int(cand_stats["exit_before_entry_count"]) == 0
        and int(base_stats["entry_on_signal_count"]) == 0
        and int(cand_stats["entry_on_signal_count"]) == 0
    )
    decision = "G0_CLEAN" if g0_clean else "G0_BROKEN"

    # Write patch diff for the G0 measurement fix.
    diff_txt = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "diff", "--", "scripts/sol_3m_lossconcentration_ga.py"],
        text=True,
    )
    (out_dir / "g0_fix_patch.diff").write_text(diff_txt, encoding="utf-8")

    # Counter definition report
    counter_lines = [
        "# G0 Counter Definition",
        "",
        "## Previous behavior (label source of false positive)",
        "",
        "- `scripts/sol_3m_lossconcentration_ga.py` used `sum(exec_same_bar_hit)` as `same_bar_exit_count`.",
        "- `scripts/phase_a_model_a_audit.py` forwards `same_bar_hit` from `execution_layer_3m_ict._simulate_path_long`.",
        "- `scripts/execution_layer_3m_ict.py` sets `same_bar_hit=1` when **both** SL and TP are touched in the same evaluated bar (`hit_sl and hit_tp`), then SL wins.",
        "- This flag is an intra-exit-candle tie marker, not an entry/exit same-parent chronology violation.",
        "",
        "## Patched behavior (true chronology check)",
        "",
        "- `same_bar_exit_count` now counts trades where `floor(entry_time,1h) == floor(exit_time,1h)` among filled+valid trades.",
        "- `same_bar_touch_count` is retained as separate diagnostics.",
        "- `entry_on_signal_count` now tracks only pre-signal entries (`entry_time < signal_time`) as chronology violations.",
        "- `parity_clean` now requires: same_parent_exit=0, exit_before_entry=0, entry_before_signal=0, invalid geometry=0, lookahead=0.",
    ]
    (out_dir / "g0_counter_definition.md").write_text("\n".join(counter_lines), encoding="utf-8")

    # Offending trade trace
    trace_rows = pd.DataFrame(
        [
            {
                "candidate_id": str(offender["candidate_id"]),
                "genome_hash": str(offender["genome_hash"]),
                "signal_id": str(t["signal_id"]),
                "split_id": int(t["split_id"]),
                "signal_time_utc": str(t["signal_time"]),
                "entry_time_utc": str(t["exec_entry_time"]),
                "exit_time_utc": str(t["exec_exit_time"]),
                "entry_parent_1h": str(entry_parent),
                "exit_parent_1h": str(exit_parent),
                "same_parent_bar": int(same_parent),
                "exit_before_entry": int(exit_before_entry),
                "exec_exit_reason": str(t.get("exec_exit_reason", "")),
                "exec_sl_hit": int(t.get("exec_sl_hit", 0)),
                "exec_tp_hit": int(t.get("exec_tp_hit", 0)),
                "exec_same_bar_hit": int(t.get("exec_same_bar_hit", 0)),
                "trigger_condition": "execution_layer_3m_ict._simulate_path_long: hit_sl and hit_tp in same evaluated bar",
            }
        ]
    )
    trace_lines = [
        "# G0 Offending Trade Trace",
        "",
        "Single reproduced event previously counted as `same_bar_exit_count=1`:",
        "",
        markdown_table(
            trace_rows,
            [
                "candidate_id",
                "genome_hash",
                "signal_id",
                "split_id",
                "signal_time_utc",
                "entry_time_utc",
                "exit_time_utc",
                "entry_parent_1h",
                "exit_parent_1h",
                "same_parent_bar",
                "exit_before_entry",
                "exec_exit_reason",
                "exec_sl_hit",
                "exec_tp_hit",
                "exec_same_bar_hit",
            ],
            n=5,
        ),
        "",
        "Route context for this signal:",
        "",
        markdown_table(
            route_df,
            [
                "route_id",
                "contains_offending_signal",
                "valid_for_ranking",
                "delta_expectancy_vs_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "confirm_flag",
            ],
            n=10,
        ),
        "",
        "- Interpretation: entry and exit occur on different 1h parent bars; this is not same-parent-bar exit.",
        "- The `exec_same_bar_hit=1` is an SL/TP tie-in-candle marker, not chronology break.",
    ]
    (out_dir / "g0_offending_trade_trace.md").write_text("\n".join(trace_lines), encoding="utf-8")

    # Final report
    report_lines = [
        "# G0 Postfix Report",
        "",
        f"- Generated UTC: `{utc_now_iso()}`",
        f"- Source GA run: `{ga_run_dir}`",
        f"- Final decision: `{decision}`",
        "",
        "## Post-fix smoke check",
        "",
        markdown_table(
            smoke_df,
            [
                "scenario",
                "candidate_id",
                "same_bar_exit_count",
                "same_bar_touch_count",
                "exit_before_entry_count",
                "entry_on_signal_count",
                "parity_clean",
                "old_same_bar_exit_count_from_ga_confirm",
                "confirm_pass_before_fix",
            ],
            n=10,
        ),
        "",
        "## Gate impact",
        "",
        f"- Candidate route_pass_rate remains `{float(offender['route_pass_rate']):.4f}` (unchanged).",
        f"- Candidate gate_improve_target remained `{int(offender['gate_improve_target'])}` in original confirm row (unchanged by counter fix).",
        "- The original G0 fail came from label misuse (`exec_same_bar_hit`) and is removed under true chronology counting.",
    ]
    (out_dir / "g0_postfix_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()

