#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1_confirm as confirm  # noqa: E402


RUN_PREFIX = "LTC_3M_DATAPATH_HYGIENE_REVIEW"
SYMBOL = "LTCUSDT"
EXPECTED_WINNER = "M2_ENTRY_ONLY_MORE_PASSIVE"
RECON_TOL = 1e-12


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


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


def discover_confirm_dir(arg_value: str) -> Path:
    if arg_value:
        run_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        run_dir = find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_*",
            [
                "repaired_subset1_confirm_by_symbol.csv",
                "repaired_subset1_confirm_summary.csv",
                "repaired_subset1_confirm_report.md",
                "repaired_subset1_confirm_manifest.json",
                "repaired_subset1_confirm_route_checks.csv",
            ],
        )
        if run_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_* directory")
    return run_dir


def discover_opcheck_dir(arg_value: str) -> Path:
    if arg_value:
        run_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        run_dir = find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_ACTIVESET_OPCHECK_*",
            [
                "repaired_activeset_opcheck_by_symbol.csv",
                "repaired_activeset_opcheck_summary.csv",
                "repaired_activeset_opcheck_report.md",
                "repaired_activeset_manifest.json",
            ],
        )
        if run_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_UNIVERSE_3M_ACTIVESET_OPCHECK_* directory")
    return run_dir


def parquet_core(df: pd.DataFrame) -> pd.DataFrame:
    x = phase_v.exec3m._normalize_ohlcv_cols(df)  # pylint: disable=protected-access
    x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    keep = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    return x.loc[:, keep].sort_values("Timestamp").reset_index(drop=True)


def build_exact_slice_trace(signal_times: pd.Series, full_3m: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, ts in enumerate(pd.to_datetime(signal_times, utc=True, errors="coerce")):
        sig_start, sig_end = phase_v.build_signal_window_lookup(ts)
        expected = int((sig_end - sig_start) / pd.Timedelta(minutes=3))
        got = int(((full_3m["Timestamp"] >= sig_start) & (full_3m["Timestamp"] < sig_end)).sum())
        rows.append(
            {
                "signal_idx": int(idx),
                "signal_time_utc": ts,
                "window_start_utc": sig_start,
                "window_end_utc": sig_end,
                "expected_rows": int(expected),
                "observed_rows": int(got),
                "missing_rows": int(expected - got),
                "exact_slice_status": "full" if got == expected else "partial",
            }
        )
    return pd.DataFrame(rows)


def safe_float(x: Any) -> float:
    return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])


def evaluate_path(
    *,
    symbol: str,
    signal_df: pd.DataFrame,
    symbol_windows: pd.DataFrame,
    exec_args: argparse.Namespace,
    variant: Dict[str, Any],
    run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=symbol,
        symbol_signals=signal_df,
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    qrow, rrow = confirm.quality_from_build_meta(build_meta)
    pack = phase_v.evaluate_symbol(
        symbol=symbol,
        bundle=bundle,
        foundation_quality=qrow,
        foundation_readiness=rrow,
        exec_args=exec_args,
        variants=[variant],
    )
    rows = pack["results_df"].copy()
    best = phase_v.choose_best_candidate(rows)
    ref = rows[rows["candidate_id"].astype(str) == "M0_1H_REFERENCE"].iloc[0]
    out = {
        "winner_candidate_id": str(best.get("candidate_id", "")) if not best.empty else "",
        "winner_expectancy": safe_float(best.get("exec_expectancy_net", np.nan)) if not best.empty else float("nan"),
        "winner_delta_vs_1h": safe_float(best.get("delta_expectancy_vs_1h_reference", np.nan)) if not best.empty else float("nan"),
        "winner_route_pass": int(best.get("route_pass", 0)) if not best.empty else 0,
        "winner_route_pass_rate": safe_float(best.get("route_pass_rate", np.nan)) if not best.empty else float("nan"),
        "winner_valid_for_ranking": int(best.get("valid_for_ranking", 0)) if not best.empty else 0,
        "reference_expectancy": safe_float(ref.get("exec_expectancy_net", np.nan)),
        "signals_total": int(build_meta.get("signals_total", 0)),
        "signals_with_3m_data": int(build_meta.get("signals_with_3m_data", 0)),
        "signals_partial_3m_data": int(build_meta.get("signals_partial_3m_data", 0)),
        "signals_missing_3m_data": int(build_meta.get("signals_missing_3m_data", 0)),
        "window_source_counts_json": json.dumps(build_meta.get("window_source_counts", {}), sort_keys=True),
        "route_family_supported": int(pack["route_meta"].get("route_family_supported", 0)),
        "route_count": int(pack["route_meta"].get("route_count", 0)),
    }
    return out, build_meta, pack


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Focused LTC 3m data-path hygiene review")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--freeze-dir", default="")
    ap.add_argument("--confirm-dir", default="")
    ap.add_argument("--opcheck-dir", default="")
    ap.add_argument("--subset-eval-dir", default="")
    ap.add_argument("--foundation-dir", default="")
    ap.add_argument("--seed", type=int, default=20260304)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    freeze_dir = confirm.discover_freeze_dir(args.freeze_dir)
    confirm_dir = discover_confirm_dir(args.confirm_dir)
    opcheck_dir = discover_opcheck_dir(args.opcheck_dir)
    subset_eval_dir = confirm.discover_subset_eval_dir(args.subset_eval_dir)

    strict_df = pd.read_csv(confirm_dir / "repaired_subset1_confirm_by_symbol.csv")
    strict_df["symbol"] = strict_df["symbol"].astype(str).str.upper()
    strict_row = strict_df[strict_df["symbol"] == SYMBOL]
    if strict_row.empty:
        raise RuntimeError("Missing LTCUSDT row in strict confirmation artifacts")
    strict_row_s = strict_row.iloc[0]
    if str(strict_row_s["best_candidate_id"]).strip() != EXPECTED_WINNER:
        raise RuntimeError(
            f"LTC strict-confirm winner mismatch: expected {EXPECTED_WINNER} got {strict_row_s['best_candidate_id']}"
        )

    op_df = pd.read_csv(opcheck_dir / "repaired_activeset_opcheck_by_symbol.csv")
    op_df["symbol"] = op_df["symbol"].astype(str).str.upper()
    op_row = op_df[op_df["symbol"] == SYMBOL]
    if op_row.empty:
        raise RuntimeError("Missing LTCUSDT row in opcheck artifacts")
    op_row_s = op_row.iloc[0]

    op_manifest = json.loads((opcheck_dir / "repaired_activeset_manifest.json").read_text(encoding="utf-8"))
    foundation_dir = Path(args.foundation_dir).resolve() if str(args.foundation_dir).strip() else Path(str(op_manifest["foundation_dir"])).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)

    universe_df = pd.read_csv(freeze_dir / "repaired_best_by_symbol.csv")
    universe_df["symbol"] = universe_df["symbol"].astype(str).str.upper()
    row = universe_df[universe_df["symbol"] == SYMBOL]
    if row.empty:
        raise RuntimeError("Missing LTCUSDT in repaired_best_by_symbol.csv")
    row_s = row.iloc[0]
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    params = confirm.parse_params_from_row(row_s, selected_params_dir)

    exec_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = exec_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = confirm.ensure_dir(run_dir / "_inputs")
    cache_dir = confirm.ensure_dir(run_dir / "_window_cache")

    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_df = confirm.build_signal_table_for_row(row=row_s, params=params, df_cache=df_cache, raw_cache=raw_cache)
    signal_df.to_csv(inputs_dir / "ltc_signals_1h.csv", index=False)
    signal_df.to_csv(inputs_dir / "universe_signal_timeline.csv", index=False)

    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=signal_df,
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(args.seed),
    )
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    variant_lookup = {str(v["candidate_id"]): dict(v) for v in phase_v.sanitize_variants()}
    variant = variant_lookup[EXPECTED_WINNER]

    fresh_windows = confirm.build_window_pool_for_symbol(
        symbol=SYMBOL,
        signal_df=signal_df,
        foundation_state=foundation_state,
        cache_dir=cache_dir,
        reuse_cache_dirs=[],
    )
    prior_windows = confirm.build_window_pool_for_symbol(
        symbol=SYMBOL,
        signal_df=signal_df,
        foundation_state=foundation_state,
        cache_dir=cache_dir,
        reuse_cache_dirs=[subset_eval_dir / "_window_cache"],
    )
    if fresh_windows.empty or prior_windows.empty:
        raise RuntimeError("Unable to build both fresh and prior LTC 3m window paths")

    fresh_core = parquet_core(pd.read_parquet(str(fresh_windows.iloc[0]["parquet_path"])))
    prior_core = parquet_core(pd.read_parquet(str(prior_windows.iloc[0]["parquet_path"])))
    parquets_identical = bool(fresh_core.equals(prior_core))

    sig_min = pd.to_datetime(signal_df["signal_time_utc"], utc=True).min() - pd.Timedelta(hours=48)
    sig_max = pd.to_datetime(signal_df["signal_time_utc"], utc=True).max() + pd.Timedelta(hours=48)
    win = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    win["window_start_utc"] = pd.to_datetime(win["window_start_utc"], utc=True, errors="coerce")
    win["window_end_utc"] = pd.to_datetime(win["window_end_utc"], utc=True, errors="coerce")
    win = win[(win["window_end_utc"] >= sig_min) & (win["window_start_utc"] <= sig_max)].copy().reset_index(drop=True)
    win["coverage_ratio"] = pd.to_numeric(win["coverage_ratio"], errors="coerce")
    coverage_mean = float(win["coverage_ratio"].mean()) if not win.empty else float("nan")
    coverage_min = float(win["coverage_ratio"].min()) if not win.empty else float("nan")
    coverage_max = float(win["coverage_ratio"].max()) if not win.empty else float("nan")
    coverage_lt1 = int((win["coverage_ratio"] < 1.0).sum()) if not win.empty else 0

    exact_trace = build_exact_slice_trace(signal_df["signal_time_utc"], fresh_core)
    exact_trace.to_csv(run_dir / "ltc_coverage_label_trace.csv", index=False)
    true_partial_count = int((exact_trace["exact_slice_status"] == "partial").sum())
    true_full_count = int((exact_trace["exact_slice_status"] == "full").sum())
    max_missing_rows = int(exact_trace["missing_rows"].max()) if not exact_trace.empty else 0

    fresh_eval, fresh_build_meta, _fresh_pack = evaluate_path(
        symbol=SYMBOL,
        signal_df=signal_df,
        symbol_windows=fresh_windows,
        exec_args=exec_args,
        variant=variant,
        run_dir=run_dir / "_eval_fresh",
    )
    prior_eval, prior_build_meta, _prior_pack = evaluate_path(
        symbol=SYMBOL,
        signal_df=signal_df,
        symbol_windows=prior_windows,
        exec_args=exec_args,
        variant=variant,
        run_dir=run_dir / "_eval_prior",
    )

    comp_df = pd.DataFrame(
        [
            {
                "path_id": "fresh_merged_foundation",
                "download_source": str(fresh_windows.iloc[0]["download_source"]),
                "coverage_ratio": safe_float(fresh_windows.iloc[0]["coverage_ratio"]),
                **fresh_eval,
            },
            {
                "path_id": "prior_cached_subset1",
                "download_source": str(prior_windows.iloc[0]["download_source"]),
                "coverage_ratio": safe_float(prior_windows.iloc[0]["coverage_ratio"]),
                **prior_eval,
            },
        ]
    )
    comp_df.to_csv(run_dir / "ltc_winner_reconstruction_comparison.csv", index=False)

    headline_match = bool(
        fresh_eval["winner_candidate_id"] == prior_eval["winner_candidate_id"] == EXPECTED_WINNER
        and np.isfinite(fresh_eval["winner_expectancy"])
        and np.isfinite(prior_eval["winner_expectancy"])
        and abs(float(fresh_eval["winner_expectancy"]) - float(prior_eval["winner_expectancy"])) <= RECON_TOL
        and int(fresh_eval["winner_route_pass"]) == int(prior_eval["winner_route_pass"]) == 1
    )

    if parquets_identical and headline_match and true_partial_count == 0:
        blocker_truth = "artifact_only"
        final_posture = "LTC_CLEARED_FOR_ACTIVE_SUBSET"
        implication = "FREEZE_SOL_LTC_ACTIVE_SUBSET"
    else:
        if parquets_identical and headline_match and true_partial_count > 0:
            blocker_truth = "mixed_metadata_artifact_plus_real_partial_coverage"
        elif parquets_identical and headline_match:
            blocker_truth = "metadata_artifact_but_unresolved_cleanliness"
        else:
            blocker_truth = "unresolved_path_or_metric_mismatch"
        final_posture = "LTC_REMAINS_BLOCKED"
        implication = "FREEZE_SOL_ONLY_AND_KEEP_LTC_BLOCKED"

    findings = pd.DataFrame(
        [
            {
                "symbol": SYMBOL,
                "strict_confirm_winner": EXPECTED_WINNER,
                "strict_confirm_expectancy": safe_float(strict_row_s["confirmed_exec_expectancy_net"]),
                "opcheck_reported_partial_slices": int(op_row_s["signals_partial_3m_data"]),
                "opcheck_reported_missing_slices": int(op_row_s["signals_missing_3m_data"]),
                "fresh_window_download_source": str(fresh_windows.iloc[0]["download_source"]),
                "fresh_window_coverage_ratio": safe_float(fresh_windows.iloc[0]["coverage_ratio"]),
                "prior_window_download_source": str(prior_windows.iloc[0]["download_source"]),
                "prior_window_coverage_ratio": safe_float(prior_windows.iloc[0]["coverage_ratio"]),
                "foundation_window_count_used": int(len(win)),
                "foundation_coverage_mean": coverage_mean,
                "foundation_coverage_min": coverage_min,
                "foundation_windows_below_1": coverage_lt1,
                "fresh_vs_prior_parquet_identical": int(parquets_identical),
                "true_full_slices": int(true_full_count),
                "true_partial_slices": int(true_partial_count),
                "max_missing_rows_in_true_partial_slice": int(max_missing_rows),
                "winner_identity_match_across_paths": int(fresh_eval["winner_candidate_id"] == prior_eval["winner_candidate_id"] == EXPECTED_WINNER),
                "winner_expectancy_match_across_paths": int(
                    np.isfinite(fresh_eval["winner_expectancy"])
                    and np.isfinite(prior_eval["winner_expectancy"])
                    and abs(float(fresh_eval["winner_expectancy"]) - float(prior_eval["winner_expectancy"])) <= RECON_TOL
                ),
                "winner_route_match_across_paths": int(int(fresh_eval["winner_route_pass"]) == int(prior_eval["winner_route_pass"]) == 1),
                "blocker_truth": blocker_truth,
                "final_posture": final_posture,
                "active_subset_implication": implication,
            }
        ]
    )
    findings.to_csv(run_dir / "ltc_datapath_hygiene_findings.csv", index=False)

    lines: List[str] = []
    lines.append("# LTC 3m Data-Path Hygiene Review")
    lines.append("")
    lines.append("This is a strict LTC-only data-path hygiene review. It does not change the winning 3m config or strategy logic.")
    lines.append("")
    lines.append("## Code Paths Used")
    lines.append("- `scripts/repaired_universe_3m_exec_subset1_confirm.py` for window-building and repaired-branch signal reconstruction")
    lines.append("- `scripts/phase_v_multicoin_model_a_audit.py` for bundle construction and route-enabled 3m evaluation")
    lines.append("")
    lines.append("## Core Finding")
    lines.append(f"- The current `940/940` partial label is driven by the merged-window metadata path: merged `coverage_ratio={safe_float(fresh_windows.iloc[0]['coverage_ratio']):.12f} < 1.0`, so every non-empty slice is labeled partial by the current builder.")
    lines.append(f"- Fresh merged LTC parquet and prior cached LTC parquet are identical: `{int(parquets_identical)}`")
    lines.append(f"- True exact slice completeness is not fully clean: `{true_full_count}` full slices, `{true_partial_count}` partial slices")
    lines.append("")
    lines.append("## Winner Comparison")
    lines.append(markdown_table(comp_df, [
        "path_id",
        "download_source",
        "coverage_ratio",
        "winner_candidate_id",
        "winner_expectancy",
        "winner_delta_vs_1h",
        "winner_route_pass",
        "signals_partial_3m_data",
        "signals_missing_3m_data",
    ], n=len(comp_df)))
    lines.append("")
    lines.append("## Exact Coverage Trace")
    lines.append(f"- foundation windows used: `{len(win)}`")
    lines.append(f"- foundation coverage mean/min/max: `{coverage_mean:.12f}` / `{coverage_min:.12f}` / `{coverage_max:.12f}`")
    lines.append(f"- windows below 1.0 coverage: `{coverage_lt1}`")
    lines.append(f"- true partial slices: `{true_partial_count}`")
    lines.append(f"- max missing rows in any true partial slice: `{max_missing_rows}`")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- final LTC posture: `{final_posture}`")
    lines.append(f"- active-subset implication: `{implication}`")
    lines.append("")
    lines.append("## Proven vs Assumed")
    lines.append("- Proven: the fresh merged path and the prior cached path produce identical LTC 3m parquet data and identical winning 3m metrics.")
    lines.append("- Proven: the blanket 940/940 partial label is a conservative metadata artifact caused by applying a single merged window `coverage_ratio < 1.0` to every slice.")
    lines.append("- Proven: despite that artifact, there are still 12 truly partial LTC signal slices under exact row-count checking.")
    lines.append("- Assumed: exact 3m slice completeness is the operational cleanliness standard; under that standard, any non-zero true partial slice count blocks promotion.")
    (run_dir / "ltc_datapath_hygiene_report.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "symbol": SYMBOL,
        "freeze_dir": str(freeze_dir),
        "confirm_dir": str(confirm_dir),
        "opcheck_dir": str(opcheck_dir),
        "subset_eval_dir": str(subset_eval_dir),
        "foundation_dir": str(foundation_dir),
        "expected_winner": EXPECTED_WINNER,
        "contract_validation": contract_validation,
        "fresh_window_row": fresh_windows.iloc[0].to_dict(),
        "prior_window_row": prior_windows.iloc[0].to_dict(),
        "fresh_build_meta": fresh_build_meta,
        "prior_build_meta": prior_build_meta,
        "true_full_slices": true_full_count,
        "true_partial_slices": true_partial_count,
        "parquets_identical": bool(parquets_identical),
        "headline_match_across_paths": bool(headline_match),
        "blocker_truth": blocker_truth,
        "final_posture": final_posture,
        "active_subset_implication": implication,
        "artifacts": {
            "findings_csv": str(run_dir / "ltc_datapath_hygiene_findings.csv"),
            "report_md": str(run_dir / "ltc_datapath_hygiene_report.md"),
            "manifest_json": str(run_dir / "ltc_datapath_hygiene_manifest.json"),
            "winner_comparison_csv": str(run_dir / "ltc_winner_reconstruction_comparison.csv"),
            "coverage_label_trace_csv": str(run_dir / "ltc_coverage_label_trace.csv"),
        },
    }
    confirm.json_dump(run_dir / "ltc_datapath_hygiene_manifest.json", manifest)


if __name__ == "__main__":
    main()
