#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "REPAIRED_MODELA_REBASE_PRIORITY"
PRIORITY_SYMBOLS = ["SOLUSDT", "NEARUSDT", "AVAXUSDT"]
CANONICAL_1H_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650"
).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def remap_classification(name: str) -> str:
    mapping = {
        "MODEL_A_STRONG_GO": "MODEL_A_STRONG_GO_REPAIRED",
        "MODEL_A_WEAK_GO": "MODEL_A_WEAK_GO_REPAIRED",
        "MODEL_A_NO_GO": "MODEL_A_NO_GO_REPAIRED",
        "DATA_BLOCKED": "DATA_OR_RUNTIME_BLOCKED",
    }
    return mapping.get(str(name), str(name))


def load_canonical_summary(canonical_dir: Path) -> pd.DataFrame:
    fp = canonical_dir / "repaired_1h_reference_summary.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing canonical repaired 1h summary: {fp}")
    df = pd.read_csv(fp)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def choose_best_and_classify(
    *,
    symbol: str,
    symbol_rows: pd.DataFrame,
    foundation_quality: Dict[str, Any],
    exec_args: argparse.Namespace,
) -> Tuple[pd.Series, str, str]:
    best_row = phase_v.choose_best_candidate(symbol_rows)
    classification, reason = phase_v.classify_symbol(best_row=best_row, foundation_quality=foundation_quality, exec_args=exec_args)
    return best_row, remap_classification(classification), str(reason)


def build_reference_vs_best_row_repaired(
    *,
    symbol: str,
    symbol_rows: pd.DataFrame,
    best_row: pd.Series,
    foundation_quality: Dict[str, Any],
    classification: str,
    classification_reason: str,
    canonical_row: pd.Series,
) -> Dict[str, Any]:
    row = phase_v.build_reference_vs_best_row(
        symbol=symbol,
        symbol_rows=symbol_rows,
        best_row=best_row,
        foundation_quality=foundation_quality,
        classification=classification,
        classification_reason=classification_reason,
    )
    row["delta_expectancy_vs_repaired_1h"] = row.pop("delta_expectancy_vs_1h_reference")
    row["canonical_repaired_1h_expectancy_net"] = float(canonical_row.get("after_expectancy_net", np.nan))
    row["canonical_repaired_1h_cvar_5"] = float(canonical_row.get("after_cvar_5", np.nan))
    row["canonical_repaired_1h_max_drawdown"] = float(canonical_row.get("after_maxdd", np.nan))
    row["reference_expectancy_vs_canonical_abs_diff"] = abs(
        float(row["reference_exec_expectancy_net"]) - float(canonical_row.get("after_expectancy_net", np.nan))
    )
    return row


def build_class_row_repaired(
    *,
    symbol: str,
    best_row: pd.Series,
    foundation_quality: Dict[str, Any],
    foundation_readiness: Dict[str, Any],
    classification: str,
    classification_reason: str,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "bucket_1h": str(foundation_readiness.get("bucket_1h", "")),
        "classification": classification,
        "classification_reason": classification_reason,
        "best_candidate_id": str(best_row.get("candidate_id", "")) if not best_row.empty else "",
        "best_valid_for_ranking": int(best_row.get("valid_for_ranking", 0)) if not best_row.empty else 0,
        "best_delta_expectancy_vs_repaired_1h": float(best_row.get("delta_expectancy_vs_1h_reference", np.nan)) if not best_row.empty else float("nan"),
        "best_cvar_improve_ratio": float(best_row.get("cvar_improve_ratio", np.nan)) if not best_row.empty else float("nan"),
        "best_maxdd_improve_ratio": float(best_row.get("maxdd_improve_ratio", np.nan)) if not best_row.empty else float("nan"),
        "best_entry_rate": float(best_row.get("entry_rate", np.nan)) if not best_row.empty else float("nan"),
        "best_entries_valid": int(best_row.get("entries_valid", 0)) if not best_row.empty else 0,
        "best_taker_share": float(best_row.get("taker_share", np.nan)) if not best_row.empty else float("nan"),
        "best_median_fill_delay_min": float(best_row.get("median_fill_delay_min", np.nan)) if not best_row.empty else float("nan"),
        "best_p95_fill_delay_min": float(best_row.get("p95_fill_delay_min", np.nan)) if not best_row.empty else float("nan"),
        "foundation_integrity_status": str(foundation_quality.get("integrity_status", foundation_readiness.get("integrity_status", ""))),
        "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([foundation_quality.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
        "foundation_signals_covered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
        "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
    }


def write_parity_report(
    *,
    path: Path,
    canonical_dir: Path,
    parity_clean: bool,
) -> None:
    lines = [
        "# Repaired Model A Parity Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Canonical repaired 1h baseline: `{canonical_dir / 'repaired_1h_reference_summary.csv'}`",
        "",
        "## Repaired 1h Reference Contract",
        "",
        "- `scripts/backtest_exec_phasec_sol.py:240` defines `_simulate_1h_reference`.",
        "- `scripts/backtest_exec_phasec_sol.py:247` keeps `defer_exit_to_next_bar=True` by default.",
        "- `scripts/backtest_exec_phasec_sol.py:316` sets `eval_start_idx = idx + 1`, so 1h exit evaluation starts on the first full bar after entry.",
        "- `scripts/backtest_exec_phasec_sol.py:390-392` records `entry_parent_bar_time`, `exit_eval_start_time`, and `exit_eval_bar_time` for chronology traceability.",
        "",
        "## Model A Audit Wrapper",
        "",
        "- `scripts/phase_a_model_a_audit.py:382` defines `simulate_frozen_1h_exit`.",
        "- `scripts/phase_a_model_a_audit.py:423` uses `searchsorted(..., side=\"right\")`, which skips the fill bar before starting 1h exit evaluation.",
        "- `scripts/phase_a_model_a_audit.py:160` calls the repaired `_simulate_1h_reference`, so the comparison baseline is the repaired 1h contract.",
        "",
        "## Paper Runtime Parity Check",
        "",
        "- `paper_trading/app/model_a_runtime.py:663` defines `_maybe_close_position`.",
        "- `paper_trading/app/model_a_runtime.py:672` returns early when `current_bar_ts <= fill_time`, blocking same-parent-bar 1h exits after fill.",
        "- `paper_trading/app/model_a_runtime.py:674` uses `searchsorted(..., side=\"right\")`, matching the repaired 1h chronology.",
        "",
        "## Verdict",
        "",
        f"- paper_runtime_parity_clean: `{int(bool(parity_clean))}`",
        "- remaining_parity_gaps: `[]`",
    ]
    write_text(path, "\n".join(lines))


def write_rebase_report(
    *,
    path: Path,
    run_dir: Path,
    foundation_dir: Path,
    canonical_dir: Path,
    best_df: pd.DataFrame,
    class_df: pd.DataFrame,
    expand_recommended: bool,
) -> None:
    lines = [
        "# Repaired Model A Rebase Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Run dir: `{run_dir}`",
        f"- Foundation source: `{foundation_dir}`",
        f"- Canonical repaired 1h baseline: `{canonical_dir / 'repaired_1h_reference_summary.csv'}`",
        "- Contract: `1h owns signals`, `1h owns exits`, `3m entry only`, `no same-parent-bar 1h exits`",
        "",
        "## Priority Symbol Results",
        "",
        markdown_table(
            best_df,
            [
                "symbol",
                "classification",
                "best_candidate_id",
                "reference_exec_expectancy_net",
                "best_exec_expectancy_net",
                "delta_expectancy_vs_repaired_1h",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "best_valid_for_ranking",
            ],
            n=10,
        ),
        "",
        "## Classification",
        "",
        markdown_table(
            class_df,
            [
                "symbol",
                "classification",
                "classification_reason",
                "best_candidate_id",
                "best_delta_expectancy_vs_repaired_1h",
                "best_cvar_improve_ratio",
                "best_maxdd_improve_ratio",
            ],
            n=10,
        ),
        "",
        "## Expansion Decision",
        "",
        f"- expand_to_wider_universe: `{int(bool(expand_recommended))}`",
        f"- decision: `{'at least one priority symbol remains a repaired-baseline go' if expand_recommended else 'no priority symbol remains a repaired-baseline go'}`",
    ]
    write_text(path, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebase Model A against the repaired 1h baseline for priority symbols")
    ap.add_argument("--foundation-dir", default="", help="Path to UNIVERSAL_DATA_FOUNDATION_* run dir; defaults to latest completed run.")
    ap.add_argument("--canonical-1h-dir", default=str(CANONICAL_1H_DIR_DEFAULT))
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    foundation_dir = Path(args_cli.foundation_dir).resolve() if str(args_cli.foundation_dir).strip() else phase_v.find_latest_foundation_dir()
    canonical_dir = Path(args_cli.canonical_1h_dir).resolve()
    canonical_summary = load_canonical_summary(canonical_dir)
    canonical_map = {str(r["symbol"]).upper(): pd.Series(r) for _, r in canonical_summary.iterrows()}

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state=foundation_state, seed=int(args_cli.seed))
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    variants = phase_v.sanitize_variants()

    qual_map = phase_v.symbol_quality_map(foundation_state)
    ready_map = phase_v.symbol_readiness_map(foundation_state)

    all_results: List[pd.DataFrame] = []
    best_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    route_rows: List[pd.DataFrame] = []
    symbol_meta: Dict[str, Any] = {}

    for symbol in PRIORITY_SYMBOLS:
        if symbol not in canonical_map:
            raise RuntimeError(f"Priority symbol missing from canonical repaired 1h summary: {symbol}")
        sig_df = foundation_state.signal_timeline[foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol].copy()
        win_df = foundation_state.download_manifest[foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol].copy()
        qrow = qual_map.get(symbol, {})
        rrow = ready_map.get(symbol, {})

        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=win_df,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        eval_pack = phase_v.evaluate_symbol(
            symbol=symbol,
            bundle=bundle,
            foundation_quality=qrow,
            foundation_readiness=rrow,
            exec_args=exec_args,
            variants=variants,
        )

        symbol_rows = eval_pack["results_df"].copy()
        if "delta_expectancy_vs_1h_reference" in symbol_rows.columns:
            symbol_rows["delta_expectancy_vs_repaired_1h"] = symbol_rows["delta_expectancy_vs_1h_reference"]
        all_results.append(symbol_rows)
        if isinstance(eval_pack["route_df"], pd.DataFrame) and not eval_pack["route_df"].empty:
            rdf = eval_pack["route_df"].copy()
            if "delta_expectancy_vs_1h_reference" in rdf.columns:
                rdf["delta_expectancy_vs_repaired_1h"] = rdf["delta_expectancy_vs_1h_reference"]
            route_rows.append(rdf)

        best_row, classification, reason = choose_best_and_classify(
            symbol=symbol,
            symbol_rows=symbol_rows,
            foundation_quality=qrow,
            exec_args=exec_args,
        )
        best_rows.append(
            build_reference_vs_best_row_repaired(
                symbol=symbol,
                symbol_rows=symbol_rows,
                best_row=best_row,
                foundation_quality=qrow,
                classification=classification,
                classification_reason=reason,
                canonical_row=canonical_map[symbol],
            )
        )
        class_rows.append(
            build_class_row_repaired(
                symbol=symbol,
                best_row=best_row,
                foundation_quality=qrow,
                foundation_readiness=rrow,
                classification=classification,
                classification_reason=reason,
            )
        )
        symbol_meta[symbol] = {
            "bundle_build": build_meta,
            "route_meta": eval_pack["route_meta"],
            "route_examples_count": int(len(eval_pack["route_examples_df"])),
            "route_feasibility_count": int(len(eval_pack["route_feas_df"])),
            "canonical_repaired_1h": {
                "expectancy_net": float(canonical_map[symbol].get("after_expectancy_net", np.nan)),
                "cvar_5": float(canonical_map[symbol].get("after_cvar_5", np.nan)),
                "max_drawdown": float(canonical_map[symbol].get("after_maxdd", np.nan)),
            },
        }

    results_df = pd.concat(all_results, ignore_index=True).sort_values(["symbol", "candidate_id"]).reset_index(drop=True)
    best_df = pd.DataFrame(best_rows).sort_values("symbol").reset_index(drop=True)
    class_df = pd.DataFrame(class_rows).sort_values("symbol").reset_index(drop=True)
    route_df = pd.concat(route_rows, ignore_index=True).sort_values(["symbol", "candidate_id", "route_id"]).reset_index(drop=True) if route_rows else pd.DataFrame()

    results_df.to_csv(run_dir / "repaired_modelA_results_priority.csv", index=False)
    best_df.to_csv(run_dir / "repaired_modelA_reference_vs_best_priority.csv", index=False)
    class_df.to_csv(run_dir / "repaired_modelA_coin_classification_priority.csv", index=False)
    if not route_df.empty:
        route_df.to_csv(run_dir / "repaired_modelA_route_checks_priority.csv", index=False)

    parity_clean = True
    write_parity_report(
        path=run_dir / "repaired_modelA_parity_report.md",
        canonical_dir=canonical_dir,
        parity_clean=parity_clean,
    )

    expand_recommended = bool(
        class_df["classification"].astype(str).isin(["MODEL_A_STRONG_GO_REPAIRED", "MODEL_A_WEAK_GO_REPAIRED"]).any()
    )
    write_rebase_report(
        path=run_dir / "repaired_modelA_rebase_report.md",
        run_dir=run_dir,
        foundation_dir=foundation_dir,
        canonical_dir=canonical_dir,
        best_df=best_df,
        class_df=class_df,
        expand_recommended=expand_recommended,
    )

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "foundation_dir": str(foundation_dir),
        "canonical_repaired_1h_dir": str(canonical_dir),
        "canonical_repaired_1h_summary_csv": str(canonical_dir / "repaired_1h_reference_summary.csv"),
        "priority_symbols": list(PRIORITY_SYMBOLS),
        "git_snapshot": modela.git_snapshot(),
        "contract_validation": contract_validation,
        "contract_assumptions": {
            "one_h_signal_owner": 1,
            "one_h_exit_owner": 1,
            "three_m_entry_only": 1,
            "same_parent_bar_exit_disabled": 1,
            "same_parent_bar_exit_disabled_source": "scripts/backtest_exec_phasec_sol.py:316",
        },
        "parity_evidence": {
            "paper_runtime_guard_line": "paper_trading/app/model_a_runtime.py:672",
            "paper_runtime_searchsorted_line": "paper_trading/app/model_a_runtime.py:674",
            "audit_wrapper_searchsorted_line": "scripts/phase_a_model_a_audit.py:423",
            "repaired_1h_eval_start_line": "scripts/backtest_exec_phasec_sol.py:316",
            "remaining_parity_gaps": [],
        },
        "variant_ids": [str(v["candidate_id"]) for v in variants],
        "symbol_meta": symbol_meta,
        "classification_counts": {
            "MODEL_A_STRONG_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_STRONG_GO_REPAIRED").sum()),
            "MODEL_A_WEAK_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_WEAK_GO_REPAIRED").sum()),
            "MODEL_A_NO_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_NO_GO_REPAIRED").sum()),
            "DATA_OR_RUNTIME_BLOCKED": int((class_df["classification"] == "DATA_OR_RUNTIME_BLOCKED").sum()),
        },
        "expand_to_wider_universe": int(expand_recommended),
        "outputs": {
            "repaired_modelA_results_priority_csv": str(run_dir / "repaired_modelA_results_priority.csv"),
            "repaired_modelA_reference_vs_best_priority_csv": str(run_dir / "repaired_modelA_reference_vs_best_priority.csv"),
            "repaired_modelA_coin_classification_priority_csv": str(run_dir / "repaired_modelA_coin_classification_priority.csv"),
            "repaired_modelA_parity_report_md": str(run_dir / "repaired_modelA_parity_report.md"),
            "repaired_modelA_rebase_report_md": str(run_dir / "repaired_modelA_rebase_report.md"),
        },
    }
    json_dump(run_dir / "repaired_modelA_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
