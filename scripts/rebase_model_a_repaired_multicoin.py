#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import rebase_model_a_repaired_priority as repaired
from scripts import phase_v_multicoin_model_a_audit as phase_v
RUN_PREFIX = "REPAIRED_MULTICOIN_MODELA_AUDIT"
UNIVERSE = [
    "SOLUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "CRVUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "AXSUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "OGUSDT",
    "PAXGUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "ZECUSDT",
]
CANONICAL_1H_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650"
).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_canonical_row_from_reference(symbol_rows: pd.DataFrame) -> pd.Series:
    ref = symbol_rows[symbol_rows["candidate_id"].astype(str) == "M0_1H_REFERENCE"].iloc[0]
    return pd.Series(
        {
            "symbol": str(ref["symbol"]).upper(),
            "after_expectancy_net": float(ref["exec_expectancy_net"]),
            "after_cvar_5": float(ref["reference_cvar_5"]),
            "after_maxdd": float(ref["reference_max_drawdown"]),
            "source": "runtime_repaired_contract",
        }
    )


def build_data_blocked_rows(
    *,
    symbol: str,
    qrow: Dict[str, Any],
    rrow: Dict[str, Any],
    reason: str,
) -> Dict[str, Dict[str, Any]]:
    base = {
        "symbol": symbol,
        "bucket_1h": str(rrow.get("bucket_1h", "")),
        "foundation_integrity_status": str(qrow.get("integrity_status", rrow.get("integrity_status", "DATA_BLOCKED"))),
        "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
        "foundation_signals_covered": int(pd.to_numeric(pd.Series([qrow.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
        "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([qrow.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
    }
    best = {
        **base,
        "classification": "DATA_BLOCKED",
        "classification_reason": reason,
        "reference_exec_expectancy_net": float("nan"),
        "reference_entries_valid": 0,
        "reference_entry_rate": float("nan"),
        "reference_taker_share": float("nan"),
        "reference_median_fill_delay_min": float("nan"),
        "reference_p95_fill_delay_min": float("nan"),
        "reference_cvar_5": float("nan"),
        "reference_max_drawdown": float("nan"),
        "best_candidate_id": "",
        "best_candidate_label": "",
        "best_valid_for_ranking": 0,
        "best_exec_expectancy_net": float("nan"),
        "delta_expectancy_vs_repaired_1h": float("nan"),
        "cvar_improve_ratio": float("nan"),
        "maxdd_improve_ratio": float("nan"),
        "best_entries_valid": 0,
        "best_entry_rate": float("nan"),
        "best_taker_share": float("nan"),
        "best_median_fill_delay_min": float("nan"),
        "best_p95_fill_delay_min": float("nan"),
        "best_route_pass": 0,
        "best_route_pass_rate": float("nan"),
        "best_min_subperiod_delta": float("nan"),
        "best_invalid_reason": "",
        "canonical_repaired_1h_expectancy_net": float("nan"),
        "canonical_repaired_1h_cvar_5": float("nan"),
        "canonical_repaired_1h_max_drawdown": float("nan"),
        "reference_expectancy_vs_canonical_abs_diff": float("nan"),
        "canonical_repaired_1h_source": "unavailable",
    }
    cls = {
        "symbol": symbol,
        "bucket_1h": str(rrow.get("bucket_1h", "")),
        "classification": "DATA_BLOCKED",
        "classification_reason": reason,
        "best_candidate_id": "",
        "best_valid_for_ranking": 0,
        "best_delta_expectancy_vs_repaired_1h": float("nan"),
        "best_cvar_improve_ratio": float("nan"),
        "best_maxdd_improve_ratio": float("nan"),
        "best_entry_rate": float("nan"),
        "best_entries_valid": 0,
        "best_taker_share": float("nan"),
        "best_median_fill_delay_min": float("nan"),
        "best_p95_fill_delay_min": float("nan"),
        "foundation_integrity_status": str(qrow.get("integrity_status", rrow.get("integrity_status", "DATA_BLOCKED"))),
        "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
        "foundation_signals_covered": int(pd.to_numeric(pd.Series([qrow.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
        "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([qrow.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
    }
    return {"best": best, "class": cls}


def classify_universe(class_df: pd.DataFrame) -> str:
    go_mask = class_df["classification"].astype(str).isin(["MODEL_A_STRONG_GO_REPAIRED", "MODEL_A_WEAK_GO_REPAIRED"])
    strong_mask = class_df["classification"].astype(str) == "MODEL_A_STRONG_GO_REPAIRED"
    blocked_mask = class_df["classification"].astype(str) == "DATA_BLOCKED"
    go_count = int(go_mask.sum())
    strong_count = int(strong_mask.sum())
    blocked_count = int(blocked_mask.sum())
    total = int(len(class_df))
    if blocked_count > 0 and blocked_count == total:
        return "NEED_DATA_REPAIR_BEFORE_DECISION"
    if go_count == 0:
        return "MODEL_A_TOO_NARROW_FOR_UNIVERSAL_PIPELINE"
    if strong_count >= 3 and go_count >= max(3, total // 3):
        return "MODEL_A_MULTICOIN_SURVIVES_REPAIR"
    return "MODEL_A_COIN_SELECTIVE_ONLY"


def write_universe_report(
    *,
    run_dir: Path,
    foundation_dir: Path,
    canonical_dir: Path,
    best_df: pd.DataFrame,
    class_df: pd.DataFrame,
    universe_status: str,
) -> None:
    lines = [
        "# Repaired Multicoin Model A Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Foundation source: `{foundation_dir}`",
        f"- Canonical repaired 1h baseline dir: `{canonical_dir}`",
        f"- Universe size: `{len(UNIVERSE)}`",
        "- Contract: `1h signals`, `1h exits`, `3m entry only`, `no same-parent-bar exit after fill`",
        "",
        "## Per-Coin Classification",
        "",
        repaired.markdown_table(
            class_df,
            [
                "symbol",
                "classification",
                "best_candidate_id",
                "best_delta_expectancy_vs_repaired_1h",
                "best_cvar_improve_ratio",
                "best_maxdd_improve_ratio",
                "best_valid_for_ranking",
                "classification_reason",
            ],
            n=40,
        ),
        "",
        "## Reference Vs Best",
        "",
        repaired.markdown_table(
            best_df,
            [
                "symbol",
                "classification",
                "reference_exec_expectancy_net",
                "best_exec_expectancy_net",
                "delta_expectancy_vs_repaired_1h",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "best_valid_for_ranking",
            ],
            n=40,
        ),
        "",
        "## Universe Status",
        "",
        f"- classification: `{universe_status}`",
        f"- strong_go_count: `{int((class_df['classification'] == 'MODEL_A_STRONG_GO_REPAIRED').sum())}`",
        f"- weak_go_count: `{int((class_df['classification'] == 'MODEL_A_WEAK_GO_REPAIRED').sum())}`",
        f"- no_go_count: `{int((class_df['classification'] == 'MODEL_A_NO_GO_REPAIRED').sum())}`",
        f"- data_blocked_count: `{int((class_df['classification'] == 'DATA_BLOCKED').sum())}`",
    ]
    repaired.write_text(run_dir / "repaired_multicoin_modelA_report.md", "\n".join(lines))
    repaired.write_text(
        run_dir / "repaired_multicoin_universe_status.md",
        "\n".join(
            [
                "# Repaired Multicoin Universe Status",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- classification: `{universe_status}`",
                "",
                repaired.markdown_table(
                    class_df,
                    [
                        "symbol",
                        "classification",
                        "best_candidate_id",
                        "best_delta_expectancy_vs_repaired_1h",
                        "best_valid_for_ranking",
                        "classification_reason",
                    ],
                    n=40,
                ),
            ]
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Repaired multicoin Model A audit using the repaired 1h contract as baseline")
    ap.add_argument("--foundation-dir", default="", help="Path to UNIVERSAL_DATA_FOUNDATION_* run dir; defaults to latest completed run.")
    ap.add_argument("--canonical-1h-dir", default=str(CANONICAL_1H_DIR_DEFAULT))
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    foundation_dir = Path(args_cli.foundation_dir).resolve() if str(args_cli.foundation_dir).strip() else phase_v.find_latest_foundation_dir()
    canonical_dir = Path(args_cli.canonical_1h_dir).resolve()
    canonical_summary = repaired.load_canonical_summary(canonical_dir)
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
    all_route_rows: List[pd.DataFrame] = []
    best_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    invalid_hist: Dict[str, int] = {}
    readiness_rows: List[Dict[str, Any]] = []
    symbol_meta: Dict[str, Any] = {}

    for symbol in UNIVERSE:
        qrow = qual_map.get(symbol, {})
        rrow = ready_map.get(symbol, {})
        sig_df = foundation_state.signal_timeline[foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol].copy()
        win_df = foundation_state.download_manifest[foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol].copy()

        readiness = "READY"
        readiness_reason = ""
        if sig_df.empty:
            readiness = "DATA_BLOCKED"
            readiness_reason = "no_signal_rows"
        elif win_df.empty:
            readiness = "DATA_BLOCKED"
            readiness_reason = "no_3m_windows"
        elif str(qrow.get("integrity_status", rrow.get("integrity_status", ""))).upper() == "PARTIAL":
            readiness = "PARTIAL"

        readiness_rows.append(
            {
                "symbol": symbol,
                "bucket_1h": str(rrow.get("bucket_1h", "")),
                "readiness": readiness,
                "readiness_reason": readiness_reason,
                "signals_total": int(len(sig_df)),
                "windows_total": int(len(win_df)),
                "windows_ready": int(pd.to_numeric(pd.Series([qrow.get("windows_ready", rrow.get("windows_ready", 0))]), errors="coerce").fillna(0).iloc[0]),
                "windows_partial": int(pd.to_numeric(pd.Series([qrow.get("windows_partial", rrow.get("windows_partial", 0))]), errors="coerce").fillna(0).iloc[0]),
                "windows_blocked": int(pd.to_numeric(pd.Series([qrow.get("windows_blocked", rrow.get("windows_blocked", 0))]), errors="coerce").fillna(0).iloc[0]),
                "missing_window_rate": float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
                "frozen_repaired_baseline_available": int(symbol in canonical_map),
            }
        )

        if readiness == "DATA_BLOCKED":
            blocked = build_data_blocked_rows(symbol=symbol, qrow=qrow, rrow=rrow, reason=readiness_reason)
            best_rows.append(blocked["best"])
            class_rows.append(blocked["class"])
            symbol_meta[symbol] = {
                "readiness": readiness,
                "readiness_reason": readiness_reason,
                "canonical_repaired_baseline_source": "unavailable",
            }
            continue

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
        symbol_rows["delta_expectancy_vs_repaired_1h"] = symbol_rows["delta_expectancy_vs_1h_reference"]
        all_results.append(symbol_rows)
        if isinstance(eval_pack["route_df"], pd.DataFrame) and not eval_pack["route_df"].empty:
            route_df = eval_pack["route_df"].copy()
            route_df["delta_expectancy_vs_repaired_1h"] = route_df["delta_expectancy_vs_1h_reference"]
            all_route_rows.append(route_df)

        for row in symbol_rows[symbol_rows["candidate_id"].astype(str) != "M0_1H_REFERENCE"].itertuples(index=False):
            for part in [x for x in str(getattr(row, "invalid_reason", "")).split("|") if x]:
                invalid_hist[part] = int(invalid_hist.get(part, 0)) + 1

        best_row = phase_v.choose_best_candidate(symbol_rows)
        classification, classification_reason = phase_v.classify_symbol(best_row=best_row, foundation_quality=qrow, exec_args=exec_args)
        classification = {
            "MODEL_A_STRONG_GO": "MODEL_A_STRONG_GO_REPAIRED",
            "MODEL_A_WEAK_GO": "MODEL_A_WEAK_GO_REPAIRED",
            "MODEL_A_NO_GO": "MODEL_A_NO_GO_REPAIRED",
            "DATA_BLOCKED": "DATA_BLOCKED",
        }.get(str(classification), str(classification))

        canonical_row = canonical_map.get(symbol)
        canonical_source = "frozen_repaired_summary" if canonical_row is not None else "runtime_repaired_contract"
        if canonical_row is None:
            canonical_row = make_canonical_row_from_reference(symbol_rows)

        best_ref = repaired.build_reference_vs_best_row_repaired(
            symbol=symbol,
            symbol_rows=symbol_rows,
            best_row=best_row,
            foundation_quality=qrow,
            classification=classification,
            classification_reason=classification_reason,
            canonical_row=canonical_row,
        )
        best_ref["canonical_repaired_1h_source"] = canonical_source
        best_rows.append(best_ref)

        class_rows.append(
            repaired.build_class_row_repaired(
                symbol=symbol,
                best_row=best_row,
                foundation_quality=qrow,
                foundation_readiness=rrow,
                classification=classification,
                classification_reason=classification_reason,
            )
        )
        symbol_meta[symbol] = {
            "readiness": readiness,
            "readiness_reason": readiness_reason,
            "bundle_build": build_meta,
            "route_meta": eval_pack["route_meta"],
            "route_examples_count": int(len(eval_pack["route_examples_df"])),
            "route_feasibility_count": int(len(eval_pack["route_feas_df"])),
            "canonical_repaired_baseline_source": canonical_source,
        }

    results_df = pd.concat(all_results, ignore_index=True).sort_values(["symbol", "candidate_id"]).reset_index(drop=True) if all_results else pd.DataFrame()
    route_df = pd.concat(all_route_rows, ignore_index=True).sort_values(["symbol", "candidate_id", "route_id"]).reset_index(drop=True) if all_route_rows else pd.DataFrame()
    best_df = pd.DataFrame(best_rows).sort_values("symbol").reset_index(drop=True)
    class_df = pd.DataFrame(class_rows).sort_values("symbol").reset_index(drop=True)
    readiness_df = pd.DataFrame(readiness_rows).sort_values("symbol").reset_index(drop=True)

    results_df.to_csv(run_dir / "repaired_multicoin_modelA_results.csv", index=False)
    best_df.to_csv(run_dir / "repaired_multicoin_modelA_reference_vs_best.csv", index=False)
    class_df.to_csv(run_dir / "repaired_multicoin_modelA_coin_classification.csv", index=False)
    repaired.json_dump(run_dir / "repaired_multicoin_modelA_invalid_reason_histogram.json", dict(sorted(invalid_hist.items(), key=lambda kv: (-kv[1], kv[0]))))
    if not route_df.empty:
        route_df.to_csv(run_dir / "repaired_multicoin_modelA_route_checks.csv", index=False)
    readiness_df.to_csv(run_dir / "repaired_multicoin_modelA_readiness.csv", index=False)

    universe_status = classify_universe(class_df)
    write_universe_report(
        run_dir=run_dir,
        foundation_dir=foundation_dir,
        canonical_dir=canonical_dir,
        best_df=best_df,
        class_df=class_df,
        universe_status=universe_status,
    )

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "foundation_dir": str(foundation_dir),
        "canonical_repaired_1h_dir": str(canonical_dir),
        "canonical_repaired_1h_summary_csv": str(canonical_dir / "repaired_1h_reference_summary.csv"),
        "symbol_universe": list(UNIVERSE),
        "git_snapshot": repaired.modela.git_snapshot(),
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
        "readiness_counts": {
            "READY": int((readiness_df["readiness"] == "READY").sum()),
            "PARTIAL": int((readiness_df["readiness"] == "PARTIAL").sum()),
            "DATA_BLOCKED": int((readiness_df["readiness"] == "DATA_BLOCKED").sum()),
        },
        "classification_counts": {
            "MODEL_A_STRONG_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_STRONG_GO_REPAIRED").sum()),
            "MODEL_A_WEAK_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_WEAK_GO_REPAIRED").sum()),
            "MODEL_A_NO_GO_REPAIRED": int((class_df["classification"] == "MODEL_A_NO_GO_REPAIRED").sum()),
            "DATA_BLOCKED": int((class_df["classification"] == "DATA_BLOCKED").sum()),
        },
        "universe_status": universe_status,
        "outputs": {
            "repaired_multicoin_modelA_results_csv": str(run_dir / "repaired_multicoin_modelA_results.csv"),
            "repaired_multicoin_modelA_reference_vs_best_csv": str(run_dir / "repaired_multicoin_modelA_reference_vs_best.csv"),
            "repaired_multicoin_modelA_coin_classification_csv": str(run_dir / "repaired_multicoin_modelA_coin_classification.csv"),
            "repaired_multicoin_modelA_invalid_reason_histogram_json": str(run_dir / "repaired_multicoin_modelA_invalid_reason_histogram.json"),
            "repaired_multicoin_modelA_report_md": str(run_dir / "repaired_multicoin_modelA_report.md"),
            "repaired_multicoin_modelA_run_manifest_json": str(run_dir / "repaired_multicoin_modelA_run_manifest.json"),
            "repaired_multicoin_universe_status_md": str(run_dir / "repaired_multicoin_universe_status.md"),
        },
    }
    repaired.json_dump(run_dir / "repaired_multicoin_modelA_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
