#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
    if df.empty:
        return "| (empty) |\n| --- |\n| (no rows) |"
    use = [c for c in cols if c in df.columns]
    d = df[use].head(n).copy()
    header = "| " + " | ".join(use) + " |"
    sep = "| " + " | ".join(["---"] * len(use)) + " |"
    rows = [header, sep]
    for _, r in d.iterrows():
        vals: List[str] = []
        for c in use:
            v = r[c]
            if isinstance(v, float):
                vals.append("" if not math.isfinite(v) else f"{v:.10g}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Freeze repaired long universe from fresh repaired 1h rerun outputs")
    ap.add_argument("--rerun-dir", default="")
    ap.add_argument("--legacy-best-by-symbol", default="reports/params_scan/20260220_044949/best_by_symbol.csv")
    ap.add_argument("--outdir", default="reports/execution_layer")
    return ap


def discover_rerun_dir(arg_value: str) -> Path:
    if arg_value:
        p = Path(arg_value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()
    root = PROJECT_ROOT / "reports" / "execution_layer"
    cands = sorted([p for p in root.glob("REPAIRED_1H_GA_RERUN_*") if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        required = [
            cand / "repaired_1h_ga_results.csv",
            cand / "repaired_1h_frontier.csv",
            cand / "repaired_1h_shortlist.csv",
            cand / "repaired_1h_universe_candidates.csv",
            cand / "rerun_manifest.json",
            cand / "repaired_1h_ga_rerun_report.md",
        ]
        if all(x.exists() for x in required):
            return cand.resolve()
    raise FileNotFoundError("missing repaired rerun dir with required artifacts")


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required column(s): {missing}")


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        rerun_dir = discover_rerun_dir(args.rerun_dir)
    except Exception as exc:
        print(f"missing artifact / missing required column / ambiguity in candidate selection: {exc}")
        print("/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_1h_ga_rerun.py")
        print("/root/analysis/0.87/reports/execution_layer")
        return

    legacy_fp = Path(args.legacy_best_by_symbol)
    if not legacy_fp.is_absolute():
        legacy_fp = (PROJECT_ROOT / legacy_fp).resolve()

    ga_results_fp = rerun_dir / "repaired_1h_ga_results.csv"
    frontier_fp = rerun_dir / "repaired_1h_frontier.csv"
    shortlist_fp = rerun_dir / "repaired_1h_shortlist.csv"
    universe_candidates_fp = rerun_dir / "repaired_1h_universe_candidates.csv"
    manifest_fp = rerun_dir / "rerun_manifest.json"
    report_fp = rerun_dir / "repaired_1h_ga_rerun_report.md"
    required_paths = [ga_results_fp, frontier_fp, shortlist_fp, universe_candidates_fp, manifest_fp, report_fp, legacy_fp]
    missing_paths = [p for p in required_paths if not p.exists()]
    if missing_paths:
        print("missing artifact / missing required column / ambiguity in candidate selection: " + ", ".join(str(p) for p in missing_paths))
        print("/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_1h_ga_rerun.py")
        print(str(rerun_dir))
        return

    ga_results = pd.read_csv(ga_results_fp)
    universe_candidates = pd.read_csv(universe_candidates_fp)
    legacy = pd.read_csv(legacy_fp)
    rerun_manifest = load_json(manifest_fp)

    require_columns(
        ga_results,
        [
            "candidate_id",
            "symbol",
            "side",
            "param_hash",
            "valid_for_ranking",
            "repaired_score",
            "repaired_expectancy_net",
            "repaired_max_dd_pct",
            "repaired_cvar_5",
            "repaired_profit_factor",
            "repaired_trades",
            "repaired_pass",
            "params_json",
            "params_file_source",
        ],
        "repaired_1h_ga_results.csv",
    )
    require_columns(
        universe_candidates,
        [
            "candidate_id",
            "symbol",
            "param_hash",
            "valid_for_ranking",
            "repaired_score",
            "repaired_expectancy_net",
            "repaired_max_dd_pct",
            "repaired_cvar_5",
            "repaired_profit_factor",
            "repaired_trades",
            "params_json",
            "params_file_source",
        ],
        "repaired_1h_universe_candidates.csv",
    )
    require_columns(legacy, ["symbol", "side", "params_file", "score", "pass"], str(legacy_fp))

    long_valid = ga_results[
        (ga_results["side"].astype(str).str.lower() == "long")
        & (pd.to_numeric(ga_results["valid_for_ranking"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(ga_results["repaired_pass"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if long_valid.empty:
        print("missing artifact / missing required column / ambiguity in candidate selection: no passing long repaired candidates found")
        print("/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_1h_ga_rerun.py")
        print(str(rerun_dir))
        return

    sort_cols = ["symbol", "repaired_score", "repaired_expectancy_net", "repaired_max_dd_pct", "candidate_id", "param_hash"]
    ascending = [True, False, False, True, True, True]
    selected = (
        long_valid.sort_values(sort_cols, ascending=ascending)
        .groupby("symbol", as_index=False, sort=False)
        .head(1)
        .sort_values(["repaired_score", "repaired_expectancy_net", "repaired_max_dd_pct", "candidate_id", "param_hash"], ascending=[False, False, True, True, True])
        .reset_index(drop=True)
    )
    selected["repaired_rank"] = selected.index + 1

    # Verification against the rerun's own per-symbol candidate table.
    check = selected[["symbol", "candidate_id"]].merge(
        universe_candidates[["symbol", "candidate_id"]].rename(columns={"candidate_id": "candidate_id_universe_candidates"}),
        on="symbol",
        how="left",
    )
    check["match"] = check["candidate_id"] == check["candidate_id_universe_candidates"]
    if not bool(check["match"].fillna(False).all()):
        mism = check[check["match"] != True]
        print("missing artifact / missing required column / ambiguity in candidate selection: rebuilt selection does not match repaired_1h_universe_candidates.csv")
        print("/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_1h_universe_freeze.py")
        print(str(universe_candidates_fp))
        return

    legacy_long = legacy[legacy["side"].astype(str).str.lower() == "long"].copy()
    legacy_long = legacy_long.assign(
        legacy_pass=legacy_long["pass"].map(lambda x: int(str(x).strip().lower() in {"1", "true", "yes"})),
        legacy_score=pd.to_numeric(legacy_long["score"], errors="coerce"),
    ).sort_values(["legacy_pass", "legacy_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    legacy_long["legacy_rank"] = legacy_long.index + 1

    legacy_by_symbol = legacy_long.set_index("symbol")
    selected["legacy_pass"] = selected["symbol"].map(legacy_by_symbol["legacy_pass"]).fillna(0).astype(int)
    selected["legacy_score"] = pd.to_numeric(selected["symbol"].map(legacy_by_symbol["legacy_score"]), errors="coerce")
    selected["legacy_rank"] = pd.to_numeric(selected["symbol"].map(legacy_by_symbol["legacy_rank"]), errors="coerce")
    selected["legacy_params_file"] = selected["symbol"].map(legacy_by_symbol["params_file"])
    selected["score_delta_vs_legacy"] = pd.to_numeric(selected["repaired_score"], errors="coerce") - pd.to_numeric(selected["legacy_score"], errors="coerce")
    selected["rank_delta_vs_legacy"] = pd.to_numeric(selected["repaired_rank"], errors="coerce") - pd.to_numeric(selected["legacy_rank"], errors="coerce")

    # Build clean frozen best-by-symbol replacement.
    repaired_best = selected[
        [
            "symbol",
            "side",
            "candidate_id",
            "param_hash",
            "params_file_source",
            "params_json",
            "repaired_score",
            "repaired_expectancy_net",
            "repaired_cvar_5",
            "repaired_max_dd_pct",
            "repaired_profit_factor",
            "repaired_trades",
            "repaired_cagr_pct",
            "repaired_final_equity",
            "repaired_net_profit",
            "repaired_pass",
            "valid_for_ranking",
            "repaired_rank",
            "period_start",
            "period_end",
            "initial_equity",
        ]
    ].copy()
    repaired_best = repaired_best.rename(
        columns={
            "params_file_source": "params_source",
            "params_json": "params_payload_json",
            "repaired_score": "score",
            "repaired_expectancy_net": "expectancy_net",
            "repaired_cvar_5": "cvar_5",
            "repaired_max_dd_pct": "max_dd_pct",
            "repaired_profit_factor": "profit_factor",
            "repaired_trades": "trades",
            "repaired_cagr_pct": "cagr_pct",
            "repaired_final_equity": "final_equity",
            "repaired_net_profit": "net_profit",
            "repaired_pass": "pass",
            "repaired_rank": "rank",
        }
    )

    # Membership diff across the union of symbols.
    selected_by_symbol = selected.set_index("symbol")
    selected_pass_set = set(selected["symbol"].astype(str))
    legacy_pass_set = set(legacy_long.loc[legacy_long["legacy_pass"] == 1, "symbol"].astype(str))
    union_symbols = sorted(set(legacy_long["symbol"].astype(str)) | selected_pass_set)
    diff_rows: List[Dict[str, Any]] = []
    for sym in union_symbols:
        in_legacy = sym in legacy_by_symbol.index
        in_sel = sym in selected_by_symbol.index
        legacy_row = legacy_by_symbol.loc[sym] if in_legacy else None
        repaired_row = selected_by_symbol.loc[sym] if in_sel else None
        legacy_pass = int(legacy_row["legacy_pass"]) if in_legacy else 0
        repaired_pass = 1 if in_sel else 0
        if legacy_pass == 1 and repaired_pass == 1:
            action = "STAY_PASS"
        elif legacy_pass == 1 and repaired_pass == 0:
            action = "DROP_FROM_PASS"
        elif legacy_pass == 0 and repaired_pass == 1:
            action = "NEW_PASS"
        else:
            action = "STAY_FAIL"
        diff_rows.append(
            {
                "symbol": sym,
                "legacy_pass": legacy_pass,
                "repaired_pass": repaired_pass,
                "membership_action": action,
                "legacy_rank": int(legacy_row["legacy_rank"]) if in_legacy else "",
                "repaired_rank": int(repaired_row["repaired_rank"]) if in_sel else "",
                "legacy_score": float(legacy_row["legacy_score"]) if in_legacy else "",
                "repaired_score": float(repaired_row["repaired_score"]) if in_sel else "",
                "score_delta": float(repaired_row["repaired_score"] - legacy_row["legacy_score"]) if (in_legacy and in_sel) else "",
                "rank_delta": int(repaired_row["repaired_rank"] - legacy_row["legacy_rank"]) if (in_legacy and in_sel) else "",
                "legacy_params_file": str(legacy_row["params_file"]) if in_legacy else "",
                "repaired_candidate_id": str(repaired_row["candidate_id"]) if in_sel else "",
                "repaired_param_hash": str(repaired_row["param_hash"]) if in_sel else "",
            }
        )
    diff_df = pd.DataFrame(diff_rows)

    out_dir = (PROJECT_ROOT / args.outdir).resolve() / f"REPAIRED_1H_UNIVERSE_FREEZE_{utc_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = out_dir / "repaired_universe_selected_params"
    params_dir.mkdir(parents=True, exist_ok=True)

    # Freeze one params payload per selected symbol.
    frozen_param_files: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        sym = str(row["symbol"]).upper()
        payload = {
            "symbol": sym,
            "side": str(row["side"]),
            "candidate_id": str(row["candidate_id"]),
            "param_hash": str(row["param_hash"]),
            "params_source": str(row["params_file_source"]),
            "repaired_metrics": {
                "score": float(row["repaired_score"]),
                "expectancy_net": float(row["repaired_expectancy_net"]),
                "cvar_5": float(row["repaired_cvar_5"]),
                "max_dd_pct": float(row["repaired_max_dd_pct"]),
                "profit_factor": float(row["repaired_profit_factor"]),
                "trades": int(row["repaired_trades"]),
                "pass": int(row["repaired_pass"]),
                "valid_for_ranking": int(row["valid_for_ranking"]),
                "rank": int(row["repaired_rank"]),
            },
            "selection_rule": {
                "primary": "repaired_score desc",
                "secondary": "repaired_expectancy_net desc",
                "tertiary": "repaired_max_dd_pct asc",
                "tie_break_1": "candidate_id asc",
                "tie_break_2": "param_hash asc",
            },
            "params": json.loads(str(row["params_json"])),
        }
        fp = params_dir / f"{sym}_repaired_selected_params.json"
        json_dump(fp, payload)
        frozen_param_files.append({"symbol": sym, "path": str(fp)})

    repaired_best.to_csv(out_dir / "repaired_best_by_symbol.csv", index=False)
    diff_df.to_csv(out_dir / "repaired_universe_membership_diff_vs_legacy.csv", index=False)

    retained = sorted(selected_pass_set & legacy_pass_set)
    dropped = sorted(legacy_pass_set - selected_pass_set)
    new_symbols = sorted(selected_pass_set - legacy_pass_set)

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "source_of_truth": str(rerun_dir),
        "source_files": {
            "repaired_1h_ga_results": str(ga_results_fp),
            "repaired_1h_frontier": str(frontier_fp),
            "repaired_1h_shortlist": str(shortlist_fp),
            "repaired_1h_universe_candidates": str(universe_candidates_fp),
            "rerun_manifest": str(manifest_fp),
            "rerun_report": str(report_fp),
            "legacy_best_by_symbol_reference": str(legacy_fp),
        },
        "selection_rule": {
            "eligibility": [
                "side == long",
                "valid_for_ranking == 1",
                "repaired_pass == 1",
            ],
            "grouping": "exactly one canonical best candidate per symbol",
            "sort_order": [
                "repaired_score desc",
                "repaired_expectancy_net desc",
                "repaired_max_dd_pct asc",
                "candidate_id asc",
                "param_hash asc",
            ],
            "verification_against_repaired_1h_universe_candidates_csv": "exact match",
        },
        "summary": {
            "selected_symbol_count": int(len(repaired_best)),
            "retained_symbols": retained,
            "dropped_symbols": dropped,
            "new_symbols": new_symbols,
            "ready_for_downstream_use": True,
            "downstream_branch_note": "This frozen rebuilt universe replaces the old best_by_symbol-driven long set for the repaired branch.",
        },
        "rerun_recommendation": rerun_manifest.get("recommendation", ""),
        "frozen_param_files": frozen_param_files,
        "artifacts": {
            "repaired_best_by_symbol": str(out_dir / "repaired_best_by_symbol.csv"),
            "repaired_universe_membership_diff_vs_legacy": str(out_dir / "repaired_universe_membership_diff_vs_legacy.csv"),
            "repaired_universe_freeze_report": str(out_dir / "repaired_universe_freeze_report.md"),
            "repaired_universe_selected_params_dir": str(params_dir),
        },
    }
    json_dump(out_dir / "repaired_universe_freeze_manifest.json", manifest)

    report_lines: List[str] = []
    report_lines.append("# Repaired 1H Universe Freeze Report")
    report_lines.append("")
    report_lines.append(f"- Generated UTC: `{utc_now().isoformat()}`")
    report_lines.append(f"- Fresh repaired rerun source: `{rerun_dir}`")
    report_lines.append("- This rebuilt universe now replaces the old `best_by_symbol.csv`-driven long set for the repaired branch.")
    report_lines.append("- Downstream 3m execution work must use this frozen repaired universe, not the legacy universe.")
    report_lines.append("")
    report_lines.append("## Selection Rule")
    report_lines.append("- Eligibility: `side == long`, `valid_for_ranking == 1`, `repaired_pass == 1`")
    report_lines.append("- Exactly one canonical best candidate per symbol")
    report_lines.append("- Primary sort: `repaired_score desc`")
    report_lines.append("- Secondary sort: `repaired_expectancy_net desc`")
    report_lines.append("- Tertiary sort: `repaired_max_dd_pct asc`")
    report_lines.append("- Stable tie-breaks: `candidate_id asc`, then `param_hash asc`")
    report_lines.append("- Verification: reconstructed selection matches `repaired_1h_universe_candidates.csv` exactly")
    report_lines.append("")
    report_lines.append("## Frozen Repaired Universe Summary")
    report_lines.append(f"- Selected symbols: `{len(repaired_best)}`")
    report_lines.append(f"- Retained from legacy long pass set: `{len(retained)}`")
    report_lines.append(f"- New symbols: `{len(new_symbols)}`")
    report_lines.append(f"- Dropped symbols: `{len(dropped)}`")
    report_lines.append("")
    report_lines.append(markdown_table(repaired_best, ["rank", "symbol", "candidate_id", "score", "expectancy_net", "cvar_5", "max_dd_pct", "profit_factor", "trades", "pass"], n=len(repaired_best)))
    report_lines.append("")
    report_lines.append("## Legacy vs Repaired Membership Changes")
    report_lines.append(f"- Retained symbols: `{', '.join(retained) if retained else '(none)'}`")
    report_lines.append(f"- Dropped symbols: `{', '.join(dropped) if dropped else '(none)'}`")
    report_lines.append(f"- New symbols: `{', '.join(new_symbols) if new_symbols else '(none)'}`")
    report_lines.append("")
    report_lines.append(markdown_table(diff_df, ["symbol", "legacy_pass", "repaired_pass", "membership_action", "legacy_rank", "repaired_rank", "legacy_score", "repaired_score", "rank_delta"], n=len(diff_df)))
    report_lines.append("")
    report_lines.append("## Readiness Decision")
    report_lines.append("- Ready for downstream use: `YES`")
    report_lines.append("- No additional universe sanity check is required before downstream execution evaluation, because the rebuilt universe is sourced directly from the fresh repaired 1h rerun and frozen here.")
    (out_dir / "repaired_universe_freeze_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "freeze_dir": str(out_dir),
                "selected_symbols": int(len(repaired_best)),
                "new_symbols": new_symbols,
                "dropped_symbols": dropped,
                "ready_for_downstream_use": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
