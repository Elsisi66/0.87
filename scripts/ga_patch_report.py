#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _latest_ga_run(base_dir: Path) -> Path:
    cands = sorted([p for p in base_dir.glob("GA_EXEC_OPT_*") if p.is_dir() and (p / "best_genome.json").exists()], key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError(f"No GA_EXEC_OPT_* run directories under {base_dir}")
    return cands[-1].resolve()


def run(args: argparse.Namespace) -> Path:
    base_dir = _resolve_path(args.base_dir)
    source_run = _resolve_path(args.ga_run_dir) if str(args.ga_run_dir).strip() else _latest_ga_run(base_dir)
    out = _resolve_path(args.outdir) / f"GA_PATCH_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    best = json.loads((source_run / "best_genome.json").read_text(encoding="utf-8"))
    metrics: Dict[str, Any] = dict(best.get("metrics", {}))
    sym = pd.read_csv(source_run / "risk_rollup_by_symbol.csv") if (source_run / "risk_rollup_by_symbol.csv").exists() else pd.DataFrame()
    invalid_hist = json.loads((source_run / "invalid_reason_histogram.json").read_text(encoding="utf-8")) if (source_run / "invalid_reason_histogram.json").exists() else {}
    if not invalid_hist:
        c = Counter()
        gdf = pd.read_csv(source_run / "genomes.csv") if (source_run / "genomes.csv").exists() else pd.DataFrame()
        if "invalid_reason" in gdf.columns:
            for raw in gdf["invalid_reason"].fillna("").astype(str):
                for part in [x.strip() for x in raw.split("|") if x.strip()]:
                    c[part] += 1
        invalid_hist = dict(sorted(c.items()))

    required_finite = [
        "overall_exec_expectancy_net",
        "overall_exec_cvar_5",
        "overall_exec_max_drawdown",
        "overall_entry_rate",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_exec_p95_fill_delay_min",
    ]
    finite_pass = all(np.isfinite(float(metrics.get(k, np.nan))) for k in required_finite)
    valid_pass = int(metrics.get("valid_for_ranking", 0)) == 1
    no_trade_best = bool(float(metrics.get("overall_entries_valid", 0)) <= 0.0)
    entry_gate_overall_pass = bool(np.isfinite(float(metrics.get("overall_entry_rate", np.nan))) and float(metrics.get("overall_entry_rate", np.nan)) >= float(args.min_entry_rate_overall))
    per_symbol_entry_pass = True
    if "pass_entry_rate" in sym.columns:
        per_symbol_entry_pass = bool((pd.to_numeric(sym["pass_entry_rate"], errors="coerce").fillna(0).astype(int) == 1).all())

    smoke = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(source_run),
        "best_hash": best.get("best_hash", ""),
        "best_valid_for_ranking": int(valid_pass),
        "best_overall_entries_valid": int(metrics.get("overall_entries_valid", 0)),
        "best_overall_entry_rate": float(metrics.get("overall_entry_rate", np.nan)),
        "best_has_no_trade": int(no_trade_best),
        "finite_required_metrics_pass": int(finite_pass),
        "overall_entry_rate_gate_pass": int(entry_gate_overall_pass),
        "per_symbol_entry_rate_gate_pass": int(per_symbol_entry_pass),
        "assert_no_trade_cannot_be_best": int(valid_pass and (not no_trade_best)),
        "assertions_pass": int(valid_pass and finite_pass and entry_gate_overall_pass and per_symbol_entry_pass and (not no_trade_best)),
    }
    (out / "smoke_test_result.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    (out / "invalid_reason_histogram.json").write_text(json.dumps(invalid_hist, indent=2, sort_keys=True), encoding="utf-8")

    md: list[str] = []
    md.append("# GA Patch Report (Phase B)")
    md.append("")
    md.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"- Source run: `{source_run}`")
    md.append(f"- Best genome hash: `{best.get('best_hash', '')}`")
    md.append("")
    md.append("## Anti-Cheat Results")
    md.append("")
    md.append(f"- best_valid_for_ranking: {int(valid_pass)}")
    md.append(f"- best_overall_entries_valid: {int(metrics.get('overall_entries_valid', 0))}")
    md.append(f"- best_overall_entry_rate: {float(metrics.get('overall_entry_rate', np.nan)):.6f}")
    md.append(f"- finite_required_metrics_pass: {int(finite_pass)}")
    md.append(f"- no_trade_genome_selected_as_best: {int(no_trade_best)}")
    md.append(f"- per_symbol_entry_rate_gate_pass: {int(per_symbol_entry_pass)}")
    md.append("")
    md.append("## Invalid Reason Histogram")
    md.append("")
    if invalid_hist:
        for k, v in sorted(invalid_hist.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            md.append(f"- {k}: {int(v)}")
    else:
        md.append("- (none)")
    md.append("")
    md.append("## Smoke Assertions")
    md.append("")
    md.append(f"- assertions_pass: **{int(smoke['assertions_pass'])}**")
    md.append(f"- smoke_test_result: `{out / 'smoke_test_result.json'}`")
    (out / "ga_patch_report.md").write_text("\n".join(md).strip() + "\n", encoding="utf-8")

    repro = [
        "# Repro",
        "",
        "```bash",
        "cd /root/analysis/0.87",
        f".venv/bin/python scripts/ga_patch_report.py --ga-run-dir {source_run}",
        "```",
    ]
    (out / "repro.md").write_text("\n".join(repro).strip() + "\n", encoding="utf-8")

    (out / "phase_result.md").write_text(
        "\n".join(
            [
                "Phase: B (GA Anti-Cheat Patch + Smoke)",
                f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
                f"Status: {'PASS' if int(smoke['assertions_pass']) == 1 else 'FAIL'}",
                f"Source run: {source_run}",
                f"Best valid_for_ranking: {int(valid_pass)}",
                f"Best entries_valid: {int(metrics.get('overall_entries_valid', 0))}",
                f"Best entry_rate: {float(metrics.get('overall_entry_rate', np.nan)):.6f}",
                f"No-trade best blocked: {int(not no_trade_best)}",
                f"Artifacts: {out / 'ga_patch_report.md'}, {out / 'invalid_reason_histogram.json'}, {out / 'smoke_test_result.json'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out / "config_snapshot").mkdir(parents=True, exist_ok=True)
    if (source_run / "ga_config.yaml").exists():
        (out / "config_snapshot" / "ga_config.yaml").write_text((source_run / "ga_config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    if (source_run / "fee_model.json").exists():
        (out / "config_snapshot" / "fee_model.json").write_text((source_run / "fee_model.json").read_text(encoding="utf-8"), encoding="utf-8")
    os.system(f"git -C {PROJECT_ROOT} status --short > {out / 'git_status.txt'}")

    print(str(out))
    print(str(out / "ga_patch_report.md"))
    print(str(out / "phase_result.md"))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Phase B GA patch smoke report from a GA run directory.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--ga-run-dir", default="", help="Optional GA_EXEC_OPT_* run dir. Default: latest.")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--min-entry-rate-overall", type=float, default=0.70)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

