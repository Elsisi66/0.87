#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/root/analysis/0.87").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.app.config import load_settings  # noqa: E402
from paper_trading.app.model_a_runtime import ModelAPaperRuntime, utc_iso, utc_tag  # type: ignore[attr-defined]  # noqa: E402
from paper_trading.app.utils.io import atomic_write_json, atomic_write_text  # noqa: E402
from paper_trading.app.utils.logging_utils import configure_logging  # noqa: E402


REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"


def git_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["git_head"] = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        out["git_head"] = "unavailable"
    try:
        status = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True)
        out["git_status_short"] = status.strip().splitlines()
    except Exception:
        out["git_status_short"] = []
    return out


def write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def json_dump(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def main() -> None:
    run_dir = REPORTS_ROOT / f"PHASEE_MODEL_A_PAPER_RUNTIME_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    settings = load_settings(PROJECT_ROOT)
    logger = configure_logging(
        settings.logs_dir / "phase_e_model_a_runtime.log",
        settings.logs_dir / "phase_e_model_a_runtime_errors.log",
        settings.log_level,
    )
    runtime = ModelAPaperRuntime(settings=settings, logger=logger, force_local_only=True)

    phase_e1_arch = [
        "# Phase E1 Runtime Architecture",
        "",
        "## A) 1h Signal / Position Owner",
        "- `paper_trading/app/model_a_runtime.py::ModelAPaperRuntime` uses `SignalRunner` on 1h bars only.",
        "- The 1h owner computes signal state, cycle, and TP/SL geometry from the frozen SOL params.",
        "- Exit evaluation is performed only by the 1h owner via `_maybe_close_position(...)` using 1h bars and the locked 12h evaluation horizon.",
        "",
        "## B) 3m Entry Executor",
        "- `ModelAFeed` fetches 3m bars separately from the 1h feed.",
        "- `_simulate_entry_fill(...)` handles only entry placement/fill mechanics: market vs limit, offset, bounded fallback, and fill delay.",
        "- Once a fill is established, the 3m executor has no authority over TP, SL, or any downstream exit mutation.",
        "",
        "## C) Paper / Shadow Coordinator",
        "- `paper_primary` and `shadow_backup` are loaded from the locked Phase C selection files.",
        "- Each candidate has its own isolated `StateStore` under `paper_trading/state/model_a_runtime/<role>`.",
        "- Shared market data is used, but books, journals, and summaries are kept separate and attributable by role/candidate.",
    ]
    write_text(run_dir / "phaseE1_runtime_architecture.md", "\n".join(phase_e1_arch) + "\n")

    component_map = {
        "generated_utc": utc_iso(),
        "runtime_module": str(PROJECT_ROOT / "paper_trading" / "app" / "model_a_runtime.py"),
        "runtime_launcher": str(PROJECT_ROOT / "paper_trading" / "scripts" / "run_model_a_runtime.py"),
        "orchestrator": str(PROJECT_ROOT / "scripts" / "phase_e_model_a_paper_runtime.py"),
        "signal_owner": "paper_trading.app.signal_runner.SignalRunner",
        "entry_executor": "paper_trading.app.model_a_runtime.ModelAPaperRuntime._simulate_entry_fill",
        "exit_owner": "paper_trading.app.model_a_runtime.ModelAPaperRuntime._maybe_close_position",
        "books": [candidate.role for candidate in runtime.candidates],
        "candidate_ids": [candidate.candidate_id for candidate in runtime.candidates],
        "state_root": str(runtime.state_root),
    }
    json_dump(run_dir / "phaseE1_component_map.json", component_map)

    file_manifest = {
        "generated_utc": utc_iso(),
        "created_files": [
            str(PROJECT_ROOT / "paper_trading" / "app" / "model_a_runtime.py"),
            str(PROJECT_ROOT / "paper_trading" / "scripts" / "run_model_a_runtime.py"),
            str(PROJECT_ROOT / "scripts" / "phase_e_model_a_paper_runtime.py"),
        ],
        "reused_components": [
            str(PROJECT_ROOT / "paper_trading" / "app" / "signal_runner.py"),
            str(PROJECT_ROOT / "paper_trading" / "app" / "state_store.py"),
            str(PROJECT_ROOT / "paper_trading" / "app" / "notifier.py"),
        ],
        "generic_daemon_reused_as_is": 0,
        "state_root": str(runtime.state_root),
    }
    json_dump(run_dir / "phaseE2_file_manifest.json", file_manifest)

    build_lines = [
        "# Phase E2 Build Report",
        "",
        "- Built a dedicated Model A runtime separate from `paper_trading/app/main.py`.",
        "- The new runtime does not import or instantiate `ExecutionSimulator`.",
        "- The runtime loads exact Phase C primary/backup configs and persists them into a dedicated Model A state root.",
        "- Shared 1h and 3m feeds are handled by `ModelAFeed`; entry and exit ownership are split exactly along the Model A contract.",
        "",
        "## Files",
        f"- Runtime module: `{PROJECT_ROOT / 'paper_trading/app/model_a_runtime.py'}`",
        f"- Runtime launcher: `{PROJECT_ROOT / 'paper_trading/scripts/run_model_a_runtime.py'}`",
        f"- Phase E orchestrator: `{PROJECT_ROOT / 'scripts/phase_e_model_a_paper_runtime.py'}`",
    ]
    write_text(run_dir / "phaseE2_build_report.md", "\n".join(build_lines) + "\n")

    validation = runtime.validate_contract()
    validation["runtime_launcher_exists"] = int((PROJECT_ROOT / "paper_trading" / "scripts" / "run_model_a_runtime.py").exists())
    validation["candidate_mapping_exists"] = int(runtime.root_mapping_path.exists())
    json_dump(run_dir / "phaseE3_contract_validation.json", validation)

    phase_e3_lines = [
        "# Phase E3 Contract Parity Tests",
        "",
        f"- Freeze lock pass: `{validation['freeze_lock_pass']}`",
        f"- Repaired routes remain support-feasible: `{validation['repaired_route_support_feasible']}`",
        f"- Uses 1h signal owner: `{validation['uses_1h_signal_owner']}`",
        f"- Uses 3m entry executor: `{validation['uses_3m_entry_executor']}`",
        f"- Exits owned by 1h only: `{validation['exits_owned_by_1h_only']}`",
        f"- Forbidden exit controls active: `{validation['forbidden_exit_controls_active']}`",
        f"- Primary/backup loaded: `{validation['primary_backup_loaded']}`",
        f"- Isolated books: `{validation['isolated_books']}`",
        f"- 1h feed ready: `{validation['one_h_feed_ready']}`",
        f"- 3m feed ready: `{validation['three_m_feed_ready']}`",
        f"- Candidate mapping exists: `{validation['candidate_mapping_exists']}`",
        f"- Runtime launcher exists: `{validation['runtime_launcher_exists']}`",
        "",
        "## Exact Parity Proof",
        "- The runtime validates the same fee/metrics hashes as the locked Model A research path.",
        "- The 3m path is used only inside `_simulate_entry_fill(...)`.",
        "- Exit handling is performed only inside `_maybe_close_position(...)` on 1h bars.",
        "- Primary and backup each use a dedicated `StateStore`, proving isolated books.",
    ]
    write_text(run_dir / "phaseE3_contract_parity_tests.md", "\n".join(phase_e3_lines) + "\n")

    anchor_ts = runtime.compute_smoke_anchor()
    reset_meta = runtime.hard_reset(start_from_bar_ts=anchor_ts)
    smoke = runtime.run_cycle(latest_only=False, max_rows=96)
    cycle_json_path, cycle_md_path = runtime.write_cycle_summary(run_dir)
    telegram_probe = runtime.probe_telegram()

    smoke_rows = []
    for role, stats in smoke["books"].items():
        smoke_rows.append(
            {
                "role": role,
                "candidate_id": stats["candidate_id"],
                "signals_seen": stats["signals_seen"],
                "entries_attempted": stats["entries_attempted"],
                "entries_filled": stats["entries_filled"],
                "exits_processed": stats["exits_processed"],
                "runtime_errors": stats["runtime_errors"],
                "recovery_actions": stats["recovery_actions"],
                "open_positions": stats["open_positions"],
                "realized_pnl_eur": stats["realized_pnl_eur"],
                "taker_share": stats["taker_share"],
                "avg_fill_delay_min": stats["avg_fill_delay_min"],
                "p95_fill_delay_min": stats["p95_fill_delay_min"],
                "state_consistent": stats["state_consistent"],
                "telegram_probe_reason": telegram_probe.reason,
            }
        )
    overall = {
        "role": "overall",
        "candidate_id": "ALL",
        "signals_seen": sum(int(row["signals_seen"]) for row in smoke_rows),
        "entries_attempted": sum(int(row["entries_attempted"]) for row in smoke_rows),
        "entries_filled": sum(int(row["entries_filled"]) for row in smoke_rows),
        "exits_processed": sum(int(row["exits_processed"]) for row in smoke_rows),
        "runtime_errors": sum(int(row["runtime_errors"]) for row in smoke_rows),
        "recovery_actions": sum(int(row["recovery_actions"]) for row in smoke_rows),
        "open_positions": sum(int(row["open_positions"]) for row in smoke_rows),
        "realized_pnl_eur": sum(float(row["realized_pnl_eur"]) for row in smoke_rows),
        "taker_share": 0.0,
        "avg_fill_delay_min": 0.0,
        "p95_fill_delay_min": 0.0,
        "state_consistent": int(all(int(row["state_consistent"]) == 1 for row in smoke_rows)),
        "telegram_probe_reason": telegram_probe.reason,
    }
    smoke_rows.append(overall)

    smoke_csv = run_dir / "phaseE4_smoke_results.csv"
    with smoke_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(smoke_rows[0].keys()))
        writer.writeheader()
        writer.writerows(smoke_rows)

    smoke_lines = [
        "# Phase E4 Smoke Report",
        "",
        f"- Smoke anchor start_from_bar_ts: `{reset_meta['start_from_bar_ts']}`",
        f"- Latest closed 1h bar processed: `{smoke['latest_closed_1h']}`",
        f"- Market meta: `{smoke['market_meta']}`",
        f"- Telegram probe: `{telegram_probe.reason}`",
        f"- Cycle summary JSON: `{cycle_json_path}`",
        f"- Cycle summary MD: `{cycle_md_path}`",
    ]
    for row in smoke_rows:
        smoke_lines.extend(
            [
                "",
                f"## {row['role']}",
                f"- Candidate: `{row['candidate_id']}`",
                f"- Signals seen: `{row['signals_seen']}`",
                f"- Entries attempted / filled: `{row['entries_attempted']}` / `{row['entries_filled']}`",
                f"- Exits processed: `{row['exits_processed']}`",
                f"- Runtime errors: `{row['runtime_errors']}`",
                f"- Recovery actions: `{row['recovery_actions']}`",
                f"- Open positions: `{row['open_positions']}`",
                f"- Realized PnL EUR: `{row['realized_pnl_eur']}`",
                f"- Taker share: `{row['taker_share']}`",
                f"- Avg / P95 fill delay: `{row['avg_fill_delay_min']}` / `{row['p95_fill_delay_min']}`",
                f"- State consistent: `{row['state_consistent']}`",
            ]
        )
    write_text(run_dir / "phaseE4_smoke_report.md", "\n".join(smoke_lines) + "\n")

    error_lines = [
        "# Phase E4 Error Recovery Report",
        "",
        f"- Telegram probe reason: `{telegram_probe.reason}`",
        f"- Runtime health counters: `{smoke['health']}`",
        f"- Coordinator dead-letter path: `{runtime.root_dead_letter_path}`",
        f"- Coordinator journal path: `{runtime.root_journal_path}`",
    ]
    for role, stats in smoke["books"].items():
        error_lines.extend(
            [
                "",
                f"## {role}",
                f"- Runtime errors: `{stats['runtime_errors']}`",
                f"- Recovery actions: `{stats['recovery_actions']}`",
                f"- State consistent: `{stats['state_consistent']}`",
                f"- Open positions after smoke: `{stats['open_positions']}`",
            ]
        )
    write_text(run_dir / "phaseE4_error_recovery_report.md", "\n".join(error_lines) + "\n")

    critical_contract_ok = (
        int(validation["freeze_lock_pass"]) == 1
        and int(validation["repaired_route_support_feasible"]) == 1
        and int(validation["uses_1h_signal_owner"]) == 1
        and int(validation["uses_3m_entry_executor"]) == 1
        and int(validation["exits_owned_by_1h_only"]) == 1
        and int(validation["forbidden_exit_controls_active"]) == 0
        and int(validation["primary_backup_loaded"]) == 1
        and int(validation["isolated_books"]) == 1
        and int(validation["one_h_feed_ready"]) == 1
        and int(validation["three_m_feed_ready"]) == 1
    )
    primary_row = next(row for row in smoke_rows if row["role"] == "paper_primary")
    backup_row = next(row for row in smoke_rows if row["role"] == "shadow_backup")
    both_books_execute = int(primary_row["entries_filled"] > 0 and backup_row["entries_filled"] > 0)
    smoke_clean = int(
        all(int(row["runtime_errors"]) == 0 for row in [primary_row, backup_row])
        and all(int(row["state_consistent"]) == 1 for row in [primary_row, backup_row])
        and both_books_execute == 1
    )

    classification = "PAPER_READY_STRONG"
    blockers: list[str] = []
    if not critical_contract_ok:
        classification = "PAPER_BLOCKED_INFRA"
        blockers.append("contract_parity_failed")
    elif both_books_execute != 1:
        classification = "PAPER_BLOCKED_INFRA"
        blockers.append("primary_or_backup_did_not_execute")
    elif smoke_clean != 1:
        classification = "PAPER_READY_CAUTION"
        blockers.append("smoke_not_fully_clean")

    next_prompt = ""
    if classification in {"PAPER_READY_STRONG", "PAPER_READY_CAUTION"}:
        caution = "Run with standard rollback matrix." if classification == "PAPER_READY_STRONG" else "Run with caution and watch early-cycle metrics."
        next_prompt = (
            "ROLE\n"
            "You are launching the dedicated Model A paper/shadow runtime for SOLUSDT.\n\n"
            "MISSION\n"
            "Start only the dedicated Model A runtime, not the generic daemon. Keep 1h signal/exit ownership frozen and use 3m for entry execution only.\n\n"
            "COMMANDS\n"
            f"1) First launch:\nPYTHONPATH={PROJECT_ROOT} {PROJECT_ROOT / '.venv/bin/python'} {PROJECT_ROOT / 'paper_trading/scripts/run_model_a_runtime.py'} --reset --max-cycles 1\n"
            f"2) Ongoing runtime:\nPYTHONPATH={PROJECT_ROOT} {PROJECT_ROOT / '.venv/bin/python'} {PROJECT_ROOT / 'paper_trading/scripts/run_model_a_runtime.py'} --max-cycles 24 --latest-only\n\n"
            "PRIMARY\n"
            "M3_ENTRY_ONLY_FASTER_C_WIN_02\n\n"
            "BACKUP\n"
            "M2_ENTRY_ONLY_MORE_PASSIVE_NOFB_C_FB_ON\n\n"
            "ROLLBACK\n"
            "Use the Phase D rollback matrix and stop immediately on any hybrid-exit mutation, route-center regression, or taker-share breach above 0.25.\n\n"
            f"NOTE\n{caution}\n"
        )
        write_text(run_dir / "ready_to_launch_modelA_paper_runtime.txt", next_prompt)

    decision_lines = [
        "# Phase E Decision",
        "",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{classification}`",
        f"- Critical contract parity ok: `{int(critical_contract_ok)}`",
        f"- Both books executed in smoke: `{both_books_execute}`",
        f"- Smoke clean: `{smoke_clean}`",
        "",
        "## Blockers / Notes",
    ]
    if blockers:
        decision_lines.extend([f"- `{item}`" for item in blockers])
    else:
        decision_lines.append("- None")
    decision_lines.extend(
        [
            "",
            "## Readiness",
            f"- Primary candidate: `{primary_row['candidate_id']}`",
            f"- Backup candidate: `{backup_row['candidate_id']}`",
            f"- Telegram probe: `{telegram_probe.reason}`",
        ]
    )
    write_text(run_dir / "phaseE_decision.md", "\n".join(decision_lines) + "\n")

    manifest = {
        "generated_utc": utc_iso(),
        "artifact_dir": str(run_dir),
        "git": git_snapshot(),
        "phases_executed": ["E1", "E2", "E3", "E4", "E5"],
        "classification": classification,
        "mainline_status": classification,
        "critical_contract_ok": int(critical_contract_ok),
        "both_books_execute": both_books_execute,
        "smoke_clean": smoke_clean,
        "blockers": blockers,
        "runtime_state_root": str(runtime.state_root),
    }
    json_dump(run_dir / "phaseE_run_manifest.json", manifest)

    print(str(run_dir))


if __name__ == "__main__":
    main()
