#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from paper_trading.app.config import load_settings
from paper_trading.app.main import run_daemon
from paper_trading.app.state_store import StateStore
from paper_trading.app.utils.io import atomic_write_json, atomic_write_text
from paper_trading.app.utils.time_utils import utc_iso


def _count_fills(journal_path: Path) -> tuple[int, int]:
    if not journal_path.exists():
        return 0, 0
    open_count = 0
    close_count = 0
    with journal_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = row.get("event")
            if evt == "fill_open":
                open_count += 1
            elif evt == "fill_close":
                close_count += 1
    return open_count, close_count


def main() -> None:
    settings = load_settings(Path("/root/analysis/0.87"))
    state = StateStore(settings.state_dir)

    # Run 1: hard reset + immediate cycle. Forward-only mode should not replay backlog.
    run_daemon(
        settings,
        once=True,
        max_cycles=1,
        replay_bars=None,
        reset_on_start=True,
    )

    open_1, close_1 = _count_fills(state.journal_path)
    processed_1 = state.load_processed_bars()
    runtime_1 = state.load_runtime_meta()

    # Run 2: restart without reset; should not duplicate already processed bars.
    run_daemon(
        settings,
        once=True,
        max_cycles=1,
        replay_bars=None,
        reset_on_start=False,
    )

    open_2, close_2 = _count_fills(state.journal_path)
    processed_2 = state.load_processed_bars()
    runtime_2 = state.load_runtime_meta()

    duplicate_delta_open = open_2 - open_1
    duplicate_delta_close = close_2 - close_1

    result = {
        "generated_utc": utc_iso(),
        "run1_fill_open_count": open_1,
        "run1_fill_close_count": close_1,
        "run2_fill_open_count": open_2,
        "run2_fill_close_count": close_2,
        "delta_fill_open_after_restart": duplicate_delta_open,
        "delta_fill_close_after_restart": duplicate_delta_close,
        "processed_bars_count_run1": len(processed_1),
        "processed_bars_count_run2": len(processed_2),
        "start_from_bar_ts_run1": runtime_1.get("start_from_bar_ts"),
        "last_processed_bar_ts_run1": runtime_1.get("last_global_processed_bar_ts"),
        "last_processed_bar_ts_run2": runtime_2.get("last_global_processed_bar_ts"),
        "forward_only_no_backlog_on_reset": open_1 == 0 and close_1 == 0,
        "idempotent_restart_check": duplicate_delta_open == 0 and duplicate_delta_close == 0,
    }

    json_path = settings.reports_dir / "paper_phaseP4_smoke_results.json"
    md_path = settings.reports_dir / "paper_phaseP4_smoke_report.md"
    atomic_write_json(json_path, result)

    lines = [
        "# Paper Smoke Test Report",
        "",
        f"- Generated UTC: `{result['generated_utc']}`",
        f"- Run1 fill counts (open/close): `{open_1}/{close_1}`",
        f"- Run2 fill counts (open/close): `{open_2}/{close_2}`",
        f"- Delta after restart (open/close): `{duplicate_delta_open}/{duplicate_delta_close}`",
        f"- Start from bar TS (run1): `{result['start_from_bar_ts_run1']}`",
        f"- Last processed bar TS (run1/run2): `{result['last_processed_bar_ts_run1']}` / `{result['last_processed_bar_ts_run2']}`",
        f"- No backlog on reset: `{result['forward_only_no_backlog_on_reset']}`",
        f"- Idempotent restart check: `{result['idempotent_restart_check']}`",
    ]
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
