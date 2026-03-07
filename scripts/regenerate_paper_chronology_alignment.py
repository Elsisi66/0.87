#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.app.config import load_settings
from paper_trading.app.execution_sim import ExecutionSimulator
from paper_trading.app.portfolio import default_portfolio
from paper_trading.app.signal_runner import SignalRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate paper chronology alignment report with fixed timestamp normalization")
    p.add_argument(
        "--project-root",
        default="/root/analysis/0.87",
        help="Project root path",
    )
    p.add_argument(
        "--window-start",
        default="2026-03-04T00:00:00Z",
        help="Alignment window start UTC",
    )
    p.add_argument(
        "--window-end",
        default="2026-03-06T00:00:00Z",
        help="Alignment window end UTC",
    )
    p.add_argument(
        "--history-start",
        default="2025-12-01T00:00:00Z",
        help="History start UTC for indicator warmup",
    )
    p.add_argument(
        "--history-end",
        default="2026-03-06T23:59:59Z",
        help="History end UTC for replay data fetch",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional explicit output directory under reports/execution_layer",
    )
    return p.parse_args()


def fetch_klines(symbol: str, interval: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    rows: list[list[Any]] = []
    start_ms = int(start_ts.value // 10**6)
    end_ms = int(end_ts.value // 10**6)
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            break
        rows.extend(payload)
        next_start = int(payload[-1][0]) + 3600 * 1000
        if next_start > end_ms or len(payload) < 1000:
            break
        start_ms = next_start

    if not rows:
        raise RuntimeError("No Binance klines fetched for alignment window")

    return pd.DataFrame(
        {
            "Timestamp": [pd.to_datetime(int(r[0]), unit="ms", utc=True) for r in rows],
            "Open": [float(r[1]) for r in rows],
            "High": [float(r[2]) for r in rows],
            "Low": [float(r[3]) for r in rows],
            "Close": [float(r[4]) for r in rows],
            "Volume": [float(r[5]) for r in rows],
        }
    )


def _ts_iso(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.isoformat()


def run_replay(
    frame_window: pd.DataFrame,
    params: dict[str, Any],
    *,
    fee_bps: float,
    slippage_choices: list[int],
    defer_guard: bool,
) -> dict[str, Any]:
    sim = ExecutionSimulator(
        fee_bps,
        slippage_choices,
        seed=87,
        defer_exit_to_next_bar=defer_guard,
    )
    sim.reset_guard_stats()
    portfolio = default_portfolio(320.0)
    positions: dict[str, dict[str, Any]] = {}

    signal_count = int(frame_window["SIGNAL"].astype(bool).sum())
    event_rows: list[dict[str, Any]] = []
    opens = closes = pos_closes = neg_closes = 0
    same_bar_closes = 0
    entry_before_exit_violations = 0
    entry_on_signal_violations = 0

    # BUGFIX: canonicalize all signal timestamps to ISO before lookup.
    signal_map = {
        _ts_iso(ts): bool(sig)
        for ts, sig in zip(frame_window["Timestamp"], frame_window["SIGNAL"])
    }

    for _, row in frame_window.iterrows():
        result = sim.process_bar(
            symbol="SOLUSDT",
            row=row.to_dict(),
            params=params,
            quote_to_eur=1.0,
            portfolio=portfolio,
            positions=positions,
        )
        for ev in result.events:
            ev_type = str(ev.get("event", ""))
            if ev_type not in {"fill_open", "fill_close", "exit_deferred_to_next_bar", "exit_blocked_pre_entry"}:
                continue

            # BUGFIX: normalize event timestamps before comparisons/lookups.
            bar_ts_iso = _ts_iso(ev.get("bar_ts"))
            entry_ts_iso = _ts_iso(ev.get("entry_ts"))

            rec = {
                "mode": "GUARD" if defer_guard else "NO_GUARD",
                "event": ev_type,
                "symbol": "SOLUSDT",
                "bar_ts": bar_ts_iso,
                "entry_ts": entry_ts_iso,
                "hold_hours": ev.get("hold_hours", ""),
                "same_bar_exit": "",
                "entry_before_exit_ok": "",
                "entry_signal_expected": "",
                "net_pnl_eur": ev.get("net_pnl_eur", ""),
                "fee_bps": ev.get("fee_bps", ""),
                "slippage_bps": ev.get("slippage_bps", ""),
            }

            if ev_type == "fill_open":
                opens += 1
                signal_expected = int(signal_map.get(bar_ts_iso, False))
                rec["entry_signal_expected"] = signal_expected
                if signal_expected != 1:
                    entry_on_signal_violations += 1

            if ev_type == "fill_close":
                closes += 1
                hold = float(ev.get("hold_hours", 0.0))
                same = int(hold <= 0)
                rec["same_bar_exit"] = same
                same_bar_closes += same

                entry_ts = pd.to_datetime(ev.get("entry_ts"), utc=True, errors="coerce")
                exit_ts = pd.to_datetime(ev.get("bar_ts"), utc=True, errors="coerce")
                strict_ok = int(pd.notna(entry_ts) and pd.notna(exit_ts) and entry_ts < exit_ts)
                rec["entry_before_exit_ok"] = strict_ok
                if strict_ok == 0:
                    entry_before_exit_violations += 1

                pnl = float(ev.get("net_pnl_eur", 0.0))
                if pnl > 0:
                    pos_closes += 1
                elif pnl < 0:
                    neg_closes += 1

            event_rows.append(rec)

    summary = {
        "signals": signal_count,
        "opens": opens,
        "closes": closes,
        "pos_closes": pos_closes,
        "neg_closes": neg_closes,
        "same_bar_closes": same_bar_closes,
        "same_bar_close_rate": float(same_bar_closes / closes) if closes else 0.0,
        "entry_on_signal_violations": entry_on_signal_violations,
        "entry_before_exit_violations": entry_before_exit_violations,
        "close_win_rate": float(pos_closes / closes) if closes else 0.0,
        "all_negative_close_pathology": bool(closes > 0 and pos_closes == 0 and neg_closes == closes),
        "guard_stats": sim.snapshot_guard_stats(),
    }
    return {"events": event_rows, "summary": summary}


def latest_prior_chronology_report(reports_dir: Path) -> Path | None:
    candidates = sorted(
        reports_dir.glob("PAPER_REPAIRED_CHRONOLOGY_FIX_*"),
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        marker = c / "updated_paper_alignment_report.md"
        if marker.exists():
            return c
    return None


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    reports_dir = root / "reports" / "execution_layer"
    settings = load_settings(root)

    if args.outdir:
        out_dir = Path(args.outdir).resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = reports_dir / f"PAPER_REPAIRED_CHRONOLOGY_FIX_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)

    start_window = pd.Timestamp(args.window_start)
    end_window = pd.Timestamp(args.window_end)
    start_history = pd.Timestamp(args.history_start)
    end_history = pd.Timestamp(args.history_end)

    params_path = Path(settings.repaired_active_params_dir) / "SOLUSDT_repaired_selected_params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    market = fetch_klines("SOLUSDT", "1h", start_history, end_history)
    runner = SignalRunner(logger=None)
    params = runner.load_symbol_params("SOLUSDT", str(params_path))
    frame = runner.build_signal_frame("SOLUSDT", market, params).frame.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True, errors="coerce")
    window = frame[(frame["Timestamp"] >= start_window) & (frame["Timestamp"] < end_window)].copy().reset_index(drop=True)
    if window.empty:
        raise RuntimeError("Alignment window empty after market fetch and signal reconstruction")

    baseline = run_replay(
        window,
        params,
        fee_bps=settings.fee_bps,
        slippage_choices=settings.slippage_bps_choices,
        defer_guard=False,
    )
    guarded = run_replay(
        window,
        params,
        fee_bps=settings.fee_bps,
        slippage_choices=settings.slippage_bps_choices,
        defer_guard=True,
    )

    # Compare with immediately previous chronology report, if present.
    prev_dir = latest_prior_chronology_report(reports_dir)
    prev_false_pos = None
    if prev_dir and prev_dir != out_dir:
        prev_stats_path = prev_dir / "chronology_guard_stats.json"
        if prev_stats_path.exists():
            prev_stats = json.loads(prev_stats_path.read_text(encoding="utf-8"))
            prev_entry_flags = int(prev_stats.get("guarded_summary", {}).get("entry_on_signal_violations", 0))
            curr_entry_flags = int(guarded["summary"]["entry_on_signal_violations"])
            prev_false_pos = {
                "prior_report_dir": str(prev_dir),
                "prior_entry_on_signal_flags": prev_entry_flags,
                "current_entry_on_signal_flags": curr_entry_flags,
                "flags_removed": max(0, prev_entry_flags - curr_entry_flags),
            }

    # Required outputs
    align_rows = baseline["events"] + guarded["events"]
    align_csv = out_dir / "updated_paper_alignment_check.csv"
    with align_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(align_rows[0].keys()) if align_rows else [
            "mode",
            "event",
            "symbol",
            "bar_ts",
            "entry_ts",
            "hold_hours",
            "same_bar_exit",
            "entry_before_exit_ok",
            "entry_signal_expected",
            "net_pnl_eur",
            "fee_bps",
            "slippage_bps",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in align_rows:
            w.writerow(row)

    summary_table = {
        "total_trades_guarded": int(guarded["summary"]["closes"]),
        "same_bar_exit_count_guarded": int(guarded["summary"]["same_bar_closes"]),
        "entry_on_signal_flag_count_guarded": int(guarded["summary"]["entry_on_signal_violations"]),
        "exit_before_entry_count_guarded": int(guarded["summary"]["entry_before_exit_violations"]),
    }

    stats_payload = {
        "window_start_utc": str(start_window),
        "window_end_utc": str(end_window),
        "normalization_fix": {
            "bug_location": str(root / "scripts" / "regenerate_paper_chronology_alignment.py"),
            "bug_lines": "126-134 and 149-151",
            "before_behavior": "raw event bar_ts string compared to ISO keys caused false entry_on_signal misses",
            "after_behavior": "event bar_ts normalized via pd.to_datetime(...).isoformat() before lookup",
        },
        "baseline_no_guard_summary": baseline["summary"],
        "guarded_summary": guarded["summary"],
        "summary_table": summary_table,
        "prior_report_comparison": prev_false_pos,
    }
    (out_dir / "chronology_guard_stats.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    files_touched = [
        "paper_trading/app/universe.py",
        "paper_trading/app/main.py",
        "paper_trading/app/execution_sim.py",
        "scripts/regenerate_paper_chronology_alignment.py",
    ]
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--"] + files_touched,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        diff = ""
    patch_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files_touched": files_touched,
        "git_diff_available": bool(diff),
        "git_diff": diff if diff else "unavailable (repo tracks these paths as untracked in current state)",
        "window_start_utc": str(start_window),
        "window_end_utc": str(end_window),
    }
    (out_dir / "patch_manifest.json").write_text(json.dumps(patch_manifest, indent=2), encoding="utf-8")

    outcome = (
        "PAPER_ALIGNED"
        if (
            summary_table["same_bar_exit_count_guarded"] == 0
            and summary_table["entry_on_signal_flag_count_guarded"] == 0
            and summary_table["exit_before_entry_count_guarded"] == 0
            and guarded["summary"]["all_negative_close_pathology"] is False
        )
        else "PAPER_MISMATCH_REMAINS"
    )

    lines = [
        "# Updated Paper Alignment Report (Timestamp Normalization Fix)",
        "",
        "## Fixed Reporter Bug",
        "- File: `scripts/regenerate_paper_chronology_alignment.py`",
        "- Lines: `126-134` and `149-151`",
        "- Before: `bar_ts` from events used raw string format (`YYYY-MM-DD HH:MM:SS+00:00`) against ISO-keyed signal map (`YYYY-MM-DDTHH:MM:SS+00:00`).",
        "- After: event timestamps are normalized with `pd.to_datetime(..., utc=True).isoformat()` before signal-map lookup.",
        "",
        "## Window",
        f"- {start_window} to {end_window} UTC",
        "",
        "## Summary Table (Guarded Replay)",
        "| metric | value |",
        "|---|---:|",
        f"| same-bar exit count | {summary_table['same_bar_exit_count_guarded']} |",
        f"| entry-on-signal flag count | {summary_table['entry_on_signal_flag_count_guarded']} |",
        f"| exit-before-entry count | {summary_table['exit_before_entry_count_guarded']} |",
        f"| total trades | {summary_table['total_trades_guarded']} |",
        "",
    ]
    if prev_false_pos is not None:
        lines.extend(
            [
                "## Outcome Change vs Prior Chronology Report",
                f"- Prior entry-on-signal flags: {prev_false_pos['prior_entry_on_signal_flags']}",
                f"- Current entry-on-signal flags: {prev_false_pos['current_entry_on_signal_flags']}",
                f"- False positives removed by normalization fix: {prev_false_pos['flags_removed']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Final Outcome",
            f"- {outcome}",
            "",
        ]
    )

    if outcome == "PAPER_MISMATCH_REMAINS":
        lines.extend(
            [
                "## Top 3 Remaining Root-Cause Candidates (Trading Path)",
                "1. `paper_trading/app/execution_sim.py:151-213` chronology guard edge handling for deferred exits.",
                "2. `paper_trading/app/reconciler.py:73-104` per-bar processing order / state transition edge cases.",
                "3. `paper_trading/app/main.py:355-453` symbol cycle sequencing and feed/meta interaction.",
                "",
            ]
        )

    (out_dir / "updated_paper_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
