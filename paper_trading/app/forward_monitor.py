from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from .config import Settings
from .state_store import StateStore
from .universe import ResolvedUniverse
from .utils.io import atomic_write_json, atomic_write_text, ensure_dir
from .utils.time_utils import date_yyyymmdd, utc_iso, utc_tag


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ts(value: Any) -> datetime | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _to_iso(value: Any) -> str:
    ts = _parse_ts(value)
    return ts.isoformat() if ts is not None else ""


def _day_bounds_utc(target_day_utc: datetime) -> tuple[datetime, datetime]:
    start = datetime(target_day_utc.year, target_day_utc.month, target_day_utc.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ensure_forward_monitor_dir(
    *,
    settings: Settings,
    state: StateStore,
    universe: ResolvedUniverse,
    logger,
) -> Path:
    runtime_meta = state.load_runtime_meta()
    existing = str(runtime_meta.get("forward_monitor_dir", "")).strip()
    if existing:
        monitor_dir = Path(existing)
        ensure_dir(monitor_dir)
    else:
        monitor_dir = ensure_dir(
            settings.project_root
            / "reports"
            / "execution_layer"
            / f"PAPER_SOL_FORWARD_MONITOR_{utc_tag()}"
        )
        runtime_meta["forward_monitor_dir"] = str(monitor_dir)
        runtime_meta["forward_monitor_started_utc"] = utc_iso()
        state.save_runtime_meta(runtime_meta)

    _write_manifest(settings=settings, universe=universe, monitor_dir=monitor_dir)
    logger.info("paper_forward_monitor_dir path=%s", monitor_dir)
    return monitor_dir


def _write_manifest(*, settings: Settings, universe: ResolvedUniverse, monitor_dir: Path) -> None:
    sol_params_path = Path(universe.symbol_params.get("SOLUSDT", ""))
    posture_table = Path(settings.repaired_posture_freeze_dir) / "repaired_3m_posture_table.csv"
    manifest = {
        "generated_utc": utc_iso(),
        "paper_mode": bool(settings.paper_mode),
        "binance_mode": settings.binance_mode,
        "allowlist": [str(x).upper() for x in settings.paper_symbol_allowlist],
        "required_active_strategy_id": settings.required_active_strategy_id,
        "repaired_contract_defer_exit_to_next_bar": bool(settings.repaired_contract_defer_exit_to_next_bar),
        "posture_freeze_dir": settings.repaired_posture_freeze_dir,
        "active_subset_csv": settings.repaired_active_subset_csv,
        "active_params_dir": settings.repaired_active_params_dir,
        "sol_params_path": str(sol_params_path),
        "sol_winner_config_id": universe.winner_config_ids.get("SOLUSDT"),
        "hashes": {
            "settings_yaml_sha256": _sha256_file(settings.settings_yaml),
            "env_sha256": _sha256_file(settings.env_file),
            "active_subset_csv_sha256": _sha256_file(Path(settings.repaired_active_subset_csv)),
            "posture_table_sha256": _sha256_file(posture_table),
            "sol_params_sha256": _sha256_file(sol_params_path),
        },
    }
    atomic_write_json(monitor_dir / "paper_forward_manifest.json", manifest)


def _trade_record_from_open(row: dict[str, Any], fallback_trade_id: str) -> dict[str, Any]:
    trade_id = str(row.get("trade_id") or fallback_trade_id)
    signal_time = _to_iso(row.get("signal_time") or row.get("bar_ts"))
    entry_time = _to_iso(row.get("entry_time") or row.get("entry_ts") or row.get("bar_ts"))
    return {
        "trade_id": trade_id,
        "symbol": "SOLUSDT",
        "side": str(row.get("side", "LONG") or "LONG"),
        "signal_time_utc": signal_time,
        "entry_time_utc": entry_time,
        "exit_time_utc": "",
        "entry_px_quote": _safe_float(row.get("entry_px_quote")),
        "exit_px_quote": None,
        "units": _safe_float(row.get("units")),
        "entry_fee_eur": _safe_float(row.get("entry_fee_eur")),
        "exit_fee_eur": 0.0,
        "fees_eur": _safe_float(row.get("entry_fee_eur")),
        "entry_slippage_eur": _safe_float(row.get("entry_slippage_eur")),
        "exit_slippage_eur": 0.0,
        "slippage_eur": _safe_float(row.get("entry_slippage_eur")),
        "pnl_eur": None,
        "pnl_pct": None,
        "exit_reason": "",
        "hold_hours": None,
        "same_bar_exit": 0,
        "exit_before_entry": 0,
        "entry_on_signal_violation": 0,
        "event_recorded_open_utc": _to_iso(row.get("event_recorded_ts") or row.get("ts_utc")),
        "event_recorded_close_utc": "",
    }


def write_daily_truth_pack(
    *,
    settings: Settings,
    state: StateStore,
    monitor_dir: Path,
    target_day_utc: datetime,
    logger,
) -> dict[str, Any]:
    rows = _iter_jsonl(state.journal_path)
    if not rows:
        day_tag = date_yyyymmdd(target_day_utc)
        empty_summary = {
            "date_utc": day_tag,
            "total_trades": 0,
            "closed_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "expectancy_eur": 0.0,
            "same_bar_exit_count": 0,
            "entry_on_signal_violations": 0,
            "exit_before_entry_count": 0,
            "guard_deferrals": {
                "same_bar_exit_attempts": 0,
                "exits_deferred_to_next_bar": 0,
                "exits_blocked_pre_entry": 0,
            },
            "health": "GREEN",
        }
        atomic_write_json(monitor_dir / f"paper_daily_truthpack_{day_tag}.json", empty_summary)
        return empty_summary

    start_utc, end_utc = _day_bounds_utc(target_day_utc)
    day_tag = date_yyyymmdd(target_day_utc)

    def _event_time(row: dict[str, Any]) -> datetime:
        return (
            _parse_ts(row.get("event_recorded_ts"))
            or _parse_ts(row.get("ts_utc"))
            or _parse_ts(row.get("bar_ts"))
            or datetime(1970, 1, 1, tzinfo=timezone.utc)
        )

    rows_sorted = sorted(rows, key=_event_time)

    signal_map: dict[str, bool] = {}
    guard_day = {
        "same_bar_exit_attempts": 0,
        "exits_deferred_to_next_bar": 0,
        "exits_blocked_pre_entry": 0,
    }
    trades: dict[str, dict[str, Any]] = {}
    open_queue: list[str] = []
    open_seq = 0

    for row in rows_sorted:
        event = str(row.get("event", ""))
        symbol = str(row.get("symbol", ""))
        evt_ts = _event_time(row)

        if event == "signal_decision" and symbol == "SOLUSDT":
            bar_key = _to_iso(row.get("bar_ts"))
            signal_map[bar_key] = bool(row.get("signal", False))
            continue

        if event == "chronology_guard_summary" and start_utc <= evt_ts < end_utc:
            guard_day["same_bar_exit_attempts"] += _safe_int(row.get("same_bar_exit_attempts"))
            guard_day["exits_deferred_to_next_bar"] += _safe_int(row.get("exits_deferred_to_next_bar"))
            guard_day["exits_blocked_pre_entry"] += _safe_int(row.get("exits_blocked_pre_entry"))
            continue

        if symbol != "SOLUSDT":
            continue

        if event == "fill_open":
            open_seq += 1
            fallback_trade_id = f"SOLUSDT:{_to_iso(row.get('entry_time') or row.get('bar_ts'))}:{open_seq}"
            rec = _trade_record_from_open(row, fallback_trade_id)
            trade_id = rec["trade_id"]
            if trade_id in trades:
                trade_id = f"{trade_id}:{open_seq}"
                rec["trade_id"] = trade_id
            trades[trade_id] = rec
            open_queue.append(trade_id)
            continue

        if event == "fill_close":
            trade_id = str(row.get("trade_id", "")).strip()
            if not trade_id or trade_id not in trades:
                trade_id = open_queue[0] if open_queue else f"SOLUSDT:orphan_close:{open_seq + 1}"
                if trade_id not in trades:
                    trades[trade_id] = _trade_record_from_open({}, trade_id)
            if trade_id in open_queue:
                open_queue.remove(trade_id)

            rec = trades[trade_id]
            rec["exit_time_utc"] = _to_iso(row.get("exit_time") or row.get("bar_ts"))
            rec["exit_px_quote"] = _safe_float(row.get("exit_px_quote"))
            rec["exit_reason"] = str(row.get("reason", ""))
            rec["hold_hours"] = _safe_int(row.get("hold_hours"), default=0)
            rec["exit_fee_eur"] = _safe_float(row.get("exit_fee_eur"))
            rec["exit_slippage_eur"] = _safe_float(row.get("exit_slippage_eur"))
            rec["fees_eur"] = _safe_float(row.get("fees_eur"), rec["entry_fee_eur"] + rec["exit_fee_eur"])
            rec["slippage_eur"] = _safe_float(
                row.get("slippage_eur"),
                rec["entry_slippage_eur"] + rec["exit_slippage_eur"],
            )
            rec["pnl_eur"] = _safe_float(row.get("net_pnl_eur"))
            rec["pnl_pct"] = _safe_float(row.get("pnl_pct"))
            rec["event_recorded_close_utc"] = _to_iso(row.get("event_recorded_ts") or row.get("ts_utc"))

            entry_ts = _parse_ts(rec.get("entry_time_utc"))
            exit_ts = _parse_ts(rec.get("exit_time_utc"))
            rec["exit_before_entry"] = int(bool(entry_ts and exit_ts and exit_ts <= entry_ts))
            rec["same_bar_exit"] = int(bool(rec["exit_before_entry"] or _safe_int(rec.get("hold_hours")) <= 0))

    trade_rows = list(trades.values())
    trade_rows.sort(key=lambda x: (x.get("entry_time_utc") or "", x.get("trade_id") or ""))

    for rec in trade_rows:
        signal_ts = str(rec.get("signal_time_utc", "")).strip()
        expected = signal_map.get(signal_ts)
        rec["entry_on_signal_violation"] = int(expected is not True)

        rec["same_bar_exit_attempts_day"] = guard_day["same_bar_exit_attempts"]
        rec["exits_deferred_to_next_bar_day"] = guard_day["exits_deferred_to_next_bar"]
        rec["exits_blocked_pre_entry_day"] = guard_day["exits_blocked_pre_entry"]

    def _in_day(ts_raw: Any) -> bool:
        ts = _parse_ts(ts_raw)
        return bool(ts is not None and start_utc <= ts < end_utc)

    day_rows = [r for r in trade_rows if _in_day(r.get("entry_time_utc")) or _in_day(r.get("exit_time_utc"))]
    closed_day = [r for r in day_rows if _in_day(r.get("exit_time_utc")) and r.get("pnl_eur") is not None]
    opens_day = [r for r in day_rows if _in_day(r.get("entry_time_utc"))]

    pnls = [float(r["pnl_eur"]) for r in closed_day]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_win = sum(wins)
    gross_loss_abs = abs(sum(losses))
    if gross_loss_abs <= 1e-12:
        profit_factor: float | str | None
        profit_factor = "inf" if gross_win > 0 else None
    else:
        profit_factor = gross_win / gross_loss_abs

    same_bar_exit_count = sum(_safe_int(r.get("same_bar_exit")) for r in closed_day)
    exit_before_entry_count = sum(_safe_int(r.get("exit_before_entry")) for r in closed_day)
    entry_on_signal_violations = sum(_safe_int(r.get("entry_on_signal_violation")) for r in opens_day)

    worst5 = sorted(closed_day, key=lambda x: _safe_float(x.get("pnl_eur")))[:5]
    worst5_rows = [
        {
            "trade_id": str(r.get("trade_id")),
            "pnl_eur": _safe_float(r.get("pnl_eur")),
            "pnl_pct": _safe_float(r.get("pnl_pct")),
            "entry_time_utc": str(r.get("entry_time_utc")),
            "exit_time_utc": str(r.get("exit_time_utc")),
            "exit_reason": str(r.get("exit_reason")),
        }
        for r in worst5
    ]

    health = "GREEN"
    if same_bar_exit_count > 0 or exit_before_entry_count > 0 or entry_on_signal_violations > 0:
        health = "RED"
    elif len(closed_day) == 0:
        health = "YELLOW"

    summary = {
        "generated_utc": utc_iso(),
        "date_utc": f"{start_utc.date()}",
        "total_trades": len(day_rows),
        "opened_trades": len(opens_day),
        "closed_trades": len(closed_day),
        "win_rate_pct": (100.0 * len(wins) / len(closed_day)) if closed_day else 0.0,
        "profit_factor": profit_factor,
        "expectancy_eur": mean(pnls) if pnls else 0.0,
        "mean_pnl_eur": mean(pnls) if pnls else 0.0,
        "median_pnl_eur": median(pnls) if pnls else 0.0,
        "same_bar_exit_count": same_bar_exit_count,
        "entry_on_signal_violations": entry_on_signal_violations,
        "exit_before_entry_count": exit_before_entry_count,
        "guard_deferrals": dict(guard_day),
        "health": health,
        "worst_5_trades": worst5_rows,
    }

    csv_path = monitor_dir / f"paper_trades_{day_tag}.csv"
    fieldnames = [
        "trade_id",
        "symbol",
        "side",
        "signal_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "entry_px_quote",
        "exit_px_quote",
        "units",
        "entry_fee_eur",
        "exit_fee_eur",
        "fees_eur",
        "entry_slippage_eur",
        "exit_slippage_eur",
        "slippage_eur",
        "pnl_eur",
        "pnl_pct",
        "exit_reason",
        "hold_hours",
        "same_bar_exit",
        "exit_before_entry",
        "entry_on_signal_violation",
        "same_bar_exit_attempts_day",
        "exits_deferred_to_next_bar_day",
        "exits_blocked_pre_entry_day",
        "event_recorded_open_utc",
        "event_recorded_close_utc",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in day_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    md_lines = [
        f"# SOL Paper Daily Truth Pack ({start_utc.date()})",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Total trades (any event in day): `{summary['total_trades']}`",
        f"- Opened/Closed: `{summary['opened_trades']}` / `{summary['closed_trades']}`",
        f"- Win rate: `{summary['win_rate_pct']:.2f}%`",
        f"- Profit factor: `{summary['profit_factor']}`",
        f"- Expectancy (EUR): `{summary['expectancy_eur']:.6f}`",
        f"- Mean/Median PnL (EUR): `{summary['mean_pnl_eur']:.6f}` / `{summary['median_pnl_eur']:.6f}`",
        "",
        "## Sanity",
        f"- Same-bar exits: `{summary['same_bar_exit_count']}`",
        f"- Exit-before-entry violations: `{summary['exit_before_entry_count']}`",
        f"- Entry-on-signal violations: `{summary['entry_on_signal_violations']}`",
        "",
        "## Chronology Guard Deferrals",
        f"- same_bar_exit_attempts: `{guard_day['same_bar_exit_attempts']}`",
        f"- exits_deferred_to_next_bar: `{guard_day['exits_deferred_to_next_bar']}`",
        f"- exits_blocked_pre_entry: `{guard_day['exits_blocked_pre_entry']}`",
        "",
        f"- Health: `{summary['health']}`",
    ]

    if worst5_rows:
        md_lines.extend(
            [
                "",
                "## Worst 5 Trades",
                "| trade_id | pnl_eur | pnl_pct | entry_time_utc | exit_time_utc | exit_reason |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        for row in worst5_rows:
            md_lines.append(
                f"| {row['trade_id']} | {row['pnl_eur']:.6f} | {row['pnl_pct']:.6f} | {row['entry_time_utc']} | {row['exit_time_utc']} | {row['exit_reason']} |"
            )

    md_text = "\n".join(md_lines) + "\n"
    atomic_write_text(monitor_dir / f"paper_daily_truthpack_{day_tag}.md", md_text)
    atomic_write_text(monitor_dir / f"paper_day_summary_{day_tag}.md", md_text)
    atomic_write_json(monitor_dir / f"paper_daily_truthpack_{day_tag}.json", summary)
    logger.info(
        "paper_daily_truthpack_written date=%s total=%s closed=%s sanity=(same_bar=%s,entry_signal=%s,exit_before_entry=%s)",
        start_utc.date(),
        len(day_rows),
        len(closed_day),
        same_bar_exit_count,
        entry_on_signal_violations,
        exit_before_entry_count,
    )
    return summary

