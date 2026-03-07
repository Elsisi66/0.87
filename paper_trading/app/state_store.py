from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils.io import append_jsonl, atomic_write_json, ensure_dir, read_json
from .utils.time_utils import utc_iso, utc_tag


@dataclass
class StateStore:
    state_dir: Path

    def __post_init__(self) -> None:
        ensure_dir(self.state_dir)
        self.portfolio_path = self.state_dir / "portfolio_state.json"
        self.positions_path = self.state_dir / "positions.json"
        self.orders_path = self.state_dir / "orders.json"
        self.processed_bars_path = self.state_dir / "processed_bars.json"
        self.journal_path = self.state_dir / "journal.jsonl"
        self.dead_letter_path = self.state_dir / "dead_letter_queue.jsonl"
        self.quarantine_path = self.state_dir / "symbol_quarantine.json"
        self.health_path = self.state_dir / "health_counters.json"
        self.reset_marker_path = self.state_dir / "startup_reset_marker.json"
        self.runtime_meta_path = self.state_dir / "runtime_meta.json"
        self.archive_dir = ensure_dir(self.state_dir / "archive")

    def initialize(self, start_equity_eur: float) -> None:
        if not self.portfolio_path.exists():
            self.save_portfolio(
                {
                    "cash_eur": float(start_equity_eur),
                    "initial_equity_eur": float(start_equity_eur),
                    "realized_pnl_eur": 0.0,
                    "fees_paid_eur": 0.0,
                    "slippage_paid_eur": 0.0,
                    "trade_count_opened": 0,
                    "trade_count_closed": 0,
                    "wins": 0,
                    "losses": 0,
                    "last_summary_date": None,
                    "last_updated_utc": utc_iso(),
                    "degraded_mode": False,
                    "mode_note": "startup",
                }
            )
        if not self.positions_path.exists():
            self.save_positions({})
        if not self.orders_path.exists():
            self.save_orders([])
        if not self.processed_bars_path.exists():
            self.save_processed_bars({})
        if not self.quarantine_path.exists():
            self.save_quarantine({})
        if not self.health_path.exists():
            self.save_health_counters(
                {
                    "api_retries": 0,
                    "api_failures": 0,
                    "signal_errors": 0,
                    "execution_errors": 0,
                    "recovery_events": 0,
                    "telegram_errors": 0,
                }
            )
        if not self.runtime_meta_path.exists():
            self.save_runtime_meta({})

    def load_portfolio(self) -> dict[str, Any]:
        return read_json(self.portfolio_path, {}) or {}

    def save_portfolio(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["last_updated_utc"] = utc_iso()
        atomic_write_json(self.portfolio_path, payload)

    def load_positions(self) -> dict[str, Any]:
        return read_json(self.positions_path, {}) or {}

    def save_positions(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.positions_path, payload)

    def load_orders(self) -> list[dict[str, Any]]:
        raw = read_json(self.orders_path, [])
        return raw if isinstance(raw, list) else []

    def save_orders(self, payload: list[dict[str, Any]]) -> None:
        atomic_write_json(self.orders_path, payload)

    def load_processed_bars(self) -> dict[str, Any]:
        return read_json(self.processed_bars_path, {}) or {}

    def save_processed_bars(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.processed_bars_path, payload)

    def load_quarantine(self) -> dict[str, Any]:
        return read_json(self.quarantine_path, {}) or {}

    def save_quarantine(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.quarantine_path, payload)

    def load_health_counters(self) -> dict[str, Any]:
        return read_json(self.health_path, {}) or {}

    def save_health_counters(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.health_path, payload)

    def append_journal(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        recorded_ts = utc_iso()
        payload.setdefault("event_recorded_ts", recorded_ts)
        payload.setdefault("ts_utc", recorded_ts)
        append_jsonl(self.journal_path, payload)

    def append_dead_letter(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        recorded_ts = utc_iso()
        payload.setdefault("event_recorded_ts", recorded_ts)
        payload.setdefault("ts_utc", recorded_ts)
        append_jsonl(self.dead_letter_path, payload)

    def save_reset_marker(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.reset_marker_path, payload)

    def load_reset_marker(self) -> dict[str, Any]:
        return read_json(self.reset_marker_path, {}) or {}

    def load_runtime_meta(self) -> dict[str, Any]:
        return read_json(self.runtime_meta_path, {}) or {}

    def save_runtime_meta(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.runtime_meta_path, payload)

    def archive_state(self, label: str = "hard_reset") -> Path:
        backup_dir = ensure_dir(self.archive_dir / f"{label}_{utc_tag()}")
        for path in [
            self.portfolio_path,
            self.positions_path,
            self.orders_path,
            self.processed_bars_path,
            self.journal_path,
            self.dead_letter_path,
            self.quarantine_path,
            self.health_path,
            self.reset_marker_path,
            self.runtime_meta_path,
        ]:
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
        return backup_dir
