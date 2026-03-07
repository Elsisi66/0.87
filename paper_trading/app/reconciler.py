from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .execution_sim import ExecutionSimulator
from .health import HealthTracker
from .signal_runner import SignalFrame, SignalRunner
from .state_store import StateStore
from .utils.time_utils import utc_iso


@dataclass
class ReconcileResult:
    symbol: str
    bars_processed: int
    opened: int
    closed: int
    fill_events: list[dict[str, Any]]
    last_processed_bar_ts: str | None


class Reconciler:
    def __init__(
        self,
        *,
        state_store: StateStore,
        signal_runner: SignalRunner,
        execution_sim: ExecutionSimulator,
        health: HealthTracker,
        logger,
    ) -> None:
        self.state = state_store
        self.signal_runner = signal_runner
        self.execution_sim = execution_sim
        self.health = health
        self.logger = logger

    def _load_mutable_state(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        portfolio = self.state.load_portfolio()
        positions = self.state.load_positions()
        processed = self.state.load_processed_bars()
        return portfolio, positions, processed

    def _save_mutable_state(
        self,
        portfolio: dict[str, Any],
        positions: dict[str, Any],
        processed: dict[str, Any],
    ) -> None:
        portfolio["last_updated_utc"] = utc_iso()
        self.state.save_portfolio(portfolio)
        self.state.save_positions(positions)
        self.state.save_processed_bars(processed)
        self.state.save_health_counters(self.health.as_state())

    def bootstrap_processed_marker(self, symbol: str, marker_ts: pd.Timestamp) -> None:
        _, _, processed = self._load_mutable_state()
        processed[symbol] = pd.to_datetime(marker_ts, utc=True).isoformat()
        self.state.save_processed_bars(processed)

    def process_symbol_rows(
        self,
        *,
        symbol: str,
        signal_frame: SignalFrame,
        quote_to_eur: float,
        max_bar_ts: pd.Timestamp,
        start_from_bar_ts: pd.Timestamp | None,
    ) -> ReconcileResult:
        portfolio, positions, processed = self._load_mutable_state()

        last_ts_raw = processed.get(symbol)
        last_ts = pd.to_datetime(last_ts_raw, utc=True) if last_ts_raw else None
        rows = self.signal_runner.rows_after(
            signal_frame,
            last_ts,
            pd.to_datetime(max_bar_ts, utc=True),
            start_from_bar_ts=start_from_bar_ts,
            latest_only=True,
        )

        opened = 0
        closed = 0
        bars_processed = 0
        fill_events: list[dict[str, Any]] = []

        for _, row in rows.iterrows():
            bars_processed += 1
            row_payload = row.to_dict()
            result = self.execution_sim.process_bar(
                symbol=symbol,
                row=row_payload,
                params=signal_frame.params,
                quote_to_eur=float(quote_to_eur),
                portfolio=portfolio,
                positions=positions,
            )
            opened += result.opened
            closed += result.closed

            for event in result.events:
                recorded_ts = utc_iso()
                event_payload = dict(event)
                event_payload["event_recorded_ts"] = recorded_ts
                event_payload["ts_utc"] = recorded_ts
                event_payload["symbol"] = symbol
                self.state.append_journal(event_payload)
                if event_payload.get("event") in {"fill_open", "fill_close"}:
                    fill_events.append(event_payload)

            processed[symbol] = pd.to_datetime(row["Timestamp"], utc=True).isoformat()

        self._save_mutable_state(portfolio, positions, processed)
        return ReconcileResult(
            symbol=symbol,
            bars_processed=bars_processed,
            opened=opened,
            closed=closed,
            fill_events=fill_events,
            last_processed_bar_ts=(processed.get(symbol) if bars_processed else last_ts_raw),
        )
