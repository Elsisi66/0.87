from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HealthSnapshot:
    counters: dict[str, int]
    degraded_mode: bool
    strategy_health: str


class HealthTracker:
    def __init__(self, counters: dict[str, Any] | None = None) -> None:
        base = counters or {}
        self.counters: dict[str, int] = {
            "api_retries": int(base.get("api_retries", 0)),
            "api_failures": int(base.get("api_failures", 0)),
            "signal_errors": int(base.get("signal_errors", 0)),
            "execution_errors": int(base.get("execution_errors", 0)),
            "recovery_events": int(base.get("recovery_events", 0)),
            "telegram_errors": int(base.get("telegram_errors", 0)),
            "quarantined_symbols": int(base.get("quarantined_symbols", 0)),
        }
        self.degraded_mode = bool(base.get("degraded_mode", False))

    def inc(self, key: str, amount: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0) + amount)

    def set_quarantined_symbols(self, count: int) -> None:
        self.counters["quarantined_symbols"] = int(max(0, count))

    def set_degraded_mode(self, value: bool) -> None:
        self.degraded_mode = bool(value)

    def meter(self) -> str:
        severe = (
            self.counters.get("execution_errors", 0)
            + self.counters.get("signal_errors", 0)
            + self.counters.get("api_failures", 0)
        )
        if self.degraded_mode or severe >= 20 or self.counters.get("quarantined_symbols", 0) >= 3:
            return "RED"
        if severe >= 6 or self.counters.get("api_retries", 0) >= 15:
            return "YELLOW"
        return "GREEN"

    def snapshot(self) -> HealthSnapshot:
        return HealthSnapshot(
            counters=dict(self.counters),
            degraded_mode=self.degraded_mode,
            strategy_health=self.meter(),
        )

    def as_state(self) -> dict[str, Any]:
        return {
            **self.counters,
            "degraded_mode": self.degraded_mode,
            "strategy_health": self.meter(),
        }
