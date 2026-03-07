from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Scheduler:
    poll_seconds: int

    def sleep(self) -> None:
        time.sleep(max(1, int(self.poll_seconds)))

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)
