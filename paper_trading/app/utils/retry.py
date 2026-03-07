from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    attempts: int = 4
    base_delay_sec: float = 0.6
    max_delay_sec: float = 8.0
    jitter_sec: float = 0.3


def retry_call(
    fn: Callable[[], T],
    *,
    cfg: RetryConfig,
    retry_exceptions: Iterable[type[Exception]] = (Exception,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    errors = tuple(retry_exceptions)
    last_exc: Exception | None = None
    for attempt in range(1, max(1, cfg.attempts) + 1):
        try:
            return fn()
        except errors as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt >= cfg.attempts:
                break
            delay = min(cfg.max_delay_sec, cfg.base_delay_sec * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, cfg.jitter_sec)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
