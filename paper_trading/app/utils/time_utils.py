from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def date_yyyymmdd(dt: datetime | None = None) -> str:
    cur = dt or utc_now()
    return cur.strftime("%Y%m%d")
