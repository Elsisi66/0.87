from __future__ import annotations

from typing import Iterable


def redact_secret(value: str | None, keep_prefix: int = 4, keep_suffix: int = 2) -> str:
    if not value:
        return ""
    if len(value) <= keep_prefix + keep_suffix:
        return "*" * len(value)
    return f"{value[:keep_prefix]}***{value[-keep_suffix:]}"


def redact_text(text: str, secrets: Iterable[str | None]) -> str:
    out = text
    for secret in secrets:
        if not secret:
            continue
        out = out.replace(secret, redact_secret(secret))
    return out
