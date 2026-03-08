from __future__ import annotations

from typing import Iterable


def redact_secret(value: str | None, keep_prefix: int = 0, keep_suffix: int = 0) -> str:
    if not value:
        return ""

    text = str(value)
    prefix = max(0, int(keep_prefix))
    suffix = max(0, int(keep_suffix))

    # Default behavior fully redacts secret material to prevent partial leakage.
    if prefix == 0 and suffix == 0:
        return "[REDACTED]"
    if len(text) <= prefix + suffix:
        return "[REDACTED]"
    return f"{text[:prefix]}***{text[-suffix:]}"


def redact_text(text: str, secrets: Iterable[str | None]) -> str:
    out = text
    for secret in secrets:
        if not secret:
            continue
        out = out.replace(secret, redact_secret(secret))
    return out
