from __future__ import annotations

import errno
import json
import random
import socket
import time
import urllib.parse
import urllib.request
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError


_RETRY_REASON_DNS = "dns"
_RETRY_REASON_429 = "429"
_RETRY_REASON_TIMEOUT = "timeout"
_RETRY_REASON_OTHER = "other"
_RETRY_REASON_SET = {_RETRY_REASON_DNS, _RETRY_REASON_429, _RETRY_REASON_TIMEOUT, _RETRY_REASON_OTHER}

_RETRYABLE_ERRNOS = {
    errno.ECONNRESET,
    errno.ETIMEDOUT,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ENETDOWN,
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.EPIPE,
}


class FetchRetryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        attempts: int,
        status_code: Optional[int] = None,
        last_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason if reason in _RETRY_REASON_SET else _RETRY_REASON_OTHER
        self.attempts = int(attempts)
        self.status_code = int(status_code) if status_code is not None else None
        self.last_error = last_error


def _iter_exceptions(ex: BaseException) -> Iterable[BaseException]:
    seen = set()
    stack = [ex]
    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        yield cur
        if isinstance(cur, URLError) and isinstance(getattr(cur, "reason", None), BaseException):
            stack.append(cur.reason)  # type: ignore[arg-type]
        for nxt in (getattr(cur, "__cause__", None), getattr(cur, "__context__", None), getattr(cur, "reason", None)):
            if isinstance(nxt, BaseException):
                stack.append(nxt)


def _status_code_from_exception(ex: BaseException) -> Optional[int]:
    if isinstance(ex, HTTPError):
        try:
            return int(ex.code)
        except Exception:
            return None
    for node in _iter_exceptions(ex):
        code = getattr(node, "code", None)
        if isinstance(code, int):
            return int(code)
    return None


def classify_retry_reason(ex: BaseException, *, status_code: Optional[int] = None) -> str:
    code = int(status_code) if status_code is not None else _status_code_from_exception(ex)
    if code == 429:
        return _RETRY_REASON_429
    for node in _iter_exceptions(ex):
        if isinstance(node, socket.gaierror):
            return _RETRY_REASON_DNS
        if isinstance(node, (socket.timeout, TimeoutError)):
            return _RETRY_REASON_TIMEOUT
    text = str(ex).lower()
    if (
        "reason=429" in text
        or "status=429" in text
        or "http error 429" in text
        or "too many requests" in text
    ):
        return _RETRY_REASON_429
    if "temporary failure in name resolution" in text or "name or service not known" in text or "nodename nor servname provided" in text:
        return _RETRY_REASON_DNS
    if "timed out" in text or "timeout" in text:
        return _RETRY_REASON_TIMEOUT
    return _RETRY_REASON_OTHER


def _is_retryable(ex: BaseException, *, status_code: Optional[int] = None) -> bool:
    code = int(status_code) if status_code is not None else _status_code_from_exception(ex)
    if code == 429 or (code is not None and 500 <= code <= 599):
        return True

    for node in _iter_exceptions(ex):
        if isinstance(
            node,
            (
                URLError,
                socket.gaierror,
                socket.timeout,
                TimeoutError,
                ConnectionResetError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                BrokenPipeError,
                RemoteDisconnected,
                IncompleteRead,
                HTTPException,
            ),
        ):
            return True
        if isinstance(node, OSError):
            err_no = getattr(node, "errno", None)
            if isinstance(err_no, int) and err_no in _RETRYABLE_ERRNOS:
                return True

    text = str(ex).lower()
    if "temporary failure in name resolution" in text:
        return True
    if "timed out" in text or "timeout" in text:
        return True
    return False


def http_get_json_with_retry(
    *,
    base: str,
    path: str,
    params: Dict[str, str],
    timeout: int = 30,
    user_agent: str = "bot087-exec-gate/1.0",
    max_retries: int = 8,
    retry_base_sleep_sec: float = 0.5,
    retry_max_sleep_sec: float = 30.0,
    log_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_context: Optional[Dict[str, Any]] = None,
) -> object:
    retries = max(0, int(max_retries))
    base_sleep = max(0.0, float(retry_base_sleep_sec))
    max_sleep = max(0.0, float(retry_max_sleep_sec))
    ctx = dict(log_context or {})

    qs = urllib.parse.urlencode(params)
    url = f"{base}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as ex:
            last_error = ex
            status_code = _status_code_from_exception(ex)
            reason = classify_retry_reason(ex, status_code=status_code)
            retryable = _is_retryable(ex, status_code=status_code)
            attempt_no = int(attempt + 1)
            exhausted = attempt >= retries
            sleep_sec = 0.0
            if retryable and not exhausted:
                backoff = min(max_sleep, base_sleep * (2.0 ** attempt))
                sleep_sec = float(min(max_sleep, max(0.0, backoff * random.uniform(0.8, 1.2))))
            if log_cb is not None:
                ev = {
                    "event": "http_retry",
                    "url": url,
                    "attempt": attempt_no,
                    "max_retries": retries,
                    "status_code": status_code,
                    "reason": reason,
                    "retryable": bool(retryable),
                    "error_type": type(ex).__name__,
                    "error": str(ex),
                    "sleep_sec": sleep_sec,
                }
                if ctx:
                    ev.update(ctx)
                log_cb(ev)
            if not retryable or exhausted:
                msg = (
                    f"HTTP fetch failed after {attempt_no} attempts "
                    f"(reason={reason}, status={status_code}): {ex}"
                )
                raise FetchRetryError(
                    msg,
                    reason=reason,
                    attempts=attempt_no,
                    status_code=status_code,
                    last_error=ex,
                ) from ex
            time.sleep(sleep_sec)

    # Defensive: retries loop always exits by return or raise.
    msg = f"HTTP fetch failed without explicit error after retries={retries}"
    raise FetchRetryError(msg, reason=_RETRY_REASON_OTHER, attempts=retries + 1, last_error=last_error)
