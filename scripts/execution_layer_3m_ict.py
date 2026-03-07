#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.execution.http_retry import http_get_json_with_retry  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402


BINANCE_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"


def _utc_now() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    return now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _unwrap_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("params"), dict):
        return dict(payload["params"])
    return dict(payload)


def _normalize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    for src, dst in (
        ("timestamp", "Timestamp"),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if src in out.columns and dst not in out.columns:
            rename[src] = dst
    if rename:
        out = out.rename(columns=rename)

    if "Timestamp" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={"index": "Timestamp"})

    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    out = out.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    return out


def _find_latest_scan_dir() -> Path:
    base = PROJECT_ROOT / "reports" / "params_scan"
    if not base.exists():
        raise SystemExit(f"Missing directory: {base}")
    dirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise SystemExit(f"No scan runs found under {base}")
    return dirs[-1]


def _pick_symbol_from_best(best_csv: Path, rank: int, side: str = "long") -> Tuple[str, Path, pd.Series]:
    df = pd.read_csv(best_csv)
    if df.empty:
        raise SystemExit(f"No rows in {best_csv}")

    rows = df.copy()
    if "side" in rows.columns:
        rows = rows[rows["side"].astype(str).str.lower() == str(side).lower()].copy()
    if "pass" in rows.columns:
        rows = rows[rows["pass"].map(_as_bool)].copy()
    if rows.empty:
        rows = df.copy()

    rows["score"] = pd.to_numeric(rows.get("score"), errors="coerce").fillna(-1e18)
    rows = rows.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)

    rk = max(1, int(rank))
    if rk > len(rows):
        raise SystemExit(f"Requested rank={rk}, but only {len(rows)} candidate rows in {best_csv}")

    row = rows.iloc[rk - 1]
    symbol = str(row.get("symbol", "")).strip().upper()
    if not symbol:
        raise SystemExit(f"Invalid symbol at rank={rk} in {best_csv}")

    params_raw = str(row.get("params_file", "")).strip()
    if not params_raw:
        raise SystemExit(f"Missing params_file for symbol={symbol} in {best_csv}")

    params_file = _resolve_path(params_raw)
    if not params_file.exists():
        raise SystemExit(f"Missing params file for {symbol}: {params_file}")

    return symbol, params_file, row


def _load_symbol_df(symbol: str, tf: str = "1h") -> pd.DataFrame:
    fp_full = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if fp_full.exists():
        return _normalize_ohlcv_cols(pd.read_parquet(fp_full))

    fp_par = PROJECT_ROOT / "data" / "parquet" / f"{symbol}.parquet"
    if fp_par.exists():
        return _normalize_ohlcv_cols(pd.read_parquet(fp_par))

    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if files:
        return _normalize_ohlcv_cols(pd.concat([pd.read_csv(p) for p in files], ignore_index=True))

    raise FileNotFoundError(f"No dataset found for {symbol} tf={tf}")


def _tf_to_ms(tf: str) -> int:
    s = str(tf).strip().lower()
    if not s:
        raise ValueError("Empty timeframe")
    unit = s[-1]
    n = int(s[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in mult:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return int(n * mult[unit])


def _cache_file_for_range(cache_root: Path, symbol: str, tf: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Path:
    tag = f"{start_ts.strftime('%Y%m%dT%H%M%S')}_{end_ts.strftime('%Y%m%dT%H%M%S')}"
    return cache_root / symbol.upper() / tf.lower() / f"{symbol.upper()}_{tf.lower()}_{tag}.parquet"


def _read_local_full_slice(symbol: str, tf: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet",
        PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_features.parquet",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            try:
                df = pd.read_parquet(
                    fp,
                    columns=cols,
                    filters=[("Timestamp", ">=", start_ts), ("Timestamp", "<", end_ts)],
                )
            except Exception:
                df = pd.read_parquet(fp)
            df = _normalize_ohlcv_cols(df)
            out = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].reset_index(drop=True)
            if not out.empty:
                return out
        except Exception:
            continue
    return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])


def _fetch_binance_klines(
    *,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_retries: int,
    retry_base_sleep_sec: float,
    retry_max_sleep_sec: float,
    pause_sec: float,
) -> pd.DataFrame:
    if end_ts <= start_ts:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    tf_ms = _tf_to_ms(timeframe)
    start_ms = int(start_ts.value // 1_000_000)
    end_ms = int(end_ts.value // 1_000_000)
    cursor_ms = start_ms

    rows: List[List[Any]] = []
    while cursor_ms < end_ms:
        payload = http_get_json_with_retry(
            base=BINANCE_BASE,
            path=KLINES_PATH,
            params={
                "symbol": symbol.upper(),
                "interval": timeframe,
                "startTime": str(cursor_ms),
                "endTime": str(max(cursor_ms, end_ms - 1)),
                "limit": "1000",
            },
            timeout=30,
            max_retries=int(max_retries),
            retry_base_sleep_sec=float(retry_base_sleep_sec),
            retry_max_sleep_sec=float(retry_max_sleep_sec),
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected klines payload for {symbol} {timeframe}: {type(payload)}")
        if not payload:
            break

        rows.extend(payload)
        last_open = int(payload[-1][0])
        next_cursor = last_open + tf_ms
        if next_cursor <= cursor_ms:
            next_cursor = cursor_ms + tf_ms
        cursor_ms = next_cursor

        if pause_sec > 0:
            time.sleep(float(pause_sec))

        if len(payload) < 1000 and cursor_ms >= end_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    raw = pd.DataFrame(rows, columns=cols)
    out = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(pd.to_numeric(raw["open_time"], errors="coerce"), unit="ms", utc=True),
            "Open": pd.to_numeric(raw["open"], errors="coerce"),
            "High": pd.to_numeric(raw["high"], errors="coerce"),
            "Low": pd.to_numeric(raw["low"], errors="coerce"),
            "Close": pd.to_numeric(raw["close"], errors="coerce"),
            "Volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "QuoteVolume": pd.to_numeric(raw["quote_volume"], errors="coerce"),
            "Trades": pd.to_numeric(raw["trades"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    out = out.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    out = out[(out["Timestamp"] >= start_ts) & (out["Timestamp"] < end_ts)].reset_index(drop=True)
    return out


def _load_or_fetch_klines(
    *,
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_root: Path,
    max_retries: int,
    retry_base_sleep_sec: float,
    retry_max_sleep_sec: float,
    pause_sec: float,
) -> pd.DataFrame:
    cache_fp = _cache_file_for_range(cache_root, symbol, timeframe, start_ts, end_ts)
    if cache_fp.exists():
        try:
            df = pd.read_parquet(cache_fp)
            df = _normalize_ohlcv_cols(df)
            return df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].reset_index(drop=True)
        except Exception:
            pass

    local = _read_local_full_slice(symbol, timeframe, start_ts, end_ts)
    if not local.empty:
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        local.to_parquet(cache_fp, index=False)
        return local

    df = _fetch_binance_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        max_retries=max_retries,
        retry_base_sleep_sec=retry_base_sleep_sec,
        retry_max_sleep_sec=retry_max_sleep_sec,
        pause_sec=pause_sec,
    )
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_fp, index=False)
    return df


def _compute_atr14(df: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().fillna(0.0)


def _detect_swings(low: np.ndarray, high: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(low)
    k = max(1, int(k))
    swing_low = np.zeros(n, dtype=bool)
    swing_high = np.zeros(n, dtype=bool)
    if n < (2 * k + 1):
        return swing_low, swing_high

    for i in range(k, n - k):
        lv = float(low[i])
        hv = float(high[i])
        if np.isfinite(lv):
            left = low[i - k : i]
            right = low[i + 1 : i + 1 + k]
            if np.all(lv < left) and np.all(lv <= right):
                swing_low[i] = True
        if np.isfinite(hv):
            left_h = high[i - k : i]
            right_h = high[i + 1 : i + 1 + k]
            if np.all(hv > left_h) and np.all(hv >= right_h):
                swing_high[i] = True

    return swing_low, swing_high


def _previous_true_index(mask: np.ndarray) -> np.ndarray:
    out = np.full(len(mask), -1, dtype=int)
    last = -1
    for i, v in enumerate(mask.tolist()):
        out[i] = last
        if bool(v):
            last = int(i)
    return out


def _fmt_ts(x: Any) -> str:
    if x is None:
        return ""
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return ""
        return str(ts)
    except Exception:
        return ""


def _to_local_str(x: Any, local_tz: str) -> str:
    if x is None:
        return ""
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    if ZoneInfo is None:
        return str(ts)
    try:
        z = ZoneInfo(str(local_tz))
        return str(ts.tz_convert(z))
    except Exception:
        return str(ts)


def _parse_killzones(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    txt = str(raw).strip()
    if not txt:
        return out
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            continue
        a, b = part.split("-", 1)
        ah, am = a.strip().split(":")
        bh, bm = b.strip().split(":")
        start_min = int(ah) * 60 + int(am)
        end_min = int(bh) * 60 + int(bm)
        start_min = max(0, min(24 * 60, start_min))
        end_min = max(0, min(24 * 60, end_min))
        if end_min > start_min:
            out.append((start_min, end_min))
    return out


def _is_in_killzone(ts: pd.Timestamp, killzones: List[Tuple[int, int]]) -> bool:
    if not killzones:
        return True
    t = _to_utc_ts(ts)
    mm = int(t.hour) * 60 + int(t.minute)
    for s, e in killzones:
        if s <= mm < e:
            return True
    return False


def _find_latest_swing_low_idx(
    *,
    swing_low: np.ndarray,
    low: np.ndarray,
    idx: int,
    lookback_swings: int,
) -> int:
    i = int(idx) - 1
    found: List[int] = []
    while i >= 0 and len(found) < max(1, int(lookback_swings)):
        if bool(swing_low[i]) and np.isfinite(low[i]):
            found.append(i)
        i -= 1
    return found[0] if found else -1


def _find_liquidity_sweep_long(
    *,
    low: np.ndarray,
    close: np.ndarray,
    swing_low: np.ndarray,
    start_idx: int,
    max_wait_bars: int,
    closeback_bars: int,
    lookback_swings: int,
    sweep_ticks: float,
    sweep_pct: float,
    last_idx: int,
) -> Optional[Dict[str, Any]]:
    j_end = min(int(last_idx), int(start_idx) + max(1, int(max_wait_bars)))
    for j in range(int(start_idx), j_end + 1):
        ref_idx = _find_latest_swing_low_idx(
            swing_low=swing_low,
            low=low,
            idx=j,
            lookback_swings=lookback_swings,
        )
        if ref_idx < 0:
            continue
        ref_low = float(low[ref_idx])
        if not np.isfinite(ref_low):
            continue

        min_break = max(float(sweep_ticks), abs(ref_low) * max(0.0, float(sweep_pct)))
        if float(low[j]) >= (ref_low - min_break):
            continue

        k_end = min(j_end, j + max(1, int(closeback_bars)))
        for k in range(j, k_end + 1):
            if float(close[k]) >= ref_low:
                return {
                    "sweep_idx": int(j),
                    "confirm_idx": int(k),
                    "ref_swing_idx": int(ref_idx),
                    "ref_swing_low": float(ref_low),
                    "swept_low": float(low[j]),
                }
    return None


def _find_displacement_long(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    swing_high: np.ndarray,
    start_idx: int,
    max_confirm_bars: int,
    body_atr_mult: float,
    body_pct_thr: float,
    last_idx: int,
) -> Optional[Dict[str, Any]]:
    prev_swing_high_idx = _previous_true_index(swing_high)
    j_end = min(int(last_idx), int(start_idx) + max(1, int(max_confirm_bars)))

    for j in range(int(start_idx), j_end + 1):
        o = float(open_[j])
        c = float(close[j])
        body = c - o
        if body <= 0.0:
            continue

        atrv = float(atr[j])
        body_pct = body / max(1e-12, abs(o))
        body_ok = (body >= float(body_atr_mult) * atrv) or (body_pct >= float(body_pct_thr))
        if not body_ok:
            continue

        sh_idx = int(prev_swing_high_idx[j])
        if sh_idx < 0:
            continue
        sh = float(high[sh_idx])
        if not np.isfinite(sh):
            continue
        if c > sh:
            return {
                "disp_idx": int(j),
                "swing_high_idx": int(sh_idx),
                "mss_level": float(sh),
                "body": float(body),
                "body_pct": float(body_pct),
            }

    return None


def _find_bullish_fvg(
    *,
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    max_search_bars: int,
    last_idx: int,
) -> Optional[Dict[str, Any]]:
    i0 = max(2, int(start_idx))
    i1 = min(int(last_idx), int(start_idx) + max(1, int(max_search_bars)))
    for i in range(i0, i1 + 1):
        hi2 = float(high[i - 2])
        li = float(low[i])
        if li > hi2:
            return {
                "fvg_idx": int(i),
                "zone_low": float(hi2),
                "zone_high": float(li),
                "zone_mid": float((hi2 + li) / 2.0),
            }
    return None


def _find_bullish_ob(
    *,
    open_: np.ndarray,
    close: np.ndarray,
    disp_idx: int,
    start_idx: int,
) -> Optional[Dict[str, Any]]:
    i0 = max(0, int(start_idx))
    for i in range(int(disp_idx) - 1, i0 - 1, -1):
        o = float(open_[i])
        c = float(close[i])
        if c < o:
            return {
                "ob_idx": int(i),
                "open": float(o),
                "close": float(c),
                "mid": float((o + c) / 2.0),
                "body_low": float(min(o, c)),
                "body_high": float(max(o, c)),
            }
    return None


def _find_limit_fill_long(
    *,
    ts: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    price: float,
    start_idx: int,
    timeout_bars: int,
    use_killzones: bool,
    killzones: List[Tuple[int, int]],
    last_idx: int,
) -> Tuple[Optional[int], bool]:
    i0 = max(0, int(start_idx))
    i1 = min(int(last_idx), i0 + max(1, int(timeout_bars)))
    touched_outside_kz = False
    for i in range(i0, i1 + 1):
        if float(low[i]) <= float(price) <= float(high[i]):
            if use_killzones and not _is_in_killzone(pd.to_datetime(ts[i], utc=True), killzones):
                touched_outside_kz = True
                continue
            return int(i), touched_outside_kz
    return None, touched_outside_kz


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _parse_ict_score_weights(raw: str) -> Dict[str, float]:
    default = {
        "sweep": 1.0,
        "displacement": 1.0,
        "killzone": 1.0,
        "fvg": 1.0,
        "ob": 1.0,
    }
    txt = str(raw).strip()
    if not txt:
        return default
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            return default
        out = dict(default)
        for k in list(default.keys()):
            if k in parsed:
                out[k] = float(parsed[k])
        return out
    except Exception:
        return default


def _debug_append(
    debug_events: Optional[List[Dict[str, Any]]],
    *,
    signal_id: str,
    stage: str,
    decision_ts_ns: Optional[int],
    feature_last_ts_ns: Optional[int],
    note: str = "",
) -> None:
    if debug_events is None:
        return
    leak = 0
    if decision_ts_ns is not None and feature_last_ts_ns is not None and int(feature_last_ts_ns) > int(decision_ts_ns):
        leak = 1
    debug_events.append(
        {
            "signal_id": str(signal_id),
            "stage": str(stage),
            "decision_time_utc": _fmt_ts(pd.to_datetime(int(decision_ts_ns), utc=True)) if decision_ts_ns is not None else "",
            "feature_last_time_utc": _fmt_ts(pd.to_datetime(int(feature_last_ts_ns), utc=True)) if feature_last_ts_ns is not None else "",
            "lookahead_violation": int(leak),
            "note": str(note),
        }
    )


def _idx_at_or_before_ts(ts_ns: np.ndarray, target_ns: int, min_idx: int, max_idx: int) -> int:
    if len(ts_ns) == 0:
        return int(min_idx)
    j = int(np.searchsorted(ts_ns, int(target_ns), side="right")) - 1
    j = max(int(min_idx), j)
    j = min(int(max_idx), j)
    return int(j)


def _compute_eval_end_ns(
    *,
    entry_ts_ns: int,
    eval_horizon_hours: float,
    baseline_exit_time: Optional[pd.Timestamp] = None,
) -> int:
    horizon_ns = int(max(0.0, float(eval_horizon_hours)) * 3600.0 * 1e9)
    end_ns = int(entry_ts_ns + horizon_ns)
    if baseline_exit_time is not None:
        b_ns = int(_to_utc_ts(baseline_exit_time).value)
        end_ns = min(int(end_ns), int(b_ns))
    return int(end_ns)


def _simulate_path_long(
    *,
    ts_ns: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    max_exit_ts_ns: Optional[int] = None,
    tie_touch_policy: str = "sl_first",
) -> Dict[str, Any]:
    n = len(ts_ns)
    if n == 0 or entry_idx < 0 or entry_idx >= n or not np.isfinite(entry_price) or entry_price <= 0:
        return {
            "filled": False,
            "exit_reason": "invalid_entry",
            "invalid_stop_geometry": 0,
            "invalid_tp_geometry": 0,
            "same_bar_hit": 0,
            "valid_for_metrics": 0,
        }

    invalid_stop_geometry = int((not np.isfinite(sl_price)) or (float(sl_price) >= float(entry_price)))
    invalid_tp_geometry = int((not np.isfinite(tp_price)) or (float(tp_price) <= float(entry_price)))
    valid_for_metrics = int((invalid_stop_geometry == 0) and (invalid_tp_geometry == 0))

    entry_ts_ns = int(ts_ns[int(entry_idx)])
    if max_exit_ts_ns is None:
        eval_end_idx = n - 1
    else:
        eval_end_idx = _idx_at_or_before_ts(ts_ns=ts_ns, target_ns=int(max_exit_ts_ns), min_idx=int(entry_idx), max_idx=n - 1)
    eval_end_ts = pd.to_datetime(int(ts_ns[int(eval_end_idx)]), utc=True)

    if valid_for_metrics == 0:
        entry_ts = pd.to_datetime(int(ts_ns[entry_idx]), utc=True)
        return {
            "filled": True,
            "entry_idx": int(entry_idx),
            "entry_time": entry_ts,
            "entry_price": float(entry_price),
            "sl": float(sl_price) if np.isfinite(sl_price) else float("nan"),
            "tp": float(tp_price) if np.isfinite(tp_price) else float("nan"),
            "sl_hit": False,
            "tp_hit": False,
            "sl_hit_time": None,
            "tp_hit_time": None,
            "sl_hit_price": float("nan"),
            "tp_hit_price": float("nan"),
            "exit_time": entry_ts,
            "exit_price": float(entry_price),
            "exit_reason": "invalid_geometry",
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "time_to_mae_min": float("nan"),
            "time_to_mfe_min": float("nan"),
            "invalid_stop_geometry": int(invalid_stop_geometry),
            "invalid_tp_geometry": int(invalid_tp_geometry),
            "same_bar_hit": 0,
            "valid_for_metrics": 0,
            "eval_window_end_time": eval_end_ts,
            "eval_window_minutes": float((eval_end_ts - entry_ts).total_seconds() / 60.0),
        }

    sl_hit = False
    tp_hit = False
    sl_hit_idx: Optional[int] = None
    tp_hit_idx: Optional[int] = None
    same_bar_hit = 0
    exit_idx = int(eval_end_idx)
    exit_reason = "window_end"
    exit_price = float("nan")

    tie_policy = str(tie_touch_policy).strip().lower()
    if tie_policy not in {"sl_first", "tp_first", "distance_to_entry"}:
        tie_policy = "sl_first"

    for i in range(int(entry_idx), int(eval_end_idx) + 1):
        hit_sl = np.isfinite(sl_price) and float(low[i]) <= float(sl_price)
        hit_tp = np.isfinite(tp_price) and float(high[i]) >= float(tp_price)

        if hit_sl and hit_tp:
            sl_hit = True
            tp_hit = True
            same_bar_hit = 1
            sl_hit_idx = i
            tp_hit_idx = i
            exit_idx = i
            if tie_policy == "tp_first":
                exit_reason = "tp"
                exit_price = float(tp_price)
            elif tie_policy == "distance_to_entry":
                d_sl = abs(float(entry_price) - float(sl_price))
                d_tp = abs(float(tp_price) - float(entry_price))
                # Deterministic tie-break: if equal distance, keep conservative stop-first.
                if np.isfinite(d_tp) and np.isfinite(d_sl) and d_tp < d_sl:
                    exit_reason = "tp"
                    exit_price = float(tp_price)
                else:
                    exit_reason = "sl"
                    exit_price = float(sl_price)
            else:
                exit_reason = "sl"
                exit_price = float(sl_price)
            break
        if hit_sl:
            sl_hit = True
            sl_hit_idx = i
            exit_idx = i
            exit_reason = "sl"
            exit_price = float(sl_price)
            break
        if hit_tp:
            tp_hit = True
            tp_hit_idx = i
            exit_idx = i
            exit_reason = "tp"
            exit_price = float(tp_price)
            break

    if not np.isfinite(exit_price):
        exit_idx = int(eval_end_idx)
        exit_reason = "window_end"
        exit_price = float(close[exit_idx]) if np.isfinite(close[exit_idx]) else float(entry_price)

    lows = np.asarray(low[entry_idx : exit_idx + 1], dtype=float)
    highs = np.asarray(high[entry_idx : exit_idx + 1], dtype=float)

    if lows.size:
        rel_low = _safe_div(float(np.nanmin(lows)), float(entry_price)) - 1.0
        mae_loc = int(np.nanargmin(lows))
    else:
        rel_low = float("nan")
        mae_loc = 0

    if highs.size:
        rel_high = _safe_div(float(np.nanmax(highs)), float(entry_price)) - 1.0
        mfe_loc = int(np.nanargmax(highs))
    else:
        rel_high = float("nan")
        mfe_loc = 0

    mae_idx = int(entry_idx + mae_loc)
    mfe_idx = int(entry_idx + mfe_loc)

    entry_ts = pd.to_datetime(int(ts_ns[entry_idx]), utc=True)
    mae_ts = pd.to_datetime(int(ts_ns[mae_idx]), utc=True)
    mfe_ts = pd.to_datetime(int(ts_ns[mfe_idx]), utc=True)

    return {
        "filled": True,
        "entry_idx": int(entry_idx),
        "entry_time": entry_ts,
        "entry_price": float(entry_price),
        "sl": float(sl_price) if np.isfinite(sl_price) else float("nan"),
        "tp": float(tp_price) if np.isfinite(tp_price) else float("nan"),
        "sl_hit": bool(sl_hit),
        "tp_hit": bool(tp_hit),
        "sl_hit_time": pd.to_datetime(int(ts_ns[sl_hit_idx]), utc=True) if sl_hit_idx is not None else None,
        "tp_hit_time": pd.to_datetime(int(ts_ns[tp_hit_idx]), utc=True) if tp_hit_idx is not None else None,
        "sl_hit_price": float(sl_price) if sl_hit else float("nan"),
        "tp_hit_price": float(tp_price) if tp_hit else float("nan"),
        "exit_time": pd.to_datetime(int(ts_ns[exit_idx]), utc=True),
        "exit_price": float(exit_price),
        "exit_reason": str(exit_reason),
        "mae_pct": float(rel_low),
        "mfe_pct": float(rel_high),
        "time_to_mae_min": float((mae_ts - entry_ts).total_seconds() / 60.0),
        "time_to_mfe_min": float((mfe_ts - entry_ts).total_seconds() / 60.0),
        "invalid_stop_geometry": int(invalid_stop_geometry),
        "invalid_tp_geometry": int(invalid_tp_geometry),
        "same_bar_hit": int(same_bar_hit),
        "valid_for_metrics": int(valid_for_metrics),
        "eval_window_end_time": eval_end_ts,
        "eval_window_minutes": float((eval_end_ts - entry_ts).total_seconds() / 60.0),
    }


def _simulate_baseline_long(
    *,
    df3m: pd.DataFrame,
    signal_time: pd.Timestamp,
    tp_mult: float,
    sl_mult: float,
    eval_horizon_hours: float,
) -> Dict[str, Any]:
    if df3m.empty:
        return {"filled": False, "skip_reason": "no_3m_data"}

    ts_ser = pd.to_datetime(df3m["Timestamp"], utc=True, errors="coerce")
    open_ = pd.to_numeric(df3m["Open"], errors="coerce")
    high = pd.to_numeric(df3m["High"], errors="coerce")
    low = pd.to_numeric(df3m["Low"], errors="coerce")
    close = pd.to_numeric(df3m["Close"], errors="coerce")

    good = ts_ser.notna() & open_.notna() & high.notna() & low.notna() & close.notna()
    if not bool(good.any()):
        return {"filled": False, "skip_reason": "bad_3m_data"}

    ts_ok = ts_ser[good].tolist()
    ts_ns = np.array([int(t.value) for t in ts_ok], dtype=np.int64)
    open_np = open_[good].to_numpy(dtype=float)
    high_np = high[good].to_numpy(dtype=float)
    low_np = low[good].to_numpy(dtype=float)
    close_np = close[good].to_numpy(dtype=float)

    sig_ns = int(_to_utc_ts(signal_time).value)
    idx = int(np.searchsorted(ts_ns, sig_ns, side="left"))
    if idx >= len(ts_ns):
        return {"filled": False, "skip_reason": "no_bar_after_signal"}

    entry_price = float(open_np[idx])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"filled": False, "skip_reason": "bad_entry_price"}

    sl = float(entry_price * float(sl_mult))
    tp = float(entry_price * float(tp_mult))
    max_exit_ts_ns = _compute_eval_end_ns(
        entry_ts_ns=int(ts_ns[idx]),
        eval_horizon_hours=float(eval_horizon_hours),
        baseline_exit_time=None,
    )

    out = _simulate_path_long(
        ts_ns=ts_ns,
        close=close_np,
        high=high_np,
        low=low_np,
        entry_idx=idx,
        entry_price=entry_price,
        sl_price=sl,
        tp_price=tp,
        max_exit_ts_ns=max_exit_ts_ns,
    )
    out["entry_type"] = "market"
    out["fill_delay_minutes"] = float((pd.to_datetime(int(ts_ns[idx]), utc=True) - _to_utc_ts(signal_time)).total_seconds() / 60.0)
    out["skip_reason"] = ""
    return out


def _simulate_ict_long(
    *,
    df3m: pd.DataFrame,
    signal_time: pd.Timestamp,
    tp_mult: float,
    sl_mult: float,
    args: argparse.Namespace,
    killzones: List[Tuple[int, int]],
    debug_events: Optional[List[Dict[str, Any]]] = None,
    signal_id: str = "",
    baseline_exit_time: Optional[pd.Timestamp] = None,
    eval_horizon_hours: float = 12.0,
) -> Dict[str, Any]:
    if df3m.empty:
        return {"filled": False, "skip_reason": "no_3m_data", "ict_score": 0.0, "ict_score_components": "", "fallback_used": 0}

    x = df3m.copy()
    x["ATR14"] = _compute_atr14(x)

    ts_ser = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    open_ = pd.to_numeric(x["Open"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce")
    low = pd.to_numeric(x["Low"], errors="coerce")
    close = pd.to_numeric(x["Close"], errors="coerce")
    atr = pd.to_numeric(x["ATR14"], errors="coerce")

    good = ts_ser.notna() & open_.notna() & high.notna() & low.notna() & close.notna()
    if not bool(good.any()):
        return {"filled": False, "skip_reason": "bad_3m_data", "ict_score": 0.0, "ict_score_components": "", "fallback_used": 0}

    ts_ok = ts_ser[good].tolist()
    ts_ns = np.array([int(t.value) for t in ts_ok], dtype=np.int64)
    open_ = open_[good].to_numpy(dtype=float)
    high = high[good].to_numpy(dtype=float)
    low = low[good].to_numpy(dtype=float)
    close = close[good].to_numpy(dtype=float)
    atr = atr[good].fillna(0.0).to_numpy(dtype=float)

    n = len(ts_ns)
    if n == 0:
        return {"filled": False, "skip_reason": "no_3m_data", "ict_score": 0.0, "ict_score_components": "", "fallback_used": 0}

    score_weights = getattr(args, "_ict_score_weights", _parse_ict_score_weights(str(getattr(args, "ict_score_weights", ""))))
    ict_mode = str(getattr(args, "ict_mode", "rules")).strip().lower()
    score_threshold = float(getattr(args, "ict_score_threshold", 3))

    use_sweep = _as_bool(args.use_sweep)
    use_disp = _as_bool(args.use_displacement)
    use_fvg = _as_bool(args.use_fvg)
    use_ob = _as_bool(args.use_ob)
    use_killzones = _as_bool(args.use_killzones)
    enforce_killzone = bool(use_killzones and ict_mode == "rules")
    fallback = str(args.fallback).strip().lower()
    entry_mode = str(args.entry_mode).strip().lower()

    def _skip(reason: str, *, score: float, score_components: List[str], fallback_used: int = 0) -> Dict[str, Any]:
        return {
            "filled": False,
            "skip_reason": str(reason),
            "ict_score": float(score),
            "ict_score_components": ",".join(score_components),
            "fallback_used": int(fallback_used),
            "exit_reason": "no_fill",
        }

    sig_ns = int(_to_utc_ts(signal_time).value)
    sig_idx = int(np.searchsorted(ts_ns, sig_ns, side="left"))
    if sig_idx >= n:
        return _skip("no_bar_after_signal", score=0.0, score_components=[])

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="signal_anchor",
        decision_ts_ns=int(ts_ns[sig_idx]),
        feature_last_ts_ns=int(ts_ns[sig_idx]),
        note="signal aligned to first 3m candle at/after signal_time",
    )

    if _as_bool(args.use_volatility_filter):
        lb = max(1, int(args.atr_lookback_bars))
        min_bars = max(30, int(args.atr_min_bars))
        hist_start = max(0, sig_idx - lb)
        hist = atr[hist_start:sig_idx]
        hist = hist[np.isfinite(hist)]
        if hist.size >= min_bars:
            cut = float(np.percentile(hist, float(args.atr_spike_percentile)))
            if float(atr[sig_idx]) > cut:
                _debug_append(
                    debug_events,
                    signal_id=signal_id,
                    stage="volatility_gate",
                    decision_ts_ns=int(ts_ns[sig_idx]),
                    feature_last_ts_ns=int(ts_ns[sig_idx]),
                    note=f"atr={float(atr[sig_idx]):.8f} cut={cut:.8f} -> skip",
                )
                return _skip("volatility_spike", score=0.0, score_components=[])

    swing_low, swing_high = _detect_swings(low=low, high=high, k=int(args.swing_k))

    sweep_meta: Optional[Dict[str, Any]] = None
    if use_sweep:
        sweep_meta = _find_liquidity_sweep_long(
            low=low,
            close=close,
            swing_low=swing_low,
            start_idx=sig_idx,
            max_wait_bars=int(args.max_wait_bars),
            closeback_bars=int(args.sweep_closeback_bars),
            lookback_swings=int(args.sweep_lookback_swings),
            sweep_ticks=float(args.sweep_ticks),
            sweep_pct=float(args.sweep_pct),
            last_idx=n - 1,
        )
        if sweep_meta is None:
            sweep_end_idx = min(n - 1, sig_idx + max(1, int(args.max_wait_bars)))
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="sweep",
                decision_ts_ns=int(ts_ns[sweep_end_idx]),
                feature_last_ts_ns=int(ts_ns[sweep_end_idx]),
                note="not found",
            )
        else:
            conf = int(sweep_meta["confirm_idx"])
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="sweep",
                decision_ts_ns=int(ts_ns[conf]),
                feature_last_ts_ns=int(ts_ns[conf]),
                note=f"found ref_low={float(sweep_meta['ref_swing_low']):.8f}",
            )

    disp_start = int(sweep_meta["confirm_idx"]) if sweep_meta is not None else int(sig_idx)
    disp_meta: Optional[Dict[str, Any]] = None
    if use_disp:
        disp_meta = _find_displacement_long(
            open_=open_,
            high=high,
            close=close,
            atr=atr,
            swing_high=swing_high,
            start_idx=disp_start,
            max_confirm_bars=int(args.max_confirm_bars),
            body_atr_mult=float(args.disp_body_atr_mult),
            body_pct_thr=float(args.disp_body_pct),
            last_idx=n - 1,
        )
        if disp_meta is None:
            disp_end_idx = min(n - 1, disp_start + max(1, int(args.max_confirm_bars)))
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="displacement_mss",
                decision_ts_ns=int(ts_ns[disp_end_idx]),
                feature_last_ts_ns=int(ts_ns[disp_end_idx]),
                note="not found",
            )
        else:
            d = int(disp_meta["disp_idx"])
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="displacement_mss",
                decision_ts_ns=int(ts_ns[d]),
                feature_last_ts_ns=int(ts_ns[d]),
                note=f"found body={float(disp_meta['body']):.8f}",
            )

    scan_anchor = int(disp_meta["disp_idx"]) if disp_meta is not None else (int(sweep_meta["confirm_idx"]) if sweep_meta is not None else int(sig_idx))

    fvg_meta: Optional[Dict[str, Any]] = None
    if use_fvg:
        fvg_meta = _find_bullish_fvg(
            high=high,
            low=low,
            start_idx=scan_anchor,
            max_search_bars=int(args.max_confirm_bars),
            last_idx=n - 1,
        )
        if fvg_meta is None:
            fvg_end_idx = min(n - 1, scan_anchor + max(1, int(args.max_confirm_bars)))
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="fvg",
                decision_ts_ns=int(ts_ns[fvg_end_idx]),
                feature_last_ts_ns=int(ts_ns[fvg_end_idx]),
                note="not found",
            )
        else:
            fi = int(fvg_meta["fvg_idx"])
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="fvg",
                decision_ts_ns=int(ts_ns[fi]),
                feature_last_ts_ns=int(ts_ns[fi]),
                note=f"found zone=[{float(fvg_meta['zone_low']):.8f},{float(fvg_meta['zone_high']):.8f}]",
            )

    ob_meta: Optional[Dict[str, Any]] = None
    if use_ob and disp_meta is not None:
        ob_meta = _find_bullish_ob(
            open_=open_,
            close=close,
            disp_idx=int(disp_meta["disp_idx"]),
            start_idx=sig_idx,
        )
        if ob_meta is None:
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="ob",
                decision_ts_ns=int(ts_ns[int(disp_meta["disp_idx"])]),
                feature_last_ts_ns=int(ts_ns[int(disp_meta["disp_idx"])]),
                note="not found",
            )
        else:
            oi = int(ob_meta["ob_idx"])
            d = int(disp_meta["disp_idx"])
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="ob",
                decision_ts_ns=int(ts_ns[d]),
                feature_last_ts_ns=int(ts_ns[d]),
                note=f"found ob_idx={oi}",
            )

    candidate_entry_idx = min(n - 1, max(0, scan_anchor + 1))
    if entry_mode.startswith("fvg") and fvg_meta is not None:
        candidate_entry_idx = min(n - 1, int(fvg_meta["fvg_idx"]) + 1)

    killzone_ok = True
    if use_killzones:
        killzone_ok = _is_in_killzone(pd.to_datetime(int(ts_ns[candidate_entry_idx]), utc=True), killzones)
        _debug_append(
            debug_events,
            signal_id=signal_id,
            stage="killzone",
            decision_ts_ns=int(ts_ns[candidate_entry_idx]),
            feature_last_ts_ns=int(ts_ns[candidate_entry_idx]),
            note=f"killzone_ok={int(bool(killzone_ok))}",
        )

    components: Dict[str, bool] = {}
    if use_sweep:
        components["sweep"] = sweep_meta is not None
    if use_disp:
        components["displacement"] = disp_meta is not None
    if use_killzones:
        components["killzone"] = bool(killzone_ok)
    if use_fvg:
        components["fvg"] = fvg_meta is not None
    if use_ob:
        components["ob"] = ob_meta is not None

    ict_score = 0.0
    score_components: List[str] = []
    for name, ok in components.items():
        if bool(ok):
            w = float(score_weights.get(name, 1.0))
            ict_score += w
            score_components.append(name)

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="score_gate",
        decision_ts_ns=int(ts_ns[candidate_entry_idx]),
        feature_last_ts_ns=int(ts_ns[candidate_entry_idx]),
        note=f"mode={ict_mode} score={ict_score:.4f} threshold={score_threshold:.4f} components={','.join(score_components)}",
    )

    if ict_mode == "rules":
        if use_sweep and sweep_meta is None:
            return _skip("no_sweep", score=ict_score, score_components=score_components)
        if use_disp and disp_meta is None:
            return _skip("no_displacement", score=ict_score, score_components=score_components)
    else:
        if float(ict_score) < float(score_threshold):
            return _skip("score_below_threshold", score=ict_score, score_components=score_components)

    def _market_fill(idx: int, entry_type: str, reason_if_kz: str) -> Dict[str, Any]:
        i = min(max(0, int(idx)), n - 1)
        t = pd.to_datetime(int(ts_ns[i]), utc=True)
        if enforce_killzone and (not _is_in_killzone(t, killzones)):
            return {"filled": False, "skip_reason": reason_if_kz}
        px = float(open_[i])
        if not np.isfinite(px) or px <= 0:
            return {"filled": False, "skip_reason": "bad_entry_price"}
        _debug_append(
            debug_events,
            signal_id=signal_id,
            stage="entry_market",
            decision_ts_ns=int(ts_ns[i]),
            feature_last_ts_ns=int(ts_ns[i]),
            note=f"entry_type={entry_type}",
        )
        return {
            "filled": True,
            "entry_idx": int(i),
            "entry_time": t,
            "entry_price": float(px),
            "entry_type": entry_type,
        }

    fill: Dict[str, Any]
    fallback_used = 0

    if entry_mode == "market":
        fill = _market_fill(scan_anchor + 1, "market", "outside_killzone")
        if not fill.get("filled", False):
            return _skip(str(fill.get("skip_reason", "no_fill")), score=ict_score, score_components=score_components)
    else:
        limit_price: Optional[float] = None
        limit_start_idx: Optional[int] = None
        touched_outside = False
        limit_type = ""

        if entry_mode.startswith("fvg"):
            if use_fvg and fvg_meta is not None:
                if entry_mode == "fvg_mid":
                    limit_price = float(fvg_meta["zone_mid"])
                else:
                    limit_price = float(fvg_meta["zone_low"])
                limit_start_idx = int(fvg_meta["fvg_idx"] + 1)
                limit_type = "limit_fvg"
            else:
                if fallback == "market":
                    fallback_used = 1
                    fill = _market_fill(scan_anchor + int(args.entry_timeout_bars), "market_fallback", "outside_killzone")
                    if not fill.get("filled", False):
                        return _skip(str(fill.get("skip_reason", "no_fill")), score=ict_score, score_components=score_components, fallback_used=fallback_used)
                else:
                    return _skip("no_fvg", score=ict_score, score_components=score_components)

        elif entry_mode.startswith("ob"):
            if use_ob and ob_meta is not None:
                if entry_mode == "ob_mid":
                    limit_price = float(ob_meta["mid"])
                else:
                    limit_price = float(ob_meta["open"])
                limit_start_idx = int(scan_anchor + 1)
                limit_type = "limit_ob"
            else:
                if fallback == "market":
                    fallback_used = 1
                    fill = _market_fill(scan_anchor + int(args.entry_timeout_bars), "market_fallback", "outside_killzone")
                    if not fill.get("filled", False):
                        return _skip(str(fill.get("skip_reason", "no_fill")), score=ict_score, score_components=score_components, fallback_used=fallback_used)
                else:
                    return _skip("no_ob", score=ict_score, score_components=score_components)
        else:
            return _skip(f"unsupported_entry_mode:{entry_mode}", score=ict_score, score_components=score_components)

        if limit_price is not None and limit_start_idx is not None:
            timeout = min(max(1, int(args.entry_timeout_bars)), max(1, int(args.fvg_retrace_bars)))
            fill_idx, touched_outside = _find_limit_fill_long(
                ts=ts_ns,
                low=low,
                high=high,
                price=float(limit_price),
                start_idx=limit_start_idx,
                timeout_bars=timeout,
                use_killzones=enforce_killzone,
                killzones=killzones,
                last_idx=n - 1,
            )
            if fill_idx is None:
                if fallback == "market":
                    fallback_used = 1
                    fill = _market_fill(limit_start_idx + int(args.entry_timeout_bars), "market_fallback", "outside_killzone")
                    if not fill.get("filled", False):
                        reason = "outside_killzone" if touched_outside else str(fill.get("skip_reason", "no_fill"))
                        return _skip(reason, score=ict_score, score_components=score_components, fallback_used=fallback_used)
                else:
                    return _skip("outside_killzone" if touched_outside else "no_retrace", score=ict_score, score_components=score_components)
            else:
                t = pd.to_datetime(int(ts_ns[fill_idx]), utc=True)
                px = float(limit_price)
                _debug_append(
                    debug_events,
                    signal_id=signal_id,
                    stage="entry_limit_fill",
                    decision_ts_ns=int(ts_ns[fill_idx]),
                    feature_last_ts_ns=int(ts_ns[fill_idx]),
                    note=f"type={limit_type} limit={float(limit_price):.8f}",
                )
                fill = {
                    "filled": True,
                    "entry_idx": int(fill_idx),
                    "entry_time": t,
                    "entry_price": float(px),
                    "entry_type": str(limit_type),
                }

    if not fill.get("filled", False):
        return _skip(str(fill.get("skip_reason", "no_fill")), score=ict_score, score_components=score_components, fallback_used=fallback_used)

    entry_idx = int(fill["entry_idx"])
    entry_price = float(fill["entry_price"])
    entry_time = pd.to_datetime(fill["entry_time"], utc=True)
    if baseline_exit_time is not None and entry_time > _to_utc_ts(baseline_exit_time):
        return _skip("after_baseline_exit", score=ict_score, score_components=score_components, fallback_used=fallback_used)

    strategy_sl = float(entry_price * float(sl_mult))
    strategy_tp = float(entry_price * float(tp_mult))
    final_sl = float(strategy_sl)

    if _as_bool(args.tighten_sl_to_sweep) and sweep_meta is not None:
        swept_low = float(sweep_meta.get("swept_low", np.nan))
        if np.isfinite(swept_low):
            ict_sl = swept_low * (1.0 - float(args.sl_buffer_pct)) - float(args.sl_buffer_ticks)
            if np.isfinite(ict_sl):
                final_sl = max(float(strategy_sl), float(ict_sl))
                if final_sl >= entry_price:
                    final_sl = float(strategy_sl)

    out = _simulate_path_long(
        ts_ns=ts_ns,
        close=close,
        high=high,
        low=low,
        entry_idx=entry_idx,
        entry_price=entry_price,
        sl_price=final_sl,
        tp_price=strategy_tp,
        max_exit_ts_ns=_compute_eval_end_ns(
            entry_ts_ns=int(ts_ns[entry_idx]),
            eval_horizon_hours=float(eval_horizon_hours),
            baseline_exit_time=baseline_exit_time,
        ),
    )
    out["entry_type"] = str(fill.get("entry_type", "market"))
    out["fill_delay_minutes"] = float((pd.to_datetime(out["entry_time"], utc=True) - _to_utc_ts(signal_time)).total_seconds() / 60.0)
    out["skip_reason"] = ""
    out["ict_score"] = float(ict_score)
    out["ict_score_components"] = ",".join(score_components)
    out["fallback_used"] = int(fallback_used)

    if sweep_meta is not None:
        out["swept_low"] = float(sweep_meta.get("swept_low", np.nan))
        out["ref_swing_low"] = float(sweep_meta.get("ref_swing_low", np.nan))

    if disp_meta is not None:
        out["mss_level"] = float(disp_meta.get("mss_level", np.nan))

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="trade_complete",
        decision_ts_ns=int(pd.to_datetime(out["exit_time"], utc=True).value),
        feature_last_ts_ns=int(pd.to_datetime(out["exit_time"], utc=True).value),
        note=f"exit_reason={out.get('exit_reason','')}",
    )

    return out


def _simulate_exec_limit_long(
    *,
    df3m: pd.DataFrame,
    signal_time: pd.Timestamp,
    tp_mult: float,
    sl_mult: float,
    args: argparse.Namespace,
    debug_events: Optional[List[Dict[str, Any]]] = None,
    signal_id: str = "",
    baseline_exit_time: Optional[pd.Timestamp] = None,
    eval_horizon_hours: float = 12.0,
) -> Dict[str, Any]:
    if df3m.empty:
        return {"filled": False, "skip_reason": "no_3m_data", "vol_skip": 0}

    x = df3m.copy()
    x["ATR14"] = _compute_atr14(x)

    ts_ser = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    open_ = pd.to_numeric(x["Open"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce")
    low = pd.to_numeric(x["Low"], errors="coerce")
    close = pd.to_numeric(x["Close"], errors="coerce")
    atr = pd.to_numeric(x["ATR14"], errors="coerce")

    good = ts_ser.notna() & open_.notna() & high.notna() & low.notna() & close.notna() & atr.notna()
    if not bool(good.any()):
        return {"filled": False, "skip_reason": "bad_3m_data", "vol_skip": 0}

    ts_ok = ts_ser[good].tolist()
    ts_ns = np.array([int(t.value) for t in ts_ok], dtype=np.int64)
    open_np = open_[good].to_numpy(dtype=float)
    high_np = high[good].to_numpy(dtype=float)
    low_np = low[good].to_numpy(dtype=float)
    close_np = close[good].to_numpy(dtype=float)
    atr_np = atr[good].to_numpy(dtype=float)

    n = len(ts_ns)
    if n == 0:
        return {"filled": False, "skip_reason": "no_3m_data", "vol_skip": 0}

    sig_ns = int(_to_utc_ts(signal_time).value)
    sig_idx = int(np.searchsorted(ts_ns, sig_ns, side="left"))
    if sig_idx >= n:
        return {"filled": False, "skip_reason": "no_bar_after_signal", "vol_skip": 0}

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="signal_anchor",
        decision_ts_ns=int(ts_ns[sig_idx]),
        feature_last_ts_ns=int(ts_ns[sig_idx]),
        note="exec_limit signal anchor",
    )

    atr_ref_idx = max(0, sig_idx - 1)
    atr_ref = float(atr_np[atr_ref_idx])
    entry_ref = float(open_np[sig_idx])
    if not np.isfinite(entry_ref) or entry_ref <= 0.0 or (not np.isfinite(atr_ref)):
        return {"filled": False, "skip_reason": "bad_entry_ref_or_atr", "vol_skip": 0}

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="atr_snapshot",
        decision_ts_ns=int(ts_ns[sig_idx]),
        feature_last_ts_ns=int(ts_ns[atr_ref_idx]),
        note=f"atr_ref={atr_ref:.8f}",
    )

    lb = 7 * 24 * 20  # 7 days of 3m bars
    hist_start = max(0, atr_ref_idx - lb)
    hist = np.asarray(atr_np[hist_start : atr_ref_idx + 1], dtype=float)
    hist = hist[np.isfinite(hist)]
    atr_z = 0.0
    atr_pct = float("nan")
    if hist.size >= 100:
        mean = float(np.nanmean(hist))
        std = float(np.nanstd(hist))
        atr_z = float((atr_ref - mean) / std) if std > 1e-12 else 0.0
        atr_pct = float(100.0 * np.mean(hist <= atr_ref))

    vol_skip = 0
    if _as_bool(getattr(args, "use_vol_gate", 0)):
        z_thr = float(getattr(args, "vol_z_thr", 2.5))
        p_thr = float(getattr(args, "vol_p_thr", 95.0))
        if np.isfinite(atr_pct) and ((atr_z > z_thr) or (atr_pct > p_thr)):
            vol_skip = 1
            _debug_append(
                debug_events,
                signal_id=signal_id,
                stage="vol_gate",
                decision_ts_ns=int(ts_ns[sig_idx]),
                feature_last_ts_ns=int(ts_ns[atr_ref_idx]),
                note=f"skip z={atr_z:.4f} pct={atr_pct:.2f} z_thr={z_thr:.2f} p_thr={p_thr:.2f}",
            )
            return {
                "filled": False,
                "skip_reason": "volatility_gate",
                "vol_skip": int(vol_skip),
                "entry_ref_price": float(entry_ref),
                "atr_ref": float(atr_ref),
                "atr_zscore": float(atr_z),
                "atr_percentile": float(atr_pct),
            }
        _debug_append(
            debug_events,
            signal_id=signal_id,
            stage="vol_gate",
            decision_ts_ns=int(ts_ns[sig_idx]),
            feature_last_ts_ns=int(ts_ns[atr_ref_idx]),
            note=f"pass z={atr_z:.4f} pct={atr_pct:.2f}",
        )

    if _as_bool(getattr(args, "use_micro_panic", 0)):
        prev_idx = max(0, sig_idx - 1)
        rr = float(high_np[prev_idx] - low_np[prev_idx])
        atr_prev = float(atr_np[prev_idx])
        close_pos = float((close_np[prev_idx] - low_np[prev_idx]) / rr) if rr > 1e-12 else 1.0
        panic_mult = float(getattr(args, "panic_mult", 2.5))
        panic = np.isfinite(rr) and np.isfinite(atr_prev) and rr > panic_mult * atr_prev and close_pos <= 0.2
        _debug_append(
            debug_events,
            signal_id=signal_id,
            stage="micro_panic",
            decision_ts_ns=int(ts_ns[sig_idx]),
            feature_last_ts_ns=int(ts_ns[prev_idx]),
            note=f"panic={int(bool(panic))} range={rr:.8f} atr={atr_prev:.8f} close_pos={close_pos:.4f}",
        )
        if panic:
            return {
                "filled": False,
                "skip_reason": "micro_panic",
                "vol_skip": int(vol_skip),
                "entry_ref_price": float(entry_ref),
                "atr_ref": float(atr_ref),
                "atr_zscore": float(atr_z),
                "atr_percentile": float(atr_pct),
            }

    k_mult = 1.0
    if _as_bool(getattr(args, "exec_adaptive_k", 0)):
        k_mult = float(np.clip(1.0 + 0.15 * float(atr_z), 0.7, 1.4))
        _debug_append(
            debug_events,
            signal_id=signal_id,
            stage="adaptive_k",
            decision_ts_ns=int(ts_ns[sig_idx]),
            feature_last_ts_ns=int(ts_ns[atr_ref_idx]),
            note=f"atr_z={atr_z:.4f} k_mult={k_mult:.4f}",
        )

    timeout_bars = max(1, int(getattr(args, "exec_timeout_bars", 20)))
    i0 = int(sig_idx)
    use_ladder = _as_bool(getattr(args, "exec_use_ladder", 0))
    use_two_stage = _as_bool(getattr(args, "exec_two_stage", 0))
    base_k = float(getattr(args, "exec_k", 0.5)) * float(k_mult)
    k1 = float(getattr(args, "exec_k1", 0.3)) * float(k_mult)
    k2 = float(getattr(args, "exec_k2", 0.8)) * float(k_mult)
    if k2 <= k1:
        k2 = k1 + 1e-6

    l1 = float(entry_ref - base_k * atr_ref)
    l2 = float("nan")

    def _first_limit_fill(start_idx: int, end_idx: int, px: float, tag: str) -> Tuple[Optional[int], float, str]:
        if (not np.isfinite(px)) or start_idx > end_idx:
            return None, float("nan"), ""
        for j in range(int(start_idx), int(end_idx) + 1):
            lj = float(low_np[j])
            if np.isfinite(lj) and lj <= float(px):
                return int(j), float(px), str(tag)
        return None, float("nan"), ""

    fill_idx: Optional[int] = None
    fill_px = float("nan")
    fill_type = "limit"
    fallback = str(getattr(args, "exec_fallback", "market")).strip().lower()
    fallback_used = 0
    fallback_from_idx = i0 + timeout_bars + 1

    if use_two_stage:
        s1 = max(0, i0)
        s1_end = min(n - 1, s1 + max(1, int(getattr(args, "exec_stage1_bars", 10))) - 1)
        s2_bars = max(1, int(getattr(args, "exec_stage2_bars", 10)))
        move_away_thr = float(getattr(args, "exec_move_away_thr", 1.0))
        l1 = float(entry_ref - k1 * atr_ref)
        l2 = float(entry_ref - k2 * atr_ref)

        fill_idx, fill_px, fill_type = _first_limit_fill(s1, s1_end, l1, "limit_stage1")
        if fill_idx is None:
            moved_away = False
            if s1_end >= s1:
                moved_away = bool(np.nanmax(high_np[s1 : s1_end + 1]) - entry_ref >= move_away_thr * atr_ref)
            s2 = s1_end + 1
            if moved_away:
                fallback_from_idx = int(s2)
                _debug_append(
                    debug_events,
                    signal_id=signal_id,
                    stage="two_stage_move_away",
                    decision_ts_ns=int(ts_ns[min(s1_end, n - 1)]),
                    feature_last_ts_ns=int(ts_ns[min(s1_end, n - 1)]),
                    note=f"triggered thr={move_away_thr:.4f}",
                )
                if str(getattr(args, "exec_fallback", "market")).strip().lower() == "market":
                    if s2 < n:
                        fill_idx = int(s2)
                        fill_px = float(open_np[fill_idx])
                        fill_type = "market_early_fallback"
                        fallback_used = 1
                    else:
                        fill_idx = None
                else:
                    return {
                        "filled": False,
                        "skip_reason": "move_away",
                        "vol_skip": int(vol_skip),
                        "entry_ref_price": float(entry_ref),
                        "atr_ref": float(atr_ref),
                        "atr_zscore": float(atr_z),
                        "atr_percentile": float(atr_pct),
                    }
            else:
                s2_end = min(n - 1, s2 + s2_bars - 1)
                fill_idx, fill_px, fill_type = _first_limit_fill(s2, s2_end, l2, "limit_stage2")
                if fill_idx is None:
                    timeout_market_idx = s2_end + 1
                    fallback_from_idx = int(timeout_market_idx)
                    if str(getattr(args, "exec_fallback", "market")).strip().lower() == "market":
                        if timeout_market_idx < n:
                            fill_idx = int(timeout_market_idx)
                            fill_px = float(open_np[fill_idx])
                            fill_type = "market_fallback"
                            fallback_used = 1
                        else:
                            fill_idx = None
                    else:
                        return {
                            "filled": False,
                            "skip_reason": "timeout_no_fill",
                            "vol_skip": int(vol_skip),
                            "entry_ref_price": float(entry_ref),
                            "atr_ref": float(atr_ref),
                            "atr_zscore": float(atr_z),
                            "atr_percentile": float(atr_pct),
                        }
    else:
        i1 = min(n - 1, i0 + timeout_bars)
        fallback_from_idx = int(i1 + 1)
        if use_ladder:
            l1 = float(entry_ref - k1 * atr_ref)
            l2 = float(entry_ref - k2 * atr_ref)
        else:
            l2 = float("nan")

        for i in range(i0, i1 + 1):
            li = float(low_np[i])
            if not np.isfinite(li):
                continue
            if use_ladder:
                if li <= l2:
                    fill_idx = int(i)
                    fill_px = float(l2)
                    fill_type = "limit_l2"
                    break
                if li <= l1:
                    fill_idx = int(i)
                    fill_px = float(l1)
                    fill_type = "limit_l1"
                    break
            else:
                if li <= l1:
                    fill_idx = int(i)
                    fill_px = float(l1)
                    fill_type = "limit"
                    break

    if fill_idx is None:
        if fallback == "market":
            m_idx = int(fallback_from_idx)
            if m_idx >= n:
                return {
                    "filled": False,
                    "skip_reason": "timeout_no_bar_for_market_fallback",
                    "vol_skip": int(vol_skip),
                    "entry_ref_price": float(entry_ref),
                    "atr_ref": float(atr_ref),
                    "atr_zscore": float(atr_z),
                    "atr_percentile": float(atr_pct),
                }
            fill_idx = int(m_idx)
            fill_px = float(open_np[fill_idx])
            fill_type = "market_fallback"
            fallback_used = 1
        else:
            return {
                "filled": False,
                "skip_reason": "timeout_no_fill",
                "vol_skip": int(vol_skip),
                "entry_ref_price": float(entry_ref),
                "atr_ref": float(atr_ref),
                "atr_zscore": float(atr_z),
                "atr_percentile": float(atr_pct),
            }

    if not np.isfinite(fill_px) or fill_px <= 0.0:
        return {
            "filled": False,
            "skip_reason": "bad_entry_price",
            "vol_skip": int(vol_skip),
            "entry_ref_price": float(entry_ref),
            "atr_ref": float(atr_ref),
            "atr_zscore": float(atr_z),
            "atr_percentile": float(atr_pct),
        }

    fill_time = pd.to_datetime(int(ts_ns[int(fill_idx)]), utc=True)
    if baseline_exit_time is not None and fill_time > _to_utc_ts(baseline_exit_time):
        return {
            "filled": False,
            "skip_reason": "after_baseline_exit",
            "vol_skip": int(vol_skip),
            "entry_ref_price": float(entry_ref),
            "atr_ref": float(atr_ref),
            "atr_zscore": float(atr_z),
            "atr_percentile": float(atr_pct),
        }

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="entry_fill",
        decision_ts_ns=int(ts_ns[int(fill_idx)]),
        feature_last_ts_ns=int(ts_ns[int(fill_idx)]),
        note=f"type={fill_type} px={fill_px:.8f}",
    )

    strategy_sl = float(fill_px * float(sl_mult))
    strategy_tp = float(fill_px * float(tp_mult))
    out = _simulate_path_long(
        ts_ns=ts_ns,
        close=close_np,
        high=high_np,
        low=low_np,
        entry_idx=int(fill_idx),
        entry_price=float(fill_px),
        sl_price=float(strategy_sl),
        tp_price=float(strategy_tp),
        max_exit_ts_ns=_compute_eval_end_ns(
            entry_ts_ns=int(ts_ns[int(fill_idx)]),
            eval_horizon_hours=float(eval_horizon_hours),
            baseline_exit_time=baseline_exit_time,
        ),
    )
    out["entry_type"] = str(fill_type)
    out["fill_delay_minutes"] = float((fill_time - _to_utc_ts(signal_time)).total_seconds() / 60.0)
    out["skip_reason"] = ""
    out["fallback_used"] = int(fallback_used)
    out["vol_skip"] = int(vol_skip)
    out["entry_ref_price"] = float(entry_ref)
    out["atr_ref"] = float(atr_ref)
    out["atr_zscore"] = float(atr_z)
    out["atr_percentile"] = float(atr_pct)
    out["k_effective"] = float(base_k)
    out["limit_price"] = float(l1)
    out["limit_price_l2"] = float(l2) if np.isfinite(l2) else float("nan")

    _debug_append(
        debug_events,
        signal_id=signal_id,
        stage="trade_complete",
        decision_ts_ns=int(pd.to_datetime(out["exit_time"], utc=True).value),
        feature_last_ts_ns=int(pd.to_datetime(out["exit_time"], utc=True).value),
        note=f"exec_limit exit_reason={out.get('exit_reason','')}",
    )
    return out


@dataclass
class SignalRow:
    signal_id: str
    signal_time: pd.Timestamp
    cycle: int
    tp_mult: float
    sl_mult: float
    signal_open_1h: float
    stop_distance_pct: float = float("nan")
    atr_1h: float = float("nan")
    atr_percentile_1h: float = float("nan")
    trend_up_1h: int = 0


def _pick_ema_cols(df: pd.DataFrame, fast_hint: str = "", slow_hint: str = "") -> Tuple[str, str]:
    fh = str(fast_hint).strip()
    sh = str(slow_hint).strip()
    if fh in df.columns and sh in df.columns:
        return fh, sh
    if "EMA_50" in df.columns and "EMA_120" in df.columns:
        return "EMA_50", "EMA_120"
    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        return "EMA_20", "EMA_50"
    cands = [c for c in df.columns if str(c).upper().startswith("EMA_")]
    if len(cands) < 2:
        return "", ""

    def _key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**9

    cands = sorted(cands, key=_key)
    return cands[0], cands[-1]


def _trailing_percentile(arr: np.ndarray, lookback: int, min_hist: int = 30) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    lb = max(1, int(lookback))
    for i in range(n):
        hist = arr[max(0, i - lb) : i]
        hist = hist[np.isfinite(hist)]
        if hist.size < int(min_hist) or not np.isfinite(arr[i]):
            continue
        out[i] = float(100.0 * np.mean(hist <= arr[i]))
    return out


def _build_1h_signals(
    df_1h: pd.DataFrame,
    p: Dict[str, Any],
    max_signals: int,
    order: str,
    gate_cfg: Optional[Dict[str, Any]] = None,
) -> List[SignalRow]:
    gate_cfg = dict(gate_cfg or {})
    use_vol_regime_gate = _as_bool(gate_cfg.get("use_vol_regime_gate", 0))
    vol_regime_max_percentile = float(gate_cfg.get("vol_regime_max_percentile", 90.0))
    vol_regime_lookback_bars = int(gate_cfg.get("vol_regime_lookback_bars", 2160))
    use_trend_gate = _as_bool(gate_cfg.get("use_trend_gate", 0))
    trend_min_slope = float(gate_cfg.get("trend_min_slope", 0.0))
    stop_distance_min_pct = float(gate_cfg.get("stop_distance_min_pct", 0.0))
    trend_fast_col_hint = str(gate_cfg.get("trend_fast_col", ""))
    trend_slow_col_hint = str(gate_cfg.get("trend_slow_col", ""))

    df = ga_long._ensure_indicators(df_1h.copy(), p)
    sig = ga_long.build_entry_signal(df, p, assume_prepared=True)
    cycles = ga_long._shift_cycles(
        ga_long.compute_cycles(df, p),
        shift=int(p.get("cycle_shift", 1)),
        fill=int(p.get("cycle_fill", 2)),
    )

    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    op = pd.to_numeric(df["Open"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df.get("ATR", np.nan), errors="coerce").to_numpy(dtype=float)
    atr_pct = _trailing_percentile(atr, lookback=vol_regime_lookback_bars, min_hist=30)
    ema_fast_col, ema_slow_col = _pick_ema_cols(df, fast_hint=trend_fast_col_hint, slow_hint=trend_slow_col_hint)
    if ema_fast_col and ema_slow_col:
        ema_fast = pd.to_numeric(df[ema_fast_col], errors="coerce").to_numpy(dtype=float)
        ema_slow = pd.to_numeric(df[ema_slow_col], errors="coerce").to_numpy(dtype=float)
        ema_slope = np.concatenate([[np.nan], np.diff(ema_fast)])
    else:
        ema_fast = np.full(len(df), np.nan, dtype=float)
        ema_slow = np.full(len(df), np.nan, dtype=float)
        ema_slope = np.full(len(df), np.nan, dtype=float)

    rows: List[SignalRow] = []
    for i, flag in enumerate(np.asarray(sig, dtype=bool).tolist()):
        if not flag:
            continue
        cyc = int(cycles[i])
        tp_mult = float(p["tp_mult_by_cycle"][cyc])
        sl_mult = float(p["sl_mult_by_cycle"][cyc])
        stop_distance_pct = float(max(0.0, 1.0 - sl_mult))
        atr_i = float(atr[i]) if np.isfinite(atr[i]) else float("nan")
        atr_pct_i = float(atr_pct[i]) if np.isfinite(atr_pct[i]) else float("nan")
        trend_up = int(
            np.isfinite(ema_fast[i])
            and np.isfinite(ema_slow[i])
            and np.isfinite(ema_slope[i])
            and (ema_fast[i] > ema_slow[i])
            and (ema_slope[i] > trend_min_slope)
        )

        if use_vol_regime_gate and np.isfinite(atr_pct_i) and atr_pct_i > vol_regime_max_percentile:
            continue
        if use_trend_gate and trend_up != 1:
            continue
        if stop_distance_min_pct > 0.0 and stop_distance_pct < stop_distance_min_pct:
            continue

        t = pd.to_datetime(ts.iloc[i], utc=True)
        rows.append(
            SignalRow(
                signal_id=f"sig_raw_{i:06d}",
                signal_time=t,
                cycle=cyc,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                signal_open_1h=float(op[i]),
                stop_distance_pct=stop_distance_pct,
                atr_1h=atr_i,
                atr_percentile_1h=atr_pct_i,
                trend_up_1h=trend_up,
            )
        )

    if order.lower() == "latest":
        rows = sorted(rows, key=lambda r: r.signal_time, reverse=True)
    else:
        rows = sorted(rows, key=lambda r: r.signal_time)

    if max_signals > 0:
        rows = rows[: int(max_signals)]

    rows = sorted(rows, key=lambda r: r.signal_time)
    out: List[SignalRow] = []
    for i, r in enumerate(rows, start=1):
        out.append(
            SignalRow(
                signal_id=f"sig_{i:05d}",
                signal_time=r.signal_time,
                cycle=r.cycle,
                tp_mult=r.tp_mult,
                sl_mult=r.sl_mult,
                signal_open_1h=r.signal_open_1h,
                stop_distance_pct=r.stop_distance_pct,
                atr_1h=r.atr_1h,
                atr_percentile_1h=r.atr_percentile_1h,
                trend_up_1h=int(r.trend_up_1h),
            )
        )
    return out


def _emit_summary_md(
    *,
    path: Path,
    symbol: str,
    rank: int,
    best_csv: Path,
    params_file: Path,
    signals_csv: Path,
    diag_csv: Path,
    df: pd.DataFrame,
) -> None:
    n = int(len(df))
    def _num_series(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(n, dtype=float))

    baseline_entries = int(_num_series("baseline_filled").astype(int).sum()) if n else 0
    ict_entries = int(_num_series("ict_filled").astype(int).sum()) if n else 0
    b_valid = _num_series("baseline_valid_for_metrics").astype(int) if "baseline_valid_for_metrics" in df.columns else pd.Series(np.ones(n, dtype=int))
    i_valid = (
        _num_series("exec_valid_for_metrics").astype(int)
        if "exec_valid_for_metrics" in df.columns
        else (_num_series("ict_valid_for_metrics").astype(int) if "ict_valid_for_metrics" in df.columns else pd.Series(np.ones(n, dtype=int)))
    )
    b_valid_entries = int(((_num_series("baseline_filled").astype(int) == 1) & (b_valid == 1)).sum()) if n else 0
    i_valid_entries = int(((_num_series("ict_filled").astype(int) == 1) & (i_valid == 1)).sum()) if n else 0

    b_sl_hits = int(_num_series("baseline_sl_hit").astype(int).sum()) if n else 0
    i_sl_hits = int(_num_series("ict_sl_hit").astype(int).sum()) if n else 0
    b_tp_hits = int(_num_series("baseline_tp_hit").astype(int).sum()) if n else 0
    i_tp_hits = int(_num_series("ict_tp_hit").astype(int).sum()) if n else 0

    b_sl_rate = float((((_num_series("baseline_sl_hit").astype(int) == 1) & (_num_series("baseline_filled").astype(int) == 1) & (b_valid == 1)).sum()) / b_valid_entries) if b_valid_entries > 0 else float("nan")
    i_sl_rate = float((((_num_series("ict_sl_hit").astype(int) == 1) & (_num_series("ict_filled").astype(int) == 1) & (i_valid == 1)).sum()) / i_valid_entries) if i_valid_entries > 0 else float("nan")
    b_tp_rate = float((((_num_series("baseline_tp_hit").astype(int) == 1) & (_num_series("baseline_filled").astype(int) == 1) & (b_valid == 1)).sum()) / b_valid_entries) if b_valid_entries > 0 else float("nan")
    i_tp_rate = float((((_num_series("ict_tp_hit").astype(int) == 1) & (_num_series("ict_filled").astype(int) == 1) & (i_valid == 1)).sum()) / i_valid_entries) if i_valid_entries > 0 else float("nan")

    skip_rate = 1.0 - (ict_entries / n) if n > 0 else float("nan")
    med_entry_improvement = float(pd.to_numeric(df["entry_price_delta_pct"], errors="coerce").median()) if (n and "entry_price_delta_pct" in df.columns) else float("nan")

    b_mae_med = float(pd.to_numeric(df["baseline_mae_pct"], errors="coerce").median()) if (n and "baseline_mae_pct" in df.columns) else float("nan")
    i_mae_med = float(pd.to_numeric(df["ict_mae_pct"], errors="coerce").median()) if (n and "ict_mae_pct" in df.columns) else float("nan")
    b_mfe_med = float(pd.to_numeric(df["baseline_mfe_pct"], errors="coerce").median()) if (n and "baseline_mfe_pct" in df.columns) else float("nan")
    i_mfe_med = float(pd.to_numeric(df["ict_mfe_pct"], errors="coerce").median()) if (n and "ict_mfe_pct" in df.columns) else float("nan")

    raw_skip = df["ict_skip_reason"].tolist() if "ict_skip_reason" in df.columns else []
    skip_reasons = [str(x).strip() for x in raw_skip if pd.notna(x) and str(x).strip()]
    hist = Counter(skip_reasons)
    top3 = hist.most_common(3)

    b_fill_mask = (_num_series("baseline_filled").astype(int) == 1) & (b_valid == 1)
    i_fill_mask = (_num_series("ict_filled").astype(int) == 1) & (i_valid == 1)
    b_gross = _num_series("baseline_pnl_gross_pct") if "baseline_pnl_gross_pct" in df.columns else _num_series("baseline_pnl_pct")
    i_gross = _num_series("exec_pnl_gross_pct") if "exec_pnl_gross_pct" in df.columns else (_num_series("ict_pnl_gross_pct") if "ict_pnl_gross_pct" in df.columns else _num_series("ict_pnl_pct"))
    b_net = _num_series("baseline_pnl_net_pct") if "baseline_pnl_net_pct" in df.columns else b_gross
    i_net = _num_series("exec_pnl_net_pct") if "exec_pnl_net_pct" in df.columns else (_num_series("ict_pnl_net_pct") if "ict_pnl_net_pct" in df.columns else i_gross)
    b_exp_net = float(b_net[b_fill_mask].sum() / b_valid_entries) if b_valid_entries > 0 and b_net[b_fill_mask].notna().any() else float("nan")
    i_exp_net = float(i_net[i_fill_mask].sum() / i_valid_entries) if i_valid_entries > 0 and i_net[i_fill_mask].notna().any() else float("nan")
    i_liq_col = "exec_fill_liquidity_type" if "exec_fill_liquidity_type" in df.columns else ("ict_fill_liquidity_type" if "ict_fill_liquidity_type" in df.columns else "")
    if i_liq_col:
        i_liq = df[i_liq_col].fillna("").astype(str).str.lower()
        i_taker_share = float(((i_liq == "taker") & i_fill_mask).sum() / max(1, i_valid_entries)) if i_valid_entries > 0 else float("nan")
    else:
        i_taker_share = float("nan")

    lines: List[str] = []
    lines.append(f"# 3m Execution Layer Summary: {symbol}")
    lines.append("")
    lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    lines.append(f"- Rank requested: {int(rank)}")
    lines.append(f"- best_by_symbol.csv: `{best_csv}`")
    lines.append(f"- params_file: `{params_file}`")
    lines.append(f"- 1h signals CSV: `{signals_csv}`")
    lines.append(f"- diagnostics CSV: `{diag_csv}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Signal count: {n}")
    lines.append(f"- Baseline entries: {baseline_entries}")
    lines.append(f"- ICT entries: {ict_entries}")
    lines.append(f"- ICT skip rate: {skip_rate:.4f}" if np.isfinite(skip_rate) else "- ICT skip rate: n/a")
    lines.append("")
    lines.append("## Outcome Rates")
    lines.append("")
    lines.append(f"- Baseline SL hit rate: {b_sl_rate:.4f}" if np.isfinite(b_sl_rate) else "- Baseline SL hit rate: n/a")
    lines.append(f"- ICT SL hit rate: {i_sl_rate:.4f}" if np.isfinite(i_sl_rate) else "- ICT SL hit rate: n/a")
    lines.append(f"- Baseline TP hit rate: {b_tp_rate:.4f}" if np.isfinite(b_tp_rate) else "- Baseline TP hit rate: n/a")
    lines.append(f"- ICT TP hit rate: {i_tp_rate:.4f}" if np.isfinite(i_tp_rate) else "- ICT TP hit rate: n/a")
    lines.append("")
    lines.append("## Costs / Net")
    lines.append("")
    lines.append(f"- Baseline expectancy_net (%): {b_exp_net:.6f}" if np.isfinite(b_exp_net) else "- Baseline expectancy_net (%): n/a")
    lines.append(f"- EXEC expectancy_net (%): {i_exp_net:.6f}" if np.isfinite(i_exp_net) else "- EXEC expectancy_net (%): n/a")
    lines.append(f"- EXEC taker_share: {i_taker_share:.4f}" if np.isfinite(i_taker_share) else "- EXEC taker_share: n/a")
    lines.append("")
    lines.append("## Entry / Excursion")
    lines.append("")
    lines.append(
        f"- Median entry improvement vs baseline: {med_entry_improvement:.6f}"
        if np.isfinite(med_entry_improvement)
        else "- Median entry improvement vs baseline: n/a"
    )
    lines.append(f"- Median MAE% baseline vs ICT: {b_mae_med:.6f} vs {i_mae_med:.6f}" if np.isfinite(b_mae_med) or np.isfinite(i_mae_med) else "- Median MAE% baseline vs ICT: n/a")
    lines.append(f"- Median MFE% baseline vs ICT: {b_mfe_med:.6f} vs {i_mfe_med:.6f}" if np.isfinite(b_mfe_med) or np.isfinite(i_mfe_med) else "- Median MFE% baseline vs ICT: n/a")
    lines.append("")
    lines.append("## Top Skip Reasons")
    lines.append("")
    if top3:
        for k, v in top3:
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- none")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _emit_audit_summary_md(
    *,
    path: Path,
    symbol: str,
    exec_mode: str,
    df: pd.DataFrame,
    eval_horizon_hours: float,
) -> Dict[str, Any]:
    n = int(len(df))

    def _num(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(np.full(n, default, dtype=float))

    baseline_filled = _num("baseline_filled").astype(int)
    baseline_sl = _num("baseline_sl_hit").astype(int)
    baseline_tp = _num("baseline_tp_hit").astype(int)
    b_inv_stop = _num("baseline_invalid_stop_geometry").astype(int)
    b_inv_tp = _num("baseline_invalid_tp_geometry").astype(int)
    b_same_bar = _num("baseline_same_bar_hit").astype(int)
    b_valid = ((b_inv_stop == 0) & (b_inv_tp == 0))

    exec_filled = _num("exec_filled").astype(int) if "exec_filled" in df.columns else _num("ict_filled").astype(int)
    exec_sl = _num("exec_sl_hit").astype(int) if "exec_sl_hit" in df.columns else _num("ict_sl_hit").astype(int)
    exec_tp = _num("exec_tp_hit").astype(int) if "exec_tp_hit" in df.columns else _num("ict_tp_hit").astype(int)
    e_inv_stop = _num("exec_invalid_stop_geometry").astype(int) if "exec_invalid_stop_geometry" in df.columns else _num("invalid_stop_geometry").astype(int)
    e_inv_tp = _num("exec_invalid_tp_geometry").astype(int) if "exec_invalid_tp_geometry" in df.columns else _num("invalid_tp_geometry").astype(int)
    e_same_bar = _num("exec_same_bar_hit").astype(int) if "exec_same_bar_hit" in df.columns else _num("same_bar_hit").astype(int)
    e_valid = ((e_inv_stop == 0) & (e_inv_tp == 0))

    b_entries_valid = int(((baseline_filled == 1) & b_valid).sum())
    e_entries_valid = int(((exec_filled == 1) & e_valid).sum())
    b_sl_rate = float(((baseline_sl == 1) & (baseline_filled == 1) & b_valid).sum() / b_entries_valid) if b_entries_valid > 0 else float("nan")
    e_sl_rate = float(((exec_sl == 1) & (exec_filled == 1) & e_valid).sum() / e_entries_valid) if e_entries_valid > 0 else float("nan")
    b_tp_rate = float(((baseline_tp == 1) & (baseline_filled == 1) & b_valid).sum() / b_entries_valid) if b_entries_valid > 0 else float("nan")
    e_tp_rate = float(((exec_tp == 1) & (exec_filled == 1) & e_valid).sum() / e_entries_valid) if e_entries_valid > 0 else float("nan")

    out_lines: List[str] = []
    out_lines.append(f"# Execution Audit Summary: {symbol}")
    out_lines.append("")
    out_lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    out_lines.append(f"- exec_mode: `{exec_mode}`")
    out_lines.append(f"- signals_total: {n}")
    out_lines.append(f"- eval_horizon_hours_default: {float(eval_horizon_hours):.2f}")
    out_lines.append("")
    out_lines.append("## Geometry / Trigger Flags")
    out_lines.append("")
    out_lines.append(f"- baseline_invalid_stop_geometry: {int(b_inv_stop.sum())}")
    out_lines.append(f"- baseline_invalid_tp_geometry: {int(b_inv_tp.sum())}")
    out_lines.append(f"- baseline_same_bar_hit: {int(b_same_bar.sum())}")
    out_lines.append(f"- exec_invalid_stop_geometry: {int(e_inv_stop.sum())}")
    out_lines.append(f"- exec_invalid_tp_geometry: {int(e_inv_tp.sum())}")
    out_lines.append(f"- exec_same_bar_hit: {int(e_same_bar.sum())}")
    out_lines.append("")
    out_lines.append("## Rates (Invalid Geometry Removed)")
    out_lines.append("")
    out_lines.append(f"- baseline_entries_valid: {b_entries_valid}")
    out_lines.append(f"- exec_entries_valid: {e_entries_valid}")
    out_lines.append(f"- baseline_sl_hit_rate_valid: {b_sl_rate:.6f}" if np.isfinite(b_sl_rate) else "- baseline_sl_hit_rate_valid: n/a")
    out_lines.append(f"- exec_sl_hit_rate_valid: {e_sl_rate:.6f}" if np.isfinite(e_sl_rate) else "- exec_sl_hit_rate_valid: n/a")
    out_lines.append(f"- baseline_tp_hit_rate_valid: {b_tp_rate:.6f}" if np.isfinite(b_tp_rate) else "- baseline_tp_hit_rate_valid: n/a")
    out_lines.append(f"- exec_tp_hit_rate_valid: {e_tp_rate:.6f}" if np.isfinite(e_tp_rate) else "- exec_tp_hit_rate_valid: n/a")

    top_examples_str = ""
    if np.isfinite(e_sl_rate) and e_sl_rate >= 0.98 and e_entries_valid > 0:
        fill_col_name = "exec_filled" if "exec_filled" in df.columns else ("ict_filled" if "ict_filled" in df.columns else "")
        sl_col_name = "exec_sl_hit" if "exec_sl_hit" in df.columns else ("ict_sl_hit" if "ict_sl_hit" in df.columns else "")
        fill_s = pd.to_numeric(df[fill_col_name], errors="coerce").fillna(0).astype(int) if fill_col_name else pd.Series(np.zeros(n, dtype=int))
        sl_s = pd.to_numeric(df[sl_col_name], errors="coerce").fillna(0).astype(int) if sl_col_name else pd.Series(np.zeros(n, dtype=int))
        cand = df[(fill_s == 1) & (sl_s == 1)].copy()
        keep_cols = [
            "signal_id",
            "signal_time",
            "baseline_entry_price",
            "baseline_sl",
            "baseline_tp",
            "exec_entry_price",
            "exec_sl",
            "exec_tp",
            "exec_entry_type",
            "exec_exit_reason",
            "exec_skip_reason",
            "exec_fill_delay_min",
        ]
        keep_cols = [c for c in keep_cols if c in cand.columns]
        top20 = cand[keep_cols].head(20)
        if not top20.empty:
            top_examples_str = top20.to_string(index=False)
            print("WARNING: exec SL-hit rate >= 0.98 after geometry checks; top 20 examples:", flush=True)
            print(top_examples_str, flush=True)
            out_lines.append("")
            out_lines.append("## Top 20 SL Examples")
            out_lines.append("")
            out_lines.append("```text")
            out_lines.append(top_examples_str)
            out_lines.append("```")

    path.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")
    return {
        "baseline_entries_valid": b_entries_valid,
        "exec_entries_valid": e_entries_valid,
        "baseline_sl_hit_rate_valid": b_sl_rate,
        "exec_sl_hit_rate_valid": e_sl_rate,
        "top_examples_printed": int(bool(top_examples_str)),
    }


def _pnl_pct_from_entry_exit(entry_price: Any, exit_price: Any) -> float:
    e = float(pd.to_numeric(pd.Series([entry_price]), errors="coerce").iloc[0])
    x = float(pd.to_numeric(pd.Series([exit_price]), errors="coerce").iloc[0])
    if not np.isfinite(e) or not np.isfinite(x) or e <= 0.0:
        return float("nan")
    return float((x / e) - 1.0)


def _liquidity_type_from_entry_type(entry_type: Any) -> str:
    et = str(entry_type).strip().lower()
    if not et:
        return ""
    if et.startswith("limit"):
        return "maker"
    return "taker"


def _costed_pnl_long(
    *,
    entry_price: Any,
    exit_price: Any,
    entry_liquidity_type: str,
    fee_bps_maker: float,
    fee_bps_taker: float,
    slippage_bps_limit: float,
    slippage_bps_market: float,
) -> Dict[str, float]:
    e = float(pd.to_numeric(pd.Series([entry_price]), errors="coerce").iloc[0])
    x = float(pd.to_numeric(pd.Series([exit_price]), errors="coerce").iloc[0])
    if (not np.isfinite(e)) or (not np.isfinite(x)) or e <= 0.0:
        return {
            "pnl_gross_pct": float("nan"),
            "pnl_net_pct": float("nan"),
            "entry_fee_bps": float("nan"),
            "exit_fee_bps": float("nan"),
            "entry_slippage_bps": float("nan"),
            "exit_slippage_bps": float("nan"),
            "total_cost_bps": float("nan"),
        }

    liq = str(entry_liquidity_type).strip().lower()
    is_maker = liq == "maker"
    entry_fee_bps = float(fee_bps_maker if is_maker else fee_bps_taker)
    exit_fee_bps = float(fee_bps_taker)
    entry_slip_bps = float(slippage_bps_limit if is_maker else slippage_bps_market)
    exit_slip_bps = float(slippage_bps_market)

    entry_eff = float(e * (1.0 + entry_slip_bps / 1e4))
    exit_eff = float(x * (1.0 - exit_slip_bps / 1e4))
    gross = float((x / e) - 1.0)
    net = float((exit_eff / entry_eff) - 1.0 - (entry_fee_bps + exit_fee_bps) / 1e4)
    return {
        "pnl_gross_pct": float(gross),
        "pnl_net_pct": float(net),
        "entry_fee_bps": float(entry_fee_bps),
        "exit_fee_bps": float(exit_fee_bps),
        "entry_slippage_bps": float(entry_slip_bps),
        "exit_slippage_bps": float(exit_slip_bps),
        "total_cost_bps": float((gross - net) * 1e4),
    }


def _run_ablation_for_run(
    *,
    symbol: str,
    signals: List[SignalRow],
    args: argparse.Namespace,
    killzones: List[Tuple[int, int]],
    chunks_dir: Path,
    cache_root: Path,
    out_root: Path,
) -> Path:
    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("none", {}),
        ("killzones_off", {"use_killzones": 0}),
        ("sweep_off", {"use_sweep": 0}),
        ("displacement_off", {"use_displacement": 0}),
        ("fvg_off", {"use_fvg": 0}),
        ("ob_off", {"use_ob": 0}),
        ("volatility_filter_off", {"use_volatility_filter": 0}),
    ]

    rows: List[Dict[str, Any]] = []
    for tag, overrides in variants:
        v_args = copy.deepcopy(args)
        for k, v in overrides.items():
            setattr(v_args, k, v)

        if hasattr(args, "_ict_score_weights"):
            setattr(v_args, "_ict_score_weights", dict(getattr(args, "_ict_score_weights")))

        ict_entries = 0
        ict_sl_hits = 0
        ict_tp_hits = 0
        pnl_vals: List[float] = []
        skips: List[str] = []

        for s in signals:
            signal_time = _to_utc_ts(s.signal_time)
            start_ts = signal_time - pd.Timedelta(hours=float(args.pre_buffer_hours))
            end_ts = signal_time + pd.Timedelta(hours=float(args.exec_horizon_hours))

            chunk_fp = chunks_dir / f"{s.signal_id}_3m.parquet"
            if chunk_fp.exists():
                df3m = pd.read_parquet(chunk_fp)
            else:
                df3m = _load_or_fetch_klines(
                    symbol=symbol,
                    timeframe=str(args.timeframe),
                    start_ts=start_ts,
                    end_ts=end_ts,
                    cache_root=cache_root,
                    max_retries=int(args.max_fetch_retries),
                    retry_base_sleep_sec=float(args.retry_base_sleep),
                    retry_max_sleep_sec=float(args.retry_max_sleep),
                    pause_sec=float(args.fetch_pause_sec),
                )
            df3m = _normalize_ohlcv_cols(df3m)
            df3m = df3m[(df3m["Timestamp"] >= start_ts) & (df3m["Timestamp"] < end_ts)].reset_index(drop=True)

            baseline = _simulate_baseline_long(
                df3m=df3m,
                signal_time=signal_time,
                tp_mult=float(s.tp_mult),
                sl_mult=float(s.sl_mult),
                eval_horizon_hours=float(args.exec_horizon_hours),
            )
            ict = _simulate_ict_long(
                df3m=df3m,
                signal_time=signal_time,
                tp_mult=float(s.tp_mult),
                sl_mult=float(s.sl_mult),
                args=v_args,
                killzones=killzones,
                debug_events=None,
                signal_id=s.signal_id,
                baseline_exit_time=baseline.get("exit_time"),
                eval_horizon_hours=float(args.exec_horizon_hours),
            )

            filled = int(bool(ict.get("filled", False)))
            ict_entries += filled
            ict_sl_hits += int(bool(ict.get("sl_hit", False))) if filled else 0
            ict_tp_hits += int(bool(ict.get("tp_hit", False))) if filled else 0
            if filled:
                pnl = _pnl_pct_from_entry_exit(ict.get("entry_price"), ict.get("exit_price"))
                if np.isfinite(pnl):
                    pnl_vals.append(float(pnl))
            else:
                rs = str(ict.get("skip_reason", "")).strip()
                if rs:
                    skips.append(rs)

        top_skip = Counter(skips).most_common(3)
        top_skip_str = "|".join([f"{k}:{int(v)}" for k, v in top_skip])

        rows.append(
            {
                "filter_disabled": tag,
                "signals_total": int(len(signals)),
                "ict_entries": int(ict_entries),
                "ict_sl_hits": int(ict_sl_hits),
                "ict_tp_hits": int(ict_tp_hits),
                "ict_pnl_sum": float(np.nansum(np.asarray(pnl_vals, dtype=float))) if pnl_vals else float("nan"),
                "ict_pnl_median": float(np.nanmedian(np.asarray(pnl_vals, dtype=float))) if pnl_vals else float("nan"),
                "entry_rate_ict": float(ict_entries / max(1, len(signals))),
                "top_skip_reasons": top_skip_str,
            }
        )

    out = out_root / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Execution-layer 3m slicer + ICT simulator for one long coin.")

    ap.add_argument("--scan-dir", default="", help="Path to reports/params_scan/<run_id>. Defaults to latest run.")
    ap.add_argument("--best-csv", default="", help="Path to best_by_symbol.csv. Defaults to <scan-dir>/best_by_symbol.csv")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--symbol", default="", help="Optional explicit symbol override")
    ap.add_argument("--params-file", default="", help="Optional explicit params file override")

    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--max-signals", type=int, default=100)
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--exec-mode", choices=["baseline", "ict_gate", "exec_limit"], default="ict_gate")

    ap.add_argument("--use-killzones", type=int, default=1)
    ap.add_argument("--killzones", default="07:00-10:00,13:00-16:00")

    ap.add_argument("--use-sweep", type=int, default=1)
    ap.add_argument("--swing-k", type=int, default=2)
    ap.add_argument("--sweep-lookback-swings", type=int, default=5)
    ap.add_argument("--sweep-ticks", type=float, default=0.0)
    ap.add_argument("--sweep-pct", type=float, default=0.0005)
    ap.add_argument("--sweep-closeback-bars", type=int, default=3)

    ap.add_argument("--use-displacement", type=int, default=1)
    ap.add_argument("--disp-body-atr-mult", type=float, default=1.0)
    ap.add_argument("--disp-body-pct", type=float, default=0.002)

    ap.add_argument("--use-fvg", type=int, default=1)
    ap.add_argument("--use-ob", type=int, default=0)
    ap.add_argument("--fvg-retrace-bars", type=int, default=20)

    ap.add_argument("--max-wait-bars", type=int, default=60)
    ap.add_argument("--max-confirm-bars", type=int, default=20)
    ap.add_argument("--entry-timeout-bars", type=int, default=40)
    ap.add_argument("--entry-mode", choices=["fvg_mid", "fvg_full", "ob_mid", "ob_open", "market"], default="fvg_mid")
    ap.add_argument("--fallback", choices=["market", "skip"], default="market")
    ap.add_argument("--limit-slippage-bps", type=float, default=0.0)
    ap.add_argument("--ict-mode", choices=["rules", "score"], default="rules")
    ap.add_argument("--ict-score-threshold", type=float, default=3.0)
    ap.add_argument(
        "--ict-score-weights",
        default="",
        help='Optional JSON, e.g. \'{"sweep":1,"displacement":1,"killzone":1,"fvg":1,"ob":1}\'',
    )

    ap.add_argument("--use-volatility-filter", type=int, default=0)
    ap.add_argument("--atr-spike-percentile", type=float, default=99.0)
    ap.add_argument("--atr-lookback-bars", type=int, default=3360)
    ap.add_argument("--atr-min-bars", type=int, default=240)

    ap.add_argument("--tighten-sl-to-sweep", type=int, default=0)
    ap.add_argument("--sl-buffer-pct", type=float, default=0.0005)
    ap.add_argument("--sl-buffer-ticks", type=float, default=0.0)

    ap.add_argument("--exec-k", type=float, default=0.5)
    ap.add_argument("--exec-timeout-bars", type=int, default=20)
    ap.add_argument("--exec-fallback", choices=["market", "skip"], default="market")
    ap.add_argument("--exec-use-ladder", type=int, default=0)
    ap.add_argument("--exec-k1", type=float, default=0.3)
    ap.add_argument("--exec-k2", type=float, default=0.8)
    ap.add_argument("--exec-adaptive-k", type=int, default=0)
    ap.add_argument("--exec-two-stage", type=int, default=0)
    ap.add_argument("--exec-stage1-bars", type=int, default=10)
    ap.add_argument("--exec-stage2-bars", type=int, default=10)
    ap.add_argument("--exec-move-away-thr", type=float, default=1.0)
    ap.add_argument("--use-micro-panic", type=int, default=0)
    ap.add_argument("--panic-mult", type=float, default=2.5)
    ap.add_argument("--use-vol-gate", type=int, default=1)
    ap.add_argument("--vol-z-thr", type=float, default=2.5)
    ap.add_argument("--vol-p-thr", type=float, default=95.0)
    ap.add_argument("--fee-bps-maker", type=float, default=2.0)
    ap.add_argument("--fee-bps-taker", type=float, default=4.0)
    ap.add_argument("--slippage-bps-limit", type=float, default=0.5)
    ap.add_argument("--slippage-bps-market", type=float, default=2.0)
    ap.add_argument("--use-vol-regime-gate", type=int, default=0)
    ap.add_argument("--vol-regime-max-percentile", type=float, default=90.0)
    ap.add_argument("--vol-regime-lookback-bars", type=int, default=2160)
    ap.add_argument("--use-trend-gate", type=int, default=0)
    ap.add_argument("--trend-fast-col", default="EMA_50")
    ap.add_argument("--trend-slow-col", default="EMA_120")
    ap.add_argument("--trend-min-slope", type=float, default=0.0)
    ap.add_argument("--stop-distance-min-pct", type=float, default=0.0)

    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--local-timezone", default="Africa/Cairo")

    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    ap.add_argument("--run-ablation", type=int, default=0)
    ap.add_argument("--debug-ict", type=int, default=0)
    return ap


def run(args: argparse.Namespace) -> Path:
    scan_dir = _resolve_path(args.scan_dir) if str(args.scan_dir).strip() else _find_latest_scan_dir()
    best_csv = _resolve_path(args.best_csv) if str(args.best_csv).strip() else (scan_dir / "best_by_symbol.csv").resolve()
    if not best_csv.exists():
        raise SystemExit(f"Missing best_by_symbol.csv: {best_csv}")

    if str(args.symbol).strip() and str(args.params_file).strip():
        symbol = str(args.symbol).strip().upper()
        params_file = _resolve_path(args.params_file)
        if not params_file.exists():
            raise SystemExit(f"Missing params file: {params_file}")
        selected_row = pd.Series({"symbol": symbol, "params_file": str(params_file)})
    else:
        symbol, params_file, selected_row = _pick_symbol_from_best(best_csv=best_csv, rank=int(args.rank), side="long")

    payload = json.loads(params_file.read_text(encoding="utf-8"))
    p = ga_long._norm_params(_unwrap_params(payload))

    df_1h = _load_symbol_df(symbol, tf="1h")
    gate_cfg = {
        "use_vol_regime_gate": int(args.use_vol_regime_gate),
        "vol_regime_max_percentile": float(args.vol_regime_max_percentile),
        "vol_regime_lookback_bars": int(args.vol_regime_lookback_bars),
        "use_trend_gate": int(args.use_trend_gate),
        "trend_fast_col": str(args.trend_fast_col),
        "trend_slow_col": str(args.trend_slow_col),
        "trend_min_slope": float(args.trend_min_slope),
        "stop_distance_min_pct": float(args.stop_distance_min_pct),
    }
    signals = _build_1h_signals(
        df_1h=df_1h,
        p=p,
        max_signals=int(args.max_signals),
        order=str(args.signal_order),
        gate_cfg=gate_cfg,
    )

    run_id = _utc_tag()
    out_root = _resolve_path(args.outdir) / run_id
    chunks_dir = out_root / "chunks" / symbol
    out_root.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    signals_rows: List[Dict[str, Any]] = []
    for r in signals:
        signals_rows.append(
            {
                "signal_id": r.signal_id,
                "signal_time": str(r.signal_time),
                "signal_time_local": _to_local_str(r.signal_time, str(args.local_timezone)),
                "direction": "long",
                "cycle": int(r.cycle),
                "baseline_entry_ref": "next_3m_open",
                "strategy_tp_mult": float(r.tp_mult),
                "strategy_sl_mult": float(r.sl_mult),
                "signal_open_1h": float(r.signal_open_1h),
                "strategy_tp_on_1h_open": float(r.signal_open_1h * r.tp_mult),
                "strategy_sl_on_1h_open": float(r.signal_open_1h * r.sl_mult),
                "stop_distance_pct": float(r.stop_distance_pct),
                "atr_1h": float(r.atr_1h),
                "atr_percentile_1h": float(r.atr_percentile_1h),
                "trend_up_1h": int(r.trend_up_1h),
            }
        )

    signals_df = pd.DataFrame(signals_rows)
    signals_csv = out_root / f"{symbol}_signals_1h.csv"
    signals_df.to_csv(signals_csv, index=False)

    cache_root = _resolve_path(args.cache_dir)
    killzones = _parse_killzones(str(args.killzones))
    setattr(args, "_ict_score_weights", _parse_ict_score_weights(str(args.ict_score_weights)))
    debug_events: Optional[List[Dict[str, Any]]] = [] if _as_bool(args.debug_ict) else None

    diag_rows: List[Dict[str, Any]] = []

    for i, s in enumerate(signals, start=1):
        signal_time = _to_utc_ts(s.signal_time)
        start_ts = signal_time - pd.Timedelta(hours=float(args.pre_buffer_hours))
        end_ts = signal_time + pd.Timedelta(hours=float(args.exec_horizon_hours))

        row: Dict[str, Any] = {
            "signal_id": s.signal_id,
            "symbol": symbol,
            "signal_time": str(signal_time),
            "signal_time_local": _to_local_str(signal_time, str(args.local_timezone)),
            "direction": "long",
            "cycle": int(s.cycle),
            "window_start": str(start_ts),
            "window_end": str(end_ts),
            "strategy_tp_mult": float(s.tp_mult),
            "strategy_sl_mult": float(s.sl_mult),
            "baseline_entry_ref": "next_3m_open",
            "rank": int(args.rank),
            "scan_dir": str(scan_dir),
            "best_csv": str(best_csv),
            "params_file": str(params_file),
            "selected_score": float(pd.to_numeric(pd.Series([selected_row.get("score")]), errors="coerce").fillna(np.nan).iloc[0]) if "score" in selected_row else float("nan"),
            "error": "",
        }

        try:
            df3m = _load_or_fetch_klines(
                symbol=symbol,
                timeframe=str(args.timeframe),
                start_ts=start_ts,
                end_ts=end_ts,
                cache_root=cache_root,
                max_retries=int(args.max_fetch_retries),
                retry_base_sleep_sec=float(args.retry_base_sleep),
                retry_max_sleep_sec=float(args.retry_max_sleep),
                pause_sec=float(args.fetch_pause_sec),
            )
            df3m = _normalize_ohlcv_cols(df3m)
            df3m = df3m[(df3m["Timestamp"] >= start_ts) & (df3m["Timestamp"] < end_ts)].reset_index(drop=True)
            row["bars_3m"] = int(len(df3m))

            chunk_fp = chunks_dir / f"{s.signal_id}_3m.parquet"
            df3m.to_parquet(chunk_fp, index=False)
            row["chunk_path"] = str(chunk_fp)

            baseline = _simulate_baseline_long(
                df3m=df3m,
                signal_time=signal_time,
                tp_mult=float(s.tp_mult),
                sl_mult=float(s.sl_mult),
                eval_horizon_hours=float(args.exec_horizon_hours),
            )
            exec_mode = str(getattr(args, "exec_mode", "ict_gate")).strip().lower()
            if exec_mode == "baseline":
                exec_res = dict(baseline)
                exec_res["entry_type"] = "market_baseline"
                exec_res["skip_reason"] = ""
                exec_res["fallback_used"] = 0
                exec_res["vol_skip"] = 0
            elif exec_mode == "exec_limit":
                exec_res = _simulate_exec_limit_long(
                    df3m=df3m,
                    signal_time=signal_time,
                    tp_mult=float(s.tp_mult),
                    sl_mult=float(s.sl_mult),
                    args=args,
                    debug_events=debug_events,
                    signal_id=s.signal_id,
                    baseline_exit_time=baseline.get("exit_time"),
                    eval_horizon_hours=float(args.exec_horizon_hours),
                )
            else:
                exec_res = _simulate_ict_long(
                    df3m=df3m,
                    signal_time=signal_time,
                    tp_mult=float(s.tp_mult),
                    sl_mult=float(s.sl_mult),
                    args=args,
                    killzones=killzones,
                    debug_events=debug_events,
                    signal_id=s.signal_id,
                    baseline_exit_time=baseline.get("exit_time"),
                    eval_horizon_hours=float(args.exec_horizon_hours),
                )

            row.update(
                {
                    "baseline_filled": int(bool(baseline.get("filled", False))),
                    "baseline_entry_time": _fmt_ts(baseline.get("entry_time")),
                    "baseline_entry_ts": _fmt_ts(baseline.get("entry_time")),
                    "baseline_entry_time_local": _to_local_str(baseline.get("entry_time"), str(args.local_timezone)),
                    "baseline_entry_price": float(baseline.get("entry_price", np.nan)),
                    "baseline_entry_type": str(baseline.get("entry_type", "")),
                    "baseline_fill_delay_minutes": float(baseline.get("fill_delay_minutes", np.nan)),
                    "baseline_sl": float(baseline.get("sl", np.nan)),
                    "baseline_tp": float(baseline.get("tp", np.nan)),
                    "baseline_sl_hit": int(bool(baseline.get("sl_hit", False))),
                    "baseline_tp_hit": int(bool(baseline.get("tp_hit", False))),
                    "baseline_sl_hit_time": _fmt_ts(baseline.get("sl_hit_time")),
                    "baseline_tp_hit_time": _fmt_ts(baseline.get("tp_hit_time")),
                    "baseline_sl_hit_price": float(baseline.get("sl_hit_price", np.nan)),
                    "baseline_tp_hit_price": float(baseline.get("tp_hit_price", np.nan)),
                    "baseline_exit_reason": str(
                        ("timeout" if str(baseline.get("exit_reason", "")).strip().lower() == "window_end" else baseline.get("exit_reason"))
                        if bool(baseline.get("filled", False))
                        else "no_fill"
                    ),
                    "baseline_exit_time": _fmt_ts(baseline.get("exit_time")),
                    "baseline_exit_ts": _fmt_ts(baseline.get("exit_time")),
                    "baseline_exit_price": float(baseline.get("exit_price", np.nan)),
                    "baseline_mae_pct": float(baseline.get("mae_pct", np.nan)),
                    "baseline_mfe_pct": float(baseline.get("mfe_pct", np.nan)),
                    "baseline_time_to_mae_min": float(baseline.get("time_to_mae_min", np.nan)),
                    "baseline_time_to_mfe_min": float(baseline.get("time_to_mfe_min", np.nan)),
                    "baseline_skip_reason": str(baseline.get("skip_reason", "")),
                    "baseline_invalid_stop_geometry": int(bool(baseline.get("invalid_stop_geometry", 0))),
                    "baseline_invalid_tp_geometry": int(bool(baseline.get("invalid_tp_geometry", 0))),
                    "baseline_same_bar_hit": int(bool(baseline.get("same_bar_hit", 0))),
                    "baseline_valid_for_metrics": int(bool(baseline.get("valid_for_metrics", 0))),
                    "baseline_eval_window_end_ts": _fmt_ts(baseline.get("eval_window_end_time")),
                    "baseline_eval_window_minutes": float(baseline.get("eval_window_minutes", np.nan)),
                }
            )

            row.update(
                {
                    "exec_mode": str(exec_mode),
                    "exec_filled": int(bool(exec_res.get("filled", False))),
                    "exec_entry_time": _fmt_ts(exec_res.get("entry_time")),
                    "exec_entry_ts": _fmt_ts(exec_res.get("entry_time")),
                    "exec_entry_time_local": _to_local_str(exec_res.get("entry_time"), str(args.local_timezone)),
                    "exec_entry_price": float(exec_res.get("entry_price", np.nan)),
                    "exec_entry_type": str(exec_res.get("entry_type", "")),
                    "exec_fill_delay_minutes": float(exec_res.get("fill_delay_minutes", np.nan)),
                    "exec_fill_delay_min": float(exec_res.get("fill_delay_minutes", np.nan)),
                    "exec_sl": float(exec_res.get("sl", np.nan)),
                    "exec_tp": float(exec_res.get("tp", np.nan)),
                    "exec_sl_hit": int(bool(exec_res.get("sl_hit", False))),
                    "exec_tp_hit": int(bool(exec_res.get("tp_hit", False))),
                    "exec_sl_hit_time": _fmt_ts(exec_res.get("sl_hit_time")),
                    "exec_tp_hit_time": _fmt_ts(exec_res.get("tp_hit_time")),
                    "exec_sl_hit_price": float(exec_res.get("sl_hit_price", np.nan)),
                    "exec_tp_hit_price": float(exec_res.get("tp_hit_price", np.nan)),
                    "exec_exit_reason": str(
                        ("timeout" if str(exec_res.get("exit_reason", "")).strip().lower() == "window_end" else exec_res.get("exit_reason"))
                        if bool(exec_res.get("filled", False))
                        else "no_fill"
                    ),
                    "exec_exit_time": _fmt_ts(exec_res.get("exit_time")),
                    "exec_exit_ts": _fmt_ts(exec_res.get("exit_time")),
                    "exec_exit_price": float(exec_res.get("exit_price", np.nan)),
                    "exec_mae_pct": float(exec_res.get("mae_pct", np.nan)),
                    "exec_mfe_pct": float(exec_res.get("mfe_pct", np.nan)),
                    "exec_time_to_mae_min": float(exec_res.get("time_to_mae_min", np.nan)),
                    "exec_time_to_mfe_min": float(exec_res.get("time_to_mfe_min", np.nan)),
                    "exec_skip_reason": str(exec_res.get("skip_reason", "")),
                    "exec_fallback_used": int(bool(exec_res.get("fallback_used", 0))),
                    "exec_invalid_stop_geometry": int(bool(exec_res.get("invalid_stop_geometry", 0))),
                    "exec_invalid_tp_geometry": int(bool(exec_res.get("invalid_tp_geometry", 0))),
                    "exec_same_bar_hit": int(bool(exec_res.get("same_bar_hit", 0))),
                    "exec_valid_for_metrics": int(bool(exec_res.get("valid_for_metrics", 0))),
                    "exec_eval_window_end_ts": _fmt_ts(exec_res.get("eval_window_end_time")),
                    "exec_eval_window_minutes": float(exec_res.get("eval_window_minutes", np.nan)),
                    "ict_filled": int(bool(exec_res.get("filled", False))),
                    "ict_entry_time": _fmt_ts(exec_res.get("entry_time")),
                    "ict_entry_ts": _fmt_ts(exec_res.get("entry_time")),
                    "ict_entry_time_local": _to_local_str(exec_res.get("entry_time"), str(args.local_timezone)),
                    "ict_entry_price": float(exec_res.get("entry_price", np.nan)),
                    "ict_entry_type": str(exec_res.get("entry_type", "")),
                    "ict_fill_delay_minutes": float(exec_res.get("fill_delay_minutes", np.nan)),
                    "ict_sl": float(exec_res.get("sl", np.nan)),
                    "ict_tp": float(exec_res.get("tp", np.nan)),
                    "ict_sl_hit": int(bool(exec_res.get("sl_hit", False))),
                    "ict_tp_hit": int(bool(exec_res.get("tp_hit", False))),
                    "ict_sl_hit_time": _fmt_ts(exec_res.get("sl_hit_time")),
                    "ict_tp_hit_time": _fmt_ts(exec_res.get("tp_hit_time")),
                    "ict_sl_hit_price": float(exec_res.get("sl_hit_price", np.nan)),
                    "ict_tp_hit_price": float(exec_res.get("tp_hit_price", np.nan)),
                    "ict_exit_reason": str(
                        ("timeout" if str(exec_res.get("exit_reason", "")).strip().lower() == "window_end" else exec_res.get("exit_reason"))
                        if bool(exec_res.get("filled", False))
                        else "no_fill"
                    ),
                    "ict_exit_time": _fmt_ts(exec_res.get("exit_time")),
                    "ict_exit_ts": _fmt_ts(exec_res.get("exit_time")),
                    "ict_exit_price": float(exec_res.get("exit_price", np.nan)),
                    "ict_mae_pct": float(exec_res.get("mae_pct", np.nan)),
                    "ict_mfe_pct": float(exec_res.get("mfe_pct", np.nan)),
                    "ict_time_to_mae_min": float(exec_res.get("time_to_mae_min", np.nan)),
                    "ict_time_to_mfe_min": float(exec_res.get("time_to_mfe_min", np.nan)),
                    "ict_skip_reason": str(exec_res.get("skip_reason", "")),
                    "ict_score": float(exec_res.get("ict_score", np.nan)),
                    "ict_score_components": str(exec_res.get("ict_score_components", "")),
                    "ict_fallback_used": int(bool(exec_res.get("fallback_used", 0))),
                }
            )
            row["k"] = float(args.exec_k) if exec_mode == "exec_limit" else float("nan")
            row["timeout_bars"] = int(args.exec_timeout_bars) if exec_mode == "exec_limit" else float("nan")
            row["fallback"] = str(args.exec_fallback if exec_mode == "exec_limit" else args.fallback)
            row["vol_skip"] = int(bool(exec_res.get("vol_skip", 0)))
            row["atr_zscore"] = float(exec_res.get("atr_zscore", np.nan))
            row["atr_percentile"] = float(exec_res.get("atr_percentile", np.nan))
            row["k_effective"] = float(exec_res.get("k_effective", np.nan))
            row["two_stage_used"] = int(bool(_as_bool(getattr(args, "exec_two_stage", 0))))
            row["adaptive_k_used"] = int(bool(_as_bool(getattr(args, "exec_adaptive_k", 0))))
            row["panic_filter_used"] = int(bool(_as_bool(getattr(args, "use_micro_panic", 0))))
            row["invalid_stop_geometry"] = int(row.get("exec_invalid_stop_geometry", 0))
            row["invalid_tp_geometry"] = int(row.get("exec_invalid_tp_geometry", 0))
            row["same_bar_hit"] = int(row.get("exec_same_bar_hit", 0))

            baseline_liq = _liquidity_type_from_entry_type(row.get("baseline_entry_type", "")) if int(row.get("baseline_filled", 0)) == 1 else ""
            exec_liq = _liquidity_type_from_entry_type(row.get("exec_entry_type", "")) if int(row.get("exec_filled", 0)) == 1 else ""
            row["baseline_fill_liquidity_type"] = baseline_liq
            row["exec_fill_liquidity_type"] = exec_liq
            row["ict_fill_liquidity_type"] = exec_liq
            row["fill_liquidity_type"] = exec_liq

            b_entry = float(row.get("baseline_entry_price", np.nan))
            e_entry = float(row.get("exec_entry_price", np.nan))
            if np.isfinite(b_entry) and b_entry > 0 and np.isfinite(e_entry):
                row["entry_price_delta_pct"] = float((b_entry - e_entry) / b_entry)
                row["entry_improvement_pct"] = float((b_entry - e_entry) / b_entry)
            else:
                row["entry_price_delta_pct"] = float("nan")
                row["entry_improvement_pct"] = float("nan")

            b_cost = _costed_pnl_long(
                entry_price=row.get("baseline_entry_price"),
                exit_price=row.get("baseline_exit_price"),
                entry_liquidity_type=baseline_liq,
                fee_bps_maker=float(args.fee_bps_maker),
                fee_bps_taker=float(args.fee_bps_taker),
                slippage_bps_limit=float(args.slippage_bps_limit),
                slippage_bps_market=float(args.slippage_bps_market),
            )
            e_cost = _costed_pnl_long(
                entry_price=row.get("exec_entry_price"),
                exit_price=row.get("exec_exit_price"),
                entry_liquidity_type=exec_liq,
                fee_bps_maker=float(args.fee_bps_maker),
                fee_bps_taker=float(args.fee_bps_taker),
                slippage_bps_limit=float(args.slippage_bps_limit),
                slippage_bps_market=float(args.slippage_bps_market),
            )
            row["fee_bps_maker"] = float(args.fee_bps_maker)
            row["fee_bps_taker"] = float(args.fee_bps_taker)
            row["slippage_bps_limit"] = float(args.slippage_bps_limit)
            row["slippage_bps_market"] = float(args.slippage_bps_market)

            row["baseline_entry_fee_bps"] = float(b_cost["entry_fee_bps"])
            row["baseline_exit_fee_bps"] = float(b_cost["exit_fee_bps"])
            row["baseline_entry_slippage_bps"] = float(b_cost["entry_slippage_bps"])
            row["baseline_exit_slippage_bps"] = float(b_cost["exit_slippage_bps"])
            row["baseline_total_cost_bps"] = float(b_cost["total_cost_bps"])
            row["baseline_pnl_gross_pct"] = float(b_cost["pnl_gross_pct"])
            row["baseline_pnl_net_pct"] = float(b_cost["pnl_net_pct"])
            row["baseline_pnl_pct"] = float(b_cost["pnl_gross_pct"])

            row["exec_entry_fee_bps"] = float(e_cost["entry_fee_bps"])
            row["exec_exit_fee_bps"] = float(e_cost["exit_fee_bps"])
            row["exec_entry_slippage_bps"] = float(e_cost["entry_slippage_bps"])
            row["exec_exit_slippage_bps"] = float(e_cost["exit_slippage_bps"])
            row["exec_total_cost_bps"] = float(e_cost["total_cost_bps"])
            row["exec_pnl_gross_pct"] = float(e_cost["pnl_gross_pct"])
            row["exec_pnl_net_pct"] = float(e_cost["pnl_net_pct"])
            row["exec_pnl_pct"] = float(e_cost["pnl_gross_pct"])
            row["ict_pnl_gross_pct"] = float(e_cost["pnl_gross_pct"])
            row["ict_pnl_net_pct"] = float(e_cost["pnl_net_pct"])
            row["ict_pnl_pct"] = float(e_cost["pnl_gross_pct"])

            b_mae = float(row.get("baseline_mae_pct", np.nan))
            e_mae = float(row.get("exec_mae_pct", np.nan))
            row["exec_reduced_mae"] = int(np.isfinite(b_mae) and np.isfinite(e_mae) and e_mae >= b_mae)
            row["exec_avoided_sl"] = int(bool(row.get("baseline_sl_hit", 0)) and not bool(row.get("exec_sl_hit", 0)))
            row["ict_reduced_mae"] = int(row["exec_reduced_mae"])
            row["ict_avoided_sl"] = int(row["exec_avoided_sl"])

        except Exception as ex:
            row["error"] = f"{type(ex).__name__}:{ex}"
            row.setdefault("bars_3m", 0)
            row.setdefault("baseline_filled", 0)
            row.setdefault("baseline_invalid_stop_geometry", 0)
            row.setdefault("baseline_invalid_tp_geometry", 0)
            row.setdefault("baseline_same_bar_hit", 0)
            row.setdefault("baseline_valid_for_metrics", 0)
            row.setdefault("exec_mode", str(getattr(args, "exec_mode", "ict_gate")))
            row.setdefault("exec_filled", 0)
            row.setdefault("exec_skip_reason", "error")
            row.setdefault("exec_fallback_used", 0)
            row.setdefault("exec_invalid_stop_geometry", 0)
            row.setdefault("exec_invalid_tp_geometry", 0)
            row.setdefault("exec_same_bar_hit", 0)
            row.setdefault("ict_filled", 0)
            row.setdefault("ict_skip_reason", "error")
            row.setdefault("ict_score", float("nan"))
            row.setdefault("ict_score_components", "")
            row.setdefault("ict_fallback_used", 0)
            row.setdefault("baseline_exit_reason", "no_fill")
            row.setdefault("exec_exit_reason", "no_fill")
            row.setdefault("ict_exit_reason", "no_fill")

        diag_rows.append(row)
        print(
            f"[{i}/{len(signals)}] signal={s.signal_id} bars={row.get('bars_3m', 0)} baseline={row.get('baseline_filled', 0)} exec={row.get('exec_filled', row.get('ict_filled', 0))}",
            flush=True,
        )

    diag_df = pd.DataFrame(diag_rows)
    diag_suffix = {
        "baseline": "exec_baseline_vs_baseline",
        "ict_gate": "exec_ict_vs_baseline",
        "exec_limit": "exec_limit_vs_baseline",
    }.get(str(getattr(args, "exec_mode", "ict_gate")).strip().lower(), "exec_ict_vs_baseline")
    diag_csv = out_root / f"{symbol}_{diag_suffix}.csv"
    diag_df.to_csv(diag_csv, index=False)

    debug_csv: Optional[Path] = None
    if debug_events is not None:
        debug_df = pd.DataFrame(debug_events)
        debug_csv = out_root / f"{symbol}_debug_exec.csv"
        debug_df.to_csv(debug_csv, index=False)
        if "lookahead_violation" in debug_df.columns:
            print(
                f"debug_ict rows={len(debug_df)} lookahead_violations={int(pd.to_numeric(debug_df['lookahead_violation'], errors='coerce').fillna(0).astype(int).sum())}",
                flush=True,
            )

    ablation_csv: Optional[Path] = None
    if _as_bool(args.run_ablation) and str(getattr(args, "exec_mode", "ict_gate")).strip().lower() == "ict_gate":
        ablation_csv = _run_ablation_for_run(
            symbol=symbol,
            signals=signals,
            args=args,
            killzones=killzones,
            chunks_dir=chunks_dir,
            cache_root=cache_root,
            out_root=out_root,
        )
    elif _as_bool(args.run_ablation):
        print("run_ablation requested but skipped because exec_mode is not ict_gate", flush=True)

    summary_md = out_root / f"{symbol}_exec_summary.md"
    _emit_summary_md(
        path=summary_md,
        symbol=symbol,
        rank=int(args.rank),
        best_csv=best_csv,
        params_file=params_file,
        signals_csv=signals_csv,
        diag_csv=diag_csv,
        df=diag_df,
    )
    audit_md = out_root / f"{symbol}_audit_summary.md"
    audit_stats = _emit_audit_summary_md(
        path=audit_md,
        symbol=symbol,
        exec_mode=str(getattr(args, "exec_mode", "ict_gate")),
        df=diag_df,
        eval_horizon_hours=float(args.exec_horizon_hours),
    )

    meta = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "rank": int(args.rank),
        "scan_dir": str(scan_dir),
        "best_csv": str(best_csv),
        "params_file": str(params_file),
        "timeframe": str(args.timeframe),
        "signals_count": int(len(signals)),
        "signals_csv": str(signals_csv),
        "diag_csv": str(diag_csv),
        "summary_md": str(summary_md),
        "audit_md": str(audit_md),
        "cache_dir": str(cache_root),
        "ablation_csv": str(ablation_csv) if ablation_csv is not None else "",
        "debug_csv": str(debug_csv) if debug_csv is not None else "",
        "exec_mode": str(getattr(args, "exec_mode", "ict_gate")),
        "exec_k": float(getattr(args, "exec_k", 0.5)),
        "exec_timeout_bars": int(getattr(args, "exec_timeout_bars", 20)),
        "exec_fallback": str(getattr(args, "exec_fallback", "market")),
        "exec_adaptive_k": int(bool(_as_bool(getattr(args, "exec_adaptive_k", 0)))),
        "exec_two_stage": int(bool(_as_bool(getattr(args, "exec_two_stage", 0)))),
        "exec_stage1_bars": int(getattr(args, "exec_stage1_bars", 10)),
        "exec_stage2_bars": int(getattr(args, "exec_stage2_bars", 10)),
        "exec_move_away_thr": float(getattr(args, "exec_move_away_thr", 1.0)),
        "use_micro_panic": int(bool(_as_bool(getattr(args, "use_micro_panic", 0)))),
        "panic_mult": float(getattr(args, "panic_mult", 2.5)),
        "use_vol_gate": int(bool(_as_bool(getattr(args, "use_vol_gate", 0)))),
        "vol_z_thr": float(getattr(args, "vol_z_thr", 2.5)),
        "vol_p_thr": float(getattr(args, "vol_p_thr", 95.0)),
        "use_vol_regime_gate": int(bool(_as_bool(getattr(args, "use_vol_regime_gate", 0)))),
        "vol_regime_max_percentile": float(getattr(args, "vol_regime_max_percentile", 90.0)),
        "vol_regime_lookback_bars": int(getattr(args, "vol_regime_lookback_bars", 2160)),
        "use_trend_gate": int(bool(_as_bool(getattr(args, "use_trend_gate", 0)))),
        "trend_fast_col": str(getattr(args, "trend_fast_col", "EMA_50")),
        "trend_slow_col": str(getattr(args, "trend_slow_col", "EMA_120")),
        "trend_min_slope": float(getattr(args, "trend_min_slope", 0.0)),
        "stop_distance_min_pct": float(getattr(args, "stop_distance_min_pct", 0.0)),
        "fee_bps_maker": float(getattr(args, "fee_bps_maker", 2.0)),
        "fee_bps_taker": float(getattr(args, "fee_bps_taker", 4.0)),
        "slippage_bps_limit": float(getattr(args, "slippage_bps_limit", 0.5)),
        "slippage_bps_market": float(getattr(args, "slippage_bps_market", 2.0)),
        "audit_stats": audit_stats,
        "ict_mode": str(args.ict_mode),
        "ict_score_threshold": float(args.ict_score_threshold),
        "ict_score_weights": dict(getattr(args, "_ict_score_weights", {})),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(str(out_root))
    return out_root


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
