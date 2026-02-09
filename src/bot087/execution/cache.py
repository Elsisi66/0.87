from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RangeRecord:
    symbol: str
    source: str
    mode: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    rows: int
    bytes_written: int
    path: str


def _to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _index_path(cache_root: Path, symbol: str, mode: str) -> Path:
    return cache_root / symbol.upper() / mode.lower() / "range_index.json"


def _read_index(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


def _write_index(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_index(cache_root: Path, symbol: str, mode: str) -> List[RangeRecord]:
    out: List[RangeRecord] = []
    for row in _read_index(_index_path(cache_root, symbol, mode)):
        try:
            out.append(
                RangeRecord(
                    symbol=str(row.get("symbol", symbol)).upper(),
                    source=str(row.get("source", "")),
                    mode=str(row.get("mode", mode)).lower(),
                    start_ts=_to_utc_ts(row["start_ts"]),
                    end_ts=_to_utc_ts(row["end_ts"]),
                    rows=int(row.get("rows", 0)),
                    bytes_written=int(row.get("bytes_written", 0)),
                    path=str(row.get("path", "")),
                )
            )
        except Exception:
            continue
    return out


def append_index_record(
    cache_root: Path,
    *,
    symbol: str,
    mode: str,
    source: str,
    start_ts,
    end_ts,
    rows: int,
    bytes_written: int,
    path: Path,
) -> None:
    idx = _index_path(cache_root, symbol, mode)
    payload = _read_index(idx)
    payload.append(
        {
            "symbol": symbol.upper(),
            "source": source,
            "mode": mode.lower(),
            "start_ts": str(_to_utc_ts(start_ts)),
            "end_ts": str(_to_utc_ts(end_ts)),
            "rows": int(rows),
            "bytes_written": int(bytes_written),
            "path": str(path),
        }
    )
    payload.sort(key=lambda x: x["start_ts"])
    _write_index(idx, payload)


def write_parquet(df: pd.DataFrame, out_path: Path, compression: str = "zstd") -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression=compression)
    return int(out_path.stat().st_size)


def estimate_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            total += int(fp.stat().st_size)
    return int(total)


def estimate_cache_size_gb(cache_root: Path) -> float:
    return float(estimate_dir_size_bytes(cache_root) / (1024 ** 3))


def get_covering_paths(cache_root: Path, symbol: str, mode: str, start_ts, end_ts) -> List[Path]:
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    out: List[Path] = []
    for rec in load_index(cache_root, symbol, mode):
        if rec.end_ts < s or rec.start_ts > e:
            continue
        fp = Path(rec.path)
        if fp.exists():
            out.append(fp)
    dedup = sorted(set(out), key=lambda x: str(x))
    return dedup


def missing_ranges(cache_root: Path, symbol: str, mode: str, start_ts, end_ts) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return uncovered ranges inside [start_ts, end_ts]."""
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    records = sorted(load_index(cache_root, symbol, mode), key=lambda r: r.start_ts)

    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for r in records:
        if r.end_ts < s or r.start_ts > e:
            continue
        a = max(s, r.start_ts)
        b = min(e, r.end_ts)
        if not merged:
            merged.append((a, b))
            continue
        prev_s, prev_e = merged[-1]
        if a <= prev_e:
            merged[-1] = (prev_s, max(prev_e, b))
        else:
            merged.append((a, b))

    if not merged:
        return [(s, e)]

    miss: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = s
    for a, b in merged:
        if a > cur:
            miss.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        miss.append((cur, e))
    return miss
