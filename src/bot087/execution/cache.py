from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    created_ts: pd.Timestamp
    last_access_ts: pd.Timestamp


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow().tz_convert("UTC")


def _to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _index_path(cache_root: Path, symbol: str, mode: str) -> Path:
    return cache_root / symbol.upper() / mode.lower() / "index.json"


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
            cts = row.get("created_ts", row.get("last_access_ts", str(_now_utc())))
            ats = row.get("last_access_ts", cts)
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
                    created_ts=_to_utc_ts(cts),
                    last_access_ts=_to_utc_ts(ats),
                )
            )
        except Exception:
            continue
    return out


def _save_records(cache_root: Path, symbol: str, mode: str, records: Iterable[RangeRecord]) -> None:
    idx = _index_path(cache_root, symbol, mode)
    payload = []
    for r in records:
        payload.append(
            {
                "symbol": r.symbol,
                "source": r.source,
                "mode": r.mode,
                "start_ts": str(r.start_ts),
                "end_ts": str(r.end_ts),
                "rows": int(r.rows),
                "bytes_written": int(r.bytes_written),
                "path": str(r.path),
                "created_ts": str(r.created_ts),
                "last_access_ts": str(r.last_access_ts),
            }
        )
    payload.sort(key=lambda x: (x["start_ts"], x["path"]))
    _write_index(idx, payload)


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
    now = _now_utc()
    records = load_index(cache_root, symbol, mode)
    records.append(
        RangeRecord(
            symbol=symbol.upper(),
            source=source,
            mode=mode.lower(),
            start_ts=_to_utc_ts(start_ts),
            end_ts=_to_utc_ts(end_ts),
            rows=int(rows),
            bytes_written=int(bytes_written),
            path=str(path),
            created_ts=now,
            last_access_ts=now,
        )
    )
    _save_records(cache_root, symbol, mode, records)


def append_index_records(
    cache_root: Path,
    *,
    symbol: str,
    mode: str,
    rows: List[Dict],
) -> None:
    if not rows:
        return
    records = load_index(cache_root, symbol, mode)
    for row in rows:
        try:
            records.append(
                RangeRecord(
                    symbol=str(row.get("symbol", symbol)).upper(),
                    source=str(row.get("source", "")),
                    mode=str(row.get("mode", mode)).lower(),
                    start_ts=_to_utc_ts(row["start_ts"]),
                    end_ts=_to_utc_ts(row["end_ts"]),
                    rows=int(row.get("rows", 0)),
                    bytes_written=int(row.get("bytes_written", 0)),
                    path=str(row.get("path", "")),
                    created_ts=_to_utc_ts(row.get("created_ts", _now_utc())),
                    last_access_ts=_to_utc_ts(row.get("last_access_ts", row.get("created_ts", _now_utc()))),
                )
            )
        except Exception:
            continue
    _save_records(cache_root, symbol, mode, records)


def touch_paths(cache_root: Path, symbol: str, mode: str, paths: List[Path]) -> None:
    if not paths:
        return
    pathset = {str(p.resolve()) for p in paths}
    now = _now_utc()
    records = load_index(cache_root, symbol, mode)
    updated: List[RangeRecord] = []
    for r in records:
        rp = Path(r.path)
        key = str(rp.resolve()) if rp.exists() else str(rp)
        if key in pathset or str(r.path) in pathset:
            updated.append(
                RangeRecord(
                    symbol=r.symbol,
                    source=r.source,
                    mode=r.mode,
                    start_ts=r.start_ts,
                    end_ts=r.end_ts,
                    rows=r.rows,
                    bytes_written=r.bytes_written,
                    path=r.path,
                    created_ts=r.created_ts,
                    last_access_ts=now,
                )
            )
        else:
            updated.append(r)
    _save_records(cache_root, symbol, mode, updated)


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


def merge_ranges(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not ranges:
        return []
    norm = [(_to_utc_ts(s), _to_utc_ts(e)) for s, e in ranges if _to_utc_ts(e) > _to_utc_ts(s)]
    if not norm:
        return []
    norm.sort(key=lambda x: x[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = [norm[0]]
    for s, e in norm[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _effective_end(rec: RangeRecord) -> pd.Timestamp:
    # Execution cache stores timestamped bars; treat end as exclusive by adding one bar.
    # Current use is 1s bars from klines/aggtrades.
    m = str(rec.mode).lower()
    if "1s" in m or "aggtrade" in m:
        return rec.end_ts + pd.Timedelta(seconds=1)
    return rec.end_ts


def get_covering_paths(cache_root: Path, symbol: str, mode: str, start_ts, end_ts) -> List[Path]:
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    out: List[Path] = []
    for rec in load_index(cache_root, symbol, mode):
        rec_end = _effective_end(rec)
        if rec_end <= s or rec.start_ts >= e:
            continue
        fp = Path(rec.path)
        if fp.exists():
            out.append(fp)
    dedup = sorted(set(out), key=lambda x: str(x))
    touch_paths(cache_root, symbol, mode, dedup)
    return dedup


def missing_ranges(cache_root: Path, symbol: str, mode: str, start_ts, end_ts) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    records = sorted(load_index(cache_root, symbol, mode), key=lambda r: r.start_ts)

    merged_cov: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for r in records:
        rec_end = _effective_end(r)
        if rec_end <= s or r.start_ts >= e:
            continue
        a = max(s, r.start_ts)
        b = min(e, rec_end)
        if not merged_cov:
            merged_cov.append((a, b))
            continue
        ps, pe = merged_cov[-1]
        if a <= pe:
            merged_cov[-1] = (ps, max(pe, b))
        else:
            merged_cov.append((a, b))

    if not merged_cov:
        return [(s, e)]

    miss: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = s
    for a, b in merged_cov:
        if a > cur:
            miss.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        miss.append((cur, e))
    return merge_ranges(miss)


def evict_lru(cache_root: Path, cap_gb: float, protected_paths: Optional[Iterable[Path]] = None) -> List[str]:
    """Evict oldest-accessed cache files until cache <= cap_gb. Returns evicted paths."""
    protected = {str(p.resolve()) for p in (protected_paths or []) if p.exists()}
    evicted: List[str] = []

    def _all_indices() -> List[Tuple[Path, str, str, List[RangeRecord]]]:
        out = []
        if not cache_root.exists():
            return out
        for idx in cache_root.rglob("index.json"):
            try:
                mode = idx.parent.name
                symbol = idx.parent.parent.name
                recs = load_index(cache_root, symbol, mode)
                out.append((idx, symbol, mode, recs))
            except Exception:
                continue
        return out

    while estimate_cache_size_gb(cache_root) > float(cap_gb):
        candidates: List[Tuple[pd.Timestamp, Path, str, str, RangeRecord]] = []
        for idx, symbol, mode, recs in _all_indices():
            for r in recs:
                fp = Path(r.path)
                if not fp.exists():
                    continue
                resolved = str(fp.resolve())
                if resolved in protected:
                    continue
                candidates.append((r.last_access_ts, idx, symbol, mode, r))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        _, _, symbol, mode, victim = candidates[0]
        vf = Path(victim.path)
        try:
            if vf.exists():
                vf.unlink()
                evicted.append(str(vf))
        except Exception:
            pass

        recs = [r for r in load_index(cache_root, symbol, mode) if r.path != victim.path]
        _save_records(cache_root, symbol, mode, recs)

    return evicted
