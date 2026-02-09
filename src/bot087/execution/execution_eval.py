from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.bot087.optim.ga import _apply_cost

from .cache import (
    RangeRecord,
    _to_utc_ts,
    append_index_records,
    evict_lru,
    load_index,
    merge_ranges,
    touch_paths,
    write_parquet,
)
from .fetch_1s_klines import fetch_1s_klines
from .fetch_aggtrades import fetch_precision_1s_from_aggtrades


@dataclass(frozen=True)
class ExecutionEvalConfig:
    mode: str = "klines1s"  # klines1s | aggtrades
    window_sec: int = 15
    model: str = "default"  # default | stress
    market: str = "spot"  # spot | futures

    # compat fields kept for prior tools (not used by exec_gate default flow)
    entry_delay_sec: int = 0
    exit_delay_sec: int = 0

    cache_cap_gb: float = 20.0
    cap_gb: float = 20.0
    cache_root: str = "data/processed/execution_1s"

    fee_bps: float = 7.0
    slippage_bps: float = 2.0
    initial_equity: float = 10_000.0

    fetch_pause_sec: float = 0.02
    pause_sec: float = 0.02
    fetch_max_seconds_per_request: int = 1000
    merge_gap_sec: int = 0
    fetch_workers: int = 1


def _resolve_cap_gb(cfg: ExecutionEvalConfig) -> float:
    # Prefer new name; keep backward compatibility with cap_gb.
    return float(cfg.cache_cap_gb if cfg.cache_cap_gb is not None else cfg.cap_gb)


def _resolve_pause(cfg: ExecutionEvalConfig) -> float:
    return float(cfg.fetch_pause_sec if cfg.fetch_pause_sec is not None else cfg.pause_sec)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _logger(fetch_log_path: Optional[Path]) -> Callable[[Dict[str, Any]], None]:
    def _log(event: Dict[str, Any]) -> None:
        if fetch_log_path is None:
            return
        _append_jsonl(fetch_log_path, event)

    return _log


def _max_dd(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    runmax = np.maximum.accumulate(equity)
    dd = (runmax - equity) / np.maximum(runmax, 1e-12)
    return float(dd.max()) if dd.size else 0.0


def _pf(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    gp = float(pnl[pnl > 0.0].sum()) if (pnl > 0.0).any() else 0.0
    gl = float(pnl[pnl < 0.0].sum()) if (pnl < 0.0).any() else 0.0
    if gl < -1e-12:
        return float(gp / abs(gl))
    return 10.0 if gp > 0.0 else 0.0


def _equity_curve(initial_equity: float, pnl: np.ndarray) -> np.ndarray:
    if pnl.size == 0:
        return np.array([float(initial_equity)], dtype=float)
    return np.concatenate([[float(initial_equity)], float(initial_equity) + np.cumsum(pnl, dtype=float)])


def _fallback_raw_from_adj(px_adj: float, side: str, fee_bps: float, slip_bps: float) -> float:
    m = (float(fee_bps) + float(slip_bps)) / 1e4
    if side == "buy":
        return float(px_adj / (1.0 + m)) if (1.0 + m) > 0 else float(px_adj)
    return float(px_adj / (1.0 - m)) if (1.0 - m) > 0 else float(px_adj)


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported trades file extension: {path.suffix}")

    req = ["entry_ts", "exit_ts", "entry_px", "exit_px"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Trades file missing required columns: {miss}")

    x = df.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")
    x["exit_ts"] = pd.to_datetime(x["exit_ts"], utc=True, errors="coerce")
    x["entry_px"] = pd.to_numeric(x["entry_px"], errors="coerce")
    x["exit_px"] = pd.to_numeric(x["exit_px"], errors="coerce")

    if "units" in x.columns:
        x["units"] = pd.to_numeric(x["units"], errors="coerce")
    elif "size" in x.columns:
        x["units"] = pd.to_numeric(x["size"], errors="coerce")
    else:
        raise ValueError("Trades file must have `units` or `size` column.")

    if "fee_bps" in x.columns:
        x["fee_bps"] = pd.to_numeric(x["fee_bps"], errors="coerce")
    else:
        x["fee_bps"] = np.nan

    if "slippage_bps" in x.columns:
        x["slippage_bps"] = pd.to_numeric(x["slippage_bps"], errors="coerce")
    else:
        x["slippage_bps"] = np.nan

    # Raw opens are preferred for baseline raw fills; infer if absent.
    if "entry_open_raw" in x.columns:
        x["entry_open_raw"] = pd.to_numeric(x["entry_open_raw"], errors="coerce")
    else:
        x["entry_open_raw"] = np.nan

    if "exit_open_raw" in x.columns:
        x["exit_open_raw"] = pd.to_numeric(x["exit_open_raw"], errors="coerce")
    else:
        x["exit_open_raw"] = np.nan

    x = x.dropna(subset=["entry_ts", "exit_ts", "entry_px", "exit_px", "units"]).reset_index(drop=True)
    if x.empty:
        raise ValueError("Trades file has no valid rows after normalization.")
    return x


def _build_windows(trades: pd.DataFrame, window_sec: int, merge_gap_sec: int = 0) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    w = int(max(1, window_sec))
    ints: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in trades.itertuples(index=False):
        ets = _to_utc_ts(row.entry_ts)
        xts = _to_utc_ts(row.exit_ts)
        ints.append((ets, ets + pd.Timedelta(seconds=w)))
        ints.append((xts, xts + pd.Timedelta(seconds=w)))

    if not ints:
        return []

    ints.sort(key=lambda x: x[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = [ints[0]]
    gap = pd.Timedelta(seconds=int(max(0, merge_gap_sec)))
    for s, e in ints[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merge_ranges(merged)


def _subtract_ranges(
    wanted: List[Tuple[pd.Timestamp, pd.Timestamp]],
    covered: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not wanted:
        return []
    cov = merge_ranges(covered)
    if not cov:
        return merge_ranges(wanted)

    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for ws, we in merge_ranges(wanted):
        cur = _to_utc_ts(ws)
        end = _to_utc_ts(we)
        if end <= cur:
            continue
        for cs, ce in cov:
            csu = _to_utc_ts(cs)
            ceu = _to_utc_ts(ce)
            if ceu <= cur:
                continue
            if csu >= end:
                break
            if csu > cur:
                out.append((cur, min(csu, end)))
            cur = max(cur, ceu)
            if cur >= end:
                break
        if cur < end:
            out.append((cur, end))
    return merge_ranges(out)


def _missing_from_coverage(
    ws: pd.Timestamp,
    we: pd.Timestamp,
    coverage: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = _to_utc_ts(ws)
    end = _to_utc_ts(we)
    if end <= start:
        return []
    if not coverage:
        return [(start, end)]

    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    for cs, ce in coverage:
        c0 = _to_utc_ts(cs)
        c1 = _to_utc_ts(ce)
        if c1 <= cur:
            continue
        if c0 >= end:
            break
        if c0 > cur:
            out.append((cur, min(c0, end)))
        cur = max(cur, c1)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out


def _missing_windows_from_coverage(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    coverage: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        return []
    w = [(_to_utc_ts(a), _to_utc_ts(b)) for a, b in windows if _to_utc_ts(b) > _to_utc_ts(a)]
    if not w:
        return []
    c = merge_ranges(coverage)
    if not c:
        return w

    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    j = 0
    n = len(c)
    for ws, we in w:
        cur = ws
        while j < n and _to_utc_ts(c[j][1]) <= cur:
            j += 1
        k = j
        while k < n and _to_utc_ts(c[k][0]) < we:
            cs = _to_utc_ts(c[k][0])
            ce = _to_utc_ts(c[k][1])
            if cs > cur:
                out.append((cur, min(cs, we)))
            if ce > cur:
                cur = ce
            if cur >= we:
                break
            k += 1
        if cur < we:
            out.append((cur, we))
    return merge_ranges(out)


def _rec_effective_end(rec: RangeRecord) -> pd.Timestamp:
    m = str(rec.mode).lower()
    if "1s" in m or "aggtrade" in m:
        return _to_utc_ts(rec.end_ts) + pd.Timedelta(seconds=1)
    return _to_utc_ts(rec.end_ts)


def _overlaps(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return _to_utc_ts(a0) < _to_utc_ts(b1) and _to_utc_ts(a1) > _to_utc_ts(b0)


def _candidate_local_1s_files(project_root: Path, symbol: str) -> List[Path]:
    sym = symbol.upper()
    cands = [
        project_root / "data" / "processed" / "_sec" / f"{sym}_1s_ohlcv.parquet",
        project_root / "data" / "processed" / "_sec" / f"{sym}_1s_ohlcv_targeted.parquet",
        project_root / "data" / "processed" / "_sec" / f"{sym}_1s.parquet",
    ]
    return [p for p in cands if p.exists()]


def _local_range(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        ts = pd.read_parquet(path, columns=["Timestamp"])
    except Exception:
        return None
    if ts.empty:
        return None
    ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], utc=True, errors="coerce")
    ts = ts.dropna(subset=["Timestamp"]) 
    if ts.empty:
        return None
    return _to_utc_ts(ts["Timestamp"].min()), _to_utc_ts(ts["Timestamp"].max())


def _fetch_interval(
    *,
    symbol: str,
    mode: str,
    market: str,
    start_ts,
    end_ts,
    cfg: ExecutionEvalConfig,
    log: Callable[[Dict[str, Any]], None],
) -> pd.DataFrame:
    if mode == "klines1s":
        return fetch_1s_klines(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            interval="1s",
            max_seconds_per_request=int(cfg.fetch_max_seconds_per_request),
            pause_sec=_resolve_pause(cfg),
            log_cb=log,
        )
    if mode == "aggtrades":
        return fetch_precision_1s_from_aggtrades(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            market=market,
            max_window_sec=min(int(cfg.fetch_max_seconds_per_request), 3500),
            pause_sec=_resolve_pause(cfg),
            log_cb=log,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _ensure_cache_for_windows(
    *,
    symbol: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    cfg: ExecutionEvalConfig,
    cache_root: Path,
    fetch_log_path: Optional[Path],
) -> List[Path]:
    log = _logger(fetch_log_path)
    mode = str(cfg.mode).lower()
    market = str(cfg.market).lower()

    cap = _resolve_cap_gb(cfg)
    evicted0 = evict_lru(cache_root, cap)
    if evicted0:
        log({"event": "cache_evict_pre", "count": len(evicted0), "files": evicted0[:50]})

    downloaded_paths: List[Path] = []
    index_rows_to_append: List[Dict[str, Any]] = []
    idx_records = load_index(cache_root, symbol, mode)
    indexed_cov = merge_ranges([(r.start_ts, _rec_effective_end(r)) for r in idx_records])

    # If local 1s cache fully covers all requested windows, skip API.
    proj = Path(__file__).resolve().parents[3]
    locals_ = _candidate_local_1s_files(proj, symbol)
    local_cov: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for lp in locals_:
        rr = _local_range(lp)
        if rr is not None:
            local_cov.append(rr)
            log({"event": "local_cache_found", "path": str(lp), "start": str(rr[0]), "end": str(rr[1])})
    local_cov = merge_ranges(local_cov)

    miss_all: List[Tuple[pd.Timestamp, pd.Timestamp]] = _missing_windows_from_coverage(windows, indexed_cov)
    if local_cov:
        miss_all = _missing_windows_from_coverage(miss_all, local_cov)

    def _fetch_one(ms: pd.Timestamp, me: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, str]:
        if me <= ms:
            return ms, me, pd.DataFrame(), ""
        try:
            df = _fetch_interval(
                symbol=symbol,
                mode=mode,
                market=market,
                start_ts=ms,
                end_ts=me,
                cfg=cfg,
                log=log,
            )
            source = f"{market}_{mode}"
        except Exception as ex:
            log(
                {
                    "event": "primary_fetch_failed",
                    "symbol": symbol.upper(),
                    "mode": mode,
                    "market": market,
                    "start": str(ms),
                    "end": str(me),
                    "error": str(ex),
                }
            )
            # Fallback: aggTrades (spot/futures)
            df = fetch_precision_1s_from_aggtrades(
                symbol=symbol,
                start_ts=ms,
                end_ts=me,
                market=market,
                max_window_sec=3500,
                pause_sec=_resolve_pause(cfg),
                log_cb=log,
            )
            source = f"{market}_aggtrades_fallback"
        return ms, me, df, source

    def _write_fetched(ms: pd.Timestamp, me: pd.Timestamp, df: pd.DataFrame, source: str) -> None:
        if me <= ms:
            return
        if df.empty:
            log({"event": "fetch_empty", "symbol": symbol.upper(), "start": str(ms), "end": str(me)})
            return

        out_dir = cache_root / symbol.upper() / mode / "chunks"
        tag = f"{ms.strftime('%Y%m%dT%H%M%S')}_{me.strftime('%Y%m%dT%H%M%S')}"
        out_path = out_dir / f"{symbol.upper()}_{mode}_{tag}.parquet"
        bytes_written = write_parquet(df, out_path, compression="zstd")
        now = pd.Timestamp.utcnow()
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        index_rows_to_append.append(
            {
                "symbol": symbol.upper(),
                "source": source,
                "mode": mode.lower(),
                "start_ts": str(df["Timestamp"].min()),
                "end_ts": str(df["Timestamp"].max()),
                "rows": int(len(df)),
                "bytes_written": int(bytes_written),
                "path": str(out_path),
                "created_ts": str(now),
                "last_access_ts": str(now),
            }
        )
        downloaded_paths.append(out_path)
        log(
            {
                "event": "cache_write",
                "path": str(out_path),
                "rows": int(len(df)),
                "bytes": int(bytes_written),
                "start": str(df["Timestamp"].min()),
                "end": str(df["Timestamp"].max()),
            }
        )

    workers = max(1, int(cfg.fetch_workers))
    if workers > 1 and len(miss_all) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_fetch_one, ms, me) for ms, me in miss_all]
            for fut in as_completed(futs):
                ms, me, df, source = fut.result()
                _write_fetched(ms, me, df, source)
    else:
        # Sequential fallback.
        for ms, me in miss_all:
            ms, me, df, source = _fetch_one(ms, me)
            _write_fetched(ms, me, df, source)

    if index_rows_to_append:
        append_index_records(
            cache_root,
            symbol=symbol,
            mode=mode,
            rows=index_rows_to_append,
        )
        ev = evict_lru(cache_root, cap)
        if ev:
            log({"event": "cache_evict_post_batch", "count": len(ev), "files": ev[:50]})

    return downloaded_paths


def _load_1s_data(
    *,
    symbol: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    cfg: ExecutionEvalConfig,
    cache_root: Path,
    fetch_log_path: Optional[Path],
) -> pd.DataFrame:
    log = _logger(fetch_log_path)
    mode = str(cfg.mode).lower()

    idx_records = load_index(cache_root, symbol, mode)
    indexed_cov = merge_ranges([(r.start_ts, _rec_effective_end(r)) for r in idx_records])
    paths: List[Path] = [Path(r.path) for r in idx_records if Path(r.path).exists()]

    g0 = min(s for s, _ in windows)
    g1 = max(e for _, e in windows)

    # Include local prepared 1s parquet files only where indexed cache has gaps.
    # This avoids loading large full-history files unnecessarily.
    proj = Path(__file__).resolve().parents[3]
    local_files = _candidate_local_1s_files(proj, symbol)
    need_local = bool(local_files) and bool(_missing_windows_from_coverage(windows, indexed_cov))
    if need_local:
        for lp in local_files:
            rr = _local_range(lp)
            if rr is None:
                continue
            ls, le = rr
            if le < _to_utc_ts(g0) or ls > _to_utc_ts(g1):
                continue
            paths.append(lp)

    paths = sorted(set(paths), key=lambda p: str(p))
    touch_paths(cache_root, symbol, mode, paths)
    if not paths:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            try:
                df = pd.read_parquet(
                    p,
                    columns=cols,
                    filters=[
                        ("Timestamp", ">=", _to_utc_ts(g0)),
                        ("Timestamp", "<", _to_utc_ts(g1)),
                    ],
                )
            except Exception:
                df = pd.read_parquet(p)
            for c in cols:
                if c not in df.columns:
                    if c == "Volume":
                        df[c] = 0.0
                    else:
                        raise KeyError(c)
            dfs.append(df[cols].copy())
            log({"event": "cache_read", "path": str(p), "rows": int(len(df))})
        except Exception as ex:
            log({"event": "cache_read_error", "path": str(p), "error": str(ex)})
            continue

    if not dfs:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    sec = pd.concat(dfs, ignore_index=True)
    sec["Timestamp"] = pd.to_datetime(sec["Timestamp"], utc=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        sec[c] = pd.to_numeric(sec[c], errors="coerce")
    sec = sec.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    sec = sec.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last").reset_index(drop=True)

    # Slice to requested global bounds to reduce memory.
    sec = sec[(sec["Timestamp"] >= _to_utc_ts(g0)) & (sec["Timestamp"] < _to_utc_ts(g1))].reset_index(drop=True)
    return sec


def _compute_trade_level(
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    cfg: ExecutionEvalConfig,
    model: str,
) -> pd.DataFrame:
    if sec_df.empty:
        raise ValueError("sec_df is empty")

    sec = sec_df.sort_values("Timestamp").reset_index(drop=True)
    ts = pd.to_datetime(sec["Timestamp"], utc=True, errors="coerce")
    valid = ts.notna().to_numpy()
    sec = sec.loc[valid].reset_index(drop=True)
    ts = ts.loc[valid]

    ts_ns = ts.astype("int64").to_numpy()
    open_np = pd.to_numeric(sec["Open"], errors="coerce").to_numpy(dtype=float)
    high_np = pd.to_numeric(sec["High"], errors="coerce").to_numpy(dtype=float)
    low_np = pd.to_numeric(sec["Low"], errors="coerce").to_numpy(dtype=float)
    close_np = pd.to_numeric(sec["Close"], errors="coerce").to_numpy(dtype=float)
    vol_series = sec["Volume"] if "Volume" in sec.columns else pd.Series(np.zeros(len(sec)), index=sec.index, dtype=float)
    vol_np = pd.to_numeric(vol_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    win_ns = int(max(1, int(cfg.window_sec)) * 1_000_000_000)
    m = str(model).lower()
    if m not in {"default", "stress"}:
        raise ValueError(f"Unsupported model: {model}")

    def _fill_for(ts0: pd.Timestamp, side: str, fallback: float) -> Tuple[float, float, float, float, float]:
        if ts_ns.size == 0:
            return float(fallback), float(fallback), float(fallback), float(fallback), 0.0

        s_ns = int(_to_utc_ts(ts0).value)
        e_ns = s_ns + win_ns
        i0 = int(np.searchsorted(ts_ns, s_ns, side="left"))
        i1 = int(np.searchsorted(ts_ns, e_ns, side="left"))
        if i1 <= i0:
            return float(fallback), float(fallback), float(fallback), float(fallback), 0.0

        first_open = float(open_np[i0])
        hh = high_np[i0:i1]
        ll = low_np[i0:i1]
        cc = close_np[i0:i1]
        vv = vol_np[i0:i1]

        high_max = float(np.nanmax(hh)) if hh.size else float(first_open)
        low_min = float(np.nanmin(ll)) if ll.size else float(first_open)
        v_sum = float(np.nansum(vv)) if vv.size else 0.0

        if v_sum > 0:
            typ = (hh + ll + cc) / 3.0
            vwap = float(np.nansum(typ * vv) / v_sum)
        else:
            vwap = float(cc[0]) if cc.size else float(first_open)

        if m == "default":
            fill = max(first_open, vwap) if side == "buy" else min(first_open, vwap)
        else:
            fill = high_max if side == "buy" else low_min
        return float(fill), first_open, vwap, high_max, low_min

    rows: List[Dict[str, Any]] = []

    for r in trades.itertuples(index=False):
        fee_bps = float(r.fee_bps) if pd.notna(r.fee_bps) else float(cfg.fee_bps)
        slip_bps = float(r.slippage_bps) if pd.notna(r.slippage_bps) else float(cfg.slippage_bps)

        old_entry_adj = float(r.entry_px)
        old_exit_adj = float(r.exit_px)

        old_entry_raw = (
            float(r.entry_open_raw)
            if pd.notna(r.entry_open_raw)
            else _fallback_raw_from_adj(old_entry_adj, "buy", fee_bps, slip_bps)
        )
        old_exit_raw = (
            float(r.exit_open_raw)
            if pd.notna(r.exit_open_raw)
            else _fallback_raw_from_adj(old_exit_adj, "sell", fee_bps, slip_bps)
        )

        e_fill, e_first_open, e_vwap, _, _ = _fill_for(_to_utc_ts(r.entry_ts), "buy", old_entry_raw)
        x_fill, x_first_open, x_vwap, _, _ = _fill_for(_to_utc_ts(r.exit_ts), "sell", old_exit_raw)

        new_entry_raw = float(e_fill)
        new_exit_raw = float(x_fill)

        # Apply execution cost exactly once per leg.
        new_entry_adj = float(_apply_cost(new_entry_raw, fee_bps, slip_bps, "buy"))
        new_exit_adj = float(_apply_cost(new_exit_raw, fee_bps, slip_bps, "sell"))

        units = float(r.units)
        old_pnl = float(r.net_pnl) if hasattr(r, "net_pnl") and pd.notna(r.net_pnl) else float((old_exit_adj - old_entry_adj) * units)
        new_pnl = float((new_exit_adj - new_entry_adj) * units)

        entry_slip_bps = float((new_entry_raw - old_entry_raw) / max(old_entry_raw, 1e-12) * 1e4)
        exit_slip_bps = float((old_exit_raw - new_exit_raw) / max(old_exit_raw, 1e-12) * 1e4)

        old_fill = float(old_exit_adj - old_entry_adj)
        new_fill = float(new_exit_adj - new_entry_adj)

        rows.append(
            {
                "symbol": getattr(r, "symbol", ""),
                "cycle": getattr(r, "cycle", np.nan),
                "entry_ts": str(_to_utc_ts(r.entry_ts)),
                "exit_ts": str(_to_utc_ts(r.exit_ts)),
                "units": units,
                "old_entry_fill": old_entry_adj,
                "new_entry_fill": new_entry_adj,
                "old_exit_fill": old_exit_adj,
                "new_exit_fill": new_exit_adj,
                "old_fill": old_fill,
                "new_fill": new_fill,
                "entry_slip_bps": entry_slip_bps,
                "exit_slip_bps": exit_slip_bps,
                "old_pnl": old_pnl,
                "new_pnl": new_pnl,
                "pnl_delta": float(new_pnl - old_pnl),
                "entry_model_first_open": float(e_first_open),
                "entry_model_vwap": float(e_vwap),
                "exit_model_first_open": float(x_first_open),
                "exit_model_vwap": float(x_vwap),
            }
        )

    out = pd.DataFrame(rows)
    return out


def _summary_from_trade_level(tr: pd.DataFrame, initial_equity: float) -> Dict[str, Any]:
    if tr.empty:
        return {
            "trades": 0,
            "net_before": 0.0,
            "net_after": 0.0,
            "pf_before": 0.0,
            "pf_after": 0.0,
            "dd_before": 0.0,
            "dd_after": 0.0,
            "edge_decay": 0.0,
            "dd_delta": 0.0,
            "p95_entry_slip_bps": 0.0,
            "p95_exit_slip_bps": 0.0,
            "exec_pass": False,
            "fail_reasons": ["no_trades"],
        }

    old_pnl = tr["old_pnl"].astype(float).to_numpy()
    new_pnl = tr["new_pnl"].astype(float).to_numpy()

    net_before = float(old_pnl.sum())
    net_after = float(new_pnl.sum())
    pf_before = _pf(old_pnl)
    pf_after = _pf(new_pnl)

    eq_before = _equity_curve(float(initial_equity), old_pnl)
    eq_after = _equity_curve(float(initial_equity), new_pnl)
    dd_before = _max_dd(eq_before)
    dd_after = _max_dd(eq_after)

    if net_before > 0:
        edge_decay = float(net_after / net_before)
    elif net_before == 0:
        edge_decay = 0.0
    else:
        edge_decay = float(net_after / abs(net_before))

    dd_delta = float(dd_after - dd_before)

    entry_adverse = np.maximum(0.0, tr["entry_slip_bps"].astype(float).to_numpy())
    exit_adverse = np.maximum(0.0, tr["exit_slip_bps"].astype(float).to_numpy())

    p95_entry = float(np.quantile(entry_adverse, 0.95)) if entry_adverse.size else 0.0
    p95_exit = float(np.quantile(exit_adverse, 0.95)) if exit_adverse.size else 0.0

    fail_reasons: List[str] = []
    if edge_decay < 0.70:
        fail_reasons.append("edge_decay<0.70")
    if pf_after < 1.10:
        fail_reasons.append("pf_after<1.10")
    if dd_delta > 0.05:
        fail_reasons.append("dd_after-dd_before>0.05")

    exec_pass = len(fail_reasons) == 0

    return {
        "trades": int(len(tr)),
        "orig_net": net_before,
        "adj_net": net_after,
        "delta_net": float(net_after - net_before),
        "net_before": net_before,
        "net_after": net_after,
        "pf_before": float(pf_before),
        "pf_after": float(pf_after),
        "dd_before": float(dd_before),
        "dd_after": float(dd_after),
        "edge_decay": float(edge_decay),
        "dd_delta": dd_delta,
        "p95_entry_slip_bps": p95_entry,
        "p95_exit_slip_bps": p95_exit,
        "improved_trades": int((tr["pnl_delta"].astype(float) > 0.0).sum()),
        "improved_ratio": float((tr["pnl_delta"].astype(float) > 0.0).mean()) if len(tr) else 0.0,
        "exec_pass": bool(exec_pass),
        "fail_reasons": fail_reasons,
    }


def evaluate_with_sec_data(
    *,
    symbol: str,
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    trade_level = _compute_trade_level(trades=trades, sec_df=sec_df, cfg=cfg, model=cfg.model)
    summary = _summary_from_trade_level(trade_level, initial_equity=float(cfg.initial_equity))
    return {
        "symbol": symbol.upper(),
        "mode": cfg.mode,
        "market": cfg.market,
        "window_sec": int(cfg.window_sec),
        "model": cfg.model,
        "summary": summary,
        "trade_level": trade_level,
    }


def load_and_prepare_execution_data(
    *,
    symbol: str,
    trades_path: str,
    cfg: ExecutionEvalConfig,
    fetch_log_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    trades = _load_trades(Path(trades_path))
    windows = _build_windows(trades, window_sec=int(cfg.window_sec), merge_gap_sec=int(cfg.merge_gap_sec))
    if not windows:
        raise ValueError("No trade windows generated.")

    cache_root = Path(cfg.cache_root).resolve()
    fetch_log = Path(fetch_log_path).resolve() if fetch_log_path else None

    _ensure_cache_for_windows(
        symbol=symbol,
        windows=windows,
        cfg=cfg,
        cache_root=cache_root,
        fetch_log_path=fetch_log,
    )

    sec_df = _load_1s_data(
        symbol=symbol,
        windows=windows,
        cfg=cfg,
        cache_root=cache_root,
        fetch_log_path=fetch_log,
    )
    if sec_df.empty:
        raise RuntimeError("No 1s data available after cache/fetch stage.")

    g0 = min(s for s, _ in windows)
    g1 = max(e for _, e in windows)

    return trades, sec_df, {"start": str(g0), "end": str(g1)}


def evaluate_execution_from_trades(
    *,
    symbol: str,
    trades_path: str,
    cfg: ExecutionEvalConfig,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    rd = Path(run_dir).resolve() if run_dir else None
    fetch_log_path = (rd / "fetch_log.jsonl") if rd else None

    trades, sec_df, tr = load_and_prepare_execution_data(
        symbol=symbol,
        trades_path=trades_path,
        cfg=cfg,
        fetch_log_path=str(fetch_log_path) if fetch_log_path else None,
    )

    default_cfg = ExecutionEvalConfig(**{**cfg.__dict__, "model": "default"})
    stress_cfg = ExecutionEvalConfig(**{**cfg.__dict__, "model": "stress"})

    default_eval = evaluate_with_sec_data(symbol=symbol, trades=trades, sec_df=sec_df, cfg=default_cfg)
    stress_eval = evaluate_with_sec_data(symbol=symbol, trades=trades, sec_df=sec_df, cfg=stress_cfg)

    out = {
        "symbol": symbol.upper(),
        "mode": cfg.mode,
        "market": cfg.market,
        "window_sec": int(cfg.window_sec),
        "cache_root": str(Path(cfg.cache_root).resolve()),
        "time_range": tr,
        "summary": default_eval["summary"],
        "summary_stress": stress_eval["summary"],
        "trade_level": default_eval["trade_level"],
        "adjusted_trades": default_eval["trade_level"],
    }

    if rd is not None:
        rd.mkdir(parents=True, exist_ok=True)
        default_eval["trade_level"].to_csv(rd / "trade_level.csv", index=False)
        summary_payload = {
            "symbol": out["symbol"],
            "mode": out["mode"],
            "market": out["market"],
            "window_sec": out["window_sec"],
            "time_range": out["time_range"],
            "summary": out["summary"],
            "summary_stress": out["summary_stress"],
        }
        (rd / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return out
