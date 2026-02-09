from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.bot087.optim.ga import _apply_cost

from .cache import (
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
    market: str = "spot"  # spot | futures

    # execution replay model
    window_sec: int = 15
    model: str = "default"  # default | stress

    # cache/fetch
    cache_cap_gb: float = 20.0
    cap_gb: float = 20.0
    cache_root: str = "data/processed/execution_1s"
    fetch_pause_sec: float = 0.02
    pause_sec: float = 0.02
    fetch_max_seconds_per_request: int = 1000
    merge_gap_sec: int = 0
    fetch_workers: int = 1

    # fees/costs
    fee_bps: float = 7.0
    slippage_bps: float = 2.0
    initial_equity: float = 10_000.0

    # alignment/sanity
    alignment_max_gap_sec: float = 2.0
    alignment_open_tol_pct: float = 0.01  # 1%
    debug_sample_size: int = 8

    # overlay trigger
    overlay_mode: str = "none"  # none | breakout | pullback
    overlay_window_sec: int = 30
    overlay_breakout_lookback_sec: int = 10
    overlay_breakout_bps: float = 3.0
    overlay_pullback_dip_bps: float = 8.0
    overlay_pullback_atr_k: float = 1.0
    overlay_ema_span: int = 5

    # optional quick partial TP
    overlay_partial_tp_bps: float = 15.0
    overlay_partial_tp_frac: float = 0.0
    overlay_partial_tp_window_sec: int = 60

    # compatibility fields for older callers
    entry_delay_sec: int = 0
    exit_delay_sec: int = 0


@dataclass(frozen=True)
class _AlignmentInfo:
    ok: bool
    reason: str
    first_open: float
    first_ts: Optional[pd.Timestamp]
    gap_sec: float
    mismatch_pct: float
    i0: int
    i1: int


@dataclass(frozen=True)
class _ReplayMeta:
    trades: int
    net: float
    pf: float
    dd: float
    pnl: np.ndarray
    pnl_before: np.ndarray
    equity_curve: np.ndarray
    equity_before: np.ndarray
    counts: Dict[str, int]
    fallback_count: int
    alignment_fail_count: int
    used_1s_count: int
    overlay_triggered: int
    overlay_skipped: int
    worst_entry_mismatch_pct: float
    worst_exit_mismatch_pct: float
    samples: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _resolve_cap_gb(cfg: ExecutionEvalConfig) -> float:
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


def _pf(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    gp = float(pnl[pnl > 0.0].sum()) if (pnl > 0.0).any() else 0.0
    gl = float(pnl[pnl < 0.0].sum()) if (pnl < 0.0).any() else 0.0
    if gl < -1e-12:
        return float(gp / abs(gl))
    return 10.0 if gp > 0.0 else 0.0


def _max_dd(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    runmax = np.maximum.accumulate(equity)
    dd = (runmax - equity) / np.maximum(runmax, 1e-12)
    return float(dd.max()) if dd.size else 0.0


def _fallback_raw_from_adj(px_adj: float, side: str, fee_bps: float, slip_bps: float) -> float:
    m = (float(fee_bps) + float(slip_bps)) / 1e4
    if side == "buy":
        return float(px_adj / (1.0 + m)) if (1.0 + m) > 0 else float(px_adj)
    return float(px_adj / (1.0 - m)) if (1.0 - m) > 0 else float(px_adj)


def _cfg_replace(cfg: ExecutionEvalConfig, **overrides: Any) -> ExecutionEvalConfig:
    d = asdict(cfg)
    d.update(overrides)
    return ExecutionEvalConfig(**d)


# -----------------------------------------------------------------------------
# Trades + windows
# -----------------------------------------------------------------------------


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")

    sfx = path.suffix.lower()
    if sfx == ".csv":
        df = pd.read_csv(path)
    elif sfx in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif sfx == ".json":
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
        raise ValueError("Trades file must have `units` or `size` column")

    if "fee_bps" in x.columns:
        x["fee_bps"] = pd.to_numeric(x["fee_bps"], errors="coerce")
    else:
        x["fee_bps"] = np.nan

    if "slippage_bps" in x.columns:
        x["slippage_bps"] = pd.to_numeric(x["slippage_bps"], errors="coerce")
    else:
        x["slippage_bps"] = np.nan

    if "entry_open_raw" in x.columns:
        x["entry_open_raw"] = pd.to_numeric(x["entry_open_raw"], errors="coerce")
    else:
        x["entry_open_raw"] = np.nan

    if "exit_open_raw" in x.columns:
        x["exit_open_raw"] = pd.to_numeric(x["exit_open_raw"], errors="coerce")
    else:
        x["exit_open_raw"] = np.nan

    x = x.dropna(subset=["entry_ts", "exit_ts", "entry_px", "exit_px", "units"]).copy()
    if x.empty:
        raise ValueError("Trades file has no valid rows after normalization")

    x = x.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
    x["trade_id"] = np.arange(len(x), dtype=int)
    return x


def _merge_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], gap_sec: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    xs = [(_to_utc_ts(a), _to_utc_ts(b)) for a, b in intervals if _to_utc_ts(b) > _to_utc_ts(a)]
    if not xs:
        return []
    xs.sort(key=lambda x: x[0])
    gap = pd.Timedelta(seconds=max(0, int(gap_sec)))
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = [xs[0]]
    for s, e in xs[1:]:
        ps, pe = out[-1]
        if s <= pe + gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _build_windows(trades: pd.DataFrame, cfg: ExecutionEvalConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    entry_w = max(1, int(max(cfg.window_sec, cfg.overlay_window_sec, cfg.overlay_partial_tp_window_sec)))
    exit_w = max(1, int(cfg.window_sec))
    ints: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for r in trades.itertuples(index=False):
        et = _to_utc_ts(r.entry_ts)
        xt = _to_utc_ts(r.exit_ts)
        ints.append((et, et + pd.Timedelta(seconds=entry_w)))
        ints.append((xt, xt + pd.Timedelta(seconds=exit_w)))
    return _merge_intervals(ints, gap_sec=int(cfg.merge_gap_sec))


def _needs_1s_data(cfg: ExecutionEvalConfig) -> bool:
    if int(cfg.window_sec) > 0:
        return True
    if str(cfg.overlay_mode).lower() in {"breakout", "pullback"}:
        return True
    if float(cfg.overlay_partial_tp_frac) > 0.0:
        return True
    return False


# -----------------------------------------------------------------------------
# Cache/fetch
# -----------------------------------------------------------------------------


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
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
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


def _window_missing_from_cov(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    coverage: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        return []
    w = [(_to_utc_ts(a), _to_utc_ts(b)) for a, b in windows if _to_utc_ts(b) > _to_utc_ts(a)]
    if not w:
        return []
    cov = merge_ranges(coverage)
    if not cov:
        return w

    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    j = 0
    n = len(cov)
    for ws, we in w:
        cur = ws
        while j < n and _to_utc_ts(cov[j][1]) <= cur:
            j += 1
        k = j
        while k < n and _to_utc_ts(cov[k][0]) < we:
            cs = _to_utc_ts(cov[k][0])
            ce = _to_utc_ts(cov[k][1])
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


def _ensure_cache_for_windows(
    *,
    symbol: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    cfg: ExecutionEvalConfig,
    cache_root: Path,
    fetch_log_path: Optional[Path],
) -> None:
    log = _logger(fetch_log_path)
    mode = str(cfg.mode).lower()
    market = str(cfg.market).lower()

    cap = _resolve_cap_gb(cfg)
    ev_pre = evict_lru(cache_root, cap)
    if ev_pre:
        log({"event": "cache_evict_pre", "count": len(ev_pre), "files": ev_pre[:50]})

    idx = load_index(cache_root, symbol, mode)
    idx_cov = merge_ranges([(r.start_ts, _to_utc_ts(r.end_ts) + pd.Timedelta(seconds=1)) for r in idx])

    proj = Path(__file__).resolve().parents[3]
    local_cov: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for lp in _candidate_local_1s_files(proj, symbol):
        rr = _local_range(lp)
        if rr is not None:
            local_cov.append(rr)
            log({"event": "local_cache_found", "path": str(lp), "start": str(rr[0]), "end": str(rr[1])})
    local_cov = merge_ranges(local_cov)

    miss = _window_missing_from_cov(windows, idx_cov)
    if local_cov:
        miss = _window_missing_from_cov(miss, local_cov)

    if not miss:
        return

    rows_to_append: List[Dict[str, Any]] = []

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
            src = f"{market}_{mode}"
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
            df = fetch_precision_1s_from_aggtrades(
                symbol=symbol,
                start_ts=ms,
                end_ts=me,
                market=market,
                max_window_sec=3500,
                pause_sec=_resolve_pause(cfg),
                log_cb=log,
            )
            src = f"{market}_aggtrades_fallback"
        return ms, me, df, src

    def _write_chunk(ms: pd.Timestamp, me: pd.Timestamp, df: pd.DataFrame, src: str) -> None:
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
        rows_to_append.append(
            {
                "symbol": symbol.upper(),
                "source": src,
                "mode": mode,
                "start_ts": str(_to_utc_ts(df["Timestamp"].min())),
                "end_ts": str(_to_utc_ts(df["Timestamp"].max())),
                "rows": int(len(df)),
                "bytes_written": int(bytes_written),
                "path": str(out_path),
                "created_ts": str(now),
                "last_access_ts": str(now),
            }
        )
        log(
            {
                "event": "cache_write",
                "path": str(out_path),
                "rows": int(len(df)),
                "bytes": int(bytes_written),
                "start": str(_to_utc_ts(df["Timestamp"].min())),
                "end": str(_to_utc_ts(df["Timestamp"].max())),
            }
        )

    workers = max(1, int(cfg.fetch_workers))
    if workers > 1 and len(miss) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_fetch_one, ms, me) for ms, me in miss]
            for fut in as_completed(futs):
                ms, me, df, src = fut.result()
                _write_chunk(ms, me, df, src)
    else:
        for ms, me in miss:
            ms, me, df, src = _fetch_one(ms, me)
            _write_chunk(ms, me, df, src)

    if rows_to_append:
        append_index_records(cache_root, symbol=symbol, mode=mode, rows=rows_to_append)
        ev_post = evict_lru(cache_root, cap)
        if ev_post:
            log({"event": "cache_evict_post_batch", "count": len(ev_post), "files": ev_post[:50]})


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

    g0 = min(_to_utc_ts(a) for a, _ in windows)
    g1 = max(_to_utc_ts(b) for _, b in windows)

    idx = load_index(cache_root, symbol, mode)
    paths: List[Path] = []
    for r in idx:
        fp = Path(r.path)
        if not fp.exists():
            continue
        rs = _to_utc_ts(r.start_ts)
        re = _to_utc_ts(r.end_ts) + pd.Timedelta(seconds=1)
        if re <= g0 or rs >= g1:
            continue
        paths.append(fp)

    idx_cov = merge_ranges([(r.start_ts, _to_utc_ts(r.end_ts) + pd.Timedelta(seconds=1)) for r in idx])
    proj = Path(__file__).resolve().parents[3]
    if _window_missing_from_cov(windows, idx_cov):
        for lp in _candidate_local_1s_files(proj, symbol):
            rr = _local_range(lp)
            if rr is None:
                continue
            ls, le = rr
            if le < g0 or ls > g1:
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
                        ("Timestamp", ">=", g0),
                        ("Timestamp", "<", g1),
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

    if not dfs:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    sec = pd.concat(dfs, ignore_index=True)
    sec["Timestamp"] = pd.to_datetime(sec["Timestamp"], utc=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        sec[c] = pd.to_numeric(sec[c], errors="coerce")
    sec = sec.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    sec = sec.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last").reset_index(drop=True)
    sec = sec[(sec["Timestamp"] >= g0) & (sec["Timestamp"] < g1)].reset_index(drop=True)
    return sec


# -----------------------------------------------------------------------------
# Replay internals
# -----------------------------------------------------------------------------


def _sec_arrays(sec_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if sec_df.empty:
        zf = np.array([], dtype=float)
        zi = np.array([], dtype=np.int64)
        return {"ts_ns": zi, "open": zf, "high": zf, "low": zf, "close": zf, "vol": zf}

    sec = sec_df.sort_values("Timestamp").reset_index(drop=True)
    ts = pd.to_datetime(sec["Timestamp"], utc=True, errors="coerce")
    good = ts.notna().to_numpy()
    sec = sec.loc[good].reset_index(drop=True)
    ts = ts.loc[good]

    vol_series = sec["Volume"] if "Volume" in sec.columns else pd.Series(np.zeros(len(sec)), index=sec.index, dtype=float)
    return {
        "ts_ns": ts.astype("int64").to_numpy(),
        "open": pd.to_numeric(sec["Open"], errors="coerce").to_numpy(dtype=float),
        "high": pd.to_numeric(sec["High"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(sec["Low"], errors="coerce").to_numpy(dtype=float),
        "close": pd.to_numeric(sec["Close"], errors="coerce").to_numpy(dtype=float),
        "vol": pd.to_numeric(vol_series, errors="coerce").fillna(0.0).to_numpy(dtype=float),
    }


def _slice_idx(sec: Dict[str, np.ndarray], ts0: pd.Timestamp, window_sec: int) -> Tuple[int, int]:
    ts_ns = sec["ts_ns"]
    if ts_ns.size == 0:
        return 0, 0
    s = int(_to_utc_ts(ts0).value)
    if int(window_sec) <= 0:
        i = int(np.searchsorted(ts_ns, s, side="left"))
        return i, i
    e = s + int(window_sec) * 1_000_000_000
    i0 = int(np.searchsorted(ts_ns, s, side="left"))
    i1 = int(np.searchsorted(ts_ns, e, side="left"))
    return i0, i1


def _alignment_check(
    sec: Dict[str, np.ndarray],
    ts0: pd.Timestamp,
    open_1h: float,
    window_sec: int,
    max_gap_sec: float,
    tol_pct: float,
) -> _AlignmentInfo:
    ts_ns = sec["ts_ns"]
    if ts_ns.size == 0:
        return _AlignmentInfo(False, "NO_1S_DATA", np.nan, None, np.inf, np.inf, 0, 0)

    start_ns = int(_to_utc_ts(ts0).value)
    i0 = int(np.searchsorted(ts_ns, start_ns, side="left"))
    if i0 >= ts_ns.size:
        return _AlignmentInfo(False, "NO_1S_AT_OR_AFTER_TS", np.nan, None, np.inf, np.inf, i0, i0)

    first_ns = int(ts_ns[i0])
    first_ts = pd.to_datetime(first_ns, utc=True)
    gap_sec = max(0.0, float((first_ns - start_ns) / 1e9))
    first_open = float(sec["open"][i0]) if i0 < sec["open"].size else np.nan

    if not np.isfinite(first_open) or first_open <= 0.0:
        return _AlignmentInfo(False, "BAD_FIRST_OPEN", first_open, first_ts, gap_sec, np.inf, i0, i0)

    if gap_sec > float(max_gap_sec):
        return _AlignmentInfo(False, "GAP_GT_MAX", first_open, first_ts, gap_sec, np.inf, i0, i0)

    if np.isfinite(open_1h) and float(open_1h) > 0.0:
        mismatch = abs(first_open / float(open_1h) - 1.0)
    else:
        mismatch = np.inf

    if mismatch > float(tol_pct):
        return _AlignmentInfo(False, "OPEN_MISMATCH_GT_TOL", first_open, first_ts, gap_sec, mismatch, i0, i0)

    i0w, i1w = _slice_idx(sec, ts0, window_sec)
    if int(window_sec) > 0 and i1w <= i0w:
        return _AlignmentInfo(False, "EMPTY_WINDOW", first_open, first_ts, gap_sec, mismatch, i0w, i1w)

    return _AlignmentInfo(True, "OK", first_open, first_ts, gap_sec, mismatch, i0w, i1w)


def _model_fill_raw(
    sec: Dict[str, np.ndarray],
    i0: int,
    i1: int,
    side: str,
    model: str,
    fallback_open: float,
) -> Tuple[float, float]:
    if i1 <= i0:
        return float(fallback_open), float(fallback_open)

    hh = sec["high"][i0:i1]
    ll = sec["low"][i0:i1]
    cc = sec["close"][i0:i1]
    vv = sec["vol"][i0:i1]
    first_open = float(sec["open"][i0])

    v_sum = float(np.nansum(vv)) if vv.size else 0.0
    if v_sum > 0:
        typ = (hh + ll + cc) / 3.0
        vwap = float(np.nansum(typ * vv) / v_sum)
    else:
        vwap = float(cc[0]) if cc.size else float(first_open)

    m = str(model).lower()
    s = str(side).lower()

    if m == "stress":
        fill = float(np.nanmax(hh)) if s == "buy" else float(np.nanmin(ll))
    else:
        fill = max(first_open, vwap) if s == "buy" else min(first_open, vwap)

    return float(fill), float(vwap)


def _overlay_breakout_trigger(
    sec: Dict[str, np.ndarray],
    entry_ts: pd.Timestamp,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    w = max(1, int(cfg.overlay_window_sec))
    i0, i1 = _slice_idx(sec, entry_ts, w)
    if i1 <= i0:
        return {"triggered": False, "reason": "OVERLAY_NO_WINDOW"}

    m = max(1, int(cfg.overlay_breakout_lookback_sec))
    if (i1 - i0) <= m:
        return {"triggered": False, "reason": "OVERLAY_SHORT_WINDOW"}

    bump = float(cfg.overlay_breakout_bps) / 1e4
    ts_ns = sec["ts_ns"]
    open_np = sec["open"]
    high_np = sec["high"]

    for j in range(i0 + m, i1):
        micro_high = float(np.nanmax(high_np[j - m : j]))
        level = micro_high * (1.0 + bump)
        if float(high_np[j]) >= level:
            fill_raw = max(float(open_np[j]), float(level))
            return {
                "triggered": True,
                "reason": "OVERLAY_BREAKOUT",
                "entry_ts": pd.to_datetime(int(ts_ns[j]), utc=True),
                "fill_raw": float(fill_raw),
            }

    return {"triggered": False, "reason": "OVERLAY_NO_TRIGGER"}


def _overlay_pullback_trigger(
    sec: Dict[str, np.ndarray],
    entry_ts: pd.Timestamp,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    w = max(1, int(cfg.overlay_window_sec))
    i0, i1 = _slice_idx(sec, entry_ts, w)
    if i1 <= i0:
        return {"triggered": False, "reason": "OVERLAY_NO_WINDOW"}
    if (i1 - i0) < 4:
        return {"triggered": False, "reason": "OVERLAY_SHORT_WINDOW"}

    op = sec["open"][i0:i1]
    hi = sec["high"][i0:i1]
    lo = sec["low"][i0:i1]
    cl = sec["close"][i0:i1]
    ts_ns = sec["ts_ns"][i0:i1]

    first_open = float(op[0])
    dip_level = first_open * (1.0 - float(cfg.overlay_pullback_dip_bps) / 1e4)

    prev_close = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum.reduce([hi - lo, np.abs(hi - prev_close), np.abs(lo - prev_close)])
    atr = pd.Series(tr).ewm(span=max(2, int(cfg.overlay_ema_span)), adjust=False).mean().to_numpy(dtype=float)

    ema = pd.Series(cl).ewm(span=max(2, int(cfg.overlay_ema_span)), adjust=False).mean().to_numpy(dtype=float)

    dip_seen = False
    for j in range(1, len(cl)):
        dip_cond = (float(lo[j]) <= dip_level) or (float(lo[j]) <= first_open - float(cfg.overlay_pullback_atr_k) * float(atr[j]))
        if dip_cond:
            dip_seen = True
        if not dip_seen:
            continue

        crossed_up = float(cl[j - 1]) <= float(ema[j - 1]) and float(cl[j]) > float(ema[j])
        if crossed_up:
            fill_raw = max(float(op[j]), float(cl[j]))
            return {
                "triggered": True,
                "reason": "OVERLAY_PULLBACK",
                "entry_ts": pd.to_datetime(int(ts_ns[j]), utc=True),
                "fill_raw": float(fill_raw),
            }

    return {"triggered": False, "reason": "OVERLAY_NO_TRIGGER"}


def _overlay_trigger(
    sec: Dict[str, np.ndarray],
    entry_ts: pd.Timestamp,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    mode = str(cfg.overlay_mode).lower()
    if mode == "none":
        return {"triggered": False, "reason": "OVERLAY_OFF"}
    if mode == "breakout":
        return _overlay_breakout_trigger(sec, entry_ts, cfg)
    if mode == "pullback":
        return _overlay_pullback_trigger(sec, entry_ts, cfg)
    return {"triggered": False, "reason": f"OVERLAY_UNKNOWN_MODE:{mode}"}


def _apply_partial_tp_overlay(
    sec: Dict[str, np.ndarray],
    entry_exec_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_fill_raw: float,
    base_exit_adj: float,
    fee_bps: float,
    slip_bps: float,
    cfg: ExecutionEvalConfig,
) -> Tuple[float, bool, str]:
    frac = float(cfg.overlay_partial_tp_frac)
    if frac <= 0.0:
        return float(base_exit_adj), False, ""

    horizon = min(_to_utc_ts(exit_ts), _to_utc_ts(entry_exec_ts) + pd.Timedelta(seconds=max(1, int(cfg.overlay_partial_tp_window_sec))))
    if horizon <= _to_utc_ts(entry_exec_ts):
        return float(base_exit_adj), False, ""

    i0, i1 = _slice_idx(sec, entry_exec_ts, int((horizon - _to_utc_ts(entry_exec_ts)).total_seconds()))
    if i1 <= i0:
        return float(base_exit_adj), False, ""

    target_raw = float(entry_fill_raw) * (1.0 + float(cfg.overlay_partial_tp_bps) / 1e4)
    hh = sec["high"][i0:i1]
    ts_ns = sec["ts_ns"][i0:i1]

    hit_idx = -1
    for j in range(len(hh)):
        if float(hh[j]) >= target_raw:
            hit_idx = j
            break

    if hit_idx < 0:
        return float(base_exit_adj), False, ""

    target_adj = float(_apply_cost(target_raw, fee_bps, slip_bps, "sell"))
    f = min(max(frac, 0.0), 1.0)
    blended_exit = f * target_adj + (1.0 - f) * float(base_exit_adj)
    hit_ts = str(pd.to_datetime(int(ts_ns[hit_idx]), utc=True))
    return float(blended_exit), True, hit_ts


# -----------------------------------------------------------------------------
# Core replay
# -----------------------------------------------------------------------------


def _replay_trades(
    *,
    symbol: str,
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    cfg: ExecutionEvalConfig,
    model: str,
    log: Callable[[Dict[str, Any]], None],
) -> Tuple[pd.DataFrame, _ReplayMeta]:
    eps = 1e-12
    sec = _sec_arrays(sec_df)

    cash_before = float(cfg.initial_equity)
    cash_after = float(cfg.initial_equity)

    pnl_before_list: List[float] = []
    pnl_after_list: List[float] = []
    eq_before = [cash_before]
    eq_after = [cash_after]

    counts: Dict[str, int] = {}

    def _inc(k: str, n: int = 1) -> None:
        counts[k] = int(counts.get(k, 0) + n)

    fallback_count = 0
    alignment_fail_count = 0
    used_1s_count = 0
    overlay_triggered_count = 0
    overlay_skipped_count = 0

    worst_entry_mismatch = 0.0
    worst_exit_mismatch = 0.0
    samples: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []

    use_1s = _needs_1s_data(cfg)

    trades_sorted = trades.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)

    for r in trades_sorted.itertuples(index=False):
        trade_id = int(getattr(r, "trade_id", len(rows)))
        entry_ts = _to_utc_ts(r.entry_ts)
        exit_ts = _to_utc_ts(r.exit_ts)

        fee_bps = float(r.fee_bps) if pd.notna(r.fee_bps) else float(cfg.fee_bps)
        slip_bps = float(r.slippage_bps) if pd.notna(r.slippage_bps) else float(cfg.slippage_bps)

        old_entry_adj = float(r.entry_px)
        old_exit_adj = float(r.exit_px)

        old_entry_raw_from_adj = _fallback_raw_from_adj(old_entry_adj, "buy", fee_bps, slip_bps)
        old_exit_raw_from_adj = _fallback_raw_from_adj(old_exit_adj, "sell", fee_bps, slip_bps)

        entry_open_1h = float(r.entry_open_raw) if pd.notna(r.entry_open_raw) and float(r.entry_open_raw) > 0 else float(old_entry_raw_from_adj)
        exit_open_1h = float(r.exit_open_raw) if pd.notna(r.exit_open_raw) and float(r.exit_open_raw) > 0 else float(old_exit_raw_from_adj)

        desired_units = max(0.0, float(r.units))
        if desired_units <= 0.0:
            _inc("bad_units")

        # defaults: fallback to original fills
        entry_fill_raw = float(old_entry_raw_from_adj)
        exit_fill_raw = float(old_exit_raw_from_adj)
        entry_fill_adj_new = float(old_entry_adj)
        exit_fill_adj_new = float(old_exit_adj)

        entry_1s_open = np.nan
        exit_1s_open = np.nan

        used_1s = False
        alignment_ok = True
        overlay_triggered = False
        overlay_skipped = False

        reasons: List[str] = []

        if int(cfg.window_sec) <= 0 and str(cfg.overlay_mode).lower() == "none" and float(cfg.overlay_partial_tp_frac) <= 0.0:
            reasons.append("WINDOW0_FALLBACK_ORIG")
        elif use_1s:
            ent_al = _alignment_check(
                sec,
                entry_ts,
                entry_open_1h,
                window_sec=max(1, int(cfg.window_sec)),
                max_gap_sec=float(cfg.alignment_max_gap_sec),
                tol_pct=float(cfg.alignment_open_tol_pct),
            )
            ex_al = _alignment_check(
                sec,
                exit_ts,
                exit_open_1h,
                window_sec=max(1, int(cfg.window_sec)),
                max_gap_sec=float(cfg.alignment_max_gap_sec),
                tol_pct=float(cfg.alignment_open_tol_pct),
            )

            if np.isfinite(ent_al.first_open):
                entry_1s_open = float(ent_al.first_open)
            if np.isfinite(ex_al.first_open):
                exit_1s_open = float(ex_al.first_open)

            worst_entry_mismatch = max(worst_entry_mismatch, float(ent_al.mismatch_pct) if np.isfinite(ent_al.mismatch_pct) else 0.0)
            worst_exit_mismatch = max(worst_exit_mismatch, float(ex_al.mismatch_pct) if np.isfinite(ex_al.mismatch_pct) else 0.0)

            if not ent_al.ok:
                fallback_count += 1
                alignment_fail_count += 1
                alignment_ok = False
                _inc(f"entry_align_fail:{ent_al.reason}")
                reasons.append(f"BAD_1S_ALIGNMENT_ENTRY:{ent_al.reason}")
                log(
                    {
                        "event": "sanity_violation",
                        "symbol": symbol.upper(),
                        "trade_id": trade_id,
                        "leg": "entry",
                        "reason": ent_al.reason,
                        "gap_sec": float(ent_al.gap_sec),
                        "mismatch_pct": float(ent_al.mismatch_pct) if np.isfinite(ent_al.mismatch_pct) else None,
                        "entry_ts": str(entry_ts),
                    }
                )
                if len(samples) < int(cfg.debug_sample_size):
                    samples.append(
                        {
                            "trade_id": trade_id,
                            "leg": "entry",
                            "entry_ts": str(entry_ts),
                            "open_1h": float(entry_open_1h),
                            "open_1s": float(entry_1s_open) if np.isfinite(entry_1s_open) else None,
                            "gap_sec": float(ent_al.gap_sec),
                            "mismatch_pct": float(ent_al.mismatch_pct) if np.isfinite(ent_al.mismatch_pct) else None,
                            "reason": ent_al.reason,
                        }
                    )

            if not ex_al.ok:
                fallback_count += 1
                alignment_fail_count += 1
                alignment_ok = False
                _inc(f"exit_align_fail:{ex_al.reason}")
                reasons.append(f"BAD_1S_ALIGNMENT_EXIT:{ex_al.reason}")
                log(
                    {
                        "event": "sanity_violation",
                        "symbol": symbol.upper(),
                        "trade_id": trade_id,
                        "leg": "exit",
                        "reason": ex_al.reason,
                        "gap_sec": float(ex_al.gap_sec),
                        "mismatch_pct": float(ex_al.mismatch_pct) if np.isfinite(ex_al.mismatch_pct) else None,
                        "exit_ts": str(exit_ts),
                    }
                )
                if len(samples) < int(cfg.debug_sample_size):
                    samples.append(
                        {
                            "trade_id": trade_id,
                            "leg": "exit",
                            "exit_ts": str(exit_ts),
                            "open_1h": float(exit_open_1h),
                            "open_1s": float(exit_1s_open) if np.isfinite(exit_1s_open) else None,
                            "gap_sec": float(ex_al.gap_sec),
                            "mismatch_pct": float(ex_al.mismatch_pct) if np.isfinite(ex_al.mismatch_pct) else None,
                            "reason": ex_al.reason,
                        }
                    )

            entry_exec_ts = entry_ts
            if ent_al.ok and ex_al.ok:
                mode = str(cfg.overlay_mode).lower()
                if mode in {"breakout", "pullback"}:
                    trig = _overlay_trigger(sec, entry_ts, cfg)
                    if bool(trig.get("triggered", False)):
                        overlay_triggered = True
                        overlay_triggered_count += 1
                        _inc(f"overlay_trigger:{mode}")
                        entry_exec_ts = _to_utc_ts(trig["entry_ts"])
                        entry_fill_raw = float(trig["fill_raw"])
                        reasons.append(str(trig.get("reason", "OVERLAY_TRIGGER")))
                        used_1s = True
                    else:
                        overlay_skipped = True
                        overlay_skipped_count += 1
                        _inc("overlay_skipped")
                        reasons.append(str(trig.get("reason", "OVERLAY_NO_TRIGGER")))

                if not overlay_skipped:
                    if not overlay_triggered:
                        ei0, ei1 = _slice_idx(sec, entry_exec_ts, max(1, int(cfg.window_sec)))
                        entry_fill_raw, _ = _model_fill_raw(sec, ei0, ei1, "buy", model, float(ent_al.first_open))
                        used_1s = True

                    xi0, xi1 = _slice_idx(sec, exit_ts, max(1, int(cfg.window_sec)))
                    exit_fill_raw, _ = _model_fill_raw(sec, xi0, xi1, "sell", model, float(ex_al.first_open))
                    used_1s = True

                    if not np.isfinite(entry_fill_raw) or entry_fill_raw <= 0.0:
                        fallback_count += 1
                        _inc("entry_fill_sanity_fallback")
                        reasons.append("BAD_FILL_SANITY_ENTRY")
                        entry_fill_raw = float(old_entry_raw_from_adj)
                        used_1s = False

                    if not np.isfinite(exit_fill_raw) or exit_fill_raw <= 0.0:
                        fallback_count += 1
                        _inc("exit_fill_sanity_fallback")
                        reasons.append("BAD_FILL_SANITY_EXIT")
                        exit_fill_raw = float(old_exit_raw_from_adj)
                        used_1s = False

                    entry_fill_adj_new = float(_apply_cost(float(entry_fill_raw), fee_bps, slip_bps, "buy"))
                    exit_fill_adj_new = float(_apply_cost(float(exit_fill_raw), fee_bps, slip_bps, "sell"))

                    if float(cfg.overlay_partial_tp_frac) > 0.0 and overlay_triggered:
                        blended_exit_adj, partial_done, hit_ts = _apply_partial_tp_overlay(
                            sec,
                            entry_exec_ts=entry_exec_ts,
                            exit_ts=exit_ts,
                            entry_fill_raw=float(entry_fill_raw),
                            base_exit_adj=float(exit_fill_adj_new),
                            fee_bps=fee_bps,
                            slip_bps=slip_bps,
                            cfg=cfg,
                        )
                        if partial_done:
                            _inc("overlay_partial_tp")
                            reasons.append(f"OVERLAY_PARTIAL_TP:{hit_ts}")
                            exit_fill_adj_new = float(blended_exit_adj)

        # Before replay (cash constrained)
        old_units = min(desired_units, cash_before / max(old_entry_adj, eps)) if old_entry_adj > 0 else 0.0
        if old_units < desired_units - 1e-9:
            _inc("before_units_capped")

        cost_before = old_units * old_entry_adj
        proceeds_before = old_units * old_exit_adj
        cash_before = cash_before - cost_before + proceeds_before
        pnl_before = float(proceeds_before - cost_before)

        if cash_before < -1e-6:
            raise RuntimeError(f"BUG: negative equity in before replay at trade_id={trade_id} cash={cash_before}")

        # After replay (cash constrained)
        effective_exit_adj = float(exit_fill_adj_new)
        if overlay_skipped:
            units_after = 0.0
            pnl_after = 0.0
            _inc("overlay_trade_skipped")
        else:
            units_after = min(desired_units, cash_after / max(entry_fill_adj_new, eps)) if entry_fill_adj_new > 0 else 0.0
            if units_after < desired_units - 1e-9:
                _inc("after_units_capped")
                reasons.append("UNITS_CAPPED_CASH")

            if units_after <= 0.0:
                pnl_after = 0.0
                reasons.append("INSUFFICIENT_CASH_SKIP")
            else:
                cost_after = units_after * entry_fill_adj_new
                proceeds_after = units_after * effective_exit_adj
                cash_after = cash_after - cost_after + proceeds_after
                pnl_after = float(proceeds_after - cost_after)

        if cash_after < -1e-6:
            raise RuntimeError(f"BUG: negative equity in after replay at trade_id={trade_id} cash={cash_after}")

        eq_before.append(float(cash_before))
        eq_after.append(float(cash_after))

        pnl_before_list.append(float(pnl_before))
        pnl_after_list.append(float(pnl_after))

        old_entry_raw_slip = float(old_entry_raw_from_adj)
        old_exit_raw_slip = float(old_exit_raw_from_adj)

        if overlay_skipped:
            new_entry_raw_slip = float(old_entry_raw_slip)
            new_exit_raw_slip = float(old_exit_raw_slip)
        else:
            new_entry_raw_slip = _fallback_raw_from_adj(float(entry_fill_adj_new), "buy", fee_bps, slip_bps)
            new_exit_raw_slip = _fallback_raw_from_adj(float(effective_exit_adj), "sell", fee_bps, slip_bps)

        entry_slip_bps = float((new_entry_raw_slip - old_entry_raw_slip) / max(abs(old_entry_raw_slip), eps) * 1e4)
        exit_slip_bps = float((old_exit_raw_slip - new_exit_raw_slip) / max(abs(old_exit_raw_slip), eps) * 1e4)

        if used_1s:
            used_1s_count += 1

        rows.append(
            {
                "symbol": getattr(r, "symbol", symbol.upper()),
                "cycle": getattr(r, "cycle", np.nan),
                "entry_ts": str(entry_ts),
                "exit_ts": str(exit_ts),
                "entry_1h_open": float(entry_open_1h),
                "entry_1s_open": float(entry_1s_open) if np.isfinite(entry_1s_open) else np.nan,
                "entry_fill_old": float(old_entry_adj),
                "entry_fill_new": float(entry_fill_adj_new),
                "exit_1h_open": float(exit_open_1h),
                "exit_1s_open": float(exit_1s_open) if np.isfinite(exit_1s_open) else np.nan,
                "exit_fill_old": float(old_exit_adj),
                "exit_fill_new": float(effective_exit_adj),
                "entry_slip_bps": float(entry_slip_bps),
                "exit_slip_bps": float(exit_slip_bps),
                "used_1s": bool(used_1s and not overlay_skipped),
                "alignment_ok": bool(alignment_ok),
                "overlay_triggered": bool(overlay_triggered),
                "overlay_skipped": bool(overlay_skipped),
                "reason": "|".join(reasons) if reasons else "OK",
                "units_requested": float(desired_units),
                "units_before": float(old_units),
                "units_after": float(units_after),
                "pnl_before": float(pnl_before),
                "pnl_after": float(pnl_after),
                "pnl_delta": float(pnl_after - pnl_before),
            }
        )

    tr = pd.DataFrame(rows)
    pnl_before_np = np.array(pnl_before_list, dtype=float)
    pnl_after_np = np.array(pnl_after_list, dtype=float)
    eq_before_np = np.array(eq_before, dtype=float)
    eq_after_np = np.array(eq_after, dtype=float)

    if (eq_before_np < -1e-6).any() or (eq_after_np < -1e-6).any():
        raise RuntimeError("BUG: equity became negative in replay")

    dd_val = _max_dd(eq_after_np)
    if dd_val > 1.000001:
        raise RuntimeError(f"BUG: drawdown out of range for long-only replay (dd={dd_val})")

    meta = _ReplayMeta(
        trades=int(len(tr)),
        net=float(pnl_after_np.sum()),
        pf=float(_pf(pnl_after_np)),
        dd=float(dd_val),
        pnl=pnl_after_np,
        pnl_before=pnl_before_np,
        equity_curve=eq_after_np,
        equity_before=eq_before_np,
        counts=counts,
        fallback_count=int(fallback_count),
        alignment_fail_count=int(alignment_fail_count),
        used_1s_count=int(used_1s_count),
        overlay_triggered=int(overlay_triggered_count),
        overlay_skipped=int(overlay_skipped_count),
        worst_entry_mismatch_pct=float(worst_entry_mismatch),
        worst_exit_mismatch_pct=float(worst_exit_mismatch),
        samples=samples,
    )

    return tr, meta


def _build_summary(
    *,
    before_meta: _ReplayMeta,
    after_meta: _ReplayMeta,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    before_pnl = np.array(before_meta.pnl_before, dtype=float)
    before_eq = np.array(before_meta.equity_before, dtype=float)

    net_before = float(before_pnl.sum()) if before_pnl.size else 0.0
    net_after = float(after_meta.net)

    pf_before = float(_pf(before_pnl)) if before_pnl.size else 0.0
    pf_after = float(after_meta.pf)

    dd_before = float(_max_dd(before_eq)) if before_eq.size else 0.0
    dd_after = float(after_meta.dd)

    if net_before > 0:
        edge_decay = float(net_after / net_before)
    elif net_before == 0:
        edge_decay = 0.0
    else:
        edge_decay = float(net_after / abs(net_before))

    dd_delta = float(dd_after - dd_before)

    tr_n = max(1, int(after_meta.trades))
    fallback_rate = float(after_meta.fallback_count / tr_n)
    alignment_fail_rate = float(after_meta.alignment_fail_count / tr_n)
    overlay_skip_rate = float(after_meta.overlay_skipped / tr_n)

    fail_reasons: List[str] = []
    if edge_decay < 0.70:
        fail_reasons.append("edge_decay<0.70")
    if pf_after < 1.10:
        fail_reasons.append("pf_after<1.10")
    if dd_delta > 0.05:
        fail_reasons.append("dd_after-dd_before>0.05")

    exec_pass = len(fail_reasons) == 0

    overlay_mode = str(cfg.overlay_mode).lower()
    overlay_improve = float(net_after - before_meta.net)
    if overlay_mode == "none":
        overlay_ok = True
    else:
        overlay_ok = bool(overlay_improve >= 0.0 and overlay_skip_rate <= 0.80)

    return {
        "trades": int(after_meta.trades),
        "net_before": net_before,
        "net_after": net_after,
        "delta_net": float(net_after - net_before),
        "pf_before": pf_before,
        "pf_after": pf_after,
        "dd_before": dd_before,
        "dd_after": dd_after,
        "dd_delta": dd_delta,
        "edge_decay": edge_decay,
        "fallback_rate": fallback_rate,
        "alignment_fail_rate": alignment_fail_rate,
        "overlay_skip_rate": overlay_skip_rate,
        "overlay_triggered": int(after_meta.overlay_triggered),
        "overlay_improvement_stats": {
            "overlay_mode": overlay_mode,
            "overlay_vs_no_overlay_net_delta": overlay_improve,
            "overlay_triggered": int(after_meta.overlay_triggered),
            "overlay_skipped": int(after_meta.overlay_skipped),
        },
        "worst_entry_open_mismatch_pct": float(after_meta.worst_entry_mismatch_pct),
        "worst_exit_open_mismatch_pct": float(after_meta.worst_exit_mismatch_pct),
        "exec_pass": bool(exec_pass),
        "exec_gate_ok": bool(exec_pass),
        "overlay_ok": bool(overlay_ok),
        "fail_reasons": fail_reasons,
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def evaluate_with_sec_data(
    *,
    symbol: str,
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    log = lambda *_args, **_kwargs: None

    # baseline replay: same model, but no overlay
    base_cfg = _cfg_replace(
        cfg,
        overlay_mode="none",
        overlay_partial_tp_frac=0.0,
    )

    tr_base, meta_base = _replay_trades(symbol=symbol, trades=trades, sec_df=sec_df, cfg=base_cfg, model=str(cfg.model).lower(), log=log)
    tr_after, meta_after = _replay_trades(symbol=symbol, trades=trades, sec_df=sec_df, cfg=cfg, model=str(cfg.model).lower(), log=log)

    summary = _build_summary(before_meta=meta_base, after_meta=meta_after, cfg=cfg)

    return {
        "symbol": symbol.upper(),
        "mode": cfg.mode,
        "market": cfg.market,
        "window_sec": int(cfg.window_sec),
        "model": str(cfg.model).lower(),
        "summary": summary,
        "trade_level": tr_after,
        "trade_level_base": tr_base,
        "meta_after": meta_after,
    }


def load_and_prepare_execution_data(
    *,
    symbol: str,
    trades_path: str,
    cfg: ExecutionEvalConfig,
    fetch_log_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    trades = _load_trades(Path(trades_path))

    if not _needs_1s_data(cfg):
        g0 = _to_utc_ts(trades["entry_ts"].min())
        g1 = _to_utc_ts(trades["exit_ts"].max())
        return trades, pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]), {"start": str(g0), "end": str(g1)}

    windows = _build_windows(trades, cfg)
    if not windows:
        raise ValueError("No trade windows generated")

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
        raise RuntimeError("No 1s data available after cache/fetch stage")

    g0 = min(_to_utc_ts(a) for a, _ in windows)
    g1 = max(_to_utc_ts(b) for _, b in windows)
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
    log = _logger(fetch_log_path)

    trades, sec_df, tr = load_and_prepare_execution_data(
        symbol=symbol,
        trades_path=trades_path,
        cfg=cfg,
        fetch_log_path=str(fetch_log_path) if fetch_log_path else None,
    )

    # default replay (with configured overlay)
    base_cfg = _cfg_replace(cfg, model="default")
    default_eval = evaluate_with_sec_data(symbol=symbol, trades=trades, sec_df=sec_df, cfg=base_cfg)

    # stress replay (overlay disabled)
    stress_cfg = _cfg_replace(cfg, model="stress", overlay_mode="none", overlay_partial_tp_frac=0.0)
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

    # Debug summary
    meta_after: _ReplayMeta = default_eval["meta_after"]
    debug_summary = {
        "symbol": symbol.upper(),
        "run_cfg": asdict(cfg),
        "counts": meta_after.counts,
        "fallback_count": int(meta_after.fallback_count),
        "alignment_fail_count": int(meta_after.alignment_fail_count),
        "used_1s_count": int(meta_after.used_1s_count),
        "overlay_triggered": int(meta_after.overlay_triggered),
        "overlay_skipped": int(meta_after.overlay_skipped),
        "worst_entry_open_mismatch_pct": float(meta_after.worst_entry_mismatch_pct),
        "worst_exit_open_mismatch_pct": float(meta_after.worst_exit_mismatch_pct),
        "samples": meta_after.samples,
        "summary": out["summary"],
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

        dbg_root = Path(__file__).resolve().parents[3] / "artifacts" / "execution_debug" / symbol.upper() / rd.name
        dbg_root.mkdir(parents=True, exist_ok=True)
        (dbg_root / "debug_summary.json").write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
        log({"event": "debug_summary_written", "path": str(dbg_root / "debug_summary.json")})

    return out
