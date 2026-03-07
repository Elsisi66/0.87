#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402


RUN_PREFIX = "UNIVERSAL_DATA_FOUNDATION"
PRE_BUFFER_HOURS = 6.0
POST_BUFFER_HOURS = 12.0
MAX_MERGED_WINDOW_HOURS = 72.0
WINDOW_TIMEFRAME = "3m"
RAW_SCHEMA_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "_universal_exec_3m_window_cache"
RAW_CACHE_ROOT = CACHE_ROOT / "raw_csv"
PARQUET_CACHE_ROOT = CACHE_ROOT / "parquet"
LEGACY_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "_exec_klines_cache"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
PARAM_SCAN_CSV = PROJECT_ROOT / "reports" / "params_scan" / "20260220_044949" / "best_by_symbol.csv"
FEE_MODEL_PATH = PROJECT_ROOT / "reports" / "execution_layer" / "BASELINE_AUDIT_20260221_214310" / "fee_model.json"
METRICS_DEF_PATH = PROJECT_ROOT / "reports" / "execution_layer" / "BASELINE_AUDIT_20260221_214310" / "metrics_definition.md"
LATEST_SOL_SIGNAL_SNAPSHOT = (
    PROJECT_ROOT
    / "reports"
    / "execution_layer"
    / "PHASEE2_SOL_REPRESENTATIVE_20260222_021052"
    / "config_snapshot"
    / "SOLUSDT_signals_1h.csv"
)

PASSED_SYMBOLS = ["SOLUSDT", "AVAXUSDT", "BCHUSDT", "CRVUSDT", "NEARUSDT"]
FAILED_SYMBOLS = ["ADAUSDT", "AXSUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "OGUSDT", "PAXGUSDT", "TRXUSDT", "XRPUSDT", "ZECUSDT"]
UNIVERSE = PASSED_SYMBOLS + FAILED_SYMBOLS
USER_BUCKET = {symbol: "passed_1h_long" for symbol in PASSED_SYMBOLS}
USER_BUCKET.update({symbol: "failed_1h_long" for symbol in FAILED_SYMBOLS})

WINDOW_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})\.parquet$")


@dataclass
class SymbolContext:
    symbol: str
    user_bucket: str
    params_file: Optional[Path]
    params_row: Dict[str, Any]
    params: Optional[Dict[str, Any]]
    best_score: float
    max_hold_hours: float
    source_notes: List[str]


@dataclass
class SignalWindow:
    symbol: str
    signal_id: str
    signal_time: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp


@dataclass
class MergedWindow:
    symbol: str
    window_id: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    signal_count: int
    signal_ids: List[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, obj: Any) -> None:
    def _default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(value, utc=True))
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return str(value)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def next_3m_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    ts = to_utc_ts(ts)
    minute = int(ts.minute)
    second = int(ts.second)
    micro = int(ts.microsecond)
    offset_min = minute % 3
    if offset_min == 0 and second == 0 and micro == 0:
        return ts
    floor = ts.floor("3min")
    return floor + pd.Timedelta(minutes=3)


def parse_window_bounds_from_name(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    m = WINDOW_RE.search(path.name)
    if not m:
        return None
    try:
        start_ts = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True)
        end_ts = pd.to_datetime(m.group(2), format="%Y%m%dT%H%M%S", utc=True)
    except Exception:
        return None
    return start_ts, end_ts


def find_covering_cache_file(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[Path]:
    if not LEGACY_CACHE_ROOT.exists():
        return None
    symbol = str(symbol).upper()
    dirs = sorted([p for p in LEGACY_CACHE_ROOT.glob(f"{symbol}*") if p.is_dir()], key=lambda p: p.name)
    for base_dir in dirs:
        tf_dir = base_dir / WINDOW_TIMEFRAME
        if not tf_dir.exists():
            continue
        for fp in sorted(tf_dir.glob(f"{symbol}_{WINDOW_TIMEFRAME}_*.parquet"), key=lambda p: p.name):
            bounds = parse_window_bounds_from_name(fp)
            if bounds is None:
                continue
            fp_start, fp_end = bounds
            if fp_start <= start_ts and fp_end >= end_ts:
                return fp
    return None


def window_cache_paths(symbol: str, window_id: str) -> Tuple[Path, Path]:
    raw_fp = RAW_CACHE_ROOT / symbol / f"{window_id}.csv"
    parquet_fp = PARQUET_CACHE_ROOT / symbol / f"{window_id}.parquet"
    return raw_fp, parquet_fp


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = exec3m._normalize_ohlcv_cols(df)
    use_cols = [c for c in RAW_SCHEMA_COLUMNS if c in out.columns]
    out = out.loc[:, use_cols].copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out.loc[:, RAW_SCHEMA_COLUMNS]
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    return out


def load_or_build_symbol_contexts() -> Dict[str, SymbolContext]:
    best_rows: Dict[str, Dict[str, Any]] = {}
    if PARAM_SCAN_CSV.exists():
        df = pd.read_csv(PARAM_SCAN_CSV)
        if "symbol" in df.columns:
            for _, row in df.iterrows():
                symbol = str(row.get("symbol", "")).strip().upper()
                if symbol:
                    best_rows[symbol] = row.to_dict()

    contexts: Dict[str, SymbolContext] = {}
    for symbol in UNIVERSE:
        params_row = dict(best_rows.get(symbol, {}))
        notes: List[str] = []

        params_file: Optional[Path] = None
        params_raw = str(params_row.get("params_file", "")).strip()
        if params_raw:
            cand = exec3m._resolve_path(params_raw)
            if cand.exists():
                params_file = cand
                notes.append("params_from_best_by_symbol")

        if params_file is None:
            cands = sorted((PROJECT_ROOT / "data" / "metadata" / "params").glob(f"{symbol}*active_params*.json"))
            if cands:
                params_file = cands[0]
                notes.append("params_from_fallback_glob")

        params_obj: Optional[Dict[str, Any]] = None
        max_hold_hours = float("nan")
        if params_file is not None and params_file.exists():
            raw = json.loads(params_file.read_text(encoding="utf-8"))
            params_obj = exec3m._unwrap_params(raw)
            params_obj = exec3m.ga_long._norm_params(params_obj)
            max_hold_hours = float(params_obj.get("max_hold_hours", np.nan))
        else:
            notes.append("missing_params_file")

        score = float(pd.to_numeric(pd.Series([params_row.get("score")]), errors="coerce").iloc[0]) if params_row else float("nan")
        contexts[symbol] = SymbolContext(
            symbol=symbol,
            user_bucket=USER_BUCKET[symbol],
            params_file=params_file,
            params_row=params_row,
            params=params_obj,
            best_score=score,
            max_hold_hours=max_hold_hours,
            source_notes=notes,
        )
    return contexts


def reconstruct_signals(ctx: SymbolContext) -> Tuple[pd.DataFrame, Optional[str]]:
    if ctx.params is None:
        return pd.DataFrame(), "params_missing"
    try:
        df_1h = exec3m._load_symbol_df(ctx.symbol, tf="1h")
    except Exception as exc:
        return pd.DataFrame(), f"missing_1h_dataset:{type(exc).__name__}:{exc}"

    try:
        signals = exec3m._build_1h_signals(
            df_1h=df_1h,
            p=ctx.params,
            max_signals=0,
            order="oldest",
            gate_cfg={},
        )
    except Exception as exc:
        return pd.DataFrame(), f"signal_reconstruction_failed:{type(exc).__name__}:{exc}"

    rows: List[Dict[str, Any]] = []
    for idx, sig in enumerate(signals, start=1):
        signal_time = to_utc_ts(sig.signal_time)
        entry_ref_time = next_3m_boundary(signal_time)
        rows.append(
            {
                "symbol": ctx.symbol,
                "signal_id": f"{ctx.symbol}_sig_{idx:06d}",
                "signal_time_utc": signal_time,
                "entry_reference_time_utc": entry_ref_time,
                "expected_holding_horizon_hours": float(ctx.max_hold_hours),
                "execution_pre_buffer_hours": float(PRE_BUFFER_HOURS),
                "execution_post_buffer_hours": float(POST_BUFFER_HOURS),
                "side": "long",
                "bucket_1h": ctx.user_bucket,
                "model_source": "src/bot087/optim/ga.py::build_entry_signal",
                "params_source": str(ctx.params_file) if ctx.params_file is not None else "",
                "params_scan_source": str(PARAM_SCAN_CSV) if PARAM_SCAN_CSV.exists() else "",
                "best_scan_score": float(ctx.best_score),
                "cycle": int(sig.cycle),
                "strategy_tp_mult": float(sig.tp_mult),
                "strategy_sl_mult": float(sig.sl_mult),
                "signal_open_1h": float(sig.signal_open_1h),
                "stop_distance_pct": float(sig.stop_distance_pct),
                "atr_1h": float(sig.atr_1h),
                "atr_percentile_1h": float(sig.atr_percentile_1h),
                "trend_up_1h": int(sig.trend_up_1h),
            }
        )
    if not rows:
        return pd.DataFrame(), "no_signals"
    out = pd.DataFrame(rows).sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
    return out, None


def build_signal_windows(signal_df: pd.DataFrame) -> List[SignalWindow]:
    windows: List[SignalWindow] = []
    if signal_df.empty:
        return windows
    for row in signal_df.itertuples(index=False):
        signal_time = to_utc_ts(getattr(row, "signal_time_utc"))
        windows.append(
            SignalWindow(
                symbol=str(getattr(row, "symbol")),
                signal_id=str(getattr(row, "signal_id")),
                signal_time=signal_time,
                window_start=signal_time - pd.Timedelta(hours=PRE_BUFFER_HOURS),
                window_end=signal_time + pd.Timedelta(hours=POST_BUFFER_HOURS),
            )
        )
    return windows


def merge_windows(symbol: str, windows: Sequence[SignalWindow]) -> List[MergedWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: (w.window_start, w.window_end, w.signal_id))
    merged: List[MergedWindow] = []

    cur_start = ordered[0].window_start
    cur_end = ordered[0].window_end
    cur_signal_ids = [ordered[0].signal_id]

    def _emit(idx: int) -> None:
        merged.append(
            MergedWindow(
                symbol=symbol,
                window_id=f"{symbol}_w{idx:04d}_{cur_start.strftime('%Y%m%dT%H%M%S')}_{cur_end.strftime('%Y%m%dT%H%M%S')}",
                start_ts=cur_start,
                end_ts=cur_end,
                signal_count=len(cur_signal_ids),
                signal_ids=list(cur_signal_ids),
            )
        )

    for win in ordered[1:]:
        proposed_end = max(cur_end, win.window_end)
        proposed_hours = float((proposed_end - cur_start).total_seconds() / 3600.0)
        if win.window_start <= cur_end and proposed_hours <= MAX_MERGED_WINDOW_HOURS:
            cur_end = proposed_end
            cur_signal_ids.append(win.signal_id)
            continue
        _emit(len(merged) + 1)
        cur_start = win.window_start
        cur_end = win.window_end
        cur_signal_ids = [win.signal_id]
    _emit(len(merged) + 1)
    return merged


def window_coverage_stats(signal_windows: Sequence[SignalWindow], merged: Sequence[MergedWindow]) -> Dict[str, float]:
    raw_hours = float(sum((w.window_end - w.window_start).total_seconds() for w in signal_windows) / 3600.0)
    merged_hours = float(sum((w.end_ts - w.start_ts).total_seconds() for w in merged) / 3600.0)
    savings_hours = float(raw_hours - merged_hours)
    savings_pct = float((savings_hours / raw_hours) * 100.0) if raw_hours > 0 else 0.0
    return {
        "raw_signal_window_hours": raw_hours,
        "merged_window_hours": merged_hours,
        "overlap_savings_hours": savings_hours,
        "overlap_savings_pct": savings_pct,
    }


def load_window_from_existing_parquet(
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    parquet_fp: Path,
    source_tag: str,
) -> Optional[Tuple[pd.DataFrame, str]]:
    try:
        df = pd.read_parquet(parquet_fp)
    except Exception:
        return None
    out = normalize_raw(df)
    out = out[(out["Timestamp"] >= start_ts) & (out["Timestamp"] < end_ts)].reset_index(drop=True)
    if out.empty:
        return None
    return out, source_tag


def fetch_window_data(
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    remote_state: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, str]:
    start_ts = to_utc_ts(start_ts)
    end_ts = to_utc_ts(end_ts)

    source_cache_fp = exec3m._cache_file_for_range(PARQUET_CACHE_ROOT, symbol, WINDOW_TIMEFRAME, start_ts, end_ts)
    if source_cache_fp.exists():
        loaded = load_window_from_existing_parquet(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            parquet_fp=source_cache_fp,
            source_tag="universal_parquet_cache",
        )
        if loaded is not None:
            return loaded[0], loaded[1], ""

    legacy_cover = find_covering_cache_file(symbol, start_ts, end_ts)
    if legacy_cover is not None:
        loaded = load_window_from_existing_parquet(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            parquet_fp=legacy_cover,
            source_tag="legacy_exec_cache_cover",
        )
        if loaded is not None:
            return loaded[0], loaded[1], ""

    local = exec3m._read_local_full_slice(symbol, WINDOW_TIMEFRAME, start_ts, end_ts)
    if not local.empty:
        out = normalize_raw(local)
        if not out.empty:
            return out, "local_full_3m", ""

    if remote_state.get("disabled", False):
        reason = str(remote_state.get("reason", "remote_fetch_disabled"))
        return pd.DataFrame(columns=RAW_SCHEMA_COLUMNS), "remote_blocked", reason

    try:
        fetched = exec3m._fetch_binance_klines(
            symbol=symbol,
            timeframe=WINDOW_TIMEFRAME,
            start_ts=start_ts,
            end_ts=end_ts,
            max_retries=1,
            retry_base_sleep_sec=0.2,
            retry_max_sleep_sec=1.0,
            pause_sec=0.0,
        )
        out = normalize_raw(fetched)
        return out, "remote_fetch", ""
    except Exception as exc:
        reason = f"remote_fetch_failed:{type(exc).__name__}:{exc}"
        remote_state["disabled"] = True
        remote_state["reason"] = reason
        return pd.DataFrame(columns=RAW_SCHEMA_COLUMNS), "remote_blocked", reason


def persist_window(
    *,
    df: pd.DataFrame,
    symbol: str,
    window_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[Path, Path]:
    raw_fp, parquet_fp = window_cache_paths(symbol, window_id)
    ensure_dir(raw_fp.parent)
    ensure_dir(parquet_fp.parent)

    if raw_fp.exists() and parquet_fp.exists():
        try:
            pd.read_parquet(parquet_fp, columns=["Timestamp", "Open", "High", "Low", "Close"])
            return raw_fp, parquet_fp
        except Exception:
            pass

    out = normalize_raw(df)
    out = out[(out["Timestamp"] >= start_ts) & (out["Timestamp"] < end_ts)].reset_index(drop=True)
    out.to_csv(raw_fp, index=False)
    out.to_parquet(parquet_fp, index=False)

    source_cache_fp = exec3m._cache_file_for_range(PARQUET_CACHE_ROOT, symbol, WINDOW_TIMEFRAME, start_ts, end_ts)
    ensure_dir(source_cache_fp.parent)
    if not source_cache_fp.exists():
        out.to_parquet(source_cache_fp, index=False)
    return raw_fp, parquet_fp


def compute_signal_coverage(
    signal_df: pd.DataFrame,
    window_manifests: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_id",
                "signal_time_utc",
                "window_start_utc",
                "window_end_utc",
                "coverage_status",
                "covered",
                "bars_covered",
                "bars_expected",
                "covering_window_id",
            ]
        )

    manifest_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in window_manifests:
        manifest_by_symbol.setdefault(str(row["symbol"]), []).append(dict(row))

    coverage_rows: List[Dict[str, Any]] = []
    expected_bars = int((PRE_BUFFER_HOURS + POST_BUFFER_HOURS) * 60 / 3)

    for row in signal_df.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        signal_time = to_utc_ts(getattr(row, "signal_time_utc"))
        window_start = signal_time - pd.Timedelta(hours=PRE_BUFFER_HOURS)
        window_end = signal_time + pd.Timedelta(hours=POST_BUFFER_HOURS)
        selected: Optional[Dict[str, Any]] = None
        for cand in manifest_by_symbol.get(symbol, []):
            if to_utc_ts(cand["window_start_utc"]) <= window_start and to_utc_ts(cand["window_end_utc"]) >= window_end:
                selected = cand
                break

        status = "uncovered"
        covered = 0
        bars_covered = 0
        covering_window_id = ""
        if selected is not None:
            covering_window_id = str(selected["window_id"])
            bars_covered = int(selected.get("bars_3m", 0))
            if str(selected.get("download_status", "")).lower() == "ready":
                status = "covered"
                covered = 1
            elif str(selected.get("download_status", "")).lower() == "partial":
                status = "partial"

        coverage_rows.append(
            {
                "symbol": symbol,
                "signal_id": str(getattr(row, "signal_id")),
                "signal_time_utc": signal_time,
                "window_start_utc": window_start,
                "window_end_utc": window_end,
                "coverage_status": status,
                "covered": int(covered),
                "bars_covered": int(bars_covered),
                "bars_expected": int(expected_bars),
                "covering_window_id": covering_window_id,
            }
        )
    return pd.DataFrame(coverage_rows).sort_values(["symbol", "signal_time_utc", "signal_id"]).reset_index(drop=True)


def readiness_from_quality(
    *,
    symbol: str,
    signal_count: int,
    signal_blocker: str,
    merged_windows: Sequence[MergedWindow],
    download_rows: Sequence[Dict[str, Any]],
    coverage_df: pd.DataFrame,
) -> Dict[str, Any]:
    sym_downloads = [row for row in download_rows if str(row["symbol"]) == symbol]
    sym_cov = coverage_df[coverage_df["symbol"] == symbol].copy() if not coverage_df.empty else pd.DataFrame()

    missing_windows = int(sum(1 for row in sym_downloads if str(row.get("download_status", "")).lower() == "blocked"))
    partial_windows = int(sum(1 for row in sym_downloads if str(row.get("download_status", "")).lower() == "partial"))
    ready_windows = int(sum(1 for row in sym_downloads if str(row.get("download_status", "")).lower() == "ready"))
    signals_covered = int((pd.to_numeric(sym_cov.get("covered", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int) == 1).sum()) if not sym_cov.empty else 0
    uncovered_signals = int((sym_cov["coverage_status"] == "uncovered").sum()) if not sym_cov.empty else signal_count
    partially_covered_signals = int((sym_cov["coverage_status"] == "partial").sum()) if not sym_cov.empty else 0
    missing_rate = float(uncovered_signals / signal_count) if signal_count > 0 else 1.0

    parquet_ok = 1
    parquet_failures = 0
    for row in sym_downloads:
        if not row.get("parquet_path"):
            parquet_ok = 0
            parquet_failures += 1
            continue
        try:
            pd.read_parquet(Path(str(row["parquet_path"])), columns=["Timestamp", "Open", "High", "Low", "Close"])
        except Exception:
            parquet_ok = 0
            parquet_failures += 1

    blockers: List[str] = []
    if signal_blocker:
        blockers.append(signal_blocker)
    blockers.extend([str(row.get("error", "")) for row in sym_downloads if str(row.get("error", "")).strip()])
    blocker_text = " | ".join(sorted(set([b for b in blockers if b])))

    if signal_count <= 0 or signal_blocker:
        integrity = "BLOCKED"
    elif ready_windows == len(merged_windows) and uncovered_signals == 0 and parquet_ok == 1:
        integrity = "READY"
    elif ready_windows > 0 or partial_windows > 0 or signals_covered > 0:
        integrity = "PARTIAL"
    else:
        integrity = "BLOCKED"

    return {
        "symbol": symbol,
        "signals_total": int(signal_count),
        "merged_windows_total": int(len(merged_windows)),
        "windows_ready": int(ready_windows),
        "windows_partial": int(partial_windows),
        "windows_blocked": int(missing_windows),
        "signals_covered": int(signals_covered),
        "signals_uncovered": int(uncovered_signals),
        "signals_partially_covered": int(partially_covered_signals),
        "missing_window_rate": float(missing_rate),
        "parquet_readable": int(parquet_ok),
        "parquet_failures": int(parquet_failures),
        "integrity_status": integrity,
        "blockers": blocker_text,
    }


def write_report(
    *,
    out_dir: Path,
    symbol_contexts: Dict[str, SymbolContext],
    signal_quality_rows: Sequence[Dict[str, Any]],
    readiness_rows: Sequence[Dict[str, Any]],
    window_plan_rows: Sequence[Dict[str, Any]],
    download_rows: Sequence[Dict[str, Any]],
) -> None:
    quality_by_symbol = {str(row["symbol"]): dict(row) for row in signal_quality_rows}
    ready = [row["symbol"] for row in readiness_rows if row["integrity_status"] == "READY"]
    partial = [row["symbol"] for row in readiness_rows if row["integrity_status"] == "PARTIAL"]
    blocked = [row["symbol"] for row in readiness_rows if row["integrity_status"] == "BLOCKED"]
    lines: List[str] = []
    lines.append(f"# {RUN_PREFIX}")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Universe size: {len(UNIVERSE)}")
    lines.append(f"- Pre-buffer hours: {PRE_BUFFER_HOURS:.2f}")
    lines.append(f"- Post-buffer hours: {POST_BUFFER_HOURS:.2f}")
    lines.append(f"- Max merged window hours: {MAX_MERGED_WINDOW_HOURS:.2f}")
    lines.append(f"- Canonical 1h engine: `src/bot087/optim/ga.py`")
    lines.append(f"- 3m bounded slicer template: `scripts/execution_layer_3m_ict.py`")
    lines.append(f"- Combined harness reference: `scripts/phase_u_combined_1h3m_pilot.py`")
    lines.append(f"- SOL audit reference: `scripts/phase_a_model_a_audit.py`")
    lines.append(f"- Param scan source: `{PARAM_SCAN_CSV}`")
    lines.append(f"- Frozen fee model: `{FEE_MODEL_PATH}`")
    lines.append(f"- Frozen metrics definition: `{METRICS_DEF_PATH}`")
    lines.append(f"- SOL signal snapshot reference: `{LATEST_SOL_SIGNAL_SNAPSHOT}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- READY: {', '.join(ready) if ready else 'none'}")
    lines.append(f"- PARTIAL: {', '.join(partial) if partial else 'none'}")
    lines.append(f"- BLOCKED: {', '.join(blocked) if blocked else 'none'}")
    lines.append("")
    lines.append("## Reconstruction Summary")
    lines.append("")
    for row in readiness_rows:
        ctx = symbol_contexts[row["symbol"]]
        q = quality_by_symbol.get(str(row["symbol"]), {})
        notes = ", ".join(ctx.source_notes) if ctx.source_notes else "none"
        lines.append(
            f"- {row['symbol']}: signals={int(row['signals_total'])}, windows_ready={int(row['windows_ready'])}/{int(q.get('merged_windows_total', 0))}, "
            f"status={row['integrity_status']}, bucket={ctx.user_bucket}, source_notes={notes}"
        )
    lines.append("")
    lines.append("## Biggest Blockers")
    lines.append("")
    blocker_rows = [row for row in readiness_rows if row.get("blockers")]
    if blocker_rows:
        for row in blocker_rows:
            lines.append(f"- {row['symbol']}: {row['blockers']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Artifact Files")
    lines.append("")
    for name in [
        "universe_signal_timeline.csv",
        "universe_signal_timeline.parquet",
        "universe_3m_window_plan.csv",
        "universe_3m_download_manifest.csv",
        "universe_3m_data_quality.csv",
        "universe_symbol_readiness.csv",
        "run_manifest.json",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "universe_data_foundation_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    run_dir = ensure_dir(REPORTS_ROOT / f"{RUN_PREFIX}_{utc_tag()}")
    ensure_dir(RAW_CACHE_ROOT)
    ensure_dir(PARQUET_CACHE_ROOT)

    symbol_contexts = load_or_build_symbol_contexts()
    remote_state: Dict[str, Any] = {"disabled": False, "reason": ""}

    all_signal_frames: List[pd.DataFrame] = []
    signal_blockers: Dict[str, str] = {}
    per_symbol_windows: Dict[str, List[SignalWindow]] = {}
    merged_by_symbol: Dict[str, List[MergedWindow]] = {}
    window_plan_rows: List[Dict[str, Any]] = []
    download_rows: List[Dict[str, Any]] = []

    run_manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "symbol_universe": [
            {"symbol": symbol, "bucket_1h": USER_BUCKET[symbol]} for symbol in UNIVERSE
        ],
        "source_scripts": {
            "canonical_1h_engine": "src/bot087/optim/ga.py",
            "execution_3m_slicer": "scripts/execution_layer_3m_ict.py",
            "combined_harness": "scripts/phase_u_combined_1h3m_pilot.py",
            "sol_audit_reference": "scripts/phase_a_model_a_audit.py",
        },
        "data_sources": {
            "param_scan_csv": str(PARAM_SCAN_CSV),
            "legacy_exec_cache_root": str(LEGACY_CACHE_ROOT),
            "universal_raw_cache_root": str(RAW_CACHE_ROOT),
            "universal_parquet_cache_root": str(PARQUET_CACHE_ROOT),
        },
        "frozen_references": {
            "fee_model": str(FEE_MODEL_PATH),
            "metrics_definition": str(METRICS_DEF_PATH),
            "sol_signal_snapshot": str(LATEST_SOL_SIGNAL_SNAPSHOT),
        },
        "window_template": {
            "timeframe": WINDOW_TIMEFRAME,
            "pre_buffer_hours": PRE_BUFFER_HOURS,
            "post_buffer_hours": POST_BUFFER_HOURS,
            "max_merged_window_hours": MAX_MERGED_WINDOW_HOURS,
        },
        "symbol_context": {},
    }

    for symbol in UNIVERSE:
        ctx = symbol_contexts[symbol]
        run_manifest["symbol_context"][symbol] = {
            "bucket_1h": ctx.user_bucket,
            "params_file": str(ctx.params_file) if ctx.params_file is not None else "",
            "best_scan_score": float(ctx.best_score),
            "max_hold_hours": float(ctx.max_hold_hours) if np.isfinite(ctx.max_hold_hours) else None,
            "source_notes": list(ctx.source_notes),
        }

        sig_df, blocker = reconstruct_signals(ctx)
        signal_blockers[symbol] = blocker or ""
        if not sig_df.empty:
            all_signal_frames.append(sig_df)
            windows = build_signal_windows(sig_df)
            per_symbol_windows[symbol] = windows
            merged = merge_windows(symbol, windows)
            merged_by_symbol[symbol] = merged

            stats = window_coverage_stats(windows, merged)
            earliest = min(w.window_start for w in windows)
            latest = max(w.window_end for w in windows)
            for merged_window in merged:
                window_plan_rows.append(
                    {
                        "symbol": symbol,
                        "bucket_1h": ctx.user_bucket,
                        "window_id": merged_window.window_id,
                        "window_start_utc": merged_window.start_ts,
                        "window_end_utc": merged_window.end_ts,
                        "window_hours": float((merged_window.end_ts - merged_window.start_ts).total_seconds() / 3600.0),
                        "signals_in_window": int(merged_window.signal_count),
                        "signal_ids": "|".join(merged_window.signal_ids),
                        "earliest_needed_3m_utc": earliest,
                        "latest_needed_3m_utc": latest,
                        "unique_windows_for_symbol": int(len(merged)),
                        "raw_signal_windows_for_symbol": int(len(windows)),
                        "raw_signal_window_hours": float(stats["raw_signal_window_hours"]),
                        "merged_window_hours": float(stats["merged_window_hours"]),
                        "overlap_savings_hours": float(stats["overlap_savings_hours"]),
                        "overlap_savings_pct": float(stats["overlap_savings_pct"]),
                    }
                )

            for merged_window in merged:
                df3m, source_tag, error = fetch_window_data(
                    symbol=symbol,
                    start_ts=merged_window.start_ts,
                    end_ts=merged_window.end_ts,
                    remote_state=remote_state,
                )
                bars = int(len(df3m))
                expected_bars = int(max(0.0, math.ceil((merged_window.end_ts - merged_window.start_ts).total_seconds() / 180.0)))
                download_status = "BLOCKED"
                if bars >= expected_bars and bars > 0:
                    download_status = "READY"
                elif bars > 0:
                    download_status = "PARTIAL"

                raw_path = ""
                parquet_path = ""
                if bars > 0:
                    raw_fp, parquet_fp = persist_window(
                        df=df3m,
                        symbol=symbol,
                        window_id=merged_window.window_id,
                        start_ts=merged_window.start_ts,
                        end_ts=merged_window.end_ts,
                    )
                    raw_path = str(raw_fp)
                    parquet_path = str(parquet_fp)

                download_rows.append(
                    {
                        "symbol": symbol,
                        "bucket_1h": ctx.user_bucket,
                        "window_id": merged_window.window_id,
                        "window_start_utc": merged_window.start_ts,
                        "window_end_utc": merged_window.end_ts,
                        "download_source": source_tag,
                        "download_status": download_status,
                        "bars_3m": int(bars),
                        "expected_bars_3m": int(expected_bars),
                        "coverage_ratio": float(bars / expected_bars) if expected_bars > 0 else float("nan"),
                        "raw_path": raw_path,
                        "parquet_path": parquet_path,
                        "cache_hit": int(source_tag in {"universal_parquet_cache", "legacy_exec_cache_cover", "local_full_3m"}),
                        "error": error,
                    }
                )
        else:
            per_symbol_windows[symbol] = []
            merged_by_symbol[symbol] = []

    signal_timeline = (
        pd.concat(all_signal_frames, ignore_index=True)
        if all_signal_frames
        else pd.DataFrame(
            columns=[
                "symbol",
                "signal_id",
                "signal_time_utc",
                "entry_reference_time_utc",
                "expected_holding_horizon_hours",
                "execution_pre_buffer_hours",
                "execution_post_buffer_hours",
                "side",
                "bucket_1h",
                "model_source",
                "params_source",
            ]
        )
    )
    signal_timeline = signal_timeline.sort_values(["symbol", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    signal_timeline.to_csv(run_dir / "universe_signal_timeline.csv", index=False)
    signal_timeline.to_parquet(run_dir / "universe_signal_timeline.parquet", index=False)

    window_plan_df = pd.DataFrame(window_plan_rows).sort_values(["symbol", "window_start_utc", "window_id"]).reset_index(drop=True) if window_plan_rows else pd.DataFrame()
    window_plan_df.to_csv(run_dir / "universe_3m_window_plan.csv", index=False)

    download_df = pd.DataFrame(download_rows).sort_values(["symbol", "window_start_utc", "window_id"]).reset_index(drop=True) if download_rows else pd.DataFrame()
    download_df.to_csv(run_dir / "universe_3m_download_manifest.csv", index=False)

    signal_coverage_df = compute_signal_coverage(signal_timeline, download_rows)
    quality_rows: List[Dict[str, Any]] = []
    readiness_rows: List[Dict[str, Any]] = []
    for symbol in UNIVERSE:
        sig_count = int((signal_timeline["symbol"] == symbol).sum()) if not signal_timeline.empty else 0
        quality = readiness_from_quality(
            symbol=symbol,
            signal_count=sig_count,
            signal_blocker=signal_blockers.get(symbol, ""),
            merged_windows=merged_by_symbol.get(symbol, []),
            download_rows=download_rows,
            coverage_df=signal_coverage_df,
        )
        quality_rows.append(quality)
        readiness_rows.append(
            {
                "symbol": symbol,
                "bucket_1h": USER_BUCKET[symbol],
                "integrity_status": quality["integrity_status"],
                "signals_total": int(quality["signals_total"]),
                "windows_ready": int(quality["windows_ready"]),
                "windows_partial": int(quality["windows_partial"]),
                "windows_blocked": int(quality["windows_blocked"]),
                "blockers": quality["blockers"],
            }
        )

    quality_df = pd.DataFrame(quality_rows).sort_values("symbol").reset_index(drop=True)
    readiness_df = pd.DataFrame(readiness_rows).sort_values("symbol").reset_index(drop=True)
    quality_df.to_csv(run_dir / "universe_3m_data_quality.csv", index=False)
    readiness_df.to_csv(run_dir / "universe_symbol_readiness.csv", index=False)

    write_report(
        out_dir=run_dir,
        symbol_contexts=symbol_contexts,
        signal_quality_rows=quality_rows,
        readiness_rows=readiness_rows,
        window_plan_rows=window_plan_rows,
        download_rows=download_rows,
    )

    run_manifest["outputs"] = {
        "universe_signal_timeline_csv": str(run_dir / "universe_signal_timeline.csv"),
        "universe_signal_timeline_parquet": str(run_dir / "universe_signal_timeline.parquet"),
        "universe_3m_window_plan_csv": str(run_dir / "universe_3m_window_plan.csv"),
        "universe_3m_download_manifest_csv": str(run_dir / "universe_3m_download_manifest.csv"),
        "universe_3m_data_quality_csv": str(run_dir / "universe_3m_data_quality.csv"),
        "universe_symbol_readiness_csv": str(run_dir / "universe_symbol_readiness.csv"),
        "universe_data_foundation_report_md": str(run_dir / "universe_data_foundation_report.md"),
    }
    run_manifest["result_summary"] = {
        "ready_symbols": [row["symbol"] for row in readiness_rows if row["integrity_status"] == "READY"],
        "partial_symbols": [row["symbol"] for row in readiness_rows if row["integrity_status"] == "PARTIAL"],
        "blocked_symbols": [row["symbol"] for row in readiness_rows if row["integrity_status"] == "BLOCKED"],
        "remote_fetch_disabled": int(bool(remote_state.get("disabled", False))),
        "remote_fetch_disable_reason": str(remote_state.get("reason", "")),
    }
    json_dump(run_dir / "run_manifest.json", run_manifest)

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
