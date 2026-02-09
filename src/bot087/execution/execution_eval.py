from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .cache import (
    _to_utc_ts,
    append_index_record,
    estimate_cache_size_gb,
    get_covering_paths,
    missing_ranges,
    write_parquet,
)
from .fetch_1s_klines import fetch_1s_klines
from .fetch_aggtrades import fetch_precision_1s_from_aggtrades


@dataclass(frozen=True)
class ExecutionEvalConfig:
    mode: str = "klines1s"  # klines1s | aggtrades
    window_sec: int = 15
    entry_delay_sec: int = 0
    exit_delay_sec: int = 0
    cap_gb: float = 20.0
    fee_bps: float = 7.0
    slippage_bps: float = 2.0
    pause_sec: float = 0.05
    cache_root: str = "data/processed/execution_1s"


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".json":
        raw = pd.read_json(path)
        df = raw
    else:
        raise ValueError(f"Unsupported trades extension: {path.suffix}")

    req_any = ["units", "size"]
    if not any(c in df.columns for c in req_any):
        raise ValueError(f"Trades file must have one of {req_any}")

    for col in ["entry_ts", "exit_ts"]:
        if col not in df.columns:
            raise ValueError(f"Trades file missing required column: {col}")
    for col in ["entry_px", "exit_px"]:
        if col not in df.columns:
            raise ValueError(f"Trades file missing required column: {col}")

    x = df.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], utc=True, errors="coerce")
    x["exit_ts"] = pd.to_datetime(x["exit_ts"], utc=True, errors="coerce")
    x["entry_px"] = pd.to_numeric(x["entry_px"], errors="coerce")
    x["exit_px"] = pd.to_numeric(x["exit_px"], errors="coerce")
    if "units" in x.columns:
        x["units"] = pd.to_numeric(x["units"], errors="coerce")
    else:
        x["units"] = pd.to_numeric(x["size"], errors="coerce")

    x = x.dropna(subset=["entry_ts", "exit_ts", "entry_px", "exit_px", "units"]).reset_index(drop=True)
    if x.empty:
        raise ValueError("Trades file contains no valid rows")
    return x


def _price_with_cost(px: float, side: str, fee_bps: float, slippage_bps: float) -> float:
    cost = (fee_bps + slippage_bps) / 1e4
    if side == "buy":
        return float(px * (1.0 + cost))
    return float(px * (1.0 - cost))


def _build_windows(trades: pd.DataFrame, window_sec: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ints: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    delta = pd.Timedelta(seconds=int(window_sec))
    for row in trades.itertuples(index=False):
        e0 = _to_utc_ts(row.entry_ts)
        x0 = _to_utc_ts(row.exit_ts)
        ints.append((e0, e0 + delta))
        ints.append((x0, x0 + delta))

    ints.sort(key=lambda z: z[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in ints:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _fetch_interval(symbol: str, mode: str, start_ts, end_ts, pause_sec: float) -> pd.DataFrame:
    if mode == "klines1s":
        return fetch_1s_klines(symbol=symbol, start_ts=start_ts, end_ts=end_ts, interval="1s", pause_sec=pause_sec)
    if mode == "aggtrades":
        return fetch_precision_1s_from_aggtrades(symbol=symbol, start_ts=start_ts, end_ts=end_ts, pause_sec=pause_sec)
    raise ValueError(f"Unsupported mode: {mode}")


def _ensure_cache_for_windows(symbol: str, cfg: ExecutionEvalConfig, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Path:
    cache_root = Path(cfg.cache_root)
    sym_mode_root = cache_root / symbol.upper() / cfg.mode.lower()
    sym_mode_root.mkdir(parents=True, exist_ok=True)
    chunks_dir = sym_mode_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for s, e in windows:
        for ms, me in missing_ranges(cache_root, symbol, cfg.mode, s, e):
            before = estimate_cache_size_gb(cache_root)
            if before >= float(cfg.cap_gb):
                raise RuntimeError(f"Storage cap reached ({before:.2f}GB >= {cfg.cap_gb:.2f}GB)")

            df = _fetch_interval(symbol=symbol, mode=cfg.mode, start_ts=ms, end_ts=me, pause_sec=cfg.pause_sec)
            if df.empty:
                continue

            tag = f"{ms.strftime('%Y%m%dT%H%M%S')}_{me.strftime('%Y%m%dT%H%M%S')}"
            out_path = chunks_dir / f"{symbol.upper()}_{cfg.mode}_{tag}.parquet"
            bytes_written = write_parquet(df, out_path)
            append_index_record(
                cache_root,
                symbol=symbol,
                mode=cfg.mode,
                source="binance_spot_api",
                start_ts=df["Timestamp"].min(),
                end_ts=df["Timestamp"].max(),
                rows=len(df),
                bytes_written=bytes_written,
                path=out_path,
            )

            after = estimate_cache_size_gb(cache_root)
            if after > float(cfg.cap_gb):
                raise RuntimeError(f"Storage cap exceeded after write ({after:.2f}GB > {cfg.cap_gb:.2f}GB)")

    return sym_mode_root


def _load_1s_for_range(symbol: str, cfg: ExecutionEvalConfig, start_ts, end_ts) -> pd.DataFrame:
    cache_root = Path(cfg.cache_root)
    paths = get_covering_paths(cache_root, symbol, cfg.mode, start_ts, end_ts)
    if not paths:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close"])

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p, columns=["Timestamp", "Open", "High", "Low", "Close"]))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close"])

    out = pd.concat(dfs, ignore_index=True)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp")
    out = out[(out["Timestamp"] >= _to_utc_ts(start_ts)) & (out["Timestamp"] <= _to_utc_ts(end_ts))].reset_index(drop=True)
    return out


def _first_open_in_window(
    sec: pd.DataFrame,
    ts: pd.Timestamp,
    window_sec: int,
    delay_sec: int,
    fallback_px: float,
) -> float:
    start = _to_utc_ts(ts) + pd.Timedelta(seconds=int(delay_sec))
    end = start + pd.Timedelta(seconds=int(window_sec))
    x = sec[(sec["Timestamp"] >= start) & (sec["Timestamp"] < end)]
    if x.empty:
        return float(fallback_px)
    return float(x.iloc[0]["Open"])


def _adjust_trades_with_1s(trades: pd.DataFrame, sec_df: pd.DataFrame, cfg: ExecutionEvalConfig) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for r in trades.itertuples(index=False):
        entry_px_raw = _first_open_in_window(
            sec_df,
            r.entry_ts,
            cfg.window_sec,
            cfg.entry_delay_sec,
            float(r.entry_px),
        )
        exit_px_raw = _first_open_in_window(
            sec_df,
            r.exit_ts,
            cfg.window_sec,
            cfg.exit_delay_sec,
            float(r.exit_px),
        )

        entry_px_adj = _price_with_cost(entry_px_raw, side="buy", fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)
        exit_px_adj = _price_with_cost(exit_px_raw, side="sell", fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)

        original_net = (float(r.exit_px) - float(r.entry_px)) * float(r.units)
        adjusted_net = (exit_px_adj - entry_px_adj) * float(r.units)

        out_rows.append(
            {
                "entry_ts": str(_to_utc_ts(r.entry_ts)),
                "exit_ts": str(_to_utc_ts(r.exit_ts)),
                "units": float(r.units),
                "entry_px_orig": float(r.entry_px),
                "exit_px_orig": float(r.exit_px),
                "entry_px_adj": float(entry_px_adj),
                "exit_px_adj": float(exit_px_adj),
                "net_pnl_orig": float(original_net),
                "net_pnl_adj": float(adjusted_net),
                "pnl_delta": float(adjusted_net - original_net),
            }
        )

    out = pd.DataFrame(out_rows)
    return out


def _summary_from_adjusted(adjusted: pd.DataFrame) -> Dict[str, Any]:
    if adjusted.empty:
        return {
            "trades": 0,
            "orig_net": 0.0,
            "adj_net": 0.0,
            "delta_net": 0.0,
            "improved_trades": 0,
            "improved_ratio": 0.0,
        }
    orig = float(adjusted["net_pnl_orig"].sum())
    adj = float(adjusted["net_pnl_adj"].sum())
    improved = int((adjusted["pnl_delta"] > 0.0).sum())
    return {
        "trades": int(len(adjusted)),
        "orig_net": orig,
        "adj_net": adj,
        "delta_net": float(adj - orig),
        "improved_trades": improved,
        "improved_ratio": float(improved / max(1, len(adjusted))),
    }


def load_and_prepare_execution_data(
    *,
    symbol: str,
    trades_path: str,
    cfg: ExecutionEvalConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    trades = _load_trades(Path(trades_path))
    windows = _build_windows(trades, cfg.window_sec)
    if not windows:
        raise ValueError("No execution windows found from trades file")

    global_start = min(s for s, _ in windows)
    global_end = max(e for _, e in windows)

    _ensure_cache_for_windows(symbol=symbol, cfg=cfg, windows=windows)
    sec_df = _load_1s_for_range(symbol=symbol, cfg=cfg, start_ts=global_start, end_ts=global_end)
    if sec_df.empty:
        raise RuntimeError("No 1s data available after fetch/cache stage")
    return trades, sec_df, {"start": str(global_start), "end": str(global_end)}


def evaluate_with_sec_data(
    *,
    symbol: str,
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    adjusted = _adjust_trades_with_1s(trades, sec_df, cfg)
    summary = _summary_from_adjusted(adjusted)
    return {
        "symbol": symbol.upper(),
        "mode": cfg.mode,
        "window_sec": int(cfg.window_sec),
        "entry_delay_sec": int(cfg.entry_delay_sec),
        "exit_delay_sec": int(cfg.exit_delay_sec),
        "summary": summary,
        "adjusted_trades": adjusted,
    }


def evaluate_execution_from_trades(
    *,
    symbol: str,
    trades_path: str,
    cfg: ExecutionEvalConfig,
) -> Dict[str, Any]:
    trades, sec_df, tr = load_and_prepare_execution_data(symbol=symbol, trades_path=trades_path, cfg=cfg)
    out = evaluate_with_sec_data(symbol=symbol, trades=trades, sec_df=sec_df, cfg=cfg)
    out["cache_root"] = str(Path(cfg.cache_root).resolve())
    out["time_range"] = tr
    return out
