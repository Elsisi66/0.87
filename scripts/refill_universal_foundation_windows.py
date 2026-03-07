#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402
from scripts import phase_u_universal_data_foundation as foundation  # noqa: E402


def _latest_foundation_dir() -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("UNIVERSAL_DATA_FOUNDATION_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        if (p / "universe_3m_download_manifest.csv").exists() and (p / "universe_signal_timeline.csv").exists():
            return p.resolve()
    raise FileNotFoundError("No completed UNIVERSAL_DATA_FOUNDATION run directory found")


def _parse_symbols(raw: str) -> List[str]:
    if not str(raw).strip():
        return list(foundation.UNIVERSE)
    return [x.strip().upper() for x in str(raw).split(",") if x.strip()]


def _parse_statuses(raw: str) -> List[str]:
    if not str(raw).strip():
        return ["BLOCKED", "PARTIAL"]
    return [x.strip().upper() for x in str(raw).split(",") if x.strip()]


def _load_existing_window_df(row: Dict[str, Any]) -> pd.DataFrame:
    fp_raw = str(row.get("parquet_path", "")).strip()
    if not fp_raw:
        return pd.DataFrame(columns=foundation.RAW_SCHEMA_COLUMNS)
    fp = Path(fp_raw)
    if not fp.exists():
        return pd.DataFrame(columns=foundation.RAW_SCHEMA_COLUMNS)
    try:
        return foundation.normalize_raw(pd.read_parquet(fp))
    except Exception:
        return pd.DataFrame(columns=foundation.RAW_SCHEMA_COLUMNS)


def _merge_dataframes(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [foundation.normalize_raw(df) for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=foundation.RAW_SCHEMA_COLUMNS)
    out = pd.concat(non_empty, ignore_index=True)
    out = foundation.normalize_raw(out)
    return out.reset_index(drop=True)


def _direct_refetch_window(
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_retries: int,
    retry_base_sleep_sec: float,
    retry_max_sleep_sec: float,
    pause_sec: float,
) -> pd.DataFrame:
    df = exec3m._fetch_binance_klines(
        symbol=str(symbol).upper(),
        timeframe=foundation.WINDOW_TIMEFRAME,
        start_ts=pd.to_datetime(start_ts, utc=True),
        end_ts=pd.to_datetime(end_ts, utc=True),
        max_retries=int(max_retries),
        retry_base_sleep_sec=float(retry_base_sleep_sec),
        retry_max_sleep_sec=float(retry_max_sleep_sec),
        pause_sec=float(pause_sec),
    )
    return foundation.normalize_raw(df)


def _status_from_bars(bars: int, expected_bars: int) -> str:
    if bars >= expected_bars and bars > 0:
        return "READY"
    if bars > 0:
        return "PARTIAL"
    return "BLOCKED"


def _update_run_manifest(
    *,
    run_dir: Path,
    ready_rows: List[Dict[str, Any]],
    refill_summary: Dict[str, Any],
) -> None:
    manifest_fp = run_dir / "run_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_fp.exists():
        try:
            manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest["last_refill_utc"] = foundation.utc_now()
    manifest["refill_summary"] = refill_summary
    manifest["result_summary"] = {
        "ready_symbols": [row["symbol"] for row in ready_rows if str(row.get("integrity_status", "")).upper() == "READY"],
        "partial_symbols": [row["symbol"] for row in ready_rows if str(row.get("integrity_status", "")).upper() == "PARTIAL"],
        "blocked_symbols": [row["symbol"] for row in ready_rows if str(row.get("integrity_status", "")).upper() == "BLOCKED"],
    }
    foundation.json_dump(manifest_fp, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refill blocked/partial windows in an existing UNIVERSAL_DATA_FOUNDATION run")
    ap.add_argument("--run-dir", default="", help="Existing UNIVERSAL_DATA_FOUNDATION run dir; defaults to latest completed run.")
    ap.add_argument("--symbols", default="", help="Comma-separated symbol list; defaults to full universe.")
    ap.add_argument("--statuses", default="BLOCKED,PARTIAL", help="Comma-separated statuses to refill.")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_foundation_dir()
    symbols = _parse_symbols(args.symbols)
    statuses = set(_parse_statuses(args.statuses))

    signal_fp = run_dir / "universe_signal_timeline.csv"
    download_fp = run_dir / "universe_3m_download_manifest.csv"
    quality_fp = run_dir / "universe_3m_data_quality.csv"
    readiness_fp = run_dir / "universe_symbol_readiness.csv"
    if not signal_fp.exists() or not download_fp.exists():
        raise FileNotFoundError(f"Missing required foundation files under {run_dir}")

    signal_timeline = pd.read_csv(signal_fp)
    signal_timeline["signal_time_utc"] = pd.to_datetime(signal_timeline["signal_time_utc"], utc=True, errors="coerce")
    download_df = pd.read_csv(download_fp)
    for c in ["window_start_utc", "window_end_utc"]:
        download_df[c] = pd.to_datetime(download_df[c], utc=True, errors="coerce")

    target_mask = download_df["symbol"].astype(str).str.upper().isin(symbols) & download_df["download_status"].astype(str).str.upper().isin(statuses)
    target_idx = download_df.index[target_mask].tolist()

    fetched = 0
    improved = 0
    still_blocked = 0
    errors: List[str] = []

    for idx in target_idx:
        row = dict(download_df.loc[idx].to_dict())
        symbol = str(row["symbol"]).upper()
        start_ts = pd.to_datetime(row["window_start_utc"], utc=True)
        end_ts = pd.to_datetime(row["window_end_utc"], utc=True)
        expected_bars = int(pd.to_numeric(pd.Series([row.get("expected_bars_3m", 0)]), errors="coerce").fillna(0).iloc[0])
        prev_status = str(row.get("download_status", "")).upper()

        existing_df = _load_existing_window_df(row)
        try:
            fetched_df = _direct_refetch_window(
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                max_retries=int(args.max_fetch_retries),
                retry_base_sleep_sec=float(args.retry_base_sleep),
                retry_max_sleep_sec=float(args.retry_max_sleep),
                pause_sec=float(args.fetch_pause_sec),
            )
            merged_df = _merge_dataframes([existing_df, fetched_df])
            fetched += 1
            error_text = ""
            source = "remote_refill"
        except Exception as exc:
            merged_df = existing_df.copy()
            error_text = f"remote_refill_failed:{type(exc).__name__}:{exc}"
            errors.append(f"{symbol}:{error_text}")
            source = str(row.get("download_source", "")).strip() or "remote_refill_failed"

        bars = int(len(merged_df))
        new_status = _status_from_bars(bars=bars, expected_bars=expected_bars)
        if bars > 0:
            raw_fp, parquet_fp = foundation.persist_window(
                df=merged_df,
                symbol=symbol,
                window_id=str(row["window_id"]),
                start_ts=start_ts,
                end_ts=end_ts,
            )
            download_df.at[idx, "raw_path"] = str(raw_fp)
            download_df.at[idx, "parquet_path"] = str(parquet_fp)
        else:
            still_blocked += 1

        if prev_status != "READY" and new_status == "READY":
            improved += 1

        download_df.at[idx, "download_source"] = source
        download_df.at[idx, "download_status"] = new_status
        download_df.at[idx, "bars_3m"] = int(bars)
        download_df.at[idx, "coverage_ratio"] = float(bars / expected_bars) if expected_bars > 0 else float("nan")
        download_df.at[idx, "cache_hit"] = 0
        download_df.at[idx, "error"] = error_text

    download_df = download_df.sort_values(["symbol", "window_start_utc", "window_id"]).reset_index(drop=True)
    download_df.to_csv(download_fp, index=False)

    coverage_df = foundation.compute_signal_coverage(signal_timeline, download_df.to_dict("records"))
    quality_rows: List[Dict[str, Any]] = []
    readiness_rows: List[Dict[str, Any]] = []
    for symbol in foundation.UNIVERSE:
        sig_count = int((signal_timeline["symbol"].astype(str).str.upper() == symbol).sum())
        q = foundation.readiness_from_quality(
            symbol=symbol,
            signal_count=sig_count,
            signal_blocker="",
            merged_windows=[],
            download_rows=download_df.to_dict("records"),
            coverage_df=coverage_df,
        )
        merged_total = int((download_df["symbol"].astype(str).str.upper() == symbol).sum())
        q["merged_windows_total"] = merged_total
        quality_rows.append(q)
        readiness_rows.append(
            {
                "symbol": symbol,
                "bucket_1h": foundation.USER_BUCKET[symbol],
                "integrity_status": q["integrity_status"],
                "signals_total": int(q["signals_total"]),
                "windows_ready": int(q["windows_ready"]),
                "windows_partial": int(q["windows_partial"]),
                "windows_blocked": int(q["windows_blocked"]),
                "blockers": q["blockers"],
            }
        )

    quality_df = pd.DataFrame(quality_rows).sort_values("symbol").reset_index(drop=True)
    readiness_df = pd.DataFrame(readiness_rows).sort_values("symbol").reset_index(drop=True)
    quality_df.to_csv(quality_fp, index=False)
    readiness_df.to_csv(readiness_fp, index=False)

    foundation.write_report(
        out_dir=run_dir,
        symbol_contexts=foundation.load_or_build_symbol_contexts(),
        signal_quality_rows=quality_rows,
        readiness_rows=readiness_rows,
        window_plan_rows=[],
        download_rows=download_df.to_dict("records"),
    )

    refill_summary = {
        "run_dir": str(run_dir),
        "requested_symbols": symbols,
        "target_statuses": sorted(statuses),
        "target_windows": int(len(target_idx)),
        "remote_fetch_attempts": int(fetched),
        "windows_promoted_to_ready": int(improved),
        "windows_still_blocked_or_empty": int(still_blocked),
        "errors": errors[:50],
    }
    _update_run_manifest(run_dir=run_dir, ready_rows=quality_rows, refill_summary=refill_summary)
    print(json.dumps(refill_summary, sort_keys=True))


if __name__ == "__main__":
    main()
