from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (_p / "data").is_dir() and (_p / "src").is_dir():
        PROJECT_ROOT = _p
        break

os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.execution.http_retry import http_get_json_with_retry  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from src.bot087.optim import ga_short  # noqa: E402

BINANCE_BASE = "https://api.binance.com"
ONE_HOUR_MS = 60 * 60 * 1000
_RUNID_RE = re.compile(r"[0-9]{8}_[0-9]{6}")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_STABLE_TOKENS = {
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "FDUSD",
    "USDP",
    "PAX",
    "DAI",
    "EURT",
    "AEUR",
    "SUSD",
}
_FIAT_TOKENS = {
    "USD",
    "EUR",
    "GBP",
    "TRY",
    "RUB",
    "UAH",
    "BRL",
    "AUD",
    "BIDR",
    "NGN",
    "ZAR",
    "ARS",
    "IDRT",
}
_EXCLUDED_SYMBOLS = {
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "BUSDUSDT",
    "USDPUSDT",
    "EURUSDT",
}


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} {msg}", flush=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _is_date_only(value: str) -> bool:
    return bool(_DATE_ONLY_RE.match(str(value).strip()))


def _inclusive_end_ts(end_value: str) -> pd.Timestamp:
    ts = _to_utc_ts(end_value)
    if _is_date_only(end_value):
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _exclusive_end_str(end_value: str) -> str:
    ts = _to_utc_ts(end_value)
    if _is_date_only(end_value):
        ts = ts + pd.Timedelta(days=1)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_thread_caps() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(os.environ.get(var, "1"))
    _log(
        "thread_caps="
        f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} "
        f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']} "
        f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']} "
        f"NUMEXPR_NUM_THREADS={os.environ['NUMEXPR_NUM_THREADS']}"
    )


def _extract_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def _safe_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _binance_get_json(
    path: str,
    params: Dict[str, Any],
    *,
    retries: int,
    base_sleep_sec: float,
    max_sleep_sec: float,
) -> Any:
    pp = {str(k): str(v) for k, v in params.items()}
    return http_get_json_with_retry(
        base=BINANCE_BASE,
        path=path,
        params=pp,
        timeout=30,
        max_retries=int(retries),
        retry_base_sleep_sec=float(base_sleep_sec),
        retry_max_sleep_sec=float(max_sleep_sec),
    )


def _is_leveraged_token(base_asset: str, symbol: str) -> bool:
    b = str(base_asset).upper()
    s = str(symbol).upper()
    for suf in ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "2L", "2S"):
        if b.endswith(suf):
            return True
    return bool(re.search(r"(UP|DOWN|BULL|BEAR)USDT$", s))


def _history_years(earliest_ms: int, end_ts: pd.Timestamp) -> float:
    end_ms = int(end_ts.value // 1_000_000)
    span_sec = max(0.0, float(end_ms - int(earliest_ms)) / 1000.0)
    return float(span_sec / (365.25 * 24.0 * 3600.0))


def _local_universe(
    *,
    count: int,
    min_history_years: float,
    end_ts: pd.Timestamp,
) -> Dict[str, Any]:
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    symbols = sorted(
        {
            p.name.split("_", 1)[0].upper()
            for p in full_dir.glob("*_1h_full.parquet")
            if p.is_file()
        }
    )
    selected: List[Dict[str, Any]] = []
    for sym in symbols:
        fp = full_dir / f"{sym}_1h_full.parquet"
        try:
            ts = pd.read_parquet(fp, columns=["Timestamp"])
            if ts.empty:
                continue
            t0 = pd.to_datetime(ts["Timestamp"], utc=True, errors="coerce").dropna().min()
            if pd.isna(t0):
                continue
        except Exception:
            continue
        earliest_ms = int(pd.Timestamp(t0).value // 1_000_000)
        years = _history_years(earliest_ms, end_ts)
        if years < float(min_history_years):
            continue
        selected.append(
            {
                "symbol": sym,
                "base_asset": sym[:-4] if sym.endswith("USDT") else sym,
                "quote_asset": "USDT" if sym.endswith("USDT") else "",
                "quote_volume_24h": None,
                "earliest_1h_ms": int(earliest_ms),
                "earliest_1h_ts": str(pd.to_datetime(earliest_ms, unit="ms", utc=True)),
                "history_years": float(round(years, 3)),
                "source": "local_cache",
            }
        )
        if len(selected) >= int(count):
            break
    return {
        "source": "local_fallback",
        "selected": selected,
        "warning": (
            "Binance API unavailable; selected from local cached symbols only. "
            f"requested={count} available={len(selected)}"
        ),
    }


def _fetch_universe_from_binance(args: argparse.Namespace, end_ts: pd.Timestamp) -> Dict[str, Any]:
    _log("universe: fetching exchangeInfo + ticker/24hr from Binance")
    exchange = _binance_get_json(
        "/api/v3/exchangeInfo",
        {},
        retries=args.http_retries,
        base_sleep_sec=args.http_retry_base_sleep,
        max_sleep_sec=args.http_retry_max_sleep,
    )
    ticker24 = _binance_get_json(
        "/api/v3/ticker/24hr",
        {},
        retries=args.http_retries,
        base_sleep_sec=args.http_retry_base_sleep,
        max_sleep_sec=args.http_retry_max_sleep,
    )
    if not isinstance(exchange, dict) or not isinstance(exchange.get("symbols"), list):
        raise RuntimeError("Unexpected exchangeInfo payload")
    if not isinstance(ticker24, list):
        raise RuntimeError("Unexpected ticker/24hr payload")

    qv_map: Dict[str, float] = {}
    for row in ticker24:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper().strip()
        if not sym:
            continue
        try:
            qv_map[sym] = float(row.get("quoteVolume", 0.0) or 0.0)
        except Exception:
            qv_map[sym] = 0.0

    candidates: List[Dict[str, Any]] = []
    for row in exchange["symbols"]:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper().strip()
        base_asset = str(row.get("baseAsset", "")).upper().strip()
        quote_asset = str(row.get("quoteAsset", "")).upper().strip()
        status = str(row.get("status", "")).upper().strip()
        is_spot = bool(row.get("isSpotTradingAllowed", False))
        if not sym or quote_asset != "USDT":
            continue
        if status != "TRADING" or not is_spot:
            continue
        if sym in _EXCLUDED_SYMBOLS:
            continue
        if base_asset in _STABLE_TOKENS or base_asset in _FIAT_TOKENS:
            continue
        if _is_leveraged_token(base_asset, sym):
            continue
        qv = float(qv_map.get(sym, 0.0))
        if qv <= 0.0:
            continue
        candidates.append(
            {
                "symbol": sym,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "quote_volume_24h": qv,
            }
        )

    candidates.sort(key=lambda x: float(x["quote_volume_24h"]), reverse=True)
    if not candidates:
        raise RuntimeError("No Binance USDT spot candidates after filters")

    max_probes = int(max(args.count, args.count * args.candidate_multiplier))
    if int(args.max_probes) > 0:
        max_probes = min(max_probes, int(args.max_probes))

    selected: List[Dict[str, Any]] = []
    checked = 0
    _log(
        f"universe: candidates={len(candidates)} max_probes={max_probes} "
        f"target_count={args.count} min_history_years={args.min_history_years}"
    )
    for cand in candidates:
        if checked >= max_probes:
            break
        checked += 1
        sym = str(cand["symbol"])
        try:
            kline = _binance_get_json(
                "/api/v3/klines",
                {"symbol": sym, "interval": "1h", "startTime": 0, "limit": 1},
                retries=args.http_retries,
                base_sleep_sec=args.http_retry_base_sleep,
                max_sleep_sec=args.http_retry_max_sleep,
            )
            if not isinstance(kline, list) or not kline:
                continue
            earliest_ms = int(kline[0][0])
        except Exception:
            continue
        yrs = _history_years(earliest_ms, end_ts)
        if yrs < float(args.min_history_years):
            continue
        selected.append(
            {
                **cand,
                "earliest_1h_ms": int(earliest_ms),
                "earliest_1h_ts": str(pd.to_datetime(earliest_ms, unit="ms", utc=True)),
                "history_years": float(round(yrs, 3)),
                "source": "binance_api",
            }
        )
        _log(
            f"universe: selected={sym} quoteVolume={cand['quote_volume_24h']:.2f} "
            f"history_years={yrs:.2f} ({len(selected)}/{args.count})"
        )
        if len(selected) >= int(args.count):
            break

    return {
        "source": "binance_api",
        "selected": selected,
        "checked_candidates": checked,
        "total_candidates": len(candidates),
    }


def _build_universe(args: argparse.Namespace) -> Dict[str, Any]:
    universe_path = (PROJECT_ROOT / args.universe_file).resolve()
    end_ts = _inclusive_end_ts(args.end)

    if universe_path.exists() and not bool(args.refresh_universe):
        payload = _read_json(universe_path)
        if isinstance(payload, dict) and isinstance(payload.get("selected"), list):
            _log(f"universe: using existing file {universe_path}")
            return payload

    try:
        core = _fetch_universe_from_binance(args, end_ts=end_ts)
    except Exception as ex:
        if not bool(args.offline_fallback):
            raise
        _log(f"universe: Binance API unavailable ({type(ex).__name__}: {ex}); falling back to local cache")
        core = _local_universe(
            count=int(args.count),
            min_history_years=float(args.min_history_years),
            end_ts=end_ts,
        )

    payload = {
        "generated_utc": _utc_now_iso(),
        "requested_count": int(args.count),
        "selected_count": int(len(core.get("selected", []))),
        "min_history_years": float(args.min_history_years),
        "start": str(args.start),
        "end": str(args.end),
        "source": str(core.get("source", "unknown")),
        "checked_candidates": int(core.get("checked_candidates", 0)),
        "total_candidates": int(core.get("total_candidates", 0)),
        "warning": core.get("warning"),
        "selected": core.get("selected", []),
    }
    _write_json(universe_path, payload)
    _log(f"universe: wrote {universe_path} selected_count={payload['selected_count']}")
    return payload


def _default_long_seed() -> Dict[str, Any]:
    return {
        "entry_rsi_min": 52.0,
        "entry_rsi_max": 64.0,
        "entry_rsi_buffer": 2.5,
        "willr_floor": -100.0,
        "willr_by_cycle": [-78.0, -32.0, -90.0, -18.0, -50.0],
        "ema_span": 35,
        "ema_trend_long": 120,
        "ema_align": True,
        "require_ema200_slope": True,
        "adx_min": 18.0,
        "require_plus_di": True,
        "tp_mult_by_cycle": [1.035, 1.08, 1.05, 1.07, 1.05],
        "sl_mult_by_cycle": [0.985, 0.98, 0.96, 0.99, 0.97],
        "exit_rsi_by_cycle": [50.0, 56.0, 50.0, 62.0, 52.0],
        "risk_per_trade": 0.02,
        "max_allocation": 0.7,
        "atr_k": 1.0,
        "trade_cycles": [1, 3],
        "max_hold_hours": 48,
        "cycle_shift": 1,
        "cycle_fill": 2,
        "two_candle_confirm": False,
        "require_trade_cycles": True,
    }


def _default_short_seed() -> Dict[str, Any]:
    return {
        "entry_rsi_min": 58.0,
        "entry_rsi_max": 86.0,
        "willr_by_cycle": [-22.0, -18.0, -30.0, -16.0, -24.0],
        "tp_mult_by_cycle": [0.96, 0.94, 0.95, 0.93, 0.95],
        "sl_mult_by_cycle": [1.03, 1.04, 1.05, 1.03, 1.04],
        "exit_rsi_by_cycle": [45.0, 50.0, 45.0, 55.0, 50.0],
        "ema_span": 35,
        "ema_trend_long": 120,
        "ema_align": True,
        "require_ema200_slope": True,
        "adx_min": 18.0,
        "require_minus_di": True,
        "risk_per_trade": 0.02,
        "max_allocation": 0.7,
        "atr_k": 1.0,
        "trade_cycles": [0, 4],
        "max_hold_hours": 48,
        "cycle_shift": 1,
        "cycle_fill": 2,
        "two_candle_confirm": False,
        "require_trade_cycles": True,
    }


def _load_long_seed(symbol: str) -> Dict[str, Any]:
    params_root = PROJECT_ROOT / "data" / "metadata" / "params"
    candidates = [
        params_root / f"{symbol}_C13_active_params_long.json",
        params_root / f"{symbol}_active_params.json",
        params_root / f"{symbol}_seed_params.json",
        params_root / "BTCUSDT_C13_active_params.json",
        params_root / "BTCUSDT_active_params.json",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            return ga_long._norm_params(_extract_params(_read_json(fp)))
        except Exception:
            continue
    return ga_long._norm_params(_default_long_seed())


def _load_short_seed(symbol: str) -> Dict[str, Any]:
    params_root = PROJECT_ROOT / "data" / "metadata" / "params"
    candidates = [
        params_root / f"{symbol}_C13_active_params_short.json",
        params_root / f"{symbol}_active_params_short.json",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            return ga_short._norm_params(_extract_params(_read_json(fp)))
        except Exception:
            continue
    return ga_short._norm_params(_default_short_seed())


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Timestamp" not in out.columns:
        raise ValueError("missing Timestamp column")
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp"])
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in out.columns:
            raise ValueError(f"missing {c} column")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    return out


def _load_existing_full(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    try:
        return _normalize_ohlcv(pd.read_parquet(fp))
    except Exception:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])


def _fetch_1h_klines(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if end_ts <= start_ts:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    rows: List[List[Any]] = []
    cursor = int(start_ts.value // 1_000_000)
    end_ms = int(end_ts.value // 1_000_000)
    while cursor < end_ms:
        payload = _binance_get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": max(cursor, end_ms - 1),
                "limit": 1000,
            },
            retries=args.http_retries,
            base_sleep_sec=args.http_retry_base_sleep,
            max_sleep_sec=args.http_retry_max_sleep,
        )
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        last_open = int(payload[-1][0])
        next_cursor = last_open + ONE_HOUR_MS
        if next_cursor <= cursor:
            next_cursor = cursor + ONE_HOUR_MS
        cursor = next_cursor
        if len(payload) < 1000 and cursor >= end_ms:
            break
    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

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
        }
    )
    out = _normalize_ohlcv(out)
    out = out[(out["Timestamp"] >= start_ts) & (out["Timestamp"] <= end_ts)].reset_index(drop=True)
    return out


def _save_symbol_feature_parquets(symbol: str, df_features: pd.DataFrame) -> None:
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_fp = full_dir / f"{symbol}_1h_full.parquet"
    feat_fp = full_dir / f"{symbol}_1h_features.parquet"
    df_features.to_parquet(full_fp, index=False)
    df_features.to_parquet(feat_fp, index=False)


def _prepare_symbol_dataset(symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    _log(f"{symbol}: ensure 1h data/features (resumable)")
    start_ts = _to_utc_ts(args.start)
    end_ts = _inclusive_end_ts(args.end)
    df_existing = _load_existing_full(symbol)
    missing_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if bool(args.force_download) or df_existing.empty:
        missing_intervals = [(start_ts, end_ts)]
    else:
        cur_min = pd.Timestamp(df_existing["Timestamp"].min())
        cur_max = pd.Timestamp(df_existing["Timestamp"].max())
        if start_ts < cur_min:
            missing_intervals.append((start_ts, cur_min - pd.Timedelta(milliseconds=1)))
        if end_ts > cur_max:
            missing_intervals.append((cur_max + pd.Timedelta(hours=1), end_ts))

    fetched_parts: List[pd.DataFrame] = []
    for ms, me in missing_intervals:
        if me <= ms:
            continue
        _log(f"{symbol}: fetching missing 1h range {ms} -> {me}")
        try:
            part = _fetch_1h_klines(symbol, ms, me, args)
        except Exception as ex:
            if bool(args.offline_fallback) and not df_existing.empty:
                _log(
                    f"{symbol}: missing-range fetch failed in offline fallback mode "
                    f"({type(ex).__name__}: {ex}); using existing cached range only"
                )
                continue
            raise
        if not part.empty:
            fetched_parts.append(part)

    if fetched_parts:
        merged = pd.concat([df_existing] + fetched_parts, ignore_index=True)
    else:
        merged = df_existing.copy()
    merged = _normalize_ohlcv(merged) if not merged.empty else merged
    merged = merged[(merged["Timestamp"] >= start_ts) & (merged["Timestamp"] <= end_ts)].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"{symbol}: no data in requested range")

    # Reuse existing indicator code path (no 1s dependency).
    seed = _load_long_seed(symbol)
    feat = ga_long._ensure_indicators(merged.copy(), ga_long._norm_params(seed))
    feat["Timestamp"] = pd.to_datetime(feat["Timestamp"], utc=True, errors="coerce")
    feat = feat.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    _save_symbol_feature_parquets(symbol, feat)
    return feat


def _build_train_test(df: pd.DataFrame, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts0 = _to_utc_ts(test_start)
    ts1 = _inclusive_end_ts(test_end)
    train = df[df["Timestamp"] < ts0].reset_index(drop=True)
    test = df[(df["Timestamp"] >= ts0) & (df["Timestamp"] <= ts1)].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("train split empty (check --test-start)")
    if test.empty:
        raise RuntimeError("test split empty (check --test-start/--test-end)")
    return train, test


def _long_cfg(args: argparse.Namespace) -> ga_long.GAConfig:
    return ga_long.GAConfig(
        pop_size=int(args.pop_size),
        generations=int(args.ga_generations),
        n_procs=int(args.eval_procs),
        mc_splits=int(args.mc_splits),
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
        min_trades_train=int(args.min_trades_train),
        min_trades_val=int(args.min_trades_val),
        resume=True,
        early_stop_patience=int(args.early_stop),
    )


def _short_cfg(args: argparse.Namespace) -> ga_short.GAConfig:
    return ga_short.GAConfig(
        pop_size=int(args.pop_size),
        generations=int(args.ga_generations),
        n_procs=int(args.eval_procs),
        mc_splits=int(args.mc_splits),
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
        min_trades_train=int(args.min_trades_train),
        min_trades_val=int(args.min_trades_val),
        resume=True,
    )


def _long_stability(df_train: pd.DataFrame, params: Dict[str, Any], cfg: ga_long.GAConfig) -> float:
    try:
        splits = ga_long.make_mc_splits(df_train, cfg, gen=9913)
    except Exception:
        return 0.0
    if not splits:
        return 0.0
    pos = 0
    for _tr0, _tr1, va0, va1, _te0, _te1 in splits:
        dval = df_train.iloc[va0:va1].reset_index(drop=True)
        if dval.empty:
            continue
        _, m = ga_long.run_backtest_long_only(
            dval,
            symbol="VAL",
            p=params,
            initial_equity=float(cfg.initial_equity),
            fee_bps=float(cfg.fee_bps),
            slippage_bps=float(cfg.slippage_bps),
            collect_trades=False,
        )
        if float(m.get("net_profit", 0.0)) > 0.0:
            pos += 1
    return float(pos / max(1, len(splits)))


def _short_stability(df_train: pd.DataFrame, params: Dict[str, Any], cfg: ga_short.GAConfig) -> float:
    try:
        splits = ga_short.make_mc_splits(df_train, cfg, gen=9917)
    except Exception:
        return 0.0
    if not splits:
        return 0.0
    pos = 0
    for _tr0, _tr1, va0, va1, _te0, _te1 in splits:
        dval = df_train.iloc[va0:va1].reset_index(drop=True)
        if dval.empty:
            continue
        _, m = ga_short.run_backtest_short_only(
            dval,
            symbol="VAL",
            p=params,
            initial_equity=float(cfg.initial_equity),
            fee_bps=float(cfg.fee_bps),
            slippage_bps=float(cfg.slippage_bps),
        )
        if float(m.get("net_profit", 0.0)) > 0.0:
            pos += 1
    return float(pos / max(1, len(splits)))


def _pass_fail(metrics: Dict[str, Any], stability: float, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if float(metrics.get("profit_factor", 0.0)) < float(args.pass_pf_min):
        reasons.append(f"pf<{args.pass_pf_min}")
    if float(metrics.get("max_dd", 1.0)) > float(args.pass_dd_max):
        reasons.append(f"dd>{args.pass_dd_max}")
    if float(metrics.get("trades", 0.0)) < float(args.pass_trades_min):
        reasons.append(f"trades<{args.pass_trades_min}")
    if float(stability) < float(args.pass_stability_min):
        reasons.append(f"stability<{args.pass_stability_min}")
    return len(reasons) == 0, reasons


def _save_side_params(
    *,
    symbol: str,
    side: str,
    params: Dict[str, Any],
    pipeline_run_id: str,
    ga_report: Dict[str, Any],
    test_metrics: Dict[str, Any],
    stability: float,
    passed: bool,
    reasons: List[str],
) -> Path:
    out = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_C13_active_params_{side}.json"
    payload = {
        "symbol": symbol,
        "side": side,
        "params": params,
        "meta": {
            "saved_at_utc": _utc_now_iso(),
            "pipeline_run_id": pipeline_run_id,
            "ga_run_id": ga_report.get("run_id"),
            "ga_symbol": ga_report.get("symbol"),
            "ga_saved": ga_report.get("saved", {}),
            "test_metrics": test_metrics,
            "stability": float(stability),
            "pass": bool(passed),
            "reasons": reasons,
        },
    }
    _write_json(out, payload)
    return out


def _optimize_long(
    symbol: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    args: argparse.Namespace,
    pipeline_run_id: str,
) -> Dict[str, Any]:
    cfg = _long_cfg(args)
    seed = _load_long_seed(symbol)
    ga_symbol = f"{symbol}__UNIVERSE_LONG"
    _log(f"{symbol} long: GA start pop={cfg.pop_size} gens={cfg.generations} eval_procs={cfg.n_procs}")
    best_params, report = ga_long.run_ga_montecarlo(
        symbol=ga_symbol,
        df=df_train,
        seed_params=seed,
        cfg=cfg,
    )
    best_params = ga_long._norm_params(best_params)
    stability = _long_stability(df_train, best_params, cfg)
    _, test_metrics = ga_long.run_backtest_long_only(
        df_test,
        symbol=symbol,
        p=best_params,
        initial_equity=float(args.initial_equity),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        collect_trades=False,
    )
    passed, reasons = _pass_fail(test_metrics, stability, args)
    param_path = _save_side_params(
        symbol=symbol,
        side="long",
        params=best_params,
        pipeline_run_id=pipeline_run_id,
        ga_report=report,
        test_metrics=test_metrics,
        stability=stability,
        passed=passed,
        reasons=reasons,
    )
    return {
        "symbol": symbol,
        "side": "long",
        "test_net": float(test_metrics.get("net_profit", 0.0)),
        "test_pf": float(test_metrics.get("profit_factor", 0.0)),
        "test_dd": float(test_metrics.get("max_dd", 1.0)),
        "test_trades": float(test_metrics.get("trades", 0.0)),
        "stability": float(stability),
        "PASS/FAIL": "PASS" if passed else "FAIL",
        "fail_reasons": ",".join(reasons),
        "param_path": str(param_path.resolve()),
        "pipeline_run_id": pipeline_run_id,
        "ga_run_id": str(report.get("run_id", "")),
        "ga_symbol": ga_symbol,
        "updated_utc": _utc_now_iso(),
    }


def _optimize_short(
    symbol: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    args: argparse.Namespace,
    pipeline_run_id: str,
) -> Dict[str, Any]:
    cfg = _short_cfg(args)
    seed = _load_short_seed(symbol)
    ga_symbol = f"{symbol}__UNIVERSE_SHORT"
    _log(f"{symbol} short: GA start pop={cfg.pop_size} gens={cfg.generations} eval_procs={cfg.n_procs}")
    best_params, report = ga_short.run_ga_montecarlo(
        symbol=ga_symbol,
        df=df_train,
        seed_params=seed,
        cfg=cfg,
    )
    best_params = ga_short._norm_params(best_params)
    stability = _short_stability(df_train, best_params, cfg)
    _, test_metrics = ga_short.run_backtest_short_only(
        df_test,
        symbol=symbol,
        p=best_params,
        initial_equity=float(args.initial_equity),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
    )
    passed, reasons = _pass_fail(test_metrics, stability, args)
    param_path = _save_side_params(
        symbol=symbol,
        side="short",
        params=best_params,
        pipeline_run_id=pipeline_run_id,
        ga_report=report,
        test_metrics=test_metrics,
        stability=stability,
        passed=passed,
        reasons=reasons,
    )
    return {
        "symbol": symbol,
        "side": "short",
        "test_net": float(test_metrics.get("net_profit", 0.0)),
        "test_pf": float(test_metrics.get("profit_factor", 0.0)),
        "test_dd": float(test_metrics.get("max_dd", 1.0)),
        "test_trades": float(test_metrics.get("trades", 0.0)),
        "stability": float(stability),
        "PASS/FAIL": "PASS" if passed else "FAIL",
        "fail_reasons": ",".join(reasons),
        "param_path": str(param_path.resolve()),
        "pipeline_run_id": pipeline_run_id,
        "ga_run_id": str(report.get("run_id", "")),
        "ga_symbol": ga_symbol,
        "updated_utc": _utc_now_iso(),
    }


def _load_summary(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in df.to_dict("records"):
        sym = str(row.get("symbol", "")).upper().strip()
        side = str(row.get("side", "")).lower().strip()
        if sym and side:
            out[(sym, side)] = row
    return out


def _write_summary(path: Path, rows: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[k] for k in sorted(rows.keys())]
    if not ordered:
        pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "test_net",
                "test_pf",
                "test_dd",
                "test_trades",
                "stability",
                "PASS/FAIL",
                "fail_reasons",
                "param_path",
                "pipeline_run_id",
                "ga_run_id",
                "ga_symbol",
                "updated_utc",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(ordered).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--min-history-years", type=float, default=4.0)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--test-end", default="2025-12-31")

    ap.add_argument("--eval-procs", type=int, default=3)
    ap.add_argument("--fetch-workers", type=int, default=1)
    ap.add_argument("--ga-generations", type=int, default=30)
    ap.add_argument("--pop-size", type=int, default=24)
    ap.add_argument("--mc-splits", type=int, default=6)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--min-trades-train", type=int, default=40)
    ap.add_argument("--min-trades-val", type=int, default=15)

    ap.add_argument("--pass-pf-min", type=float, default=1.2)
    ap.add_argument("--pass-dd-max", type=float, default=0.20)
    ap.add_argument("--pass-trades-min", type=float, default=30.0)
    ap.add_argument("--pass-stability-min", type=float, default=0.75)

    ap.add_argument("--candidate-multiplier", type=int, default=5)
    ap.add_argument("--max-probes", type=int, default=200)
    ap.add_argument("--http-retries", type=int, default=8)
    ap.add_argument("--http-retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--http-retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--offline-fallback", dest="offline_fallback", action="store_true", default=True)
    ap.add_argument("--no-offline-fallback", dest="offline_fallback", action="store_false")

    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-process", action="store_true")
    ap.add_argument("--force-optimize", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")

    ap.add_argument("--universe-file", default="artifacts/universe/universe_20.json")
    ap.add_argument("--summary-file", default="artifacts/universe/summary.csv")
    args = ap.parse_args()

    _apply_thread_caps()
    _log(
        f"pipeline_start count={args.count} min_history_years={args.min_history_years} "
        f"eval_procs={args.eval_procs} fetch_workers={args.fetch_workers}"
    )

    pipeline_run_id = _safe_run_id()
    universe_payload = _build_universe(args)
    selected = list(universe_payload.get("selected", []))
    if not selected:
        raise SystemExit("No symbols selected for universe")

    symbols = [str(row.get("symbol", "")).upper().strip() for row in selected if str(row.get("symbol", "")).strip()]
    symbols = list(dict.fromkeys([s for s in symbols if s]))
    if not symbols:
        raise SystemExit("No usable symbols in universe selection")
    _log(f"universe_symbols={symbols}")

    summary_path = (PROJECT_ROOT / args.summary_file).resolve()
    summary_rows = _load_summary(summary_path) if bool(args.resume) else {}

    for sym in symbols:
        _log(f"{sym}: preparing data/features")
        try:
            df = _prepare_symbol_dataset(sym, args)
            df_train, df_test = _build_train_test(df, args.test_start, args.test_end)
            _log(f"{sym}: rows total={len(df)} train={len(df_train)} test={len(df_test)}")
        except Exception as ex:
            _log(f"{sym}: data preparation failed: {type(ex).__name__}: {ex}")
            for side in ("long", "short"):
                summary_rows[(sym, side)] = {
                    "symbol": sym,
                    "side": side,
                    "test_net": 0.0,
                    "test_pf": 0.0,
                    "test_dd": 1.0,
                    "test_trades": 0.0,
                    "stability": 0.0,
                    "PASS/FAIL": "FAIL",
                    "fail_reasons": f"data_prep_error:{type(ex).__name__}",
                    "param_path": "",
                    "pipeline_run_id": pipeline_run_id,
                    "ga_run_id": "",
                    "ga_symbol": "",
                    "updated_utc": _utc_now_iso(),
                }
            _write_summary(summary_path, summary_rows)
            continue

        for side in ("long", "short"):
            key = (sym, side)
            param_path = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_C13_active_params_{side}.json"
            if bool(args.resume) and not bool(args.force_optimize) and key in summary_rows and param_path.exists():
                _log(f"{sym} {side}: resume skip (summary + params already present)")
                continue

            try:
                if side == "long":
                    row = _optimize_long(sym, df_train, df_test, args, pipeline_run_id)
                else:
                    row = _optimize_short(sym, df_train, df_test, args, pipeline_run_id)
                _log(
                    f"{sym} {side}: {row['PASS/FAIL']} "
                    f"pf={row['test_pf']:.3f} dd={row['test_dd']:.3f} "
                    f"trades={row['test_trades']:.1f} stability={row['stability']:.3f}"
                )
                summary_rows[key] = row
                _write_summary(summary_path, summary_rows)
            except Exception as ex:
                _log(f"{sym} {side}: optimization failed: {type(ex).__name__}: {ex}")
                summary_rows[key] = {
                    "symbol": sym,
                    "side": side,
                    "test_net": 0.0,
                    "test_pf": 0.0,
                    "test_dd": 1.0,
                    "test_trades": 0.0,
                    "stability": 0.0,
                    "PASS/FAIL": "FAIL",
                    "fail_reasons": f"opt_error:{type(ex).__name__}",
                    "param_path": "",
                    "pipeline_run_id": pipeline_run_id,
                    "ga_run_id": "",
                    "ga_symbol": "",
                    "updated_utc": _utc_now_iso(),
                }
                _write_summary(summary_path, summary_rows)

    _write_summary(summary_path, summary_rows)
    _log(f"pipeline_done summary_csv={summary_path}")


if __name__ == "__main__":
    main()
