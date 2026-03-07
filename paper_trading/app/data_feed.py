from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import Settings
from .utils.retry import RetryConfig, retry_call
from .utils.time_utils import utc_now


@dataclass
class CircuitState:
    fail_count: int = 0
    open_until_epoch: float = 0.0

    @property
    def is_open(self) -> bool:
        return time.time() < self.open_until_epoch


class DataFeed:
    def __init__(self, settings: Settings, logger, health: dict[str, Any]) -> None:
        self.settings = settings
        self.logger = logger
        self.health = health
        self.session = requests.Session()
        self.timeout_sec = 12
        self.retry_cfg = RetryConfig(attempts=4, base_delay_sec=0.7, max_delay_sec=6.0, jitter_sec=0.35)
        self.circuit = CircuitState()
        self.local_cache: dict[str, pd.DataFrame] = {}
        # Remote market-data reads are allowed in paper mode for both testnet and production endpoints.
        # Safety is enforced by keeping execution strictly local paper (no live order placement).
        self._allow_remote_marketdata = self.settings.binance_mode.lower() in {"testnet", "marketdata_only"}

    def _mark_api_failure(self, reason: str) -> None:
        self.circuit.fail_count += 1
        self.health["api_failures"] = int(self.health.get("api_failures", 0)) + 1
        if self.circuit.fail_count >= self.settings.api_circuit_fail_threshold:
            self.circuit.open_until_epoch = time.time() + float(self.settings.api_circuit_cooldown_sec)
            self.logger.error(
                "api_circuit_open fail_count=%s cooldown_sec=%s reason=%s",
                self.circuit.fail_count,
                self.settings.api_circuit_cooldown_sec,
                reason,
            )

    def _mark_api_success(self) -> None:
        self.circuit.fail_count = 0
        self.circuit.open_until_epoch = 0.0

    def _fetch_binance_klines(self, symbol: str, limit: int) -> pd.DataFrame:
        base_url = self.settings.binance_base_url.rstrip("/")
        url = f"{base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": int(limit)}

        def call() -> pd.DataFrame:
            response = self.session.get(url, params=params, timeout=self.timeout_sec)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected kline payload type: {type(payload)}")
            rows = []
            for item in payload:
                if not isinstance(item, list) or len(item) < 6:
                    continue
                rows.append(
                    {
                        "Timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                        "Open": float(item[1]),
                        "High": float(item[2]),
                        "Low": float(item[3]),
                        "Close": float(item[4]),
                        "Volume": float(item[5]),
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                raise RuntimeError(f"empty kline payload for {symbol}")
            return df.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last").reset_index(drop=True)

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            self.health["api_retries"] = int(self.health.get("api_retries", 0)) + 1
            self.logger.warning(
                "kline_retry symbol=%s attempt=%s delay=%.2fs err=%s",
                symbol,
                attempt,
                delay,
                str(exc),
            )

        return retry_call(call, cfg=self.retry_cfg, on_retry=on_retry)

    def _load_local(self, symbol: str) -> pd.DataFrame:
        if symbol in self.local_cache:
            return self.local_cache[symbol]

        root = self.settings.project_root
        candidates = [
            root / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet",
            root / "data" / "processed" / "_full" / f"{symbol}_1h_features.parquet",
        ]

        df: pd.DataFrame | None = None
        for path in candidates:
            if path.exists():
                loaded = pd.read_parquet(path)
                df = loaded
                break

        if df is None:
            csv_paths = sorted((root / "data" / "processed").glob(f"{symbol}_*_proc.csv"))
            if not csv_paths:
                raise FileNotFoundError(f"No local OHLC source found for {symbol}")
            frames = [pd.read_csv(path) for path in csv_paths]
            df = pd.concat(frames, ignore_index=True)

        rename_map = {}
        if "timestamp" in df.columns and "Timestamp" not in df.columns:
            rename_map["timestamp"] = "Timestamp"
        for src, dst in [
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ]:
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        if rename_map:
            df = df.rename(columns=rename_map)

        required = ["Timestamp", "Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(f"local data missing {missing} for {symbol}")

        x = df.copy()
        x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
        x = x.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

        self.local_cache[symbol] = x
        return x

    def fetch_ohlcv_1h(self, symbol: str, limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        bars = int(limit or self.settings.max_bars_fetch)
        source = "local"
        degraded = False

        if (
            self._allow_remote_marketdata
            and self.settings.binance_mode.lower() in {"testnet", "marketdata_only"}
            and not self.circuit.is_open
        ):
            try:
                df = self._fetch_binance_klines(symbol, bars)
                self._mark_api_success()
                source = "binance_api"
                return df.tail(bars).reset_index(drop=True), {
                    "source": source,
                    "degraded": degraded,
                    "circuit_open": self.circuit.is_open,
                }
            except Exception as exc:
                degraded = True
                self._mark_api_failure(str(exc))
                self.logger.error("api_fetch_failed symbol=%s err=%s", symbol, str(exc))

        df = self._load_local(symbol)
        return df.tail(bars).reset_index(drop=True), {
            "source": source,
            "degraded": degraded or self.circuit.is_open,
            "circuit_open": self.circuit.is_open,
        }

    def fetch_usdt_to_eur(self) -> tuple[float, str]:
        if (
            self._allow_remote_marketdata
            and self.settings.binance_mode.lower() in {"testnet", "marketdata_only"}
            and not self.circuit.is_open
        ):
            base_url = self.settings.binance_base_url.rstrip("/")
            url = f"{base_url}/api/v3/ticker/price"
            params = {"symbol": "EURUSDT"}
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
                resp.raise_for_status()
                obj = resp.json()
                eurusdt = float(obj["price"])
                if eurusdt > 0:
                    return 1.0 / eurusdt, "binance_api"
            except Exception as exc:
                self._mark_api_failure(f"fx:{exc}")
                self.logger.warning("fx_fetch_failed err=%s", str(exc))

        # Degraded fallback: assume USDT≈EUR.
        return 1.0, "fallback_1_to_1"

    def quote_to_eur(self, quote_asset: str) -> tuple[float, str]:
        quote = quote_asset.upper()
        if quote == "EUR":
            return 1.0, "native"
        if quote in {"USDT", "USDC", "BUSD"}:
            return self.fetch_usdt_to_eur()
        return 1.0, "fallback_unknown_quote"

    @staticmethod
    def latest_closed_bar_ts(df: pd.DataFrame) -> pd.Timestamp:
        if df.empty:
            raise ValueError("empty dataframe")
        if len(df) == 1:
            return pd.to_datetime(df.iloc[0]["Timestamp"], utc=True)

        now = utc_now()
        last_ts = pd.to_datetime(df.iloc[-1]["Timestamp"], utc=True)
        # If the last candle is still within its 1h window, treat previous as last closed.
        if last_ts <= now < (last_ts + pd.Timedelta(hours=1)):
            return pd.to_datetime(df.iloc[-2]["Timestamp"], utc=True)
        return last_ts
