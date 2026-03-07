from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bot087.optim.ga import _ensure_indicators, _norm_params, _shift_cycles, build_entry_signal, compute_cycles


@dataclass
class SignalFrame:
    frame: pd.DataFrame
    params: dict[str, Any]


class SignalRunner:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.params_cache: dict[str, dict[str, Any]] = {}

    def load_symbol_params(self, symbol: str, params_path: str) -> dict[str, Any]:
        symbol_u = symbol.upper()
        if symbol_u in self.params_cache:
            return self.params_cache[symbol_u]

        path = Path(params_path)
        if not path.exists():
            raise FileNotFoundError(f"params not found for {symbol_u}: {path}")

        raw = pd.read_json(path, typ="series").to_dict()
        if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
            raw = raw["params"]

        params = _norm_params(raw)
        # Force strict no-lookahead timing contract.
        params["cycle_shift"] = int(params.get("cycle_shift", 1) or 1)
        params["cycle_fill"] = int(params.get("cycle_fill", 2) or 2)
        params["two_candle_confirm"] = bool(params.get("two_candle_confirm", False))

        self.params_cache[symbol_u] = params
        return params

    @staticmethod
    def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        rename = {}
        if "timestamp" in x.columns and "Timestamp" not in x.columns:
            rename["timestamp"] = "Timestamp"
        for src, dst in [
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ]:
            if src in x.columns and dst not in x.columns:
                rename[src] = dst
        if rename:
            x = x.rename(columns=rename)

        required = ["Timestamp", "Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in x.columns]
        if missing:
            raise ValueError(f"ohlc frame missing columns: {missing}")

        x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
        x = x.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")

        x = x.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
        return x

    def build_signal_frame(self, symbol: str, raw_df: pd.DataFrame, params: dict[str, Any]) -> SignalFrame:
        x = self._standardize_ohlc(raw_df)
        if x.empty:
            raise RuntimeError(f"empty signal frame for {symbol}")

        x = _ensure_indicators(x, params)

        signals = np.asarray(build_entry_signal(x, params, assume_prepared=True), dtype=bool)
        cycles_raw = compute_cycles(x, params)
        cycles = _shift_cycles(cycles_raw, int(params.get("cycle_shift", 1)), int(params.get("cycle_fill", 2)))

        x = x.copy()
        x["SIGNAL"] = signals.astype(bool)
        x["CYCLE"] = cycles.astype(int)
        x["ATR_PREV"] = x["ATR"].astype(float).shift(1).fillna(0.0)
        x["RSI_PREV"] = x["RSI"].astype(float).shift(1).fillna(50.0)
        x["BAR_INDEX"] = np.arange(len(x), dtype=int)

        return SignalFrame(frame=x, params=params)

    @staticmethod
    def rows_after(
        signal_frame: SignalFrame,
        last_processed_ts: pd.Timestamp | None,
        max_bar_ts: pd.Timestamp,
        start_from_bar_ts: pd.Timestamp | None = None,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        x = signal_frame.frame
        max_ts = pd.to_datetime(max_bar_ts, utc=True)
        floor_ts: pd.Timestamp | None = None
        thresholds = [ts for ts in [last_processed_ts, start_from_bar_ts] if ts is not None]
        if thresholds:
            floor_ts = max(pd.to_datetime(ts, utc=True) for ts in thresholds)

        if floor_ts is None:
            out = x[x["Timestamp"] <= max_ts]
        else:
            out = x[(x["Timestamp"] > floor_ts) & (x["Timestamp"] <= max_ts)]

        out = out.reset_index(drop=True)
        if latest_only and not out.empty:
            out = out.tail(1).reset_index(drop=True)
        return out
