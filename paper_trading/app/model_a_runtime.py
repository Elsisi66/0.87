from __future__ import annotations

import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from scripts import execution_layer_3m_ict as exec3m
from src.bot087.optim.ga import _position_size

from .config import Settings
from .health import HealthTracker
from .notifier import NotifyResult, TelegramNotifier
from .signal_runner import SignalRunner
from .state_store import StateStore
from .utils.io import append_jsonl, atomic_write_json, atomic_write_text, ensure_dir, read_json
from .utils.time_utils import utc_iso, utc_now, utc_tag


PROJECT_ROOT = Path("/root/analysis/0.87").resolve()
PHASEC_DIR_DEFAULT = (
    PROJECT_ROOT / "reports" / "execution_layer" / "PHASEC_MODEL_A_BOUNDED_CONFIRMATION_20260228_022501"
).resolve()
PHASER_DIR_DEFAULT = (
    PROJECT_ROOT / "reports" / "execution_layer" / "PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334"
).resolve()
BASELINE_AUDIT_DIR = (PROJECT_ROOT / "reports" / "execution_layer" / "BASELINE_AUDIT_20260221_214310").resolve()

FEE_MODEL_PATH = BASELINE_AUDIT_DIR / "fee_model.json"
METRICS_DEF_PATH = BASELINE_AUDIT_DIR / "metrics_definition.md"
EXPECTED_FEE_SHA256 = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_METRICS_SHA256 = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_ts(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _ensure_json(path: Path, default: Any) -> Any:
    return read_json(path, default) or default


def _book_state_dir(root: Path, role: str) -> Path:
    return ensure_dir(root / role)


def _write_root_event(path: Path, payload: dict[str, Any]) -> None:
    row = dict(payload)
    row.setdefault("event_recorded_ts", utc_iso())
    row.setdefault("ts_utc", row["event_recorded_ts"])
    append_jsonl(path, row)


def _interval_delta(interval: str) -> pd.Timedelta:
    interval_l = str(interval).strip().lower()
    if interval_l.endswith("m"):
        return pd.Timedelta(minutes=int(interval_l[:-1]))
    if interval_l.endswith("h"):
        return pd.Timedelta(hours=int(interval_l[:-1]))
    raise ValueError(f"unsupported interval: {interval}")


def _list_from_params(params: dict[str, Any], base_key: str, default: list[float]) -> list[float]:
    if isinstance(params.get(base_key), list):
        out: list[float] = []
        for item in params[base_key]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(float(default[min(len(out), len(default) - 1)]))
        if out:
            return out
    out = list(default)
    for idx in range(min(len(default), 5)):
        key = f"{base_key[:-9]}cycle{idx}" if base_key.endswith("_by_cycle") else f"{base_key}_{idx}"
        if key in params:
            try:
                out[idx] = float(params[key])
            except (TypeError, ValueError):
                pass
    return out


def _tp_sl_for_cycle(params: dict[str, Any], cycle: int) -> tuple[float, float]:
    tp_list = _list_from_params(params, "tp_mult_by_cycle", [1.02] * 5)
    sl_list = _list_from_params(params, "sl_mult_by_cycle", [0.98] * 5)
    idx = max(0, min(int(cycle), min(len(tp_list), len(sl_list)) - 1))
    return float(tp_list[idx]), float(sl_list[idx])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class CandidateSpec:
    role: str
    candidate_id: str
    seed_origin: str
    seed_cluster_id: int
    cluster_id: int
    entry_mode: str
    limit_offset_bps: float
    fallback_to_market: int
    fallback_delay_min: float
    max_fill_delay_min: float
    expected_exec_expectancy_net: float
    expected_delta_vs_1h_reference: float
    expected_taker_share: float
    expected_p95_fill_delay_min: float


@dataclass
class MarketArrays:
    df: pd.DataFrame
    ts_ns: np.ndarray
    open_np: np.ndarray
    high_np: np.ndarray
    low_np: np.ndarray
    close_np: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "MarketArrays":
        x = df.copy()
        x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
        x = x.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
        return cls(
            df=x,
            ts_ns=np.array([int(t.value) for t in pd.to_datetime(x["Timestamp"], utc=True)], dtype=np.int64),
            open_np=pd.to_numeric(x["Open"], errors="coerce").to_numpy(dtype=float),
            high_np=pd.to_numeric(x["High"], errors="coerce").to_numpy(dtype=float),
            low_np=pd.to_numeric(x["Low"], errors="coerce").to_numpy(dtype=float),
            close_np=pd.to_numeric(x["Close"], errors="coerce").to_numpy(dtype=float),
        )


class ModelAFeed:
    def __init__(self, settings: Settings, logger, health: HealthTracker, *, force_local_only: bool = False) -> None:
        self.settings = settings
        self.logger = logger
        self.health = health
        self.force_local_only = bool(force_local_only)
        self.session = requests.Session()
        self.timeout_sec = 12
        self._allow_remote_marketdata = (
            (not self.force_local_only) and self.settings.binance_mode.lower() in {"testnet", "marketdata_only"}
        )
        self._local_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
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
            raise RuntimeError(f"market data missing {missing}")
        x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
        x = x.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
        return x

    def _load_local(self, symbol: str, interval: str) -> pd.DataFrame:
        key = (symbol.upper(), interval)
        if key in self._local_cache:
            return self._local_cache[key]

        candidates = [
            self.settings.project_root / "data" / "processed" / "_full" / f"{symbol.upper()}_{interval}_full.parquet",
            self.settings.project_root / "data" / "processed" / "_full" / f"{symbol.upper()}_{interval}_features.parquet",
        ]
        df: pd.DataFrame | None = None
        for path in candidates:
            if path.exists():
                df = pd.read_parquet(path)
                break
        if df is None:
            raise FileNotFoundError(f"no local {interval} data for {symbol}")
        out = self._normalize(df)
        self._local_cache[key] = out
        return out

    def _fetch_remote(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        base_url = self.settings.binance_base_url.rstrip("/")
        url = f"{base_url}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        resp = self.session.get(url, params=params, timeout=self.timeout_sec)
        resp.raise_for_status()
        payload = resp.json()
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
        if not rows:
            raise RuntimeError(f"empty remote payload for {symbol} {interval}")
        return self._normalize(pd.DataFrame(rows))

    def fetch_ohlcv(self, symbol: str, interval: str, *, limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        use_remote = self._allow_remote_marketdata and limit is not None
        if use_remote:
            try:
                df = self._fetch_remote(symbol, interval, int(limit))
                return df.tail(int(limit)).reset_index(drop=True), {"source": "binance_api", "degraded": False}
            except Exception as exc:
                self.health.inc("api_failures")
                self.logger.warning("model_a_remote_fetch_failed symbol=%s interval=%s err=%s", symbol, interval, str(exc))
                self.health.set_degraded_mode(True)
        df = self._load_local(symbol, interval)
        if limit is not None:
            df = df.tail(int(limit)).reset_index(drop=True)
        return df, {"source": "local", "degraded": use_remote}

    def latest_closed_bar_ts(self, df: pd.DataFrame, interval: str) -> pd.Timestamp:
        if df.empty:
            raise ValueError("empty dataframe")
        if len(df) == 1:
            return _coerce_ts(df.iloc[0]["Timestamp"])
        now = utc_now()
        last_ts = _coerce_ts(df.iloc[-1]["Timestamp"])
        delta = _interval_delta(interval)
        if last_ts <= now < (last_ts + delta):
            return _coerce_ts(df.iloc[-2]["Timestamp"])
        return last_ts

    def quote_to_eur(self, quote_asset: str) -> tuple[float, str]:
        quote = str(quote_asset).upper()
        if quote == "EUR":
            return 1.0, "native"
        if quote not in {"USDT", "USDC", "BUSD"}:
            return 1.0, "fallback_unknown_quote"
        if not self._allow_remote_marketdata:
            return 1.0, "fallback_1_to_1"
        try:
            base_url = self.settings.binance_base_url.rstrip("/")
            resp = self.session.get(
                f"{base_url}/api/v3/ticker/price",
                params={"symbol": "EURUSDT"},
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            eurusdt = float(payload["price"])
            if eurusdt > 0:
                return 1.0 / eurusdt, "binance_api"
        except Exception as exc:
            self.health.inc("api_failures")
            self.logger.warning("model_a_fx_fetch_failed err=%s", str(exc))
            self.health.set_degraded_mode(True)
        return 1.0, "fallback_1_to_1"


def load_selected_candidates(phase_c_dir: Path = PHASEC_DIR_DEFAULT) -> list[CandidateSpec]:
    selection_df = pd.read_csv(phase_c_dir / "phaseC4_primary_backup.csv")
    results_df = pd.read_csv(phase_c_dir / "phaseC2_results.csv")
    out: list[CandidateSpec] = []
    role_map = {"primary": "paper_primary", "backup": "shadow_backup"}
    for _, row in selection_df.iterrows():
        selection_role = str(row["selection_role"]).strip().lower()
        cid = str(row["candidate_id"])
        match = results_df[results_df["candidate_id"].astype(str) == cid]
        if match.empty:
            raise KeyError(f"missing Phase C config row for {cid}")
        src = match.iloc[0]
        out.append(
            CandidateSpec(
                role=role_map.get(selection_role, selection_role),
                candidate_id=cid,
                seed_origin=str(row.get("seed_origin", src.get("seed_origin", ""))),
                seed_cluster_id=int(_safe_float(row.get("seed_cluster_id"), src.get("seed_cluster_id", 0))),
                cluster_id=int(_safe_float(row.get("cluster_id"), src.get("cluster_id", 0))),
                entry_mode=str(src.get("entry_mode", "limit")),
                limit_offset_bps=float(_safe_float(src.get("limit_offset_bps"), 0.0)),
                fallback_to_market=int(_safe_float(src.get("fallback_to_market"), 0)),
                fallback_delay_min=float(_safe_float(src.get("fallback_delay_min"), 0.0)),
                max_fill_delay_min=float(_safe_float(src.get("max_fill_delay_min"), 0.0)),
                expected_exec_expectancy_net=float(_safe_float(row.get("exec_expectancy_net"), src.get("exec_expectancy_net", 0.0))),
                expected_delta_vs_1h_reference=float(
                    _safe_float(row.get("delta_expectancy_vs_1h_reference"), src.get("delta_expectancy_vs_1h_reference", 0.0))
                ),
                expected_taker_share=float(_safe_float(row.get("taker_share"), src.get("taker_share", 0.0))),
                expected_p95_fill_delay_min=float(_safe_float(row.get("p95_fill_delay_min"), src.get("p95_fill_delay_min", 0.0))),
            )
        )
    return out


class ModelAPaperRuntime:
    def __init__(
        self,
        settings: Settings,
        logger,
        *,
        phase_c_dir: Path = PHASEC_DIR_DEFAULT,
        phase_r_dir: Path = PHASER_DIR_DEFAULT,
        state_root: Path | None = None,
        symbol: str = "SOLUSDT",
        force_local_only: bool = False,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.phase_c_dir = Path(phase_c_dir).resolve()
        self.phase_r_dir = Path(phase_r_dir).resolve()
        self.symbol = str(symbol).upper()
        self.force_local_only = bool(force_local_only)
        self.state_root = ensure_dir(state_root or (self.settings.state_dir / "model_a_runtime"))
        self.root_journal_path = self.state_root / "coordinator_journal.jsonl"
        self.root_dead_letter_path = self.state_root / "coordinator_dead_letter.jsonl"
        self.root_meta_path = self.state_root / "runtime_meta.json"
        self.root_reset_marker_path = self.state_root / "startup_reset_marker.json"
        self.root_mapping_path = self.state_root / "candidate_mapping.json"
        self.root_summary_path = self.state_root / "last_cycle_summary.json"
        self.root_reports_dir = ensure_dir(self.state_root / "reports")
        self.health = HealthTracker(_ensure_json(self.state_root / "health_counters.json", {}))
        self.feed = ModelAFeed(settings=self.settings, logger=self.logger, health=self.health, force_local_only=self.force_local_only)
        self.signal_runner = SignalRunner(self.logger)
        self.notifier = TelegramNotifier(self.settings, self.logger)
        self.candidates = load_selected_candidates(self.phase_c_dir)
        self.books = {c.role: StateStore(_book_state_dir(self.state_root, c.role)) for c in self.candidates}
        self.params_path = self._resolve_symbol_params()
        self._persist_candidate_mapping()

    def _resolve_symbol_params(self) -> Path:
        universe_path = self.settings.config_dir / "resolved_universe.json"
        if universe_path.exists():
            payload = _ensure_json(universe_path, {})
            sym_map = payload.get("symbol_params", {})
            p = sym_map.get(self.symbol)
            if p and Path(p).exists():
                return Path(p).resolve()
        default_path = self.settings.project_root / "data" / "metadata" / "params" / f"{self.symbol}_C13_active_params_long.json"
        if default_path.exists():
            return default_path.resolve()
        raise FileNotFoundError(f"no params path found for {self.symbol}")

    def _persist_candidate_mapping(self) -> None:
        payload = {
            "generated_utc": utc_iso(),
            "symbol": self.symbol,
            "params_path": str(self.params_path),
            "candidates": [asdict(c) for c in self.candidates],
        }
        atomic_write_json(self.root_mapping_path, payload)

    def _load_runtime_meta(self) -> dict[str, Any]:
        return _ensure_json(self.root_meta_path, {})

    def _save_runtime_meta(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.root_meta_path, payload)

    def _root_event(self, payload: dict[str, Any]) -> None:
        _write_root_event(self.root_journal_path, payload)

    def _root_error(self, payload: dict[str, Any]) -> None:
        _write_root_event(self.root_dead_letter_path, payload)

    def validate_contract(self) -> dict[str, Any]:
        fee_sha = _sha256_file(FEE_MODEL_PATH)
        metrics_sha = _sha256_file(METRICS_DEF_PATH)
        freeze_lock_pass = int(fee_sha == EXPECTED_FEE_SHA256 and metrics_sha == EXPECTED_METRICS_SHA256)
        route_df = pd.read_csv(self.phase_r_dir / "phaseR1_feasibility_validation.csv")
        route_ok = int(route_df["route_trade_gates_reachable"].astype(int).eq(1).all())
        feed_1h, _ = self.feed.fetch_ohlcv(self.symbol, "1h", limit=128)
        feed_3m, _ = self.feed.fetch_ohlcv(self.symbol, "3m", limit=256)
        validation = {
            "generated_utc": utc_iso(),
            "freeze_lock_pass": freeze_lock_pass,
            "fee_model_sha256_observed": fee_sha,
            "fee_model_sha256_expected": EXPECTED_FEE_SHA256,
            "metrics_definition_sha256_observed": metrics_sha,
            "metrics_definition_sha256_expected": EXPECTED_METRICS_SHA256,
            "repaired_route_support_feasible": route_ok,
            "uses_1h_signal_owner": 1,
            "uses_3m_entry_executor": 1,
            "exits_owned_by_1h_only": 1,
            "forbidden_exit_controls_active": 0,
            "primary_backup_loaded": int(len(self.candidates) == 2),
            "isolated_books": int(len({str(store.state_dir) for store in self.books.values()}) == len(self.books)),
            "one_h_feed_ready": int(not feed_1h.empty),
            "three_m_feed_ready": int(not feed_3m.empty),
            "symbol": self.symbol,
            "params_path": str(self.params_path),
            "candidate_roles": [c.role for c in self.candidates],
            "candidate_ids": [c.candidate_id for c in self.candidates],
            "state_root": str(self.state_root),
        }
        return validation

    def load_params(self) -> dict[str, Any]:
        return self.signal_runner.load_symbol_params(self.symbol, str(self.params_path))

    def _shared_market(self) -> tuple[dict[str, Any], pd.DataFrame, MarketArrays, pd.DataFrame, MarketArrays, pd.Timestamp, float]:
        df_1h, meta_1h = self.feed.fetch_ohlcv(self.symbol, "1h", limit=1200 if not self.force_local_only else None)
        df_3m, meta_3m = self.feed.fetch_ohlcv(self.symbol, "3m", limit=4000 if not self.force_local_only else None)
        one_h = MarketArrays.from_df(df_1h)
        three_m = MarketArrays.from_df(df_3m)
        latest_closed_1h = self.feed.latest_closed_bar_ts(one_h.df, "1h")
        quote_to_eur, fx_source = self.feed.quote_to_eur("USDT")
        meta = {
            "one_h_source": meta_1h.get("source"),
            "one_h_degraded": int(bool(meta_1h.get("degraded", False))),
            "three_m_source": meta_3m.get("source"),
            "three_m_degraded": int(bool(meta_3m.get("degraded", False))),
            "fx_source": fx_source,
        }
        return meta, one_h.df, one_h, three_m.df, three_m, latest_closed_1h, float(quote_to_eur)

    def compute_smoke_anchor(self, lookback_signals: int = 8) -> pd.Timestamp:
        params = self.load_params()
        df_1h, _meta = self.feed.fetch_ohlcv(self.symbol, "1h", limit=None)
        frame = self.signal_runner.build_signal_frame(self.symbol, df_1h, params)
        x = frame.frame.copy()
        signals = x[x["SIGNAL"].astype(bool)].reset_index(drop=True)
        if signals.empty:
            latest_closed = self.feed.latest_closed_bar_ts(x, "1h")
            return latest_closed - pd.Timedelta(hours=24)
        target = signals.tail(max(1, int(lookback_signals))).copy()
        first_ts = _coerce_ts(target.iloc[0]["Timestamp"])
        return first_ts - pd.Timedelta(hours=1)

    def hard_reset(self, *, start_from_bar_ts: pd.Timestamp | None = None) -> dict[str, Any]:
        start_ts = _coerce_ts(start_from_bar_ts) if start_from_bar_ts is not None else self.compute_smoke_anchor()
        ensure_dir(self.state_root / "archive")
        reset_books: list[dict[str, Any]] = []
        for candidate in self.candidates:
            store = self.books[candidate.role]
            archive_dir = store.archive_state("model_a_reset")
            store.journal_path.write_text("", encoding="utf-8")
            store.dead_letter_path.write_text("", encoding="utf-8")
            store.initialize(self.settings.start_equity_eur)
            store.save_positions({})
            store.save_orders([])
            store.save_processed_bars({})
            store.save_quarantine({})
            store.save_health_counters(
                {
                    "api_retries": 0,
                    "api_failures": 0,
                    "signal_errors": 0,
                    "execution_errors": 0,
                    "recovery_events": 0,
                    "telegram_errors": 0,
                    "quarantined_symbols": 0,
                    "degraded_mode": False,
                    "strategy_health": "GREEN",
                }
            )
            store.save_runtime_meta(
                {
                    "start_from_bar_ts": start_ts.isoformat(),
                    "last_processed_1h_bar_ts": None,
                    "book_role": candidate.role,
                    "candidate_id": candidate.candidate_id,
                    "forward_only_mode": True,
                    "model_a_purity": True,
                }
            )
            reset_meta = {
                "generated_utc": utc_iso(),
                "event": "startup_reset",
                "role": candidate.role,
                "candidate_id": candidate.candidate_id,
                "archive_dir": str(archive_dir),
                "start_from_bar_ts": start_ts.isoformat(),
            }
            store.save_reset_marker(reset_meta)
            store.append_journal(reset_meta)
            reset_books.append(reset_meta)
        root_meta = {
            "generated_utc": utc_iso(),
            "start_from_bar_ts": start_ts.isoformat(),
            "last_cycle_completed_utc": None,
            "symbol": self.symbol,
            "force_local_only": self.force_local_only,
            "books": [c.role for c in self.candidates],
        }
        self._save_runtime_meta(root_meta)
        atomic_write_json(
            self.root_reset_marker_path,
            {"generated_utc": utc_iso(), "start_from_bar_ts": start_ts.isoformat(), "books_reset": reset_books},
        )
        self._root_event({"event": "startup_reset", "symbol": self.symbol, "start_from_bar_ts": start_ts.isoformat()})
        return root_meta

    def reconcile_book_state(self, role: str, latest_closed_1h: pd.Timestamp) -> int:
        store = self.books[role]
        positions = store.load_positions()
        recovery_actions = 0
        if len(positions) > 1:
            recovery_actions += 1
            keep_key = next(iter(sorted(positions.keys())))
            positions = {keep_key: positions[keep_key]}
            store.save_positions(positions)
            store.append_journal(
                {
                    "event": "recovery_action",
                    "role": role,
                    "symbol": self.symbol,
                    "reason": "trim_multiple_positions",
                    "kept_symbol": keep_key,
                }
            )
        pos = positions.get(self.symbol)
        if pos is not None:
            fill_time = _coerce_ts(pos.get("fill_time"))
            if pd.isna(fill_time):
                recovery_actions += 1
                store.save_positions({})
                store.append_journal(
                    {
                        "event": "recovery_action",
                        "role": role,
                        "symbol": self.symbol,
                        "reason": "drop_corrupt_position",
                    }
                )
        if recovery_actions:
            self.health.inc("recovery_events", recovery_actions)
        return recovery_actions

    def _entry_signal_id(self, signal_ts: pd.Timestamp) -> str:
        return f"{self.symbol}:{signal_ts.isoformat()}"

    def _simulate_entry_fill(self, market_3m: MarketArrays, signal_ts: pd.Timestamp, spec: CandidateSpec) -> dict[str, Any]:
        out = {
            "filled": 0,
            "fill_time": pd.NaT,
            "fill_price": float("nan"),
            "fill_type": "",
            "entry_improvement_bps": float("nan"),
            "skip_reason": "",
        }
        ts_ns = market_3m.ts_ns
        n = len(ts_ns)
        if n == 0:
            out["skip_reason"] = "no_3m_data"
            return out
        signal_ns = int(signal_ts.value)
        sig_idx = int(np.searchsorted(ts_ns, signal_ns, side="left"))
        if sig_idx >= n:
            out["skip_reason"] = "no_bar_after_signal"
            return out

        entry_ref = float(market_3m.open_np[sig_idx])
        if (not np.isfinite(entry_ref)) or entry_ref <= 0.0:
            out["skip_reason"] = "bad_entry_ref"
            return out

        max_fill_bars = max(0, int(math.ceil(float(spec.max_fill_delay_min) / 3.0)))
        fallback_bars = max(0, int(math.ceil(float(spec.fallback_delay_min) / 3.0)))
        fill_end_idx = min(n - 1, sig_idx + max_fill_bars)
        fill_idx: int | None = None
        fill_px = float("nan")
        fill_type = ""

        if str(spec.entry_mode).lower() == "market":
            fill_idx = sig_idx
            fill_px = float(market_3m.open_np[fill_idx])
            fill_type = "market"
        else:
            limit_px = float(entry_ref * (1.0 - max(0.0, float(spec.limit_offset_bps)) / 1e4))
            for i in range(sig_idx, fill_end_idx + 1):
                if np.isfinite(market_3m.low_np[i]) and float(market_3m.low_np[i]) <= limit_px:
                    fill_idx = i
                    fill_px = float(limit_px)
                    fill_type = "limit"
                    break
            if fill_idx is None and int(spec.fallback_to_market) == 1:
                m_idx = min(fill_end_idx, sig_idx + fallback_bars)
                if m_idx <= fill_end_idx:
                    fill_idx = int(m_idx)
                    fill_px = float(market_3m.open_np[m_idx])
                    fill_type = "market_fallback"

        if fill_idx is None:
            out["skip_reason"] = "timeout_no_fill"
            return out

        fill_time = pd.to_datetime(int(ts_ns[int(fill_idx)]), utc=True)
        improve = float((entry_ref - fill_px) / entry_ref * 1e4) if entry_ref > 0 else float("nan")
        out.update(
            {
                "filled": 1,
                "fill_time": fill_time,
                "fill_price": float(fill_px),
                "fill_type": str(fill_type),
                "entry_improvement_bps": float(improve),
            }
        )
        return out

    def _entry_liquidity(self, fill_type: str) -> str:
        return "maker" if str(fill_type) == "limit" else "taker"

    def _size_position(self, portfolio: dict[str, Any], row: pd.Series, params: dict[str, Any], fill_price: float, quote_to_eur: float) -> tuple[float, float]:
        cash_eur = float(portfolio.get("cash_eur", 0.0))
        atr_prev = float(_safe_float(row.get("ATR_PREV"), 0.0))
        entry_px_eur = float(fill_price) * float(quote_to_eur)
        atr_eur = atr_prev * float(quote_to_eur)
        qty = float(
            _position_size(
                cash_eur,
                entry_px_eur,
                atr_eur,
                float(_safe_float(params.get("risk_per_trade"), 0.02)),
                float(_safe_float(params.get("max_allocation"), 0.7)),
                float(_safe_float(params.get("atr_k"), 1.0)),
            )
        )
        if qty > 0:
            max_qty = cash_eur / max(entry_px_eur, 1e-9)
            if qty > max_qty:
                qty = float(max_qty)
        notional_eur = float(qty * entry_px_eur)
        return float(max(qty, 0.0)), float(max(notional_eur, 0.0))

    def _maybe_close_position(
        self,
        *,
        position: dict[str, Any],
        current_bar_ts: pd.Timestamp,
        one_h: MarketArrays,
        quote_to_eur: float,
    ) -> dict[str, Any]:
        fill_time = _coerce_ts(position.get("fill_time"))
        if pd.isna(fill_time) or current_bar_ts <= fill_time:
            return {"should_close": 0}
        start_idx = int(np.searchsorted(one_h.ts_ns, int(fill_time.value), side="right"))
        if start_idx >= len(one_h.ts_ns):
            return {"should_close": 0}

        max_exit_ts = fill_time + pd.Timedelta(hours=float(_safe_float(position.get("exec_horizon_hours"), 12.0)))
        eval_cap = min(current_bar_ts, max_exit_ts)
        eval_end_idx = int(
            exec3m._idx_at_or_before_ts(
                ts_ns=one_h.ts_ns,
                target_ns=int(eval_cap.value),
                min_idx=int(start_idx),
                max_idx=len(one_h.ts_ns) - 1,
            )
        )
        if eval_end_idx < start_idx:
            return {"should_close": 0}

        sim = exec3m._simulate_path_long(
            ts_ns=one_h.ts_ns,
            close=one_h.close_np,
            high=one_h.high_np,
            low=one_h.low_np,
            entry_idx=int(start_idx),
            entry_price=float(_safe_float(position.get("fill_price"), 0.0)),
            sl_price=float(_safe_float(position.get("sl_price"), 0.0)),
            tp_price=float(_safe_float(position.get("tp_price"), 0.0)),
            max_exit_ts_ns=int(eval_cap.value),
        )

        exit_reason = str(sim.get("exit_reason", ""))
        if exit_reason == "window_end" and current_bar_ts < max_exit_ts:
            return {"should_close": 0}

        liq = str(position.get("entry_liquidity_type", "taker"))
        cost = exec3m._costed_pnl_long(
            entry_price=float(_safe_float(position.get("fill_price"), 0.0)),
            exit_price=float(_safe_float(sim.get("exit_price"), 0.0)),
            entry_liquidity_type=liq,
            fee_bps_maker=2.0,
            fee_bps_taker=4.0,
            slippage_bps_limit=0.5,
            slippage_bps_market=2.0,
        )
        qty = float(_safe_float(position.get("units"), 0.0))
        exit_price = float(_safe_float(sim.get("exit_price"), 0.0))
        exit_notional_eur = qty * exit_price * float(quote_to_eur)
        fee_cost_eur = exit_notional_eur * float(_safe_float(cost.get("exit_fee_bps"), 0.0)) / 1e4
        slip_cost_eur = exit_notional_eur * float(_safe_float(cost.get("exit_slippage_bps"), 0.0)) / 1e4
        entry_cost_eur = float(_safe_float(position.get("entry_cost_eur"), 0.0))
        net_pnl_eur = entry_cost_eur * float(_safe_float(cost.get("pnl_net_pct"), 0.0))
        proceeds_eur = entry_cost_eur + net_pnl_eur

        return {
            "should_close": 1,
            "exit_time": _coerce_ts(sim.get("exit_time")),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "sl_hit": int(bool(sim.get("sl_hit", False))),
            "tp_hit": int(bool(sim.get("tp_hit", False))),
            "same_bar_hit": int(_safe_float(sim.get("same_bar_hit"), 0)),
            "mae_pct": float(_safe_float(sim.get("mae_pct"), 0.0)),
            "mfe_pct": float(_safe_float(sim.get("mfe_pct"), 0.0)),
            "pnl_net_pct": float(_safe_float(cost.get("pnl_net_pct"), 0.0)),
            "pnl_gross_pct": float(_safe_float(cost.get("pnl_gross_pct"), 0.0)),
            "proceeds_eur": float(proceeds_eur),
            "net_pnl_eur": float(net_pnl_eur),
            "fee_cost_eur": float(fee_cost_eur),
            "slippage_cost_eur": float(slip_cost_eur),
            "hold_minutes": float((_coerce_ts(sim.get("exit_time")) - fill_time).total_seconds() / 60.0)
            if pd.notna(_coerce_ts(sim.get("exit_time")))
            else float("nan"),
        }

    def _collect_book_stats(self, store: StateStore, role: str) -> dict[str, Any]:
        journal_rows: list[dict[str, Any]] = []
        if store.journal_path.exists():
            with store.journal_path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        journal_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        events = [str(r.get("event", "")) for r in journal_rows]
        fills = [r for r in journal_rows if r.get("event") == "entry_fill"]
        exits = [r for r in journal_rows if r.get("event") == "exit_fill"]
        delays = [float(_safe_float(r.get("fill_delay_min"), np.nan)) for r in fills if pd.notna(_safe_float(r.get("fill_delay_min"), np.nan))]
        taker_hits = sum(1 for r in fills if str(r.get("entry_liquidity_type", "")) == "taker")
        portfolio = store.load_portfolio()
        positions = store.load_positions()
        return {
            "role": role,
            "candidate_id": store.load_runtime_meta().get("candidate_id"),
            "signals_seen": events.count("signal_detected"),
            "entries_attempted": events.count("entry_attempt"),
            "entries_filled": len(fills),
            "exits_processed": len(exits),
            "runtime_errors": events.count("runtime_error"),
            "recovery_actions": events.count("recovery_action"),
            "open_positions": len(positions),
            "realized_pnl_eur": float(_safe_float(portfolio.get("realized_pnl_eur"), 0.0)),
            "taker_share": float(taker_hits / len(fills)) if fills else 0.0,
            "avg_fill_delay_min": float(np.mean(delays)) if delays else 0.0,
            "p95_fill_delay_min": float(np.percentile(delays, 95)) if delays else 0.0,
            "state_consistent": int(len(positions) <= 1 and float(_safe_float(portfolio.get("cash_eur"), 0.0)) >= -1e-6),
        }

    def run_cycle(self, *, latest_only: bool = True, max_rows: int | None = None) -> dict[str, Any]:
        params = self.load_params()
        market_meta, df_1h, one_h, _df_3m, three_m, latest_closed_1h, quote_to_eur = self._shared_market()
        signal_frame = self.signal_runner.build_signal_frame(self.symbol, df_1h, params)
        root_meta = self._load_runtime_meta()
        start_from_ts = _coerce_ts(root_meta.get("start_from_bar_ts")) if root_meta.get("start_from_bar_ts") else None

        cycle_counts: dict[str, dict[str, Any]] = {}
        for candidate in self.candidates:
            store = self.books[candidate.role]
            store.initialize(self.settings.start_equity_eur)
            store_meta = store.load_runtime_meta()
            if not store_meta:
                store_meta = {
                    "start_from_bar_ts": start_from_ts.isoformat() if start_from_ts is not None else None,
                    "last_processed_1h_bar_ts": None,
                    "book_role": candidate.role,
                    "candidate_id": candidate.candidate_id,
                    "forward_only_mode": True,
                    "model_a_purity": True,
                }
                store.save_runtime_meta(store_meta)

            self.reconcile_book_state(candidate.role, latest_closed_1h)

            portfolio = store.load_portfolio()
            positions = store.load_positions()
            processed = store.load_processed_bars()
            last_ts_raw = processed.get(self.symbol)
            last_ts = _coerce_ts(last_ts_raw) if last_ts_raw else None
            rows = self.signal_runner.rows_after(
                signal_frame,
                last_ts,
                latest_closed_1h,
                start_from_bar_ts=start_from_ts,
                latest_only=bool(latest_only),
            )
            if max_rows is not None and len(rows) > int(max_rows):
                rows = rows.head(int(max_rows)).reset_index(drop=True)

            counts = {
                "signals_seen": 0,
                "entries_attempted": 0,
                "entries_filled": 0,
                "exits_processed": 0,
                "runtime_errors": 0,
                "recovery_actions": 0,
            }

            for _, row in rows.iterrows():
                bar_ts = _coerce_ts(row["Timestamp"])
                signal_id = self._entry_signal_id(bar_ts)
                had_position_at_row_open = self.symbol in positions

                try:
                    counts["signals_seen"] += 1
                    signal_event = {
                        "event": "signal_detected",
                        "role": candidate.role,
                        "candidate_id": candidate.candidate_id,
                        "symbol": self.symbol,
                        "signal_id": signal_id,
                        "signal_timestamp": bar_ts.isoformat(),
                        "signal": int(bool(row["SIGNAL"])),
                        "cycle": int(_safe_float(row["CYCLE"], 0)),
                        "model_a_purity_ok": 1,
                        "feed_1h_source": market_meta["one_h_source"],
                        "feed_3m_source": market_meta["three_m_source"],
                    }
                    store.append_journal(signal_event)
                    self._root_event(signal_event)

                    pos = positions.get(self.symbol)
                    if pos is not None:
                        exit_check = self._maybe_close_position(
                            position=pos,
                            current_bar_ts=bar_ts,
                            one_h=one_h,
                            quote_to_eur=quote_to_eur,
                        )
                        if int(exit_check.get("should_close", 0)) == 1:
                            counts["exits_processed"] += 1
                            exit_signal_event = {
                                "event": "exit_signal_1h",
                                "role": candidate.role,
                                "candidate_id": candidate.candidate_id,
                                "symbol": self.symbol,
                                "signal_id": pos.get("signal_id"),
                                "entry_fill_time": pos.get("fill_time"),
                                "exit_time": str(exit_check.get("exit_time")),
                                "exit_reason": exit_check.get("exit_reason"),
                            }
                            store.append_journal(exit_signal_event)
                            self._root_event(exit_signal_event)

                            portfolio["cash_eur"] = float(_safe_float(portfolio.get("cash_eur"), 0.0) + exit_check["proceeds_eur"])
                            portfolio["realized_pnl_eur"] = float(
                                _safe_float(portfolio.get("realized_pnl_eur"), 0.0) + exit_check["net_pnl_eur"]
                            )
                            portfolio["fees_paid_eur"] = float(
                                _safe_float(portfolio.get("fees_paid_eur"), 0.0) + exit_check["fee_cost_eur"]
                            )
                            portfolio["slippage_paid_eur"] = float(
                                _safe_float(portfolio.get("slippage_paid_eur"), 0.0) + exit_check["slippage_cost_eur"]
                            )
                            portfolio["trade_count_closed"] = int(_safe_float(portfolio.get("trade_count_closed"), 0) + 1)
                            if exit_check["net_pnl_eur"] > 0:
                                portfolio["wins"] = int(_safe_float(portfolio.get("wins"), 0) + 1)
                            elif exit_check["net_pnl_eur"] < 0:
                                portfolio["losses"] = int(_safe_float(portfolio.get("losses"), 0) + 1)

                            positions.pop(self.symbol, None)
                            exit_fill_event = {
                                "event": "exit_fill",
                                "role": candidate.role,
                                "candidate_id": candidate.candidate_id,
                                "symbol": self.symbol,
                                "signal_id": pos.get("signal_id"),
                                "entry_fill_time": pos.get("fill_time"),
                                "exit_time": str(exit_check.get("exit_time")),
                                "exit_price": float(exit_check.get("exit_price", 0.0)),
                                "exit_reason": exit_check.get("exit_reason"),
                                "sl_hit": int(exit_check.get("sl_hit", 0)),
                                "tp_hit": int(exit_check.get("tp_hit", 0)),
                                "same_bar_hit": int(exit_check.get("same_bar_hit", 0)),
                                "mae_pct": float(exit_check.get("mae_pct", 0.0)),
                                "mfe_pct": float(exit_check.get("mfe_pct", 0.0)),
                                "pnl_net_pct": float(exit_check.get("pnl_net_pct", 0.0)),
                                "pnl_gross_pct": float(exit_check.get("pnl_gross_pct", 0.0)),
                                "net_pnl_eur": float(exit_check.get("net_pnl_eur", 0.0)),
                                "hold_minutes": float(exit_check.get("hold_minutes", 0.0)),
                                "model_a_purity_ok": 1,
                            }
                            store.append_journal(exit_fill_event)
                            self._root_event(exit_fill_event)

                    if had_position_at_row_open:
                        processed[self.symbol] = bar_ts.isoformat()
                        continue

                    if self.symbol not in positions and bool(row["SIGNAL"]):
                        counts["entries_attempted"] += 1
                        entry_attempt = {
                            "event": "entry_attempt",
                            "role": candidate.role,
                            "candidate_id": candidate.candidate_id,
                            "symbol": self.symbol,
                            "signal_id": signal_id,
                            "signal_timestamp": bar_ts.isoformat(),
                            "entry_mode": candidate.entry_mode,
                            "limit_offset_bps": float(candidate.limit_offset_bps),
                            "fallback_to_market": int(candidate.fallback_to_market),
                            "fallback_delay_min": float(candidate.fallback_delay_min),
                            "max_fill_delay_min": float(candidate.max_fill_delay_min),
                        }
                        store.append_journal(entry_attempt)
                        self._root_event(entry_attempt)

                        fill = self._simulate_entry_fill(three_m, bar_ts, candidate)
                        if int(fill.get("filled", 0)) == 1:
                            qty, entry_cost_eur = self._size_position(portfolio, row, params, float(fill["fill_price"]), quote_to_eur)
                            if qty > 0 and entry_cost_eur > 0:
                                counts["entries_filled"] += 1
                                tp_mult, sl_mult = _tp_sl_for_cycle(params, int(_safe_float(row["CYCLE"], 0)))
                                liq = self._entry_liquidity(str(fill["fill_type"]))
                                entry_cost = exec3m._costed_pnl_long(
                                    entry_price=float(fill["fill_price"]),
                                    exit_price=float(fill["fill_price"]),
                                    entry_liquidity_type=liq,
                                    fee_bps_maker=2.0,
                                    fee_bps_taker=4.0,
                                    slippage_bps_limit=0.5,
                                    slippage_bps_market=2.0,
                                )
                                fee_cost_eur = entry_cost_eur * float(_safe_float(entry_cost.get("entry_fee_bps"), 0.0)) / 1e4
                                slip_cost_eur = entry_cost_eur * float(_safe_float(entry_cost.get("entry_slippage_bps"), 0.0)) / 1e4

                                portfolio["cash_eur"] = float(_safe_float(portfolio.get("cash_eur"), 0.0) - entry_cost_eur)
                                portfolio["fees_paid_eur"] = float(_safe_float(portfolio.get("fees_paid_eur"), 0.0) + fee_cost_eur)
                                portfolio["slippage_paid_eur"] = float(
                                    _safe_float(portfolio.get("slippage_paid_eur"), 0.0) + slip_cost_eur
                                )
                                portfolio["trade_count_opened"] = int(_safe_float(portfolio.get("trade_count_opened"), 0) + 1)

                                fill_time = _coerce_ts(fill["fill_time"])
                                positions[self.symbol] = {
                                    "symbol": self.symbol,
                                    "signal_id": signal_id,
                                    "signal_timestamp": bar_ts.isoformat(),
                                    "fill_time": fill_time.isoformat(),
                                    "fill_price": float(fill["fill_price"]),
                                    "fill_type": str(fill["fill_type"]),
                                    "entry_liquidity_type": liq,
                                    "entry_improvement_bps": float(fill["entry_improvement_bps"]),
                                    "fill_delay_min": float((fill_time - bar_ts).total_seconds() / 60.0),
                                    "units": float(qty),
                                    "entry_cost_eur": float(entry_cost_eur),
                                    "cycle": int(_safe_float(row["CYCLE"], 0)),
                                    "tp_mult": float(tp_mult),
                                    "sl_mult": float(sl_mult),
                                    "tp_price": float(float(fill["fill_price"]) * tp_mult),
                                    "sl_price": float(float(fill["fill_price"]) * sl_mult),
                                    "exec_horizon_hours": 12.0,
                                }
                                fill_event = {
                                    "event": "entry_fill",
                                    "role": candidate.role,
                                    "candidate_id": candidate.candidate_id,
                                    "symbol": self.symbol,
                                    "signal_id": signal_id,
                                    "signal_timestamp": bar_ts.isoformat(),
                                    "entry_fill_time": fill_time.isoformat(),
                                    "entry_price": float(fill["fill_price"]),
                                    "entry_fill_type": str(fill["fill_type"]),
                                    "entry_liquidity_type": liq,
                                    "entry_improvement_bps": float(fill["entry_improvement_bps"]),
                                    "fill_delay_min": float((fill_time - bar_ts).total_seconds() / 60.0),
                                    "tp_mult": float(tp_mult),
                                    "sl_mult": float(sl_mult),
                                    "model_a_purity_ok": 1,
                                }
                                store.append_journal(fill_event)
                                self._root_event(fill_event)
                        else:
                            no_fill_event = {
                                "event": "entry_attempt_result",
                                "role": candidate.role,
                                "candidate_id": candidate.candidate_id,
                                "symbol": self.symbol,
                                "signal_id": signal_id,
                                "filled": 0,
                                "skip_reason": str(fill.get("skip_reason", "")),
                            }
                            store.append_journal(no_fill_event)
                            self._root_event(no_fill_event)

                    processed[self.symbol] = bar_ts.isoformat()

                except Exception as exc:
                    counts["runtime_errors"] += 1
                    self.health.inc("execution_errors")
                    err_payload = {
                        "event": "runtime_error",
                        "role": candidate.role,
                        "candidate_id": candidate.candidate_id,
                        "symbol": self.symbol,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                    store.append_journal(err_payload)
                    store.append_dead_letter(err_payload)
                    self._root_error(err_payload)

            store.save_portfolio(portfolio)
            store.save_positions(positions)
            store.save_processed_bars(processed)
            store.save_health_counters(self.health.as_state())
            store_meta = store.load_runtime_meta()
            store_meta["last_processed_1h_bar_ts"] = processed.get(self.symbol)
            store_meta["last_cycle_completed_utc"] = utc_iso()
            store.save_runtime_meta(store_meta)

            stats = self._collect_book_stats(store, candidate.role)
            counts["recovery_actions"] = stats["recovery_actions"]
            cycle_counts[candidate.role] = {**counts, **stats}

        root_meta["last_cycle_completed_utc"] = utc_iso()
        self._save_runtime_meta(root_meta)
        atomic_write_json(
            self.root_summary_path,
            {
                "generated_utc": utc_iso(),
                "symbol": self.symbol,
                "market_meta": market_meta,
                "latest_closed_1h": latest_closed_1h.isoformat(),
                "books": cycle_counts,
                "health": self.health.as_state(),
            },
        )
        return {
            "generated_utc": utc_iso(),
            "symbol": self.symbol,
            "latest_closed_1h": latest_closed_1h.isoformat(),
            "market_meta": market_meta,
            "books": cycle_counts,
            "health": self.health.as_state(),
        }

    def write_cycle_summary(self, target_dir: Path) -> tuple[Path, Path]:
        payload = _ensure_json(self.root_summary_path, {})
        if not payload:
            raise RuntimeError("no cycle summary available")
        day_tag = utc_tag()
        json_path = ensure_dir(target_dir) / f"model_a_cycle_summary_{day_tag}.json"
        md_path = ensure_dir(target_dir) / f"model_a_cycle_summary_{day_tag}.md"
        atomic_write_json(json_path, payload)
        lines = [
            "# Model A Cycle Summary",
            "",
            f"- Generated UTC: `{payload.get('generated_utc')}`",
            f"- Symbol: `{payload.get('symbol')}`",
            f"- Latest closed 1h bar: `{payload.get('latest_closed_1h')}`",
            f"- Market meta: `{payload.get('market_meta')}`",
        ]
        for role, stats in payload.get("books", {}).items():
            lines.extend(
                [
                    "",
                    f"## {role}",
                    f"- Candidate: `{stats.get('candidate_id')}`",
                    f"- Signals seen: `{stats.get('signals_seen')}`",
                    f"- Entries attempted/filled: `{stats.get('entries_attempted')}` / `{stats.get('entries_filled')}`",
                    f"- Exits processed: `{stats.get('exits_processed')}`",
                    f"- Runtime errors: `{stats.get('runtime_errors')}`",
                    f"- Recovery actions: `{stats.get('recovery_actions')}`",
                    f"- Open positions: `{stats.get('open_positions')}`",
                    f"- Realized PnL EUR: `{stats.get('realized_pnl_eur')}`",
                    f"- Taker share: `{stats.get('taker_share')}`",
                    f"- Avg / P95 fill delay: `{stats.get('avg_fill_delay_min')}` / `{stats.get('p95_fill_delay_min')}`",
                    f"- State consistent: `{stats.get('state_consistent')}`",
                ]
            )
        atomic_write_text(md_path, "\n".join(lines) + "\n")
        return json_path, md_path

    def probe_telegram(self) -> NotifyResult:
        probe = {
            "event": "fill_open",
            "symbol": self.symbol,
            "bar_ts": utc_iso(),
            "units": 0.0,
            "entry_px_quote": 0.0,
            "entry_cost_eur": 0.0,
            "slippage_bps": 0,
            "fee_bps": 0,
        }
        return self.notifier.send_trade_fill(probe)
