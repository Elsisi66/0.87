#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_trading.app.config import load_settings
from paper_trading.app.data_feed import DataFeed
from paper_trading.app.execution_sim import ExecutionSimulator
from paper_trading.app.portfolio import default_portfolio
from paper_trading.app.signal_runner import SignalRunner
from paper_trading.app.universe import resolve_universe
from paper_trading.app.utils.io import atomic_write_json, atomic_write_text
from paper_trading.app.utils.logging_utils import configure_logging
from paper_trading.app.utils.time_utils import utc_iso
from src.bot087.optim.ga import _ensure_indicators, _shift_cycles, build_entry_signal, compute_cycles, run_backtest_long_only


def _ts_str(value: Any) -> str:
    return pd.to_datetime(value, utc=True).isoformat()


def _closed_only(feed: DataFeed, df: pd.DataFrame) -> pd.DataFrame:
    closed_ts = feed.latest_closed_bar_ts(df)
    return df[df["Timestamp"] <= closed_ts].copy().reset_index(drop=True)


def _structural_parity(
    *,
    symbol: str,
    df: pd.DataFrame,
    params: dict[str, Any],
    runner: SignalRunner,
) -> dict[str, Any]:
    frame = runner.build_signal_frame(symbol, df, params)
    x = frame.frame.copy()

    y = _ensure_indicators(x, params)
    expected_signal = np.asarray(build_entry_signal(y, params, assume_prepared=True), dtype=bool)
    cycles_raw = compute_cycles(y, params)
    expected_cycle = _shift_cycles(cycles_raw, int(params.get("cycle_shift", 1)), int(params.get("cycle_fill", 2)))
    expected_atr_prev = y["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    expected_rsi_prev = y["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()

    actual_signal = x["SIGNAL"].astype(bool).to_numpy()
    actual_cycle = x["CYCLE"].astype(int).to_numpy()
    actual_atr_prev = x["ATR_PREV"].astype(float).to_numpy()
    actual_rsi_prev = x["RSI_PREV"].astype(float).to_numpy()

    signal_mismatch = int(np.sum(actual_signal != expected_signal))
    cycle_mismatch = int(np.sum(actual_cycle != expected_cycle))
    atr_prev_max_abs_diff = float(np.nanmax(np.abs(actual_atr_prev - expected_atr_prev))) if len(x) else 0.0
    rsi_prev_max_abs_diff = float(np.nanmax(np.abs(actual_rsi_prev - expected_rsi_prev))) if len(x) else 0.0

    return {
        "symbol": symbol,
        "bars_checked": int(len(x)),
        "signal_mismatch_count": signal_mismatch,
        "cycle_mismatch_count": cycle_mismatch,
        "atr_prev_max_abs_diff": atr_prev_max_abs_diff,
        "rsi_prev_max_abs_diff": rsi_prev_max_abs_diff,
        "signal_match": signal_mismatch == 0,
        "cycle_match": cycle_mismatch == 0,
        "shift_feature_match": atr_prev_max_abs_diff <= 1e-12 and rsi_prev_max_abs_diff <= 1e-12,
    }


def _execution_replay_events(
    *,
    symbol: str,
    frame: pd.DataFrame,
    params: dict[str, Any],
    fee_bps: float,
    slippage_bps: int,
    initial_equity: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Any], dict[str, Any]]:
    sim = ExecutionSimulator(fee_bps=fee_bps, slippage_choices_bps=[slippage_bps], seed=7)
    portfolio = default_portfolio(initial_equity)
    positions: dict[str, Any] = {}
    entries: list[tuple[str, str]] = []
    exits: list[tuple[str, str]] = []

    for _, row in frame.iterrows():
        out = sim.process_bar(
            symbol=symbol,
            row=row.to_dict(),
            params=params,
            quote_to_eur=1.0,
            portfolio=portfolio,
            positions=positions,
        )
        for evt in out.events:
            if evt.get("event") == "fill_open":
                entries.append((_ts_str(evt["bar_ts"]), "open"))
            elif evt.get("event") == "fill_close":
                exits.append((_ts_str(evt["bar_ts"]), str(evt.get("reason"))))

    return entries, exits, portfolio, positions


def _execution_parity(
    *,
    symbol: str,
    frame: pd.DataFrame,
    params: dict[str, Any],
    fee_bps: float,
    slippage_bps: int,
    initial_equity: float,
) -> dict[str, Any]:
    bt_trades, bt_metrics = run_backtest_long_only(
        frame,
        symbol,
        params,
        initial_equity,
        fee_bps,
        slippage_bps,
        collect_trades=True,
        assume_prepared=True,
        return_equity_curve=False,
    )

    bt_entries = [(_ts_str(t["entry_ts"]), "open") for t in bt_trades]
    bt_exits = [(_ts_str(t["exit_ts"]), str(t.get("reason"))) for t in bt_trades]

    sim_entries, sim_exits, sim_portfolio, sim_positions = _execution_replay_events(
        symbol=symbol,
        frame=frame,
        params=params,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
    )

    return {
        "symbol": symbol,
        "slippage_bps_fixed": int(slippage_bps),
        "entry_count_backtest": len(bt_entries),
        "entry_count_paper_sim": len(sim_entries),
        "exit_count_backtest": len(bt_exits),
        "exit_count_paper_sim": len(sim_exits),
        "entry_timing_match": bt_entries == sim_entries,
        "exit_timing_reason_match": bt_exits == sim_exits,
        "backtest_net_profit_quote": float(bt_metrics.get("net_profit", 0.0)),
        "paper_realized_pnl_quote": float(sim_portfolio.get("realized_pnl_eur", 0.0)),
        "paper_open_positions_after_replay": int(len(sim_positions)),
    }


def _write_signal_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PaperSignalFrameRow",
        "type": "object",
        "required": [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "RSI",
            "ATR",
            "WILLR",
            "ADX",
            "PLUS_DI",
            "MINUS_DI",
            "SIGNAL",
            "CYCLE",
            "ATR_PREV",
            "RSI_PREV",
            "BAR_INDEX",
        ],
        "properties": {
            "Timestamp": {"type": "string", "format": "date-time"},
            "Open": {"type": "number"},
            "High": {"type": "number"},
            "Low": {"type": "number"},
            "Close": {"type": "number"},
            "Volume": {"type": "number"},
            "RSI": {"type": "number"},
            "ATR": {"type": "number"},
            "WILLR": {"type": "number"},
            "ADX": {"type": "number"},
            "PLUS_DI": {"type": "number"},
            "MINUS_DI": {"type": "number"},
            "SIGNAL": {"type": "boolean"},
            "CYCLE": {"type": "integer", "minimum": 0, "maximum": 4},
            "ATR_PREV": {"type": "number"},
            "RSI_PREV": {"type": "number"},
            "BAR_INDEX": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": True,
    }
    atomic_write_json(path, schema)


def main() -> None:
    root = Path("/root/analysis/0.87")
    settings = load_settings(root)
    logger = configure_logging(settings.logs_dir / "service.log", settings.logs_dir / "errors.log", settings.log_level)
    feed = DataFeed(settings, logger, {})
    runner = SignalRunner(logger)
    universe = resolve_universe(settings)

    symbols = universe.symbols[: min(5, len(universe.symbols))]
    structural_results: list[dict[str, Any]] = []

    first_symbol_frame: pd.DataFrame | None = None
    first_symbol_params: dict[str, Any] | None = None
    first_symbol: str | None = None

    for symbol in symbols:
        params = runner.load_symbol_params(symbol, universe.symbol_params[symbol])
        raw_df, _meta = feed.fetch_ohlcv_1h(symbol, limit=min(settings.max_bars_fetch, 1200))
        df = _closed_only(feed, raw_df)
        if len(df) > 600:
            df = df.tail(600).reset_index(drop=True)
        structural = _structural_parity(symbol=symbol, df=df, params=params, runner=runner)
        structural_results.append(structural)
        if first_symbol_frame is None:
            first_symbol = symbol
            first_symbol_params = params
            first_symbol_frame = runner.build_signal_frame(symbol, df, params).frame

    if first_symbol is None or first_symbol_frame is None or first_symbol_params is None:
        raise RuntimeError("No symbols available for parity check")

    exec_parity = _execution_parity(
        symbol=first_symbol,
        frame=first_symbol_frame,
        params=first_symbol_params,
        fee_bps=settings.fee_bps,
        slippage_bps=2,
        initial_equity=10000.0,
    )

    overall_structural_ok = all(
        r["signal_match"] and r["cycle_match"] and r["shift_feature_match"] for r in structural_results
    )
    overall_ok = (
        overall_structural_ok
        and bool(exec_parity["entry_timing_match"])
        and bool(exec_parity["exit_timing_reason_match"])
    )

    payload = {
        "generated_utc": utc_iso(),
        "universe_source_path": universe.source_path,
        "symbols_checked": symbols,
        "structural_results": structural_results,
        "execution_parity_reference_symbol": first_symbol,
        "execution_parity": exec_parity,
        "overall_structural_ok": overall_structural_ok,
        "overall_parity_ok": overall_ok,
    }

    json_path = settings.reports_dir / "paper_phaseP2_parity_check_results.json"
    md_path = settings.reports_dir / "paper_phaseP2_parity_check_report.md"
    signal_schema_path = settings.reports_dir / "paper_phaseP2_signal_schema.json"

    atomic_write_json(json_path, payload)
    _write_signal_schema(signal_schema_path)

    lines = [
        "# Phase P2 Parity Check Report",
        "",
        f"- Generated UTC: `{payload['generated_utc']}`",
        f"- Universe source: `{universe.source_path}`",
        f"- Symbols checked: `{', '.join(symbols)}`",
        f"- Structural parity OK: `{overall_structural_ok}`",
        f"- Execution parity reference symbol: `{first_symbol}`",
        f"- Execution entry timing match: `{exec_parity['entry_timing_match']}`",
        f"- Execution exit timing/reason match: `{exec_parity['exit_timing_reason_match']}`",
        f"- Overall parity OK: `{overall_ok}`",
        "",
        "## Structural Checks",
    ]
    for row in structural_results:
        lines.append(
            f"- {row['symbol']}: bars={row['bars_checked']} "
            f"signal_mismatch={row['signal_mismatch_count']} "
            f"cycle_mismatch={row['cycle_mismatch_count']} "
            f"atr_prev_max_diff={row['atr_prev_max_abs_diff']:.6g} "
            f"rsi_prev_max_diff={row['rsi_prev_max_abs_diff']:.6g}"
        )

    lines.extend(
        [
            "",
            "## Deterministic Execution Parity",
            f"- Symbol: `{first_symbol}`",
            f"- Fixed slippage bps: `{exec_parity['slippage_bps_fixed']}`",
            f"- Entry count backtest/paper: `{exec_parity['entry_count_backtest']}/{exec_parity['entry_count_paper_sim']}`",
            f"- Exit count backtest/paper: `{exec_parity['exit_count_backtest']}/{exec_parity['exit_count_paper_sim']}`",
            f"- Backtest net profit (quote): `{exec_parity['backtest_net_profit_quote']:.6f}`",
            f"- Paper replay realized pnl (quote): `{exec_parity['paper_realized_pnl_quote']:.6f}`",
        ]
    )

    atomic_write_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
