from __future__ import annotations

import argparse
import hashlib
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings, load_settings
from .data_feed import DataFeed
from .execution_sim import ExecutionSimulator
from .forward_monitor import ensure_forward_monitor_dir, write_daily_truth_pack
from .health import HealthTracker
from .notifier import TelegramNotifier, build_daily_summary, should_emit_daily_summary
from .portfolio import default_portfolio, total_equity_eur
from .reconciler import Reconciler
from .scheduler import Scheduler
from .signal_runner import SignalRunner
from .state_store import StateStore
from .universe import ResolvedUniverse, resolve_universe
from .utils.io import atomic_write_json, atomic_write_text
from .utils.logging_utils import configure_logging
from .utils.time_utils import date_yyyymmdd, utc_iso, utc_now, utc_tag


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_repaired_posture_lock(
    *,
    settings: Settings,
    universe: ResolvedUniverse,
    logger,
) -> dict[str, Any]:
    allowlist = sorted({str(sym).upper() for sym in settings.paper_symbol_allowlist if str(sym).strip()})
    observed = sorted({str(sym).upper() for sym in universe.symbols if str(sym).strip()})
    if observed != allowlist:
        raise RuntimeError(
            f"paper universe mismatch against allowlist: observed={observed} expected={allowlist}"
        )
    if observed != ["SOLUSDT"]:
        raise RuntimeError(f"paper deployment guard requires SOLUSDT-only active set, observed={observed}")

    sol_params = Path(universe.symbol_params.get("SOLUSDT", ""))
    if not sol_params.exists():
        raise RuntimeError(f"SOLUSDT params path missing: {sol_params}")
    sol_hash = _sha256_file(sol_params)

    winner_id = str(universe.winner_config_ids.get("SOLUSDT", "")).strip()
    if winner_id != settings.required_active_strategy_id:
        raise RuntimeError(
            f"SOLUSDT winner mismatch: observed={winner_id} expected={settings.required_active_strategy_id}"
        )

    if not settings.repaired_contract_defer_exit_to_next_bar:
        raise RuntimeError("repaired contract flag mismatch: defer_exit_to_next_bar must be true")

    payload = {
        "posture_freeze_dir": str(universe.posture_freeze_dir or settings.repaired_posture_freeze_dir),
        "active_subset_csv": str(settings.repaired_active_subset_csv),
        "active_params_dir": str(settings.repaired_active_params_dir),
        "active_symbols": observed,
        "sol_params_path": str(sol_params.resolve()),
        "sol_params_sha256": sol_hash,
        "sol_winner_config_id": winner_id,
        "required_active_strategy_id": settings.required_active_strategy_id,
        "repaired_contract_flag": "defer_exit_to_next_bar=True",
        "startup_assertions_passed": True,
    }
    logger.critical("REPAIRED_POSTURE_ASSERT %s", payload)
    return payload


def _discover_forward_anchor(
    *,
    settings: Settings,
    universe: ResolvedUniverse,
    feed: DataFeed,
    logger,
) -> tuple[pd.Timestamp, list[str]]:
    anchors: list[pd.Timestamp] = []
    failures: list[str] = []
    for symbol in universe.symbols:
        try:
            df, _meta = feed.fetch_ohlcv_1h(symbol, limit=min(settings.max_bars_fetch, 8))
            if df.empty:
                failures.append(symbol)
                continue
            anchors.append(feed.latest_closed_bar_ts(df))
        except Exception as exc:
            failures.append(symbol)
            logger.error("forward_anchor_failed symbol=%s err=%s", symbol, str(exc))
    if not anchors:
        raise RuntimeError("unable to determine start_from_bar_ts for any tracked symbol")
    return max(anchors), failures


def _ensure_forward_start_meta(
    *,
    settings: Settings,
    state: StateStore,
    universe: ResolvedUniverse,
    feed: DataFeed,
    logger,
    reason: str,
) -> dict[str, Any]:
    runtime_meta = state.load_runtime_meta()
    if runtime_meta.get("start_from_bar_ts"):
        runtime_meta.setdefault("backlog_replay_disabled", True)
        runtime_meta.setdefault("forward_only_mode", True)
        runtime_meta.setdefault("anchor_reason", reason)
        state.save_runtime_meta(runtime_meta)
        return runtime_meta

    anchor_ts, anchor_failures = _discover_forward_anchor(
        settings=settings,
        universe=universe,
        feed=feed,
        logger=logger,
    )
    runtime_meta = {
        **runtime_meta,
        "start_from_bar_ts": anchor_ts.isoformat(),
        "last_global_processed_bar_ts": None,
        "backlog_replay_disabled": True,
        "forward_only_mode": True,
        "anchor_reason": reason,
        "anchor_failures": anchor_failures,
        "updated_utc": utc_iso(),
    }
    state.save_runtime_meta(runtime_meta)
    logger.info(
        "forward_start_anchor_initialized start_from_bar_ts=%s reason=%s",
        runtime_meta["start_from_bar_ts"],
        reason,
    )
    return runtime_meta


def _startup_reset(
    *,
    settings: Settings,
    state: StateStore,
    universe: ResolvedUniverse,
    feed: DataFeed,
    logger,
    start_capital_eur: float | None = None,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    actions: list[str] = []
    reset_mode = "local_hard_reset"
    reset_flag = "local_hard_reset_applied"
    start_equity_eur = float(start_capital_eur if start_capital_eur is not None else settings.start_equity_eur)

    if settings.allow_testnet_reset:
        actions.append("testnet_reset_requested_but_not_configured_in_this_daemon")
        actions.append("falling_back_to_local_hard_reset")

    archive_dir = state.archive_state("hard_reset")

    portfolio = default_portfolio(start_equity_eur)
    portfolio["mode_note"] = "paper_local_ledger"
    if settings.unsafe_live_endpoint:
        portfolio["degraded_mode"] = True
        portfolio["mode_note"] = "unsafe_base_url_live_endpoint_forced_local_paper"

    state.journal_path.write_text("", encoding="utf-8")
    state.dead_letter_path.write_text("", encoding="utf-8")
    state.save_portfolio(portfolio)
    state.save_positions({})
    state.save_orders([])
    state.save_processed_bars({})
    state.save_quarantine({})
    state.save_health_counters(
        {
            "api_retries": 0,
            "api_failures": 0,
            "signal_errors": 0,
            "execution_errors": 0,
            "recovery_events": 0,
            "telegram_errors": 0,
            "quarantined_symbols": 0,
            "degraded_mode": bool(portfolio.get("degraded_mode", False)),
            "strategy_health": "GREEN",
        }
    )

    anchor_ts, anchor_failures = _discover_forward_anchor(
        settings=settings,
        universe=universe,
        feed=feed,
        logger=logger,
    )
    runtime_meta = {
        "start_from_bar_ts": anchor_ts.isoformat(),
        "last_global_processed_bar_ts": None,
        "last_reset_utc": utc_iso(),
        "backlog_replay_disabled": True,
        "forward_only_mode": True,
        "anchor_reason": "hard_reset_now",
        "anchor_failures": anchor_failures,
        "state_archive_dir": str(archive_dir),
    }
    state.save_runtime_meta(runtime_meta)

    actions.append(f"state_archived:{archive_dir}")
    actions.append("open_orders_cleared:local")
    actions.append("open_positions_cleared:local")
    actions.append("processed_bar_history_cleared")
    actions.append(f"start_from_bar_ts:{runtime_meta['start_from_bar_ts']}")
    if anchor_failures:
        actions.append(f"anchor_failures:{','.join(sorted(anchor_failures))}")

    reset_meta = {
        "generated_utc": utc_iso(),
        "reset_mode": reset_mode,
        "reset_flag": reset_flag,
        "actions": actions,
        "symbols": universe.symbols,
        "anchor_failures": anchor_failures,
        "archive_dir": str(archive_dir),
        "start_equity_eur": start_equity_eur,
        "start_from_bar_ts": runtime_meta["start_from_bar_ts"],
        "backlog_replay_disabled": True,
    }
    state.save_reset_marker(reset_meta)

    report_path = settings.reports_dir / f"startup_reset_report_{utc_tag()}.md"
    lines = [
        "# Startup Reset Report",
        "",
        f"- Timestamp UTC: `{reset_meta['generated_utc']}`",
        f"- Reset mode: `{reset_mode}`",
        f"- Reset flag: `{reset_flag}`",
        f"- Start equity EUR: `{start_equity_eur:.4f}`",
        f"- Start from bar TS: `{reset_meta['start_from_bar_ts']}`",
        f"- State archive dir: `{archive_dir}`",
        f"- Symbols tracked: `{', '.join(universe.symbols)}`",
        "",
        "## Actions",
    ]
    lines.extend([f"- {item}" for item in actions])
    if anchor_failures:
        lines.append("")
        lines.append("## Anchor Failures")
        lines.extend([f"- {sym}" for sym in anchor_failures])
    atomic_write_text(report_path, "\n".join(lines) + "\n")

    reset_meta["report_path"] = str(report_path)
    if notifier is not None:
        notify = notifier.send_reset_completed(reset_meta)
        reset_meta["telegram_sent"] = notify.sent
        reset_meta["telegram_reason"] = notify.reason

    logger.info(
        "startup_reset_complete mode=%s symbols=%s start_from_bar_ts=%s",
        reset_mode,
        len(universe.symbols),
        reset_meta["start_from_bar_ts"],
    )
    return reset_meta


def _load_quarantine(state: StateStore) -> dict[str, Any]:
    raw = state.load_quarantine()
    now = utc_now()
    active: dict[str, Any] = {}
    for symbol, payload in raw.items():
        until_str = payload.get("until_utc")
        if not until_str:
            continue
        until = pd.to_datetime(until_str, utc=True)
        if until > now:
            active[symbol] = payload
    if active != raw:
        state.save_quarantine(active)
    return active


def _quarantine_symbol(state: StateStore, symbol: str, *, minutes: int, reason: str, logger) -> None:
    q = state.load_quarantine()
    now = utc_now()
    until = now + timedelta(minutes=int(minutes))
    current = q.get(symbol, {})
    error_count = int(current.get("error_count", 0)) + 1
    q[symbol] = {
        "until_utc": until.isoformat(),
        "reason": reason,
        "error_count": error_count,
        "updated_utc": now.isoformat(),
    }
    state.save_quarantine(q)
    logger.error("symbol_quarantined symbol=%s until=%s reason=%s", symbol, until.isoformat(), reason)


def _clear_symbol_quarantine_if_any(state: StateStore, symbol: str) -> None:
    q = state.load_quarantine()
    if symbol in q:
        q.pop(symbol, None)
        state.save_quarantine(q)


def _record_equity_snapshot(
    *,
    state: StateStore,
    universe: ResolvedUniverse,
    feed: DataFeed,
) -> tuple[float, dict[str, float], dict[str, float]]:
    portfolio = state.load_portfolio()
    positions = state.load_positions()

    mark_prices: dict[str, float] = {}
    quote_to_eur_map: dict[str, float] = {}

    for symbol in universe.symbols:
        quote_asset = universe.quote_assets.get(symbol, "USDT")
        quote_to_eur, _fx_source = feed.quote_to_eur(quote_asset)
        quote_to_eur_map[symbol] = quote_to_eur

        pos = positions.get(symbol)
        if pos is not None:
            try:
                mark_df, _meta = feed.fetch_ohlcv_1h(symbol, limit=3)
                if mark_df.empty:
                    raise RuntimeError("empty mark frame")
                closed_ts = feed.latest_closed_bar_ts(mark_df)
                match = mark_df[mark_df["Timestamp"] == closed_ts]
                if match.empty:
                    mark_prices[symbol] = float(mark_df.iloc[-1]["Close"])
                else:
                    mark_prices[symbol] = float(match.iloc[-1]["Close"])
            except Exception:
                mark_prices[symbol] = float(pos.get("entry_px_quote", 0.0))

    equity = total_equity_eur(portfolio, positions, mark_prices, quote_to_eur_map)
    state.append_journal(
        {
            "event": "equity_snapshot",
            "equity_eur": equity,
            "cash_eur": float(portfolio.get("cash_eur", 0.0)),
            "open_positions": len(positions),
        }
    )
    return equity, mark_prices, quote_to_eur_map


def _process_cycle(
    *,
    settings: Settings,
    state: StateStore,
    universe: ResolvedUniverse,
    signal_runner: SignalRunner,
    reconciler: Reconciler,
    feed: DataFeed,
    health: HealthTracker,
    notifier: TelegramNotifier,
    logger,
) -> dict[str, Any]:
    allowed_symbols = {str(sym).upper() for sym in settings.paper_symbol_allowlist if str(sym).strip()}
    quarantine = _load_quarantine(state)
    runtime_meta = state.load_runtime_meta()
    start_from_raw = runtime_meta.get("start_from_bar_ts")
    start_from_ts = pd.to_datetime(start_from_raw, utc=True) if start_from_raw else None

    total_opened = 0
    total_closed = 0
    total_bars = 0
    degraded = settings.unsafe_live_endpoint or feed.circuit.is_open
    processed_bar_updates: list[pd.Timestamp] = []

    for symbol in universe.symbols:
        if symbol.upper() not in allowed_symbols:
            raise RuntimeError(f"symbol_not_allowlisted_for_paper_runtime:{symbol}")
        if symbol in quarantine:
            logger.warning("symbol_skipped_quarantine symbol=%s until=%s", symbol, quarantine[symbol].get("until_utc"))
            continue

        try:
            params = signal_runner.load_symbol_params(symbol, universe.symbol_params[symbol])
            df, meta = feed.fetch_ohlcv_1h(symbol, limit=settings.max_bars_fetch)
            if bool(meta.get("degraded", False)):
                degraded = True

            frame = signal_runner.build_signal_frame(symbol, df, params)
            max_bar_ts = feed.latest_closed_bar_ts(df)

            quote_asset = universe.quote_assets.get(symbol, "USDT")
            quote_to_eur, _fx_source = feed.quote_to_eur(quote_asset)

            result = reconciler.process_symbol_rows(
                symbol=symbol,
                signal_frame=frame,
                quote_to_eur=quote_to_eur,
                max_bar_ts=max_bar_ts,
                start_from_bar_ts=start_from_ts,
            )

            total_opened += result.opened
            total_closed += result.closed
            total_bars += result.bars_processed
            if result.bars_processed and result.last_processed_bar_ts:
                processed_bar_updates.append(pd.to_datetime(result.last_processed_bar_ts, utc=True))

            for fill_event in result.fill_events:
                notify = notifier.send_trade_fill(fill_event)
                if not notify.sent and notify.reason.startswith("telegram_error"):
                    health.inc("telegram_errors")
                state.append_journal(
                    {
                        "event": "trade_notification",
                        "symbol": symbol,
                        "bar_ts": fill_event.get("bar_ts"),
                        "fill_event": fill_event.get("event"),
                        "telegram_sent": notify.sent,
                        "telegram_reason": notify.reason,
                    }
                )

            _clear_symbol_quarantine_if_any(state, symbol)

        except Exception as exc:
            health.inc("signal_errors")
            state.append_dead_letter(
                {
                    "event": "symbol_cycle_error",
                    "symbol": symbol,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            q = state.load_quarantine().get(symbol, {})
            err_count = int(q.get("error_count", 0)) + 1
            if err_count >= settings.symbol_error_quarantine_threshold:
                _quarantine_symbol(
                    state,
                    symbol,
                    minutes=settings.symbol_quarantine_minutes,
                    reason=f"repeated_symbol_error:{exc}",
                    logger=logger,
                )
                health.inc("recovery_events")
            else:
                # Keep a soft counter even before quarantine threshold.
                _quarantine_symbol(state, symbol, minutes=5, reason=f"transient_error:{exc}", logger=logger)

    q_active = _load_quarantine(state)
    health.set_quarantined_symbols(len(q_active))
    health.set_degraded_mode(degraded)
    state.save_health_counters(health.as_state())
    if processed_bar_updates:
        runtime_meta = state.load_runtime_meta()
        runtime_meta["last_global_processed_bar_ts"] = max(processed_bar_updates).isoformat()
        runtime_meta["last_cycle_completed_utc"] = utc_iso()
        state.save_runtime_meta(runtime_meta)

    return {
        "bars_processed": total_bars,
        "opened": total_opened,
        "closed": total_closed,
        "degraded": degraded,
        "quarantined": len(q_active),
    }


def run_daemon(settings: Settings, *, once: bool, max_cycles: int | None, replay_bars: int | None, reset_on_start: bool) -> None:
    logger = configure_logging(settings.logs_dir / "service.log", settings.logs_dir / "errors.log", settings.log_level)

    logger.info("paper_daemon_start paper_mode=%s binance_mode=%s", settings.paper_mode, settings.binance_mode)
    if settings.require_repaired_posture_pack:
        logger.critical(
            "REPAIRED_POSTURE_MODE: fallback disabled freeze_dir=%s",
            settings.repaired_posture_freeze_dir,
        )
    if settings.unsafe_live_endpoint:
        logger.error(
            "unsafe_live_base_url_detected base_url=%s forcing_local_paper_mode",
            settings.binance_base_url,
        )

    state = StateStore(settings.state_dir)
    state.initialize(settings.start_equity_eur)

    health = HealthTracker(state.load_health_counters())
    feed = DataFeed(settings, logger, health.counters)
    signal_runner = SignalRunner(logger)
    execution_sim = ExecutionSimulator(
        settings.fee_bps,
        settings.slippage_bps_choices,
        seed=87,
        defer_exit_to_next_bar=bool(settings.repaired_contract_defer_exit_to_next_bar),
    )
    reconciler = Reconciler(
        state_store=state,
        signal_runner=signal_runner,
        execution_sim=execution_sim,
        health=health,
        logger=logger,
    )
    scheduler = Scheduler(poll_seconds=settings.poll_seconds)
    notifier = TelegramNotifier(settings, logger)

    universe = resolve_universe(settings)
    if not universe.symbols:
        msg = "no symbols resolved for paper universe"
        state.append_dead_letter({"event": "startup_fatal", "error": msg})
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        "resolved_universe symbols=%s source=%s priority=%s",
        ",".join(universe.symbols),
        universe.source_path,
        universe.source_priority,
    )
    posture_assert_payload = _assert_repaired_posture_lock(settings=settings, universe=universe, logger=logger)
    state.append_journal({"event": "startup_posture_assertions", **posture_assert_payload})

    startup_reset_meta = None
    if reset_on_start:
        startup_reset_meta = _startup_reset(
            settings=settings,
            state=state,
            universe=universe,
            feed=feed,
            logger=logger,
            notifier=notifier,
        )
        state.append_journal({"event": "startup_reset", **startup_reset_meta})

    _ensure_forward_start_meta(
        settings=settings,
        state=state,
        universe=universe,
        feed=feed,
        logger=logger,
        reason="daemon_bootstrap",
    )
    forward_monitor_dir = ensure_forward_monitor_dir(
        settings=settings,
        state=state,
        universe=universe,
        logger=logger,
    )

    if replay_bars and replay_bars > 0:
        logger.warning("replay_bars_ignored_forward_only_mode replay_bars=%s", replay_bars)

    cycles = 0
    while True:
        cycles += 1
        execution_sim.reset_guard_stats()
        cycle_out = _process_cycle(
            settings=settings,
            state=state,
            universe=universe,
            signal_runner=signal_runner,
            reconciler=reconciler,
            feed=feed,
            health=health,
            notifier=notifier,
            logger=logger,
        )
        guard_stats = execution_sim.snapshot_guard_stats()
        cycle_out["chronology_guard_stats"] = guard_stats
        logger.info(
            "chronology_guard_summary cycle=%s defer_exit_to_next_bar=%s stats=%s",
            cycles,
            bool(settings.repaired_contract_defer_exit_to_next_bar),
            guard_stats,
        )
        state.append_journal(
            {
                "event": "chronology_guard_summary",
                "defer_exit_to_next_bar": bool(settings.repaired_contract_defer_exit_to_next_bar),
                **guard_stats,
            }
        )

        equity, mark_prices, quote_to_eur_map = _record_equity_snapshot(state=state, universe=universe, feed=feed)
        logger.info(
            "cycle_done cycle=%s bars=%s opened=%s closed=%s equity_eur=%.4f degraded=%s quarantined=%s",
            cycles,
            cycle_out["bars_processed"],
            cycle_out["opened"],
            cycle_out["closed"],
            equity,
            cycle_out["degraded"],
            cycle_out["quarantined"],
        )

        now = scheduler.now_utc()
        portfolio = state.load_portfolio()
        portfolio["degraded_mode"] = bool(cycle_out["degraded"])
        portfolio["mode_note"] = "degraded" if cycle_out["degraded"] else "normal"
        state.save_portfolio(portfolio)

        write_daily_truth_pack(
            settings=settings,
            state=state,
            monitor_dir=forward_monitor_dir,
            target_day_utc=now,
            logger=logger,
        )

        if should_emit_daily_summary(
            now_utc=now,
            daily_summary_hour_utc=settings.daily_summary_hour_utc,
            last_summary_date=portfolio.get("last_summary_date"),
        ):
            summary = build_daily_summary(
                settings=settings,
                target_day_utc=now,
                portfolio=portfolio,
                positions=state.load_positions(),
                mark_prices_quote=mark_prices,
                quote_to_eur_map=quote_to_eur_map,
                health_snapshot=health.snapshot(),
            )
            notify = notifier.send_daily_summary(summary)
            if not notify.sent and notify.reason.startswith("telegram_error"):
                health.inc("telegram_errors")

            portfolio = state.load_portfolio()
            portfolio["last_summary_date"] = date_yyyymmdd(now)
            state.save_portfolio(portfolio)
            state.save_health_counters(health.as_state())
            state.append_journal(
                {
                    "event": "daily_summary",
                    "summary_date": summary.get("date_utc"),
                    "telegram_sent": notify.sent,
                    "telegram_reason": notify.reason,
                }
            )

        if once:
            break
        if max_cycles is not None and cycles >= max_cycles:
            break

        scheduler.sleep()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading daemon for bot087")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--max-cycles", type=int, default=None, help="Maximum cycles to run")
    parser.add_argument(
        "--replay-bars",
        type=int,
        default=None,
        help="Deprecated; ignored in forward-only mode",
    )
    parser.add_argument(
        "--no-startup-reset",
        action="store_true",
        help="Disable startup hard reset (for controlled smoke/restart tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(Path("/root/analysis/0.87"))
    try:
        run_daemon(
            settings,
            once=bool(args.once),
            max_cycles=args.max_cycles,
            replay_bars=args.replay_bars,
            reset_on_start=not bool(args.no_startup_reset),
        )
    except Exception as exc:
        logger = configure_logging(settings.logs_dir / "service.log", settings.logs_dir / "errors.log", settings.log_level)
        state = StateStore(settings.state_dir)
        state.append_dead_letter({"event": "fatal_runtime_error", "error": str(exc), "traceback": traceback.format_exc(limit=8)})
        TelegramNotifier(settings, logger).send_fatal_error(str(exc))
        raise


if __name__ == "__main__":
    main()
