#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_trading.app.config import load_settings
from paper_trading.app.data_feed import DataFeed
from paper_trading.app.main import _startup_reset
from paper_trading.app.notifier import TelegramNotifier
from paper_trading.app.state_store import StateStore
from paper_trading.app.universe import resolve_universe
from paper_trading.app.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard reset the paper trader state and anchor it forward-only from now.")
    parser.add_argument("--hard-reset-now", action="store_true", help="Execute the hard reset immediately")
    parser.add_argument("--start-capital", type=float, default=320.0, help="Reset paper capital to this EUR amount")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hard_reset_now:
        raise SystemExit("Refusing to reset without --hard-reset-now")

    settings = load_settings(Path("/root/analysis/0.87"))
    logger = configure_logging(settings.logs_dir / "service.log", settings.logs_dir / "errors.log", settings.log_level)
    state = StateStore(settings.state_dir)
    state.initialize(settings.start_equity_eur)
    health_stub = state.load_health_counters()
    feed = DataFeed(settings, logger, health_stub)
    universe = resolve_universe(settings)
    notifier = TelegramNotifier(settings, logger)
    result = _startup_reset(
        settings=settings,
        state=state,
        universe=universe,
        feed=feed,
        logger=logger,
        start_capital_eur=float(args.start_capital),
        notifier=notifier,
    )
    state.append_journal({"event": "startup_reset", **result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
