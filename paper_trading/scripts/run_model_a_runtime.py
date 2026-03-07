#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path("/root/analysis/0.87").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.app.config import load_settings  # noqa: E402
from paper_trading.app.model_a_runtime import ModelAPaperRuntime  # noqa: E402
from paper_trading.app.utils.logging_utils import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dedicated Model A paper/shadow runtime for SOLUSDT")
    ap.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    ap.add_argument("--max-cycles", type=int, default=None, help="Maximum cycles to run")
    ap.add_argument("--latest-only", action="store_true", help="Process only the latest closed 1h bar")
    ap.add_argument("--max-rows", type=int, default=None, help="Cap processed 1h rows per cycle")
    ap.add_argument("--reset", action="store_true", help="Reset Model A state before starting")
    ap.add_argument("--force-local-only", action="store_true", help="Disable remote market data and use local data only")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(PROJECT_ROOT)
    logger = configure_logging(
        settings.logs_dir / "model_a_runtime.log",
        settings.logs_dir / "model_a_runtime_errors.log",
        settings.log_level,
    )

    runtime = ModelAPaperRuntime(
        settings=settings,
        logger=logger,
        force_local_only=bool(args.force_local_only),
    )
    if args.reset:
        runtime.hard_reset()

    cycles = 0
    latest_only = bool(args.latest_only)
    while True:
        cycles += 1
        result = runtime.run_cycle(latest_only=latest_only, max_rows=args.max_rows)
        runtime.write_cycle_summary(settings.reports_dir)
        logger.info("model_a_cycle_done cycle=%s result=%s", cycles, json.dumps(result))

        if args.once:
            break
        if args.max_cycles is not None and cycles >= int(args.max_cycles):
            break
        time.sleep(max(5, int(settings.poll_seconds)))


if __name__ == "__main__":
    main()
