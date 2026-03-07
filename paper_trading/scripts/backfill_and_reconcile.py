#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper_trading.app.config import load_settings
from paper_trading.app.main import run_daemon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a bounded number of bars and reconcile state")
    parser.add_argument("--replay-bars", type=int, default=72)
    parser.add_argument("--max-cycles", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(Path("/root/analysis/0.87"))
    run_daemon(
        settings,
        once=False,
        max_cycles=int(args.max_cycles),
        replay_bars=int(args.replay_bars),
        reset_on_start=False,
    )


if __name__ == "__main__":
    main()
