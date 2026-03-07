from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir


def configure_logging(log_path: Path, error_path: Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(log_path.parent)
    ensure_dir(error_path.parent)

    logger = logging.getLogger("paper_trading")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)

    err_handler = logging.FileHandler(error_path, encoding="utf-8")
    err_handler.setFormatter(formatter)
    err_handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logger.level)

    logger.addHandler(file_handler)
    logger.addHandler(err_handler)
    logger.addHandler(stream_handler)

    return logger
