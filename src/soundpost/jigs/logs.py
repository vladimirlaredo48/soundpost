from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, rank: int = 0) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if rank == 0 else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
