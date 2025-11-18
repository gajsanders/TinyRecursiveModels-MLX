"""Simple logging utilities used by tests.

The tests capture stderr; to make outputs predictable we write a simple
formatted message directly to stderr rather than depending on the global
logging configuration (which pytest may override).
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any


def _format(level: str, msg: Any) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} - {level} - {msg}\n"


def log_msg(msg: Any, level: str = "INFO") -> None:
    """Write a log message to stderr with a simple format.

    This keeps behavior deterministic for tests which capture stderr.
    """
    level = str(level).upper()
    # validate level; default to INFO for unknown levels (tests expect this)
    valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid:
        level = "INFO"
    sys.stderr.write(_format(level, msg))


def debug(msg: Any) -> None:
    log_msg(msg, "DEBUG")


def info(msg: Any) -> None:
    log_msg(msg, "INFO")


def warning(msg: Any) -> None:
    log_msg(msg, "WARNING")


def error(msg: Any) -> None:
    log_msg(msg, "ERROR")


def critical(msg: Any) -> None:
    log_msg(msg, "CRITICAL")
