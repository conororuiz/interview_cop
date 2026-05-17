"""Structured logging with rotation.

We deliberately keep stdlib `logging` rather than a third-party log lib to
avoid extra dependencies and to keep Textual's console capture clean.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import get_settings

_CONFIGURED = False


def setup_logging(quiet_console: bool = False) -> logging.Logger:
    """Configure root logging once. Safe to call multiple times.

    Args:
        quiet_console: if True, skip the stderr handler. Required when running
            inside Textual's TUI — otherwise log lines get smeared across the UI.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not quiet_console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(fmt)
        root.addHandler(stream)

    # Rotating file handler — 5 MB x 5 backups.
    log_file = settings.logs_dir / "transcriber.log"
    file_h = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # Quiet known noisy libs.
    for noisy in ("urllib3", "huggingface_hub", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.debug("Logging initialised at level %s", settings.log_level)
    return root
