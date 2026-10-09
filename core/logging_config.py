"""Logging configuration for the framework.

Architecture reference: ``architecture.md`` §2.2, §5; conventions §4.

Called exactly once from ``app.py``.  Every module obtains its own logger with
``logging.getLogger(__name__)``; nothing else calls ``basicConfig``.

Import rules (architecture.md §2.3): ``core.logging_config`` may import
``core.config`` only.
"""

from __future__ import annotations

import logging
import sys

from core.config import AppConfig, get_config

__all__ = ["configure_logging"]

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(config: AppConfig | None = None) -> logging.Logger:
    """Install a single stderr handler at the configured level.

    Idempotent: calling it twice replaces the handler this function installed
    rather than stacking duplicates, so Streamlit's repeated script execution
    cannot multiply log lines.

    Args:
        config: Configuration supplying ``log_level``; ``None`` uses
            ``get_config()``.

    Returns:
        logging.Logger: The application root logger.
    """
    active = config if config is not None else get_config()
    level = getattr(logging, str(active.log_level).upper(), logging.INFO)

    logger = logging.getLogger("stf")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_stf_handler", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.setLevel(level)
    handler._stf_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)

    logger.info(
        "Logging configured at level %s (random_seed=%s)",
        active.log_level,
        active.random_seed,
    )
    return logger
