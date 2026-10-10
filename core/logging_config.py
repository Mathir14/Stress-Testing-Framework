"""Logging configuration for the framework.

Architecture reference: ``architecture.md`` §2.2, §5; conventions §4; ADR-022.

Called exactly once from ``app.py``.  Every module obtains its own logger with
``logging.getLogger(__name__)``; nothing else calls ``basicConfig``.

The one handler lives on the **root** logger (ADR-022), which is the namespace
those ``getLogger(__name__)`` names propagate to — the previous private
``"stf"`` logger had no writers, so every module record was dropped.  The call
is idempotent: Streamlit re-executes ``app.py`` on every interaction, and a
handler that stacked duplicates would multiply every log line.

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

#: Third-party loggers known to be noisy at DEBUG/INFO on the pinned stack.
#: Enumerated, never discovered at runtime (conventions §6): the list is part
#: of the contract asserted by ``tests/test_logging_config.py``.
NOISY_LOGGERS = ("streamlit", "urllib3", "matplotlib", "PIL")


def configure_logging(config: AppConfig | None = None) -> logging.Logger:
    """Install the single framework handler on the root logger.

    Idempotent: calling it twice replaces the ``_stf_handler`` this function
    installed (and only that handler) rather than stacking duplicates, so
    Streamlit's repeated script execution cannot multiply log lines.  Other
    handlers — pytest's ``caplog`` and any third-party filters — are left
    untouched.

    Args:
        config: Configuration supplying ``log_level``.  ``None`` uses
            ``get_config()``.  The level is already validated by
            ``AppConfig.__post_init__`` (ADR-023), so this function never has
            to fall back on an unrecognised name.

    Returns:
        logging.Logger: The root logger.
    """
    active = config if config is not None else get_config()
    level = logging.getLevelName(active.log_level)
    if not isinstance(level, int):  # pragma: no cover - unreachable post-ADR-023
        level = logging.INFO

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.setLevel(level)

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