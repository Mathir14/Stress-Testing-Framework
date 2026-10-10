"""Logging contract tests for ``core.logging_config`` (ADR-022).

Tier T0-safe: this file imports stdlib + pytest + ``core.config`` only — the
same closure ``core.logging_config`` needs — so it also runs in a trimmed
environment.

The load-bearing behaviour is *where* the handler lives: the previous private
``"stf"`` logger had no writers, so every ``getLogger(__name__)`` record was
dropped.  The handler now lives on the root logger, the namespace those names
propagate to, and repeat calls must not stack duplicates (Streamlit re-executes
``app.py`` on every interaction).
"""

from __future__ import annotations

import logging

import pytest

from core.config import get_config
from core.logging_config import NOISY_LOGGERS, configure_logging


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Undo everything a test's ``configure_logging`` call did.

    Root-logger configuration is process-global; without restoration one test's
    level or handler would leak into every later test.  Only ``_stf_handler``
    handlers are removed — pytest's own root handlers are left alone.
    """
    root = logging.getLogger()
    saved_root_level = root.level
    saved_noisy = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    for handler in list(root.handlers):
        if getattr(handler, "_stf_handler", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_root_level)
    for name, level in saved_noisy.items():
        logging.getLogger(name).setLevel(level)


def test_configure_logging_returns_the_root_logger() -> None:
    """ADR-022: the handler must live where ``getLogger(__name__)`` propagates."""
    assert configure_logging() is logging.getLogger()


def test_getlogger_name_records_reach_the_handler(capsys) -> None:
    """A module-named logger's record must reach the handler.

    This is the regression the whole ADR exists for: the old private ``"stf"``
    logger had no writers, so ``getLogger("core.config.anything").info(...)``
    was dropped entirely.
    """
    configure_logging(get_config({"log_level": "DEBUG"}))
    logging.getLogger("core.config.record_probe").info("reach-probe")
    captured = capsys.readouterr()
    assert "reach-probe" in captured.err


def test_warning_and_error_records_reach_the_handler(capsys) -> None:
    configure_logging()
    logging.getLogger("modules.probe").warning("warn-probe")
    logging.getLogger("modules.probe").error("error-probe")
    captured = capsys.readouterr()
    assert "warn-probe" in captured.err
    assert "error-probe" in captured.err


def test_configure_logging_is_idempotent() -> None:
    """Repeat calls replace rather than stack the ``_stf_handler``."""
    configure_logging()
    configure_logging()
    configure_logging()
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_stf_handler", False)]
    assert len(ours) == 1


def test_foreign_handlers_are_left_untouched() -> None:
    """Only the ``_stf_handler`` is ever removed; other handlers survive.

    This is what keeps ``pytest``'s own logging capture working after a call.
    """
    import io

    root = logging.getLogger()
    sentinel = logging.StreamHandler(stream=io.StringIO())
    sentinel.name = "foreign-sentinel"
    root.addHandler(sentinel)
    try:
        configure_logging()
        configure_logging()
        assert any(h is sentinel for h in root.handlers), (
            "a handler this function did not install must not be removed"
        )
    finally:
        root.removeHandler(sentinel)
        sentinel.close()


def test_log_level_is_the_validated_gate() -> None:
    """``AppConfig.log_level`` (ADR-023-validated) drives the root level."""
    configure_logging(get_config({"log_level": "ERROR"}))
    assert logging.getLogger().level == logging.ERROR


def test_default_uses_the_process_config() -> None:
    """``config=None`` resolves the singleton; its default level is INFO."""
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_noisy_third_party_loggers_are_quieted_to_warning() -> None:
    """streamlit/urllib3/matplotlib/PIL are quieted even under DEBUG."""
    configure_logging(get_config({"log_level": "DEBUG"}))
    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING, name