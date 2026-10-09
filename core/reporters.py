"""Message-reporting protocol for domain services.

Architecture reference: ``architecture.md`` §4.6, ADR-002.

Domain code raises typed errors and emits *messages*; only the presentation
layer knows how to display them.  ``views/reporters.py`` holds the single
Streamlit-backed implementation, so ``core/``, ``modules/`` and ``utils/`` can
never import Streamlit (architecture.md §2.1) while still reporting lifecycle
events to the user.

Import rules (architecture.md §2.3): ``core.reporters`` imports nothing
project-local.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Reporter", "NullReporter", "CollectingReporter"]


@runtime_checkable
class Reporter(Protocol):
    """A minimal message sink with four severity levels."""

    def info(self, message: str) -> None:
        """Report an informational lifecycle event."""
        ...

    def success(self, message: str) -> None:
        """Report that a mutating action completed."""
        ...

    def warning(self, message: str) -> None:
        """Report a recoverable degradation."""
        ...

    def error(self, message: str) -> None:
        """Report a failure the user should act on."""
        ...


class NullReporter:
    """Default reporter for domain services: discards every message.

    Chosen over a logging reporter so that domain code stays usable from a
    plain script and from tests without configuring logging, and so a missing
    reporter can never turn into a crash (architecture.md §4.6).
    """

    def info(self, message: str) -> None:
        """Discard an informational message."""

    def success(self, message: str) -> None:
        """Discard a success message."""

    def warning(self, message: str) -> None:
        """Discard a warning message."""

    def error(self, message: str) -> None:
        """Discard an error message."""


class CollectingReporter:
    """Records messages for assertions in tests.

    Examples:
        >>> reporter = CollectingReporter()
        >>> reporter.success("saved")
        >>> reporter.messages
        [('success', 'saved')]
    """

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        """Record an informational message."""
        self.messages.append(("info", message))

    def success(self, message: str) -> None:
        """Record a success message."""
        self.messages.append(("success", message))

    def warning(self, message: str) -> None:
        """Record a warning message."""
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        """Record an error message."""
        self.messages.append(("error", message))

    def of_level(self, level: str) -> list[str]:
        """Return only the messages recorded at ``level``.

        Args:
            level: One of ``info`` / ``success`` / ``warning`` / ``error``.

        Returns:
            list[str]: The recorded messages, in order.
        """
        return [message for recorded, message in self.messages if recorded == level]

    def clear(self) -> None:
        """Discard all recorded messages."""
        self.messages.clear()
