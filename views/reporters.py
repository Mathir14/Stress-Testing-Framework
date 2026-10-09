"""The one Streamlit messaging adapter.

Architecture reference: ``architecture.md`` §4.6, ADR-002.

``core.reporters`` defines the :class:`~core.reporters.Reporter` protocol;
``StreamlitReporter` is the **only** implementation that touches ``st.*``.  Every
``st.success`` / ``st.error`` call that previously lived in
``modules/data_module.py`` and ``modules/model_module.py`` now goes through this
class, which is what allows the §2.1 matrix to forbid Streamlit imports below
the view layer without losing user feedback.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from core.errors import FrameworkError

__all__ = ["StreamlitReporter", "render_error", "render_errors"]


class StreamlitReporter:
    """Route domain-layer messages to Streamlit widgets.

    Implements the :class:`core.reporters.Reporter` protocol structurally; it is
    not a subclass, so the domain protocol stays framework-agnostic.

    Args:
        key_prefix: Optional prefix prepended to every message, so messages from
            two reporters in one script run stay distinguishable.  It is part of
            the **text**, not a Streamlit ``key``: the ``st.info`` /
            ``st.success`` / ``st.warning`` / ``st.error`` alert elements accept
            no ``key`` on ``streamlit==1.54.0`` (nor ``st.caption``), so passing
            one raised ``TypeError: AlertMixin.error() got an unexpected keyword
            argument 'key'`` from *every* error render — see ADR-013.
    """

    def __init__(self, key_prefix: str = "") -> None:
        self.key_prefix = key_prefix

    def _decorate(self, message: str) -> str:
        """Return ``message`` with this reporter's prefix applied, if any."""
        return f"{self.key_prefix}{message}" if self.key_prefix else message

    def info(self, message: str) -> None:
        """Render an informational message."""
        st.info(self._decorate(message))

    def success(self, message: str) -> None:
        """Render a success message."""
        st.success(self._decorate(message))

    def warning(self, message: str) -> None:
        """Render a warning message."""
        st.warning(self._decorate(message))

    def error(self, message: str) -> None:
        """Render an error message."""
        st.error(self._decorate(message))


def render_error(exc: BaseException, reporter: StreamlitReporter | None = None) -> None:
    """Render a domain error uniformly so users always get an actionable path.

    The pattern is fixed by architecture.md §5: the message is always a
    user-actionable sentence, and the machine-readable ``context`` payload is
    rendered as a caption when present.

    Args:
        exc: The exception to render.
        reporter: Reporter to use; a default :class:`StreamlitReporter` when
            omitted.

    Returns:
        None
    """
    sink = reporter if reporter is not None else StreamlitReporter()
    sink.error(str(exc))
    context = getattr(exc, "context", None)
    if context:
        st.caption(f"Details: {context}")
    if isinstance(exc, FrameworkError):
        st.caption(f"Error type: {type(exc).__name__}")


@contextmanager
def render_errors(reporter: StreamlitReporter | None = None) -> Iterator[None]:
    """Render a :class:`~core.errors.FrameworkError` raised inside the block.

    conventions §7 requires that *every* mutating action report success **or** a
    rendered ``FrameworkError``.  The uniform way to satisfy that in a view is
    this context manager, which is why it exists rather than a ``try`` block at
    each of the nine call sites: a copy-pasted ``try`` is easy to forget at the
    next button, whereas ``tests/test_architecture.py`` can assert that every
    view owning an ``st.button(`` uses *this*.

    Semantics match an explicit ``try``/``except FrameworkError: render_error``:
    the exception is **suppressed** and the remainder of the block is skipped,
    so a rejected domain call cannot fall through into display code that
    references a variable the failed call was going to bind.  Anything that is
    not a ``FrameworkError`` still propagates — a ``NameError`` or a pandas bug
    is a defect to be fixed, not a message to be shown to the user.

    Args:
        reporter: Reporter to use; a default :class:`StreamlitReporter` when
            omitted.

    Yields:
        None: control to the guarded block.

    Raises:
        BaseException: anything that is not a ``FrameworkError``, unchanged.
    """
    try:
        yield
    except FrameworkError as exc:
        render_error(exc, reporter)
