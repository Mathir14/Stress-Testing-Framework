"""Streamlit reporting adapter tests (``views/reporters.py``).

Architecture reference: ``architecture.md`` §4.6, ADR-002; conventions §3, §7.

**Tier T1** — requires Streamlit, and skips cleanly without it.

The headline regression is :func:`test_render_error_reaches_the_user`, and the
reason it needs a *real* Streamlit runtime rather than a monkeypatched ``st`` is
the whole point of it.  :class:`~views.reporters.StreamlitReporter` passed
``key=`` to ``st.error`` / ``st.success`` / ``st.warning`` / ``st.info``.  Those
are *alert elements*, not widgets: on the pinned ``streamlit==1.54.0`` their
signatures are ``(body, *, icon=None, width='stretch')``, so the call raised::

    TypeError: AlertMixin.error() got an unexpected keyword argument 'key'

— from *every* rendered error in the application, including the one
conventions §7 mandates.  A test that replaced ``views.reporters.st`` with a
mock would have accepted the call and passed, which is why no such test existed:
the suite was 645 green while every error path in the app raised a ``TypeError``
instead of showing the user a message.  :func:`test_alert_elements_take_no_key`
pins the underlying fact, and
:func:`test_render_error_reaches_the_user` pins the behaviour end to end.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("streamlit", reason="the Streamlit reporter needs the Streamlit runtime")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.errors import (  # noqa: E402
    FrameworkError,
    ParameterOutOfRangeError,
    ValidationError,
)
from core.reporters import CollectingReporter, NullReporter, Reporter  # noqa: E402
from views.reporters import StreamlitReporter, render_error  # noqa: E402


#: Streamlit surface version this file's claims are about.  A bump here is a
#: prompt to re-check ``key`` support on the alert elements, not a routine bump.
PINNED = "1.54.0"


def _script(tmp_path, body: str):
    """Write ``body`` as a standalone Streamlit script and return an ``AppTest``.

    The script is executed by Streamlit's own script runner in-process, so
    ``views.reporters`` resolves through the ``sys.path`` entry ``conftest.py``
    installs.  Nothing is stubbed: this is the only kind of test that can catch
    a keyword the real ``st.error`` does not accept.
    """
    path = tmp_path / "reporter_probe.py"
    path.write_text(body, encoding="utf-8")
    app = AppTest.from_file(str(path), default_timeout=120)
    app.run()
    return app


def _no_exception(app: AppTest, context: str) -> None:
    """Assert the probe script produced no uncaught exception."""
    assert not app.exception, f"{context} raised: {[str(e.value) for e in app.exception]}"


# ── The pinned fact ───────────────────────────────────────────────────────────


def test_alert_elements_take_no_key():
    """The premise of the fix: alert elements accept no ``key``.

    If a future Streamlit adds ``key`` to ``st.error``, this test fails and asks
    whether the reporter should go back to using it — which is the correct way
    to find out, rather than discovering it from a live ``TypeError``.
    """
    assert st.__version__ == PINNED, f"this file pins streamlit=={PINNED}; found {st.__version__}"
    for level in ("error", "success", "warning", "info", "caption"):
        parameters = inspect.signature(getattr(st, level)).parameters
        assert "key" not in parameters, (
            f"st.{level} now accepts key=; revisit StreamlitReporter._decorate"
        )


# ── The regressions ───────────────────────────────────────────────────────────


def test_render_error_reaches_the_user(tmp_path):
    """Run-002 MAJOR-3: a domain error renders as ``st.error``, it does not raise.

    Before the fix this raised ``TypeError: AlertMixin.error() got an unexpected
    keyword argument 'key'``, so the mandated conventions §7 rendering never
    happened for any error in the app.
    """
    app = _script(
        tmp_path,
        "from core.errors import ParameterOutOfRangeError\n"
        "from views.reporters import render_error\n"
        "render_error(ParameterOutOfRangeError(\n"
        "    \"Parameter 'scale_factor' must be greater than 1.0; got 1.0.\",\n"
        "    field='scale_factor',\n"
        "    context={'op': 'scale_perturbation'},\n"
        "))\n",
    )
    _no_exception(app, "render_error(ParameterOutOfRangeError(...))")

    rendered = [element.value for element in app.error]
    assert any("greater than 1.0" in text for text in rendered), rendered
    # ``.value`` is load-bearing: an ``AppTest`` element defines ``__contains__``,
    # so ``"needle" in element`` is a membership test over the element's *data*
    # and silently answers ``False`` for every string -- which would make this
    # assertion unfalsifiable rather than merely wrong.
    assert any("ParameterOutOfRangeError" in element.value for element in app.caption), [
        element.value for element in app.caption
    ]


def test_render_error_shows_the_machine_context_as_a_caption(tmp_path):
    """``exc.context`` is for the machine; the user still needs to read it."""
    app = _script(
        tmp_path,
        "from core.errors import ValidationError\n"
        "from views.reporters import render_error\n"
        "render_error(ValidationError(\n"
        '    "Feature frame repeats 1 column name(s): a.",\n'
        "    context={'duplicate_columns': ['a']},\n"
        "))\n",
    )
    _no_exception(app, "render_error(ValidationError(...))")

    # The machine payload is the *caption*; the human-readable sentence is the
    # alert.  Asserting the message in the caption would pass for the wrong
    # reason -- ``render_error`` renders it with ``st.error``, and conventions §3
    # requires the actionable sentence to be the thing the user is shown.
    captions = [element.value for element in app.caption]
    assert any("duplicate_columns" in text for text in captions), captions
    assert any("repeats 1 column name" in element.value for element in app.error), [
        element.value for element in app.error
    ]


def test_a_non_framework_error_renders_without_an_error_type_caption(tmp_path):
    """``render_error`` accepts any exception; only framework ones are typed."""
    app = _script(
        tmp_path,
        "from views.reporters import render_error\nrender_error(RuntimeError('plain failure'))\n",
    )
    _no_exception(app, "render_error(RuntimeError(...))")
    assert any("plain failure" in e.value for e in app.error)
    assert not any("Error type" in element.value for element in app.caption), (
        "a non-framework error has no type caption to show"
    )


def test_every_reporter_level_renders(tmp_path):
    """All four protocol levels reach Streamlit — none of them raises."""
    app = _script(
        tmp_path,
        "from views.reporters import StreamlitReporter\n"
        "reporter = StreamlitReporter()\n"
        "reporter.info('info body')\n"
        "reporter.success('success body')\n"
        "reporter.warning('warning body')\n"
        "reporter.error('error body')\n",
    )
    _no_exception(app, "all four StreamlitReporter levels")
    for element_name, needle in (
        ("info", "info body"),
        ("success", "success body"),
        ("warning", "warning body"),
        ("error", "error body"),
    ):
        rendered = [element.value for element in getattr(app, element_name)]
        assert any(needle in text for text in rendered), (element_name, rendered)


# ── The reporter's own contract ───────────────────────────────────────────────


def test_key_prefix_reaches_the_message_text(tmp_path):
    """``key_prefix`` still distinguishes two reporters in one script run.

    It is now a *text* prefix rather than a widget key, because alert elements
    have no key to set — but it must not be silently dropped.
    """
    app = _script(
        tmp_path,
        "from views.reporters import StreamlitReporter\n"
        "StreamlitReporter(key_prefix='[model] ').success('saved')\n",
    )
    _no_exception(app, "StreamlitReporter(key_prefix=...)")
    assert any("[model] saved" in e.value for e in app.success)


def test_the_default_prefix_leaves_the_message_verbatim(tmp_path):
    """No prefix means byte-identical text — no stray separator characters."""
    app = _script(
        tmp_path,
        "from views.reporters import StreamlitReporter\nStreamlitReporter().success('saved')\n",
    )
    _no_exception(app, "StreamlitReporter()")
    assert any(e.value == "saved" for e in app.success), [e.value for e in app.success]


def test_render_error_uses_a_supplied_reporter(tmp_path):
    """The ``reporter`` parameter is honoured, so a view can supply its own sink."""
    seen = CollectingReporter()
    render_error(ValidationError("through the supplied sink"), seen)
    assert seen.of_level("error") == ["through the supplied sink"]


def test_streamlit_reporter_satisfies_the_domain_protocol():
    """Structural conformance (ADR-002): the protocol stays framework-agnostic."""
    assert isinstance(StreamlitReporter(), Reporter)
    for name in ("info", "success", "warning", "error"):
        assert callable(getattr(StreamlitReporter(), name))


def test_the_domain_sinks_still_satisfy_the_protocol():
    """The other two implementations must not have drifted either."""
    assert isinstance(NullReporter(), Reporter)
    assert isinstance(CollectingReporter(), Reporter)


def test_parameter_out_of_range_is_a_framework_error():
    """The hierarchy the view layer catches is the one it advertises."""
    assert issubclass(ParameterOutOfRangeError, ValidationError)
    assert issubclass(ValidationError, FrameworkError)
