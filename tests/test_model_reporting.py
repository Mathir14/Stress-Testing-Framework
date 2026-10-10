"""ModelTrainer reporter-boundary tests (run-014 M10, ADR-025).

**Tier T1** — importing :mod:`modules.model_module` pulls in the scientific
stack (scikit-learn, xgboost, plotly).

When the artifact store has already persisted a model, a reporter that cannot
render its confirmation message must not turn a successful save into a failure.
The pre-remediation code was ``except Exception: pass`` around the
``reporter.success(...)`` call — a blanket swallow that hid *any* reporter
defect, including the framework's own typed ``FrameworkError`` subclasses that a
headless caller (the CLI, the tests) legitimately raises.

The corrected contract, asserted here in both directions:

* a :class:`FrameworkError` from the reporter is **logged and swallowed** — the
  save or load already succeeded, so the confirmation is best-effort; and
* any other exception is a genuine defect and **propagates**, so a broken
  reporter is not silently discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.errors import FrameworkError
from core.model_io import ArtifactRecord
from modules.model_module import ModelTrainer


class _RecordingReporter:
    """A reporter double that records each call and can raise on demand."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc
        self.calls: list[str] = []

    def success(self, message: str) -> None:
        self.calls.append(message)
        if self._exc is not None:
            raise self._exc


def _record(name: str = "m.pkl") -> ArtifactRecord:
    """A fabricated provenance record; the store is stubbed, so nothing is on disk."""
    return ArtifactRecord(
        name=name,
        path=Path("/tmp") / name,
        sha256="0" * 64,
        size_bytes=1,
        created_at="2026-10-10T00:00:00Z",
        model_name="Logistic Regression",
        library_versions={},
    )


def _trainer(reporter: _RecordingReporter):
    """A trainer whose store is stubbed so no real file is written or read."""
    trainer = ModelTrainer(reporter=reporter)
    trainer.trained_models["Logistic Regression"] = object()
    trainer._store.save = lambda *args, **kwargs: _record()  # type: ignore[method-assign]
    trainer._store.load = lambda *args, **kwargs: (object(), _record())  # type: ignore[method-assign]
    return trainer


def test_save_logs_a_framework_error_from_the_reporter(caplog: pytest.LogCaptureFixture):
    """A typed confirmation failure is logged, and the save still returns."""
    reporter = _RecordingReporter(FrameworkError("no Streamlit runtime"))
    trainer = _trainer(reporter)

    with caplog.at_level(logging.WARNING, logger="modules.model_module"):
        record = trainer.save_model("Logistic Regression", "m.pkl")

    assert record.name == "m.pkl"
    assert reporter.calls == ["✅ Model saved to m.pkl"]
    assert any("Could not report the save" in r.message for r in caplog.records), (
        "the swallowed confirmation was not logged"
    )


def test_load_logs_a_framework_error_from_the_reporter(caplog: pytest.LogCaptureFixture):
    """The load path has the same contract as the save path."""
    reporter = _RecordingReporter(FrameworkError("no Streamlit runtime"))
    trainer = _trainer(reporter)

    with caplog.at_level(logging.WARNING, logger="modules.model_module"):
        record = trainer.load_model("Logistic Regression", "m.pkl")

    assert record.name == "m.pkl"
    assert reporter.calls == ["✅ Model loaded from m.pkl"]
    assert any("Could not report the load" in r.message for r in caplog.records)


def test_save_surfaces_a_non_framework_reporter_defect():
    """A plain ``RuntimeError`` is a bug, not a best-effort confirmation."""
    reporter = _RecordingReporter(RuntimeError("reporter bug"))
    trainer = _trainer(reporter)
    with pytest.raises(RuntimeError):
        trainer.save_model("Logistic Regression", "m.pkl")


def test_load_surfaces_a_non_framework_reporter_defect():
    """As above, on the load path: the blanket swallow is gone."""
    reporter = _RecordingReporter(RuntimeError("reporter bug"))
    trainer = _trainer(reporter)
    with pytest.raises(RuntimeError):
        trainer.load_model("Logistic Regression", "m.pkl")
