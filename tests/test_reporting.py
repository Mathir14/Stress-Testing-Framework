"""Regression tests for the ``compile_report`` rated gate (ADR-018 / §9 item 14).

**Tier T1** — requires numpy/pandas via the reporting module's imports.

The run-012 Reviewer MAJOR: ``ReportGenerator.compile_report`` selected
``best_reliability`` with ``max(Total)`` over every reliability row, so a model
the scorer had marked ``rated=False`` (too few measured components) could still
be crowned best.  The first remediation attempt added
``[r for r in rel_rows if r.get("rated", True)]`` — but the flattened ``rel_rows``
entries never carried a ``rated`` key, so the filter kept every row and was a
no-op that merely *looked* like a gate.  That is the "gate passes for the wrong
reason" pattern the project's ADRs warn against.

These tests pin the honest behaviour: the flag is read from the source
``reliability_scores`` entry, an unrated model can never win, and a tree with no
rated model reports ``"N/A"`` rather than falling back to ranking the unrated.
"""

from __future__ import annotations

from typing import Any

from modules.reporting_module import ReportGenerator


class _FakeTrainer:
    """Minimal stand-in carrying only what ``compile_report`` reads."""

    def __init__(self, models: list[str]) -> None:
        self.trained_models = {m: object() for m in models}
        self.metrics: dict[str, Any] = {}


def _score(total: float, *, rated: bool) -> dict[str, Any]:
    """A reliability score entry shaped like ``ReliabilityScorer.score_model``."""
    return {
        "total": total,
        "grade": "D",
        "performance": round(total, 2),
        "calibration": 0.0,
        "robustness": 0.0,
        "confidence": 0.0,
        "rated": rated,
    }


def test_unrated_model_is_not_crowned_best_reliability() -> None:
    """An unrated total that outranks a rated total must not win (run-012 MAJOR).

    Uses the exact figures from the Reviewer's reproduction: the unrated model
    scores ``22.5`` and the rated model ``21.25``.  Before the fix the unrated
    model was crowned; after it, the rated model is the only candidate.
    """
    report = ReportGenerator().compile_report(
        _FakeTrainer(["rated_model", "unrated_model"]),
        None,
        reliability_scores={
            "rated_model": _score(21.25, rated=True),
            "unrated_model": _score(22.5, rated=False),
        },
    )
    assert report["summary"]["best_reliability"] == "rated_model"


def test_unrated_model_cannot_be_crowned_even_when_it_is_the_only_one() -> None:
    """With no rated model the summary is ``N/A``, not an unrated winner."""
    report = ReportGenerator().compile_report(
        _FakeTrainer(["only_unrated"]),
        None,
        reliability_scores={"only_unrated": _score(22.5, rated=False)},
    )
    assert report["summary"]["best_reliability"] == "N/A"


def test_rated_models_are_still_ranked_by_total() -> None:
    """The gate must not change the ordinary path: highest rated total wins."""
    report = ReportGenerator().compile_report(
        _FakeTrainer(["low", "high"]),
        None,
        reliability_scores={
            "low": _score(30.0, rated=True),
            "high": _score(80.0, rated=True),
        },
    )
    assert report["summary"]["best_reliability"] == "high"


def test_scores_without_a_rated_flag_are_treated_as_rated() -> None:
    """Backward compatibility: legacy entries with no ``rated`` key still rank."""
    legacy = _score(80.0, rated=True)
    del legacy["rated"]
    report = ReportGenerator().compile_report(
        _FakeTrainer(["legacy"]),
        None,
        reliability_scores={"legacy": legacy},
    )
    assert report["summary"]["best_reliability"] == "legacy"
