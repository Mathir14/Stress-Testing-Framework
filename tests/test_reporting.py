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

import json
from typing import Any

import pytest

pytest.importorskip("numpy", reason="the report generator pulls the scientific stack")
pytest.importorskip("pandas", reason="the report generator pulls the scientific stack")
pytest.importorskip("plotly", reason="the report generator pulls the scientific stack")

from core.errors import ReportExportError
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


# ── export_to_json: non-finite floats and circular references ────────────────
#
# A second boundary lives in this module.  ``export_to_json`` used a ``default``
# hook that converted numpy scalars but left non-finite floats alone, so a report
# that carried a ``nan`` (an empty-bin ECE, say) serialised to the non-standard
# tokens ``NaN``/``Infinity`` that no strict JSON parser accepts.  And it caught
# only ``TypeError``, so a self-referential report raised the raw ``ValueError``
# from ``json`` rather than the module's typed ``ReportExportError`` (run-014
# M11, ADR-025).  These tests pin the corrected contract.


def test_non_finite_floats_export_as_json_null() -> None:
    """``nan``/``±inf`` are not JSON; they must become ``null``, not raw tokens."""
    payload = ReportGenerator().export_to_json(
        {"ece": float("nan"), "up": float("inf"), "down": float("-inf")}
    )
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert json.loads(payload) == {"ece": None, "up": None, "down": None}


def test_finite_floats_are_unchanged_by_the_safety_pass() -> None:
    """The non-finite rewrite must not touch an ordinary float."""
    payload = ReportGenerator().export_to_json({"ece": 0.25, "acc": -1.5})
    assert json.loads(payload) == {"ece": 0.25, "acc": -1.5}


def test_numpy_scalars_and_arrays_export_as_python_values() -> None:
    """The old ``default`` hook did this; the rewrite must keep doing it."""
    numpy = pytest.importorskip("numpy")
    payload = ReportGenerator().export_to_json(
        {
            "i": numpy.int64(3),
            "f": numpy.float64(0.5),
            "b": numpy.bool_(True),
            "a": numpy.array([1, 2, 3]),
        }
    )
    assert json.loads(payload) == {"i": 3, "f": 0.5, "b": True, "a": [1, 2, 3]}


def test_a_circular_report_is_a_typed_export_error() -> None:
    """A self-referential report must be ``ReportExportError``, not ``ValueError``."""
    report: dict[str, Any] = {}
    report["self"] = report
    with pytest.raises(ReportExportError) as excinfo:
        ReportGenerator().export_to_json(report)
    assert excinfo.value.context["format"] == "json"


def test_non_finite_floats_nested_in_lists_are_also_nulled() -> None:
    """The rewrite walks containers, so a nested ``nan`` cannot slip through."""
    payload = ReportGenerator().export_to_json({"rows": [{"drop": float("nan")}]})
    assert json.loads(payload) == {"rows": [{"drop": None}]}
