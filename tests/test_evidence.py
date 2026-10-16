"""Contract tests for ``modules/evidence_module.py`` (run-015 R6, ADR-021).

**Tier T1** — numpy (importorskip-guarded, architecture.md §10.1 / ADR-026).

This module is the single owner of "what did we measure?" for Modules 7, 8 and
9.  Before run-015 each page re-derived the same ECE / entropy / high-confidence
error / robustness figures, so a fix in one page silently missed the other two
(run-014 M2/M3).  These tests pin the properties the pages now depend on:

* :func:`collect_evidence` performs **one** ``predict`` per model and turns a
  per-model failure into *missing* evidence rather than an abort — a broken
  model must be distinguishable from a model with no data;
* a field is ``None`` exactly when it could not be measured, never ``0.0``
  standing in for "missing" (ADR-018/ADR-021);
* :func:`robustness_scores` reuses the shared ``robustness_fraction`` with the
  configured ``drop_cap``, so it cannot disagree with the reliability scorer.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from modules.evidence_module import (  # noqa: E402 - after importorskip by design
    ModelEvidence,
    calibration_ece,
    class_count,
    collect_evidence,
    hce_rates,
    mean_entropies,
    robustness_scores,
)


class _Model:
    """Minimal stand-in exposing only ``classes_`` (what the view reads)."""

    def __init__(self, n_classes: int = 2) -> None:
        self.classes_ = np.arange(n_classes)


class _FakeTrainer:
    """A ``ModelTrainer``-like double recording every ``predict`` call."""

    def __init__(
        self,
        trained: dict,
        metrics: dict,
        predictions: dict,
    ) -> None:
        self.trained_models = trained
        self.metrics = metrics
        self._predictions = predictions
        self.predict_calls: list[str] = []

    def predict(self, model_name: str, _x: object) -> tuple:
        self.predict_calls.append(model_name)
        result = self._predictions[model_name]
        if isinstance(result, Exception):
            raise result
        return result


class _FakeCalibration:
    """Returns a fixed ECE, or raises, so the caller's handling is observable."""

    def __init__(self, ece: float = 0.1, error: Exception | None = None) -> None:
        self._ece = ece
        self._error = error

    def compute_calibration_metrics(self, _y, _probs, classes=None) -> dict:
        if self._error is not None:
            raise self._error
        return {"ece": self._ece}


def _binary_metrics(accuracy: float = 0.9, f1: float = 0.8) -> dict:
    return {"m": {"Test": {"accuracy": accuracy, "f1": f1}}}


def test_collect_evidence_gathers_accuracy_f1_entropy_and_hce() -> None:
    """A good model yields accuracy/F1 plus measured confidence figures."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(0.75, 0.5),
        predictions={
            "m": (
                np.array([0, 1]),
                np.array([[0.5, 0.5], [0.5, 0.5]]),
            )
        },
    )
    evidence = collect_evidence(
        trainer,
        X_test=np.zeros((2, 3)),
        y_test=np.array([0, 1]),
        calibration_analyzer=_FakeCalibration(0.2),
    )
    item = evidence["m"]
    assert item.accuracy == pytest.approx(0.75)
    assert item.f1 == pytest.approx(0.5)
    assert item.ece == pytest.approx(0.2)
    assert item.avg_entropy == pytest.approx(1.0)  # uniform binary => 1 bit
    # preds are correct, so no high-confidence errors
    assert item.hce_rate == pytest.approx(0.0)


def test_collect_evidence_predicts_at_most_once_per_model() -> None:
    """One predict per model feeds ECE *and* confidence (M2: no double work)."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={"m": (np.array([0]), np.array([[0.9, 0.1]]))},
    )
    collect_evidence(
        trainer,
        X_test=np.zeros((1, 2)),
        y_test=np.array([0]),
        calibration_analyzer=_FakeCalibration(),
    )
    assert trainer.predict_calls == ["m"]


def test_collect_evidence_without_test_data_leaves_confidence_missing() -> None:
    """No test data => ECE/entropy/HCE are ``None``, never fabricated zeros."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={},
    )
    evidence = collect_evidence(
        trainer,
        X_test=None,
        y_test=None,
        calibration_analyzer=_FakeCalibration(),
    )
    item = evidence["m"]
    assert item.ece is None
    assert item.avg_entropy is None
    assert item.hce_rate is None
    assert trainer.predict_calls == []


def test_collect_evidence_isolates_a_broken_model() -> None:
    """An unpredicable model is missing evidence; a sibling is unaffected."""
    trainer = _FakeTrainer(
        trained={"broken": _Model(), "ok": _Model()},
        metrics={
            "broken": {"Test": {"accuracy": 0.5, "f1": 0.5}},
            "ok": {"Test": {"accuracy": 0.9, "f1": 0.9}},
        },
        predictions={
            "broken": RuntimeError("boom"),
            "ok": (np.array([0]), np.array([[0.9, 0.1]])),
        },
    )
    evidence = collect_evidence(
        trainer,
        X_test=np.zeros((1, 2)),
        y_test=np.array([0]),
        calibration_analyzer=_FakeCalibration(),
    )
    assert evidence["broken"].avg_entropy is None
    assert evidence["broken"].hce_rate is None
    assert evidence["broken"].accuracy == pytest.approx(0.5)
    assert evidence["ok"].avg_entropy is not None


def test_collect_evidence_calibration_failure_is_missing_not_zero() -> None:
    """A raising calibration analyzer leaves ECE ``None`` (missing) not ``0.0``."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={"m": (np.array([0]), np.array([[0.9, 0.1]]))},
    )
    evidence = collect_evidence(
        trainer,
        X_test=np.zeros((1, 2)),
        y_test=np.array([0]),
        calibration_analyzer=_FakeCalibration(error=ValueError("bad")),
    )
    assert evidence["m"].ece is None
    # entropy/HCE still measured because they share the predict, not the ECE call
    assert evidence["m"].avg_entropy is not None


def test_collect_evidence_avg_drop_is_mean_of_numeric_entries() -> None:
    """Robustness reads the mean ``performance_drop`` across stress types."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={},
    )
    evidence = collect_evidence(
        trainer,
        X_test=None,
        y_test=None,
        stress_results={
            "m": {
                "noise": {"performance_drop": 0.1},
                "shift": {"performance_drop": 0.3},
            }
        },
    )
    assert evidence["m"].avg_drop == pytest.approx(0.2)


def test_collect_evidence_empty_stress_evidence_is_none() -> None:
    """No numeric drop entries => ``avg_drop`` is ``None`` (missing, not zero)."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={},
    )
    evidence = collect_evidence(
        trainer,
        X_test=None,
        y_test=None,
        stress_results={"m": {"noise": {"note": "no drop recorded"}}},
    )
    assert evidence["m"].avg_drop is None


def test_collect_evidence_derives_and_overrides_class_count() -> None:
    """The class count comes from ``y_test`` unless explicitly supplied."""
    trainer = _FakeTrainer(
        trained={"m": _Model()},
        metrics=_binary_metrics(),
        predictions={},
    )
    derived = collect_evidence(trainer, X_test=None, y_test=np.array([0, 1, 2]))
    assert derived["m"].n_classes == 3
    overridden = collect_evidence(
        trainer, X_test=None, y_test=np.array([0, 1, 2]), n_classes=5
    )
    assert overridden["m"].n_classes == 5


def test_accessor_helpers_drop_missing_figures() -> None:
    """The ``*_rates``/``*_ece`` accessors omit models with no measurement."""
    evidence = {
        "measured": ModelEvidence("measured", 0.9, 0.9, 0.1, 0.2, 0.5, 0.05, 2),
        "blank": ModelEvidence("blank", 0.5, 0.5, None, None, None, None, 2),
    }
    assert calibration_ece(evidence) == {"measured": 0.1}
    assert mean_entropies(evidence) == {"measured": 0.5}
    assert hce_rates(evidence) == {"measured": 0.05}
    assert class_count(evidence) == 2


def test_class_count_falls_back_when_empty() -> None:
    assert class_count({}, fallback=4) == 4


def test_class_count_warns_and_falls_back_on_discrepancy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When models in evidence report conflicting class counts, log a warning and fall back."""
    import logging

    evidence = {
        "binary": ModelEvidence("binary", 0.9, 0.9, 0.1, 0.2, 0.5, 0.05, 2),
        "multiclass": ModelEvidence("multiclass", 0.8, 0.8, 0.1, 0.2, 0.5, 0.05, 3),
    }
    with caplog.at_level(logging.WARNING):
        result = class_count(evidence, fallback=5)
    assert result == 5
    assert "Discrepancy detected in model evidence class counts" in caplog.text


def test_robustness_scores_uses_shared_transform_and_cap() -> None:
    """``robustness_scores`` is the shared 0-100 transform, missing dropped."""
    evidence = {
        "zero_drop": ModelEvidence("zero_drop", 1.0, 1.0, 0.0, 0.0, None, None, 2),
        "half_drop": ModelEvidence("half_drop", 1.0, 1.0, 0.0, 0.25, None, None, 2),
        "no_drop": ModelEvidence("no_drop", 1.0, 1.0, 0.0, None, None, None, 2),
    }
    scores = robustness_scores(evidence, drop_cap=0.5)
    assert scores == {"zero_drop": 100.0, "half_drop": 50.0}
    assert "no_drop" not in scores


def test_robustness_score_clamps_a_negative_drop_to_a_perfect_score() -> None:
    """A modelled negative drop must not inflate past 100 (run-015 C1)."""
    item = ModelEvidence("m", 1.0, 1.0, 0.0, -0.3, None, None, 2)
    assert item.robustness_score(drop_cap=0.5) == 100.0


def test_robustness_score_is_none_when_drop_unmeasured() -> None:
    item = ModelEvidence("m", 1.0, 1.0, 0.0, None, None, None, 2)
    assert item.robustness_score(drop_cap=0.5) is None


# ── ADR-035 (m-6) distinguishing error logging ────────────────────────────────


def test_predict_logging_distinguishes_causes(caplog: pytest.LogCaptureFixture) -> None:
    """ADR-035: _predict logs cause-distinguishing messages for failures."""
    import logging

    from core.errors import ModelNotTrainedError, ModelPredictionError
    from modules.evidence_module import _predict

    class _TrainerStub:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def predict(self, _name: str, _x: object) -> tuple:
            raise self.exc

    # 1. Benign untrained model -> INFO log
    with caplog.at_level(logging.INFO):
        caplog.clear()
        res = _predict(_TrainerStub(ModelNotTrainedError("not trained")), "untrained", None)
        assert res is None
        assert any(r.levelno == logging.INFO and "not been trained yet" in r.message for r in caplog.records)

    # 2. Fitted estimator prediction error -> WARNING log
    with caplog.at_level(logging.INFO):
        caplog.clear()
        res = _predict(_TrainerStub(ModelPredictionError("bad shape")), "broken", None)
        assert res is None
        assert any(r.levelno == logging.WARNING and "Prediction failed" in r.message for r in caplog.records)

    # 3. Unexpected exception -> ERROR log (logger.exception)
    with caplog.at_level(logging.INFO):
        caplog.clear()
        res = _predict(_TrainerStub(RuntimeError("crash")), "crashed", None)
        assert res is None
        assert any(r.levelno == logging.ERROR and "Unexpected error while predicting" in r.message for r in caplog.records)
