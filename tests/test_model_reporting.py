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

pytest.importorskip("numpy", reason="ModelTrainer pulls the scientific stack")
pytest.importorskip("pandas", reason="ModelTrainer pulls the scientific stack")
pytest.importorskip("sklearn", reason="ModelTrainer pulls the scientific stack")
pytest.importorskip("xgboost", reason="ModelTrainer pulls the scientific stack")
pytest.importorskip("plotly", reason="ModelTrainer pulls the scientific stack")

import numpy as np

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


# ── ADR-035 (m-2) predict error typing ────────────────────────────────────────


def test_predict_distinguishes_untrained_from_prediction_failures():
    """ADR-035: untrained raises ModelNotTrainedError, inference failure raises ModelPredictionError."""
    import pandas as pd

    from core.errors import ModelNotTrainedError, ModelPredictionError

    trainer = ModelTrainer()

    # 1. Model not in trained_models -> ModelNotTrainedError
    with pytest.raises(ModelNotTrainedError):
        trainer.predict("Logistic Regression", pd.DataFrame([[1, 2]]))

    # 2. Estimator failing predict() -> ModelPredictionError
    class _BrokenPredictEstimator:
        def predict(self, _x):
            raise ValueError("bad shape")

        def predict_proba(self, _x):
            return [[0.5, 0.5]]

    trainer.trained_models["Broken"] = _BrokenPredictEstimator()
    with pytest.raises(ModelPredictionError, match="cannot predict"):
        trainer.predict("Broken", pd.DataFrame([[1, 2]]))

    # 3. Estimator failing predict_proba() -> ModelPredictionError
    class _BrokenProbaEstimator:
        def predict(self, _x):
            return [1]

        def predict_proba(self, _x):
            raise AttributeError("predict_proba not supported")

    trainer.trained_models["NoProba"] = _BrokenProbaEstimator()
    with pytest.raises(ModelPredictionError, match="cannot produce probabilities"):
        trainer.predict("NoProba", pd.DataFrame([[1, 2]]))


def test_get_best_model_raises_typed_errors_on_invalid_conditions():
    """get_best_model raises ModelNotTrainedError and ValidationError instead of sentinel (None, -1)."""
    from core.errors import ModelNotTrainedError, ValidationError

    trainer = ModelTrainer()

    # 1. No metrics at all
    with pytest.raises(ModelNotTrainedError, match="No trained models"):
        trainer.get_best_model("accuracy", "Test")

    # 2. Metrics present, but dataset_name not found
    trainer.metrics = {"m1": {"Validation": {"accuracy": 0.85, "f1": 0.84}}}
    with pytest.raises(
        ModelNotTrainedError, match="No evaluated metrics found for dataset 'Test'"
    ):
        trainer.get_best_model("accuracy", "Test")

    # 3. Invalid metric name
    with pytest.raises(ValidationError, match="not a valid evaluation metric"):
        trainer.get_best_model("nonexistent_metric", "Validation")

    # 4. Valid evaluation returns best model and score
    trainer.metrics["m2"] = {"Validation": {"accuracy": 0.92, "f1": 0.91}}
    best_name, best_score = trainer.get_best_model("accuracy", "Validation")
    assert best_name == "m2"
    assert best_score == 0.92


def test_get_best_model_metric_polarity_minimization_and_explicit_flag():
    """get_best_model supports minimization metrics and explicit lower_is_better flag."""
    trainer = ModelTrainer()
    trainer.metrics = {
        "m_low": {"Test": {"accuracy": 0.70, "loss": 0.20, "ece": 0.03, "custom_metric": 10.0}},
        "m_high": {"Test": {"accuracy": 0.90, "loss": 0.50, "ece": 0.15, "custom_metric": 50.0}},
    }

    # Maximization (default inferred)
    best_name, best_score = trainer.get_best_model("accuracy", "Test")
    assert best_name == "m_high"
    assert best_score == 0.90

    # Minimization: 'loss' (inferred)
    best_name, best_score = trainer.get_best_model("loss", "Test")
    assert best_name == "m_low"
    assert best_score == 0.20

    # Minimization: 'ece' (inferred)
    best_name, best_score = trainer.get_best_model("ece", "Test")
    assert best_name == "m_low"
    assert best_score == 0.03

    # Explicit lower_is_better=True on custom metric
    best_name, best_score = trainer.get_best_model("custom_metric", "Test", lower_is_better=True)
    assert best_name == "m_low"
    assert best_score == 10.0

    # Explicit lower_is_better=False on loss metric overrides inference
    best_name, best_score = trainer.get_best_model("loss", "Test", lower_is_better=False)
    assert best_name == "m_high"
    assert best_score == 0.50


def test_model_trainer_capability_introspection():
    """supports_probabilities, get_classes, get_class_names, supports_feature_importance work as expected."""
    from core.errors import ModelNotTrainedError

    trainer = ModelTrainer()

    # Querying untrained models raises ModelNotTrainedError
    for meth in ("supports_probabilities", "get_classes", "get_class_names", "supports_feature_importance"):
        with pytest.raises(ModelNotTrainedError):
            getattr(trainer, meth)("nonexistent_model")

    class _DummyTreeEstimator:
        classes_ = np.array([0, 1])
        feature_importances_ = np.array([0.4, 0.6])

        def predict_proba(self, _x):
            return np.array([[0.5, 0.5]])

    class _DummyLinearEstimator:
        classes_ = np.array(["cat", "dog"])

    trainer.trained_models["tree"] = _DummyTreeEstimator()
    trainer.trained_models["linear"] = _DummyLinearEstimator()

    assert trainer.supports_probabilities("tree") is True
    assert trainer.supports_probabilities("linear") is False

    np.testing.assert_array_equal(trainer.get_classes("tree"), np.array([0, 1]))
    assert trainer.get_class_names("tree") == ["0", "1"]
    assert trainer.get_class_names("linear") == ["cat", "dog"]

    assert trainer.supports_feature_importance("tree") is True
    assert trainer.supports_feature_importance("linear") is False


def test_model_trainer_query_methods_raise_typed_errors_not_none():
    """Visualization and query methods raise typed FrameworkError instead of returning None sentinels."""
    from core.errors import ModelNotTrainedError, ValidationError

    trainer = ModelTrainer()

    # Untrained model raises ModelNotTrainedError
    with pytest.raises(ModelNotTrainedError):
        trainer.get_classification_report("untrained")

    with pytest.raises(ModelNotTrainedError):
        trainer.plot_confusion_matrix("untrained")

    with pytest.raises(ModelNotTrainedError):
        trainer.plot_metrics_comparison("Test")

    with pytest.raises(ModelNotTrainedError):
        trainer.get_feature_importance("untrained", ["f1"])

    with pytest.raises(ModelNotTrainedError):
        trainer.plot_feature_importance("untrained", ["f1"])

    with pytest.raises(ModelNotTrainedError):
        trainer.get_probability_distribution("untrained")

    # Model trained without feature_importances_
    class _DummyNoFI:
        pass

    trainer.trained_models["no_fi"] = _DummyNoFI()
    with pytest.raises(ValidationError, match="does not support feature importances"):
        trainer.get_feature_importance("no_fi", ["f1"])

    # Feature names length mismatch
    class _DummyFI:
        feature_importances_ = np.array([0.5, 0.5])

    trainer.trained_models["fi"] = _DummyFI()
    with pytest.raises(ValidationError, match="Length of feature_names"):
        trainer.get_feature_importance("fi", ["f1"])

    with pytest.raises(ValidationError, match="top_n must be a positive integer"):
        trainer.plot_feature_importance("fi", ["f1", "f2"], top_n=0)
