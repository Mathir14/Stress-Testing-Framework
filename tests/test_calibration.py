"""Calibration-module tests for ADR-017 label alignment and the single Brier.

**Tier T1** — requires numpy/pandas.

The binding decisions under test:

* ``classes=None`` means **strict identity**: labels must be exactly ``0..K-1``,
  and any other label space raises a typed :class:`ValidationError` instead of
  silently mis-scoring (ADR-017, critic C1).  A call site that forgets to thread
  the estimator's ``classes_`` fails loudly.
* :func:`multiclass_brier` is the framework's **single** Brier implementation —
  the former ``utils.calculate_brier_score`` was removed so the two copies could
  not disagree about label guarding (ADR-017, critic M23).  It guards the same
  label alignment *and* refuses an index that overflows the probability columns.
* Every :class:`CalibrationAnalyzer` entry point threads ``classes`` through to
  ``align_target_indices``, so a model trained on ``["cat", "dog"]`` labels and
  an estimator whose columns are class indices agree on what column means what.
* ECE-quality bands are owned by :class:`core.config.ScoringPolicy` (ADR-018);
  ``get_calibration_quality`` delegates rather than restating the bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import ScoringPolicy, get_config
from core.errors import ValidationError
from modules.calibration_module import CalibrationAnalyzer, align_target_indices, multiclass_brier


@pytest.fixture(scope="module")
def analyzer() -> CalibrationAnalyzer:
    return CalibrationAnalyzer()


# ── align_target_indices: strict identity when classes=None ──────────────────


def test_strict_identity_accepts_binary_labels():
    """Labels already in ``0..K-1`` pass through unchanged."""
    out = align_target_indices(np.array([0, 1, 0, 1]), None)
    assert list(out) == [0, 1, 0, 1]


def test_strict_identity_accepts_multiclass_contiguous_labels():
    """Contiguous ``0..K-1`` in any row order is still valid."""
    out = align_target_indices(np.array([0, 2, 1, 2, 0]), None)
    assert list(out) == [0, 2, 1, 2, 0]


def test_non_contiguous_labels_are_rejected_without_classes():
    """``[0, 2]`` with no ``classes`` is a gap the estimator cannot explain."""
    with pytest.raises(ValidationError) as excinfo:
        align_target_indices(np.array([0, 2, 2]), None)
    assert "0..K-1" in str(excinfo.value)
    assert "labels" in excinfo.value.context


def test_string_or_offset_labels_are_rejected_without_classes():
    """Raw ``"cat"/"dog"`` labels only make sense relative to ``classes_``."""
    for bad in (np.array(["cat", "dog"]), np.array([5, 6]), np.array([-1, 0])):
        with pytest.raises(ValidationError):
            align_target_indices(bad, None)


def test_empty_targets_are_valid_under_strict_identity():
    """No labels to mis-align: the empty array must not raise."""
    out = align_target_indices(np.array([], dtype=int), None)
    assert out.shape == (0,)


# ── align_target_indices: classes mapping ────────────────────────────────────


def test_classes_map_labels_to_columns():
    out = align_target_indices(np.array(["cat", "dog", "cat"]), ["cat", "dog"])
    assert list(out) == [0, 1, 0]


def test_classes_accept_numeric_label_spaces_that_do_not_start_at_zero():
    """Binary classes ``[5, 9]`` map to columns ``[0, 1]``, not to ``5``."""
    out = align_target_indices(np.array([9, 5, 5]), [5, 9])
    assert list(out) == [1, 0, 0]


def test_classes_accept_numpy_scalar_labels():
    """``np.str_`` labels must hash like their Python equivalents (ADR-017)."""
    classes = np.array(["low", "high"])
    out = align_target_indices(np.array(["high", "low"]), classes)
    assert list(out) == [1, 0]


def test_a_label_absent_from_classes_is_a_typed_error():
    """A label from a different encoding must not be silently dropped."""
    with pytest.raises(ValidationError) as excinfo:
        align_target_indices(np.array(["cat", "bird"]), ["cat", "dog"])
    assert excinfo.value.context["label"] == "bird"
    assert "classes" in excinfo.value.context


# ── multiclass_brier ─────────────────────────────────────────────────────────


def test_brier_is_zero_for_perfect_predictions():
    one_hot = np.eye(2)[np.array([0, 1, 0, 1])]
    assert multiclass_brier(np.array([0, 1, 0, 1]), one_hot) == 0.0


def test_brier_is_half_for_uniform_guessing():
    """Two classes, 0.5/0.5 every row: squared error is 0.5 per row."""
    probs = np.full((4, 2), 0.5)
    assert multiclass_brier(np.array([0, 1, 0, 1]), probs) == 0.5


def test_brier_matches_a_hand_computation():
    """Two rows, computed by hand outside the module."""
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y_true = np.array([0, 1])
    assert multiclass_brier(y_true, probs) == pytest.approx(0.13)


def test_brier_rejects_labels_that_overflow_the_columns():
    """Strict identity passes, but the mapping exceeds the column count."""
    with pytest.raises(ValidationError) as excinfo:
        multiclass_brier(np.array([0, 1, 2]), np.eye(3)[:, :2])
    assert excinfo.value.context["n_classes"] == 2


def test_brier_rejects_string_labels_without_classes():
    with pytest.raises(ValidationError):
        multiclass_brier(np.array(["a", "b"]), np.eye(2))


def test_brier_uses_classes_to_align():
    classes = ["cat", "dog"]
    assert multiclass_brier(np.array(["cat", "dog"]), np.eye(2), classes=classes) == 0.0


def test_brier_requires_a_2d_probability_matrix():
    with pytest.raises(ValidationError) as excinfo:
        multiclass_brier(np.array([0, 1]), np.array([0.8, 0.2]))
    assert excinfo.value.context["shape"] == [2]


# ── compute_calibration_metrics ──────────────────────────────────────────────


def test_calibration_metrics_match_a_hand_computation(analyzer: CalibrationAnalyzer):
    """ECE / MCE / Brier / overconfidence, derived independently.

    ``y_true=[0,1]``, ``probs=[[1,0],[0.8,0.2]]``: the second row predicts
    class 0 with confidence 0.8 but is wrong, so the binned ECE is
    ``0.5 * |0.0 - 0.8| = 0.4`` and the per-class MCE is ``0.8``.
    """
    metrics = analyzer.compute_calibration_metrics(
        np.array([0, 1]), np.array([[1.0, 0.0], [0.8, 0.2]])
    )
    assert metrics["ece"] == pytest.approx(0.4, abs=1e-9)
    assert metrics["mce"] == pytest.approx(0.8, abs=1e-9)
    assert metrics["brier_score"] == pytest.approx(0.64, abs=1e-9)
    assert metrics["avg_confidence"] == pytest.approx(0.9)
    assert metrics["avg_accuracy"] == pytest.approx(0.5)
    assert metrics["overconfidence"] == pytest.approx(0.4)


def test_calibration_metrics_are_perfect_for_identity_predictions(
    analyzer: CalibrationAnalyzer,
):
    one_hot = np.eye(2)[np.array([0, 1, 0, 1])]
    metrics = analyzer.compute_calibration_metrics(np.array([0, 1, 0, 1]), one_hot)
    assert metrics["ece"] == 0.0
    assert metrics["mce"] == 0.0
    assert metrics["brier_score"] == 0.0
    assert metrics["overconfidence"] == 0.0


def test_calibration_metrics_thread_classes(analyzer: CalibrationAnalyzer):
    """``classes=["cat","dog"]`` scores the same as strict-identity indices."""
    classes = ["cat", "dog"]
    labels = np.array(["cat", "dog", "cat"])
    one_hot = np.eye(2)[[0, 1, 0]]
    mapped = analyzer.compute_calibration_metrics(labels, one_hot, classes=classes)
    strict = analyzer.compute_calibration_metrics(np.array([0, 1, 0]), one_hot)
    assert mapped["ece"] == strict["ece"] == 0.0
    assert mapped["brier_score"] == strict["brier_score"] == 0.0
    # The Brier inside the analyzer is the one true implementation (ADR-017).
    assert mapped["brier_score"] == multiclass_brier(labels, one_hot, classes=classes)


def test_calibration_metrics_reject_raw_strings_without_classes(
    analyzer: CalibrationAnalyzer,
):
    with pytest.raises(ValidationError):
        analyzer.compute_calibration_metrics(np.array(["cat", "dog"]), np.eye(2))


# ── compute_per_class_calibration ────────────────────────────────────────────


def test_per_class_frame_uses_classes_for_names_and_columns(
    analyzer: CalibrationAnalyzer,
):
    classes = ["cat", "dog"]
    frame = analyzer.compute_per_class_calibration(
        np.array(["cat", "dog", "cat"]),
        np.eye(2)[[0, 1, 0]],
        classes=classes,
        class_names=classes,
    )
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["Class"]) == classes
    assert list(frame["Brier Score"]) == [0.0, 0.0]
    assert list(frame["ECE"]) == [0.0, 0.0]


def test_per_class_rejects_unmapped_labels_without_classes(analyzer: CalibrationAnalyzer):
    with pytest.raises(ValidationError):
        analyzer.compute_per_class_calibration(np.array(["cat", "dog"]), np.eye(2))


# ── plot_confidence_histogram ────────────────────────────────────────────────


def test_confidence_histogram_threads_classes(analyzer: CalibrationAnalyzer):
    classes = ["cat", "dog"]
    fig = analyzer.plot_confidence_histogram(
        np.array(["cat", "dog", "cat"]), np.eye(2)[[0, 1, 0]], classes=classes
    )
    assert len(fig.data) == 2  # Correct and Incorrect
    with pytest.raises(ValidationError):
        analyzer.plot_confidence_histogram(np.array(["cat", "dog"]), np.eye(2))


# ── find_optimal_temperature / apply_temperature_scaling ─────────────────────


def test_temperature_scaling_preserves_row_sums(analyzer: CalibrationAnalyzer):
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.55, 0.45]])
    out = analyzer.apply_temperature_scaling(probs, 2.0)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(np.isfinite(out))
    # Temperature 1.0 is the identity for a valid probability distribution.
    identity = analyzer.apply_temperature_scaling(probs, 1.0)
    assert np.allclose(identity, probs)


def test_an_overconfident_model_is_softened(analyzer: CalibrationAnalyzer):
    """A confidently-wrong model improves with temperature > 1 (ADR-017).

    Twelve rows all predicting class 0 with confidence 0.99 while half the
    labels are 1: ECE at t=1 is high, and softening monotonically reduces it
    within the grid, so the optimal temperature must be strictly above 1.0.
    """
    y_true = np.ones(12, dtype=int)
    probs = np.repeat([[0.99, 0.01]], 12, axis=0)

    ece_at_one = analyzer.compute_calibration_metrics(y_true, probs, classes=[0, 1])["ece"]
    assert ece_at_one > 0.9

    optimal = analyzer.find_optimal_temperature(y_true, probs, classes=[0, 1])
    assert optimal > 1.0
    assert optimal <= 5.0

    ece_at_optimal = analyzer.compute_calibration_metrics(
        y_true,
        analyzer.apply_temperature_scaling(probs, optimal),
        classes=[0, 1],
    )["ece"]
    assert ece_at_optimal < ece_at_one


# ── get_calibration_quality: ScoringPolicy owns the bands ────────────────────


def test_calibration_quality_uses_the_bound_policy(analyzer: CalibrationAnalyzer):
    assert analyzer.get_calibration_quality(0.029) == "Excellent"
    assert analyzer.get_calibration_quality(0.03) == "Good"  # strict upper bound
    assert analyzer.get_calibration_quality(0.07) == "Moderate"
    assert analyzer.get_calibration_quality(0.15) == "Poor"  # fallthrough
    assert analyzer.get_calibration_quality(0.9) == "Poor"


def test_calibration_quality_can_be_bound_to_a_custom_policy():
    policy = ScoringPolicy(ece_quality=((0.5, "Tight"),), quality_fallthrough="Loose")
    custom = CalibrationAnalyzer(policy=policy)
    assert custom.get_calibration_quality(0.4) == "Tight"
    assert custom.get_calibration_quality(0.7) == "Loose"


def test_default_analyzer_binds_the_configuration_singleton(analyzer: CalibrationAnalyzer):
    assert analyzer._policy is get_config().scoring