"""Prediction metrics: entropy in bits and the high-confidence-error contract.

**Tier T1** — numpy.

Two bindings from ADR-017/ADR-018 are pinned here:

* :func:`utils.metrics.get_prediction_entropy` returns Shannon entropy in
  **bits** (``log2``), because the reliability scorer normalises it by
  ``log2(n_classes)`` bits.  Nats would inflate the confidence component.
* :func:`utils.metrics.identify_high_confidence_errors` returns a *mapping* whose
  error count is ``["count"]``.  The C2 defect read ``len(hce_dict)`` — the
  number of keys (4) — as the error count, so the HCE rate in two views was
  wrong for every batch size.  The regression is stated explicitly below.

The suite also pins the removals ADR-017 mandates: ``utils.calculate_brier_score``
and ``utils.get_confidence_bins`` are gone, so ``multiclass_brier`` is the only
Brier owner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils import metrics
from utils.metrics import get_prediction_entropy, identify_high_confidence_errors

# ── Entropy units ─────────────────────────────────────────────────────────────


def test_entropy_of_a_uniform_binary_distribution_is_one_bit():
    assert get_prediction_entropy(np.array([[0.5, 0.5]]))[0] == pytest.approx(1.0)


def test_entropy_of_a_uniform_four_class_distribution_is_two_bits():
    assert get_prediction_entropy(np.array([[0.25, 0.25, 0.25, 0.25]]))[0] == (
        pytest.approx(2.0)
    )


def test_entropy_is_measured_in_bits_not_nats():
    """The uniform-binary entropy is ``1`` bit, not ``ln 2 ≈ 0.693`` nats."""
    [value] = get_prediction_entropy(np.array([[0.5, 0.5]]))
    assert value == pytest.approx(1.0)
    assert value != pytest.approx(math.log(2))


def test_entropy_of_a_certain_prediction_is_near_zero():
    [value] = get_prediction_entropy(np.array([[1.0, 0.0]]))
    assert value == pytest.approx(0.0, abs=1e-6)


def test_entropy_is_never_negative_for_clipped_probabilities():
    values = get_prediction_entropy(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert (values >= 0.0).all()


def test_entropy_is_maximal_at_the_uniform_distribution():
    uniform = get_prediction_entropy(np.array([[1 / 3, 1 / 3, 1 / 3]]))[0]
    skewed = get_prediction_entropy(np.array([[0.8, 0.1, 0.1]]))[0]
    assert uniform == pytest.approx(math.log2(3))
    assert uniform > skewed


# ── High-confidence errors ────────────────────────────────────────────────────


@pytest.fixture
def hce_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two errors, only one of them above the confidence threshold."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    return y_true, y_pred, probabilities


def test_hce_count_is_not_the_number_of_keys(hce_batch):
    """C2 regression: ``len(result)`` is 4 (the keys), not the error count.

    The two views computed ``len(hce_dict) / n`` and so reported a rate near 1.0
    regardless of the model.  The count of high-confidence errors here is 1.
    """
    y_true, y_pred, probabilities = hce_batch
    info = identify_high_confidence_errors(y_true, y_pred, probabilities)
    assert info["count"] == 1
    assert len(info) == 4
    assert info["count"] != len(info)


def test_hce_count_and_percentage_are_consistent(hce_batch):
    y_true, y_pred, probabilities = hce_batch
    info = identify_high_confidence_errors(y_true, y_pred, probabilities)
    assert info["percentage"] == pytest.approx(25.0)


def test_hce_indices_point_at_the_offending_rows(hce_batch):
    y_true, y_pred, probabilities = hce_batch
    info = identify_high_confidence_errors(y_true, y_pred, probabilities)
    assert info["indices"].tolist() == [0]


def test_hce_avg_confidence_covers_only_the_offending_rows(hce_batch):
    y_true, y_pred, probabilities = hce_batch
    info = identify_high_confidence_errors(y_true, y_pred, probabilities)
    assert info["avg_confidence"] == pytest.approx(0.9)


def test_hce_returns_zero_confidence_when_nothing_is_flagged():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    info = identify_high_confidence_errors(y_true, y_pred, probabilities)
    assert info["count"] == 0
    assert info["indices"].tolist() == []
    assert info["avg_confidence"] == 0.0


def test_hce_threshold_is_inclusive(hce_batch):
    """``confidence >= threshold`` — the boundary is a high-confidence error."""
    y_true, y_pred, probabilities = hce_batch
    at_boundary = identify_high_confidence_errors(
        y_true, y_pred, probabilities, threshold=0.9
    )
    assert at_boundary["count"] == 1


# ── Removed symbols (ADR-017) ─────────────────────────────────────────────────


def test_utils_is_no_longer_a_second_brier_owner():
    """A second Brier implementation was removed so the two could not diverge."""
    assert not hasattr(metrics, "calculate_brier_score")


def test_confidence_bin_helper_is_gone():
    assert not hasattr(metrics, "get_confidence_bins")
