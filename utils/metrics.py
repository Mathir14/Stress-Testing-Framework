"""
Metrics Utilities
Helper functions for calculating various metrics
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class HighConfidenceErrorInfo(TypedDict):
    """Structured result of :func:`identify_high_confidence_errors`.

    Attributes:
        indices: Positions of the high-confidence misclassifications.
        count: Number of high-confidence misclassifications.
        percentage: ``count`` as a percentage of the batch.
        avg_confidence: Mean confidence over the offending rows, or ``0.0``.
    """

    indices: np.ndarray
    count: int
    percentage: float
    avg_confidence: float


def get_confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Get confidence scores (max probability for each prediction)

    Args:
        probabilities: Array of prediction probabilities (n_samples, n_classes)

    Returns:
        Array of confidence scores
    """
    return np.max(probabilities, axis=1)


def get_prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate prediction entropy (uncertainty measure)

    Args:
        probabilities: Array of prediction probabilities

    Returns:
        Array of Shannon entropy values in **bits** (base-2).  The reliability
        scorer normalises by ``log2(n_classes)`` bits, so the units must match
        (ADR-018); dots/nats would inflate the confidence component.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-10, 1)
    entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    return entropy


def identify_high_confidence_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.8,
) -> HighConfidenceErrorInfo:
    """
    Identify high-confidence misclassifications

    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities
        threshold: Confidence threshold

    Returns:
        HighConfidenceErrorInfo: Structured result.  Callers must read
        ``["count"]`` for the number of errors; ``len(result)`` is the number of
        keys, not the number of errors.
    """
    confidence = get_confidence_scores(probabilities)
    errors = np.asarray(y_true) != np.asarray(y_pred)

    high_conf_errors = (confidence >= threshold) & errors
    count = int(np.count_nonzero(high_conf_errors))

    return {
        "indices": np.where(high_conf_errors)[0],
        "count": count,
        "percentage": (count / len(y_true)) * 100,
        "avg_confidence": (
            float(np.mean(confidence[high_conf_errors])) if count > 0 else 0.0
        ),
    }
