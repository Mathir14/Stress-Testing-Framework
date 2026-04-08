"""
Metrics Utilities
Helper functions for calculating various metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def get_confidence_scores(probabilities):
    """
    Get confidence scores (max probability for each prediction)

    Args:
        probabilities: Array of prediction probabilities (n_samples, n_classes)

    Returns:
        Array of confidence scores
    """
    return np.max(probabilities, axis=1)


def get_prediction_entropy(probabilities):
    """
    Calculate prediction entropy (uncertainty measure)

    Args:
        probabilities: Array of prediction probabilities

    Returns:
        Array of entropy values
    """
    # Avoid log(0)
    probabilities = np.clip(probabilities, 1e-10, 1)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropy


def identify_high_confidence_errors(y_true, y_pred, probabilities, threshold=0.8):
    """
    Identify high-confidence misclassifications

    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities
        threshold: Confidence threshold

    Returns:
        Dictionary with high-confidence error information
    """
    confidence = get_confidence_scores(probabilities)
    errors = y_true != y_pred

    high_conf_errors = (confidence >= threshold) & errors

    return {
        "indices": np.where(high_conf_errors)[0],
        "count": np.sum(high_conf_errors),
        "percentage": (np.sum(high_conf_errors) / len(y_true)) * 100,
        "avg_confidence": (
            np.mean(confidence[high_conf_errors]) if np.sum(high_conf_errors) > 0 else 0
        ),
    }


def calculate_brier_score(y_true, probabilities):
    """
    Calculate Brier score for multi-class classification

    Args:
        y_true: True labels
        probabilities: Prediction probabilities

    Returns:
        Brier score
    """
    n_classes = probabilities.shape[1]
    y_true_binary = np.zeros((len(y_true), n_classes))

    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1

    return np.mean(np.sum((probabilities - y_true_binary) ** 2, axis=1))


def get_confidence_bins(confidence, n_bins=10):
    """
    Bin confidence scores

    Args:
        confidence: Array of confidence scores
        n_bins: Number of bins

    Returns:
        Bin edges and bin assignments
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bins, bin_indices
