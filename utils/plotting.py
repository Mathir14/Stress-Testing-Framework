"""
Plotting Utilities
Helper functions for creating visualizations
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_confidence_histogram(confidence, correct_mask, bins=20):
    """
    Plot confidence histogram split by correctness

    Args:
        confidence: Array of confidence scores
        correct_mask: Boolean array indicating correct predictions
        bins: Number of bins

    Returns:
        Plotly figure
    """
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=correct_conf,
            name="Correct",
            marker_color="green",
            opacity=0.7,
            nbinsx=bins,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=incorrect_conf,
            name="Incorrect",
            marker_color="red",
            opacity=0.7,
            nbinsx=bins,
        )
    )

    fig.update_layout(
        title="Confidence Distribution by Prediction Correctness",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        barmode="overlay",
        width=800,
        height=500,
    )

    return fig


def plot_confidence_by_class(confidence, predictions, true_labels, class_names=None):
    """
    Plot confidence distribution by predicted class

    Args:
        confidence: Array of confidence scores
        predictions: Predicted labels
        true_labels: True labels
        class_names: Optional list of class names

    Returns:
        Plotly figure
    """
    unique_classes = np.unique(predictions)

    if class_names is None:
        class_names = [f"Class {i}" for i in unique_classes]

    fig = go.Figure()

    for i, cls in enumerate(unique_classes):
        mask = predictions == cls
        cls_confidence = confidence[mask]

        fig.add_trace(
            go.Box(
                y=cls_confidence,
                name=class_names[i] if i < len(class_names) else f"Class {cls}",
                boxmean="sd",
            )
        )

    fig.update_layout(
        title="Confidence Distribution by Predicted Class",
        yaxis_title="Confidence Score",
        xaxis_title="Predicted Class",
        showlegend=True,
        width=800,
        height=500,
    )

    return fig


def plot_confidence_accuracy_curve(confidence, correct_mask, n_bins=10):
    """
    Plot accuracy vs confidence curve

    Args:
        confidence: Array of confidence scores
        correct_mask: Boolean array indicating correct predictions
        n_bins: Number of confidence bins

    Returns:
        Plotly figure
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    accuracies = []
    counts = []

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if i == n_bins - 1:  # Include the last bin edge
            mask = (confidence >= bins[i]) & (confidence <= bins[i + 1])

        count = np.sum(mask)
        counts.append(count)

        if count > 0:
            acc = np.mean(correct_mask[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Accuracy line
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=accuracies,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        ),
        secondary_y=False,
    )

    # Sample count bars
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            name="Sample Count",
            marker_color="lightblue",
            opacity=0.3,
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Confidence Score")
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text="Sample Count", secondary_y=True)

    fig.update_layout(
        title="Calibration Curve: Accuracy vs Confidence", width=800, height=500
    )

    return fig


def plot_error_analysis(confidence, predictions, true_labels, threshold=0.8):
    """
    Plot error analysis comparing low vs high confidence errors

    Args:
        confidence: Array of confidence scores
        predictions: Predicted labels
        true_labels: True labels
        threshold: Confidence threshold

    Returns:
        Plotly figure
    """
    errors = predictions != true_labels

    high_conf = confidence >= threshold
    low_conf = confidence < threshold

    categories = ["Correct", "Errors"]

    high_conf_correct = np.sum(~errors & high_conf)
    high_conf_errors = np.sum(errors & high_conf)
    low_conf_correct = np.sum(~errors & low_conf)
    low_conf_errors = np.sum(errors & low_conf)

    fig = go.Figure(
        data=[
            go.Bar(
                name=f"High Confidence (≥{threshold})",
                x=categories,
                y=[high_conf_correct, high_conf_errors],
                text=[high_conf_correct, high_conf_errors],
                textposition="auto",
                marker_color="darkblue",
            ),
            go.Bar(
                name=f"Low Confidence (<{threshold})",
                x=categories,
                y=[low_conf_correct, low_conf_errors],
                text=[low_conf_correct, low_conf_errors],
                textposition="auto",
                marker_color="lightblue",
            ),
        ]
    )

    fig.update_layout(
        title=f"Error Analysis by Confidence Level (Threshold: {threshold})",
        xaxis_title="Prediction Outcome",
        yaxis_title="Count",
        barmode="group",
        width=700,
        height=500,
    )

    return fig


def plot_entropy_distribution(entropy, correct_mask):
    """
    Plot entropy (uncertainty) distribution

    Args:
        entropy: Array of entropy values
        correct_mask: Boolean array indicating correct predictions

    Returns:
        Plotly figure
    """
    correct_entropy = entropy[correct_mask]
    incorrect_entropy = entropy[~correct_mask]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=correct_entropy,
            name="Correct",
            marker_color="green",
            opacity=0.7,
            nbinsx=30,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=incorrect_entropy,
            name="Incorrect",
            marker_color="red",
            opacity=0.7,
            nbinsx=30,
        )
    )

    fig.update_layout(
        title="Prediction Entropy (Uncertainty) Distribution",
        xaxis_title="Entropy",
        yaxis_title="Count",
        barmode="overlay",
        width=800,
        height=500,
    )

    return fig
