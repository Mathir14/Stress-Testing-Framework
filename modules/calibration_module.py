"""
Calibration Analysis Module
Assesses and improves model probability calibration

Architecture reference: ADR-017.  Calibration and Brier scoring operate in the
model's **class-index space**: a raw label is mapped to the probability column it
belongs to via the estimator's ``classes_``.  When ``classes`` is not supplied,
labels must already be exactly ``0..K-1`` or a typed ``ValidationError`` is
raised — a call site that forgets to thread ``classes`` fails loudly instead of
silently mis-scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.config import ScoringPolicy, get_config
from core.errors import ValidationError

logger = logging.getLogger(__name__)

__all__ = ["CalibrationAnalyzer", "align_target_indices", "multiclass_brier"]


def _hashable(value: Any) -> Any:
    """Return a plain-Python equivalent of a numpy scalar for use as a dict key.

    ``np.str_("cat") == "cat"`` and ``hash(np.str_("cat")) == hash("cat")``, so
    this is not strictly required on the pinned stack; it is here so the mapping
    cannot depend on that equality staying true across numpy versions.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def align_target_indices(y_true: np.ndarray, classes: Any) -> np.ndarray:
    """Map raw target labels onto probability-column indices.

    Args:
        y_true: Raw target labels, in the estimator's native label space.
        classes: The estimator's ``classes_`` array, or ``None`` for strict
            identity (labels must be exactly ``0..K-1``).

    Returns:
        np.ndarray: Integer column indices, one per row.

    Raises:
        ValidationError: If ``classes`` is ``None`` and the labels are not
            exactly ``0..K-1``, or if a label is absent from ``classes``, or if
            the mapping exceeds the probability array's column count.
    """
    raw = np.asarray(y_true)
    if classes is None:
        unique = np.unique(raw)
        expected = np.arange(len(unique))
        if unique.shape != expected.shape or not np.array_equal(unique, expected):
            preview = [str(value) for value in unique[:10]]
            raise ValidationError(
                "Calibration requires class labels exactly 0..K-1 when `classes` "
                f"is not supplied; got labels {preview}"
                f"{'...' if len(unique) > 10 else ''}. Pass the estimator's "
                "`classes_` so labels can be mapped to probability columns.",
                context={"labels": [str(value) for value in unique[:20]]},
            )
        return raw.astype(int)

    index_of: dict[Any, int] = {}
    for position, label in enumerate(np.asarray(classes)):
        index_of[_hashable(label)] = position

    out = np.empty(raw.shape[0], dtype=int)
    for position, label in enumerate(raw):
        key = _hashable(label)
        if key not in index_of:
            raise ValidationError(
                f"Target label {label!r} is not one of the estimator's classes "
                f"({list(np.asarray(classes))[:10]}). The target was encoded or "
                "predicted in a different label space.",
                context={"label": str(label), "classes": [str(c) for c in index_of]},
            )
        out[position] = index_of[key]
    return out


def multiclass_brier(
    y_true: np.ndarray, probabilities: np.ndarray, *, classes: Any = None
) -> float:
    """Compute the mean multi-class Brier score.

    This is the framework's **single** Brier implementation (ADR-017): the
    former ``utils.calculate_brier_score`` was removed so the two copies could
    not disagree about label guarding.

    Args:
        y_true: Raw target labels.
        probabilities: ``(n_samples, n_classes)`` predicted probabilities.
        classes: The estimator's ``classes_`` array, or ``None`` for strict
            identity (labels must be exactly ``0..K-1``).

    Returns:
        float: Mean squared error between the one-hot targets and the predicted
        probabilities.

    Raises:
        ValidationError: If labels cannot be aligned to probability columns.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValidationError(
            f"Probabilities must be 2-D (n_samples, n_classes); got shape "
            f"{list(probs.shape)}.",
            context={"shape": list(probs.shape)},
        )
    n_classes = probs.shape[1]
    indices = align_target_indices(y_true, classes)
    if indices.size and int(indices.max()) >= n_classes:
        raise ValidationError(
            f"Target label maps to column {int(indices.max())} but the "
            f"probability matrix has only {n_classes} column(s).",
            context={"max_index": int(indices.max()), "n_classes": n_classes},
        )
    onehot = np.zeros((indices.shape[0], n_classes), dtype=float)
    if indices.size:
        onehot[np.arange(indices.shape[0]), indices] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


class CalibrationAnalyzer:
    """
    Analyzes model calibration - how well predicted probabilities
    match true outcome frequencies.
    """

    def __init__(self, policy: ScoringPolicy | None = None) -> None:
        """Bind the scoring policy.

        Args:
            policy: Policy supplying the ECE quality bands.  ``None`` uses the
                process-wide configuration singleton.
        """
        self.calibration_results: Dict = {}
        self._policy = get_config().scoring if policy is None else policy

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        *,
        classes: Any = None,
        n_bins: int = 10,
    ) -> Dict:
        """Compute comprehensive calibration metrics.

        Args:
            y_true: Raw target labels.
            probabilities: ``(n_samples, n_classes)`` predicted probabilities.
            classes: Estimator ``classes_`` for label alignment.
            n_bins: Number of confidence bins.

        Returns:
            Dict: ECE, MCE, Brier, confidence/accuracy summaries and bin arrays.
        """
        probs = np.asarray(probabilities, dtype=float)
        indices = align_target_indices(y_true, classes)
        if indices.size and int(indices.max()) >= probs.shape[1]:
            raise ValidationError(
                f"Target label maps to column {int(indices.max())} but the "
                f"probability matrix has only {probs.shape[1]} column(s).",
                context={
                    "max_index": int(indices.max()),
                    "n_classes": probs.shape[1],
                },
            )

        confidence = np.max(probs, axis=1)
        y_pred = np.argmax(probs, axis=1)
        correct = (y_pred == indices).astype(int)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(confidence, bin_edges[1:-1])

        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_count = np.zeros(n_bins)

        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                bin_acc[b] = correct[mask].mean()
                bin_conf[b] = confidence[mask].mean()
                bin_count[b] = mask.sum()

        ece = float(np.sum(bin_count / len(probs) * np.abs(bin_acc - bin_conf)))

        populated = bin_count > 0
        mce = (
            float(np.max(np.abs(bin_acc[populated] - bin_conf[populated])))
            if populated.any()
            else 0.0
        )

        brier = multiclass_brier(y_true, probs, classes=classes)

        avg_confidence = float(confidence.mean()) if len(probs) else 0.0
        avg_accuracy = float(correct.mean()) if len(probs) else 0.0

        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "overconfidence": avg_confidence - avg_accuracy,
            "bin_acc": bin_acc,
            "bin_conf": bin_conf,
            "bin_count": bin_count,
            "bin_edges": bin_edges,
            "n_bins": n_bins,
        }

    def compute_per_class_calibration(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        *,
        classes: Any = None,
        class_names: List[str] | None = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compute calibration metrics per class (one-vs-rest)."""
        probs = np.asarray(probabilities, dtype=float)
        indices = align_target_indices(y_true, classes)
        n_classes = probs.shape[1]

        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        records = []
        for c in range(n_classes):
            y_bin = (indices == c).astype(int)
            prob_c = probs[:, c]

            if np.isfinite(prob_c).all():
                brier = float(np.mean((prob_c - y_bin) ** 2))
            else:
                # The single documented degradation permitted by architecture.md
                # §5: per-class Brier -> NaN.  It MUST be logged, otherwise a
                # silently-NaN column is indistinguishable from a real result.
                brier = float("nan")
                logger.warning(
                    "Brier score unavailable for class %s; recording NaN. Check "
                    "that probabilities for this class are finite and within "
                    "[0, 1].",
                    class_names[c] if c < len(class_names) else c,
                )

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_ids = np.digitize(prob_c, bin_edges[1:-1])
            ece = 0.0
            for b in range(n_bins):
                mask = bin_ids == b
                if mask.sum() > 0:
                    ece += (mask.sum() / len(indices)) * abs(
                        y_bin[mask].mean() - prob_c[mask].mean()
                    )

            records.append(
                {
                    "Class": class_names[c] if c < len(class_names) else f"Class {c}",
                    "ECE": round(ece, 4),
                    "Brier Score": round(brier, 4),
                    "Avg Predicted Prob": round(float(prob_c.mean()), 4),
                    "Actual Prevalence": round(float(y_bin.mean()), 4),
                }
            )

        return pd.DataFrame(records)

    def plot_calibration_curve(
        self, metrics: Dict, model_name: str = "Model"
    ) -> go.Figure:
        """Plot reliability diagram."""
        bin_conf = metrics["bin_conf"]
        bin_acc = metrics["bin_acc"]
        bin_count = metrics["bin_count"]
        bin_edges = metrics["bin_edges"]
        populated = bin_count > 0

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(color="gray", dash="dash"),
            )
        )

        sizes = bin_count[populated]
        norm_sizes = sizes / sizes.max() * 20 + 6 if sizes.max() > 0 else sizes + 6

        fig.add_trace(
            go.Scatter(
                x=bin_conf[populated],
                y=bin_acc[populated],
                mode="lines+markers",
                name=model_name,
                marker=dict(size=norm_sizes, color="rgb(99,110,250)"),
                line=dict(color="rgb(99,110,250)"),
                hovertemplate="Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<extra></extra>",
            )
        )

        for i in range(metrics["n_bins"]):
            if not populated[i]:
                continue
            color = (
                "rgba(255,80,80,0.15)"
                if bin_conf[i] > bin_acc[i]
                else "rgba(80,80,255,0.15)"
            )
            fig.add_shape(
                type="rect",
                x0=bin_edges[i],
                x1=bin_edges[i + 1],
                y0=min(bin_conf[i], bin_acc[i]),
                y1=max(bin_conf[i], bin_acc[i]),
                fillcolor=color,
                line=dict(width=0),
                layer="below",
            )

        fig.update_layout(
            title=f"Reliability Diagram — {model_name}",
            xaxis_title="Mean Predicted Confidence",
            yaxis_title="Observed Accuracy",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=420,
        )
        return fig

    def plot_confidence_histogram(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        *,
        classes: Any = None,
        n_bins: int = 20,
    ) -> go.Figure:
        """Histogram of confidence for correct vs incorrect predictions."""
        probs = np.asarray(probabilities, dtype=float)
        indices = align_target_indices(y_true, classes)
        confidence = np.max(probs, axis=1)
        correct = np.argmax(probs, axis=1) == indices

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=confidence[correct],
                nbinsx=n_bins,
                name="Correct",
                marker_color="rgba(50,200,100,0.7)",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=confidence[~correct],
                nbinsx=n_bins,
                name="Incorrect",
                marker_color="rgba(255,80,80,0.7)",
                opacity=0.75,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title="Confidence Distribution: Correct vs Incorrect",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            height=380,
        )
        return fig

    def plot_calibration_comparison(self, results_dict: Dict) -> go.Figure:
        """Compare ECE / MCE / Brier across models."""
        models = list(results_dict.keys())
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="ECE",
                x=models,
                y=[results_dict[m]["ece"] for m in models],
                marker_color="rgb(99,110,250)",
            )
        )
        fig.add_trace(
            go.Bar(
                name="MCE",
                x=models,
                y=[results_dict[m]["mce"] for m in models],
                marker_color="rgb(239,85,59)",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Brier Score",
                x=models,
                y=[results_dict[m]["brier_score"] for m in models],
                marker_color="rgb(0,204,150)",
            )
        )
        fig.update_layout(
            barmode="group",
            title="Calibration Metrics Comparison (lower = better)",
            xaxis_title="Model",
            yaxis_title="Error",
            height=400,
        )
        return fig

    def apply_temperature_scaling(
        self, probabilities: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Scale probabilities by temperature (>1 softens, <1 sharpens)."""
        logits = np.log(np.clip(probabilities, 1e-10, 1))
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    def find_optimal_temperature(
        self, y_true: np.ndarray, probabilities: np.ndarray, *, classes: Any = None
    ) -> float:
        """Grid-search for the temperature that minimises ECE."""
        probs = np.asarray(probabilities, dtype=float)
        best_temp, best_ece = 1.0, float("inf")
        for t in np.arange(0.1, 5.1, 0.1):
            m = self.compute_calibration_metrics(
                y_true,
                self.apply_temperature_scaling(probs, t),
                classes=classes,
            )
            if m["ece"] < best_ece:
                best_ece, best_temp = m["ece"], t
        return round(float(best_temp), 2)

    def get_calibration_quality(self, ece: float) -> str:
        """Classify an ECE value using the bound :class:`ScoringPolicy`."""
        return self._policy.resolve_ece_quality(ece)
