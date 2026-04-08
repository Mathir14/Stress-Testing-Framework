"""
Calibration Analysis Module
Assesses and improves model probability calibration
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss


class CalibrationAnalyzer:
    """
    Analyzes model calibration - how well predicted probabilities
    match true outcome frequencies.
    """

    def __init__(self):
        self.calibration_results = {}

    def compute_calibration_metrics(
        self, y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """Compute comprehensive calibration metrics."""
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)
        n_classes = probabilities.shape[1]

        confidence = np.max(probabilities, axis=1)
        y_pred = np.argmax(probabilities, axis=1)
        correct = (y_pred == y_true).astype(int)

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

        ece = float(np.sum(bin_count / len(y_true) * np.abs(bin_acc - bin_conf)))

        populated = bin_count > 0
        mce = (
            float(np.max(np.abs(bin_acc[populated] - bin_conf[populated])))
            if populated.any()
            else 0.0
        )

        brier = self._multiclass_brier(y_true, probabilities, n_classes)

        avg_confidence = float(confidence.mean())
        avg_accuracy = float(correct.mean())

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

    def _multiclass_brier(self, y_true, probabilities, n_classes):
        y_onehot = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            if 0 <= int(label) < n_classes:
                y_onehot[i, int(label)] = 1
        return float(np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1)))

    def compute_per_class_calibration(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        class_names: List[str] = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compute calibration metrics per class (one-vs-rest)."""
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)
        n_classes = probabilities.shape[1]

        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        records = []
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            prob_c = probabilities[:, c]

            try:
                brier = float(brier_score_loss(y_bin, prob_c))
            except Exception:
                brier = float("nan")

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_ids = np.digitize(prob_c, bin_edges[1:-1])
            ece = 0.0
            for b in range(n_bins):
                mask = bin_ids == b
                if mask.sum() > 0:
                    ece += (mask.sum() / len(y_true)) * abs(
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
        self, y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 20
    ) -> go.Figure:
        """Histogram of confidence for correct vs incorrect predictions."""
        y_true = np.asarray(y_true)
        confidence = np.max(probabilities, axis=1)
        correct = np.argmax(probabilities, axis=1) == y_true

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
        self, y_true: np.ndarray, probabilities: np.ndarray
    ) -> float:
        """Grid-search for the temperature that minimises ECE."""
        y_true = np.asarray(y_true)
        best_temp, best_ece = 1.0, float("inf")
        for t in np.arange(0.1, 5.1, 0.1):
            m = self.compute_calibration_metrics(
                y_true, self.apply_temperature_scaling(probabilities, t)
            )
            if m["ece"] < best_ece:
                best_ece, best_temp = m["ece"], t
        return round(float(best_temp), 2)

    def get_calibration_quality(self, ece: float) -> str:
        if ece < 0.03:
            return "Excellent"
        elif ece < 0.07:
            return "Good"
        elif ece < 0.15:
            return "Moderate"
        else:
            return "Poor"
