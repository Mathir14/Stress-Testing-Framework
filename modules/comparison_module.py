"""
Module 7: Model Comparison
Comprehensive side-by-side comparison of all trained models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.config import ScoringPolicy, get_config

#: Composite components, in display order.  The weights themselves live in
#: :class:`core.config.ScoringPolicy` (single owner, ADR-018); this tuple only
#: fixes the order components are presented in.
_COMPONENTS = ("performance", "robustness", "calibration")


class ModelComparator:
    """Aggregates and compares performance across trained models."""

    def __init__(self, policy: ScoringPolicy | None = None) -> None:
        """Bind the unified scoring policy.

        Args:
            policy: Composite weights, missing-component policy and the
                measured-components gate (ADR-018).  ``None`` (the default)
                uses the process-wide configuration singleton, which is what the
                app builds.
        """
        self.policy = get_config().scoring if policy is None else policy

    # ------------------------------------------------------------------ #
    #  Data collection helpers                                             #
    # ------------------------------------------------------------------ #

    def compile_performance_metrics(
        self, model_trainer: Any, dataset_name: str = "Test"
    ) -> dict[str, dict]:
        """
        Pull stored evaluation metrics for every trained model.

        Returns
        -------
        dict  { model_name -> {accuracy, precision, recall, f1, has_data} }
        """
        results: dict[str, dict] = {}
        for model_name in model_trainer.trained_models:
            entry = model_trainer.metrics.get(model_name, {}).get(dataset_name)
            if entry:
                results[model_name] = {
                    "accuracy": round(entry["accuracy"], 4),
                    "precision": round(entry["precision"], 4),
                    "recall": round(entry["recall"], 4),
                    "f1": round(entry["f1"], 4),
                    "has_data": True,
                }
            else:
                results[model_name] = {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "has_data": False,
                }
        return results

    def get_confusion_matrices(
        self, model_trainer: Any, dataset_name: str = "Test"
    ) -> dict:
        """Return confusion matrix + label info for each model."""
        cms: dict = {}
        for model_name in model_trainer.trained_models:
            entry = model_trainer.metrics.get(model_name, {}).get(dataset_name)
            if entry:
                cms[model_name] = {
                    "cm": entry["confusion_matrix"],
                    "true_labels": entry["true_labels"],
                    "predictions": entry["predictions"],
                }
        return cms

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    def plot_metrics_bar(self, metrics_dict: dict) -> go.Figure:
        """Grouped bar chart: accuracy / precision / recall / F1 per model."""
        models = [m for m, v in metrics_dict.items() if v["has_data"]]
        metric_names = ["accuracy", "precision", "recall", "f1"]
        colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

        fig = go.Figure()
        for metric, color in zip(metric_names, colors, strict=True):
            fig.add_trace(
                go.Bar(
                    name=metric.capitalize(),
                    x=models,
                    y=[metrics_dict[m][metric] for m in models],
                    marker_color=color,
                    text=[f"{metrics_dict[m][metric]:.3f}" for m in models],
                    textposition="outside",
                )
            )

        fig.update_layout(
            barmode="group",
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.15]),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            height=420,
        )
        return fig

    def plot_radar(self, metrics_dict: dict) -> go.Figure:
        """Spider / radar chart comparing all models across 4 metrics."""
        categories = ["Accuracy", "Precision", "Recall", "F1"]
        palette = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
        ]

        fig = go.Figure()
        for i, (model_name, vals) in enumerate(metrics_dict.items()):
            if not vals["has_data"]:
                continue
            r = [vals["accuracy"], vals["precision"], vals["recall"], vals["f1"]]
            # Close the polygon
            r_closed = r + [r[0]]
            cats_closed = categories + [categories[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=r_closed,
                    theta=cats_closed,
                    fill="toself",
                    name=model_name,
                    opacity=0.65,
                    line=dict(color=palette[i % len(palette)], width=2),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Radar Chart",
            template="plotly_white",
            height=450,
        )
        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> go.Figure:
        """Heatmap for a single confusion matrix."""
        n = cm.shape[0]
        labels = [str(i) for i in range(n)]

        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title=f"Confusion Matrix — {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=350,
        )
        return fig

    def plot_robustness_comparison(
        self, robustness_scores: dict[str, float]
    ) -> go.Figure:
        """Horizontal bar chart of robustness scores."""
        models = list(robustness_scores.keys())
        scores = [robustness_scores[m] for m in models]
        colors = [
            "#2ECC71" if s >= 70 else "#F39C12" if s >= 40 else "#E74C3C"
            for s in scores
        ]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=models,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Stress Robustness Scores by Model",
            xaxis=dict(title="Robustness Score (0–100)", range=[0, 115]),
            template="plotly_white",
            height=max(300, 60 * len(models)),
        )
        return fig

    def plot_composite_scores(self, composite: dict[str, float]) -> go.Figure:
        """Bar chart of composite reliability scores, sorted descending."""
        sorted_items = sorted(composite.items(), key=lambda x: x[1], reverse=True)
        models = [i[0] for i in sorted_items]
        scores = [i[1] for i in sorted_items]
        colors = [
            "#2ECC71" if s >= 70 else "#F39C12" if s >= 40 else "#E74C3C"
            for s in scores
        ]

        fig = go.Figure(
            go.Bar(
                x=models,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Composite Reliability Score (Performance + Robustness + Calibration)",
            xaxis_title="Model",
            yaxis=dict(title="Score (0–100)", range=[0, 115]),
            template="plotly_white",
            height=400,
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Composite scoring & recommendation                                  #
    # ------------------------------------------------------------------ #

    def _component_scores(
        self,
        model: str,
        vals: dict,
        robustness_dict: dict | None,
        calibration_ece: dict | None,
    ) -> tuple[dict[str, float | None], list[str], list[str]]:
        """Return the component scores, available names and missing names.

        Every component is scored on a 0-100 scale; an unmeasured component is
        ``None`` so the caller can apply the fixed-denominator zero policy
        without confusing it with a genuinely zero score.
        """
        components: dict[str, float | None] = {
            "performance": ((vals["accuracy"] + vals["f1"]) / 2) * 100.0
        }
        available = ["performance"]
        missing: list[str] = []

        if robustness_dict and model in robustness_dict:
            components["robustness"] = float(robustness_dict[model])
            available.append("robustness")
        else:
            components["robustness"] = None
            missing.append("robustness")

        if calibration_ece and model in calibration_ece:
            ece = float(calibration_ece[model])
            components["calibration"] = max(0.0, 1.0 - ece) * 100.0
            available.append("calibration")
        else:
            components["calibration"] = None
            missing.append("calibration")

        return components, available, missing

    def rate_models(
        self,
        metrics_dict: dict,
        robustness_dict: dict | None = None,
        calibration_ece: dict | None = None,
    ) -> dict[str, dict]:
        """Assess every model with data against the scoring policy.

        Returns:
            dict: ``{model -> {composite, rated, available_components,
            missing_components, components}}``.  ``rated`` is ``True`` when at
            least :attr:`ScoringPolicy.min_measured_components` components were
            measured; an unrated model must be shown separately rather than
            ranked against measured ones (ADR-018).
        """
        assessed: dict[str, dict] = {}
        for model, vals in metrics_dict.items():
            if not vals["has_data"]:
                continue

            components, available, missing = self._component_scores(
                model, vals, robustness_dict, calibration_ece
            )
            # Fixed denominator: a missing component contributes exactly zero
            # with no redistribution (ADR-018), so adding evidence can never
            # lower a score.
            total = sum(
                (0.0 if components[name] is None else components[name])
                * self.policy.weight_for(name)
                for name in _COMPONENTS
            )
            assessed[model] = {
                "composite": round(min(max(total, 0.0), 100.0), 2),
                "rated": len(available) >= self.policy.min_measured_components,
                "available_components": available,
                "missing_components": missing,
                "components": components,
            }
        return assessed

    def compute_composite_score(
        self,
        metrics_dict: dict,
        robustness_dict: dict | None = None,
        calibration_ece: dict | None = None,
    ) -> dict[str, float]:
        """Compute a 0-100 composite score per model (fixed denominator).

        Components and weights come from :class:`core.config.ScoringPolicy`:
        performance (accuracy+F1 average), robustness and calibration.  A
        component with no evidence scores ``missing_component_points`` (0.0)
        with **no** redistribution to the remaining components, which keeps the
        score monotone in the evidence — a model can never lose points by having
        an extra component measured (ADR-018, C3).
        """
        return {
            model: assessment["composite"]
            for model, assessment in self.rate_models(
                metrics_dict, robustness_dict, calibration_ece
            ).items()
        }

    def recommend_best_model(
        self,
        metrics_dict: dict,
        composite: dict[str, float],
        robustness_dict: dict | None = None,
        rated: dict[str, bool] | None = None,
    ) -> dict:
        """Return recommendation dict with best model and reason text.

        Args:
            metrics_dict: Compiled performance metrics per model.
            composite: Composite scores, as returned by
                :meth:`compute_composite_score`.
            robustness_dict: Optional robustness scores for the reason text.
            rated: Optional ``{model -> bool}`` gate.  Only rated models are
                eligible to be recommended or ranked; unrated models are listed
                under ``"unrated"`` instead of being silently dropped (ADR-018).
        """
        if not composite:
            return {"best_model": None, "reason": "No data available.", "ranking": []}

        is_rated = (lambda m: True) if rated is None else (
            lambda m: bool(rated.get(m, True))
        )
        eligible = [m for m in composite if is_rated(m)]
        unrated = sorted(
            (m for m in composite if not is_rated(m)),
            key=lambda m: composite[m],
            reverse=True,
        )

        if not eligible:
            return {
                "best_model": None,
                "reason": (
                    "No model has enough measured components to be rated. "
                    "Run robustness and calibration evaluation before comparing "
                    "models."
                ),
                "ranking": [],
                "unrated": unrated,
            }

        best = max(eligible, key=lambda m: composite[m])
        vals = metrics_dict[best]

        lines = [
            f"**{best}** achieves the highest composite score of **{composite[best]:.1f}/100**.",
            f"- Accuracy: {vals['accuracy']:.3f}",
            f"- F1 Score: {vals['f1']:.3f}",
        ]
        if robustness_dict and best in robustness_dict:
            lines.append(f"- Robustness Score: {robustness_dict[best]:.1f}/100")

        # Runner-up
        sorted_models = sorted(eligible, key=lambda m: composite[m], reverse=True)
        if len(sorted_models) > 1:
            runner = sorted_models[1]
            lines.append(
                f"\nRunner-up: **{runner}** "
                f"(composite score {composite[runner]:.1f}/100)"
            )

        return {
            "best_model": best,
            "composite_score": composite[best],
            "reason": "\n".join(lines),
            "ranking": sorted_models,
            "unrated": unrated,
        }

    # ------------------------------------------------------------------ #
    #  DataFrame helpers                                                   #
    # ------------------------------------------------------------------ #

    def build_comparison_df(self, metrics_dict: dict) -> pd.DataFrame:
        """Return a tidy DataFrame for tabular display."""
        rows = []
        for model, vals in metrics_dict.items():
            if vals["has_data"]:
                rows.append(
                    {
                        "Model": model,
                        "Accuracy": f"{vals['accuracy']:.4f}",
                        "Precision": f"{vals['precision']:.4f}",
                        "Recall": f"{vals['recall']:.4f}",
                        "F1 Score": f"{vals['f1']:.4f}",
                    }
                )
        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()
