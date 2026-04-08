"""
Module 7: Model Comparison
Comprehensive side-by-side comparison of all trained models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ModelComparator:
    """Aggregates and compares performance across trained models."""

    # Weights used for composite score (sum to 1.0)
    METRIC_WEIGHT = 0.5  # accuracy / F1
    ROBUSTNESS_WEIGHT = 0.25  # stress robustness score
    CALIBRATION_WEIGHT = 0.25  # 1 - ECE (lower ECE → higher component)

    # ------------------------------------------------------------------ #
    #  Data collection helpers                                             #
    # ------------------------------------------------------------------ #

    def compile_performance_metrics(
        self, model_trainer, dataset_name: str = "Test"
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

    def get_confusion_matrices(self, model_trainer, dataset_name: str = "Test") -> dict:
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
        for metric, color in zip(metric_names, colors):
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

    def compute_composite_score(
        self,
        metrics_dict: dict,
        robustness_dict: dict | None = None,
        calibration_ece: dict | None = None,
    ) -> dict[str, float]:
        """
        Compute a 0–100 composite score per model.

        Weights
        -------
        - 50 % performance  : average of F1 + accuracy (0–1 each → 0–50)
        - 25 % robustness   : 0–100 robustness score → 0–25
        - 25 % calibration  : (1 – ECE) capped to [0, 1]  → 0–25
          If robustness / calibration absent, the remaining weight shifts to performance.
        """
        composite: dict[str, float] = {}

        for model in metrics_dict:
            vals = metrics_dict[model]
            if not vals["has_data"]:
                continue

            # --- Performance component (max 50 points) ---
            perf = (vals["accuracy"] + vals["f1"]) / 2  # 0–1
            perf_score = perf * 50

            # --- Robustness component (max 25 points) ---
            if robustness_dict and model in robustness_dict:
                rob_score = (robustness_dict[model] / 100) * 25
                has_rob = True
            else:
                rob_score = 0.0
                has_rob = False

            # --- Calibration component (max 25 points) ---
            if calibration_ece and model in calibration_ece:
                ece = calibration_ece[model]
                cal_score = max(0.0, (1 - ece)) * 25
                has_cal = True
            else:
                cal_score = 0.0
                has_cal = False

            # Redistribute missing weights to performance
            missing = (0 if has_rob else 25) + (0 if has_cal else 25)
            perf_bonus = perf * missing  # proportional bonus

            total = perf_score + perf_bonus + rob_score + cal_score
            composite[model] = round(min(total, 100), 2)

        return composite

    def recommend_best_model(
        self,
        metrics_dict: dict,
        composite: dict[str, float],
        robustness_dict: dict | None = None,
    ) -> dict:
        """
        Return recommendation dict with best model and reason text.
        """
        if not composite:
            return {"best_model": None, "reason": "No data available."}

        best = max(composite, key=lambda m: composite[m])
        vals = metrics_dict[best]

        lines = [
            f"**{best}** achieves the highest composite score of **{composite[best]:.1f}/100**.",
            f"- Accuracy: {vals['accuracy']:.3f}",
            f"- F1 Score: {vals['f1']:.3f}",
        ]
        if robustness_dict and best in robustness_dict:
            lines.append(f"- Robustness Score: {robustness_dict[best]:.1f}/100")

        # Runner-up
        sorted_models = sorted(composite, key=lambda m: composite[m], reverse=True)
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
