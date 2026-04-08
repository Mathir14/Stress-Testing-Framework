"""
Module 8: Reliability Scoring
Computes a unified, multi-dimensional reliability score for each trained model.

Score breakdown (each component 0–25 pts, total 0–100):
  • Performance     – accuracy + F1
  • Calibration     – Expected Calibration Error (lower ECE → higher score)
  • Robustness      – mean performance retention across stress tests
  • Confidence Qual – entropy tightness + high-confidence error rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Grade helpers
# ──────────────────────────────────────────────────────────────────────────────


def _grade(score: float) -> tuple[str, str]:
    """Return (letter_grade, colour_hex) for a 0-100 score."""
    if score >= 90:
        return "A+", "#2ECC71"
    elif score >= 80:
        return "A", "#27AE60"
    elif score >= 70:
        return "B", "#F39C12"
    elif score >= 60:
        return "C", "#E67E22"
    elif score >= 50:
        return "D", "#E74C3C"
    else:
        return "F", "#C0392B"


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────


class ReliabilityScorer:
    """Compute and display unified reliability scores for trained models."""

    # Maximum points per component
    MAX_PERF = 25.0
    MAX_CAL = 25.0
    MAX_ROB = 25.0
    MAX_CONF = 25.0

    # ── Component calculators ────────────────────────────────────────────────

    def _performance_score(self, accuracy: float, f1: float) -> float:
        """0–25 pts.  Average of accuracy and F1, scaled to 25."""
        return round(((accuracy + f1) / 2) * self.MAX_PERF, 3)

    def _calibration_score(self, ece: float) -> float:
        """
        0–25 pts.
        ECE = 0   → 25 pts  (perfect)
        ECE = 0.5 → 0  pts  (terrible)
        Linear interpolation, clipped.
        """
        pts = max(0.0, (1.0 - ece / 0.5)) * self.MAX_CAL
        return round(pts, 3)

    def _robustness_score(self, avg_drop: float) -> float:
        """
        0–25 pts.
        avg_drop is the mean performance_drop fraction (0–1).
        0 drop → 25 pts,  ≥ 0.5 drop → 0 pts.
        """
        pts = max(0.0, (1.0 - avg_drop / 0.5)) * self.MAX_ROB
        return round(pts, 3)

    def _confidence_score(
        self,
        avg_entropy: float,
        hce_rate: float,
        max_entropy: float,
    ) -> float:
        """
        0–25 pts.
        Penalises high normalised entropy and high-confidence error rate (hce_rate).
          entropy component  : (1 - avg_entropy/max_entropy) * 12.5
          hce component      : (1 - hce_rate) * 12.5
        """
        if max_entropy <= 0:
            norm_entropy = 0.0
        else:
            norm_entropy = min(avg_entropy / max_entropy, 1.0)

        entropy_pts = (1.0 - norm_entropy) * (self.MAX_CONF / 2)
        hce_pts = max(0.0, 1.0 - hce_rate) * (self.MAX_CONF / 2)
        return round(entropy_pts + hce_pts, 3)

    # ── Core scoring pipeline ────────────────────────────────────────────────

    def score_model(
        self,
        model_name: str,
        *,
        accuracy: float | None = None,
        f1: float | None = None,
        ece: float | None = None,
        avg_drop: float | None = None,
        avg_entropy: float | None = None,
        hce_rate: float | None = None,
        n_classes: int = 2,
    ) -> dict:
        """
        Compute the reliability breakdown for one model.

        Any unavailable component defaults to the component midpoint,
        and is flagged so the UI can warn the user.

        Returns
        -------
        dict with keys:
          performance, calibration, robustness, confidence,
          total, grade, colour,
          available_components, missing_components,
          details  (sub-dict of raw inputs)
        """
        available, missing = [], []

        # ── Performance ────────────────────────────────────────────────────
        if accuracy is not None and f1 is not None:
            perf_pts = self._performance_score(accuracy, f1)
            available.append("Performance")
        else:
            perf_pts = self.MAX_PERF / 2  # neutral 12.5
            missing.append("Performance")

        # ── Calibration ────────────────────────────────────────────────────
        if ece is not None:
            cal_pts = self._calibration_score(ece)
            available.append("Calibration")
        else:
            cal_pts = self.MAX_CAL / 2
            missing.append("Calibration")

        # ── Robustness ─────────────────────────────────────────────────────
        if avg_drop is not None:
            rob_pts = self._robustness_score(avg_drop)
            available.append("Robustness")
        else:
            rob_pts = self.MAX_ROB / 2
            missing.append("Robustness")

        # ── Confidence quality ─────────────────────────────────────────────
        if avg_entropy is not None and hce_rate is not None:
            import math

            max_entropy = math.log2(max(n_classes, 2))
            conf_pts = self._confidence_score(avg_entropy, hce_rate, max_entropy)
            available.append("Confidence")
        else:
            conf_pts = self.MAX_CONF / 2
            missing.append("Confidence")

        total = round(perf_pts + cal_pts + rob_pts + conf_pts, 2)
        grade, colour = _grade(total)

        return {
            "model_name": model_name,
            "performance": perf_pts,
            "calibration": cal_pts,
            "robustness": rob_pts,
            "confidence": conf_pts,
            "total": total,
            "grade": grade,
            "colour": colour,
            "available_components": available,
            "missing_components": missing,
            "details": {
                "accuracy": accuracy,
                "f1": f1,
                "ece": ece,
                "avg_drop": avg_drop,
                "avg_entropy": avg_entropy,
                "hce_rate": hce_rate,
            },
        }

    def score_all_models(
        self,
        model_trainer,
        *,
        dataset_name: str = "Test",
        stress_results: dict | None = None,
        calibration_ece: dict | None = None,
        entropy_data: dict | None = None,
        hce_data: dict | None = None,
        n_classes: int = 2,
    ) -> dict[str, dict]:
        """
        Score every trained model, pulling data from session-state objects.

        Parameters
        ----------
        model_trainer   : ModelTrainer instance
        dataset_name    : which split to read metrics from
        stress_results  : batch_stress_results_by_model dict
        calibration_ece : {model_name: ece_float}
        entropy_data    : {model_name: avg_entropy}
        hce_data        : {model_name: hce_rate}  (0–1)
        n_classes       : number of output classes
        """
        scores: dict[str, dict] = {}

        for model_name in model_trainer.trained_models:
            entry = model_trainer.metrics.get(model_name, {}).get(dataset_name)

            accuracy = entry["accuracy"] if entry else None
            f1 = entry["f1"] if entry else None

            # Robustness: mean drop across stress types
            avg_drop: float | None = None
            if stress_results and model_name in stress_results:
                drops = [
                    r.get("performance_drop", 0)
                    for r in stress_results[model_name].values()
                    if isinstance(r, dict)
                ]
                avg_drop = float(np.mean(drops)) if drops else 0.0

            ece = (calibration_ece or {}).get(model_name)
            avg_entropy = (entropy_data or {}).get(model_name)
            hce_rate = (hce_data or {}).get(model_name)

            scores[model_name] = self.score_model(
                model_name,
                accuracy=accuracy,
                f1=f1,
                ece=ece,
                avg_drop=avg_drop,
                avg_entropy=avg_entropy,
                hce_rate=hce_rate,
                n_classes=n_classes,
            )

        return scores

    # ── Plots ────────────────────────────────────────────────────────────────

    def plot_gauge(self, score_dict: dict) -> go.Figure:
        """Gauge chart for a single model's total reliability score."""
        total = score_dict["total"]
        grade = score_dict["grade"]
        colour = score_dict["colour"]
        name = score_dict["model_name"]

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=total,
                delta={"reference": 70, "increasing": {"color": "#2ECC71"}},
                title={
                    "text": f"{name}<br><sub>Grade: {grade}</sub>",
                    "font": {"size": 16},
                },
                number={"suffix": " / 100", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": colour},
                    "steps": [
                        {"range": [0, 50], "color": "#FADBD8"},
                        {"range": [50, 70], "color": "#FAE5D3"},
                        {"range": [70, 90], "color": "#D5F5E3"},
                        {"range": [90, 100], "color": "#A9DFBF"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "value": 70,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
        return fig

    def plot_component_radar(self, scores_dict: dict) -> go.Figure:
        """Radar chart of component scores for all models."""
        categories = ["Performance", "Calibration", "Robustness", "Confidence"]
        max_vals = [self.MAX_PERF, self.MAX_CAL, self.MAX_ROB, self.MAX_CONF]
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
        for i, (model_name, sd) in enumerate(scores_dict.items()):
            r = [
                sd["performance"] / max_vals[0],
                sd["calibration"] / max_vals[1],
                sd["robustness"] / max_vals[2],
                sd["confidence"] / max_vals[3],
            ]
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
            title="Component Score Radar",
            template="plotly_white",
            height=440,
        )
        return fig

    def plot_stacked_bar(self, scores_dict: dict) -> go.Figure:
        """Stacked bar – each bar = one model, segments = 4 components."""
        models = list(scores_dict.keys())
        components = ["performance", "calibration", "robustness", "confidence"]
        labels = ["Performance", "Calibration", "Robustness", "Confidence"]
        colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]

        fig = go.Figure()
        for comp, label, color in zip(components, labels, colors):
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=models,
                    y=[scores_dict[m][comp] for m in models],
                    marker_color=color,
                )
            )

        fig.update_layout(
            barmode="stack",
            title="Reliability Score Breakdown by Component",
            xaxis_title="Model",
            yaxis=dict(title="Score (max 100)", range=[0, 110]),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            height=420,
        )
        return fig

    def plot_total_bar(self, scores_dict: dict) -> go.Figure:
        """Simple scored bar sorted by total, coloured by grade."""
        sorted_items = sorted(
            scores_dict.items(), key=lambda x: x[1]["total"], reverse=True
        )
        models = [i[0] for i in sorted_items]
        totals = [i[1]["total"] for i in sorted_items]
        colours = [i[1]["colour"] for i in sorted_items]
        grades = [i[1]["grade"] for i in sorted_items]

        fig = go.Figure(
            go.Bar(
                x=models,
                y=totals,
                marker_color=colours,
                text=[f"{t:.1f}  ({g})" for t, g in zip(totals, grades)],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Total Reliability Score (0–100)",
            xaxis_title="Model",
            yaxis=dict(title="Score", range=[0, 115]),
            template="plotly_white",
            height=380,
        )
        return fig

    # ── Summary helpers ──────────────────────────────────────────────────────

    def build_summary_df(self, scores_dict: dict) -> pd.DataFrame:
        """Return a tidy DataFrame for tabular display."""
        rows = []
        for model_name, sd in sorted(
            scores_dict.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            rows.append(
                {
                    "Model": model_name,
                    "Total": f"{sd['total']:.1f}",
                    "Grade": sd["grade"],
                    "Performance": f"{sd['performance']:.1f} / 25",
                    "Calibration": f"{sd['calibration']:.1f} / 25",
                    "Robustness": f"{sd['robustness']:.1f} / 25",
                    "Confidence": f"{sd['confidence']:.1f} / 25",
                    "Missing Data": (
                        ", ".join(sd["missing_components"])
                        if sd["missing_components"]
                        else "—"
                    ),
                }
            )
        return pd.DataFrame(rows).set_index("Model")

    def generate_recommendations(self, score_dict: dict) -> list[str]:
        """Return a list of actionable recommendation strings for one model."""
        recs = []
        sd = score_dict

        # Performance
        if sd["performance"] < 15:
            recs.append(
                "🔴 **Performance** is low. "
                "Consider adding more training data, feature engineering, "
                "or trying a more powerful model architecture."
            )
        elif sd["performance"] < 20:
            recs.append(
                "🟡 **Performance** is moderate. "
                "Hyperparameter tuning or ensemble methods may improve accuracy/F1."
            )

        # Calibration
        if sd["calibration"] < 12:
            recs.append(
                "🔴 **Calibration** is poor (high ECE). "
                "Apply temperature scaling (Module 6) or use Platt scaling / isotonic regression."
            )
        elif sd["calibration"] < 18:
            recs.append(
                "🟡 **Calibration** could be improved. "
                "Review the confidence histogram and consider post-hoc calibration."
            )

        # Robustness
        if sd["robustness"] < 12:
            recs.append(
                "🔴 **Robustness** is low. "
                "Model is sensitive to perturbations. "
                "Consider data augmentation, adversarial training, or feature standardisation."
            )
        elif sd["robustness"] < 18:
            recs.append(
                "🟡 **Robustness** is moderate. "
                "Investigate which stress types cause the largest drops (Module 4)."
            )

        # Confidence quality
        if sd["confidence"] < 12:
            recs.append(
                "🔴 **Confidence Quality** is poor. "
                "The model produces high-entropy or misleadingly confident predictions. "
                "Review high-confidence errors (Module 3)."
            )
        elif sd["confidence"] < 18:
            recs.append(
                "🟡 **Confidence Quality** is moderate. "
                "Some high-confidence errors detected — inspect borderline cases."
            )

        if not recs:
            recs.append(
                "✅ All reliability components are in good shape. "
                "Continue monitoring with new data distributions."
            )
        return recs
