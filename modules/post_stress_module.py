"""
Post-Stress Evaluation Module
Deep analysis of stress test results to assess model robustness
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class PostStressAnalyzer:
    """
    Analyzes stress test results to provide comprehensive robustness insights
    """

    def __init__(self):
        self.stress_results = {}
        self.robustness_scores = {}

    def add_stress_result(
        self, test_name: str, result: Dict, model_name: str = "default"
    ):
        """
        Store a stress test result for analysis

        Args:
            test_name: Name of the stress test
            result: Dictionary containing test metrics
            model_name: Name of the model tested
        """
        if model_name not in self.stress_results:
            self.stress_results[model_name] = {}

        self.stress_results[model_name][test_name] = result

    def add_batch_results(self, results: Dict, model_name: str = "default"):
        """
        Store batch stress test results

        Args:
            results: Dictionary of stress test results
            model_name: Name of the model tested
        """
        for test_name, result in results.items():
            self.add_stress_result(test_name, result, model_name)

    def calculate_robustness_score(
        self, model_name: str = "default", weights: Optional[Dict] = None
    ) -> float:
        """
        Calculate overall robustness score based on stress test results

        Args:
            model_name: Name of the model
            weights: Optional weights for different metrics

        Returns:
            Robustness score (0-100, higher is better)
        """
        if model_name not in self.stress_results:
            return 0.0

        results = self.stress_results[model_name]

        if not results:
            return 0.0

        # Default weights
        if weights is None:
            weights = {
                "accuracy_retention": 0.4,  # How much accuracy is retained
                "prediction_stability": 0.3,  # Prediction agreement
                "performance_consistency": 0.3,  # Consistency across tests
            }

        # Accuracy retention (average of stressed/original accuracy)
        acc_retentions = []
        for result in results.values():
            if result["accuracy_original"] > 0:
                retention = result["accuracy_stressed"] / result["accuracy_original"]
                acc_retentions.append(min(retention, 1.0))  # Cap at 1.0

        avg_retention = np.mean(acc_retentions) if acc_retentions else 0

        # Prediction stability (average prediction agreement)
        agreements = [result["prediction_agreement"] for result in results.values()]
        avg_agreement = np.mean(agreements) if agreements else 0

        # Performance consistency (1 - coefficient of variation of accuracy)
        stressed_accs = [result["accuracy_stressed"] for result in results.values()]
        if len(stressed_accs) > 1:
            consistency = 1 - (np.std(stressed_accs) / np.mean(stressed_accs))
            consistency = max(0, min(consistency, 1))  # Clip to [0, 1]
        else:
            consistency = 1.0

        # Weighted score
        score = (
            weights["accuracy_retention"] * avg_retention
            + weights["prediction_stability"] * avg_agreement
            + weights["performance_consistency"] * consistency
        ) * 100

        self.robustness_scores[model_name] = {
            "overall_score": score,
            "accuracy_retention": avg_retention * 100,
            "prediction_stability": avg_agreement * 100,
            "performance_consistency": consistency * 100,
        }

        return score

    def get_vulnerability_analysis(self, model_name: str = "default") -> pd.DataFrame:
        """
        Analyze which stress types cause the most vulnerability

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with vulnerability analysis
        """
        if model_name not in self.stress_results:
            return pd.DataFrame()

        results = self.stress_results[model_name]

        analysis = []
        for test_name, result in results.items():
            analysis.append(
                {
                    "Stress Test": test_name,
                    "Performance Drop (%)": result["performance_drop_pct"],
                    "Accuracy Loss": result["performance_drop"],
                    "Prediction Changes (%)": (1 - result["prediction_agreement"])
                    * 100,
                    "Stressed Accuracy": result["accuracy_stressed"],
                    "F1 (Stressed)": result.get("f1_stressed", 0),
                    "Severity": self._classify_severity(result["performance_drop_pct"]),
                }
            )

        df = pd.DataFrame(analysis)
        df = df.sort_values("Performance Drop (%)", ascending=False)

        return df

    def _classify_severity(self, drop_pct: float) -> str:
        """Classify performance drop severity"""
        if drop_pct < 5:
            return "Low"
        elif drop_pct < 15:
            return "Medium"
        elif drop_pct < 30:
            return "High"
        else:
            return "Critical"

    def get_stress_type_summary(self, model_name: str = "default") -> pd.DataFrame:
        """
        Summarize results by stress type category

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with category summaries
        """
        if model_name not in self.stress_results:
            return pd.DataFrame()

        results = self.stress_results[model_name]

        # Categorize tests
        categories = {
            "Noise": ["gaussian", "uniform"],
            "Dropout": ["dropout"],
            "Corruption": ["corruption"],
            "Scaling": ["scale"],
            "Distribution": ["shift", "distribution"],
        }

        category_stats = []

        for category, keywords in categories.items():
            # Filter tests matching this category
            category_results = {
                name: res
                for name, res in results.items()
                if any(kw in name.lower() for kw in keywords)
            }

            if category_results:
                drops = [r["performance_drop_pct"] for r in category_results.values()]
                agreements = [
                    r["prediction_agreement"] for r in category_results.values()
                ]

                category_stats.append(
                    {
                        "Category": category,
                        "Num Tests": len(category_results),
                        "Avg Drop (%)": np.mean(drops),
                        "Max Drop (%)": np.max(drops),
                        "Avg Agreement": np.mean(agreements),
                        "Robustness": (
                            "High"
                            if np.mean(drops) < 10
                            else "Medium" if np.mean(drops) < 20 else "Low"
                        ),
                    }
                )

        return pd.DataFrame(category_stats)

    def plot_robustness_radar(self, model_name: str = "default"):
        """
        Create radar chart showing robustness across dimensions

        Args:
            model_name: Name of the model

        Returns:
            Plotly figure
        """
        if model_name not in self.robustness_scores:
            self.calculate_robustness_score(model_name)

        if model_name not in self.robustness_scores:
            return None

        scores = self.robustness_scores[model_name]

        categories = [
            "Accuracy Retention",
            "Prediction Stability",
            "Performance Consistency",
        ]

        values = [
            scores["accuracy_retention"],
            scores["prediction_stability"],
            scores["performance_consistency"],
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=model_name,
                line=dict(color="rgb(99, 110, 250)"),
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Robustness Dimensions",
            height=400,
        )

        return fig

    def plot_vulnerability_heatmap(self, model_name: str = "default"):
        """
        Create heatmap showing vulnerability across stress tests

        Args:
            model_name: Name of the model

        Returns:
            Plotly figure
        """
        df = self.get_vulnerability_analysis(model_name)

        if df.empty:
            return None

        # Create heatmap data
        metrics = ["Performance Drop (%)", "Prediction Changes (%)"]
        tests = df["Stress Test"].tolist()

        data = df[metrics].T.values

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=tests,
                y=metrics,
                colorscale="Reds",
                text=np.round(data, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Value"),
            )
        )

        fig.update_layout(
            title="Vulnerability Heatmap",
            xaxis_title="Stress Test",
            yaxis_title="Metric",
            height=300,
        )

        return fig

    def compare_model_robustness(self, model_names: List[str] = None):
        """
        Compare robustness across multiple models

        Args:
            model_names: List of model names to compare

        Returns:
            Plotly figure
        """
        if model_names is None:
            model_names = list(self.stress_results.keys())

        # Calculate scores for all models
        comparison_data = []

        for model_name in model_names:
            if model_name in self.stress_results:
                self.calculate_robustness_score(model_name)

                if model_name in self.robustness_scores:
                    scores = self.robustness_scores[model_name]
                    comparison_data.append(
                        {
                            "Model": model_name,
                            "Overall Score": scores["overall_score"],
                            "Accuracy Retention": scores["accuracy_retention"],
                            "Prediction Stability": scores["prediction_stability"],
                            "Consistency": scores["performance_consistency"],
                        }
                    )

        if not comparison_data:
            return None

        df = pd.DataFrame(comparison_data)

        # Create grouped bar chart
        fig = go.Figure()

        metrics = [
            "Accuracy Retention",
            "Prediction Stability",
            "Consistency",
        ]

        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric]))

        fig.update_layout(
            barmode="group",
            title="Model Robustness Comparison",
            xaxis_title="Model",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
        )

        return fig

    def get_recommendations(self, model_name: str = "default") -> List[str]:
        """
        Generate recommendations based on stress test analysis

        Args:
            model_name: Name of the model

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if model_name not in self.stress_results:
            return ["No stress test results available for analysis."]

        # Get vulnerability analysis
        vuln_df = self.get_vulnerability_analysis(model_name)

        if vuln_df.empty:
            return ["Insufficient data for recommendations."]

        # Calculate robustness score
        score = self.calculate_robustness_score(model_name)

        # Overall robustness assessment
        if score >= 80:
            recommendations.append(
                f"✅ **Excellent robustness** (score: {score:.1f}/100). Model performs well under stress."
            )
        elif score >= 60:
            recommendations.append(
                f"⚠️ **Good robustness** (score: {score:.1f}/100). Some improvements possible."
            )
        elif score >= 40:
            recommendations.append(
                f"⚠️ **Moderate robustness** (score: {score:.1f}/100). Consider improvements."
            )
        else:
            recommendations.append(
                f"❌ **Low robustness** (score: {score:.1f}/100). Significant improvements needed."
            )

        # Identify worst performing tests
        worst_tests = vuln_df.head(3)

        recommendations.append("\n**Key Vulnerabilities:**")
        for _, row in worst_tests.iterrows():
            if row["Performance Drop (%)"] > 20:
                recommendations.append(
                    f"- **{row['Stress Test']}**: {row['Performance Drop (%)']:.1f}% drop ({row['Severity']} severity)"
                )

        # Specific recommendations
        high_severity = vuln_df[vuln_df["Severity"].isin(["High", "Critical"])]

        if len(high_severity) > 0:
            recommendations.append("\n**Recommended Actions:**")

            # Check for noise sensitivity
            noise_tests = high_severity[
                high_severity["Stress Test"].str.contains("noise", case=False)
            ]
            if len(noise_tests) > 0:
                recommendations.append(
                    "- **Add noise augmentation** to training data to improve noise robustness"
                )

            # Check for dropout sensitivity
            dropout_tests = high_severity[
                high_severity["Stress Test"].str.contains("dropout", case=False)
            ]
            if len(dropout_tests) > 0:
                recommendations.append(
                    "- **Feature engineering**: Some features may be too critical. Add redundant features."
                )
                recommendations.append(
                    "- Consider **dropout regularization** during training"
                )

            # Check for corruption sensitivity
            corruption_tests = high_severity[
                high_severity["Stress Test"].str.contains("corruption", case=False)
            ]
            if len(corruption_tests) > 0:
                recommendations.append(
                    "- **Outlier handling**: Improve robustness to corrupted values"
                )
                recommendations.append(
                    "- Add **data validation** in production pipeline"
                )

            # Check for scale sensitivity
            scale_tests = high_severity[
                high_severity["Stress Test"].str.contains("scale", case=False)
            ]
            if len(scale_tests) > 0:
                recommendations.append(
                    "- **Feature scaling**: Ensure consistent scaling in production"
                )
                recommendations.append("- Consider **scale-invariant features**")

        # Prediction stability
        avg_agreement = np.mean(
            [
                r["prediction_agreement"]
                for r in self.stress_results[model_name].values()
            ]
        )

        if avg_agreement < 0.8:
            recommendations.append(
                f"\n- **Low prediction stability** ({avg_agreement:.1%} agreement). Model predictions are inconsistent under stress."
            )
            recommendations.append(
                "  - Consider ensemble methods for more stable predictions"
            )
            recommendations.append("  - Increase model capacity or regularization")

        return recommendations
