"""
Baseline Modeling Module
Purpose: Train and evaluate ML models on clean data
Models: Logistic Regression, Random Forest, XGBoost
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from core.config import get_config
from core.errors import (
    FrameworkError,
    ModelNotTrainedError,
    ModelPredictionError,
    UnknownModelError,
    ValidationError,
)
from core.model_io import ArtifactRecord, ModelArtifactStore

logger = logging.getLogger(__name__)

LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({
    "loss",
    "log_loss",
    "ece",
    "brier_score",
    "brier",
    "mse",
    "mae",
    "rmse",
    "error",
    "error_rate",
    "drop",
    "performance_drop",
})


class ModelTrainer:
    """Manages training and evaluation of baseline ML models"""

    def __init__(self, reporter: Any | None = None) -> None:
        self.models = {}
        self.trained_models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        self.best_model = None
        # Wave 1/4: the store is the only place persistence happens, and the
        # reporter replaces the previous st.success calls (ADR-001, ADR-002).
        self.reporter = reporter
        self._store = ModelArtifactStore(get_config().artifacts)

    def get_model(self, model_name: str, **params: Any) -> Any:
        """
        Get model instance with specified parameters

        Args:
            model_name: Name of the model
            **params: Model-specific parameters

        Returns:
            Model instance
        """
        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 1.0),
                random_state=params.get("random_state", 42),
            )
        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=params.get("random_state", 42),
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )
        else:
            raise UnknownModelError(
                f"Unknown model {model_name!r}. Accepted models are: "
                "'Logistic Regression', 'Random Forest', 'XGBoost'.",
                context={"model_name": model_name},
            )

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **params: Any,
    ) -> Any:
        """
        Train a model

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            **params: Model parameters

        Returns:
            Trained model
        """
        model = self.get_model(model_name, **params)
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def predict(
        self, model_name: str, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained yet; train it in "
                "Module 2 before predicting.",
                context={"model_name": model_name},
            )

        model = self.trained_models[model_name]
        try:
            predictions = model.predict(X)
        except Exception as exc:  # pragma: no cover - sklearn errors are domain-bound
            raise ModelPredictionError(
                f"Model {model_name!r} cannot predict; ensure it was fitted "
                "on the same feature shape.",
                context={"model_name": model_name},
            ) from exc
        try:
            probabilities = model.predict_proba(X)
        except Exception as exc:  # pragma: no cover - sklearn errors are domain-bound
            raise ModelPredictionError(
                f"Model {model_name!r} cannot produce probabilities; check that "
                "the estimator supports predict_proba.",
                context={"model_name": model_name},
            ) from exc

        return predictions, probabilities

    def supports_probabilities(self, model_name: str) -> bool:
        """Return True if model is trained and its estimator supports predict_proba.

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        model = self.trained_models[model_name]
        return hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba", None))

    def get_classes(self, model_name: str) -> np.ndarray | None:
        """Return classes_ array for the named model, or None if classes_ is absent.

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        model = self.trained_models[model_name]
        classes = getattr(model, "classes_", None)
        return np.asarray(classes) if classes is not None else None

    def get_class_names(self, model_name: str) -> list[str] | None:
        """Return string class labels for the named model, or None if classes_ is absent.

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
        """
        classes = self.get_classes(model_name)
        if classes is None:
            return None
        return [str(c) for c in classes]

    def supports_feature_importance(self, model_name: str) -> bool:
        """Return True if model is trained and exposes feature_importances_.

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        model = self.trained_models[model_name]
        return hasattr(model, "feature_importances_")

    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str = "Test",
    ) -> dict[str, Any]:
        """
        Evaluate model performance

        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of dataset (Train/Val/Test)

        Returns:
            Dictionary of metrics
        """
        predictions, probabilities = self.predict(model_name, X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "predictions": predictions,
            "probabilities": probabilities,
            "true_labels": y_test,
        }

        # Store predictions and probabilities
        self.predictions[f"{model_name}_{dataset_name}"] = predictions
        self.probabilities[f"{model_name}_{dataset_name}"] = probabilities

        # Store metrics
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        self.metrics[model_name][dataset_name] = metrics

        return metrics

    def get_classification_report(
        self, model_name: str, dataset_name: str = "Test"
    ) -> dict[str, Any]:
        """
        Get detailed classification report

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Classification report as dict

        Raises:
            ModelNotTrainedError: If model or dataset evaluation metrics are not found.
        """
        if model_name not in self.trained_models and model_name not in self.metrics:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            raise ModelNotTrainedError(
                f"No evaluated metrics found for model {model_name!r} on dataset {dataset_name!r}.",
                context={"model_name": model_name, "dataset_name": dataset_name},
            )

        metrics = self.metrics[model_name][dataset_name]
        y_true = metrics["true_labels"]
        y_pred = metrics["predictions"]

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        return report

    def plot_confusion_matrix(
        self, model_name: str, dataset_name: str = "Test"
    ) -> go.Figure:
        """
        Create interactive confusion matrix plot

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Plotly figure

        Raises:
            ModelNotTrainedError: If model or dataset evaluation metrics are not found.
        """
        if model_name not in self.trained_models and model_name not in self.metrics:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            raise ModelNotTrainedError(
                f"No evaluated metrics found for model {model_name!r} on dataset {dataset_name!r}.",
                context={"model_name": model_name, "dataset_name": dataset_name},
            )

        cm = self.metrics[model_name][dataset_name]["confusion_matrix"]

        # Create labels
        labels = [f"Class {i}" for i in range(len(cm))]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=True,
            )
        )

        fig.update_layout(
            title=f"Confusion Matrix - {model_name} ({dataset_name})",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500,
        )

        return fig

    def plot_metrics_comparison(
        self, dataset_name: str = "Test"
    ) -> go.Figure:
        """
        Compare metrics across all trained models

        Args:
            dataset_name: Dataset to compare

        Returns:
            Plotly figure

        Raises:
            ModelNotTrainedError: If no models have been evaluated on dataset_name.
        """
        if not self.metrics:
            raise ModelNotTrainedError(
                "No trained models with evaluation metrics found.",
                context={"dataset_name": dataset_name},
            )

        matching_models = [
            m for m in self.metrics if dataset_name in self.metrics[m]
        ]
        if not matching_models:
            raise ModelNotTrainedError(
                f"No evaluated metrics found for dataset {dataset_name!r}.",
                context={"dataset_name": dataset_name},
            )

        models = []
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for model_name in matching_models:
            models.append(model_name)
            metrics = self.metrics[model_name][dataset_name]
            accuracy.append(metrics["accuracy"])
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])
            f1.append(metrics["f1"])

        # Create grouped bar chart
        fig = go.Figure(
            data=[
                go.Bar(name="Accuracy", x=models, y=accuracy),
                go.Bar(name="Precision", x=models, y=precision),
                go.Bar(name="Recall", x=models, y=recall),
                go.Bar(name="F1-Score", x=models, y=f1),
            ]
        )

        fig.update_layout(
            title=f"Model Performance Comparison ({dataset_name} Set)",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            yaxis_range=[0, 1],
            width=800,
            height=500,
        )

        return fig

    def get_feature_importance(
        self, model_name: str, feature_names: list[str]
    ) -> pd.DataFrame:
        """
        Get feature importance for tree-based models

        Args:
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
            ValidationError: If model does not support feature importances or feature lengths mismatch.
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )

        model = self.trained_models[model_name]

        # Check if model has feature_importances_
        if not hasattr(model, "feature_importances_"):
            raise ValidationError(
                f"Model {model_name!r} does not support feature importances.",
                field="model_name",
                value=model_name,
                context={"model_name": model_name},
            )

        importances = model.feature_importances_
        if len(feature_names) != len(importances):
            raise ValidationError(
                f"Length of feature_names ({len(feature_names)}) does not match "
                f"model feature importances ({len(importances)}).",
                field="feature_names",
                context={
                    "feature_names_len": len(feature_names),
                    "importances_len": len(importances),
                },
            )

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        return importance_df

    def plot_feature_importance(
        self, model_name: str, feature_names: list[str], top_n: int = 10
    ) -> go.Figure:
        """
        Plot feature importance

        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to display

        Returns:
            Plotly figure

        Raises:
            ModelNotTrainedError: If model_name has not been trained.
            ValidationError: If top_n is invalid or model does not support feature importances.
        """
        if top_n <= 0:
            raise ValidationError(
                f"top_n must be a positive integer, got {top_n}.",
                field="top_n",
                value=top_n,
                context={"top_n": top_n},
            )

        importance_df = self.get_feature_importance(model_name, feature_names)

        # Get top N features
        top_features = importance_df.head(top_n)

        fig = go.Figure(
            go.Bar(
                x=top_features["Importance"],
                y=top_features["Feature"],
                orientation="h",
                marker_color="lightblue",
            )
        )

        fig.update_layout(
            title=f"Top {top_n} Feature Importances - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            yaxis={"categoryorder": "total ascending"},
        )

        return fig

    def save_model(self, model_name: str, filepath: str) -> ArtifactRecord:
        """
        Save a trained model through the attested artifact store

        All persistence is delegated to ``core.model_io.ModelArtifactStore``
        (ADR-001).  The raw ``filepath`` is reduced to its basename so a caller
        cannot escape the store root, and the previous
        ``os.makedirs(os.path.dirname(filepath))`` call is gone — it raised
        ``FileNotFoundError`` for a bare filename, which is the first-save path
        in a fresh checkout.

        Args:
            model_name: Name of the model
            filepath: Destination path; only its filename is honoured

        Returns:
            ArtifactRecord: The attestation record written by the store

        Raises:
            ModelNotTrainedError: If the model has not been trained
            ArtifactPathError: If the filename fails sanitisation or containment
        """
        if model_name not in self.trained_models:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained yet; nothing to "
                "save.",
                context={"model_name": model_name},
            )

        try:
            record = self._store.save(
                self.trained_models[model_name],
                Path(filepath).name,
                model_name=model_name,
            )
        except Exception as exc:
            from core.errors import ArtifactError

            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(
                f"Could not save model {model_name!r} to {Path(filepath).name}.",
                context={"model_name": model_name},
            ) from exc
        logger.info(
            "Saved model %s as %s (%d bytes)",
            model_name,
            record.name,
            record.size_bytes,
        )
        if self.reporter is not None:
            try:
                self.reporter.success(f"✅ Model saved to {record.name}")
            except FrameworkError as exc:
                # The save already succeeded; only the confirmation failed.  Log
                # it (conventions §3) instead of swallowing it silently; anything
                # that is not a FrameworkError is a defect and still propagates.
                logger.warning(
                    "Could not report the save of model %r: %s", model_name, exc
                )
        return record

    def load_model(
        self, model_name: str, filepath: str, *, trust: bool = False
    ) -> ArtifactRecord:
        """
        Load a trained model through the attested artifact store

        Args:
            model_name: Name to assign to the model
            filepath: Path to load model from; only its filename is honoured
            trust: Opt in to loading an unattested or tampered file.  The UI
                only sets this from an explicit, default-unchecked confirmation
                stating that the file executes arbitrary Python code.

        Returns:
            ArtifactRecord: The attestation record read by the store

        Raises:
            ArtifactNotFoundError: No such artifact
            ArtifactIntegrityError: Hash mismatch, size cap, or wrong type
            ArtifactUntrustedError: Unattested file and ``trust`` not set
        """
        try:
            model, record = self._store.load(Path(filepath).name, trust=trust)
        except Exception as exc:
            from core.errors import ArtifactError

            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(
                f"Could not load model from {Path(filepath).name}.",
                context={"model_name": model_name},
            ) from exc
        self.trained_models[model_name] = model
        logger.info(
            "Loaded model %s from %s%s",
            model_name,
            record.name,
            " (unattested, trust=True)" if trust and not record.created_at else "",
        )
        if self.reporter is not None:
            try:
                self.reporter.success(f"✅ Model loaded from {record.name}")
            except FrameworkError as exc:
                # As above: the load succeeded, so a failed confirmation is
                # logged, not swallowed, and non-FrameworkError defects surface.
                logger.warning(
                    "Could not report the load of model %r: %s", model_name, exc
                )
        return record

    def list_saved_models(self) -> list[ArtifactRecord]:
        """Return the attested artifact records available for loading.

        Returns:
            list[ArtifactRecord]: Records in manifest insertion order.
        """
        return self._store.list_artifacts()

    def delete_saved_model(self, filepath: str) -> None:
        """Delete an artifact and its manifest entry.

        Args:
            filepath: Path whose filename identifies the artifact.
        """
        self._store.delete(Path(filepath).name)

    def get_best_model(
        self,
        metric: str = "Accuracy",
        dataset_name: str = "Test",
        *,
        lower_is_better: bool | None = None,
    ) -> tuple[str, float]:
        """Identify the best-performing model for a given metric and dataset.

        Args:
            metric: Evaluation metric name (e.g. "Accuracy", "loss", "ECE").
            dataset_name: Dataset split ("Train", "Validation", "Test").
            lower_is_better: Whether lower values represent better performance.
                If None, inferred from :data:`LOWER_IS_BETTER_METRICS`.

        Returns:
            tuple[str, float]: (best_model_name, best_score).

        Raises:
            ModelNotTrainedError: If no models have been evaluated or no metrics exist
                for dataset_name.
            ValidationError: If metric is not found in the evaluated metrics.
        """
        if not self.metrics:
            raise ModelNotTrainedError(
                "No trained models with evaluation metrics found.",
                context={"dataset_name": dataset_name, "metric": metric},
            )

        matching_models = {
            name: m[dataset_name]
            for name, m in self.metrics.items()
            if dataset_name in m
        }
        if not matching_models:
            raise ModelNotTrainedError(
                f"No evaluated metrics found for dataset {dataset_name!r}.",
                context={"dataset_name": dataset_name, "metric": metric},
            )

        # Determine metric polarity
        if lower_is_better is None:
            metric_clean = metric.strip().lower().replace(" ", "_").replace("-", "_")
            invert = (
                metric.strip().lower() in LOWER_IS_BETTER_METRICS
                or metric_clean in LOWER_IS_BETTER_METRICS
            )
        else:
            invert = bool(lower_is_better)

        best_score = float("inf") if invert else float("-inf")
        best_model: str | None = None

        for name, dmetrics in matching_models.items():
            metric_key = None
            if metric in dmetrics:
                metric_key = metric
            elif metric.lower() in dmetrics:
                metric_key = metric.lower()
            elif metric.title() in dmetrics:
                metric_key = metric.title()

            if metric_key is None:
                raise ValidationError(
                    f"Metric {metric!r} is not a valid evaluation metric.",
                    field="metric",
                    value=metric,
                    context={"metric": metric},
                )

            score = float(dmetrics[metric_key])
            if best_model is None:
                best_score = score
                best_model = name
            elif math.isnan(best_score) and not math.isnan(score):
                best_score = score
                best_model = name
            elif not math.isnan(score):
                if invert:
                    if score < best_score:
                        best_score = score
                        best_model = name
                else:
                    if score > best_score:
                        best_score = score
                        best_model = name

        if best_model is None:  # pragma: no cover - guarded by matching_models check
            raise ModelNotTrainedError(
                f"No evaluated metrics found for dataset {dataset_name!r}.",
                context={"dataset_name": dataset_name, "metric": metric},
            )

        self.best_model = best_model
        return best_model, best_score

    def get_probability_distribution(
        self, model_name: str, dataset_name: str = "Test"
    ) -> np.ndarray:
        """
        Get probability distribution for predictions

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Probabilities array

        Raises:
            ModelNotTrainedError: If model or dataset evaluation metrics are not found.
            ModelPredictionError: If probability predictions are absent.
        """
        if model_name not in self.trained_models and model_name not in self.metrics:
            raise ModelNotTrainedError(
                f"Model {model_name!r} has not been trained.",
                context={"model_name": model_name},
            )
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            raise ModelNotTrainedError(
                f"No evaluated metrics found for model {model_name!r} on dataset {dataset_name!r}.",
                context={"model_name": model_name, "dataset_name": dataset_name},
            )

        probs = self.metrics[model_name][dataset_name].get("probabilities")
        if probs is None:
            raise ModelPredictionError(
                f"Model {model_name!r} does not have probability predictions for dataset {dataset_name!r}.",
                context={"model_name": model_name, "dataset_name": dataset_name},
            )
        return probs

    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models

        Returns:
            DataFrame with model summary
        """
        summary_data = []

        for model_name in self.metrics:
            for dataset in self.metrics[model_name]:
                metrics = self.metrics[model_name][dataset]
                summary_data.append(
                    {
                        "Model": model_name,
                        "Dataset": dataset,
                        "Accuracy": f"{metrics['accuracy']:.4f}",
                        "Precision": f"{metrics['precision']:.4f}",
                        "Recall": f"{metrics['recall']:.4f}",
                        "F1-Score": f"{metrics['f1']:.4f}",
                    }
                )

        return pd.DataFrame(summary_data)
