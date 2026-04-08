"""
Baseline Modeling Module
Purpose: Train and evaluate ML models on clean data
Models: Logistic Regression, Random Forest, XGBoost
"""

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


class ModelTrainer:
    """Manages training and evaluation of baseline ML models"""

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        self.best_model = None

    def get_model(self, model_name: str, **params):
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
            raise ValueError(f"Unknown model: {model_name}")

    def train_model(self, model_name: str, X_train, y_train, **params):
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

    def predict(self, model_name: str, X):
        """
        Make predictions

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        return predictions, probabilities

    def evaluate_model(self, model_name: str, X_test, y_test, dataset_name="Test"):
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

    def get_classification_report(self, model_name: str, dataset_name="Test"):
        """
        Get detailed classification report

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Classification report as dict
        """
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            return None

        metrics = self.metrics[model_name][dataset_name]
        y_true = metrics["true_labels"]
        y_pred = metrics["predictions"]

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        return report

    def plot_confusion_matrix(self, model_name: str, dataset_name="Test"):
        """
        Create interactive confusion matrix plot

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Plotly figure
        """
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            return None

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

    def plot_metrics_comparison(self, dataset_name="Test"):
        """
        Compare metrics across all trained models

        Args:
            dataset_name: Dataset to compare

        Returns:
            Plotly figure
        """
        if not self.metrics:
            return None

        models = []
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for model_name in self.metrics:
            if dataset_name in self.metrics[model_name]:
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

    def get_feature_importance(self, model_name: str, feature_names):
        """
        Get feature importance for tree-based models

        Args:
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.trained_models:
            return None

        model = self.trained_models[model_name]

        # Check if model has feature_importances_
        if not hasattr(model, "feature_importances_"):
            return None

        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        return importance_df

    def plot_feature_importance(self, model_name: str, feature_names, top_n=10):
        """
        Plot feature importance

        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to display

        Returns:
            Plotly figure
        """
        importance_df = self.get_feature_importance(model_name, feature_names)

        if importance_df is None:
            return None

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

    def save_model(self, model_name: str, filepath: str):
        """
        Save trained model to disk

        Args:
            model_name: Name of the model
            filepath: Path to save model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.trained_models[model_name], f)

        st.success(f"✅ Model saved to {filepath}")

    def load_model(self, model_name: str, filepath: str):
        """
        Load trained model from disk

        Args:
            model_name: Name to assign to the model
            filepath: Path to load model from
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        self.trained_models[model_name] = model
        st.success(f"✅ Model loaded from {filepath}")

    def get_best_model(self, metric="accuracy", dataset_name="Test"):
        """
        Get the best performing model based on a metric

        Args:
            metric: Metric to use for comparison
            dataset_name: Dataset to evaluate on

        Returns:
            Tuple of (model_name, score)
        """
        best_score = -1
        best_model = None

        for model_name in self.metrics:
            if dataset_name in self.metrics[model_name]:
                score = self.metrics[model_name][dataset_name][metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name

        self.best_model = best_model
        return best_model, best_score

    def get_probability_distribution(self, model_name: str, dataset_name="Test"):
        """
        Get probability distribution for predictions

        Args:
            model_name: Name of the model
            dataset_name: Dataset name

        Returns:
            Probabilities array
        """
        if (
            model_name not in self.metrics
            or dataset_name not in self.metrics[model_name]
        ):
            return None

        return self.metrics[model_name][dataset_name]["probabilities"]

    def get_model_summary(self):
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
