from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import get_config
from core.errors import FrameworkError
from core.model_io import ModelArtifactStore
from core.state import AppContext, StateKeys
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from views.reporters import render_error, render_errors


def render(ctx: AppContext) -> None:
    """Render the Baseline Modeling page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    model_trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)

    st.header("2️⃣ Baseline Modeling Module")
    st.markdown("**Purpose:** Train and evaluate ML models on clean data")

    # Check if data is prepared
    if not ctx.store[StateKeys.DATA_PREPARED]:
        st.warning("⚠️ Please prepare your data first in Module 1!")
        st.info("Go to **1️⃣ Data Management** → **Train/Val/Test Split** to prepare your data")
    else:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🎯 Model Selection & Training",
                "📊 Model Evaluation",
                "📈 Feature Importance",
                "💾 Save/Load Models",
            ]
        )

        # TAB 1: Model Selection & Training
        with tab1:
            st.subheader("🎯 Model Selection & Training")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**Select Model:**")
                model_name = st.selectbox(
                    "Choose a model to train:",
                    ["Logistic Regression", "Random Forest", "XGBoost"],
                )

            with col2:
                st.write("**Dataset Info:**")
                data = data_manager.get_data()
                st.metric("Training Samples", len(data["X_train"]))
                st.metric("Validation Samples", len(data["X_val"]))
                st.metric("Test Samples", len(data["X_test"]))

            st.markdown("---")

            # Model-specific parameters
            st.write(f"**{model_name} Parameters:**")

            params = {}

            if model_name == "Logistic Regression":
                col1, col2 = st.columns(2)
                with col1:
                    params["C"] = st.slider("Regularization (C)", 0.001, 10.0, 1.0, 0.1)
                with col2:
                    params["max_iter"] = st.slider("Max Iterations", 100, 5000, 1000, 100)

            elif model_name == "Random Forest":
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100, 10)
                with col2:
                    max_depth_none = st.checkbox("No Max Depth", value=True)
                    if not max_depth_none:
                        params["max_depth"] = st.slider("Max Depth", 1, 50, 10)
                    else:
                        params["max_depth"] = None
                with col3:
                    params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)

            elif model_name == "XGBoost":
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100, 10)
                with col2:
                    params["max_depth"] = st.slider("Max Depth", 1, 15, 6)
                with col3:
                    params["learning_rate"] = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)

            params["random_state"] = 42

            st.markdown("---")

            # Train button
            if st.button(f"🚀 Train {model_name}", type="primary"):
                with render_errors():
                    # conventions §7: every mutating action reports success **or** a rendered
                    # ``FrameworkError``.  The guard wraps the whole handler, so a failed train
                    # cannot leave ``ctx.store[StateKeys.MODEL_TRAINED]`` set, and cannot fall
                    # through into display code reading metrics that were never computed.
                    with st.spinner(f"Training {model_name}..."):
                        # Get data
                        data = data_manager.get_data()
                        X_train = data["X_train"]
                        y_train = data["y_train"]
                        X_val = data["X_val"]
                        y_val = data["y_val"]
                        X_test = data["X_test"]
                        y_test = data["y_test"]

                        # Train model.  The trained estimator is not needed here:
                        # evaluate_model/feature_importance look it up by name out
                        # of ModelTrainer, so the call's side effect is the result.
                        model_trainer.train_model(model_name, X_train, y_train, **params)

                        # Evaluate on all sets
                        train_metrics = model_trainer.evaluate_model(
                            model_name, X_train, y_train, "Train"
                        )
                        val_metrics = model_trainer.evaluate_model(
                            model_name, X_val, y_val, "Validation"
                        )
                        test_metrics = model_trainer.evaluate_model(model_name, X_test, y_test, "Test")

                        st.success(f"✅ {model_name} trained successfully!")

                        # Display quick results
                        st.subheader("📊 Quick Results")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write("**Training Set**")
                            st.metric("Accuracy", f"{train_metrics['accuracy']:.4f}")
                            st.metric("F1-Score", f"{train_metrics['f1']:.4f}")

                        with col2:
                            st.write("**Validation Set**")
                            st.metric("Accuracy", f"{val_metrics['accuracy']:.4f}")
                            st.metric("F1-Score", f"{val_metrics['f1']:.4f}")

                        with col3:
                            st.write("**Test Set**")
                            st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
                            st.metric("F1-Score", f"{test_metrics['f1']:.4f}")

                        ctx.store[StateKeys.MODEL_TRAINED] = True

            # Show trained models
            if model_trainer.trained_models:
                st.markdown("---")
                st.write("**Trained Models:**")
                for trained_model in model_trainer.trained_models.keys():
                    st.success(f"✅ {trained_model}")

        # TAB 2: Model Evaluation
        with tab2:
            st.subheader("📊 Model Evaluation")

            if not model_trainer.trained_models:
                st.info("👈 Train a model first in the 'Model Selection & Training' tab")
            else:
                # Model selector
                eval_model = st.selectbox(
                    "Select model to evaluate:",
                    list(model_trainer.trained_models.keys()),
                )

                dataset_eval = st.radio(
                    "Select dataset:", ["Train", "Validation", "Test"], horizontal=True
                )

                st.markdown("---")

                # Get metrics
                metrics = model_trainer.metrics.get(eval_model, {}).get(dataset_eval)

                if metrics:
                    # Performance Metrics
                    st.subheader("🎯 Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1']:.4f}")

                    st.markdown("---")

                    # Confusion Matrix
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("📉 Confusion Matrix")
                        cm_fig = model_trainer.plot_confusion_matrix(eval_model, dataset_eval)
                        if cm_fig:
                            st.plotly_chart(cm_fig, width="stretch")

                    with col2:
                        st.subheader("📋 Classification Report")
                        report = model_trainer.get_classification_report(eval_model, dataset_eval)
                        if report:
                            # Display as dataframe
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, width="stretch")

                    st.markdown("---")

                    # Model Comparison
                    st.subheader("📊 Model Comparison")
                    comparison_fig = model_trainer.plot_metrics_comparison(dataset_eval)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, width="stretch")

                    # Summary Table
                    st.subheader("📑 All Models Summary")
                    summary_df = model_trainer.get_model_summary()
                    if not summary_df.empty:
                        st.dataframe(summary_df, width="stretch")
                else:
                    st.warning(f"No evaluation results for {eval_model} on {dataset_eval} set")

        # TAB 3: Feature Importance
        with tab3:
            st.subheader("📈 Feature Importance Analysis")

            if not model_trainer.trained_models:
                st.info("👈 Train a model first in the 'Model Selection & Training' tab")
            else:
                model_fi = st.selectbox(
                    "Select model:",
                    list(model_trainer.trained_models.keys()),
                    key="fi_model",
                )

                # Get feature names
                data = data_manager.get_data()
                feature_names = data["feature_columns"]

                if model_fi in ["Random Forest", "XGBoost"]:
                    top_n = st.slider("Number of top features to display", 5, 20, 10)

                    # Plot feature importance
                    fi_fig = model_trainer.plot_feature_importance(model_fi, feature_names, top_n)

                    if fi_fig:
                        st.plotly_chart(fi_fig, width="stretch")

                        # Show table
                        st.subheader("📊 Feature Importance Table")
                        fi_df = model_trainer.get_feature_importance(model_fi, feature_names)
                        st.dataframe(fi_df, width="stretch")
                else:
                    st.info(f"❌ Feature importance not available for {model_fi}")
                    st.write("Currently supported for: Random Forest, XGBoost")

        # TAB 4: Save/Load Models
        with tab4:
            st.subheader("💾 Save/Load Models")

            col1, col2 = st.columns(2)

            # Wave 1 / ADR-001: persistence is contained by ModelArtifactStore, and
            # every artifact is SHA-256 attested on save.  Listing therefore reads
            # the attestation manifest rather than the filesystem, so an untrusted
            # or tampered file is visibly distinguishable from an attested one.
            saved_records = ModelArtifactStore(get_config().artifacts).list_artifacts()

            with col1:
                st.write("**Save Model**")
                if model_trainer.trained_models:
                    save_model = st.selectbox(
                        "Select model to save:",
                        list(model_trainer.trained_models.keys()),
                        key="save_model",
                    )

                    filename = st.text_input(
                        "Filename:", value=f"{save_model.lower().replace(' ', '_')}.pkl"
                    )

                    if st.button("💾 Save Model"):
                        # The store sanitises the filename and asserts containment,
                        # so the UI no longer joins user input onto a path.
                        try:
                            model_trainer.save_model(save_model, filename)
                        except FrameworkError as exc:
                            render_error(exc)
                else:
                    st.info("No trained models to save")

            with col2:
                st.write("**Load Model**")

                if saved_records:
                    selected_file = st.selectbox(
                        "Select model file:",
                        [record.name for record in saved_records],
                        key="load_model_file",
                    )

                    model_name_input = st.text_input(
                        "Model name:",
                        value=selected_file.replace(".pkl", "").replace("_", " ").title(),
                        key="load_model_name",
                    )

                    # ADR-001: trust is only ever supplied from an explicit,
                    # default-unchecked confirmation stating the file will execute
                    # arbitrary Python code.
                    trust_untrusted = st.checkbox(
                        "⚠️ I understand this file executes arbitrary Python code on "
                        "my machine (only tick this for files I created myself)",
                        value=False,
                        key="load_model_trust",
                    )

                    if st.button("📂 Load Model"):
                        try:
                            model_trainer.load_model(
                                model_name_input, selected_file, trust=trust_untrusted
                            )
                            ctx.store[StateKeys.MODEL_TRAINED] = True

                            # Evaluate on all datasets
                            data = data_manager.get_data()
                            X_train = data["X_train"]
                            y_train = data["y_train"]
                            X_val = data["X_val"]
                            y_val = data["y_val"]
                            X_test = data["X_test"]
                            y_test = data["y_test"]

                            with st.spinner("Evaluating loaded model..."):
                                model_trainer.evaluate_model(
                                    model_name_input, X_train, y_train, "Train"
                                )
                                model_trainer.evaluate_model(
                                    model_name_input, X_val, y_val, "Validation"
                                )
                                model_trainer.evaluate_model(
                                    model_name_input, X_test, y_test, "Test"
                                )

                            st.success(f"✅ Model '{model_name_input}' loaded and evaluated!")
                        except FrameworkError as exc:
                            # render_error surfaces ArtifactError subclasses distinctly
                            # so the user can tell a provenance failure from a
                            # formatting failure (architecture.md §5).
                            render_error(exc)
                else:
                    st.info("No attested models found in 'saved_models'. Save a model first!")

            st.markdown("---")

            # Best Model
            if model_trainer.metrics:
                st.subheader("🏆 Best Model")

                metric_choice = st.selectbox(
                    "Select metric for comparison:",
                    ["accuracy", "precision", "recall", "f1"],
                )

                dataset_choice = st.selectbox(
                    "Select dataset:",
                    ["Train", "Validation", "Test"],
                    index=2,
                    key="best_model_dataset",
                )

                best_model, best_score = model_trainer.get_best_model(metric_choice, dataset_choice)

                if best_model:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Best Model:** {best_model}")
                    with col2:
                        st.metric(f"Best {metric_choice.capitalize()}", f"{best_score:.4f}")


__all__ = ["render"]
