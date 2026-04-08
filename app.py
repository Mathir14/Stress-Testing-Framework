"""
ML Model Reliability & Stress Testing Framework
Main Streamlit Application
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.calibration_module import CalibrationAnalyzer
from modules.comparison_module import ModelComparator
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from modules.post_stress_module import PostStressAnalyzer
from modules.reliability_module import ReliabilityScorer
from modules.reporting_module import ReportGenerator
from modules.stress_module import StressTester
from utils.metrics import (
    calculate_brier_score,
    get_confidence_scores,
    get_prediction_entropy,
    identify_high_confidence_errors,
)
from utils.plotting import (
    plot_confidence_accuracy_curve,
    plot_confidence_by_class,
    plot_confidence_histogram,
    plot_entropy_distribution,
    plot_error_analysis,
)

# Page configuration
st.set_page_config(
    page_title="ML Reliability Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "model_trainer" not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if "post_stress_analyzer" not in st.session_state:
    st.session_state.post_stress_analyzer = PostStressAnalyzer()
if "calibration_analyzer" not in st.session_state:
    st.session_state.calibration_analyzer = CalibrationAnalyzer()
if "model_comparator" not in st.session_state:
    st.session_state.model_comparator = ModelComparator()
if "reliability_scorer" not in st.session_state:
    st.session_state.reliability_scorer = ReliabilityScorer()
if "report_generator" not in st.session_state:
    st.session_state.report_generator = ReportGenerator()
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# Title and description
st.title("🔬 ML Model Reliability & Stress Testing Framework")
st.markdown(
    "### A comprehensive framework for testing ML model robustness and reliability"
)

# Sidebar navigation
st.sidebar.title("📋 Navigation")
module = st.sidebar.radio(
    "Select Module:",
    [
        "1️⃣ Data Management",
        "2️⃣ Baseline Modeling",
        "3️⃣ Prediction & Confidence",
        "4️⃣ Stress Testing",
        "5️⃣ Post-Stress Evaluation",
        "6️⃣ Calibration Analysis",
        "7️⃣ Model Comparison",
        "8️⃣ Reliability Scoring",
        "9️⃣ Visualization & Reports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Complete modules in order for best results")

# ==================== MODULE 1: DATA MANAGEMENT ====================
if module == "1️⃣ Data Management":
    st.header("1️⃣ Data Management Module")
    st.markdown("**Purpose:** Prepare dataset for modeling")

    # Create tabs for different data operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📁 Upload Data",
            "🔍 Validation",
            "🧹 Cleaning",
            "🔄 Encoding & Scaling",
            "✂️ Train/Val/Test Split",
        ]
    )

    # TAB 1: Upload Data
    with tab1:
        st.subheader("📁 Dataset Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            if st.button("Load Dataset"):
                data = st.session_state.data_manager.load_dataset(uploaded_file)
                if data is not None:
                    st.session_state.data_loaded = True
                    st.session_state.data_manager.display_summary(data)

        if hasattr(st.session_state, "data_loaded") and st.session_state.data_loaded:
            if st.session_state.data_manager.raw_data is not None:
                st.success("✅ Dataset is loaded and ready!")

    # TAB 2: Validation
    with tab2:
        st.subheader("🔍 Data Validation")

        if st.session_state.data_manager.raw_data is not None:
            df = st.session_state.data_manager.raw_data

            if st.button("Run Validation"):
                validation_report = st.session_state.data_manager.validate_dataset(df)

                # Display validation results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Rows", validation_report["shape"][0])
                    st.metric("Total Columns", validation_report["shape"][1])

                with col2:
                    st.metric(
                        "Numeric Columns", len(validation_report["numeric_columns"])
                    )
                    st.metric(
                        "Categorical Columns",
                        len(validation_report["categorical_columns"]),
                    )

                with col3:
                    st.metric("Duplicate Rows", validation_report["duplicates"])
                    st.metric(
                        "Memory Usage (MB)", f"{validation_report['memory_usage']:.2f}"
                    )

                # Missing values details
                st.subheader("Missing Values Analysis")
                missing_df = pd.DataFrame(
                    {
                        "Column": validation_report["missing_values"].keys(),
                        "Missing Count": validation_report["missing_values"].values(),
                        "Missing %": [
                            f"{v:.2f}%"
                            for v in validation_report["missing_percentage"].values()
                        ],
                    }
                )
                missing_df = missing_df[missing_df["Missing Count"] > 0]

                if len(missing_df) > 0:
                    st.warning("⚠️ Columns with missing values:")
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing values detected!")

                # Column types
                st.subheader("Column Types")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Numeric Columns:**")
                    st.write(validation_report["numeric_columns"])
                with col2:
                    st.write("**Categorical Columns:**")
                    st.write(validation_report["categorical_columns"])
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 3: Cleaning
    with tab3:
        st.subheader("🧹 Data Cleaning")

        if st.session_state.data_manager.raw_data is not None:
            df = st.session_state.data_manager.raw_data

            # Handle duplicates
            st.write("**Remove Duplicates**")
            if df.duplicated().sum() > 0:
                if st.button("Remove Duplicate Rows"):
                    df = df.drop_duplicates()
                    st.session_state.data_manager.raw_data = df
                    st.success(f"✅ Removed duplicates! New shape: {df.shape}")
            else:
                st.info("No duplicates found")

            st.markdown("---")

            # Handle missing values
            st.write("**Handle Missing Values**")
            missing_cols = df.columns[df.isnull().any()].tolist()

            if missing_cols:
                st.write(f"Columns with missing values: {len(missing_cols)}")

                strategy = {}
                for col in missing_cols:
                    st.write(f"**{col}** ({df[col].dtype})")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        if df[col].dtype in ["int64", "float64"]:
                            method = st.selectbox(
                                f"Strategy for {col}",
                                [
                                    "mean",
                                    "median",
                                    "drop",
                                    "forward_fill",
                                    "backward_fill",
                                ],
                                key=f"missing_{col}",
                            )
                        else:
                            method = st.selectbox(
                                f"Strategy for {col}",
                                ["mode", "drop", "forward_fill", "backward_fill"],
                                key=f"missing_{col}",
                            )

                    with col2:
                        st.metric("Missing", df[col].isnull().sum())

                    strategy[col] = method

                if st.button("Apply Missing Value Handling"):
                    df_cleaned = st.session_state.data_manager.handle_missing_values(
                        df, strategy
                    )
                    st.session_state.data_manager.raw_data = df_cleaned
                    st.success("✅ Missing values handled successfully!")
                    st.rerun()
            else:
                st.success("✅ No missing values to handle!")
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 4: Encoding & Scaling
    with tab4:
        st.subheader("🔄 Encoding & Scaling")

        if st.session_state.data_manager.raw_data is not None:
            df = st.session_state.data_manager.raw_data

            # Encoding section
            st.write("**Categorical Encoding**")
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            if categorical_cols:
                st.write(f"Categorical columns found: {categorical_cols}")

                cols_to_encode = st.multiselect(
                    "Select columns to encode:", categorical_cols
                )

                if cols_to_encode:
                    encoding_method = st.radio(
                        "Encoding method:",
                        ["label", "onehot"],
                        help="Label Encoding: Convert to numbers. One-Hot: Create binary columns.",
                    )

                    if st.button("Apply Encoding"):
                        df_encoded = st.session_state.data_manager.encode_categorical(
                            df, cols_to_encode, encoding_method
                        )
                        st.session_state.data_manager.raw_data = df_encoded
                        st.success("✅ Encoding applied successfully!")
                        st.write("New shape:", df_encoded.shape)
                        st.rerun()
            else:
                st.info("No categorical columns found")

            st.markdown("---")
            st.info(
                "💡 **Note:** Feature scaling will be applied automatically during train/val/test split"
            )
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 5: Train/Val/Test Split
    with tab5:
        st.subheader("✂️ Train/Validation/Test Split")

        if st.session_state.data_manager.raw_data is not None:
            df = st.session_state.data_manager.raw_data

            # Select target column
            target_col = st.selectbox("Select Target Column:", df.columns.tolist())

            col1, col2, col3 = st.columns(3)

            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05)
            with col2:
                val_size = st.slider("Validation Set Size", 0.05, 0.2, 0.1, 0.05)
            with col3:
                random_state = st.number_input("Random State", 1, 999, 42)

            train_size = 1 - test_size - val_size

            # Display split proportions
            st.write("**Split Proportions:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Train", f"{train_size*100:.1f}%")
            col2.metric("Validation", f"{val_size*100:.1f}%")
            col3.metric("Test", f"{test_size*100:.1f}%")

            # Scale features option
            apply_scaling = st.checkbox(
                "Apply Feature Scaling (StandardScaler)", value=True
            )

            if st.button("Split Dataset", type="primary"):
                # Perform split
                X_train, X_val, X_test, y_train, y_val, y_test = (
                    st.session_state.data_manager.split_data(
                        df, target_col, test_size, val_size, random_state
                    )
                )

                # Apply scaling if requested
                if apply_scaling:
                    X_train, X_val, X_test = (
                        st.session_state.data_manager.scale_features(
                            X_train, X_val, X_test
                        )
                    )
                    st.session_state.data_manager.X_train = X_train
                    st.session_state.data_manager.X_val = X_val
                    st.session_state.data_manager.X_test = X_test

                st.success("✅ Dataset split successfully!")

                # Display split summary
                summary = st.session_state.data_manager.get_split_summary()

                st.subheader("📊 Split Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Training Set**")
                    st.metric("Samples", summary["train"]["samples"])
                    st.write(
                        "Class Distribution:", summary["train"]["class_distribution"]
                    )

                with col2:
                    st.write("**Validation Set**")
                    st.metric("Samples", summary["validation"]["samples"])
                    st.write(
                        "Class Distribution:",
                        summary["validation"]["class_distribution"],
                    )

                with col3:
                    st.write("**Test Set**")
                    st.metric("Samples", summary["test"]["samples"])
                    st.write(
                        "Class Distribution:", summary["test"]["class_distribution"]
                    )

                if apply_scaling:
                    st.info("✅ Feature scaling applied using StandardScaler")

                st.session_state.data_prepared = True
        else:
            st.warning("⚠️ Please upload a dataset first!")

# ==================== OTHER MODULES (Placeholders) ====================
elif module == "2️⃣ Baseline Modeling":
    st.header("2️⃣ Baseline Modeling Module")
    st.markdown("**Purpose:** Train and evaluate ML models on clean data")

    # Check if data is prepared
    if (
        not hasattr(st.session_state, "data_prepared")
        or not st.session_state.data_prepared
    ):
        st.warning("⚠️ Please prepare your data first in Module 1!")
        st.info(
            "Go to **1️⃣ Data Management** → **Train/Val/Test Split** to prepare your data"
        )
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
                data = st.session_state.data_manager.get_data()
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
                    params["max_iter"] = st.slider(
                        "Max Iterations", 100, 5000, 1000, 100
                    )

            elif model_name == "Random Forest":
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["n_estimators"] = st.slider(
                        "Number of Trees", 10, 500, 100, 10
                    )
                with col2:
                    max_depth_none = st.checkbox("No Max Depth", value=True)
                    if not max_depth_none:
                        params["max_depth"] = st.slider("Max Depth", 1, 50, 10)
                    else:
                        params["max_depth"] = None
                with col3:
                    params["min_samples_split"] = st.slider(
                        "Min Samples Split", 2, 20, 2
                    )

            elif model_name == "XGBoost":
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["n_estimators"] = st.slider(
                        "Number of Trees", 10, 500, 100, 10
                    )
                with col2:
                    params["max_depth"] = st.slider("Max Depth", 1, 15, 6)
                with col3:
                    params["learning_rate"] = st.slider(
                        "Learning Rate", 0.01, 1.0, 0.1, 0.01
                    )

            params["random_state"] = 42

            st.markdown("---")

            # Train button
            if st.button(f"🚀 Train {model_name}", type="primary"):
                with st.spinner(f"Training {model_name}..."):
                    # Get data
                    data = st.session_state.data_manager.get_data()
                    X_train = data["X_train"]
                    y_train = data["y_train"]
                    X_val = data["X_val"]
                    y_val = data["y_val"]
                    X_test = data["X_test"]
                    y_test = data["y_test"]

                    # Train model
                    model = st.session_state.model_trainer.train_model(
                        model_name, X_train, y_train, **params
                    )

                    # Evaluate on all sets
                    train_metrics = st.session_state.model_trainer.evaluate_model(
                        model_name, X_train, y_train, "Train"
                    )
                    val_metrics = st.session_state.model_trainer.evaluate_model(
                        model_name, X_val, y_val, "Validation"
                    )
                    test_metrics = st.session_state.model_trainer.evaluate_model(
                        model_name, X_test, y_test, "Test"
                    )

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

                    st.session_state.model_trained = True

            # Show trained models
            if st.session_state.model_trainer.trained_models:
                st.markdown("---")
                st.write("**Trained Models:**")
                for (
                    trained_model
                ) in st.session_state.model_trainer.trained_models.keys():
                    st.success(f"✅ {trained_model}")

        # TAB 2: Model Evaluation
        with tab2:
            st.subheader("📊 Model Evaluation")

            if not st.session_state.model_trainer.trained_models:
                st.info(
                    "👈 Train a model first in the 'Model Selection & Training' tab"
                )
            else:
                # Model selector
                eval_model = st.selectbox(
                    "Select model to evaluate:",
                    list(st.session_state.model_trainer.trained_models.keys()),
                )

                dataset_eval = st.radio(
                    "Select dataset:", ["Train", "Validation", "Test"], horizontal=True
                )

                st.markdown("---")

                # Get metrics
                metrics = st.session_state.model_trainer.metrics.get(
                    eval_model, {}
                ).get(dataset_eval)

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
                        cm_fig = st.session_state.model_trainer.plot_confusion_matrix(
                            eval_model, dataset_eval
                        )
                        if cm_fig:
                            st.plotly_chart(cm_fig, use_container_width=True)

                    with col2:
                        st.subheader("📋 Classification Report")
                        report = (
                            st.session_state.model_trainer.get_classification_report(
                                eval_model, dataset_eval
                            )
                        )
                        if report:
                            # Display as dataframe
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)

                    st.markdown("---")

                    # Model Comparison
                    st.subheader("📊 Model Comparison")
                    comparison_fig = (
                        st.session_state.model_trainer.plot_metrics_comparison(
                            dataset_eval
                        )
                    )
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)

                    # Summary Table
                    st.subheader("📑 All Models Summary")
                    summary_df = st.session_state.model_trainer.get_model_summary()
                    if not summary_df.empty:
                        st.dataframe(summary_df, use_container_width=True)
                else:
                    st.warning(
                        f"No evaluation results for {eval_model} on {dataset_eval} set"
                    )

        # TAB 3: Feature Importance
        with tab3:
            st.subheader("📈 Feature Importance Analysis")

            if not st.session_state.model_trainer.trained_models:
                st.info(
                    "👈 Train a model first in the 'Model Selection & Training' tab"
                )
            else:
                model_fi = st.selectbox(
                    "Select model:",
                    list(st.session_state.model_trainer.trained_models.keys()),
                    key="fi_model",
                )

                # Get feature names
                data = st.session_state.data_manager.get_data()
                feature_names = data["feature_columns"]

                if model_fi in ["Random Forest", "XGBoost"]:
                    top_n = st.slider("Number of top features to display", 5, 20, 10)

                    # Plot feature importance
                    fi_fig = st.session_state.model_trainer.plot_feature_importance(
                        model_fi, feature_names, top_n
                    )

                    if fi_fig:
                        st.plotly_chart(fi_fig, use_container_width=True)

                        # Show table
                        st.subheader("📊 Feature Importance Table")
                        fi_df = st.session_state.model_trainer.get_feature_importance(
                            model_fi, feature_names
                        )
                        st.dataframe(fi_df, use_container_width=True)
                else:
                    st.info(f"❌ Feature importance not available for {model_fi}")
                    st.write("Currently supported for: Random Forest, XGBoost")

        # TAB 4: Save/Load Models
        with tab4:
            st.subheader("💾 Save/Load Models")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Save Model**")
                if st.session_state.model_trainer.trained_models:
                    save_model = st.selectbox(
                        "Select model to save:",
                        list(st.session_state.model_trainer.trained_models.keys()),
                        key="save_model",
                    )

                    filename = st.text_input(
                        "Filename:", value=f"{save_model.lower().replace(' ', '_')}.pkl"
                    )

                    if st.button("💾 Save Model"):
                        filepath = os.path.join("saved_models", filename)
                        st.session_state.model_trainer.save_model(save_model, filepath)
                else:
                    st.info("No trained models to save")

            with col2:
                st.write("**Load Model**")

                # Check if saved_models directory exists
                saved_models_dir = "saved_models"
                if os.path.exists(saved_models_dir):
                    # List all .pkl files
                    model_files = [
                        f for f in os.listdir(saved_models_dir) if f.endswith(".pkl")
                    ]

                    if model_files:
                        selected_file = st.selectbox(
                            "Select model file:", model_files, key="load_model_file"
                        )

                        model_name_input = st.text_input(
                            "Model name:",
                            value=selected_file.replace(".pkl", "")
                            .replace("_", " ")
                            .title(),
                            key="load_model_name",
                        )

                        if st.button("📂 Load Model"):
                            filepath = os.path.join(saved_models_dir, selected_file)
                            try:
                                st.session_state.model_trainer.load_model(
                                    model_name_input, filepath
                                )
                                st.session_state.model_trained = True

                                # Evaluate on all datasets
                                data = st.session_state.data_manager.get_data()
                                X_train = data["X_train"]
                                y_train = data["y_train"]
                                X_val = data["X_val"]
                                y_val = data["y_val"]
                                X_test = data["X_test"]
                                y_test = data["y_test"]

                                with st.spinner("Evaluating loaded model..."):
                                    st.session_state.model_trainer.evaluate_model(
                                        model_name_input, X_train, y_train, "Train"
                                    )
                                    st.session_state.model_trainer.evaluate_model(
                                        model_name_input, X_val, y_val, "Validation"
                                    )
                                    st.session_state.model_trainer.evaluate_model(
                                        model_name_input, X_test, y_test, "Test"
                                    )

                                st.success(
                                    f"✅ Model '{model_name_input}' loaded and evaluated!"
                                )
                            except Exception as e:
                                st.error(f"❌ Error loading model: {str(e)}")
                    else:
                        st.info("No saved models found in 'saved_models' folder")
                else:
                    st.info("No 'saved_models' folder found. Save a model first!")

            st.markdown("---")

            # Best Model
            if st.session_state.model_trainer.metrics:
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

                best_model, best_score = st.session_state.model_trainer.get_best_model(
                    metric_choice, dataset_choice
                )

                if best_model:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Best Model:** {best_model}")
                    with col2:
                        st.metric(
                            f"Best {metric_choice.capitalize()}", f"{best_score:.4f}"
                        )

elif module == "3️⃣ Prediction & Confidence":
    st.header("3️⃣ Prediction & Confidence Module")
    st.markdown(
        "Analyze model predictions, confidence levels, and identify high-confidence errors."
    )

    # Check prerequisites
    if st.session_state.data_manager is None:
        st.warning("⚠️ Please load and prepare data in Module 1 first.")
    elif (
        st.session_state.model_trainer is None
        or not st.session_state.model_trainer.trained_models
    ):
        st.warning("⚠️ Please train at least one model in Module 2 first.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Predictions Overview",
                "🎯 Confidence Analysis",
                "⚠️ High-Confidence Errors",
                "📈 Calibration & Entropy",
            ]
        )

        with tab1:
            st.subheader("Model Predictions Overview")

            # Model selection
            model_names = list(st.session_state.model_trainer.trained_models.keys())
            selected_model = st.selectbox(
                "Select Model", model_names, key="pred_model_select"
            )

            # Dataset selection
            dataset_choice = st.radio(
                "Select Dataset",
                ["Validation Set", "Test Set"],
                horizontal=True,
                key="pred_dataset_select",
            )

            if st.button("Generate Predictions", key="generate_predictions"):
                with st.spinner("Generating predictions..."):
                    model = st.session_state.model_trainer.trained_models[
                        selected_model
                    ]

                    # Select dataset
                    if dataset_choice == "Validation Set":
                        X = st.session_state.data_manager.X_val
                        y = st.session_state.data_manager.y_val
                    else:
                        X = st.session_state.data_manager.X_test
                        y = st.session_state.data_manager.y_test

                    # Get predictions and probabilities
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)

                    # Get confidence scores
                    confidence = get_confidence_scores(y_proba)

                    # Create confidence DataFrame
                    confidence_df = pd.DataFrame(
                        {
                            "True_Label": y.values,
                            "Predicted_Label": y_pred,
                            "Confidence": confidence,
                            "Correct": y.values == y_pred,
                        }
                    )

                    # Store in session state
                    st.session_state.predictions = {
                        "model_name": selected_model,
                        "dataset": dataset_choice,
                        "X": X,
                        "y_true": y,
                        "y_pred": y_pred,
                        "y_proba": y_proba,
                        "confidence_df": confidence_df,
                    }

                    st.success("✅ Predictions generated successfully!")

            # Display predictions if available
            if (
                hasattr(st.session_state, "predictions")
                and st.session_state.predictions
            ):
                pred_data = st.session_state.predictions

                st.markdown("---")
                st.markdown(f"**Model:** {pred_data['model_name']}")
                st.markdown(f"**Dataset:** {pred_data['dataset']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    accuracy = (pred_data["y_true"] == pred_data["y_pred"]).mean()
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    avg_confidence = pred_data["confidence_df"]["Confidence"].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                with col3:
                    correct_mask = pred_data["confidence_df"]["Correct"]
                    if correct_mask.sum() > 0:
                        correct_conf = pred_data["confidence_df"][correct_mask][
                            "Confidence"
                        ].mean()
                        st.metric("Avg Confidence (Correct)", f"{correct_conf:.2%}")

                # Show detailed predictions table
                st.markdown("#### Detailed Predictions")
                display_df = pred_data["confidence_df"].copy()
                display_df["Confidence"] = display_df["Confidence"].apply(
                    lambda x: f"{x:.2%}"
                )
                st.dataframe(display_df, use_container_width=True)

                # Download predictions
                csv = pred_data["confidence_df"].to_csv(index=False)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    csv,
                    f"predictions_{pred_data['model_name']}_{dataset_choice}.csv",
                    "text/csv",
                    key="download_predictions",
                )

        with tab2:
            st.subheader("Confidence Analysis")

            if (
                not hasattr(st.session_state, "predictions")
                or not st.session_state.predictions
            ):
                st.info(
                    "👈 Generate predictions in the 'Predictions Overview' tab first."
                )
            else:
                pred_data = st.session_state.predictions

                st.markdown(
                    f"**Analyzing:** {pred_data['model_name']} on {pred_data['dataset']}"
                )

                # Confidence threshold slider
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.0,
                    1.0,
                    0.8,
                    step=0.05,
                    help="Adjust to filter predictions by confidence level",
                    key="confidence_threshold",
                )

                # Overall confidence histogram
                st.markdown("#### Confidence Distribution (Correct vs Incorrect)")
                fig_hist = plot_confidence_histogram(
                    pred_data["confidence_df"]["Confidence"].values,
                    pred_data["confidence_df"]["Correct"].values,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Class-wise confidence analysis
                st.markdown("#### Confidence by Class")
                fig_class = plot_confidence_by_class(
                    pred_data["confidence_df"]["Confidence"].values,
                    pred_data["confidence_df"]["Predicted_Label"].values,
                    pred_data["confidence_df"]["True_Label"].values,
                )
                st.plotly_chart(fig_class, use_container_width=True)

                # Statistics by confidence level
                st.markdown("#### Statistics by Confidence Level")
                high_conf_mask = (
                    pred_data["confidence_df"]["Confidence"] >= confidence_threshold
                )
                low_conf_mask = (
                    pred_data["confidence_df"]["Confidence"] < confidence_threshold
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**High Confidence (≥ {confidence_threshold:.0%})**")
                    high_conf_df = pred_data["confidence_df"][high_conf_mask]
                    st.metric("Count", len(high_conf_df))
                    if len(high_conf_df) > 0:
                        high_acc = high_conf_df["Correct"].mean()
                        st.metric("Accuracy", f"{high_acc:.2%}")

                with col2:
                    st.markdown(f"**Low Confidence (< {confidence_threshold:.0%})**")
                    low_conf_df = pred_data["confidence_df"][low_conf_mask]
                    st.metric("Count", len(low_conf_df))
                    if len(low_conf_df) > 0:
                        low_acc = low_conf_df["Correct"].mean()
                        st.metric("Accuracy", f"{low_acc:.2%}")

        with tab3:
            st.subheader("High-Confidence Errors")
            st.markdown(
                "Identify predictions where the model was very confident but **wrong** - these are critical failure cases."
            )

            if (
                not hasattr(st.session_state, "predictions")
                or not st.session_state.predictions
            ):
                st.info(
                    "👈 Generate predictions in the 'Predictions Overview' tab first."
                )
            else:
                pred_data = st.session_state.predictions

                # Error threshold slider
                error_threshold = st.slider(
                    "Minimum Confidence for Errors",
                    0.5,
                    1.0,
                    0.8,
                    step=0.05,
                    help="Find errors where model confidence was above this threshold",
                    key="error_threshold",
                )

                # Identify high-confidence errors
                hc_errors_info = identify_high_confidence_errors(
                    pred_data["y_true"].values,
                    pred_data["y_pred"],
                    pred_data["y_proba"],
                    threshold=error_threshold,
                )

                if hc_errors_info["count"] == 0:
                    st.success(
                        f"✅ No high-confidence errors found (threshold: {error_threshold:.0%})"
                    )
                else:
                    st.warning(
                        f"⚠️ Found {hc_errors_info['count']} high-confidence errors ({hc_errors_info['percentage']:.1f}%)"
                    )

                    # Show error analysis plot
                    fig_errors = plot_error_analysis(
                        pred_data["confidence_df"]["Confidence"].values,
                        pred_data["confidence_df"]["Predicted_Label"].values,
                        pred_data["confidence_df"]["True_Label"].values,
                        threshold=error_threshold,
                    )
                    st.plotly_chart(fig_errors, use_container_width=True)

                    # Show error details
                    st.markdown("#### Error Details")
                    error_indices = hc_errors_info["indices"]
                    hc_errors = pred_data["confidence_df"].iloc[error_indices].copy()
                    error_display = hc_errors.copy()
                    error_display["Confidence"] = error_display["Confidence"].apply(
                        lambda x: f"{x:.2%}"
                    )
                    st.dataframe(error_display, use_container_width=True)

                    # Error statistics by class
                    st.markdown("#### Errors by True Class")
                    error_counts = hc_errors["True_Label"].value_counts()
                    st.bar_chart(error_counts)

                    # Download high-confidence errors
                    csv_errors = hc_errors.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download High-Confidence Errors CSV",
                        csv_errors,
                        f"hc_errors_{pred_data['model_name']}_{error_threshold:.0%}.csv",
                        "text/csv",
                        key="download_errors",
                    )

        with tab4:
            st.subheader("Model Calibration & Uncertainty")
            st.markdown(
                "Assess whether prediction confidence matches actual accuracy (calibration) and analyze prediction uncertainty."
            )

            if (
                not hasattr(st.session_state, "predictions")
                or not st.session_state.predictions
            ):
                st.info(
                    "👈 Generate predictions in the 'Predictions Overview' tab first."
                )
            else:
                pred_data = st.session_state.predictions

                # Calibration curve
                st.markdown("#### Calibration Curve")
                st.markdown(
                    "A well-calibrated model's confidence should match its accuracy."
                )

                fig_calib = plot_confidence_accuracy_curve(
                    pred_data["confidence_df"]["Confidence"].values,
                    pred_data["confidence_df"]["Correct"].values,
                )
                st.plotly_chart(fig_calib, use_container_width=True)

                # Brier score
                brier = calculate_brier_score(pred_data["y_true"], pred_data["y_proba"])
                st.metric(
                    "Brier Score",
                    f"{brier:.4f}",
                    help="Lower is better. Measures calibration quality (0 = perfect, 1 = worst)",
                )

                # Entropy analysis
                st.markdown("#### Prediction Uncertainty (Entropy)")
                st.markdown("High entropy = model is uncertain about the prediction.")

                entropy_series = get_prediction_entropy(pred_data["y_proba"])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Entropy", f"{entropy_series.mean():.3f}")
                    st.metric("Max Entropy", f"{entropy_series.max():.3f}")

                with col2:
                    st.metric("Min Entropy", f"{entropy_series.min():.3f}")
                    st.metric("Std Entropy", f"{entropy_series.std():.3f}")

                # Entropy distribution
                fig_entropy = plot_entropy_distribution(
                    entropy_series, pred_data["confidence_df"]["Correct"].values
                )
                st.plotly_chart(fig_entropy, use_container_width=True)

                # Show samples with highest uncertainty
                st.markdown("#### Most Uncertain Predictions")
                uncertainty_df = pred_data["confidence_df"].copy()
                uncertainty_df["Entropy"] = entropy_series
                most_uncertain = uncertainty_df.nlargest(10, "Entropy")[
                    [
                        "True_Label",
                        "Predicted_Label",
                        "Confidence",
                        "Entropy",
                        "Correct",
                    ]
                ]
                most_uncertain["Confidence"] = most_uncertain["Confidence"].apply(
                    lambda x: f"{x:.2%}"
                )
                most_uncertain["Entropy"] = most_uncertain["Entropy"].apply(
                    lambda x: f"{x:.3f}"
                )
                st.dataframe(most_uncertain, use_container_width=True)

elif module == "4️⃣ Stress Testing":
    st.header("4️⃣ Stress Testing Module")
    st.markdown(
        "Test model robustness under various data perturbations and stress conditions."
    )

    # Check prerequisites
    if st.session_state.data_manager is None:
        st.warning("⚠️ Please load and prepare data in Module 1 first.")
    elif (
        st.session_state.model_trainer is None
        or not st.session_state.model_trainer.trained_models
    ):
        st.warning("⚠️ Please train at least one model in Module 2 first.")
    else:
        # Initialize stress tester if not already done
        if not hasattr(st.session_state, "stress_tester"):
            st.session_state.stress_tester = StressTester()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🎯 Single Stress Test",
                "📊 Batch Stress Tests",
                "📈 Results Analysis",
                "💾 Export Results",
            ]
        )

        with tab1:
            st.subheader("Single Stress Test")
            st.markdown("Apply a single stress test and immediately view results.")

            col1, col2 = st.columns(2)

            with col1:
                # Model selection
                model_names = list(st.session_state.model_trainer.trained_models.keys())
                selected_model = st.selectbox(
                    "Select Model", model_names, key="stress_model_select"
                )

                # Dataset selection
                dataset_choice = st.radio(
                    "Select Dataset",
                    ["Validation Set", "Test Set"],
                    horizontal=True,
                    key="stress_dataset_select",
                )

            with col2:
                # Stress test type
                stress_type = st.selectbox(
                    "Stress Test Type",
                    [
                        "Gaussian Noise",
                        "Uniform Noise",
                        "Feature Dropout",
                        "Feature Corruption",
                        "Scale Perturbation",
                        "Distribution Shift",
                    ],
                    key="single_stress_type",
                )

            # Parameters based on stress type
            st.markdown("#### Configuration")

            if stress_type == "Gaussian Noise":
                noise_level = st.slider(
                    "Noise Level (std multiplier)",
                    0.0,
                    1.0,
                    0.1,
                    0.05,
                    help="Noise = Normal(0, feature_std * noise_level)",
                )
                params = {"noise_level": noise_level}

            elif stress_type == "Uniform Noise":
                noise_range = st.slider(
                    "Noise Range (fraction of feature range)",
                    0.0,
                    0.5,
                    0.1,
                    0.05,
                    help="Noise uniformly distributed in [-range*feature_range, +range*feature_range]",
                )
                params = {"noise_range": noise_range}

            elif stress_type == "Feature Dropout":
                dropout_rate = st.slider(
                    "Dropout Rate",
                    0.0,
                    0.8,
                    0.2,
                    0.05,
                    help="Fraction of features to randomly set to zero",
                )
                params = {"dropout_rate": dropout_rate}

            elif stress_type == "Feature Corruption":
                corruption_rate = st.slider(
                    "Corruption Rate",
                    0.0,
                    0.5,
                    0.1,
                    0.05,
                    help="Fraction of values to corrupt",
                )
                corruption_type_choice = st.selectbox(
                    "Corruption Type",
                    ["zero", "mean", "random", "extreme"],
                    help="How to corrupt values",
                )
                params = {
                    "corruption_rate": corruption_rate,
                    "corruption_type": corruption_type_choice,
                }

            elif stress_type == "Scale Perturbation":
                scale_factor = st.slider(
                    "Scale Factor",
                    1.0,
                    3.0,
                    1.5,
                    0.1,
                    help="Features randomly scaled by factor or 1/factor",
                )
                params = {"scale_factor": scale_factor}

            elif stress_type == "Distribution Shift":
                shift_type_choice = st.radio(
                    "Shift Type", ["mean", "variance"], horizontal=True
                )
                shift_amount = st.slider(
                    "Shift Amount",
                    0.0,
                    2.0,
                    0.5,
                    0.1,
                    help="For mean: shift by amount*std. For variance: scale by amount",
                )
                params = {"shift_type": shift_type_choice, "shift_amount": shift_amount}

            if st.button("🧪 Run Stress Test", key="run_single_stress"):
                with st.spinner("Running stress test..."):
                    model = st.session_state.model_trainer.trained_models[
                        selected_model
                    ]

                    # Select dataset
                    if dataset_choice == "Validation Set":
                        X = st.session_state.data_manager.X_val
                        y = st.session_state.data_manager.y_val
                    else:
                        X = st.session_state.data_manager.X_test
                        y = st.session_state.data_manager.y_test

                    # Apply stress test
                    stress_tester = st.session_state.stress_tester

                    if stress_type == "Gaussian Noise":
                        X_stressed = stress_tester.add_gaussian_noise(X, **params)
                    elif stress_type == "Uniform Noise":
                        X_stressed = stress_tester.add_uniform_noise(X, **params)
                    elif stress_type == "Feature Dropout":
                        X_stressed = stress_tester.feature_dropout(X, **params)
                    elif stress_type == "Feature Corruption":
                        X_stressed = stress_tester.feature_corruption(X, **params)
                    elif stress_type == "Scale Perturbation":
                        X_stressed = stress_tester.scale_perturbation(X, **params)
                    elif stress_type == "Distribution Shift":
                        X_stressed = stress_tester.distribution_shift(X, **params)

                    # Evaluate
                    result = stress_tester.evaluate_stress_test(model, X, X_stressed, y)

                    # Store result
                    st.session_state.single_stress_result = {
                        "model": selected_model,
                        "dataset": dataset_choice,
                        "stress_type": stress_type,
                        "params": params,
                        "result": result,
                        "X_stressed": X_stressed,
                    }

                    st.success("✅ Stress test completed!")

            # Display results
            if hasattr(st.session_state, "single_stress_result"):
                st.markdown("---")
                st.markdown("### Results")

                result_data = st.session_state.single_stress_result
                result = result_data["result"]

                st.markdown(f"**Model:** {result_data['model']}")
                st.markdown(f"**Dataset:** {result_data['dataset']}")
                st.markdown(f"**Stress Test:** {result_data['stress_type']}")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Accuracy", f"{result['accuracy_original']:.2%}")
                with col2:
                    st.metric(
                        "Stressed Accuracy",
                        f"{result['accuracy_stressed']:.2%}",
                        f"{-result['performance_drop']:.2%}",
                    )
                with col3:
                    st.metric(
                        "Performance Drop", f"{result['performance_drop_pct']:.1f}%"
                    )
                with col4:
                    st.metric(
                        "Prediction Agreement", f"{result['prediction_agreement']:.2%}"
                    )

                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Precision (Stressed)", f"{result['precision_stressed']:.2%}"
                    )
                with col2:
                    st.metric("Recall (Stressed)", f"{result['recall_stressed']:.2%}")
                with col3:
                    st.metric("F1 Score (Stressed)", f"{result['f1_stressed']:.2%}")

        with tab2:
            st.subheader("Batch Stress Tests")
            st.markdown(
                "Run multiple stress tests simultaneously to comprehensively evaluate model robustness."
            )

            col1, col2 = st.columns(2)

            with col1:
                batch_model = st.selectbox(
                    "Select Model",
                    list(st.session_state.model_trainer.trained_models.keys()),
                    key="batch_stress_model",
                )

            with col2:
                batch_dataset = st.radio(
                    "Select Dataset",
                    ["Validation Set", "Test Set"],
                    horizontal=True,
                    key="batch_stress_dataset",
                )

            st.markdown("#### Select Stress Tests")

            col1, col2 = st.columns(2)

            with col1:
                test_gaussian = st.checkbox(
                    "Gaussian Noise (Low: 0.05, Medium: 0.15, High: 0.3)", value=True
                )
                test_uniform = st.checkbox(
                    "Uniform Noise (Low: 0.05, Medium: 0.15, High: 0.3)", value=True
                )
                test_dropout = st.checkbox(
                    "Feature Dropout (10%, 20%, 30%)", value=True
                )

            with col2:
                test_corruption = st.checkbox(
                    "Feature Corruption (5%, 15%, 25%)", value=True
                )
                test_scale = st.checkbox(
                    "Scale Perturbation (1.5x, 2.0x, 2.5x)", value=True
                )
                test_shift = st.checkbox(
                    "Distribution Shift (Mean: 0.5, 1.0 / Var: 0.5, 1.5)", value=True
                )

            if st.button("🚀 Run Batch Stress Tests", key="run_batch_stress"):
                with st.spinner("Running batch stress tests..."):
                    model = st.session_state.model_trainer.trained_models[batch_model]

                    # Select dataset
                    if batch_dataset == "Validation Set":
                        X = st.session_state.data_manager.X_val
                        y = st.session_state.data_manager.y_val
                    else:
                        X = st.session_state.data_manager.X_test
                        y = st.session_state.data_manager.y_test

                    # Build stress test configurations
                    stress_configs = []

                    if test_gaussian:
                        stress_configs.extend(
                            [
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.05},
                                    "name": "Gaussian Noise (Low)",
                                },
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.15},
                                    "name": "Gaussian Noise (Medium)",
                                },
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.3},
                                    "name": "Gaussian Noise (High)",
                                },
                            ]
                        )

                    if test_uniform:
                        stress_configs.extend(
                            [
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.05},
                                    "name": "Uniform Noise (Low)",
                                },
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.15},
                                    "name": "Uniform Noise (Medium)",
                                },
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.3},
                                    "name": "Uniform Noise (High)",
                                },
                            ]
                        )

                    if test_dropout:
                        stress_configs.extend(
                            [
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.1},
                                    "name": "Feature Dropout (10%)",
                                },
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.2},
                                    "name": "Feature Dropout (20%)",
                                },
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.3},
                                    "name": "Feature Dropout (30%)",
                                },
                            ]
                        )

                    if test_corruption:
                        stress_configs.extend(
                            [
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.05,
                                        "corruption_type": "zero",
                                    },
                                    "name": "Corruption (5%, zero)",
                                },
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.15,
                                        "corruption_type": "random",
                                    },
                                    "name": "Corruption (15%, random)",
                                },
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.25,
                                        "corruption_type": "extreme",
                                    },
                                    "name": "Corruption (25%, extreme)",
                                },
                            ]
                        )

                    if test_scale:
                        stress_configs.extend(
                            [
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 1.5},
                                    "name": "Scale Perturbation (1.5x)",
                                },
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 2.0},
                                    "name": "Scale Perturbation (2.0x)",
                                },
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 2.5},
                                    "name": "Scale Perturbation (2.5x)",
                                },
                            ]
                        )

                    if test_shift:
                        stress_configs.extend(
                            [
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "mean",
                                        "shift_amount": 0.5,
                                    },
                                    "name": "Mean Shift (0.5)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "mean",
                                        "shift_amount": 1.0,
                                    },
                                    "name": "Mean Shift (1.0)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "variance",
                                        "shift_amount": 0.5,
                                    },
                                    "name": "Variance Shift (0.5)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "variance",
                                        "shift_amount": 1.5,
                                    },
                                    "name": "Variance Shift (1.5)",
                                },
                            ]
                        )

                    # Run batch tests
                    results = st.session_state.stress_tester.batch_stress_test(
                        model, X, y, stress_configs
                    )

                    # Store results per model
                    if "batch_stress_results_by_model" not in st.session_state:
                        st.session_state.batch_stress_results_by_model = {}
                    st.session_state.batch_stress_results_by_model[batch_model] = (
                        results
                    )

                    # Also keep last run for backward compat (Results Analysis tab)
                    st.session_state.batch_stress_results = {
                        "model": batch_model,
                        "dataset": batch_dataset,
                        "results": results,
                    }

                    st.success(f"✅ Completed {len(results)} stress tests!")

            # Display batch results summary
            if hasattr(st.session_state, "batch_stress_results"):
                st.markdown("---")
                st.markdown("### Batch Results Summary")

                batch_data = st.session_state.batch_stress_results
                results = batch_data["results"]

                # Create summary DataFrame
                summary_data = []
                for name, result in results.items():
                    summary_data.append(
                        {
                            "Stress Test": name,
                            "Original Acc": f"{result['accuracy_original']:.2%}",
                            "Stressed Acc": f"{result['accuracy_stressed']:.2%}",
                            "Drop %": f"{result['performance_drop_pct']:.1f}%",
                            "Agreement": f"{result['prediction_agreement']:.2%}",
                            "F1": f"{result['f1_stressed']:.2%}",
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        with tab3:
            st.subheader("Results Analysis")

            if not hasattr(st.session_state, "batch_stress_results"):
                st.info(
                    "👈 Run batch stress tests in the 'Batch Stress Tests' tab first."
                )
            else:
                batch_data = st.session_state.batch_stress_results
                results = batch_data["results"]

                st.markdown(f"**Model:** {batch_data['model']}")
                st.markdown(f"**Dataset:** {batch_data['dataset']}")

                # Performance drop chart
                st.markdown("#### Performance Drop by Stress Test")

                chart_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Performance Drop (%)": [
                            r["performance_drop_pct"] for r in results.values()
                        ],
                        "Accuracy (Stressed)": [
                            r["accuracy_stressed"] * 100 for r in results.values()
                        ],
                    }
                )

                st.bar_chart(
                    chart_data.set_index("Stress Test")["Performance Drop (%)"]
                )

                # Accuracy comparison
                st.markdown("#### Accuracy: Original vs Stressed")

                acc_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Original": [
                            r["accuracy_original"] * 100 for r in results.values()
                        ],
                        "Stressed": [
                            r["accuracy_stressed"] * 100 for r in results.values()
                        ],
                    }
                )

                st.line_chart(acc_data.set_index("Stress Test"))

                # Prediction agreement
                st.markdown("#### Prediction Agreement")
                st.markdown(
                    "Percentage of predictions that remained the same under stress"
                )

                agreement_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Agreement (%)": [
                            r["prediction_agreement"] * 100 for r in results.values()
                        ],
                    }
                )

                st.bar_chart(agreement_data.set_index("Stress Test")["Agreement (%)"])

                # Worst performing stress tests
                st.markdown("#### Most Challenging Stress Tests")
                st.markdown("Stress tests that caused the largest performance drops")

                worst_tests = sorted(
                    results.items(),
                    key=lambda x: x[1]["performance_drop_pct"],
                    reverse=True,
                )[:5]

                worst_df = pd.DataFrame(
                    [
                        {
                            "Stress Test": name,
                            "Performance Drop": f"{result['performance_drop_pct']:.1f}%",
                            "Stressed Accuracy": f"{result['accuracy_stressed']:.2%}",
                        }
                        for name, result in worst_tests
                    ]
                )

                st.dataframe(worst_df, use_container_width=True)

        with tab4:
            st.subheader("Export Results")

            if not hasattr(st.session_state, "batch_stress_results"):
                st.info("👈 Run batch stress tests first to export results.")
            else:
                batch_data = st.session_state.batch_stress_results
                results = batch_data["results"]

                st.markdown("#### Download Stress Test Results")

                # Create comprehensive export DataFrame
                export_data = []
                for name, result in results.items():
                    export_data.append(
                        {
                            "Model": batch_data["model"],
                            "Dataset": batch_data["dataset"],
                            "Stress_Test": name,
                            "Accuracy_Original": result["accuracy_original"],
                            "Accuracy_Stressed": result["accuracy_stressed"],
                            "Performance_Drop": result["performance_drop"],
                            "Performance_Drop_Pct": result["performance_drop_pct"],
                            "Precision_Stressed": result["precision_stressed"],
                            "Recall_Stressed": result["recall_stressed"],
                            "F1_Stressed": result["f1_stressed"],
                            "Prediction_Agreement": result["prediction_agreement"],
                        }
                    )

                export_df = pd.DataFrame(export_data)

                csv = export_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Stress Test Results CSV",
                    csv,
                    f"stress_test_results_{batch_data['model']}.csv",
                    "text/csv",
                    key="download_stress_results",
                )

                st.markdown("#### Preview")
                st.dataframe(export_df, use_container_width=True)

elif module == "5️⃣ Post-Stress Evaluation":
    st.header("5️⃣ Post-Stress Evaluation Module")
    st.markdown(
        "**Purpose:** Deep analysis of stress test results and robustness assessment"
    )

    # Check if stress tests have been run
    tested_models = list(
        st.session_state.get("batch_stress_results_by_model", {}).keys()
    )
    if not tested_models:
        st.warning("⚠️ No stress test results available!")
        st.info(
            "Go to **4️⃣ Stress Testing** → **Batch Stress Tests** to run comprehensive stress tests first."
        )
    else:
        st.info(f"✅ Stress test results available for: **{', '.join(tested_models)}**")

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Robustness Score",
                "🎯 Vulnerability Analysis",
                "📈 Stress Type Summary",
                "🔍 Model Comparison",
                "💡 Recommendations",
            ]
        )

        # TAB 1: Robustness Score
        with tab1:
            st.subheader("📊 Overall Robustness Score")

            selected_model = st.selectbox(
                "Select model for analysis:",
                tested_models,
                key="post_stress_model",
            )

            # Load the correct per-model results
            model_results = st.session_state.batch_stress_results_by_model[
                selected_model
            ]
            st.session_state.post_stress_analyzer.add_batch_results(
                model_results, selected_model
            )

            # Calculate robustness score
            if st.button("🔄 Calculate Robustness Score", type="primary"):
                with st.spinner("Analyzing stress test results..."):
                    score = st.session_state.post_stress_analyzer.calculate_robustness_score(
                        selected_model
                    )

                    st.success("✅ Analysis complete!")

                    # Display overall score
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)

                    scores = st.session_state.post_stress_analyzer.robustness_scores[
                        selected_model
                    ]

                    with col1:
                        st.metric(
                            "Overall Robustness",
                            f"{scores['overall_score']:.1f}/100",
                        )

                    with col2:
                        st.metric(
                            "Accuracy Retention",
                            f"{scores['accuracy_retention']:.1f}%",
                        )

                    with col3:
                        st.metric(
                            "Prediction Stability",
                            f"{scores['prediction_stability']:.1f}%",
                        )

                    with col4:
                        st.metric(
                            "Performance Consistency",
                            f"{scores['performance_consistency']:.1f}%",
                        )

                    st.markdown("---")

                    # Score interpretation
                    st.subheader("📖 Score Interpretation")

                    if score >= 80:
                        st.success(
                            "🌟 **Excellent Robustness** - Model performs very well under stress"
                        )
                    elif score >= 60:
                        st.info(
                            "✅ **Good Robustness** - Model handles most stress conditions well"
                        )
                    elif score >= 40:
                        st.warning(
                            "⚠️ **Moderate Robustness** - Model shows some vulnerabilities"
                        )
                    else:
                        st.error(
                            "❌ **Low Robustness** - Model is highly sensitive to perturbations"
                        )

                    # Radar chart
                    st.markdown("---")
                    st.subheader("📊 Robustness Dimensions")

                    radar_fig = (
                        st.session_state.post_stress_analyzer.plot_robustness_radar(
                            selected_model
                        )
                    )
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)

                    # Detailed breakdown
                    st.markdown("---")
                    st.subheader("📋 Metric Breakdown")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Accuracy Retention**")
                        st.write(
                            "Measures how much of the original accuracy is retained under stress."
                        )
                        st.write(
                            f"On average, the model retains {scores['accuracy_retention']:.1f}% of its accuracy."
                        )

                    with col2:
                        st.markdown("**Prediction Stability**")
                        st.write(
                            "Measures how often predictions remain the same despite perturbations."
                        )
                        st.write(
                            f"{scores['prediction_stability']:.1f}% of predictions stay consistent."
                        )

                    st.markdown("**Performance Consistency**")
                    st.write(
                        "Measures how consistently the model performs across different stress tests."
                    )
                    st.write(
                        f"Consistency score: {scores['performance_consistency']:.1f}%"
                    )

        # TAB 2: Vulnerability Analysis
        with tab2:
            st.subheader("🎯 Vulnerability Analysis")

            selected_model_vuln = st.selectbox(
                "Select model:",
                tested_models,
                key="vuln_model",
            )

            # Load the correct per-model results
            model_results_vuln = st.session_state.batch_stress_results_by_model[
                selected_model_vuln
            ]
            st.session_state.post_stress_analyzer.add_batch_results(
                model_results_vuln, selected_model_vuln
            )

            vuln_df = st.session_state.post_stress_analyzer.get_vulnerability_analysis(
                selected_model_vuln
            )

            if not vuln_df.empty:
                # Summary stats
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Most Vulnerable Test",
                        (
                            vuln_df.iloc[0]["Stress Test"][:20] + "..."
                            if len(vuln_df.iloc[0]["Stress Test"]) > 20
                            else vuln_df.iloc[0]["Stress Test"]
                        ),
                    )
                    st.caption(f"{vuln_df.iloc[0]['Performance Drop (%)']:.1f}% drop")

                with col2:
                    critical = len(vuln_df[vuln_df["Severity"] == "Critical"])
                    high = len(vuln_df[vuln_df["Severity"] == "High"])
                    st.metric("High Risk Tests", f"{critical + high}")
                    st.caption(f"{critical} Critical, {high} High")

                with col3:
                    avg_drop = vuln_df["Performance Drop (%)"].mean()
                    st.metric("Average Drop", f"{avg_drop:.1f}%")

                st.markdown("---")

                # Vulnerability heatmap
                st.subheader("🔥 Vulnerability Heatmap")
                heatmap_fig = (
                    st.session_state.post_stress_analyzer.plot_vulnerability_heatmap(
                        selected_model_vuln
                    )
                )
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                st.markdown("---")

                # Detailed table
                st.subheader("📊 Detailed Vulnerability Table")

                def color_severity(val):
                    if val == "Critical":
                        return "background-color: #ff4444; color: white"
                    elif val == "High":
                        return "background-color: #ff9944; color: white"
                    elif val == "Medium":
                        return "background-color: #ffcc44"
                    else:
                        return "background-color: #44ff44"

                styled_df = vuln_df.style.applymap(color_severity, subset=["Severity"])

                st.dataframe(styled_df, use_container_width=True)

                csv = vuln_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Vulnerability Analysis",
                    csv,
                    f"vulnerability_analysis_{selected_model_vuln}.csv",
                    "text/csv",
                )

        # TAB 3: Stress Type Summary
        with tab3:
            st.subheader("📈 Stress Type Summary")

            selected_model_type = st.selectbox(
                "Select model:",
                tested_models,
                key="type_model",
            )

            # Load the correct per-model results
            model_results_type = st.session_state.batch_stress_results_by_model[
                selected_model_type
            ]
            st.session_state.post_stress_analyzer.add_batch_results(
                model_results_type, selected_model_type
            )

            type_df = st.session_state.post_stress_analyzer.get_stress_type_summary(
                selected_model_type
            )

            if not type_df.empty:
                st.markdown(
                    "**Performance by Stress Category** (grouped by stress type)"
                )

                st.dataframe(type_df, use_container_width=True)

                st.markdown("---")

                st.subheader("📊 Average Performance Drop by Category")

                import plotly.graph_objects as go

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=type_df["Category"],
                        y=type_df["Avg Drop (%)"],
                        text=type_df["Avg Drop (%)"].round(1),
                        textposition="auto",
                        marker=dict(
                            color=type_df["Avg Drop (%)"],
                            colorscale="Reds",
                            showscale=True,
                        ),
                    )
                )

                fig.update_layout(
                    title="Average Performance Drop by Stress Category",
                    xaxis_title="Stress Category",
                    yaxis_title="Average Drop (%)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                st.subheader("🎯 Prediction Agreement by Category")

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Bar(
                        x=type_df["Category"],
                        y=type_df["Avg Agreement"] * 100,
                        text=(type_df["Avg Agreement"] * 100).round(1),
                        textposition="auto",
                        marker=dict(color="rgb(99, 110, 250)"),
                    )
                )

                fig2.update_layout(
                    title="Average Prediction Agreement by Category",
                    xaxis_title="Stress Category",
                    yaxis_title="Agreement (%)",
                    yaxis=dict(range=[0, 100]),
                    height=400,
                )

                st.plotly_chart(fig2, use_container_width=True)

        # TAB 4: Model Comparison
        with tab4:
            st.subheader("🔍 Model Comparison")

            st.markdown(
                "**Compare robustness across different models** (requires stress tests for each model)"
            )

            # Only allow selecting models that have actual stress test results
            if len(tested_models) < 2:
                st.info(
                    f"💡 Only **{tested_models[0]}** has been stress tested. Run batch stress tests on more models in Module 4 to compare them."
                )
            else:
                st.info(
                    "💡 Tip: Run stress tests for multiple models in Module 4, then compare them here."
                )

                # Select models to compare
                models_to_compare = st.multiselect(
                    "Select models to compare:",
                    tested_models,
                    default=tested_models[:2],
                    key="compare_models",
                )

                if len(models_to_compare) >= 2:
                    if st.button("📊 Compare Models", type="primary"):
                        # Load the correct per-model results for each
                        for model in models_to_compare:
                            per_model_results = (
                                st.session_state.batch_stress_results_by_model[model]
                            )
                            st.session_state.post_stress_analyzer.add_batch_results(
                                per_model_results, model
                            )

                        # Comparison chart
                        comparison_fig = st.session_state.post_stress_analyzer.compare_model_robustness(
                            models_to_compare
                        )

                        if comparison_fig:
                            st.plotly_chart(comparison_fig, use_container_width=True)

                            # Summary table
                            st.markdown("---")
                            st.subheader("📋 Robustness Score Summary")

                            summary_data = []
                            for model in models_to_compare:
                                st.session_state.post_stress_analyzer.calculate_robustness_score(
                                    model
                                )
                                scores = st.session_state.post_stress_analyzer.robustness_scores[
                                    model
                                ]
                                summary_data.append(
                                    {
                                        "Model": model,
                                        "Overall Score": f"{scores['overall_score']:.1f}",
                                        "Accuracy Retention": f"{scores['accuracy_retention']:.1f}%",
                                        "Prediction Stability": f"{scores['prediction_stability']:.1f}%",
                                        "Consistency": f"{scores['performance_consistency']:.1f}%",
                                    }
                                )

                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)

                            best_idx = (
                                summary_df["Overall Score"].astype(float).idxmax()
                            )
                            best_model = summary_df.iloc[best_idx]["Model"]
                            st.success(f"🏆 **Most Robust Model:** {best_model}")
                        else:
                            st.warning("Could not generate comparison.")

        # TAB 5: Recommendations
        with tab5:
            st.subheader("💡 Recommendations")

            selected_model_rec = st.selectbox(
                "Select model:",
                tested_models,
                key="rec_model",
            )

            # Load the correct per-model results
            model_results_rec = st.session_state.batch_stress_results_by_model[
                selected_model_rec
            ]
            st.session_state.post_stress_analyzer.add_batch_results(
                model_results_rec, selected_model_rec
            )

            if st.button("🔍 Generate Recommendations", type="primary"):
                with st.spinner("Analyzing and generating recommendations..."):
                    recommendations = (
                        st.session_state.post_stress_analyzer.get_recommendations(
                            selected_model_rec
                        )
                    )

                    st.markdown("---")
                    st.subheader(f"📝 Recommendations for {selected_model_rec}")

                    for rec in recommendations:
                        st.markdown(rec)

                    st.markdown("---")

                    st.subheader("🎯 General Best Practices")

                    st.markdown(
                        """
                    - **Data Augmentation**: Add perturbations similar to stress tests during training
                    - **Regularization**: Use dropout, L1/L2 regularization to prevent overfitting
                    - **Ensemble Methods**: Combine multiple models for more robust predictions
                    - **Feature Engineering**: Create robust features less sensitive to noise
                    - **Model Monitoring**: Continuously monitor model performance in production
                    - **Retraining**: Regularly retrain with new data to adapt to distribution shifts
                    - **Input Validation**: Add checks in production to detect anomalous inputs
                    """
                    )

                    st.markdown("---")

                    rec_text = "\n".join(recommendations)
                    st.download_button(
                        "📥 Download Recommendations",
                        rec_text,
                        f"recommendations_{selected_model_rec}.txt",
                        "text/plain",
                    )

elif module == "6️⃣ Calibration Analysis":
    st.header("6️⃣ Calibration Analysis Module")
    st.markdown(
        "**Purpose:** Assess how well model confidence scores reflect true accuracy"
    )

    if (
        st.session_state.model_trainer is None
        or not st.session_state.model_trainer.trained_models
    ):
        st.warning("⚠️ Please train at least one model in Module 2 first.")
    else:
        data = st.session_state.data_manager.get_data()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Calibration Curve",
                "📊 Calibration Metrics",
                "🔬 Per-Class Analysis",
                "🌡️ Temperature Scaling",
            ]
        )

        # ── shared model / dataset selector (rendered inside each tab) ──────────────
        model_options = list(st.session_state.model_trainer.trained_models.keys())

        # TAB 1 – Reliability Diagram
        with tab1:
            st.subheader("📈 Reliability Diagram")
            st.markdown(
                "A well-calibrated model follows the dashed diagonal. "
                "**Red** shading = overconfident, **blue** shading = underconfident."
            )

            col1, col2 = st.columns(2)
            with col1:
                model_cal = st.selectbox(
                    "Select model:", model_options, key="cal_model_1"
                )
            with col2:
                dataset_cal = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_1"
                )

            n_bins_cal = st.slider("Number of bins:", 5, 20, 10, key="cal_bins_1")

            if st.button(
                "📈 Generate Calibration Curve", type="primary", key="btn_cal_1"
            ):
                model = st.session_state.model_trainer.trained_models[model_cal]
                X = data["X_val"] if dataset_cal == "Validation" else data["X_test"]
                y = data["y_val"] if dataset_cal == "Validation" else data["y_test"]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)
                    y_arr = np.asarray(y)

                    metrics = st.session_state.calibration_analyzer.compute_calibration_metrics(
                        y_arr, probabilities, n_bins=n_bins_cal
                    )

                    quality = (
                        st.session_state.calibration_analyzer.get_calibration_quality(
                            metrics["ece"]
                        )
                    )

                    # Quick metric strip
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "ECE",
                            f"{metrics['ece']:.4f}",
                            help="Expected Calibration Error — lower is better",
                        )
                    with c2:
                        st.metric(
                            "MCE",
                            f"{metrics['mce']:.4f}",
                            help="Maximum Calibration Error",
                        )
                    with c3:
                        st.metric("Brier Score", f"{metrics['brier_score']:.4f}")
                    with c4:
                        st.metric("Calibration Quality", quality)

                    if quality == "Excellent":
                        st.success(
                            "🌟 **Excellent calibration** — confidence closely matches accuracy."
                        )
                    elif quality == "Good":
                        st.info(
                            "✅ **Good calibration** — minor deviations from perfect calibration."
                        )
                    elif quality == "Moderate":
                        st.warning(
                            "⚠️ **Moderate calibration** — consider applying temperature scaling."
                        )
                    else:
                        st.error(
                            "❌ **Poor calibration** — probabilities are unreliable. Apply calibration correction."
                        )

                    st.markdown("---")
                    fig = st.session_state.calibration_analyzer.plot_calibration_curve(
                        metrics, model_cal
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    hist_fig = (
                        st.session_state.calibration_analyzer.plot_confidence_histogram(
                            y_arr, probabilities
                        )
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                else:
                    st.error(
                        f"❌ {model_cal} does not support `predict_proba`. Calibration requires probability outputs."
                    )

        # TAB 2 – Calibration Metrics Table
        with tab2:
            st.subheader("📊 Calibration Metrics")
            st.markdown("Compare calibration quality across all trained models.")

            dataset_cal2 = st.radio(
                "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_2"
            )

            if st.button(
                "📊 Compute All Model Metrics", type="primary", key="btn_cal_2"
            ):
                X = data["X_val"] if dataset_cal2 == "Validation" else data["X_test"]
                y = np.asarray(
                    data["y_val"] if dataset_cal2 == "Validation" else data["y_test"]
                )

                all_metrics = {}
                for (
                    mname,
                    model,
                ) in st.session_state.model_trainer.trained_models.items():
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)
                        all_metrics[mname] = (
                            st.session_state.calibration_analyzer.compute_calibration_metrics(
                                y, probs
                            )
                        )

                if all_metrics:
                    # Summary table
                    rows = []
                    for mname, m in all_metrics.items():
                        quality = st.session_state.calibration_analyzer.get_calibration_quality(
                            m["ece"]
                        )
                        rows.append(
                            {
                                "Model": mname,
                                "ECE ↓": round(m["ece"], 4),
                                "MCE ↓": round(m["mce"], 4),
                                "Brier Score ↓": round(m["brier_score"], 4),
                                "Avg Confidence": round(m["avg_confidence"], 4),
                                "Avg Accuracy": round(m["avg_accuracy"], 4),
                                "Overconfidence": round(m["overconfidence"], 4),
                                "Quality": quality,
                            }
                        )

                    summary_df = pd.DataFrame(rows)
                    st.dataframe(summary_df, use_container_width=True)

                    st.markdown("---")

                    if len(all_metrics) > 1:
                        comp_fig = st.session_state.calibration_analyzer.plot_calibration_comparison(
                            all_metrics
                        )
                        st.plotly_chart(comp_fig, use_container_width=True)

                    # Best calibrated model
                    best_model = min(all_metrics, key=lambda m: all_metrics[m]["ece"])
                    st.success(
                        f"🏆 **Best Calibrated Model:** {best_model} (ECE = {all_metrics[best_model]['ece']:.4f})"
                    )

                    # CSV export
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Metrics CSV",
                        csv,
                        "calibration_metrics.csv",
                        "text/csv",
                    )
                else:
                    st.warning("No models with probability output found.")

        # TAB 3 – Per-Class Calibration
        with tab3:
            st.subheader("🔬 Per-Class Calibration Analysis")
            st.markdown(
                "Examines calibration for each class independently (one-vs-rest)."
            )

            col1, col2 = st.columns(2)
            with col1:
                model_cls = st.selectbox(
                    "Select model:", model_options, key="cal_model_3"
                )
            with col2:
                dataset_cls = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_3"
                )

            if st.button(
                "🔬 Analyse Per-Class Calibration", type="primary", key="btn_cal_3"
            ):
                model = st.session_state.model_trainer.trained_models[model_cls]
                X = data["X_val"] if dataset_cls == "Validation" else data["X_test"]
                y = np.asarray(
                    data["y_val"] if dataset_cls == "Validation" else data["y_test"]
                )

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    class_names = (
                        [str(c) for c in model.classes_]
                        if hasattr(model, "classes_")
                        else None
                    )

                    per_class_df = st.session_state.calibration_analyzer.compute_per_class_calibration(
                        y, probs, class_names=class_names
                    )

                    st.dataframe(per_class_df, use_container_width=True)

                    st.markdown("---")

                    # Bar chart of ECE per class
                    fig_cls = go.Figure(
                        go.Bar(
                            x=per_class_df["Class"],
                            y=per_class_df["ECE"],
                            marker=dict(
                                color=per_class_df["ECE"],
                                colorscale="Reds",
                                showscale=True,
                            ),
                            text=per_class_df["ECE"].round(4),
                            textposition="auto",
                        )
                    )
                    fig_cls.update_layout(
                        title="Per-Class ECE",
                        xaxis_title="Class",
                        yaxis_title="ECE (lower = better)",
                        height=380,
                    )
                    st.plotly_chart(fig_cls, use_container_width=True)

                    worst_cls = per_class_df.loc[per_class_df["ECE"].idxmax(), "Class"]
                    st.info(
                        f"📌 Worst-calibrated class: **{worst_cls}** — consider targeted recalibration."
                    )
                else:
                    st.error(f"❌ {model_cls} does not support `predict_proba`.")

        # TAB 4 – Temperature Scaling
        with tab4:
            st.subheader("🌡️ Temperature Scaling")
            st.markdown(
                "Temperature scaling divides the log-probabilities (logits) by a constant *T* "
                "before re-applying softmax. "
                "- **T > 1** → softer distribution (less confident)  \n"
                "- **T < 1** → sharper distribution (more confident)  \n"
                "- **T = 1** → no change"
            )

            col1, col2 = st.columns(2)
            with col1:
                model_temp = st.selectbox(
                    "Select model:", model_options, key="cal_model_4"
                )
            with col2:
                dataset_temp = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_4"
                )

            temperature = st.slider(
                "Temperature (T):", 0.1, 5.0, 1.0, 0.1, key="temp_slider"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                auto_find = st.button("🔍 Find Optimal Temperature", key="btn_autotemp")
            with col_b:
                apply_btn = st.button(
                    "🌡️ Apply & Compare", type="primary", key="btn_applytemp"
                )

            if auto_find or apply_btn:
                model = st.session_state.model_trainer.trained_models[model_temp]
                X = data["X_val"] if dataset_temp == "Validation" else data["X_test"]
                y = np.asarray(
                    data["y_val"] if dataset_temp == "Validation" else data["y_test"]
                )

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)

                    if auto_find:
                        with st.spinner("Searching for optimal temperature..."):
                            opt_t = st.session_state.calibration_analyzer.find_optimal_temperature(
                                y, probs
                            )
                        st.success(f"✅ Optimal temperature found: **T = {opt_t}**")
                        temperature = opt_t

                    # Original metrics
                    orig_metrics = st.session_state.calibration_analyzer.compute_calibration_metrics(
                        y, probs
                    )
                    # Scaled metrics
                    scaled_probs = (
                        st.session_state.calibration_analyzer.apply_temperature_scaling(
                            probs, temperature
                        )
                    )
                    scaled_metrics = st.session_state.calibration_analyzer.compute_calibration_metrics(
                        y, scaled_probs
                    )

                    st.markdown("---")
                    st.subheader("Before vs After Temperature Scaling")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        delta_ece = scaled_metrics["ece"] - orig_metrics["ece"]
                        st.metric(
                            "ECE", f"{scaled_metrics['ece']:.4f}", f"{delta_ece:+.4f}"
                        )
                    with c2:
                        delta_mce = scaled_metrics["mce"] - orig_metrics["mce"]
                        st.metric(
                            "MCE", f"{scaled_metrics['mce']:.4f}", f"{delta_mce:+.4f}"
                        )
                    with c3:
                        delta_b = (
                            scaled_metrics["brier_score"] - orig_metrics["brier_score"]
                        )
                        st.metric(
                            "Brier Score",
                            f"{scaled_metrics['brier_score']:.4f}",
                            f"{delta_b:+.4f}",
                        )

                    st.markdown("---")

                    col_left, col_right = st.columns(2)
                    with col_left:
                        fig_orig = st.session_state.calibration_analyzer.plot_calibration_curve(
                            orig_metrics, f"{model_temp} (Original)"
                        )
                        st.plotly_chart(fig_orig, use_container_width=True)
                    with col_right:
                        fig_scaled = st.session_state.calibration_analyzer.plot_calibration_curve(
                            scaled_metrics, f"{model_temp} (T={temperature})"
                        )
                        st.plotly_chart(fig_scaled, use_container_width=True)

                    q_orig = (
                        st.session_state.calibration_analyzer.get_calibration_quality(
                            orig_metrics["ece"]
                        )
                    )
                    q_scaled = (
                        st.session_state.calibration_analyzer.get_calibration_quality(
                            scaled_metrics["ece"]
                        )
                    )
                    st.info(
                        f"Calibration quality: **{q_orig}** → **{q_scaled}** (T = {temperature})"
                    )
                else:
                    st.error(f"❌ {model_temp} does not support `predict_proba`.")

elif module == "7️⃣ Model Comparison":
    st.header("7️⃣ Model Comparison Module")
    st.markdown(
        "Side-by-side evaluation of all trained models across performance, "
        "robustness, and calibration dimensions."
    )

    comparator = st.session_state.model_comparator
    trainer = st.session_state.model_trainer

    if not trainer.trained_models:
        st.warning(
            "⚠️ No trained models found. Please train at least one model in Module 2."
        )
    else:
        # ── Dataset selector ──────────────────────────────────────────────
        available_datasets = set()
        for model_metrics in trainer.metrics.values():
            available_datasets.update(model_metrics.keys())

        dataset_choice = st.selectbox(
            "Evaluate on dataset:",
            sorted(available_datasets) if available_datasets else ["Test"],
            index=0,
        )

        # ── Compile metrics ───────────────────────────────────────────────
        metrics_dict = comparator.compile_performance_metrics(trainer, dataset_choice)
        models_with_data = [m for m, v in metrics_dict.items() if v["has_data"]]

        if not models_with_data:
            st.warning(
                f"⚠️ No evaluation results for the '{dataset_choice}' dataset. "
                "Run Module 2 evaluation first."
            )
        else:
            # ── Stress robustness ─────────────────────────────────────────
            stress_results = st.session_state.get("batch_stress_results_by_model", {})
            robustness_scores: dict[str, float] = {}
            if stress_results:
                for model_name, model_stress in stress_results.items():
                    if model_name in models_with_data:
                        drops = [
                            r.get("performance_drop", 0)
                            for r in model_stress.values()
                            if isinstance(r, dict)
                        ]
                        avg_drop = float(np.mean(drops)) if drops else 0.0
                        robustness_scores[model_name] = round(
                            max(0.0, 100.0 - avg_drop * 100), 2
                        )

            # ── Calibration ECE ───────────────────────────────────────────
            cal_ece: dict[str, float] = {}
            cal_analyzer = st.session_state.calibration_analyzer
            dm = st.session_state.data_manager
            if dm.X_test is not None and dm.y_test is not None:
                for model_name in models_with_data:
                    try:
                        _, probs = trainer.predict(model_name, dm.X_test)
                        cal_metrics = cal_analyzer.compute_calibration_metrics(
                            dm.y_test, probs
                        )
                        cal_ece[model_name] = cal_metrics["ece"]
                    except Exception:
                        pass

            # ── Composite score ───────────────────────────────────────────
            composite = comparator.compute_composite_score(
                metrics_dict,
                robustness_dict=robustness_scores or None,
                calibration_ece=cal_ece or None,
            )

            # ── Tabs ──────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "📊 Performance",
                    "🕸️ Radar Chart",
                    "🗂️ Confusion Matrices",
                    "🛡️ Robustness",
                    "🏆 Best Model",
                ]
            )

            # ─── TAB 1: Performance metrics ───────────────────────────────
            with tab1:
                st.subheader("📊 Performance Metrics Comparison")

                # Table
                df_cmp = comparator.build_comparison_df(metrics_dict)
                if not df_cmp.empty:
                    st.dataframe(df_cmp, use_container_width=True)

                st.markdown("---")

                # Grouped bar chart
                fig_bar = comparator.plot_metrics_bar(metrics_dict)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Delta highlights
                st.subheader("📐 Quick Stats")
                cols_stat = st.columns(len(models_with_data))
                for i, model_name in enumerate(models_with_data):
                    v = metrics_dict[model_name]
                    with cols_stat[i]:
                        st.metric(f"**{model_name}**", "")
                        st.metric("Accuracy", f"{v['accuracy']:.3f}")
                        st.metric("F1 Score", f"{v['f1']:.3f}")

            # ─── TAB 2: Radar ─────────────────────────────────────────────
            with tab2:
                st.subheader("🕸️ Performance Radar Chart")
                st.markdown(
                    "Each axis represents a performance metric (0–1). "
                    "Larger area = better overall performance."
                )
                fig_radar = comparator.plot_radar(metrics_dict)
                st.plotly_chart(fig_radar, use_container_width=True)

                # Small table reminder
                df_cmp2 = comparator.build_comparison_df(metrics_dict)
                with st.expander("📋 Underlying values"):
                    st.dataframe(df_cmp2, use_container_width=True)

            # ─── TAB 3: Confusion matrices ────────────────────────────────
            with tab3:
                st.subheader("🗂️ Confusion Matrices")
                cms = comparator.get_confusion_matrices(trainer, dataset_choice)

                if not cms:
                    st.info("No confusion matrix data available for this dataset.")
                else:
                    n_models = len(cms)
                    cols_cm = st.columns(min(n_models, 3))
                    for idx, (model_name, cm_data) in enumerate(cms.items()):
                        col = cols_cm[idx % 3]
                        with col:
                            fig_cm = comparator.plot_confusion_matrix(
                                cm_data["cm"], model_name
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

                    # Per-class accuracy table
                    st.markdown("---")
                    st.subheader("Per-Class Accuracy")
                    class_rows = []
                    for model_name, cm_data in cms.items():
                        cm_arr = cm_data["cm"]
                        per_class = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1)
                        for cls_idx, acc in enumerate(per_class):
                            class_rows.append(
                                {
                                    "Model": model_name,
                                    "Class": str(cls_idx),
                                    "Class Accuracy": f"{acc:.3f}",
                                }
                            )
                    if class_rows:
                        df_cls = pd.DataFrame(class_rows)
                        pivot = df_cls.pivot(
                            index="Class", columns="Model", values="Class Accuracy"
                        )
                        st.dataframe(pivot, use_container_width=True)

            # ─── TAB 4: Robustness ────────────────────────────────────────
            with tab4:
                st.subheader("🛡️ Stress Robustness Comparison")

                if not robustness_scores:
                    st.info(
                        "ℹ️ No batch stress test results found. "
                        "Run batch stress tests in Module 4 to populate this section."
                    )
                else:
                    fig_rob = comparator.plot_robustness_comparison(robustness_scores)
                    st.plotly_chart(fig_rob, use_container_width=True)

                    # Score table
                    score_df = pd.DataFrame(
                        [
                            {
                                "Model": m,
                                "Robustness Score": f"{s:.1f} / 100",
                                "Rating": (
                                    "✅ High"
                                    if s >= 70
                                    else "⚠️ Medium" if s >= 40 else "❌ Low"
                                ),
                            }
                            for m, s in sorted(
                                robustness_scores.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        ]
                    )
                    st.dataframe(score_df, use_container_width=True)

                st.markdown("---")
                st.subheader("📐 Calibration (ECE)")
                if not cal_ece:
                    st.info(
                        "ℹ️ Calibration ECE not available. "
                        "Ensure test data is prepared and models are evaluated."
                    )
                else:
                    ece_df = pd.DataFrame(
                        [
                            {
                                "Model": m,
                                "ECE": f"{e:.4f}",
                                "Quality": cal_analyzer.get_calibration_quality(e),
                            }
                            for m, e in sorted(cal_ece.items(), key=lambda x: x[1])
                        ]
                    )
                    st.dataframe(ece_df, use_container_width=True)

            # ─── TAB 5: Best model ────────────────────────────────────────
            with tab5:
                st.subheader("🏆 Best Model Recommendation")

                if not composite:
                    st.warning("No composite scores available.")
                else:
                    # Composite bar chart
                    fig_comp = comparator.plot_composite_scores(composite)
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # Recommendation
                    recommendation = comparator.recommend_best_model(
                        metrics_dict, composite, robustness_scores or None
                    )
                    st.success(recommendation["reason"])

                    # Ranking table
                    st.markdown("### 📋 Full Model Ranking")
                    ranking_rows = []
                    for rank, model_name in enumerate(
                        recommendation["ranking"], start=1
                    ):
                        v = metrics_dict[model_name]
                        row = {
                            "Rank": rank,
                            "Model": model_name,
                            "Composite Score": f"{composite[model_name]:.1f}",
                            "Accuracy": f"{v['accuracy']:.4f}",
                            "F1 Score": f"{v['f1']:.4f}",
                        }
                        if robustness_scores and model_name in robustness_scores:
                            row["Robustness"] = f"{robustness_scores[model_name]:.1f}"
                        if cal_ece and model_name in cal_ece:
                            row["ECE"] = f"{cal_ece[model_name]:.4f}"
                        ranking_rows.append(row)

                    ranking_df = pd.DataFrame(ranking_rows).set_index("Rank")
                    st.dataframe(ranking_df, use_container_width=True)

                    # Composite score breakdown
                    st.markdown("### 📖 Score Breakdown")
                    st.markdown(
                        """
| Component | Weight | Source |
|-----------|--------|--------|
| Performance (Accuracy + F1) | 50 % | Module 2 evaluation |
| Stress Robustness | 25 % | Module 4 batch test |
| Calibration (1 – ECE) | 25 % | Computed from test predictions |

> If robustness or calibration data is unavailable the missing weight shifts to the
> Performance component automatically.
                        """
                    )

elif module == "8️⃣ Reliability Scoring":
    st.header("8️⃣ Reliability Scoring Module")
    st.markdown(
        "Unified reliability score per model — combining performance, calibration, "
        "robustness, and confidence quality into a single 0–100 score with letter grade."
    )

    scorer = st.session_state.reliability_scorer
    trainer = st.session_state.model_trainer
    cal_an = st.session_state.calibration_analyzer
    dm = st.session_state.data_manager

    if not trainer.trained_models:
        st.warning("⚠️ No trained models found. Please train models in Module 2 first.")
    else:
        # ── Dataset selector ──────────────────────────────────────────────
        available_datasets = set()
        for model_metrics in trainer.metrics.values():
            available_datasets.update(model_metrics.keys())
        dataset_choice = st.selectbox(
            "Evaluate on dataset:",
            sorted(available_datasets) if available_datasets else ["Test"],
            key="rel_dataset",
        )

        # ── Gather calibration ECE ────────────────────────────────────────
        cal_ece: dict = {}
        if dm.X_test is not None and dm.y_test is not None:
            for mn in trainer.trained_models:
                try:
                    _, probs = trainer.predict(mn, dm.X_test)
                    cal_metrics = cal_an.compute_calibration_metrics(dm.y_test, probs)
                    cal_ece[mn] = cal_metrics["ece"]
                except Exception:
                    pass

        # ── Gather entropy & HCE rate ─────────────────────────────────────
        entropy_data: dict = {}
        hce_data: dict = {}
        if dm.X_test is not None and dm.y_test is not None:
            from utils.metrics import (
                get_prediction_entropy,
                identify_high_confidence_errors,
            )

            for mn in trainer.trained_models:
                try:
                    preds, probs = trainer.predict(mn, dm.X_test)
                    entropies = get_prediction_entropy(probs)
                    entropy_data[mn] = float(np.mean(entropies))
                    hce_df = identify_high_confidence_errors(
                        dm.y_test, preds, probs, threshold=0.9
                    )
                    total_preds = len(preds)
                    hce_data[mn] = len(hce_df) / total_preds if total_preds > 0 else 0.0
                except Exception:
                    pass

        # ── Stress robustness ─────────────────────────────────────────────
        stress_results = st.session_state.get("batch_stress_results_by_model", {})

        # ── Number of classes ─────────────────────────────────────────────
        n_classes = 2
        if dm.y_test is not None:
            try:
                n_classes = len(np.unique(dm.y_test))
            except Exception:
                pass

        # ── Compute scores ────────────────────────────────────────────────
        scores_dict = scorer.score_all_models(
            trainer,
            dataset_name=dataset_choice,
            stress_results=stress_results or None,
            calibration_ece=cal_ece or None,
            entropy_data=entropy_data or None,
            hce_data=hce_data or None,
            n_classes=n_classes,
        )

        if not scores_dict:
            st.warning("No score data available. Evaluate models in Module 2 first.")
        else:
            # ── Data-availability notice ──────────────────────────────────
            all_missing = set()
            for sd in scores_dict.values():
                all_missing.update(sd["missing_components"])
            if all_missing:
                st.info(
                    f"ℹ️ Some components defaulted to neutral (12.5 pts) due to "
                    f"missing data: **{', '.join(sorted(all_missing))}**.  \n"
                    "Run stress tests (Module 4) and ensure test data is prepared "
                    "to get full scores."
                )

            # ── Tabs ──────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "📊 Score Overview",
                    "🕸️ Component Radar",
                    "🏅 Model Detail",
                    "💡 Recommendations",
                ]
            )

            # ─── TAB 1: Score overview ────────────────────────────────────
            with tab1:
                st.subheader("📊 Reliability Score Overview")

                # Gauge row
                n_models = len(scores_dict)
                gauge_cols = st.columns(min(n_models, 4))
                for idx, (mname, sd) in enumerate(scores_dict.items()):
                    with gauge_cols[idx % 4]:
                        fig_g = scorer.plot_gauge(sd)
                        st.plotly_chart(
                            fig_g,
                            use_container_width=True,
                            key=f"rel_gauge_overview_{mname}",
                        )

                st.markdown("---")

                # Stacked bar
                fig_stack = scorer.plot_stacked_bar(scores_dict)
                st.plotly_chart(
                    fig_stack, use_container_width=True, key="rel_stacked_bar"
                )

                st.markdown("---")

                # Total bar
                fig_total = scorer.plot_total_bar(scores_dict)
                st.plotly_chart(
                    fig_total, use_container_width=True, key="rel_total_bar"
                )

                st.markdown("---")

                # Summary table
                st.subheader("📋 Score Summary Table")
                df_sum = scorer.build_summary_df(scores_dict)
                st.dataframe(df_sum, use_container_width=True)

                # Grade legend
                with st.expander("📖 Grade Legend"):
                    st.markdown(
                        """
| Grade | Score Range | Meaning |
|-------|-------------|----------|
| A+    | 90–100      | Excellent — production ready |
| A     | 80–89       | Good — minor improvements possible |
| B     | 70–79       | Acceptable — monitor closely |
| C     | 60–69       | Below average — improvements needed |
| D     | 50–59       | Poor — significant issues |
| F     | 0–49        | Failing — major rework required |
"""
                    )

            # ─── TAB 2: Component radar ───────────────────────────────────
            with tab2:
                st.subheader("🕸️ Component Score Radar")
                st.markdown(
                    "Each axis shows normalised component score (0 = worst, 1 = best). "
                    "Each segment contributes up to 25 points."
                )
                fig_radar = scorer.plot_component_radar(scores_dict)
                st.plotly_chart(fig_radar, use_container_width=True, key="rel_radar")

                # Component breakdown explanation
                with st.expander("📖 Component Definitions"):
                    st.markdown(
                        """
| Component | Max Pts | How Calculated |
|-----------|---------|----------------|
| **Performance** | 25 | Mean of Accuracy & F1 × 25 |
| **Calibration** | 25 | `(1 − ECE/0.5) × 25` — lower ECE → more pts |
| **Robustness** | 25 | `(1 − mean_drop/0.5) × 25` — less drop → more pts |
| **Confidence** | 25 | 12.5 × (1 − norm_entropy) + 12.5 × (1 − HCE_rate) |
"""
                    )

            # ─── TAB 3: Model detail ──────────────────────────────────────
            with tab3:
                st.subheader("🏅 Individual Model Detail")
                model_sel = st.selectbox(
                    "Select model:",
                    list(scores_dict.keys()),
                    key="rel_model_sel",
                )
                sd = scores_dict[model_sel]

                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_g = scorer.plot_gauge(sd)
                    st.plotly_chart(
                        fig_g,
                        use_container_width=True,
                        key=f"rel_gauge_detail_{model_sel}",
                    )

                with c2:
                    st.markdown(f"### Grade: **{sd['grade']}**")
                    st.markdown(f"**Total Score:** {sd['total']} / 100")
                    st.markdown("---")
                    st.markdown(f"- 🎯 Performance:  **{sd['performance']:.1f}** / 25")
                    st.markdown(f"- 📐 Calibration:  **{sd['calibration']:.1f}** / 25")
                    st.markdown(f"- 🛡️ Robustness:   **{sd['robustness']:.1f}** / 25")
                    st.markdown(f"- 🔮 Confidence:   **{sd['confidence']:.1f}** / 25")

                    if sd["missing_components"]:
                        st.warning(
                            f"⚠️ Defaulted (12.5 pts each): "
                            f"{', '.join(sd['missing_components'])}"
                        )

                st.markdown("---")
                st.subheader("Raw Inputs")
                det = sd["details"]
                det_df = pd.DataFrame(
                    [
                        {
                            "Metric": "Accuracy",
                            "Value": (
                                f"{det['accuracy']:.4f}"
                                if det["accuracy"] is not None
                                else "N/A"
                            ),
                        },
                        {
                            "Metric": "F1 Score",
                            "Value": (
                                f"{det['f1']:.4f}" if det["f1"] is not None else "N/A"
                            ),
                        },
                        {
                            "Metric": "ECE",
                            "Value": (
                                f"{det['ece']:.4f}" if det["ece"] is not None else "N/A"
                            ),
                        },
                        {
                            "Metric": "Avg Drop",
                            "Value": (
                                f"{det['avg_drop']:.4f}"
                                if det["avg_drop"] is not None
                                else "N/A"
                            ),
                        },
                        {
                            "Metric": "Avg Entropy",
                            "Value": (
                                f"{det['avg_entropy']:.4f}"
                                if det["avg_entropy"] is not None
                                else "N/A"
                            ),
                        },
                        {
                            "Metric": "HCE Rate",
                            "Value": (
                                f"{det['hce_rate']:.4f}"
                                if det["hce_rate"] is not None
                                else "N/A"
                            ),
                        },
                    ]
                )
                st.dataframe(det_df, use_container_width=True, hide_index=True)

            # ─── TAB 4: Recommendations ───────────────────────────────────
            with tab4:
                st.subheader("💡 Recommendations")
                model_rec = st.selectbox(
                    "Select model:",
                    list(scores_dict.keys()),
                    key="rel_rec_model",
                )
                recs = scorer.generate_recommendations(scores_dict[model_rec])
                for rec in recs:
                    st.markdown(rec)

                st.markdown("---")
                st.subheader("🏆 Best Model by Reliability")
                best_model = max(scores_dict, key=lambda m: scores_dict[m]["total"])
                best_sd = scores_dict[best_model]
                st.success(
                    f"**{best_model}** has the highest reliability score: "
                    f"**{best_sd['total']:.1f}/100** (Grade **{best_sd['grade']}**)"
                )

elif module == "9️⃣ Visualization & Reports":
    st.header("9️⃣ Visualization & Reports Module")
    st.markdown(
        "Aggregated dashboard across all modules with one-click export "
        "to CSV, JSON, HTML, and PDF."
    )

    rgen = st.session_state.report_generator
    trainer = st.session_state.model_trainer
    dm = st.session_state.data_manager
    cal_an = st.session_state.calibration_analyzer
    scorer = st.session_state.reliability_scorer

    if not trainer.trained_models:
        st.warning("⚠️ No trained models found. Complete Modules 1–2 first.")
    else:
        # ── Dataset selector ──────────────────────────────────────────────
        available_ds = set()
        for mm in trainer.metrics.values():
            available_ds.update(mm.keys())
        rep_dataset = st.selectbox(
            "Report dataset:",
            sorted(available_ds) if available_ds else ["Test"],
            key="rep_dataset",
        )

        # ── Gather calibration ECE ────────────────────────────────────────
        rep_cal_ece: dict = {}
        if dm.X_test is not None and dm.y_test is not None:
            for mn in trainer.trained_models:
                try:
                    _, probs = trainer.predict(mn, dm.X_test)
                    cm_r = cal_an.compute_calibration_metrics(dm.y_test, probs)
                    rep_cal_ece[mn] = cm_r["ece"]
                except Exception:
                    pass

        # ── Gather reliability scores ────────────────────────────────────
        stress_res = st.session_state.get("batch_stress_results_by_model", {})
        n_cls = 2
        if dm.y_test is not None:
            try:
                n_cls = len(np.unique(dm.y_test))
            except Exception:
                pass

        rep_entropy: dict = {}
        rep_hce: dict = {}
        if dm.X_test is not None and dm.y_test is not None:
            for mn in trainer.trained_models:
                try:
                    preds_r, probs_r = trainer.predict(mn, dm.X_test)
                    ents = get_prediction_entropy(probs_r)
                    rep_entropy[mn] = float(np.mean(ents))
                    hce_df = identify_high_confidence_errors(
                        dm.y_test, preds_r, probs_r, threshold=0.9
                    )
                    rep_hce[mn] = len(hce_df) / max(len(preds_r), 1)
                except Exception:
                    pass

        rel_scores = scorer.score_all_models(
            trainer,
            dataset_name=rep_dataset,
            stress_results=stress_res or None,
            calibration_ece=rep_cal_ece or None,
            entropy_data=rep_entropy or None,
            hce_data=rep_hce or None,
            n_classes=n_cls,
        )

        # ── Compile report ────────────────────────────────────────────────
        report = rgen.compile_report(
            trainer,
            dm,
            stress_results=stress_res or None,
            cal_ece=rep_cal_ece or None,
            reliability_scores=rel_scores or None,
            dataset_name=rep_dataset,
        )
        smry = report["summary"]

        # ── KPI cards ─────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🧠 Models Trained", smry["num_models"])
        k2.metric("🖥️ Stress Tested", smry["num_stressed"])
        k3.metric("🏆 Best Performance", smry["best_performance"])
        k4.metric("🛡️ Most Robust", smry["most_robust"])
        k5.metric("⭐ Best Reliability", smry["best_reliability"])

        st.markdown("---")

        # ── Tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Dashboard",
                "📋 Summary Tables",
                "🖼️ Charts Gallery",
                "💾 Export",
            ]
        )

        # ─── TAB 1: Dashboard ──────────────────────────────────────────
        with tab1:
            st.subheader("📊 Performance Overview")
            fig_perf = rgen.plot_performance_overview(report)
            st.plotly_chart(fig_perf, use_container_width=True, key="rep_perf")

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📐 Calibration (ECE)")
                fig_cal = rgen.plot_calibration_bar(report)
                st.plotly_chart(fig_cal, use_container_width=True, key="rep_cal")
            with c_right:
                st.subheader("⭐ Reliability Radar")
                fig_rad = rgen.plot_radar_all(report)
                st.plotly_chart(fig_rad, use_container_width=True, key="rep_radar")

            st.subheader("🛡️ Robustness Heatmap")
            fig_rob = rgen.plot_robustness_heatmap(report)
            st.plotly_chart(fig_rob, use_container_width=True, key="rep_heatmap")

            # Reliability gauges
            fig_gauges = rgen.plot_reliability_gauge_row(report)
            if fig_gauges:
                st.subheader("🏅 Reliability Gauges")
                st.plotly_chart(fig_gauges, use_container_width=True, key="rep_gauges")

        # ─── TAB 2: Summary Tables ────────────────────────────────────
        with tab2:
            st.subheader("🎯 Performance Metrics")
            if report["performance"]:
                st.dataframe(
                    pd.DataFrame(report["performance"]).set_index("Model"),
                    use_container_width=True,
                )
            else:
                st.info("No performance data available.")

            st.markdown("---")
            st.subheader("🛡️ Robustness Results")
            if report["robustness"]:
                st.dataframe(
                    pd.DataFrame(report["robustness"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No stress test data. Run batch tests in Module 4.")

            st.markdown("---")
            st.subheader("📐 Calibration")
            if report["calibration"]:
                st.dataframe(
                    pd.DataFrame(report["calibration"]).set_index("Model"),
                    use_container_width=True,
                )
            else:
                st.info("No calibration data. Ensure test data is prepared.")

            st.markdown("---")
            st.subheader("⭐ Reliability Scores")
            if report["reliability"]:
                st.dataframe(
                    pd.DataFrame(report["reliability"]).set_index("Model"),
                    use_container_width=True,
                )
            else:
                st.info("No reliability data available.")

        # ─── TAB 3: Charts Gallery ────────────────────────────────────
        with tab3:
            st.subheader("🖼️ Full Charts Gallery")
            st.markdown("All key visualisations from every module in one place.")

            with st.expander("🎯 Performance Bar Chart", expanded=True):
                st.plotly_chart(
                    rgen.plot_performance_overview(report),
                    use_container_width=True,
                    key="gal_perf",
                )
            with st.expander("🛡️ Robustness Heatmap", expanded=True):
                if report["robustness"]:
                    st.plotly_chart(
                        rgen.plot_robustness_heatmap(report),
                        use_container_width=True,
                        key="gal_rob",
                    )
                else:
                    st.info("No stress data available.")
            with st.expander("📐 Calibration ECE", expanded=True):
                if report["calibration"]:
                    st.plotly_chart(
                        rgen.plot_calibration_bar(report),
                        use_container_width=True,
                        key="gal_cal",
                    )
                else:
                    st.info("No calibration data available.")
            with st.expander("⭐ Reliability Scores", expanded=True):
                if report["reliability"]:
                    st.plotly_chart(
                        rgen.plot_radar_all(report),
                        use_container_width=True,
                        key="gal_radar",
                    )
                else:
                    st.info("No reliability data available.")

        # ─── TAB 4: Export ─────────────────────────────────────────────
        with tab4:
            st.subheader("💾 Export Report")
            st.markdown(
                f"Report generated: **{report['generated_at']}** — "
                f"Dataset: **{report['dataset_name']}** — "
                f"Models: **{', '.join(report['models']) or 'none'}**"
            )
            st.markdown("---")

            col_csv, col_json, col_html, col_pdf = st.columns(4)

            with col_csv:
                st.markdown("### 📄 CSV")
                st.markdown("All tables in one CSV.")
                csv_bytes = rgen.export_all_csv(report)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"ml_report_{report['generated_at'][:10]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv",
                )

            with col_json:
                st.markdown("### {} JSON")
                st.markdown("Full structured data.")
                json_str = rgen.export_to_json(report)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name=f"ml_report_{report['generated_at'][:10]}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_json",
                )

            with col_html:
                st.markdown("### 🌐 HTML")
                st.markdown("Styled, self-contained HTML page.")
                html_str = rgen.export_to_html(report)
                st.download_button(
                    label="⬇️ Download HTML",
                    data=html_str.encode("utf-8"),
                    file_name=f"ml_report_{report['generated_at'][:10]}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html",
                )

            with col_pdf:
                st.markdown("### 📄 PDF")
                st.markdown("Formatted PDF report.")
                try:
                    pdf_bytes = rgen.export_to_pdf(report)
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"ml_report_{report['generated_at'][:10]}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

            st.markdown("---")
            st.subheader("📋 Individual Table Downloads")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                if report["performance"]:
                    st.download_button(
                        "Performance CSV",
                        rgen.export_performance_csv(report),
                        file_name="performance.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_perf_csv",
                    )
            with d2:
                if report["robustness"]:
                    st.download_button(
                        "Robustness CSV",
                        rgen.export_robustness_csv(report),
                        file_name="robustness.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_rob_csv",
                    )
            with d3:
                if report["reliability"]:
                    st.download_button(
                        "Reliability CSV",
                        rgen.export_reliability_csv(report),
                        file_name="reliability.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_rel_csv",
                    )
            with d4:
                st.download_button(
                    "Full JSON",
                    rgen.export_to_json(report).encode("utf-8"),
                    file_name="full_report.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_full_json",
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Status")
if hasattr(st.session_state, "data_loaded") and st.session_state.data_loaded:
    st.sidebar.success("✅ Data loaded")
else:
    st.sidebar.warning("⚠️ No data loaded")

if hasattr(st.session_state, "data_prepared") and st.session_state.data_prepared:
    st.sidebar.success("✅ Data prepared")
else:
    st.sidebar.warning("⚠️ Data not prepared")

if hasattr(st.session_state, "model_trained") and st.session_state.model_trained:
    trained_count = len(st.session_state.model_trainer.trained_models)
    st.sidebar.success(f"✅ {trained_count} model(s) trained")
else:
    st.sidebar.warning("⚠️ No models trained")
