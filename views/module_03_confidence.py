from __future__ import annotations

import pandas as pd
import streamlit as st

from core.errors import FrameworkError
from core.state import AppContext, StateKeys
from modules.calibration_module import multiclass_brier
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from utils.metrics import (
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
from views.reporters import render_error


def render(ctx: AppContext) -> None:
    """Render the Prediction & Confidence page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    model_trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)

    st.header("3️⃣ Prediction & Confidence Module")
    st.markdown(
        "Analyze model predictions, confidence levels, and identify high-confidence errors."
    )

    # Check prerequisites
    #
    # See the identical note in ``module_04_stress.py``: both ``is None`` arms
    # were unreachable because app.py's bootstrap created the singletons up
    # front, so they become the §4.7 façade predicates.
    if not ctx.has_data():
        st.warning("⚠️ Please load and prepare data in Module 1 first.")
    elif not ctx.has_models():
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
            model_names = list(model_trainer.trained_models.keys())
            selected_model = st.selectbox("Select Model", model_names, key="pred_model_select")

            # Dataset selection
            dataset_choice = st.radio(
                "Select Dataset",
                ["Validation Set", "Test Set"],
                horizontal=True,
                key="pred_dataset_select",
            )

            if st.button("Generate Predictions", key="generate_predictions"):
                with st.spinner("Generating predictions..."):
                    model = model_trainer.trained_models[selected_model]

                    # Select dataset
                    if dataset_choice == "Validation Set":
                        X = data_manager.X_val
                        y = data_manager.y_val
                    else:
                        X = data_manager.X_test
                        y = data_manager.y_test

                    # conventions §7: a mutating action reports success **or** a
                    # rendered ``FrameworkError``.  ``model.predict`` on an
                    # unprepared split raises from the domain layer, so the
                    # domain calls are inside the ``try`` and the session-state
                    # write is in the ``else`` arm -- a rejected run can never
                    # leave a half-written ``predictions`` entry behind.
                    try:
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
                    except FrameworkError as exc:
                        render_error(exc)
                    else:
                        # Store in session state
                        ctx.store[StateKeys.PREDICTIONS] = {
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
            if ctx.store[StateKeys.PREDICTIONS]:
                pred_data = ctx.store[StateKeys.PREDICTIONS]

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
                        correct_conf = pred_data["confidence_df"][correct_mask]["Confidence"].mean()
                        st.metric("Avg Confidence (Correct)", f"{correct_conf:.2%}")

                # Show detailed predictions table
                st.markdown("#### Detailed Predictions")
                display_df = pred_data["confidence_df"].copy()
                display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df, width="stretch")

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

            if not ctx.store[StateKeys.PREDICTIONS]:
                st.info("👈 Generate predictions in the 'Predictions Overview' tab first.")
            else:
                pred_data = ctx.store[StateKeys.PREDICTIONS]

                st.markdown(f"**Analyzing:** {pred_data['model_name']} on {pred_data['dataset']}")

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
                st.plotly_chart(fig_hist, width="stretch")

                # Class-wise confidence analysis
                st.markdown("#### Confidence by Class")
                fig_class = plot_confidence_by_class(
                    pred_data["confidence_df"]["Confidence"].values,
                    pred_data["confidence_df"]["Predicted_Label"].values,
                    pred_data["confidence_df"]["True_Label"].values,
                )
                st.plotly_chart(fig_class, width="stretch")

                # Statistics by confidence level
                st.markdown("#### Statistics by Confidence Level")
                high_conf_mask = pred_data["confidence_df"]["Confidence"] >= confidence_threshold
                low_conf_mask = pred_data["confidence_df"]["Confidence"] < confidence_threshold

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

            if not ctx.store[StateKeys.PREDICTIONS]:
                st.info("👈 Generate predictions in the 'Predictions Overview' tab first.")
            else:
                pred_data = ctx.store[StateKeys.PREDICTIONS]

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
                    st.plotly_chart(fig_errors, width="stretch")

                    # Show error details
                    st.markdown("#### Error Details")
                    error_indices = hc_errors_info["indices"]
                    hc_errors = pred_data["confidence_df"].iloc[error_indices].copy()
                    error_display = hc_errors.copy()
                    error_display["Confidence"] = error_display["Confidence"].apply(
                        lambda x: f"{x:.2%}"
                    )
                    st.dataframe(error_display, width="stretch")

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

            if not ctx.store[StateKeys.PREDICTIONS]:
                st.info("👈 Generate predictions in the 'Predictions Overview' tab first.")
            else:
                pred_data = ctx.store[StateKeys.PREDICTIONS]

                # Calibration curve
                st.markdown("#### Calibration Curve")
                st.markdown("A well-calibrated model's confidence should match its accuracy.")

                fig_calib = plot_confidence_accuracy_curve(
                    pred_data["confidence_df"]["Confidence"].values,
                    pred_data["confidence_df"]["Correct"].values,
                )
                st.plotly_chart(fig_calib, width="stretch")

                # Brier score (single owner: modules.calibration_module, ADR-017)
                classes = getattr(
                    model_trainer.trained_models.get(pred_data["model_name"]),
                    "classes_",
                    None,
                )
                brier = multiclass_brier(
                    pred_data["y_true"],
                    pred_data["y_proba"],
                    classes=classes,
                )
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
                    st.metric(
                        "Avg Entropy",
                        f"{entropy_series.mean():.3f}",
                        help="Mean Shannon entropy in bits (base-2); higher = more uncertain",
                    )
                    st.metric(
                        "Max Entropy",
                        f"{entropy_series.max():.3f}",
                        help="Entropy in bits (base-2)",
                    )

                with col2:
                    st.metric(
                        "Min Entropy",
                        f"{entropy_series.min():.3f}",
                        help="Entropy in bits (base-2)",
                    )
                    st.metric(
                        "Std Entropy",
                        f"{entropy_series.std():.3f}",
                        help="Entropy in bits (base-2)",
                    )

                # Entropy distribution
                fig_entropy = plot_entropy_distribution(
                    entropy_series, pred_data["confidence_df"]["Correct"].values
                )
                st.plotly_chart(fig_entropy, width="stretch")

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
                most_uncertain["Entropy"] = most_uncertain["Entropy"].apply(lambda x: f"{x:.3f}")
                st.dataframe(most_uncertain, width="stretch")


__all__ = ["render"]
