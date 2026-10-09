from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.state import AppContext, StateKeys
from modules.calibration_module import CalibrationAnalyzer
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from views.reporters import render_errors


def render(ctx: AppContext) -> None:
    """Render the Calibration Analysis page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    calibration_analyzer = ctx.ensure_service(StateKeys.CALIBRATION_ANALYZER, CalibrationAnalyzer)
    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    model_trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)

    st.header("6️⃣ Calibration Analysis Module")
    st.markdown("**Purpose:** Assess how well model confidence scores reflect true accuracy")

    if not ctx.has_models():
        st.warning("⚠️ Please train at least one model in Module 2 first.")
    else:
        data = data_manager.get_data()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Calibration Curve",
                "📊 Calibration Metrics",
                "🔬 Per-Class Analysis",
                "🌡️ Temperature Scaling",
            ]
        )

        # ── shared model / dataset selector (rendered inside each tab) ──────────────
        model_options = list(model_trainer.trained_models.keys())

        # TAB 1 – Reliability Diagram
        with tab1:
            st.subheader("📈 Reliability Diagram")
            st.markdown(
                "A well-calibrated model follows the dashed diagonal. "
                "**Red** shading = overconfident, **blue** shading = underconfident."
            )

            col1, col2 = st.columns(2)
            with col1:
                model_cal = st.selectbox("Select model:", model_options, key="cal_model_1")
            with col2:
                dataset_cal = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_1"
                )

            n_bins_cal = st.slider("Number of bins:", 5, 20, 10, key="cal_bins_1")

            if st.button("📈 Generate Calibration Curve", type="primary", key="btn_cal_1"):
                with render_errors():
                    model = model_trainer.trained_models[model_cal]
                    X = data["X_val"] if dataset_cal == "Validation" else data["X_test"]
                    y = data["y_val"] if dataset_cal == "Validation" else data["y_test"]

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X)
                        y_arr = np.asarray(y)

                        metrics = calibration_analyzer.compute_calibration_metrics(
                            y_arr, probabilities, n_bins=n_bins_cal
                        )

                        quality = calibration_analyzer.get_calibration_quality(metrics["ece"])

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
                        fig = calibration_analyzer.plot_calibration_curve(metrics, model_cal)
                        st.plotly_chart(fig, width="stretch")

                        st.markdown("---")
                        hist_fig = calibration_analyzer.plot_confidence_histogram(
                            y_arr, probabilities
                        )
                        st.plotly_chart(hist_fig, width="stretch")
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

            if st.button("📊 Compute All Model Metrics", type="primary", key="btn_cal_2"):
                with render_errors():
                    X = data["X_val"] if dataset_cal2 == "Validation" else data["X_test"]
                    y = np.asarray(
                        data["y_val"] if dataset_cal2 == "Validation" else data["y_test"]
                    )

                    all_metrics = {}
                    for (
                        mname,
                        model,
                    ) in model_trainer.trained_models.items():
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X)
                            all_metrics[mname] = calibration_analyzer.compute_calibration_metrics(
                                y, probs
                            )

                    if all_metrics:
                        # Summary table
                        rows = []
                        for mname, m in all_metrics.items():
                            quality = calibration_analyzer.get_calibration_quality(m["ece"])
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
                        st.dataframe(summary_df, width="stretch")

                        st.markdown("---")

                        if len(all_metrics) > 1:
                            comp_fig = calibration_analyzer.plot_calibration_comparison(all_metrics)
                            st.plotly_chart(comp_fig, width="stretch")

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
            st.markdown("Examines calibration for each class independently (one-vs-rest).")

            col1, col2 = st.columns(2)
            with col1:
                model_cls = st.selectbox("Select model:", model_options, key="cal_model_3")
            with col2:
                dataset_cls = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_3"
                )

            if st.button("🔬 Analyse Per-Class Calibration", type="primary", key="btn_cal_3"):
                with render_errors():
                    model = model_trainer.trained_models[model_cls]
                    X = data["X_val"] if dataset_cls == "Validation" else data["X_test"]
                    y = np.asarray(data["y_val"] if dataset_cls == "Validation" else data["y_test"])

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)
                        class_names = (
                            [str(c) for c in model.classes_] if hasattr(model, "classes_") else None
                        )

                        per_class_df = calibration_analyzer.compute_per_class_calibration(
                            y, probs, class_names=class_names
                        )

                        st.dataframe(per_class_df, width="stretch")

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
                        st.plotly_chart(fig_cls, width="stretch")

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
                model_temp = st.selectbox("Select model:", model_options, key="cal_model_4")
            with col2:
                dataset_temp = st.radio(
                    "Dataset:", ["Validation", "Test"], horizontal=True, key="cal_ds_4"
                )

            temperature = st.slider("Temperature (T):", 0.1, 5.0, 1.0, 0.1, key="temp_slider")

            col_a, col_b = st.columns(2)
            with col_a:
                auto_find = st.button("🔍 Find Optimal Temperature", key="btn_autotemp")
            with col_b:
                apply_btn = st.button("🌡️ Apply & Compare", type="primary", key="btn_applytemp")

            if auto_find or apply_btn:
                with render_errors():
                    model = model_trainer.trained_models[model_temp]
                    X = data["X_val"] if dataset_temp == "Validation" else data["X_test"]
                    y = np.asarray(
                        data["y_val"] if dataset_temp == "Validation" else data["y_test"]
                    )

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)

                        if auto_find:
                            with st.spinner("Searching for optimal temperature..."):
                                opt_t = calibration_analyzer.find_optimal_temperature(y, probs)
                            st.success(f"✅ Optimal temperature found: **T = {opt_t}**")
                            temperature = opt_t

                        # Original metrics
                        orig_metrics = calibration_analyzer.compute_calibration_metrics(y, probs)
                        # Scaled metrics
                        scaled_probs = calibration_analyzer.apply_temperature_scaling(
                            probs, temperature
                        )
                        scaled_metrics = calibration_analyzer.compute_calibration_metrics(
                            y, scaled_probs
                        )

                        st.markdown("---")
                        st.subheader("Before vs After Temperature Scaling")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            delta_ece = scaled_metrics["ece"] - orig_metrics["ece"]
                            st.metric("ECE", f"{scaled_metrics['ece']:.4f}", f"{delta_ece:+.4f}")
                        with c2:
                            delta_mce = scaled_metrics["mce"] - orig_metrics["mce"]
                            st.metric("MCE", f"{scaled_metrics['mce']:.4f}", f"{delta_mce:+.4f}")
                        with c3:
                            delta_b = scaled_metrics["brier_score"] - orig_metrics["brier_score"]
                            st.metric(
                                "Brier Score",
                                f"{scaled_metrics['brier_score']:.4f}",
                                f"{delta_b:+.4f}",
                            )

                        st.markdown("---")

                        col_left, col_right = st.columns(2)
                        with col_left:
                            fig_orig = calibration_analyzer.plot_calibration_curve(
                                orig_metrics, f"{model_temp} (Original)"
                            )
                            st.plotly_chart(fig_orig, width="stretch")
                        with col_right:
                            fig_scaled = calibration_analyzer.plot_calibration_curve(
                                scaled_metrics, f"{model_temp} (T={temperature})"
                            )
                            st.plotly_chart(fig_scaled, width="stretch")

                        q_orig = calibration_analyzer.get_calibration_quality(orig_metrics["ece"])
                        q_scaled = calibration_analyzer.get_calibration_quality(
                            scaled_metrics["ece"]
                        )
                        st.info(
                            f"Calibration quality: **{q_orig}** → **{q_scaled}** (T = {temperature})"
                        )
                    else:
                        st.error(f"❌ {model_temp} does not support `predict_proba`.")


__all__ = ["render"]
