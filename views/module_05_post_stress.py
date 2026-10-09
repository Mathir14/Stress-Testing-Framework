from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.errors import FrameworkError
from core.state import AppContext, StateKeys
from modules.post_stress_module import PostStressAnalyzer
from views.reporters import render_error


def render(ctx: AppContext) -> None:
    """Render the Post-Stress Evaluation page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    post_stress_analyzer = ctx.ensure_service(StateKeys.POST_STRESS_ANALYZER, PostStressAnalyzer)

    st.header("5️⃣ Post-Stress Evaluation Module")
    st.markdown("**Purpose:** Deep analysis of stress test results and robustness assessment")

    # Check if stress tests have been run
    tested_models = list(ctx.store.get(StateKeys.BATCH_STRESS_RESULTS_BY_MODEL) or {}.keys())
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
            model_results = ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL][selected_model]
            post_stress_analyzer.add_batch_results(model_results, selected_model)

            # Calculate robustness score
            if st.button("🔄 Calculate Robustness Score", type="primary"):
                with st.spinner("Analyzing stress test results..."):
                    # conventions §7: a mutating action reports success **or** a
                    # rendered ``FrameworkError``.  The analyzer raises
                    # ``FrameworkError`` when the selected model has no batch
                    # results; before this guard that escaped as a raw
                    # Streamlit traceback instead of a rendered message.
                    try:
                        score = post_stress_analyzer.calculate_robustness_score(selected_model)
                    except FrameworkError as exc:
                        render_error(exc)
                        score = None

                    if score is not None:
                        st.success("✅ Analysis complete!")

                        # Display overall score
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)

                        scores = post_stress_analyzer.robustness_scores[selected_model]

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

                        radar_fig = post_stress_analyzer.plot_robustness_radar(selected_model)
                        if radar_fig:
                            st.plotly_chart(radar_fig, width="stretch")

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
                        st.write(f"Consistency score: {scores['performance_consistency']:.1f}%")

        # TAB 2: Vulnerability Analysis
        with tab2:
            st.subheader("🎯 Vulnerability Analysis")

            selected_model_vuln = st.selectbox(
                "Select model:",
                tested_models,
                key="vuln_model",
            )

            # Load the correct per-model results
            model_results_vuln = ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL][
                selected_model_vuln
            ]
            post_stress_analyzer.add_batch_results(model_results_vuln, selected_model_vuln)

            vuln_df = post_stress_analyzer.get_vulnerability_analysis(selected_model_vuln)

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
                heatmap_fig = post_stress_analyzer.plot_vulnerability_heatmap(selected_model_vuln)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, width="stretch")

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

                styled_df = vuln_df.style.map(color_severity, subset=["Severity"])

                st.dataframe(styled_df, width="stretch")

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
            model_results_type = ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL][
                selected_model_type
            ]
            post_stress_analyzer.add_batch_results(model_results_type, selected_model_type)

            type_df = post_stress_analyzer.get_stress_type_summary(selected_model_type)

            if not type_df.empty:
                st.markdown("**Performance by Stress Category** (grouped by stress type)")

                st.dataframe(type_df, width="stretch")

                st.markdown("---")

                st.subheader("📊 Average Performance Drop by Category")

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

                st.plotly_chart(fig, width="stretch")

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

                st.plotly_chart(fig2, width="stretch")

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
                        # conventions §7: a mutating action reports success **or**
                        # a rendered ``FrameworkError`` (see the Calculate
                        # Robustness Score handler above).
                        try:
                            # Load the correct per-model results for each
                            for model in models_to_compare:
                                per_model_results = ctx.store[
                                    StateKeys.BATCH_STRESS_RESULTS_BY_MODEL
                                ][model]
                                post_stress_analyzer.add_batch_results(per_model_results, model)

                            # Comparison chart
                            comparison_fig = post_stress_analyzer.compare_model_robustness(
                                models_to_compare
                            )
                        except FrameworkError as exc:
                            render_error(exc)
                            comparison_fig = None

                        if comparison_fig:
                            st.plotly_chart(comparison_fig, width="stretch")

                            # Summary table
                            st.markdown("---")
                            st.subheader("📋 Robustness Score Summary")

                            summary_data = []
                            for model in models_to_compare:
                                post_stress_analyzer.calculate_robustness_score(model)
                                scores = post_stress_analyzer.robustness_scores[model]
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
                            st.dataframe(summary_df, width="stretch")

                            best_idx = summary_df["Overall Score"].astype(float).idxmax()
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
            model_results_rec = ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL][
                selected_model_rec
            ]
            post_stress_analyzer.add_batch_results(model_results_rec, selected_model_rec)

            if st.button("🔍 Generate Recommendations", type="primary"):
                with st.spinner("Analyzing and generating recommendations..."):
                    # conventions §7: a mutating action reports success **or** a
                    # rendered ``FrameworkError``.
                    try:
                        recommendations = post_stress_analyzer.get_recommendations(
                            selected_model_rec
                        )
                    except FrameworkError as exc:
                        render_error(exc)
                        recommendations = []

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


__all__ = ["render"]
