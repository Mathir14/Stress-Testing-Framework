from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.state import AppContext, StateKeys
from modules.calibration_module import CalibrationAnalyzer
from modules.comparison_module import ModelComparator
from modules.data_module import DataManager
from modules.model_module import ModelTrainer

logger = logging.getLogger(__name__)


def render(ctx: AppContext) -> None:
    """Render the Model Comparison page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    st.header("7️⃣ Model Comparison Module")
    st.markdown(
        "Side-by-side evaluation of all trained models across performance, "
        "robustness, and calibration dimensions."
    )

    comparator = ctx.ensure_service(StateKeys.MODEL_COMPARATOR, ModelComparator)
    trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)
    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    calibration_analyzer = ctx.ensure_service(StateKeys.CALIBRATION_ANALYZER, CalibrationAnalyzer)

    if not trainer.trained_models:
        st.warning("⚠️ No trained models found. Please train at least one model in Module 2.")
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
            stress_results = ctx.store.get(StateKeys.BATCH_STRESS_RESULTS_BY_MODEL) or {}
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
                        robustness_scores[model_name] = round(max(0.0, 100.0 - avg_drop * 100), 2)

            # ── Calibration ECE ───────────────────────────────────────────
            cal_ece: dict[str, float] = {}
            cal_analyzer = calibration_analyzer
            dm = data_manager
            if dm.X_test is not None and dm.y_test is not None:
                for model_name in models_with_data:
                    try:
                        _, probs = trainer.predict(model_name, dm.X_test)
                        cal_metrics = cal_analyzer.compute_calibration_metrics(dm.y_test, probs)
                        cal_ece[model_name] = cal_metrics["ece"]
                    except Exception as exc:
                        # Conventions §3: a blind catch must log. Silently skipping
                        # the model made a broken model indistinguishable from one
                        # with no calibration data (MAJ-004).
                        logger.warning(
                            "Calibration ECE unavailable for model %r; the model is "
                            "excluded from the composite score. Cause: %s",
                            model_name,
                            exc,
                        )

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
                    st.dataframe(df_cmp, width="stretch")

                st.markdown("---")

                # Grouped bar chart
                fig_bar = comparator.plot_metrics_bar(metrics_dict)
                st.plotly_chart(fig_bar, width="stretch")

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
                st.plotly_chart(fig_radar, width="stretch")

                # Small table reminder
                df_cmp2 = comparator.build_comparison_df(metrics_dict)
                with st.expander("📋 Underlying values"):
                    st.dataframe(df_cmp2, width="stretch")

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
                            fig_cm = comparator.plot_confusion_matrix(cm_data["cm"], model_name)
                            st.plotly_chart(fig_cm, width="stretch")

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
                        st.dataframe(pivot, width="stretch")

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
                    st.plotly_chart(fig_rob, width="stretch")

                    # Score table
                    score_df = pd.DataFrame(
                        [
                            {
                                "Model": m,
                                "Robustness Score": f"{s:.1f} / 100",
                                "Rating": (
                                    "✅ High" if s >= 70 else "⚠️ Medium" if s >= 40 else "❌ Low"
                                ),
                            }
                            for m, s in sorted(
                                robustness_scores.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        ]
                    )
                    st.dataframe(score_df, width="stretch")

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
                    st.dataframe(ece_df, width="stretch")

            # ─── TAB 5: Best model ────────────────────────────────────────
            with tab5:
                st.subheader("🏆 Best Model Recommendation")

                if not composite:
                    st.warning("No composite scores available.")
                else:
                    # Composite bar chart
                    fig_comp = comparator.plot_composite_scores(composite)
                    st.plotly_chart(fig_comp, width="stretch")

                    # Recommendation
                    recommendation = comparator.recommend_best_model(
                        metrics_dict, composite, robustness_scores or None
                    )
                    st.success(recommendation["reason"])

                    # Ranking table
                    st.markdown("### 📋 Full Model Ranking")
                    ranking_rows = []
                    for rank, model_name in enumerate(recommendation["ranking"], start=1):
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
                    st.dataframe(ranking_df, width="stretch")

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


__all__ = ["render"]
