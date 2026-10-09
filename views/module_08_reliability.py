from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.state import AppContext, StateKeys
from modules.calibration_module import CalibrationAnalyzer
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from modules.reliability_module import ReliabilityScorer
from utils.metrics import get_prediction_entropy, identify_high_confidence_errors

logger = logging.getLogger(__name__)


def render(ctx: AppContext) -> None:
    """Render the Reliability Scoring page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    st.header("8️⃣ Reliability Scoring Module")
    st.markdown(
        "Unified reliability score per model — combining performance, calibration, "
        "robustness, and confidence quality into a single 0–100 score with letter grade."
    )

    scorer = ctx.ensure_service(StateKeys.RELIABILITY_SCORER, ReliabilityScorer)
    trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)
    cal_an = ctx.ensure_service(StateKeys.CALIBRATION_ANALYZER, CalibrationAnalyzer)
    dm = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)

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
                except Exception as exc:
                    # Conventions §3: a blind catch must log, not pass. A model
                    # that cannot be predicted is now distinguishable in the log
                    # from one that simply has no ECE recorded (MAJ-004).
                    logger.warning(
                        "Calibration ECE unavailable for model %r; its score will "
                        "use the default calibration weight. Cause: %s",
                        mn,
                        exc,
                    )

        # ── Gather entropy & HCE rate ─────────────────────────────────────
        entropy_data: dict = {}
        hce_data: dict = {}
        if dm.X_test is not None and dm.y_test is not None:
            for mn in trainer.trained_models:
                try:
                    preds, probs = trainer.predict(mn, dm.X_test)
                    entropies = get_prediction_entropy(probs)
                    entropy_data[mn] = float(np.mean(entropies))
                    hce_df = identify_high_confidence_errors(dm.y_test, preds, probs, threshold=0.9)
                    total_preds = len(preds)
                    hce_data[mn] = len(hce_df) / total_preds if total_preds > 0 else 0.0
                except Exception as exc:
                    # Conventions §3: log rather than pass. Entropy and the
                    # high-confidence-error rate are both derived from a single
                    # predict() call, so one failure loses both components.
                    logger.warning(
                        "Entropy / high-confidence-error rate unavailable for model "
                        "%r; its score will use the default confidence weights. "
                        "Cause: %s",
                        mn,
                        exc,
                    )

        # ── Stress robustness ─────────────────────────────────────────────
        stress_results = ctx.store.get(StateKeys.BATCH_STRESS_RESULTS_BY_MODEL) or {}

        # ── Number of classes ─────────────────────────────────────────────
        n_classes = 2
        if dm.y_test is not None:
            try:
                n_classes = len(np.unique(dm.y_test))
            except Exception as exc:
                # n_classes=2 above is the documented binary default; log so a
                # multiclass score is never silently graded on a binary basis.
                logger.warning(
                    "Could not determine the class count from y_test; falling back "
                    "to the binary default of 2, which will mis-grade a multiclass "
                    "model. Cause: %s",
                    exc,
                )

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
                            width="stretch",
                            key=f"rel_gauge_overview_{mname}",
                        )

                st.markdown("---")

                # Stacked bar
                fig_stack = scorer.plot_stacked_bar(scores_dict)
                st.plotly_chart(fig_stack, width="stretch", key="rel_stacked_bar")

                st.markdown("---")

                # Total bar
                fig_total = scorer.plot_total_bar(scores_dict)
                st.plotly_chart(fig_total, width="stretch", key="rel_total_bar")

                st.markdown("---")

                # Summary table
                st.subheader("📋 Score Summary Table")
                df_sum = scorer.build_summary_df(scores_dict)
                st.dataframe(df_sum, width="stretch")

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
                st.plotly_chart(fig_radar, width="stretch", key="rel_radar")

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
                        width="stretch",
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
                            f"⚠️ Defaulted (12.5 pts each): {', '.join(sd['missing_components'])}"
                        )

                st.markdown("---")
                st.subheader("Raw Inputs")
                det = sd["details"]
                det_df = pd.DataFrame(
                    [
                        {
                            "Metric": "Accuracy",
                            "Value": (
                                f"{det['accuracy']:.4f}" if det["accuracy"] is not None else "N/A"
                            ),
                        },
                        {
                            "Metric": "F1 Score",
                            "Value": (f"{det['f1']:.4f}" if det["f1"] is not None else "N/A"),
                        },
                        {
                            "Metric": "ECE",
                            "Value": (f"{det['ece']:.4f}" if det["ece"] is not None else "N/A"),
                        },
                        {
                            "Metric": "Avg Drop",
                            "Value": (
                                f"{det['avg_drop']:.4f}" if det["avg_drop"] is not None else "N/A"
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
                                f"{det['hce_rate']:.4f}" if det["hce_rate"] is not None else "N/A"
                            ),
                        },
                    ]
                )
                st.dataframe(det_df, width="stretch", hide_index=True)

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


__all__ = ["render"]
