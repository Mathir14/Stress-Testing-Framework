from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from core.errors import FrameworkError
from core.state import AppContext, StateKeys
from modules.calibration_module import CalibrationAnalyzer
from modules.data_module import DataManager
from modules.evidence_module import (
    calibration_ece,
    class_count,
    collect_evidence,
    hce_rates,
    mean_entropies,
)
from modules.model_module import ModelTrainer
from modules.reliability_module import ReliabilityScorer
from modules.reporting_module import ReportGenerator
from views.reporters import render_error

logger = logging.getLogger(__name__)


def render(ctx: AppContext) -> None:
    """Render the Visualization & Reports page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    st.header("9️⃣ Visualization & Reports Module")
    st.markdown(
        "Aggregated dashboard across all modules with one-click export to CSV, JSON, HTML, and PDF."
    )

    rgen = ctx.ensure_service(StateKeys.REPORT_GENERATOR, ReportGenerator)
    trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)
    dm = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    cal_an = ctx.ensure_service(StateKeys.CALIBRATION_ANALYZER, CalibrationAnalyzer)
    scorer = ctx.ensure_service(StateKeys.RELIABILITY_SCORER, ReliabilityScorer)

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

        # ── Per-model evidence ────────────────────────────────────────────
        # One predict per model; the single evidence owner in
        # modules.evidence_module gathers ECE, entropy and HCE so the report
        # cannot disagree with Modules 7 and 8 (run-015 M2/M3, R6).
        stress_res = ctx.store.get(StateKeys.BATCH_STRESS_RESULTS_BY_MODEL) or {}
        with st.spinner("Gathering per-model evidence…"):
            evidence = collect_evidence(
                trainer,
                dm.X_test,
                dm.y_test,
                stress_results=stress_res or None,
                dataset_name=rep_dataset,
                calibration_analyzer=cal_an,
            )
        rep_cal_ece = calibration_ece(evidence)

        rel_scores = scorer.score_all_models(
            trainer,
            dataset_name=rep_dataset,
            stress_results=stress_res or None,
            calibration_ece=rep_cal_ece or None,
            entropy_data=mean_entropies(evidence) or None,
            hce_data=hce_rates(evidence) or None,
            n_classes=class_count(evidence, strict=True),
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
            st.plotly_chart(fig_perf, width="stretch", key="rep_perf")

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📐 Calibration (ECE)")
                fig_cal = rgen.plot_calibration_bar(report)
                st.plotly_chart(fig_cal, width="stretch", key="rep_cal")
            with c_right:
                st.subheader("⭐ Reliability Radar")
                fig_rad = rgen.plot_radar_all(report)
                st.plotly_chart(fig_rad, width="stretch", key="rep_radar")

            st.subheader("🛡️ Robustness Heatmap")
            fig_rob = rgen.plot_robustness_heatmap(report)
            st.plotly_chart(fig_rob, width="stretch", key="rep_heatmap")

            # Reliability gauges
            fig_gauges = rgen.plot_reliability_gauge_row(report)
            if fig_gauges:
                st.subheader("🏅 Reliability Gauges")
                st.plotly_chart(fig_gauges, width="stretch", key="rep_gauges")

        # ─── TAB 2: Summary Tables ────────────────────────────────────
        with tab2:
            st.subheader("🎯 Performance Metrics")
            if report["performance"]:
                st.dataframe(
                    pd.DataFrame(report["performance"]).set_index("Model"),
                    width="stretch",
                )
            else:
                st.info("No performance data available.")

            st.markdown("---")
            st.subheader("🛡️ Robustness Results")
            if report["robustness"]:
                st.dataframe(
                    pd.DataFrame(report["robustness"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No stress test data. Run batch tests in Module 4.")

            st.markdown("---")
            st.subheader("📐 Calibration")
            if report["calibration"]:
                st.dataframe(
                    pd.DataFrame(report["calibration"]).set_index("Model"),
                    width="stretch",
                )
            else:
                st.info("No calibration data. Ensure test data is prepared.")

            st.markdown("---")
            st.subheader("⭐ Reliability Scores")
            if report["reliability"]:
                st.dataframe(
                    pd.DataFrame(report["reliability"]).set_index("Model"),
                    width="stretch",
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
                    width="stretch",
                    key="gal_perf",
                )
            with st.expander("🛡️ Robustness Heatmap", expanded=True):
                if report["robustness"]:
                    st.plotly_chart(
                        rgen.plot_robustness_heatmap(report),
                        width="stretch",
                        key="gal_rob",
                    )
                else:
                    st.info("No stress data available.")
            with st.expander("📐 Calibration ECE", expanded=True):
                if report["calibration"]:
                    st.plotly_chart(
                        rgen.plot_calibration_bar(report),
                        width="stretch",
                        key="gal_cal",
                    )
                else:
                    st.info("No calibration data available.")
            with st.expander("⭐ Reliability Scores", expanded=True):
                if report["reliability"]:
                    st.plotly_chart(
                        rgen.plot_radar_all(report),
                        width="stretch",
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
                    width="stretch",
                    key="dl_csv",
                )

            with col_json:
                st.markdown("### 🧾 JSON")
                st.markdown("Full structured data.")
                json_bytes = rgen.export_to_json(report).encode("utf-8")
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_bytes,
                    file_name=f"ml_report_{report['generated_at'][:10]}.json",
                    mime="application/json",
                    width="stretch",
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
                    width="stretch",
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
                        width="stretch",
                        key="dl_pdf",
                    )
                except FrameworkError as exc:
                    render_error(exc)
                except Exception:
                    logger.exception("PDF generation failed: unexpected error")
                    raise

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
                        width="stretch",
                        key="dl_perf_csv",
                    )
            with d2:
                if report["robustness"]:
                    st.download_button(
                        "Robustness CSV",
                        rgen.export_robustness_csv(report),
                        file_name="robustness.csv",
                        mime="text/csv",
                        width="stretch",
                        key="dl_rob_csv",
                    )
            with d3:
                if report["reliability"]:
                    st.download_button(
                        "Reliability CSV",
                        rgen.export_reliability_csv(report),
                        file_name="reliability.csv",
                        mime="text/csv",
                        width="stretch",
                        key="dl_rel_csv",
                    )
            with d4:
                st.download_button(
                    "Full JSON",
                    json_bytes,
                    file_name="full_report.json",
                    mime="application/json",
                    width="stretch",
                    key="dl_full_json",
                )


__all__ = ["render"]
