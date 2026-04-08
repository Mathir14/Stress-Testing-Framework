"""
Module 9: Visualization & Reports
Aggregates results from all modules and generates exportable reports.
"""

from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReportGenerator:
    """Compile, visualise, and export results from all framework modules."""

    # ------------------------------------------------------------------ #
    #  Data compilation                                                    #
    # ------------------------------------------------------------------ #

    def compile_report(
        self,
        trainer,
        data_manager,
        *,
        stress_results: dict | None = None,
        cal_ece: dict | None = None,
        reliability_scores: dict | None = None,
        dataset_name: str = "Test",
    ) -> dict:
        """
        Pull data from all modules into one structured report dict.

        Returns
        -------
        dict with keys:
          generated_at, dataset_name,
          performance  – DataFrame-able list of dicts,
          robustness   – DataFrame-able list of dicts,
          calibration  – DataFrame-able list of dicts,
          reliability  – DataFrame-able list of dicts,
          summary      – top-level KPIs
        """
        models = list(trainer.trained_models.keys())
        perf_rows, rob_rows, cal_rows, rel_rows = [], [], [], []

        for m in models:
            entry = trainer.metrics.get(m, {}).get(dataset_name)

            # ── Performance ────────────────────────────────────────────
            if entry:
                perf_rows.append(
                    {
                        "Model": m,
                        "Accuracy": round(entry["accuracy"], 4),
                        "Precision": round(entry["precision"], 4),
                        "Recall": round(entry["recall"], 4),
                        "F1": round(entry["f1"], 4),
                    }
                )

            # ── Robustness ─────────────────────────────────────────────
            if stress_results and m in stress_results:
                drops = [
                    r.get("performance_drop", 0)
                    for r in stress_results[m].values()
                    if isinstance(r, dict)
                ]
                for stress_type, r in stress_results[m].items():
                    if isinstance(r, dict):
                        rob_rows.append(
                            {
                                "Model": m,
                                "Stress Type": stress_type,
                                "Orig Acc": round(r.get("accuracy_original", 0), 4),
                                "Stress Acc": round(r.get("accuracy_stressed", 0), 4),
                                "Drop": round(r.get("performance_drop", 0), 4),
                                "Drop %": round(r.get("performance_drop_pct", 0), 2),
                            }
                        )

            # ── Calibration ────────────────────────────────────────────
            if cal_ece and m in cal_ece:
                cal_rows.append(
                    {
                        "Model": m,
                        "ECE": round(cal_ece[m], 4),
                        "Quality": (
                            "Excellent"
                            if cal_ece[m] < 0.03
                            else (
                                "Good"
                                if cal_ece[m] < 0.07
                                else "Moderate" if cal_ece[m] < 0.15 else "Poor"
                            )
                        ),
                    }
                )

            # ── Reliability ────────────────────────────────────────────
            if reliability_scores and m in reliability_scores:
                sd = reliability_scores[m]
                rel_rows.append(
                    {
                        "Model": m,
                        "Total": sd["total"],
                        "Grade": sd["grade"],
                        "Performance": sd["performance"],
                        "Calibration": sd["calibration"],
                        "Robustness": sd["robustness"],
                        "Confidence": sd["confidence"],
                    }
                )

        # ── Summary KPIs ───────────────────────────────────────────────
        best_perf = (
            max(perf_rows, key=lambda r: r["F1"])["Model"] if perf_rows else "N/A"
        )
        best_rel = (
            max(rel_rows, key=lambda r: r["Total"])["Model"] if rel_rows else "N/A"
        )
        most_robust = (
            min(
                {
                    m: np.mean([r["Drop"] for r in rob_rows if r["Model"] == m])
                    for m in models
                    if any(r["Model"] == m for r in rob_rows)
                }.items(),
                key=lambda x: x[1],
            )[0]
            if rob_rows
            else "N/A"
        )

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "num_models": len(models),
            "models": models,
            "performance": perf_rows,
            "robustness": rob_rows,
            "calibration": cal_rows,
            "reliability": rel_rows,
            "summary": {
                "best_performance": best_perf,
                "best_reliability": best_rel,
                "most_robust": most_robust,
                "num_models": len(models),
                "num_stressed": len({r["Model"] for r in rob_rows}),
                "num_calibrated": len(cal_rows),
            },
        }

    # ------------------------------------------------------------------ #
    #  Charts for the dashboard                                            #
    # ------------------------------------------------------------------ #

    def plot_performance_overview(self, report: dict) -> go.Figure:
        """Grouped bar: Accuracy / Precision / Recall / F1 per model."""
        rows = report["performance"]
        if not rows:
            return go.Figure()

        models = [r["Model"] for r in rows]
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

        fig = go.Figure()
        for metric, color in zip(metrics, colors):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=models,
                    y=[r[metric] for r in rows],
                    marker_color=color,
                    text=[f"{r[metric]:.3f}" for r in rows],
                    textposition="outside",
                )
            )
        fig.update_layout(
            barmode="group",
            title="Performance Metrics by Model",
            yaxis=dict(range=[0, 1.15]),
            template="plotly_white",
            height=380,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def plot_robustness_heatmap(self, report: dict) -> go.Figure:
        """Heatmap: models × stress types coloured by performance drop."""
        rows = report["robustness"]
        if not rows:
            return go.Figure()

        models = sorted({r["Model"] for r in rows})
        stress_types = sorted({r["Stress Type"] for r in rows})
        z = [
            [
                next(
                    (
                        r["Drop"]
                        for r in rows
                        if r["Model"] == m and r["Stress Type"] == s
                    ),
                    None,
                )
                for s in stress_types
            ]
            for m in models
        ]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=stress_types,
                y=models,
                colorscale="RdYlGn_r",
                text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in z],
                texttemplate="%{text}",
                showscale=True,
                zmin=0,
                zmax=0.5,
            )
        )
        fig.update_layout(
            title="Performance Drop by Stress Type (lower = better)",
            xaxis_title="Stress Type",
            yaxis_title="Model",
            template="plotly_white",
            height=max(300, 80 * len(models)),
        )
        return fig

    def plot_reliability_gauge_row(self, report: dict) -> go.Figure | None:
        """Subplot of indicator gauges, one per model."""
        rows = report["reliability"]
        if not rows:
            return None

        n = len(rows)
        fig = make_subplots(
            rows=1,
            cols=n,
            specs=[[{"type": "indicator"}] * n],
            subplot_titles=[r["Model"] for r in rows],
        )
        grade_color = {
            "A+": "#2ECC71",
            "A": "#27AE60",
            "B": "#F39C12",
            "C": "#E67E22",
            "D": "#E74C3C",
            "F": "#C0392B",
        }
        for i, r in enumerate(rows, start=1):
            colour = grade_color.get(r["Grade"], "#999")
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=r["Total"],
                    number={"suffix": "/100", "font": {"size": 22}},
                    title={"text": f"Grade {r['Grade']}", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": colour},
                        "steps": [
                            {"range": [0, 50], "color": "#FADBD8"},
                            {"range": [50, 70], "color": "#FAE5D3"},
                            {"range": [70, 90], "color": "#D5F5E3"},
                            {"range": [90, 100], "color": "#A9DFBF"},
                        ],
                    },
                ),
                row=1,
                col=i,
            )
        fig.update_layout(
            title="Reliability Scores",
            height=280,
            margin=dict(t=60, b=10, l=20, r=20),
        )
        return fig

    def plot_calibration_bar(self, report: dict) -> go.Figure:
        """Bar chart of ECE per model (lower is better)."""
        rows = report["calibration"]
        if not rows:
            return go.Figure()

        models = [r["Model"] for r in rows]
        eces = [r["ECE"] for r in rows]
        quals = [r["Quality"] for r in rows]
        colors = [
            (
                "#2ECC71"
                if q == "Excellent"
                else (
                    "#27AE60"
                    if q == "Good"
                    else "#F39C12" if q == "Moderate" else "#E74C3C"
                )
            )
            for q in quals
        ]

        fig = go.Figure(
            go.Bar(
                x=models,
                y=eces,
                marker_color=colors,
                text=[f"{e:.4f}\n({q})" for e, q in zip(eces, quals)],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Expected Calibration Error (lower is better)",
            yaxis=dict(title="ECE", range=[0, max(eces) * 1.4 if eces else 1]),
            template="plotly_white",
            height=360,
        )
        return fig

    def plot_radar_all(self, report: dict) -> go.Figure:
        """Radar chart of reliability components across models."""
        rows = report["reliability"]
        if not rows:
            return go.Figure()

        cats = ["Performance", "Calibration", "Robustness", "Confidence"]
        maxvals = [25, 25, 25, 25]
        palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        fig = go.Figure()
        for i, r in enumerate(rows):
            vals = [
                r["Performance"] / maxvals[0],
                r["Calibration"] / maxvals[1],
                r["Robustness"] / maxvals[2],
                r["Confidence"] / maxvals[3],
            ]
            vals_c = vals + [vals[0]]
            cats_c = cats + [cats[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals_c,
                    theta=cats_c,
                    fill="toself",
                    name=r["Model"],
                    opacity=0.65,
                    line=dict(color=palette[i % len(palette)], width=2),
                )
            )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Reliability Component Radar",
            template="plotly_white",
            height=420,
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Export helpers                                                      #
    # ------------------------------------------------------------------ #

    def export_to_json(self, report: dict) -> str:
        """Serialise report dict to a JSON string."""

        # Make numpy types JSON-safe
        def default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(report, indent=2, default=default)

    def export_performance_csv(self, report: dict) -> bytes:
        df = pd.DataFrame(report["performance"])
        return df.to_csv(index=False).encode("utf-8")

    def export_robustness_csv(self, report: dict) -> bytes:
        df = pd.DataFrame(report["robustness"])
        return df.to_csv(index=False).encode("utf-8")

    def export_reliability_csv(self, report: dict) -> bytes:
        df = pd.DataFrame(report["reliability"])
        return df.to_csv(index=False).encode("utf-8")

    def export_all_csv(self, report: dict) -> bytes:
        """All tables concatenated into one CSV with a section header column."""
        frames = []
        for section in ("performance", "robustness", "calibration", "reliability"):
            rows = report.get(section, [])
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "Section", section.capitalize())
                frames.append(df)
        if not frames:
            return b""
        combined = pd.concat(frames, ignore_index=True)
        return combined.to_csv(index=False).encode("utf-8")

    def export_to_html(self, report: dict) -> str:
        """Generate a self-contained styled HTML report."""
        generated = report["generated_at"]
        dataset = report["dataset_name"]
        summary = report["summary"]

        def _table(rows: list[dict]) -> str:
            if not rows:
                return "<p><em>No data available.</em></p>"
            keys = list(rows[0].keys())
            header = "".join(f"<th>{k}</th>" for k in keys)
            body = ""
            for row in rows:
                body += "<tr>" + "".join(f"<td>{row[k]}</td>" for k in keys) + "</tr>"
            return (
                f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Reliability Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1   {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 8px; }}
  h2   {{ color: #34495E; margin-top: 36px; border-left: 4px solid #3498DB;
           padding-left: 10px; }}
  table{{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
  th   {{ background: #3498DB; color: white; padding: 8px 12px; text-align: left; }}
  td   {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #F2F6FC; }}
  .kpi-grid  {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .kpi-card  {{ background: #EBF5FB; border-radius: 8px; padding: 16px 24px;
                min-width: 160px; text-align: center; }}
  .kpi-value {{ font-size: 24px; font-weight: bold; color: #2980B9; }}
  .kpi-label {{ font-size: 12px; color: #555; margin-top: 4px; }}
  .footer    {{ margin-top: 50px; font-size: 11px; color: #999;
                border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
<h1>🔬 ML Model Reliability Report</h1>
<p>Generated: <strong>{generated}</strong> &nbsp;|&nbsp; Dataset: <strong>{dataset}</strong></p>

<h2>Summary</h2>
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-value">{summary['num_models']}</div>
    <div class="kpi-label">Models Trained</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{summary['num_stressed']}</div>
    <div class="kpi-label">Models Stress-Tested</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{summary['best_performance']}</div>
    <div class="kpi-label">Best Performance</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{summary['best_reliability']}</div>
    <div class="kpi-label">Best Reliability</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{summary['most_robust']}</div>
    <div class="kpi-label">Most Robust</div>
  </div>
</div>

<h2>Performance Metrics</h2>
{_table(report['performance'])}

<h2>Robustness (Stress Test Results)</h2>
{_table(report['robustness'])}

<h2>Calibration</h2>
{_table(report['calibration'])}

<h2>Reliability Scores</h2>
{_table(report['reliability'])}

<div class="footer">
  Generated by ML Model Reliability &amp; Stress Testing Framework &mdash; {generated}
</div>
</body>
</html>"""
        return html

    def export_to_pdf(self, report: dict) -> bytes:
        """Generate a PDF report using reportlab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#2C3E50"),
        )
        h2 = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#34495E"),
            spaceAfter=6,
        )
        body = styles["BodyText"]

        def _pdf_table(rows: list[dict]) -> Table | Paragraph:
            if not rows:
                return Paragraph("No data available.", body)
            keys = list(rows[0].keys())
            data = [keys] + [[str(r.get(k, "")) for k in keys] for r in rows]
            col_w = (A4[0] - 4 * cm) / len(keys)
            t = Table(data, colWidths=[col_w] * len(keys))
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498DB")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#F2F6FC"), colors.white],
                        ),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            return t

        story = [
            Paragraph("ML Model Reliability Report", h1),
            Spacer(1, 0.3 * cm),
            Paragraph(
                f"Generated: {report['generated_at']}  |  Dataset: {report['dataset_name']}",
                body,
            ),
            Spacer(1, 0.5 * cm),
            Paragraph("Summary", h2),
            Paragraph(
                f"Models trained: {report['summary']['num_models']}  |  "
                f"Best performance: {report['summary']['best_performance']}  |  "
                f"Best reliability: {report['summary']['best_reliability']}  |  "
                f"Most robust: {report['summary']['most_robust']}",
                body,
            ),
            Spacer(1, 0.5 * cm),
            Paragraph("Performance Metrics", h2),
            _pdf_table(report["performance"]),
            Spacer(1, 0.5 * cm),
            Paragraph("Robustness (Stress Test Results)", h2),
            _pdf_table(report["robustness"]),
            Spacer(1, 0.5 * cm),
            Paragraph("Calibration", h2),
            _pdf_table(report["calibration"]),
            Spacer(1, 0.5 * cm),
            Paragraph("Reliability Scores", h2),
            _pdf_table(report["reliability"]),
        ]

        doc.build(story)
        return buf.getvalue()
