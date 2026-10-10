"""HTML-escaping tests for the report exporters (architecture.md §6.5, §10.2).

**Tier T1** — requires numpy/pandas (the module imports the metrics/plotting
helpers).  ``reportlab`` is imported lazily inside :meth:`export_to_pdf`, so the
PDF cases skip cleanly rather than failing when it is absent.

The threat is concrete: dataset names, model names, stress-test names and metric
strings all originate from a user-supplied CSV filename or a UI label, and
``export_to_html`` writes a file the user then opens in a browser.  Unescaped
interpolation is therefore stored XSS in a local research tool — the kind of bug
that is invisible in review because ``f"<td>{value}</td>"`` reads as harmless.

Two properties are asserted separately, because they fail differently:

* the **payload is neutralised** — the raw ``<script>`` text does not survive
  into the markup, which catches a missing ``_esc``;
* the **structure is intact** — the expected number of ``<td>`` elements is
  present, which catches an over-eager fix that escapes the markup delimiters
  themselves and produces a report full of literal ``&lt;td&gt;``.
"""

from __future__ import annotations

import html as html_module
import json
from typing import Any

import pytest

pytest.importorskip("numpy", reason="the report generator pulls the scientific stack")
pytest.importorskip("pandas", reason="the report generator pulls the scientific stack")
pytest.importorskip("plotly", reason="the report generator pulls the scientific stack")

from modules.reporting_module import ReportGenerator

#: A payload that is simultaneously a tag, an attribute break and an entity.
#: ``<script>alert(1)</script>`` is the canonical case; the quote-bearing variant
#: matters because ``export_to_html`` interpolates into attribute positions too.
XSS_PAYLOAD = "<script>alert('xss')</script>"
ATTR_PAYLOAD = '" onmouseover="alert(1)'
AMPERSAND = "Tom & Jerry <Data>"


def _report(
    *,
    dataset_name: str = "Test Set",
    model: str = "Logistic Regression",
    stress_name: str = "Gaussian Noise (Medium)",
) -> dict[str, Any]:
    """Build a minimal report dict shaped like ``ReportGenerator.compile_report``'s."""
    return {
        "generated_at": "2026-10-04 12:00:00",
        "dataset_name": dataset_name,
        "summary": {
            "num_models": 2,
            "num_stressed": 2,
            "best_performance": model,
            "best_reliability": 87.5,
            "most_robust": model,
        },
        "performance": [
            {
                "Model": model,
                "Accuracy": 0.9231,
                "Precision": 0.91,
                "Recall": 0.90,
                "F1": 0.905,
            }
        ],
        "robustness": [
            {
                "Model": model,
                "Stress Type": stress_name,
                "Orig Acc": 0.92,
                "Stress Acc": 0.71,
                "Drop": 0.21,
                "Drop %": 22.83,
            }
        ],
        "calibration": [{"Model": model, "ECE": 0.0432, "Quality": "Good"}],
        "reliability": [{"Model": model, "Score": 87.5, "Grade": "A", "Perf": 23.1}],
    }


@pytest.fixture
def reporter() -> ReportGenerator:
    return ReportGenerator()


# ── The core requirement: script payloads are neutralised ─────────────────────


def test_script_payload_in_dataset_name_is_escaped(reporter: ReportGenerator):
    """§6.5: the dataset name reaches ``<strong>`` in the header line."""
    document = reporter.export_to_html(_report(dataset_name=XSS_PAYLOAD))
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


def test_script_payload_in_model_name_is_escaped(reporter: ReportGenerator):
    """Model names reach both the KPI cards and the table cells."""
    document = reporter.export_to_html(_report(model=XSS_PAYLOAD))
    assert "<script>" not in document
    assert document.count("&lt;script&gt;") >= 2


def test_script_payload_in_stress_name_is_escaped(reporter: ReportGenerator):
    """The stress-test name is a row label in the robustness table."""
    document = reporter.export_to_html(_report(stress_name=XSS_PAYLOAD))
    assert "<script>" not in document


def test_script_payload_in_every_table_is_escaped(reporter: ReportGenerator):
    """All four tables, including the header row."""
    report = _report()
    for key in ("performance", "robustness", "calibration", "reliability"):
        report[key][0]["Model"] = XSS_PAYLOAD
    document = reporter.export_to_html(report)
    assert "<script>" not in document


def test_column_headers_are_escaped(reporter: ReportGenerator):
    """Keys become ``<th>`` text; a hostile *column* name must be escaped too."""
    report = _report()
    report["performance"] = [{XSS_PAYLOAD: "value"}]
    document = reporter.export_to_html(report)
    assert "<script>" not in document
    assert "&lt;script&gt;" in document


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "<svg/onload=alert(1)>",
        "<iframe src='evil'></iframe>",
        "</td><td><script>alert(1)</script>",
        "<style>body{display:none}</style>",
        '<a href="javascript:alert(1)">x</a>',
    ],
)
def test_a_range_of_payloads_is_neutralised(reporter: ReportGenerator, payload: str):
    """One payload proves escaping; several prove it is *generic* escaping."""
    document = reporter.export_to_html(_report(dataset_name=payload, model=payload))
    assert payload not in document
    assert "<script" not in document
    assert "<img" not in document
    assert "<svg" not in document
    assert "<iframe" not in document
    assert "<style>body" not in document


def test_a_bare_javascript_uri_is_inert_in_text_context():
    """``javascript:alert(1)`` alone contains no HTML metacharacter.

    Escaping cannot and should not alter it: the report interpolates every value
    into element *text*, never into an ``href``, so a bare URI has nothing to
    break out of.  Pinned so the neutralisation tests above are not read as a
    claim that escaping rewrites content — it escapes markup, and the
    round-trip test is what proves the content itself is preserved.
    """
    document = ReportGenerator().export_to_html(_report(dataset_name="javascript:alert(1)"))
    assert "javascript:alert(1)" in document
    assert "<a " not in document


def test_attribute_breakout_payload_is_escaped(reporter: ReportGenerator):
    """``quote=True`` escapes ``"``; a value interpolated into an attribute
    position must not be able to add one."""
    document = reporter.export_to_html(_report(dataset_name=ATTR_PAYLOAD))
    assert 'onmouseover="alert(1)' not in document
    assert "&quot;" in document


def test_ampersand_payload_is_escaped_not_dropped(reporter: ReportGenerator):
    """``&`` must become ``&amp;`` — dropping it would corrupt legitimate names."""
    document = reporter.export_to_html(_report(dataset_name=AMPERSAND))
    assert AMPERSAND not in document
    assert "Tom &amp; Jerry" in document


def test_special_characters_survive_a_round_trip(reporter: ReportGenerator):
    """Escaping is reversible: unescaping the document restores the value.

    This is what proves the fix is escaping rather than stripping, which would
    silently mangle legitimate dataset names.
    """
    document = reporter.export_to_html(_report(dataset_name=AMPERSAND))
    assert html_module.unescape(AMPERSAND) in html_module.unescape(document)


# ── Structure is not broken by escaping ──────────────────────────────────────


def test_table_structure_is_intact(reporter: ReportGenerator):
    """An over-eager fix that escaped ``<td>`` would produce no real cells."""
    report = _report()
    document = reporter.export_to_html(report)
    expected_cells = sum(
        len(rows[0])
        for rows in (
            report["performance"],
            report["robustness"],
            report["calibration"],
            report["reliability"],
        )
    )
    assert document.count("<td>") == expected_cells
    assert document.count("<tr>") == 4 * 2  # header + one body row per table
    assert document.count("<th>") == sum(
        len(rows[0])
        for rows in (
            report["performance"],
            report["robustness"],
            report["calibration"],
            report["reliability"],
        )
    )
    assert "<td>" in document and "&lt;td&gt;" not in document


def test_the_document_is_well_formed_enough_to_open(reporter: ReportGenerator):
    """The five structural tags a browser needs must be intact and balanced."""
    document = reporter.export_to_html(_report(dataset_name=XSS_PAYLOAD))
    assert document.startswith("<!DOCTYPE html>")
    assert document.rstrip().endswith("</html>")
    for tag in ("html", "head", "body", "table", "style"):
        assert document.count(f"<{tag}") >= 1, tag
        assert document.count(f"</{tag}>") == document.count(f"<{tag}") - (
            document.count(f"<{tag}/>")
        ), tag


def test_static_template_content_is_not_escaped(reporter: ReportGenerator):
    """Only interpolated values are escaped; the template's own text stays live.

    The stylesheet and the emoji heading must remain real markup, or the report
    renders unstyled.
    """
    document = reporter.export_to_html(_report())
    assert ".kpi-card" in document, "the <style> block was escaped"
    assert "🔬 ML Model Reliability Report" in document
    assert "<h1>" in document


def test_benign_names_render_unescaped(reporter: ReportGenerator):
    """The ordinary path must be untouched — no stray entities in normal use."""
    document = reporter.export_to_html(_report())
    assert "Logistic Regression" in document
    assert "Gaussian Noise (Medium)" in document
    assert "Test Set" in document


def test_empty_table_renders_the_placeholder(reporter: ReportGenerator):
    """An empty section must not produce an empty ``<table>``."""
    report = _report()
    report["calibration"] = []
    document = reporter.export_to_html(report)
    assert "No data available." in document


def test_non_string_values_are_escaped_too(reporter: ReportGenerator):
    """Numbers and ``None`` reach the same ``_esc``; ``str()`` must not raise."""
    report = _report()
    report["performance"][0]["F1"] = None
    report["performance"][0]["Accuracy"] = 0.5
    document = reporter.export_to_html(report)
    assert "<td>None</td>" in document
    assert "<td>0.5</td>" in document


def test_values_are_escaped_inside_numeric_cells_when_they_are_strings(reporter: ReportGenerator):
    """A metric column holding text is still escaped — types are not trusted."""
    report = _report()
    report["performance"][0]["Accuracy"] = XSS_PAYLOAD
    document = reporter.export_to_html(report)
    assert "<script>" not in document


# ── The PDF path ─────────────────────────────────────────────────────────────


def _require_reportlab() -> None:
    pytest.importorskip("reportlab", reason="PDF export requires reportlab")


def test_pdf_escapes_script_payloads(reporter: ReportGenerator):
    """§6.5 covers the PDF path too: ``Paragraph`` consumes mini-HTML."""
    _require_reportlab()
    pdf = reporter.export_to_pdf(_report(dataset_name=XSS_PAYLOAD, model=XSS_PAYLOAD))
    assert pdf.startswith(b"%PDF")
    assert b"<script>" not in pdf, "the raw script tag survived into the PDF"


def test_pdf_escapes_attribute_payloads(reporter: ReportGenerator):
    _require_reportlab()
    pdf = reporter.export_to_pdf(_report(dataset_name=ATTR_PAYLOAD))
    assert pdf.startswith(b"%PDF")


def test_pdf_renders_a_benign_report(reporter: ReportGenerator):
    """The ordinary path still produces a valid PDF after the escaping change."""
    _require_reportlab()
    assert reporter.export_to_pdf(_report()).startswith(b"%PDF")


# ── Non-HTML exports are not affected ────────────────────────────────────────


def test_json_export_is_unchanged(reporter: ReportGenerator):
    """JSON has its own escaping; it must not be double-escaped by the HTML fix."""
    payload = reporter.export_to_json(_report(dataset_name=AMPERSAND))
    decoded = json.loads(payload)
    assert decoded["dataset_name"] == AMPERSAND, "JSON export double-escaped its values"


def test_csv_export_is_unchanged(reporter: ReportGenerator):
    """CSV is not HTML; escaping it would corrupt the data."""
    csv_bytes = reporter.export_performance_csv(_report(model=AMPERSAND))
    assert b"Tom & Jerry" in csv_bytes
    assert b"&amp;" not in csv_bytes
