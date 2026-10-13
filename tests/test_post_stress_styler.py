"""Post-stress vulnerability table: no ``FutureWarning`` on the pinned pandas.

**Tier T1** — requires numpy/pandas; ``jinja2`` is required to *render* a
``Styler`` (``pandas`` pulls it in transitively via ``streamlit``, and it is
pinned in ``requirements.txt``).  Rendering skips cleanly when it is absent
rather than erroring, so the suite stays runnable in a trimmed environment.

**ADR-011.9.**  ``DataFrame.style.map`` was introduced in pandas 2.1 as the
replacement for ``Styler.applymap``; the old spelling still exists but emits a
``FutureWarning`` on 2.3.3.  This module pins both halves of the requirement:

* the view uses the current spelling — read out of ``views/module_05_post_stress.py``
  by :mod:`ast` rather than transcribed, so a copy-paste of the deprecated API
  fails here;
* the rendered HTML is produced with **no** ``FutureWarning`` and actually
  contains the per-cell colours the severity function returns.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the post-stress Styler tests need the scientific stack")
pytest.importorskip("pandas", reason="the post-stress Styler tests need the scientific stack")
pytest.importorskip("plotly", reason="the post-stress page renders Plotly figures")

import pandas as pd  # noqa: E402 - after importorskip by design

from modules.post_stress_module import PostStressAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POST_STRESS_VIEW = PROJECT_ROOT / "views" / "module_05_post_stress.py"

#: The four severity bands and their thresholds, mirrored from
#: ``PostStressAnalyzer._classify_severity``.
SEVERITY_CSS = {
    "Critical": "background-color: #ff4444; color: white",
    "High": "background-color: #ff9944; color: white",
    "Medium": "background-color: #ffcc44",
    "Low": "background-color: #44ff44",
}


def color_severity(val: str) -> str:
    """The view's colouring function, mirrored for the render assertions."""
    return SEVERITY_CSS.get(val, SEVERITY_CSS["Low"])


def _stress_results() -> dict:
    """Four stress results spanning all four severity bands."""
    return {
        "Gaussian Noise (Low)": {
            "accuracy_original": 0.92,
            "accuracy_stressed": 0.905,
            "performance_drop": 0.015,
            "performance_drop_pct": 1.63,
            "prediction_agreement": 0.98,
        },
        "Gaussian Noise (High)": {
            "accuracy_original": 0.92,
            "accuracy_stressed": 0.83,
            "performance_drop": 0.09,
            "performance_drop_pct": 9.78,
            "prediction_agreement": 0.71,
        },
        "Feature Dropout (30%)": {
            "accuracy_original": 0.92,
            "accuracy_stressed": 0.72,
            "performance_drop": 0.20,
            "performance_drop_pct": 21.74,
            "prediction_agreement": 0.55,
        },
        "Scale Perturbation (2.5x)": {
            "accuracy_original": 0.92,
            "accuracy_stressed": 0.50,
            "performance_drop": 0.42,
            "performance_drop_pct": 45.65,
            "prediction_agreement": 0.31,
        },
    }


@pytest.fixture
def analyzer() -> PostStressAnalyzer:
    instance = PostStressAnalyzer()
    instance.add_batch_results(_stress_results(), "Logistic Regression")
    return instance


@pytest.fixture
def vulnerability(analyzer: PostStressAnalyzer) -> pd.DataFrame:
    return analyzer.get_vulnerability_analysis("Logistic Regression")


# ── The severity classification ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "drop_pct,severity",
    [
        (0.0, "Low"),
        (4.99, "Low"),
        (5.0, "Medium"),
        (14.99, "Medium"),
        (15.0, "High"),
        (29.99, "High"),
        (30.0, "Critical"),
        (100.0, "Critical"),
    ],
)
def test_severity_thresholds_are_exclusive_at_the_boundary(severity: str, drop_pct: float):
    """ADR-007: the cutoffs are ``<``, so the boundary value takes the higher band."""
    assert PostStressAnalyzer()._classify_severity(drop_pct) == severity


def test_severity_is_monotone_in_drop(vulnerability: pd.DataFrame):
    """The table is sorted by descending drop, so severity must not go back up."""
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    severities = [order[s] for s in vulnerability["Severity"]]
    assert severities == sorted(severities, reverse=True)


def test_the_fixture_covers_every_severity_band(vulnerability: pd.DataFrame):
    """All four bands must be present, or the render test proves less than it looks."""
    assert set(vulnerability["Severity"]) == set(SEVERITY_CSS)


# ── The table ────────────────────────────────────────────────────────────────


def test_table_is_sorted_by_drop_descending(vulnerability: pd.DataFrame):
    drops = vulnerability["Performance Drop (%)"].tolist()
    assert drops == sorted(drops, reverse=True)


def test_prediction_changes_are_a_percentage(
    analyzer: PostStressAnalyzer, vulnerability: pd.DataFrame
):
    """``(1 - agreement) * 100``, not the agreement itself."""
    original = _stress_results()["Feature Dropout (30%)"]
    row = vulnerability[vulnerability["Stress Test"] == "Feature Dropout (30%)"]
    assert float(row["Prediction Changes (%)"].iloc[0]) == pytest.approx(
        (1 - original["prediction_agreement"]) * 100
    )


def test_unknown_model_returns_an_empty_frame(analyzer: PostStressAnalyzer):
    """The pre-remediation contract: no silent ``None``, just an empty frame."""
    result = analyzer.get_vulnerability_analysis("never-tested")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_table_has_the_documented_columns(vulnerability: pd.DataFrame):
    """The download button and the styler both key off these names."""
    assert list(vulnerability.columns) == [
        "Stress Test",
        "Performance Drop (%)",
        "Accuracy Loss",
        "Prediction Changes (%)",
        "Stressed Accuracy",
        "F1 (Stressed)",
        "Severity",
    ]


# ── §10.2 the styler renders with no FutureWarning ──────────────────────────


def _styler_map_calls(view_path: Path) -> list[tuple[str, str]]:
    """Every ``<frame>.style.map(...)`` / ``.applymap(...)`` call in a view.

    The receiver is matched structurally (an attribute access named ``style``)
    rather than by a variable name, so a rename cannot silently disarm the
    check.  ``.map`` on a dict or ``Series`` is unrelated and is not returned.
    """
    tree = ast.parse(view_path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in {"map", "applymap"}:
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Attribute) and receiver.attr == "style":
            found.append((node.func.attr, ast.dump(node)))
    return found


def test_the_view_uses_the_current_styler_spelling():
    """ADR-011.9: ``Styler.applymap`` is deprecated in favour of ``Styler.map``.

    Read from the view's AST, not transcribed: a revert to ``applymap`` in the
    view is the actual regression, and a test asserting the *production* spelling
    from a hard-coded string would not notice.
    """
    styler_calls = _styler_map_calls(POST_STRESS_VIEW)
    assert styler_calls, "no Styler.map call found in the post-stress view"
    assert [name for name, _ in styler_calls] == ["map"], (
        "the view must style via Styler.map; applymap is deprecated on pandas "
        f"2.3.3 (ADR-011.9). Found: {[name for name, _ in styler_calls]}"
    )
    # The subset has to stay pinned to the severity column; see the test below.
    (_, dump) = styler_calls[0]
    assert "subset" in dump, "the Styler.map call lost its subset= keyword"


def test_the_styler_is_scoped_to_the_severity_column():
    """``subset=["Severity"]`` must be present in the view's own call.

    ``color_severity`` falls back to the low-severity CSS for every value it does
    not recognise, so a numeric column left inside ``subset`` would render as a
    uniformly green block rather than failing.
    """
    styler_calls = _styler_map_calls(POST_STRESS_VIEW)
    assert len(styler_calls) == 1
    source = POST_STRESS_VIEW.read_text(encoding="utf-8")
    assert 'subset=["Severity"]' in source, (
        "the post-stress view must restrict the colour map to the Severity column"
    )


def _render(styler) -> str:
    """Render a Styler to HTML with whitespace collapsed.

    pandas emits one CSS declaration per line, so the production function's
    single-line ``"background-color: #ff4444; color: white"`` string never
    appears verbatim in the document.  Identical declarations are also merged
    into one grouped selector, so the *cell ids* are the reliable signal of what
    was coloured.
    """
    pytest.importorskip("jinja2", reason="Styler.to_html requires jinja2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        html = styler.to_html()
    return re.sub(r"\s+", " ", html)


def _coloured_columns(html: str) -> set[int]:
    """The frame column indexes that received a background colour."""
    block = re.search(r"<style[^>]*>(.*?)</style>", html, re.S)
    assert block, "no style block was emitted; nothing was coloured"
    return {int(col) for col in re.findall(r"#T_[0-9a-f]+_row\d+_col(\d+)", block.group(1))}


def test_styler_render_emits_no_future_warning(vulnerability: pd.DataFrame):
    """The regression itself: rendering must be warning-free on pandas 2.3.3."""
    pytest.importorskip("jinja2", reason="Styler.to_html requires jinja2")
    styler = vulnerability.style.map(color_severity, subset=["Severity"])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        styler.to_html()
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert not future, (
        f"FutureWarning while rendering the vulnerability table: {[str(w.message) for w in future]}"
    )


def test_deprecated_applymap_really_does_warn_on_this_pandas(vulnerability: pd.DataFrame):
    """Demonstrate the defect, so the test above is not vacuous.

    If this stops warning, ADR-011.9's premise has changed and both tests need
    re-measuring.  Kept as an executable record of the pinned-pandas behaviour.
    """
    pytest.importorskip("jinja2", reason="Styler.to_html requires jinja2")
    styler = vulnerability.style
    if not hasattr(styler, "applymap"):
        pytest.skip("Styler.applymap has been removed; ADR-011.9 needs updating")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        styler.applymap(color_severity, subset=["Severity"]).to_html()
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert future, (
        "Styler.applymap no longer warns on this pandas; the ADR-011.9 premise "
        "has changed and the replacement recommendation should be revisited"
    )


def test_styler_render_contains_every_severity_colour(vulnerability: pd.DataFrame):
    """Not just warning-free — the colours must actually be applied.

    A ``map`` call that silently matched nothing would satisfy the warning test
    while producing an unstyled table.
    """
    html = _render(vulnerability.style.map(color_severity, subset=["Severity"]))
    for severity, css in SEVERITY_CSS.items():
        assert css in html, f"no cell was coloured for severity {severity!r}"


def test_styler_colours_only_the_severity_column(vulnerability: pd.DataFrame):
    """``subset=["Severity"]`` must keep the other columns unstyled."""
    html = _render(vulnerability.style.map(color_severity, subset=["Severity"]))
    severity_col = list(vulnerability.columns).index("Severity")
    assert _coloured_columns(html) == {severity_col}
    # One coloured cell per row; the table has one row per stress test.
    assert html.count(SEVERITY_CSS["Critical"]) == 1
    assert html.count("background-color:") == len(vulnerability)


def test_a_wrong_subset_has_the_green_block_signature(vulnerability: pd.DataFrame):
    """Demonstrate the failure mode the ``subset=`` assertions above guard against.

    ``color_severity`` falls back to the low-severity CSS for every value it does
    not recognise, so leaving a numeric column inside ``subset`` paints it
    uniformly green instead of raising.  This is the signature those assertions
    look for; if it stops holding, they have become vacuous.
    """
    assert len(SEVERITY_CSS) == 4, "the severity vocabulary changed"
    wrong = _render(vulnerability.style.map(color_severity, subset=["Performance Drop (%)"]))
    drop_col = list(vulnerability.columns).index("Performance Drop (%)")
    assert _coloured_columns(wrong) == {drop_col}
    # Every cell of that column gets the low-severity fallback, and the Critical
    # row — the one the correct render colours red — is left uncoloured.
    assert wrong.count(SEVERITY_CSS["Critical"]) == 0


def test_rendering_is_stable_across_calls(vulnerability: pd.DataFrame):
    """No hidden state in the Styler; the second render matches the first."""
    pytest.importorskip("jinja2", reason="Styler.to_html requires jinja2")
    styler = vulnerability.style.map(color_severity, subset=["Severity"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert styler.to_html() == styler.to_html()


def test_styler_does_not_mutate_the_frame(vulnerability: pd.DataFrame):
    """Styling must be a view over the data, not an edit to it."""
    before = vulnerability.copy(deep=True)
    pytest.importorskip("jinja2", reason="Styler.to_html requires jinja2")
    styler = vulnerability.style.map(color_severity, subset=["Severity"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        styler.to_html()
    pd.testing.assert_frame_equal(vulnerability, before)


def test_csv_export_of_the_table_is_unescaped(vulnerability: pd.DataFrame):
    """The download button serialises the *unstyled* frame."""
    csv = vulnerability.to_csv(index=False)
    assert "#ff4444" not in csv, "the CSV must not contain style markup"
    assert "Severity" in csv
    assert csv.count("\n") == len(vulnerability) + 1


def test_pandas_version_is_the_one_adr_011_9_was_verified_on():
    """Guard on the premise, as in ``test_data_preprocessing.py``."""
    assert pd.__version__ == "2.3.3", f"ADR-011.9 verified on pandas 2.3.3, found {pd.__version__}"


# ── ADR-034 / ADR-035 schema validation and policy tests ──────────────────────


def test_add_stress_result_rejects_malformed_inputs():
    """ADR-035 (m-1): add_stress_result validates required keys at ingestion."""
    from core.errors import ValidationError

    analyzer = PostStressAnalyzer()

    with pytest.raises(ValidationError, match="must be a mapping"):
        analyzer.add_stress_result("bad", [1, 2, 3])  # type: ignore[arg-type]

    partial = {
        "accuracy_original": 0.9,
        "accuracy_stressed": 0.8,
        # missing performance_drop, performance_drop_pct, prediction_agreement
    }
    with pytest.raises(ValidationError, match="missing required keys") as exc_info:
        analyzer.add_stress_result("partial", partial)
    assert "performance_drop" in str(exc_info.value)
    assert "prediction_agreement" in str(exc_info.value)


def test_post_stress_analyzer_honours_injected_scoring_policy():
    """ADR-034 (m-3): recommendations and summary bands follow injected policy."""
    from core.config import ScoringPolicy

    custom_policy = ScoringPolicy(
        recommendation_bands=((90.0, "Top"), (50.0, "Mid")),
        recommendation_fallthrough="Bottom",
        robustness_bands=((15.0, "Strong"),),
        robustness_fallthrough="Fragile",
        low_prediction_stability_threshold=0.99,
        key_vulnerability_drop_threshold=5.0,
    )
    analyzer = PostStressAnalyzer(policy=custom_policy)
    analyzer.add_batch_results(_stress_results(), "CustomModel")

    recs = analyzer.get_recommendations("CustomModel")
    # Score for _stress_results is ~82.2 -> should be "Mid", not default "Excellent"
    assert any("Mid robustness" in r for r in recs)
    # 0.99 stability threshold flags the ~0.84 average agreement
    assert any("Low prediction stability" in r for r in recs)

    summary = analyzer.get_stress_type_summary("CustomModel")
    # Average drop in Gaussian noise is ~5.7% < 15.0% -> "Strong" instead of default "High"
    noise_row = summary[summary["Category"] == "Noise"]
    assert not noise_row.empty
    assert noise_row.iloc[0]["Robustness"] == "Strong"

