"""Module 7 comparison tests: the fixed-denominator composite (ADR-018).

**Tier T1** — requires numpy/pandas for the comparator's DataFrame helpers.

The binding claims under test:

* The composite is ``performance*0.5 + robustness*0.25 + calibration*0.25``
  over a **fixed denominator**: a component with no evidence contributes
  ``missing_component_points`` (``0.0``) and its weight is **not** redistributed
  to the measured components (critic C3).
* Consequently the composite is *monotone in the evidence*: measuring an extra
  component — even one whose measured value is terrible — can never lower a
  model's score.
* ``rated`` gates ranking: a model with fewer than
  ``ScoringPolicy.min_measured_components`` measured components is surfaced
  under ``"unrated"`` instead of being ranked against measured models
  (critic M5), and ``recommend_best_model`` refuses to crown an unrated model.
* The weights, gate and bands all live in :class:`core.config.ScoringPolicy`
  (single owner); the comparator consumes it rather than restating it.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy", reason="the comparator needs the scientific stack")
pytest.importorskip("pandas", reason="the comparator needs the scientific stack")
pytest.importorskip("plotly", reason="the comparator's figures need the scientific stack")

from core.config import ReliabilityWeights, ScoringPolicy, get_config
from core.errors import ValidationError
from modules.comparison_module import ModelComparator


def _metrics(
    accuracy: float = 0.9,
    f1: float = 0.85,
    *,
    has_data: bool = True,
) -> dict[str, dict]:
    """A single-model ``metrics_dict`` shaped like ``compile_performance_metrics``."""
    return {
        "M": {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": f1,
            "f1": f1,
            "has_data": has_data,
        }
    }


def _metrics_for(names: list[str]) -> dict[str, dict]:
    """One measured entry per model name."""
    return {name: _metrics()["M"] for name in names}


@pytest.fixture(scope="module")
def comparator() -> ModelComparator:
    return ModelComparator()


# ── The policy is the single owner of the weights ────────────────────────────


def test_weights_live_in_the_binding_scoring_policy(comparator: ModelComparator):
    """The comparator reads weights from ``ScoringPolicy``, not its own copy."""
    assert comparator.policy is get_config().scoring
    assert comparator.policy.weight_for("performance") == 0.5
    assert comparator.policy.weight_for("robustness") == 0.25
    assert comparator.policy.weight_for("calibration") == 0.25
    assert sum(w for _, w in comparator.policy.composite_weights) == pytest.approx(1.0)


def test_an_unknown_component_weight_is_zero():
    """``weight_for`` returns 0.0 for a component the policy does not know."""
    comparator = ModelComparator()
    assert comparator.policy.weight_for("confidence") == 0.0


def test_a_custom_policy_is_bound_and_used():
    perf_only = ScoringPolicy(
        composite_weights=(
            ("performance", 1.0),
            ("robustness", 0.0),
            ("calibration", 0.0),
        )
    )
    custom = ModelComparator(policy=perf_only)
    composite = custom.compute_composite_score(_metrics(accuracy=0.8, f1=0.6))
    assert composite["M"] == pytest.approx(70.0)  # (0.8 + 0.6) / 2 * 100


# ── Fixed denominator: zero for missing, no redistribution ───────────────────


def test_perf_only_composite_uses_the_binding_weight():
    """``(0.9 + 0.85) / 2 * 100 = 87.5``; half of it is 43.75."""
    composite = ModelComparator().compute_composite_score(_metrics())
    assert composite["M"] == pytest.approx(43.75)


def test_adding_components_applies_each_weight():
    metrics = _metrics()
    composite = ModelComparator().compute_composite_score(
        metrics, {"M": 60.0}, {"M": 0.2}
    )
    # Calibration scores against ece_cap=0.5 (not a hardcoded 1.0): an ECE of
    # 0.2 is 1 - 0.2/0.5 = 0.6 -> a 60.0 component (run-015 M3).
    # 87.5*0.5 + 60.0*0.25 + 60.0*0.25 = 43.75 + 15.0 + 15.0
    assert composite["M"] == pytest.approx(73.75)


def test_the_calibration_cap_comes_from_the_configured_policy():
    """The calibration component uses ``ReliabilityWeights.ece_cap``.

    ``ece_cap`` defaults to 0.5, so an ECE of 0.5 is a zero component and an ECE
    of 0.25 is half credit — not the 75.0 a hardcoded ``1.0`` cap would give.
    """
    metrics = _metrics()
    comparator = ModelComparator()
    assert comparator.weights.ece_cap == get_config().reliability.ece_cap
    at_cap = comparator.compute_composite_score(metrics, None, {"M": 0.5})
    half = comparator.compute_composite_score(metrics, None, {"M": 0.25})
    assert at_cap["M"] == pytest.approx(43.75)
    assert half["M"] == pytest.approx(43.75 + 12.5)


def test_a_missing_component_is_zero_not_redistributed():
    """A measured component never inherits the missing one's weight.

    The pre-remediation composite redistributed a missing component's quota to
    the measured components, so measuring a *bad* calibration could make a
    model's composite *drop* while a missing/reliably-unknown one kept it high.
    """
    metrics = _metrics()
    base = ModelComparator().compute_composite_score(metrics, {"M": 60.0})
    with_terrible_calibration = ModelComparator().compute_composite_score(
        metrics, {"M": 60.0}, {"M": 1.5}
    )
    assert base["M"] == pytest.approx(58.75)
    assert with_terrible_calibration["M"] == pytest.approx(58.75)


def test_the_calibration_component_clips_at_zero():
    metrics = _metrics()
    comparator = ModelComparator()
    # ece_cap=0.5: ECE 0.25 -> 0.5 fraction -> a 50.0 component; ECE >= 0.5 -> 0.
    good = comparator.compute_composite_score(metrics, None, {"M": 0.25})  # 50.0 x 0.25
    awful = comparator.compute_composite_score(metrics, None, {"M": 2.0})  # clipped to 0
    assert good["M"] == pytest.approx(43.75 + 12.5)
    assert awful["M"] == pytest.approx(43.75)


def test_robustness_scores_are_used_verbatim():
    metrics = _metrics()
    composite = ModelComparator().compute_composite_score(metrics, {"M": 100.0})
    assert composite["M"] == pytest.approx(43.75 + 25.0)


def test_a_dictionary_without_the_model_is_a_missing_component():
    """A ``calibration_ece`` dict not holding the model is missing, not data."""
    composite = ModelComparator().compute_composite_score(_metrics(), None, {"Other": 0.1})
    assert composite["M"] == pytest.approx(43.75)


# ── Monotonicity over the component lattice ──────────────────────────────────


def test_composite_is_monotone_over_the_component_lattice(comparator: ModelComparator):
    """Every superset of measured components scores >= its subsets.

    Swept over all four states (nothing beyond performance, +robustness,
    +calibration, +both) and every subset relation between them.
    """
    metrics = _metrics()
    robustness = {"M": 61.2}
    calibration = {"M": 0.18}  # ece_cap=0.5 -> a 64.0 component

    def score(state: frozenset[str]) -> float:
        return comparator.compute_composite_score(
            metrics,
            robustness if "robustness" in state else None,
            calibration if "calibration" in state else None,
        )["M"]

    states = [
        frozenset(),
        frozenset({"robustness"}),
        frozenset({"calibration"}),
        frozenset({"robustness", "calibration"}),
    ]
    for subset in states:
        base = score(subset)
        for superset in states:
            if subset <= superset:
                assert score(superset) >= base, f"{sorted(subset)} -> {sorted(superset)}"


# ── rate_models: the rated gate and the component ledger ─────────────────────


def test_rate_models_marks_perf_only_unrated(comparator: ModelComparator):
    assessment = comparator.rate_models(_metrics())["M"]
    assert assessment["rated"] is False
    assert assessment["available_components"] == ["performance"]
    assert assessment["missing_components"] == ["robustness", "calibration"]
    assert assessment["components"]["performance"] == pytest.approx(87.5)
    assert assessment["components"]["robustness"] is None
    assert assessment["components"]["calibration"] is None


def test_rate_models_flips_rated_with_two_measured_components(
    comparator: ModelComparator,
):
    per_rob = comparator.rate_models(_metrics(), {"M": 60.0})["M"]
    assert per_rob["rated"] is True
    assert per_rob["missing_components"] == ["calibration"]

    per_cal = comparator.rate_models(_metrics(), None, {"M": 0.1})["M"]
    assert per_cal["rated"] is True
    assert per_cal["missing_components"] == ["robustness"]


def test_rate_models_skips_models_with_no_data(comparator: ModelComparator):
    assessed = comparator.rate_models(_metrics(has_data=False))
    assert assessed == {}


def test_compute_composite_score_delegates_to_rate_models(comparator: ModelComparator):
    metrics = _metrics()
    by_assessment = {
        m: a["composite"]
        for m, a in comparator.rate_models(metrics, {"M": 60.0}, {"M": 0.2}).items()
    }
    assert comparator.compute_composite_score(metrics, {"M": 60.0}, {"M": 0.2}) == by_assessment


# ── recommend_best_model: the rated gate ─────────────────────────────────────


def test_recommendation_ranks_only_rated_models(comparator: ModelComparator):
    metrics = {
        "Strong": _metrics()["M"],
        "Weak": _metrics(accuracy=0.5, f1=0.4)["M"],
    }
    composite = comparator.compute_composite_score(
        metrics, {"Strong": 90.0, "Weak": 30.0}, {"Strong": 0.05, "Weak": 0.4}
    )
    rec = comparator.recommend_best_model(
        metrics, composite, {"Strong": 90.0}, rated={"Strong": True, "Weak": False}
    )

    assert rec["best_model"] == "Strong"
    assert rec["ranking"] == ["Strong"]
    assert rec["unrated"] == ["Weak"]


def test_an_unrated_model_cannot_win_even_with_a_higher_composite(
    comparator: ModelComparator,
):
    """The gate is what keeps a perf-only model off the throne.

    Without ``rated``, the model with the highest *composite* would be crowned;
    the non-redistributing zero-for-missing policy makes that composite an
    artifact of missing data, so the gate must win over the raw number.
    """
    composite = {"Measured": 70.0, "Sparse": 90.0}
    metrics = _metrics_for(["Measured", "Sparse"])
    rec = comparator.recommend_best_model(
        metrics, composite, None, rated={"Measured": True, "Sparse": False}
    )
    assert rec["best_model"] == "Measured"
    assert "Sparse" in rec["unrated"]


def test_when_no_model_is_rated_the_recommendation_refuses(comparator: ModelComparator):
    rec = comparator.recommend_best_model(
        _metrics(), {"M": 43.75}, None, rated={"M": False}
    )
    assert rec["best_model"] is None
    assert rec["ranking"] == []
    assert rec["unrated"] == ["M"]
    assert "enough measured components" in rec["reason"]


def test_recommendation_without_an_rated_gate_treats_everyone_eligible(
    comparator: ModelComparator,
):
    metrics = _metrics()
    composite = {"M": 43.75}
    rec = comparator.recommend_best_model(metrics, composite)  # rated=None
    assert rec["best_model"] == "M"
    assert rec["ranking"] == ["M"]
    assert rec["unrated"] == []


def test_recommendation_with_no_composite_data(comparator: ModelComparator):
    rec = comparator.recommend_best_model(_metrics(), {})
    assert rec == {"best_model": None, "reason": "No data available.", "ranking": []}


def test_recommendation_reason_carries_the_measured_numbers(
    comparator: ModelComparator,
):
    metrics = _metrics(accuracy=0.9, f1=0.85)
    composite = comparator.compute_composite_score(metrics, {"M": 60.0}, {"M": 0.2})
    rec = comparator.recommend_best_model(metrics, composite, {"M": 60.0}, rated={"M": True})
    assert "0.900" in rec["reason"]
    assert "0.850" in rec["reason"]
    assert "60.0" in rec["reason"]


# ── Non-finite inputs must not escape into the published composite ────────────


def test_nan_evidence_never_produces_a_non_finite_composite(comparator: ModelComparator):
    """A ``NaN`` metric/reading never escapes into the published composite.

    ``NaN`` defeats both range comparisons, so before this the composite (and the
    grade derived from it) could be ``NaN`` for degenerate evidence — a silently
    corrupt published score rather than a rendered error (run-015 review). The
    performance component floors to ``0.0``; a non-finite robustness/ECE reading
    is now *missing* rather than a floored component (ADR-027).
    """
    assessment = comparator.rate_models(
        _metrics(accuracy=float("nan"), f1=float("nan")),
        {"M": float("nan")},
        {"M": float("nan")},
    )["M"]
    assert math.isfinite(assessment["composite"])
    assert assessment["composite"] == 0.0
    for value in assessment["components"].values():
        assert value is None or math.isfinite(value)


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_robustness_reading_is_missing_not_a_component(reading: float):
    """``nan``/``+inf``/``-inf`` robustness is not measured evidence.

    Previously all three floored to a ``0.0`` component that still counted as
    measured (``available``), while an absent entry was ``None``/``missing`` —
    two classes for one state.  A non-finite reading is now ``None`` and lands in
    ``missing_components``, never in ``available_components`` (ADR-027).
    """
    assessment = ModelComparator().rate_models(_metrics(), {"M": reading})["M"]
    assert assessment["components"]["robustness"] is None
    assert "robustness" in assessment["missing_components"]
    assert "robustness" not in assessment["available_components"]


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_calibration_reading_is_missing_not_a_component(reading: float):
    """A non-finite ECE is missing evidence, exactly like a non-finite robustness."""
    assessment = ModelComparator().rate_models(_metrics(), None, {"M": reading})["M"]
    assert assessment["components"]["calibration"] is None
    assert "calibration" in assessment["missing_components"]
    assert "calibration" not in assessment["available_components"]


def test_a_non_finite_reading_does_not_count_toward_rated():
    """A model padded to ``min_measured_components`` by a non-finite reading is unrated.

    With the default metrics (performance only) and a non-finite robustness the
    composite is the performance-only ``43.75``, ``rated`` is ``False`` and both
    optional components are missing (ADR-027).
    """
    assessment = ModelComparator().rate_models(_metrics(), {"M": float("inf")})["M"]
    assert assessment["composite"] == pytest.approx(43.75)
    assert assessment["rated"] is False
    assert assessment["missing_components"] == ["robustness", "calibration"]
    assert assessment["available_components"] == ["performance"]


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_reading_leaves_only_the_finite_sibling_measured(reading: float):
    """A finite reading and a non-finite sibling leave exactly one available."""
    assessment = ModelComparator().rate_models(
        _metrics(), {"M": reading}, {"M": 0.1}
    )["M"]
    assert assessment["components"]["robustness"] is None
    assert "robustness" in assessment["missing_components"]
    assert "calibration" in assessment["available_components"]
    assert assessment["rated"] is True  # performance + calibration


@pytest.mark.parametrize("bad", ["oops", None, object(), True, False])
def test_a_non_numeric_reading_raises_a_typed_validation_error(bad: object):
    """A caller-supplied non-numeric reading raises typed ``ValidationError``.

    ``bool`` is checked explicitly because it is a subclass of ``int`` and would
    otherwise coerce to ``1.0`` silently; ``None`` *as a value* is caller misuse,
    distinct from the model being absent from the dict (run-015 MIN-1, ADR-027).
    """
    for component, kwargs in (
        ("robustness", {"robustness_dict": {"M": bad}}),
        ("calibration", {"calibration_ece": {"M": bad}}),
    ):
        with pytest.raises(ValidationError) as excinfo:
            ModelComparator().rate_models(_metrics(), **kwargs)
        assert excinfo.value.context["model"] == "M"
        assert excinfo.value.context["component"] == component


def test_a_model_absent_from_a_reading_dict_stays_missing():
    """A missing key is not a caller error: it stays a missing component.

    Only a value supplied *for the model* is normalised; ``_normalise_reading``
    must never be reached for an absent key (which would raise instead of
    reporting the component as unmeasured).
    """
    assessment = ModelComparator().rate_models(
        _metrics(), {"Other": 60.0}, {"Other": 0.1}
    )["M"]
    assert assessment["components"]["robustness"] is None
    assert assessment["components"]["calibration"] is None
    assert assessment["missing_components"] == ["robustness", "calibration"]


def test_finite_readings_are_still_clamped_to_the_component_range():
    """Finite readings still clamp to ``[0.0, 100.0]`` and count as measured."""
    low = ModelComparator().rate_models(_metrics(), {"M": -5.0})["M"]
    high = ModelComparator().rate_models(_metrics(), {"M": 250.0})["M"]
    assert low["components"]["robustness"] == 0.0
    assert "robustness" in low["available_components"]
    assert high["components"]["robustness"] == 100.0
    assert "robustness" in high["available_components"]


def test_performance_component_uses_the_shared_transform(comparator: ModelComparator):
    """The comparator must not restate ``(accuracy + f1) / 2`` inline.

    A clamp or ``NaN`` policy fixed in ``core.scoring`` would silently not apply
    to a hand-copied formula here (single-owner rule, ADR-021).
    """
    from core.scoring import performance_fraction

    assessment = comparator.rate_models(_metrics(accuracy=0.8, f1=0.6))["M"]
    assert assessment["components"]["performance"] == pytest.approx(
        performance_fraction(0.8, 0.6) * 100.0
    )


def test_plot_robustness_comparison_uses_reliability_weights_palette():
    """plot_robustness_comparison derives bar colors from ReliabilityWeights.resolve_grade."""
    comparator = ModelComparator()
    scores = {"M_high": 95.0, "M_mid": 75.0, "M_low": 35.0}
    fig = comparator.plot_robustness_comparison(scores)

    expected_colors = [comparator.weights.resolve_grade(s)[1] for s in scores.values()]
    assert list(fig.data[0].marker.color) == expected_colors

    # Injected custom weights palette is respected
    custom_weights = ReliabilityWeights(
        grade_colours=(
            (90.0, "#111111"),
            (80.0, "#222222"),
            (70.0, "#333333"),
            (60.0, "#444444"),
            (50.0, "#555555"),
            (0.0, "#999999"),
        )
    )
    custom_comp = ModelComparator(weights=custom_weights)
    custom_fig = custom_comp.plot_robustness_comparison(scores)
    custom_expected = [custom_weights.resolve_grade(s)[1] for s in scores.values()]
    assert list(custom_fig.data[0].marker.color) == custom_expected


def test_plot_composite_scores_uses_reliability_weights_palette():
    """plot_composite_scores derives bar colors from ReliabilityWeights.resolve_grade in sorted order."""
    comparator = ModelComparator()
    scores = {"M_low": 42.0, "M_high": 88.0, "M_fail": 20.0}
    fig = comparator.plot_composite_scores(scores)

    # plot_composite_scores sorts descending by score: M_high (88.0), M_low (42.0), M_fail (20.0)
    sorted_scores = [88.0, 42.0, 20.0]
    expected_colors = [comparator.weights.resolve_grade(s)[1] for s in sorted_scores]
    assert list(fig.data[0].marker.color) == expected_colors