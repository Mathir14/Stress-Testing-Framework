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

import pytest

from core.config import ScoringPolicy, get_config
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
    # 87.5*0.5 + 60.0*0.25 + 80.0*0.25 = 43.75 + 15.0 + 20.0
    assert composite["M"] == pytest.approx(78.75)


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
    good = comparator.compute_composite_score(metrics, None, {"M": 0.25})  # 75.0 x 0.25
    awful = comparator.compute_composite_score(metrics, None, {"M": 2.0})  # clipped to 0
    assert good["M"] == pytest.approx(43.75 + 18.75)
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
    calibration = {"M": 0.18}  # -> an 82.0 component

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