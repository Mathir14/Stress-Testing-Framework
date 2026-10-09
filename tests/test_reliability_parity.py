"""Bit-exact parity against the frozen pre-refactor baseline.

**Tier T1** — the scorer's inputs are floats, but the module imports pandas for
its summary frames, so the numeric stack is required.

This is the **one true numeric freeze** in the milestone.  Everything else —
perturbations in particular — is explicitly *not* bit-compatible
(ADR-010.5: a vectorised implementation draws randomness in a different order),
so parity there means "same formula, seeded self-consistency".  Reliability
scoring is pure arithmetic on caller-supplied floats with no RNG and no
platform-dependent library calls, which makes an exact freeze both possible and
worth having: it is the only thing standing between a refactor and a silently
changed letter grade in a published results table.

The fixture is ``tests/fixtures/reliability_baseline.json``, produced by
``python -m tests.capture_baseline`` on the interpreter recorded inside it.
Comparisons use ``==`` on the exact rounded values, not ``approx`` — a tolerance
here would defeat the purpose of the test.

Regenerate with ``python -m tests.capture_baseline``; verify drift with
``python -m tests.capture_baseline --check`` (Wave 6 gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import FAILING_GRADE, get_config
from modules.reliability_module import ReliabilityScorer, _grade

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "reliability_baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    """Load the frozen baseline payload."""
    if not FIXTURE_PATH.is_file():
        pytest.skip(
            f"baseline fixture {FIXTURE_PATH} is missing; run `python -m tests.capture_baseline`"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scorer() -> ReliabilityScorer:
    return ReliabilityScorer()


# ── Interpreter provenance ───────────────────────────────────────────────────


def test_fixture_records_the_interpreter_it_was_captured_on(baseline: dict):
    """ADR-010.9: a fixture captured on a different interpreter is not comparable.

    The check is on the *record*, not on the running interpreter — a machine
    running 3.11 must be able to assert parity against a fixture captured on
    3.14, otherwise the gate is unusable in the devcontainer.
    """
    interpreter = baseline["interpreter"]
    assert set(interpreter) == {
        "python_version",
        "python_implementation",
        "platform",
    }
    assert interpreter["python_version"].startswith("3.")
    assert interpreter["python_implementation"] == "CPython"


def test_fixture_declares_its_schema_version(baseline: dict):
    assert baseline["schema_version"] == 1


def test_fixture_was_captured_on_the_current_platform_family(baseline: dict):
    """Recorded so a platform change is visible rather than mysterious.

    The fixture stores the lowercased form ``"cpython"``; compared case-folded so
    this stays a check on the *interpreter family* and not on formatting.
    """
    import sys

    recorded = baseline["interpreter"]["python_implementation"]
    assert recorded.lower() == sys.implementation.name.lower()
    assert recorded == recorded.strip() and recorded


# ── Grade boundaries ─────────────────────────────────────────────────────────


def test_every_grade_probe_matches(baseline: dict):
    """90/80/70/60/50 and the ``<50 → "F"`` fallthrough, exactly."""
    for label, expected in baseline["grades"].items():
        score = float(label)
        assert _grade(score)[0] == expected, f"score {score} graded {expected!r}"


def test_every_grade_colour_matches(baseline: dict):
    """The colour is part of the contract — the UI renders it as a badge."""
    for label, expected in baseline["grade_colours"].items():
        score = float(label)
        assert _grade(score)[1] == expected, f"score {score} coloured {expected!r}"


@pytest.mark.parametrize(
    "score,grade",
    [
        (100.0, "A+"),
        (90.0, "A+"),
        (89.999, "A"),
        (80.0, "A"),
        (79.999, "B"),
        (70.0, "B"),
        (69.999, "C"),
        (60.0, "C"),
        (59.999, "D"),
        (50.0, "D"),
        (49.999, "F"),
        (0.0, "F"),
        (-10.0, "F"),
    ],
)
def test_grade_boundaries_are_inclusive_at_the_cutoff(score: float, grade: str):
    """The cutoff value itself belongs to the higher grade.

    Written out rather than read from the fixture so a fixture edit cannot make
    this tautological; the fixture test above is what proves the recorded
    behaviour is unchanged.
    """
    assert _grade(score)[0] == grade


def test_below_the_last_cutoff_falls_through_to_the_declared_grade():
    """ADR-009 §6: an explicit ``F``, not an ``IndexError`` or ``None``."""
    weights = get_config().reliability
    last_cutoff = weights.grade_cutoffs[-1][0]
    assert _grade(last_cutoff - 0.001)[0] == FAILING_GRADE
    assert _grade(-1e9)[0] == FAILING_GRADE


def test_grade_is_monotone_in_score():
    """A higher score never grades worse.

    Swept in ascending order, so the rank sequence must be non-*increasing*:
    ``F`` (worst) comes first and ``A+`` (best) last.  Asserting it the other way
    round would pass trivially, which is worth stating because it is the easiest
    way to make this test meaningless.
    """
    order = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4, "F": 5}
    ranks = [order[_grade(score / 2)[0]] for score in range(0, 201)]
    assert ranks == sorted(ranks, reverse=True), "grade ranking is not monotone"
    assert ranks[0] == order["F"]
    assert ranks[-1] == order["A+"]


# ── Component scores, exactly ────────────────────────────────────────────────


def test_component_maxima_match(baseline: dict, scorer: ReliabilityScorer):
    """§4.2: the config weights and the class constants must not drift apart."""
    weights = get_config().reliability
    assert baseline["maxima"]["MAX_PERF"] == scorer.MAX_PERF == weights.max_perf
    assert baseline["maxima"]["MAX_CAL"] == scorer.MAX_CAL == weights.max_cal
    assert baseline["maxima"]["MAX_ROB"] == scorer.MAX_ROB == weights.max_rob
    assert baseline["maxima"]["MAX_CONF"] == scorer.MAX_CONF == weights.max_conf


def test_every_scoring_case_matches_bit_exactly(baseline: dict, scorer: ReliabilityScorer):
    """The four component scores, the total, the grade and the colour.

    ``==``, not ``approx``: each value was already rounded to three decimals by
    the scorer, so any difference at all is a change in the arithmetic.
    """
    from tests.capture_baseline import N_CLASS_CASES, SCORING_CASES

    checked = 0
    for name, kwargs in SCORING_CASES.items():
        for n_classes in N_CLASS_CASES:
            key = name if n_classes == 2 else f"{name}__nc{n_classes}"
            expected = baseline["cases"][key]
            actual = scorer.score_model(name, n_classes=n_classes, **kwargs)
            for field in (
                "performance",
                "calibration",
                "robustness",
                "confidence",
                "total",
            ):
                assert actual[field] == expected[field], (
                    f"{key}.{field}: {actual[field]!r} != baseline {expected[field]!r}"
                )
            assert actual["grade"] == expected["grade"], key
            assert actual["colour"] == expected["colour"], key
            assert actual["available_components"] == expected["available_components"], key
            assert actual["missing_components"] == expected["missing_components"], key
            checked += 1
    assert checked == len(SCORING_CASES) * len(N_CLASS_CASES)


def test_the_baseline_fixture_covers_every_case_name(baseline: dict):
    """A case silently dropped from the fixture would reduce coverage silently."""
    from tests.capture_baseline import N_CLASS_CASES, SCORING_CASES

    expected_keys = {
        name if n == 2 else f"{name}__nc{n}" for name in SCORING_CASES for n in N_CLASS_CASES
    }
    assert set(baseline["cases"]) == expected_keys


# ── Component formulas, derived independently of the scorer ──────────────────


def test_performance_component_formula():
    """``((accuracy + f1) / 2) * 25``, rounded to 3 dp."""
    scorer = ReliabilityScorer()
    assert scorer._performance_score(1.0, 1.0) == 25.0
    assert scorer._performance_score(0.0, 0.0) == 0.0
    assert scorer._performance_score(0.5, 0.5) == 12.5
    assert scorer._performance_score(0.9, 0.7) == round((0.8 * 25.0), 3)


def test_calibration_component_formula_and_clip():
    """``(1 - ece/0.5) * 25``, clipped at zero."""
    scorer = ReliabilityScorer()
    assert scorer._calibration_score(0.0) == 25.0
    assert scorer._calibration_score(0.25) == 12.5
    assert scorer._calibration_score(0.5) == 0.0
    assert scorer._calibration_score(5.0) == 0.0, "must clip, not go negative"


def test_robustness_component_formula_and_clip():
    """``(1 - avg_drop/0.5) * 25``, clipped at zero."""
    scorer = ReliabilityScorer()
    assert scorer._robustness_score(0.0) == 25.0
    assert scorer._robustness_score(0.25) == 12.5
    assert scorer._robustness_score(0.5) == 0.0
    assert scorer._robustness_score(10.0) == 0.0


def test_confidence_component_splits_the_cap_in_half():
    """12.5 points for entropy, 12.5 for HCE rate."""
    import math

    scorer = ReliabilityScorer()
    perfect = scorer._confidence_score(0.0, 0.0, max_entropy=math.log2(2))
    assert perfect == 25.0
    worst = scorer._confidence_score(10.0, 1.0, max_entropy=math.log2(2))
    assert worst == 0.0


def test_confidence_normalises_entropy_by_log2_n_classes():
    """ADR-005: the ``log2(n_classes)`` normalisation is part of the contract."""
    import math

    scorer = ReliabilityScorer()
    for n_classes in (2, 3, 5):
        max_entropy = math.log2(n_classes)
        partial = scorer._confidence_score(max_entropy / 2, 0.0, max_entropy)
        assert partial == pytest.approx(18.75), n_classes


def test_confidence_clamps_excess_entropy():
    """Entropy above ``log2(n)`` must not produce negative points."""
    import math

    scorer = ReliabilityScorer()
    over = scorer._confidence_score(100.0, 0.0, max_entropy=math.log2(2))
    assert over == pytest.approx(12.5), "entropy contribution must floor at 0"


def test_confidence_handles_a_zero_max_entropy():
    """A degenerate ``n_classes == 1`` must not divide by zero."""
    scorer = ReliabilityScorer()
    assert scorer._confidence_score(0.0, 0.0, 0.0) == 25.0


# ── Missing-component fallbacks ──────────────────────────────────────────────


def test_missing_components_score_zero(baseline: dict, scorer: ReliabilityScorer):
    """A missing component scores ``missing_component_points`` (0.0) — ADR-018.

    The pre-remediation midpoint (half the cap, ``12.5``) is gone: a model with
    no measured components must not be propped up by neutral points, because a
    performance-only model could then outrank one with honest measurements.
    """
    result = scorer.score_model("empty")
    for field in ("performance", "calibration", "robustness", "confidence"):
        assert result[field] == 0.0, field
    assert result["missing_components"] == [
        "Performance",
        "Calibration",
        "Robustness",
        "Confidence",
    ]
    assert result["available_components"] == []
    assert result["rated"] is False


def test_partial_input_flags_exactly_the_missing_components(scorer: ReliabilityScorer):
    """``partial_performance`` supplies only accuracy and f1."""
    result = scorer.score_model("partial", accuracy=0.83, f1=0.81)
    assert result["available_components"] == ["Performance"]
    assert result["missing_components"] == ["Calibration", "Robustness", "Confidence"]
    assert result["rated"] is False, "one measured component is below the gate"
    assert result["total"] == scorer._performance_score(0.83, 0.81)


def test_a_component_needing_both_of_its_inputs_is_unavailable_with_one():
    """Performance needs accuracy *and* f1; one alone is not a partial score."""
    scorer = ReliabilityScorer()
    only_accuracy = scorer.score_model("x", accuracy=0.9)
    assert "Performance" in only_accuracy["missing_components"]
    assert only_accuracy["performance"] == scorer.policy.missing_component_points

    only_f1 = scorer.score_model("x", f1=0.9)
    assert "Performance" in only_f1["missing_components"]


def test_zero_is_a_value_not_a_missing_input():
    """``accuracy=0.0`` must score 0, not fall back to the midpoint.

    A falsy-check bug here would be invisible for every other case and would
    quietly reward a model that predicts nothing.
    """
    scorer = ReliabilityScorer()
    result = scorer.score_model("zeros", accuracy=0.0, f1=0.0)
    assert "Performance" in result["available_components"]
    assert result["performance"] == 0.0


def test_total_is_the_sum_of_the_four_components(scorer: ReliabilityScorer):
    """No hidden weighting: the total is a plain sum of the four caps."""
    result = scorer.score_model(
        "sum",
        accuracy=0.9,
        f1=0.85,
        ece=0.08,
        avg_drop=0.2,
        avg_entropy=0.3,
        hce_rate=0.05,
    )
    assert result["total"] == (
        result["performance"] + result["calibration"] + result["robustness"] + result["confidence"]
    )


def test_total_cannot_exceed_the_sum_of_caps(scorer: ReliabilityScorer):
    """100 is the ceiling; a component overshoot would break the grade scale."""
    result = scorer.score_model(
        "max",
        accuracy=1.0,
        f1=1.0,
        ece=0.0,
        avg_drop=0.0,
        avg_entropy=0.0,
        hce_rate=0.0,
    )
    assert result["total"] == 100.0
    assert result["grade"] == "A+"


def test_details_echo_the_inputs(scorer: ReliabilityScorer):
    """The UI renders ``details``; it must reflect what was actually supplied."""
    result = scorer.score_model(
        "echo",
        accuracy=0.9,
        f1=0.85,
        ece=0.08,
        avg_drop=0.2,
        avg_entropy=0.3,
        hce_rate=0.05,
    )
    assert result["details"] == {
        "accuracy": 0.9,
        "f1": 0.85,
        "ece": 0.08,
        "avg_drop": 0.2,
        "avg_entropy": 0.3,
        "hce_rate": 0.05,
    }


def test_more_classes_raise_the_confidence_floor(scorer: ReliabilityScorer):
    """``max_entropy`` is ``log2(n)``, which grows with ``n``.

    A fixed absolute entropy therefore becomes a *smaller fraction* of the
    maximum, so the same number scores **higher** for 5 classes than for 2 —
    entropy is unbounded in absolute terms, and the normalisation is what makes
    it comparable.  Asserted in the direction the formula actually implies;
    getting this backwards would be the kind of plausible-looking but false
    claim a parity test should not encode.
    """
    binary = scorer.score_model("m", avg_entropy=0.5, hce_rate=0.0, n_classes=2)
    quinary = scorer.score_model("m", avg_entropy=0.5, hce_rate=0.0, n_classes=5)
    assert quinary["confidence"] > binary["confidence"]
    # At the maximum entropy for each arity, both score 12.5 — the normalisation
    # makes "maximum possible entropy" arity-independent.
    import math

    assert (
        scorer.score_model("m", avg_entropy=math.log2(2), hce_rate=0.0, n_classes=2)["confidence"]
        == 12.5
    )
    assert (
        scorer.score_model("m", avg_entropy=math.log2(5), hce_rate=0.0, n_classes=5)["confidence"]
        == 12.5
    )


# ── Determinism ──────────────────────────────────────────────────────────────


def test_scoring_is_deterministic(scorer: ReliabilityScorer):
    """No RNG, no clock, no dict-order dependence — the whole point of the freeze."""
    kwargs = dict(
        accuracy=0.83,
        f1=0.79,
        ece=0.11,
        avg_drop=0.22,
        avg_entropy=0.4,
        hce_rate=0.06,
        n_classes=3,
    )
    first = scorer.score_model("repeat", **kwargs)
    second = ReliabilityScorer().score_model("repeat", **kwargs)
    assert first == second


def test_scorer_instances_share_the_same_capitals(scorer: ReliabilityScorer):
    """The capitals are class attributes, not per-instance state."""
    other = ReliabilityScorer()
    assert other.MAX_PERF == scorer.MAX_PERF
    assert other._performance_score(0.5, 0.5) == scorer._performance_score(0.5, 0.5)


def test_model_name_does_not_affect_the_score(scorer: ReliabilityScorer):
    """The name is metadata; it must not leak into the arithmetic."""
    kwargs = dict(accuracy=0.9, f1=0.9, ece=0.1)
    assert (
        scorer.score_model("Logistic Regression", **kwargs)["total"]
        == scorer.score_model("Random Forest", **kwargs)["total"]
    )


# ── Monotonicity in evidence ─────────────────────────────────────────────────


def test_adding_any_measured_component_never_lowers_the_total(
    scorer: ReliabilityScorer,
):
    """ADR-018: fixed denominator, zero for missing — evidence is monotone.

    Every component contributes a non-negative number of points and the missing
    value is ``missing_component_points`` (0.0), so replacing "missing" with a
    measured component can only add to the total.  All 16 subsets of the four
    components are enumerated rather than sampled, so the property cannot be
    missed by an unlucky draw.
    """
    from itertools import combinations

    components = {
        "performance": dict(accuracy=0.9, f1=0.85),
        "calibration": dict(ece=0.1),
        "robustness": dict(avg_drop=0.2),
        "confidence": dict(avg_entropy=0.3, hce_rate=0.05),
    }
    names = list(components)
    for size in range(len(names) + 1):
        for subset in combinations(names, size):
            kwargs: dict = {}
            for name in subset:
                kwargs.update(components[name])
            base = scorer.score_model("m", **kwargs)["total"]
            for extra in names:
                if extra in subset:
                    continue
                more = {**kwargs, **components[extra]}
                assert scorer.score_model("m", **more)["total"] >= base, (
                    f"adding {extra} to {subset} lowered the total"
                )
