"""Score-monotonicity property test for the two scoring engines (ADR-018).

**Tier T1** — requires numpy.

The central promise of ADR-018 and critic C3 is that **more evidence never
lowers a score**: a missing component contributes ``0.0`` with no redistribution,
so measuring an extra component (or improving a measured one) is always safe.
This file asserts the property as a *lattice* gate over both engines:

* :class:`modules.reliability_module.ReliabilityScorer` — the four 25-point
  components (performance / calibration / robustness / confidence).
* :class:`modules.comparison_module.ModelComparator` — the 0-100 composite.

For every subset of the component groups, every superset must score at least as
high, and the total must equal the plain sum of the weighted components.  The
property is written the safe direction: scores are swept with larger and larger
evidence sets, so ``>=`` is the only honest reading.

Also pinned directly: a component whose measured value is *zero* must keep the
total exactly constant — the "no redistribution" half that a weighted-rescale
implementation would violate while still being monotone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from modules.comparison_module import ModelComparator
from modules.reliability_module import ReliabilityScorer

RNG = np.random.default_rng(11)

#: The four component groups and the keyword arguments that make each one
#: available.  Performance and Confidence each need *both* of their inputs.
RELIABILITY_GROUPS: dict[str, dict[str, float]] = {
    "Performance": {"accuracy": 0.82, "f1": 0.78},
    "Calibration": {"ece": 0.09},
    "Robustness": {"avg_drop": 0.18},
    "Confidence": {"avg_entropy": 0.35, "hce_rate": 0.04},
}


def _reliability_score(scorer: ReliabilityScorer, subset: tuple[str, ...]) -> dict:
    kwargs: dict[str, float] = {}
    for name in subset:
        kwargs.update(RELIABILITY_GROUPS[name])
    return scorer.score_model("monotone", **kwargs)


def test_reliability_total_is_monotone_over_the_component_lattice():
    """Every superset of measured components totals >= every subset's total."""
    scorer = ReliabilityScorer()
    names = list(RELIABILITY_GROUPS)
    states = [
        combo
        for size in range(len(names) + 1)
        for combo in itertools.combinations(names, size)
    ]
    checked = 0
    for subset in states:
        base = _reliability_score(scorer, subset)
        for name in names:
            if name in subset:
                continue
            grown = _reliability_score(scorer, subset + (name,))
            assert grown["total"] >= base["total"], (
                f"adding {name} lowered the total: {base['total']} -> {grown['total']}"
            )
            checked += 1
    assert checked == 4 * 2**3  # 4 edges out of every subset


def test_reliability_total_is_always_the_plain_component_sum():
    """No hidden reweighting: ``total == perf + cal + rob + conf`` for every subset."""
    scorer = ReliabilityScorer()
    names = list(RELIABILITY_GROUPS)
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            result = _reliability_score(scorer, subset)
            parts = (
                result["performance"],
                result["calibration"],
                result["robustness"],
                result["confidence"],
            )
            # total is rounded to 2 dp while each component is rounded to 3 dp,
            # so the invariant holds within the rounding error of the four parts.
            assert result["total"] == pytest.approx(sum(parts), abs=0.01), subset


def test_reliability_rated_flips_at_min_measured_components():
    scorer = ReliabilityScorer()
    gate = scorer.policy.min_measured_components
    names = list(RELIABILITY_GROUPS)
    for subset in itertools.chain.from_iterable(
        itertools.combinations(names, size) for size in range(len(names) + 1)
    ):
        result = _reliability_score(scorer, subset)
        assert result["rated"] == (
            len(result["available_components"]) >= gate
        ), subset
        assert len(result["available_components"]) == len(subset)
        assert sorted(result["missing_components"]) == sorted(
            set(names) - set(subset)
        )


def test_reliability_adding_a_zero_component_keeps_the_total_constant():
    """``ece=0.5`` scores exactly zero calibration points: total stays 25.0."""
    scorer = ReliabilityScorer()
    before = scorer.score_model("m", accuracy=1.0, f1=1.0)
    after = scorer.score_model("m", accuracy=1.0, f1=1.0, ece=scorer.weights.ece_cap)
    assert before["total"] == 25.0
    assert after["calibration"] == 0.0
    assert after["total"] == before["total"]


def test_reliability_is_monotone_over_random_inputs():
    """Random sweeps: the component-sum invariant holds for arbitrary values."""
    scorer = ReliabilityScorer()
    for _ in range(25):
        kwargs = {
            name: float(RNG.uniform(0.0, 1.0))
            for name in ("accuracy", "f1", "ece", "avg_drop", "avg_entropy", "hce_rate")
        }
        result = scorer.score_model("sweep", **kwargs)
        parts = (
            result["performance"],
            result["calibration"],
            result["robustness"],
            result["confidence"],
        )
        # Rounding context: total is 2 dp, components 3 dp (see the lattice test).
        assert result["total"] == pytest.approx(sum(parts), abs=0.01)
        assert result["total"] >= 0.0
        assert result["total"] <= 100.0
        bigger = {**kwargs, "accuracy": min(1.0, kwargs["accuracy"] + 0.1)}
        assert scorer.score_model("sweep", **bigger)["total"] >= result["total"] - 1e-9


# ── ModelComparator composite ────────────────────────────────────────────────


def _metrics(accuracy: float = 0.9, f1: float = 0.85) -> dict[str, dict]:
    return {
        "M": {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": f1,
            "f1": f1,
            "has_data": True,
        }
    }


def _composite(comparator: ModelComparator, state: frozenset[str]) -> float:
    return comparator.compute_composite_score(
        _metrics(),
        {"M": 61.2} if "robustness" in state else None,
        {"M": 0.18} if "calibration" in state else None,
    )["M"]


def test_composite_is_monotone_over_the_component_lattice():
    comparator = ModelComparator()
    states = [
        frozenset(),
        frozenset({"robustness"}),
        frozenset({"calibration"}),
        frozenset({"robustness", "calibration"}),
    ]
    for subset in states:
        base = _composite(comparator, subset)
        for superset in states:
            if subset <= superset:
                assert _composite(comparator, superset) >= base, (
                    f"{sorted(subset)} -> {sorted(superset)}"
                )


def test_composite_equals_the_weighted_component_sum():
    """Fixed denominator: ``perf*0.5 + rob*0.25 + cal*0.25``, missing -> 0."""
    comparator = ModelComparator()
    perf = 87.5  # (0.9 + 0.85) / 2 * 100
    states = {
        frozenset(): (0.0, 0.0),
        frozenset({"robustness"}): (61.2, 0.0),
        frozenset({"calibration"}): (0.0, 82.0),
        frozenset({"robustness", "calibration"}): (61.2, 82.0),
    }
    for state, (rob, cal) in states.items():
        expected = (
            perf * comparator.policy.weight_for("performance")
            + rob * comparator.policy.weight_for("robustness")
            + cal * comparator.policy.weight_for("calibration")
        )
        assert _composite(comparator, state) == pytest.approx(expected, abs=1e-9), state


def test_composite_adding_a_zero_component_keeps_the_score():
    """A terrible robustness measurement (0.0/100) must not lower the composite."""
    comparator = ModelComparator()
    before = comparator.compute_composite_score(_metrics())
    after = comparator.compute_composite_score(_metrics(), {"M": 0.0})
    assert before["M"] == after["M"]