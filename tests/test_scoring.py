"""Contract tests for ``core/scoring.py`` (ADR-021).

Tier T0-safe: this file imports stdlib + pytest + ``core.scoring`` only, which
imports nothing project-local.

The load-bearing invariant (architectural spec run-015 §4): every ``*_fraction``
returns a value in ``[0.0, 1.0]`` clamped at **both** ends; ``None`` means
"missing", never "zero"; ``combine_fractions`` gives a ``None`` fraction zero
weight with no redistribution.  These are the properties the reliability and
comparison modules delegate to, so regressing them here regresses the published
scores silently — which is exactly why they are pinned.
"""

from __future__ import annotations

import math

import pytest

from core.scoring import (
    calibration_fraction,
    combine_fractions,
    confidence_fraction,
    mean_performance_drop,
    performance_fraction,
    robustness_fraction,
)

# ── Performance ───────────────────────────────────────────────────────────────


def test_performance_fraction_is_the_mean() -> None:
    assert performance_fraction(0.9, 0.7) == pytest.approx(0.8)
    assert performance_fraction(1.0, 1.0) == 1.0
    assert performance_fraction(0.0, 0.0) == 0.0


def test_performance_fraction_clamps_both_ends() -> None:
    """A negative or out-of-range measurement must not escape [0, 1]."""
    assert performance_fraction(-0.2, 0.1) == 0.0
    assert performance_fraction(1.5, 1.5) == 1.0


# ── Calibration ───────────────────────────────────────────────────────────────


def test_calibration_fraction_linear_in_ece() -> None:
    assert calibration_fraction(0.0, ece_cap=0.5) == 1.0
    assert calibration_fraction(0.25, ece_cap=0.5) == pytest.approx(0.5)
    assert calibration_fraction(0.5, ece_cap=0.5) == 0.0


def test_calibration_fraction_clamps_both_ends() -> None:
    assert calibration_fraction(5.0, ece_cap=0.5) == 0.0
    assert calibration_fraction(-0.1, ece_cap=0.5) == 1.0


# ── Robustness ────────────────────────────────────────────────────────────────


def test_robustness_fraction_linear_in_drop() -> None:
    assert robustness_fraction(0.0, drop_cap=0.5) == 1.0
    assert robustness_fraction(0.25, drop_cap=0.5) == pytest.approx(0.5)
    assert robustness_fraction(0.5, drop_cap=0.5) == 0.0


def test_robustness_fraction_clamps_both_ends() -> None:
    """A negative drop (a stress run that looks like an improvement) must not
    inflate the fraction past 1.0 — run-015 C1's exact defect."""
    assert robustness_fraction(-1.0, drop_cap=0.5) == 1.0
    assert robustness_fraction(10.0, drop_cap=0.5) == 0.0


# ── Confidence ────────────────────────────────────────────────────────────────


def test_confidence_fraction_perfect_and_worst() -> None:
    assert confidence_fraction(0.0, 0.0, n_classes=2) == 1.0
    assert confidence_fraction(10.0, 1.0, n_classes=2) == 0.0


def test_confidence_fraction_splits_the_cap_in_half() -> None:
    """Entropy and HCE each own half the fraction (12.5 of the 25 cap)."""
    max_entropy = math.log2(5)
    partial = confidence_fraction(max_entropy / 2, 0.0, n_classes=5)
    assert partial == pytest.approx(0.75)


def test_confidence_fraction_clamps_excess_entropy() -> None:
    over = confidence_fraction(100.0, 0.0, n_classes=2)
    assert over == pytest.approx(0.5), "entropy contribution must floor at 0"


def test_confidence_fraction_handles_degenerate_n_classes() -> None:
    """``n_classes < 2`` must not divide by zero; the floor is log2(2)."""
    assert confidence_fraction(0.0, 0.0, n_classes=1) == 1.0


def test_confidence_fraction_clamps_hce_both_ends() -> None:
    assert confidence_fraction(0.0, -0.5, n_classes=2) == 1.0
    assert confidence_fraction(0.0, 2.0, n_classes=2) == pytest.approx(0.5)


# ── Mean performance drop ─────────────────────────────────────────────────────


def test_mean_performance_drop_averages_numeric_entries() -> None:
    results = {
        "noise": {"performance_drop": 0.1},
        "dropout": {"performance_drop": 0.2},
        "shift": {"performance_drop": 0.3},
    }
    assert mean_performance_drop(results) == pytest.approx(0.2)


def test_mean_performance_drop_skips_entries_without_numeric_drop() -> None:
    """An entry missing ``performance_drop`` (or carrying a non-numeric one) is
    not evidence of a zero drop — it contributes nothing."""
    results = {
        "noise": {"performance_drop": 0.1},
        "missing_key": {"performance_drop_pct": 3.0},
        "non_numeric": {"performance_drop": "n/a"},
        "not_a_dict": 5,
    }
    assert mean_performance_drop(results) == pytest.approx(0.1)


def test_mean_performance_drop_is_none_when_no_numeric_evidence() -> None:
    """run-015 C1b: empty or non-numeric evidence is ``None``, never ``0.0`` —
    an unmeasured robustness must not score full marks for a zero drop."""
    assert mean_performance_drop({}) is None
    assert (
        mean_performance_drop({"noise": {"performance_drop_pct": 3.0}}) is None
    )
    assert mean_performance_drop({"noise": 5}) is None


def test_mean_performance_drop_does_not_mutate_input() -> None:
    results = {"noise": {"performance_drop": 0.1}}
    before = dict(results)
    mean_performance_drop(results)
    assert results == before


# ── Combine fractions ─────────────────────────────────────────────────────────


def test_combine_fractions_is_the_weighted_sum() -> None:
    fractions = {"performance": 0.5, "robustness": 0.5, "calibration": 0.5}
    weights = {"performance": 0.5, "robustness": 0.25, "calibration": 0.25}
    assert combine_fractions(fractions, weights) == pytest.approx(0.5)


def test_combine_fractions_gives_none_zero_weight() -> None:
    """A ``None`` component contributes zero; its weight is NOT redistributed."""
    fractions = {"performance": 0.5, "robustness": None}
    weights = {"performance": 0.5, "robustness": 0.25, "calibration": 0.25}
    assert combine_fractions(fractions, weights) == pytest.approx(0.25)


def test_combine_fractions_missing_component_is_treated_as_none() -> None:
    weights = {"performance": 0.5, "robustness": 0.25}
    assert combine_fractions({"performance": 0.8}, weights) == pytest.approx(0.4)


def test_combine_fractions_does_not_renormalise_weights() -> None:
    """The weights are the policy's fixed denominators, never re-normalised."""
    fractions = {"performance": 0.5, "robustness": 0.5}
    weights = {"performance": 1.0, "robustness": 1.0}
    assert combine_fractions(fractions, weights) == pytest.approx(1.0)


# ── Non-finite (NaN) inputs ───────────────────────────────────────────────────
#
# ``NaN`` is the one float that defeats both range comparisons (``nan <= 0.0``
# and ``nan >= 1.0`` are both ``False``), so it escaped the module's stated
# ``[0.0, 1.0]`` contract entirely and propagated into the published total, the
# letter grade and the report gauge.  A degenerate measurement is *unknown*, not
# *good*, so it takes the no-credit floor (run-015 review regression).

NAN = float("nan")


def test_performance_fraction_rejects_nan() -> None:
    assert performance_fraction(NAN, NAN) == 0.0


def test_calibration_fraction_rejects_nan() -> None:
    assert calibration_fraction(NAN, ece_cap=0.5) == 0.0


def test_robustness_fraction_rejects_nan() -> None:
    assert robustness_fraction(NAN, drop_cap=0.5) == 0.0


def test_confidence_fraction_rejects_nan_entropy_as_worst_not_best() -> None:
    """``NaN`` entropy must score the *worst* confidence, not the best.

    ``_clamp01(nan)`` returns ``0.0`` (no damage), which is correct for
    robustness/calibration but would invert into a *perfect* confidence quarter
    here, because a low entropy is the good outcome.  ``NaN`` therefore floors
    the entropy half explicitly; the valid HCE half still scores normally.
    """
    assert confidence_fraction(NAN, 0.0, n_classes=2) == pytest.approx(0.5)
    assert confidence_fraction(0.0, NAN, n_classes=2) == pytest.approx(0.5)
    assert confidence_fraction(NAN, NAN, n_classes=2) == 0.0


def test_every_fraction_is_finite_for_nan_inputs() -> None:
    """The module contract: no ``*_fraction`` may return a non-finite value."""
    for value in (
        performance_fraction(NAN, NAN),
        calibration_fraction(NAN, ece_cap=0.5),
        robustness_fraction(NAN, drop_cap=0.5),
        confidence_fraction(NAN, NAN, n_classes=2),
    ):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0