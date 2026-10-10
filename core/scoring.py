"""Unified monotone scoring transforms — the single owner of scoring arithmetic.

Architecture reference: ``architecture.md`` §2.3 (rung-2 sibling like
``core.reporters``), §4.2; ADR-018, ADR-021.

This module owns every component *transform*: the arithmetic that turns a raw
measurement into a 0-1 fraction.  ``modules/reliability_module.py``,
``modules/comparison_module.py`` and the views delegate to it, so the same
"robustness" concept cannot be computed three different ways (run-015 C1/M2/M3).

The *policy* — component caps, ``ece_cap``/``drop_cap``, composite weights,
grade bands — lives in :mod:`core.config` and is **passed in**, never imported
back.  That keeps this module stdlib-only and importing nothing project-local:
only ``math`` and ``collections.abc``.

Invariants (test-enforced by ``tests/test_scoring.py``):

* every ``*_fraction`` returns a value in ``[0.0, 1.0]``, clamped at **both**
  ends — a negative ``avg_drop`` yields ``1.0``, never a fraction above 1;
* ``mean_performance_drop`` returns ``None`` when no entry carries a numeric
  ``performance_drop`` (never ``0.0`` — "missing" and "zero damage" are
  different values);
* ``combine_fractions`` gives a ``None`` fraction zero weight with no
  redistribution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "performance_fraction",
    "calibration_fraction",
    "robustness_fraction",
    "confidence_fraction",
    "mean_performance_drop",
    "combine_fractions",
]


def _clamp01(value: float) -> float:
    """Clamp a fraction into ``[0.0, 1.0]``."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def performance_fraction(accuracy: float, f1: float) -> float:
    """Mean of accuracy and F1 as a 0-1 fraction.

    Args:
        accuracy: Classification accuracy in ``[0, 1]``.
        f1: Harmonic-mean F1 score in ``[0, 1]``.

    Returns:
        float: ``(accuracy + f1) / 2``, clamped to ``[0.0, 1.0]``.
    """
    return _clamp01((float(accuracy) + float(f1)) / 2.0)


def calibration_fraction(ece: float, *, ece_cap: float) -> float:
    """Calibration fraction: a perfect ``ECE == 0`` scores 1, ``ECE >= ece_cap``
    scores 0, linear in between.

    Args:
        ece: Expected calibration error (lower is better).
        ece_cap: The ECE value that scores zero (``ReliabilityWeights.ece_cap``).

    Returns:
        float: ``1 - ece / ece_cap``, clamped to ``[0.0, 1.0]``.
    """
    return _clamp01(1.0 - float(ece) / float(ece_cap))


def robustness_fraction(avg_drop: float, *, drop_cap: float) -> float:
    """Robustness fraction: a zero drop scores 1, a drop at or beyond
    ``drop_cap`` scores 0, linear in between.

    Args:
        avg_drop: Mean ``performance_drop`` fraction across stress types.  A
            **negative** value (an adversarial or noisy measurement that
            looks like an improvement) is clamped to a perfect score — it must
            never inflate the fraction above 1 (run-015 C1).
        drop_cap: The drop that scores zero (``ReliabilityWeights.drop_cap``).

    Returns:
        float: ``1 - avg_drop / drop_cap``, clamped to ``[0.0, 1.0]``.
    """
    return _clamp01(1.0 - float(avg_drop) / float(drop_cap))


def confidence_fraction(
    avg_entropy: float,
    hce_rate: float,
    *,
    n_classes: int,
) -> float:
    """Confidence-quality fraction (0-1), splitting the cap evenly.

    Half the points reward low normalised entropy, half reward a low
    high-confidence-error rate (ADR-005 contract: entropy is normalised by
    ``log2(max(n_classes, 2))`` bits so units match ``get_prediction_entropy``).

    Args:
        avg_entropy: Mean prediction entropy in bits.
        hce_rate: Fraction of high-confidence predictions that are wrong.
        n_classes: Number of output classes; the entropy normaliser is
            ``log2(max(n_classes, 2))`` so single-class degeneracy cannot
            divide by zero.

    Returns:
        float: The weighted quality fraction, clamped to ``[0.0, 1.0]``.
    """
    max_entropy = math.log2(max(int(n_classes), 2))
    if max_entropy <= 0.0:  # pragma: no cover - max(n_classes, 2) >= 2 => >= 1 bit
        normalised = 0.0
    else:
        normalised = _clamp01(float(avg_entropy) / max_entropy)
    return _clamp01(
        (1.0 - normalised) * 0.5 + (1.0 - _clamp01(float(hce_rate))) * 0.5
    )


def mean_performance_drop(per_model_stress: Mapping[str, Any]) -> float | None:
    """Mean ``performance_drop`` across one model's stress-type results.

    Only entries that carry a **numeric** ``performance_drop`` count.  An entry
    missing the key, or carrying a non-numeric value, contributes nothing —
    it is *not* treated as a zero drop.  When no entry contributes, the mean is
    ``None``: the evidence is missing, not "zero damage" (run-015 C1b).

    Args:
        per_model_stress: The ``stress_results[model_name]`` mapping, keyed by
            stress-type name, each value a dict possibly holding
            ``performance_drop``.

    Returns:
        float | None: The mean drop, or ``None`` when there is no numeric
        evidence.
    """
    drops: list[float] = []
    for entry in per_model_stress.values():
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("performance_drop")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        drops.append(float(value))
    if not drops:
        return None
    return sum(drops) / len(drops)


def combine_fractions(
    fractions: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float:
    """Weighted total of component fractions.

    A ``None`` fraction contributes zero with **no redistribution**: the weights
    are the policy's fixed denominators, so a missing component never makes the
    remaining ones worth more (ADR-018 monotonicity).

    Args:
        fractions: ``{component: fraction or None}``.
        weights: ``{component: weight}``; weights normally sum to ``1.0`` but
            the transform does not re-normalise them.

    Returns:
        float: ``sum(fraction * weight)`` over the supplied weights, with
        ``None`` fractions skipped.
    """
    total = 0.0
    for name, weight in weights.items():
        fraction = fractions.get(name)
        if fraction is None:
            continue
        total += float(fraction) * float(weight)
    return total