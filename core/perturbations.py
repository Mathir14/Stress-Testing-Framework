"""The single implementation of every stress perturbation operator.

Architecture reference: ``architecture.md`` §4.4, §4.4.1, §6.1; ADR-003,
ADR-010.3, ADR-010.6, ADR-010.7, ADR-011.1, ADR-011.6, ADR-011.7.

Why this module exists
----------------------
The pre-remediation operators in ``modules/stress_module.py`` carried twelve
duplicated DataFrame/ndarray branches, and the DataFrame branches wrote through
``DataFrame.values``.  ``.values`` returns a *view* only when the resulting 2-D
block dtype already equals the frame's storage dtype; whenever pandas has to
build an upcast temporary the write is discarded.  Measured on an
``int64 + float64`` frame — the realistic post-CSV-upload shape — **five of six
operators silently no-op**.  Vectorising over a ``float64`` ndarray and keeping
the container handling here in one place removes both the duplication and the
silent discard.

What is *not* promised
----------------------
Bit-fidelity with the old numbers is explicitly **not** promised (ADR-010.5):
a vectorised implementation and an injected ``Generator`` necessarily draw
randomness in a different order from the old per-column ``np.random.*`` loop.
Parity means (a) the same formula, (b) the same distributional parameters,
(c) seeded self-consistency across runs.  The only bit-exact freeze in this
milestone is reliability scoring, which is pure arithmetic.

Recorded naming debt (ADR-010.7 — deliberately **not** fixed)
------------------------------------------------------------
``feature_dropout`` masks individual *cells*, not features-per-sample as its
old docstring and the UI label claimed; and ``distribution_shift("variance")``
contracts values toward the mean rather than scaling variance.  Both names are
load-bearing for published results, so the docstrings below state the actual
behaviour while the arithmetic is preserved exactly.

Import rules (architecture.md §2.3): ``core.perturbations`` may import
``core.config``, ``core.errors`` and ``core.validation`` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from core.config import StressBounds
from core.errors import UnsupportedStressTypeError, ValidationError
from core.validation import (  # re-exported: see ADR-011.1 for why these live here
    CORRUPTION_TYPES,
    SHIFT_TYPES,
    ensure_finite,
    resolve_bounds,
    validate_stress_params,
)

__all__ = [
    "OPERATIONS",
    "OPERATION_LABELS",
    "CORRUPTION_TYPES",
    "SHIFT_TYPES",
    "PerturbationSpec",
    "OpFn",
    "available_operations",
    "operation_key",
    "apply_perturbation",
]

logger = logging.getLogger(__name__)

OpFn = Callable[[np.ndarray, Mapping[str, Any], np.random.Generator], np.ndarray]


def _column_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column ``(mean, std)`` for a 2-D float array.

    A zero-variance column yields ``std == 0`` (never ``NaN``), so every
    noise scale derived from it is ``0`` rather than ``NaN`` — the guard required
    by architecture.md §4.4.2.
    """
    means = values.mean(axis=0)
    if values.shape[0] > 1:
        stds = values.std(axis=0)
    else:
        stds = np.zeros(values.shape[1], dtype=np.float64)
    return means, stds


# ── Operators ────────────────────────────────────────────────────────────────


def _gaussian_noise(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """``x + N(0, (std * noise_level)^2)`` per column.

    The scale is the column's own standard deviation times ``noise_level``, so a
    zero-variance column receives zero-width noise.
    """
    level = float(params["noise_level"])
    _mean, stds = _column_stats(values)
    scale = stds * level
    # The draw is accumulated into its own buffer rather than written as
    # `values + draw * scale`, which would need three live frame-sized arrays
    # (draw, `draw * scale`, and the sum) on top of `values`.  Floating-point
    # addition is commutative, so `draw += values` is bit-identical to
    # `values + draw` — the difference is allocation, not arithmetic (ADR-016.2).
    result = rng.normal(0.0, 1.0, size=values.shape)
    result *= scale
    result += values
    return result


def _uniform_noise(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """``x + U(-r, r)`` per column, where ``r = (max - min) * noise_range``."""
    width = float(params["noise_range"])
    spreads = values.max(axis=0) - values.min(axis=0)
    half = spreads * width
    # Accumulated in place for the same reason as `_gaussian_noise`.
    result = rng.uniform(-1.0, 1.0, size=values.shape)
    result *= half
    result += values
    return result


def _feature_dropout(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Multiply by a per-**cell** Bernoulli mask ``rng.random(shape) > rate``.

    Naming debt (ADR-010.7): this masks individual cells, not features per
    sample, despite the operator's name and the UI label.
    """
    rate = float(params["dropout_rate"])
    mask = rng.random(values.shape) > rate
    result = np.empty_like(values)
    np.multiply(values, mask, out=result)
    return result


def _feature_corruption(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Replace ``int(n_rows * rate)`` sampled cells per column.

    Replacement depends on ``corruption_type``:

    * ``zero``    -> ``0.0``
    * ``mean``    -> the column mean
    * ``random``  -> ``U(min, max)`` of that column
    * ``extreme`` -> a random choice of the column's ``min`` or ``max``
    """
    rate = float(params["corruption_rate"])
    corruption_type = str(params["corruption_type"])
    result = values.copy()
    n_rows = values.shape[0]
    n_corrupt = int(n_rows * rate)

    if n_corrupt <= 0:
        logger.debug("corruption_rate=%s selects no cells; returning a copy", rate)
        return result

    for column in range(values.shape[1]):
        column_values = values[:, column]
        indices = rng.choice(n_rows, n_corrupt, replace=False)
        column_min = column_values.min()
        column_max = column_values.max()
        if corruption_type == "zero":
            result[indices, column] = 0.0
        elif corruption_type == "mean":
            result[indices, column] = column_values.mean()
        elif corruption_type == "random":
            result[indices, column] = rng.uniform(column_min, column_max, n_corrupt)
        else:  # "extreme" — validation has already excluded anything else
            result[indices, column] = rng.choice([column_min, column_max], n_corrupt)
    return result


def _scale_perturbation(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Multiply each column by ``factor`` or ``1 / factor``, drawn once per column.

    ``factor = scale_factor if rng.random() > 0.5 else 1 / scale_factor``.  The
    lower bound of ``scale_factor`` is exclusive and validation runs before any
    RNG draw, so ``1 / scale_factor`` is never evaluated at zero — replacing the
    pre-remediation non-deterministic ``ZeroDivisionError`` (architecture.md
    §4.4.7, §9 item 8).
    """
    factor = float(params["scale_factor"])
    inverse = 1.0 / factor
    # One draw per column, matching the pre-remediation loop granularity.
    picks = rng.random(values.shape[1]) > 0.5
    multipliers = np.where(picks, factor, inverse)
    result = np.empty_like(values)
    np.multiply(values, multipliers, out=result)
    return result


def _distribution_shift(
    values: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Shift each column's distribution.

    * ``mean``     -> ``x + std * shift_amount``
    * ``variance`` -> ``mean + (x - mean) * shift_amount``

    Naming debt (ADR-010.7): the ``variance`` branch contracts values toward
    the mean for ``shift_amount < 1``; it does not scale variance.
    """
    amount = float(params["shift_amount"])
    shift_type = str(params["shift_type"])
    means, stds = _column_stats(values)
    if shift_type == "mean":
        result = np.empty_like(values)
        np.add(values, stds * amount, out=result)
        return result
    # `values - means` is a full-size temporary; folding it into the result
    # buffer keeps only `values` and the result live (ADR-016.2).
    result = np.empty_like(values)
    np.subtract(values, means, out=result)
    result *= amount
    result += means
    return result


# ── Registry ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PerturbationSpec:
    """One registered operator.

    Attributes:
        name: Registry key — one of the six binding ``snake_case`` strings that
            ``StressTester.batch_stress_test`` already dispatches on.
        fn: The pure operator, receiving a 2-D ``float64`` ndarray, the
            normalised parameters and the injected generator.
        allowed_params: Every accepted parameter name.
        defaults: Defaults applied by validation when a parameter is omitted.
    """

    name: str
    fn: OpFn
    allowed_params: tuple[str, ...]
    defaults: Mapping[str, Any]


OPERATIONS: Mapping[str, PerturbationSpec] = {
    "gaussian_noise": PerturbationSpec(
        name="gaussian_noise",
        fn=_gaussian_noise,
        allowed_params=("noise_level",),
        defaults={"noise_level": 0.1},
    ),
    "uniform_noise": PerturbationSpec(
        name="uniform_noise",
        fn=_uniform_noise,
        allowed_params=("noise_range",),
        defaults={"noise_range": 0.1},
    ),
    "feature_dropout": PerturbationSpec(
        name="feature_dropout",
        fn=_feature_dropout,
        allowed_params=("dropout_rate",),
        defaults={"dropout_rate": 0.2},
    ),
    "feature_corruption": PerturbationSpec(
        name="feature_corruption",
        fn=_feature_corruption,
        allowed_params=("corruption_rate", "corruption_type"),
        defaults={"corruption_rate": 0.1, "corruption_type": "zero"},
    ),
    "scale_perturbation": PerturbationSpec(
        name="scale_perturbation",
        fn=_scale_perturbation,
        allowed_params=("scale_factor",),
        defaults={"scale_factor": 1.5},
    ),
    "distribution_shift": PerturbationSpec(
        name="distribution_shift",
        fn=_distribution_shift,
        allowed_params=("shift_amount", "shift_type"),
        defaults={"shift_amount": 0.5, "shift_type": "mean"},
    ),
}

#: The six human-readable UI labels and their registry keys.  This is the single
#: declared translation point between the two dispatch chains in the single
#: stress-test UI path and the ``snake_case`` registry (ADR-011.2).  The labels
#: are **not** permitted as ``OPERATIONS`` keys — adding them would break
#: ``batch_stress_test``, which dispatches on the registry keys (ADR-010.6).
OPERATION_LABELS: Mapping[str, str] = {
    "Gaussian Noise": "gaussian_noise",
    "Uniform Noise": "uniform_noise",
    "Feature Dropout": "feature_dropout",
    "Feature Corruption": "feature_corruption",
    "Scale Perturbation": "scale_perturbation",
    "Distribution Shift": "distribution_shift",
}

# The registry and the label map must agree in both directions.
#
# This is an explicit ``raise`` rather than an ``assert`` on purpose: ``assert`` is
# stripped by ``python -O``, so an assert here would let the invariant lapse
# silently in exactly the optimised runs most likely to be automated, leaving
# ``batch_stress_test`` dispatching on a key the registry does not hold
# (ADR-010.6). Conventions §3 also prefers a typed failure to a bare assert.
if tuple(OPERATIONS) != tuple(OPERATION_LABELS.values()):
    raise RuntimeError(
        "core.perturbations is misconfigured: OPERATIONS keys "
        f"{tuple(OPERATIONS)} and OPERATION_LABELS values "
        f"{tuple(OPERATION_LABELS.values())} must be the same six strings, in "
        "the same order (ADR-011.2)."
    )


def available_operations() -> tuple[str, ...]:
    """Return the registry keys in a stable order.

    Returns:
        tuple: The six operator keys.
    """
    return tuple(OPERATIONS)


def operation_key(label: str) -> str:
    """Translate a UI label into its registry key.

    Args:
        label: Human-readable label, e.g. ``"Gaussian Noise"``.

    Returns:
        str: The registry key, e.g. ``"gaussian_noise"``.

    Raises:
        UnsupportedStressTypeError: If ``label`` is not one of the six labels.

    Examples:
        >>> operation_key("Gaussian Noise")
        'gaussian_noise'
    """
    try:
        return OPERATION_LABELS[label]
    except (KeyError, TypeError) as exc:
        raise UnsupportedStressTypeError(
            f"Unknown stress test label {label!r}. Accepted labels are: "
            f"{', '.join(OPERATION_LABELS)}.",
            field="label",
            value=label,
            context={"label": label, "accepted": list(OPERATION_LABELS)},
        ) from exc


# ── Entry point ──────────────────────────────────────────────────────────────


def _to_float_array(X: Any) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Coerce the input to a 2-D ``float64`` array, preserving frame metadata.

    Returns:
        tuple: ``(values, template)`` where ``template`` is the original frame
        (used to restore index/columns) or ``None`` for ndarray input.

    Raises:
        ValidationError: If the input is not a supported container, a frame
            repeats a column label (``exc.context["duplicate_columns"]``), or a
            frame column cannot be represented as ``float64`` (the offending
            names are in ``exc.context["offending_columns"]``).
    """
    if isinstance(X, pd.DataFrame):
        from core.validation import coerce_numeric_array  # noqa: PLC0415

        return coerce_numeric_array(X), X

    if isinstance(X, pd.Series):
        raise ValidationError(
            "apply_perturbation expects a 2-D DataFrame or ndarray, got a "
            "1-D Series. Reshape it first.",
            context={"received_type": "pandas.Series"},
        )

    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValidationError(
                f"apply_perturbation expects a 2-D array, got {X.ndim}-D. Reshape it first.",
                context={"ndim": int(X.ndim)},
            )
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Array input could not be represented as float64; it holds non-numeric values.",
                context={"dtype": str(X.dtype)},
            ) from exc
        if values is X:
            # `asarray` returned the caller's own buffer because no conversion was
            # needed.  A *view* shares that memory but has independent flags, so
            # clearing `writeable` below does not change what the caller may do
            # with its own array — see `_freeze_input` for why that matters.
            values = values.view()
        return values, None

    raise ValidationError(
        f"apply_perturbation expects a pandas DataFrame or a numpy ndarray, "
        f"got {type(X).__name__}.",
        context={"received_type": type(X).__name__},
    )


def _freeze_input(values: np.ndarray) -> np.ndarray:
    """Return ``values`` marked read-only, making operator purity structural.

    ADR-016.3.  The previous implementation defended purity with a full-size
    ``before = values.copy()`` and an ``array_equal`` comparison after the fact.
    That cost a second frame-sized buffer on every call — the largest single
    avoidable term in ``apply_perturbation``'s peak memory — and, per ADR-011.6,
    the check could not fail against the code it was written for.

    Clearing ``writeable`` moves the guarantee from *detection* to *structure*: an
    operator that tries to write into its input raises ``ValueError`` from numpy
    instead of silently corrupting the caller's data, and no buffer is allocated
    to find out whether it did.

    The flag is set on ``values`` directly, which is safe because the caller of
    this function receives a fresh array (the DataFrame path allocates, and the
    ndarray path takes a ``.view()``) rather than the caller's own object.  That
    is why ``_to_float_array`` returns a view rather than ``np.asarray``'s
    result unchanged.
    """
    values.flags.writeable = False
    return values


def apply_perturbation(
    X: Any,
    op: str,
    params: Mapping[str, Any] | None = None,
    *,
    rng: np.random.Generator | None = None,
    bounds: StressBounds | None = None,
) -> Any:
    """Apply one registered perturbation, purely.

    Args:
        X: ``DataFrame`` or 2-D ``ndarray`` of features.  Never mutated.
        op: Registry key, e.g. ``"feature_dropout"``.
        params: Raw parameters; normalised and bounds-checked first.  ``None``
            means "all defaults for ``op``".
        rng: Random generator.  ``None`` creates a fresh
            ``np.random.default_rng()``; pass an explicit seeded generator for
            reproducible runs.
        bounds: Explicit bounds, or ``None`` for ``get_config().stress_bounds``
            (single-source invariant, architecture.md §4.3).

    Returns:
        A new object of the same container type: a ``DataFrame`` with the
        original index, columns and column order, or an ``ndarray``.  Output is
        always ``float64`` — integer input is no longer truncated back to
        ``int`` (§4.4.1, ADR-010.3).

    Raises:
        UnsupportedStressTypeError: Unknown operator key.
        ParameterOutOfRangeError: Numeric parameter outside its range.
        ValidationError: Unknown parameter key, unknown enum value, a frame with
            a repeated column label or a non-numeric input column, or non-finite
            output.

    Examples:
        >>> import numpy as np
        >>> out = apply_perturbation(
        ...     np.array([[1, 2], [3, 4]]), "gaussian_noise",
        ...     {"noise_level": 0.1}, rng=np.random.default_rng(0),
        ... )
        >>> out.dtype
        dtype('float64')
    """
    if op not in OPERATIONS:
        raise UnsupportedStressTypeError(
            f"Unknown stress operator {op!r}. Accepted operators are: {', '.join(OPERATIONS)}.",
            field="op",
            value=op,
            context={"op": op, "accepted": list(OPERATIONS)},
        )

    # Validation precedes every RNG draw, which is what makes
    # `scale_factor=0` a deterministic ParameterOutOfRangeError instead of the
    # old ~50/50 ZeroDivisionError (architecture.md §4.4.7).
    normalised = validate_stress_params(op, params or {}, resolve_bounds(bounds))

    values, template = _to_float_array(X)
    if values.shape[1] == 0:
        raise ValidationError(
            f"Cannot apply {op!r} to a feature frame with no columns.",
            context={"op": op, "shape": list(values.shape)},
        )

    generator = rng if rng is not None else np.random.default_rng()

    result = np.asarray(
        OPERATIONS[op].fn(_freeze_input(values), normalised, generator), dtype=np.float64
    )

    ensure_finite(result, context=f"operator {op!r}")

    if template is None:
        return result
    return pd.DataFrame(
        result,
        index=template.index.copy(),
        columns=template.columns.copy(),
        dtype=np.float64,
    )
