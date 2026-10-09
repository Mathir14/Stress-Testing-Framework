"""Parameter, numeric and frame validation for the domain kernel.

Architecture reference: ``architecture.md`` §4.3, §2.3, ADR-011.1.

This module owns the **canonical** stress parameter tables:

* the six numeric parameters' accepted ranges (derived from
  :class:`core.config.StressBounds` at call time, never a second hard-coded
  copy — the single-source invariant), and
* the two enum tuples ``CORRUPTION_TYPES`` / ``SHIFT_TYPES``.

They live here rather than in ``core.perturbations`` for a structural reason:
``validate_stress_params`` must reject an unknown ``corruption_type`` while
``apply_perturbation`` must call ``validate_stress_params``.  If the enums were
owned by ``perturbations``, validation would import them from ``perturbations``
while ``perturbations`` imports validation — an intra-``core/`` cycle, which
architecture.md §2.1 declares a test failure.  ``core.perturbations``
re-exports them as a one-way import so the documented public path
``core.perturbations.CORRUPTION_TYPES`` keeps working.

Import rules (architecture.md §2.3): ``core.validation`` may import
``core.config`` and ``core.errors`` only.
"""

from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.config import StressBounds, get_config
from core.errors import (
    ParameterOutOfRangeError,
    UnsupportedStressTypeError,
    ValidationError,
)

__all__ = [
    "CORRUPTION_TYPES",
    "SHIFT_TYPES",
    "NUMERIC_PARAMETERS",
    "ENUM_PARAMETERS",
    "resolve_bounds",
    "numeric_parameter_table",
    "validate_stress_params",
    "validate_frame",
    "ensure_finite",
    "coerce_numeric_array",
    "coerce_numeric_frame",
]

#: Accepted ``corruption_type`` values — verified to equal the UI option list at
#: the single-stress-test tab exactly (architecture.md §4.3).
CORRUPTION_TYPES: tuple[str, ...] = ("zero", "mean", "random", "extreme")

#: Accepted ``shift_type`` values — verified equal to the UI option list.
SHIFT_TYPES: tuple[str, ...] = ("mean", "variance")

#: Registry key -> numeric parameter name.  Keys are the six binding
#: ``snake_case`` strings the batch dispatcher already uses (ADR-010.6).
NUMERIC_PARAMETERS: Mapping[str, str] = {
    "gaussian_noise": "noise_level",
    "uniform_noise": "noise_range",
    "feature_dropout": "dropout_rate",
    "feature_corruption": "corruption_rate",
    "scale_perturbation": "scale_factor",
    "distribution_shift": "shift_amount",
}

#: Registry key -> (enum parameter name, accepted members).  Keys absent here
#: have no enum parameter.
ENUM_PARAMETERS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "feature_corruption": ("corruption_type", CORRUPTION_TYPES),
    "distribution_shift": ("shift_type", SHIFT_TYPES),
}

#: ``StressBounds`` attribute per numeric parameter.
_BOUNDS_ATTRIBUTE: Mapping[str, str] = {
    "noise_level": "noise_level",
    "noise_range": "noise_range",
    "dropout_rate": "dropout_rate",
    "corruption_rate": "corruption_rate",
    "scale_factor": "scale_factor",
    "shift_amount": "shift_amount",
}

#: Parameters whose upper / lower bound is exclusive.  ``dropout_rate=1.0``
#: would zero every cell; ``scale_factor=0`` is the division-by-zero path.
_EXCLUSIVE_LOW: frozenset[str] = frozenset({"scale_factor"})
_EXCLUSIVE_HIGH: frozenset[str] = frozenset({"dropout_rate", "corruption_rate"})

#: Default for each numeric parameter, applied when a caller omits it.  Mirrors
#: the pre-remediation keyword defaults in ``modules/stress_module.py``.
_NUMERIC_DEFAULTS: Mapping[str, float] = {
    "noise_level": 0.1,
    "noise_range": 0.1,
    "dropout_rate": 0.2,
    "corruption_rate": 0.1,
    "scale_factor": 1.5,
    "shift_amount": 0.5,
}

_ENUM_DEFAULTS: Mapping[tuple[str, str], str] = {
    ("feature_corruption", "corruption_type"): "zero",
    ("distribution_shift", "shift_type"): "mean",
}


def resolve_bounds(bounds: StressBounds | None = None) -> StressBounds:
    """Return the bounds to enforce.

    Single-source invariant (architecture.md §4.3): both
    :func:`validate_stress_params` and
    :func:`core.perturbations.apply_perturbation` route ``None`` through this
    function, so validation and execution can never disagree about the accepted
    range.

    Args:
        bounds: Explicit bounds, or ``None`` to use ``get_config().stress_bounds``.

    Returns:
        StressBounds: The bounds that will be enforced.
    """
    if bounds is None:
        return get_config().stress_bounds
    return bounds


def numeric_parameter_table(
    bounds: StressBounds | None = None,
) -> Mapping[str, tuple[tuple[float, float], bool, bool]]:
    """Build the numeric parameter table from ``bounds``.

    The table is derived at call time rather than hard-coded, so it can never
    drift from :class:`core.config.StressBounds`.

    Args:
        bounds: Explicit bounds, or ``None`` to use the configured defaults.

    Returns:
        Mapping: ``parameter -> ((low, high), exclusive_low, exclusive_high)``.
    """
    active = resolve_bounds(bounds)
    table: dict[str, tuple[tuple[float, float], bool, bool]] = {}
    for parameter, attribute in _BOUNDS_ATTRIBUTE.items():
        low, high = getattr(active, attribute)
        table[parameter] = (
            (float(low), float(high)),
            parameter in _EXCLUSIVE_LOW,
            parameter in _EXCLUSIVE_HIGH,
        )
    return table


def accepted_keys(op: str) -> tuple[str, ...]:
    """Return every accepted parameter name for ``op``.

    Args:
        op: Registry key of the operator.

    Returns:
        tuple: Numeric parameter name first, then the enum name when the
        operator has one.

    Raises:
        UnsupportedStressTypeError: If ``op`` is not a registry key.
    """
    if op not in NUMERIC_PARAMETERS:
        raise UnsupportedStressTypeError(
            f"Unknown stress operator {op!r}. Accepted operators are: "
            f"{', '.join(sorted(NUMERIC_PARAMETERS))}.",
            field="op",
            value=op,
            context={"op": op, "accepted": sorted(NUMERIC_PARAMETERS)},
        )
    keys = [NUMERIC_PARAMETERS[op]]
    if op in ENUM_PARAMETERS:
        keys.append(ENUM_PARAMETERS[op][0])
    return tuple(keys)


def _coerce_number(parameter: str, value: Any, op: str) -> float:
    """Coerce a numeric stress parameter, rejecting bools and non-numbers.

    ``bool`` is rejected explicitly even though it is an ``int`` subclass:
    ``True`` would otherwise become ``1.0``, i.e. maximum noise severity from a
    value that is not a severity at all.  No UI path produces a bool — Streamlit
    sliders return ``float`` — so this closes a hole rather than changing
    reachable behaviour.
    """
    if isinstance(value, bool):
        raise ValidationError(
            f"Parameter {parameter!r} of operator {op!r} must be a number, "
            f"got {type(value).__name__}.",
            field=parameter,
            value=repr(value),
            context={"op": op, "parameter": parameter},
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Parameter {parameter!r} of operator {op!r} must be a number, "
                f"got {type(value).__name__}.",
                field=parameter,
                value=repr(value),
                context={"op": op, "parameter": parameter},
            ) from exc
    number = float(value)
    if not np.isfinite(number):
        raise ParameterOutOfRangeError(
            f"Parameter {parameter!r} of operator {op!r} must be finite, got {value!r}.",
            field=parameter,
            value=value,
            context={"op": op, "parameter": parameter},
        )
    return number


def validate_stress_params(
    op: str,
    params: Mapping[str, Any],
    bounds: StressBounds | None = None,
) -> dict[str, Any]:
    """Normalise and bounds-check the parameters for one operator.

    Args:
        op: Registry key, e.g. ``"gaussian_noise"``.
        params: Raw parameter mapping.  Missing numeric and enum parameters are
            filled from the pre-remediation keyword defaults.
        bounds: Explicit bounds, or ``None`` for ``get_config().stress_bounds``.

    Returns:
        dict: A new normalised mapping containing exactly the operator's
        accepted keys.

    Raises:
        UnsupportedStressTypeError: Unknown ``op``, or an enum value that is not
            a member of its accepted set.
        ParameterOutOfRangeError: A numeric value outside its range, including
            the exclusive bounds ``scale_factor > 0``, ``dropout_rate < 1`` and
            ``corruption_rate < 1``.
        ValidationError: An unrecognised parameter key; the message names the
            operator and its accepted keys.
    """
    if params is None:
        params = {}
    if not isinstance(params, _MappingABC):
        raise ValidationError(
            f"Parameters for operator {op!r} must be a mapping, got {type(params).__name__}.",
            context={"op": op},
        )

    table = numeric_parameter_table(bounds)
    permitted = accepted_keys(op)
    normalised: dict[str, Any] = {}

    unknown = sorted(set(params) - set(permitted))
    if unknown:
        raise ValidationError(
            f"Operator {op!r} received unrecognised parameter(s) "
            f"{', '.join(repr(k) for k in unknown)}. Accepted parameters for "
            f"{op!r} are: {', '.join(permitted)}.",
            field=unknown[0],
            context={
                "op": op,
                "unknown": unknown,
                "accepted": list(permitted),
            },
        )

    numeric = NUMERIC_PARAMETERS[op]
    raw_numeric = params.get(numeric, _NUMERIC_DEFAULTS[numeric])
    number = _coerce_number(numeric, raw_numeric, op)
    (low, high), exclusive_low, exclusive_high = table[numeric]

    if number < low or (exclusive_low and number <= low):
        bound_text = f"greater than {low}" if exclusive_low else f"at least {low}"
        raise ParameterOutOfRangeError(
            f"Parameter {numeric!r} of operator {op!r} must be {bound_text}; got {number}.",
            field=numeric,
            value=number,
            context={
                "op": op,
                "parameter": numeric,
                "accepted_range": [low, high],
                "exclusive_low": exclusive_low,
                "exclusive_high": exclusive_high,
            },
        )
    if number > high or (exclusive_high and number >= high):
        bound_text = f"less than {high}" if exclusive_high else f"at most {high}"
        raise ParameterOutOfRangeError(
            f"Parameter {numeric!r} of operator {op!r} must be {bound_text}; got {number}.",
            field=numeric,
            value=number,
            context={
                "op": op,
                "parameter": numeric,
                "accepted_range": [low, high],
                "exclusive_low": exclusive_low,
                "exclusive_high": exclusive_high,
            },
        )
    normalised[numeric] = number

    if op in ENUM_PARAMETERS:
        enum_name, enum_values = ENUM_PARAMETERS[op]
        raw_enum = params.get(enum_name, _ENUM_DEFAULTS[(op, enum_name)])
        if not isinstance(raw_enum, str) or raw_enum not in enum_values:
            raise UnsupportedStressTypeError(
                f"Unsupported {enum_name}={raw_enum!r} for operator {op!r}. "
                f"Accepted values are: {', '.join(enum_values)}.",
                field=enum_name,
                value=raw_enum,
                context={
                    "op": op,
                    "parameter": enum_name,
                    "accepted": list(enum_values),
                },
            )
        normalised[enum_name] = raw_enum

    return normalised


def validate_frame(df: pd.DataFrame, *, min_rows: int = 1) -> None:
    """Assert that a DataFrame is usable as a feature matrix.

    Args:
        df: Frame to check.
        min_rows: Minimum acceptable row count.

    Returns:
        None

    Raises:
        ValidationError: If ``df`` is not a DataFrame, has no columns, repeats a
            column label (``duplicate_columns`` in ``exc.context``), or has fewer
            than ``min_rows`` rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"Expected a pandas DataFrame, got {type(df).__name__}.",
            context={"received_type": type(df).__name__},
        )
    if df.shape[1] == 0:
        raise ValidationError(
            "Feature frame has no columns.",
            context={"shape": list(df.shape)},
        )
    # A repeated label makes ``df[label]`` return a *frame* rather than a
    # ``Series``, so every column-addressing consumer below silently builds a
    # 2-D block and the reconstruction in ``coerce_numeric_frame`` then fails
    # with a bare ``builtins.ValueError`` about shapes.  That is a user-fixable
    # data defect, so it is rejected here, as the one place the "usable as a
    # feature matrix" contract is stated (conventions §3).
    duplicate_labels = df.columns[df.columns.duplicated()].tolist()
    if duplicate_labels:
        raise ValidationError(
            f"Feature frame repeats {len(duplicate_labels)} column name(s): "
            f"{', '.join(str(label) for label in duplicate_labels)}. Rename or "
            "drop the duplicates before stress testing.",
            context={
                "duplicate_columns": [str(label) for label in duplicate_labels],
                "column_count": int(df.shape[1]),
                "unique_column_count": int(df.columns.nunique()),
            },
        )
    if df.shape[0] < min_rows:
        raise ValidationError(
            f"Feature frame needs at least {min_rows} row(s); got {df.shape[0]}.",
            context={"rows": int(df.shape[0]), "min_rows": min_rows},
        )


def ensure_finite(values: np.ndarray, *, context: str) -> np.ndarray:
    """Assert that an array contains no NaN or infinity.

    Args:
        values: Array to check.
        context: Short description used in the error message, e.g. the operator
            name.

    Returns:
        np.ndarray: ``values``, unchanged, so the call can be chained.

    Raises:
        ValidationError: If any element is NaN or infinite.

    Notes:
        The finiteness probe allocates **one** frame-sized boolean array, not
        two.  The obvious spelling, ``bad = ~np.isfinite(array)``, builds the
        ``isfinite`` mask and then a second full-size array to invert it, and the
        two are live simultaneously — measured at 0.249x the operator's own
        frame on a 200k x 51 input, which is larger than every remaining
        avoidable term in ``apply_perturbation`` combined (ADR-016.2).  The
        inverted mask is therefore built only on the error path, where the call
        is about to raise and its peak memory no longer matters.  ``.all()``
        replaces ``.any()`` over the complement so the fast path needs no
        inversion at all.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        return array
    if np.isfinite(array).all():
        return array
    # Error path only: from here the call raises, so allocating the inverted
    # mask costs nothing that the caller was trying to protect against.
    bad = ~np.isfinite(array)
    if bad.any():
        rows = int(np.count_nonzero(np.any(bad, axis=1))) if array.ndim > 1 else int(bad.sum())
        raise ValidationError(
            f"{context} produced {int(bad.sum())} non-finite value(s) across "
            f"{rows} row(s). Check for zero-variance columns or NaN inputs.",
            context={
                "stage": context,
                "non_finite_count": int(bad.sum()),
                "affected_rows": rows,
            },
        )
    return array


def _non_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Name every column that cannot be represented as ``float64``.

    This is a **dtype probe**, not a conversion.  The pre-refactor
    implementation converted every numeric column as it went and kept the result,
    which held a second full-size buffer for the whole loop (ADR-016.2).  Here
    nothing is allocated for a column that is already numeric — the bulk
    conversion happens exactly once, in :func:`coerce_numeric_array`.

    Only a *non*-numeric dtype is probed, and the probe's result is discarded
    immediately, so an object column holding ``Decimal`` or numeric strings is
    accepted here and converted by the caller in the same single pass.
    """
    offending: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            offending.append(str(column))
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        try:
            series.to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            offending.append(str(column))
    return offending


def coerce_numeric_array(df: pd.DataFrame) -> np.ndarray:
    """Coerce a feature frame to a single 2-D ``float64`` ndarray.

    Architecture reference: §4.4.1, ADR-010.3, ADR-016.2.  The pre-remediation
    operators wrote through ``DataFrame.values``, which silently discarded
    results whenever pandas had to build an upcast temporary (an
    ``int64 + float64`` frame is the realistic post-CSV-upload shape and 5 of 6
    operators no-op on it).  Widening to ``float64`` up front is the only way
    the operators are meaningful on integer data, and an un-coercible column
    must be named rather than silently no-op.

    The conversion is deliberately **one pass and one buffer**, and it is
    explicitly **C-contiguous**.  Both properties are load-bearing:

    * *One buffer.*  The pre-refactor code converted every column into a dict and
      then ``column_stack``-ed the dict, holding two frame-sized arrays at once
      (measured 2.00x).  Here a single preallocated array is filled column by
      column, so the only transient is one column.
    * *C-contiguous.*  ``DataFrame.to_numpy(dtype=float64)`` returns an
      **F-contiguous** array on the pinned pandas, with or without ``copy=False``
      — pandas stores its blocks transposed, so no layout-preserving view is
      available.  That is not a cosmetic difference: ``values.mean(axis=0)`` and
      ``values.std(axis=0)`` sum in a different order on an F-contiguous array,
      which moves the resulting noise scale by up to ~2e-15 and every perturbed
      value derived from it.  Since every noise, dropout and shift operator is
      scaled by a column statistic, *the layout of the coercion buffer changes
      every operator's output*.  Filling a C-contiguous buffer is what keeps the
      refactor bit-identical, and
      ``tests/test_perturbations.py::test_coercion_is_c_contiguous_so_reductions_are_bit_stable``
      fails if that is ever undone.

    Args:
        df: Frame whose columns must all be numeric and uniquely named.

    Returns:
        np.ndarray: A C-contiguous ``float64`` array of shape
        ``(len(df), df.shape[1])``.  It is **writeable**; callers that require
        structural purity — see :func:`core.perturbations.apply_perturbation` —
        clear the flag themselves, because a read-only return would be a
        surprising contract for a function whose name promises a value.

    Raises:
        ValidationError: If ``df`` is not a usable feature matrix (no columns, a
            repeated column label — named in ``exc.context["duplicate_columns"]``
            — or too few rows), or if any column cannot be represented as
            ``float64``; the latter are named in
            ``exc.context["offending_columns"]``.
    """
    validate_frame(df)
    offending = _non_numeric_columns(df)
    if offending:
        raise ValidationError(
            f"Feature frame has {len(offending)} non-numeric column(s) that "
            f"cannot be perturbed: {', '.join(offending)}. Encode categorical "
            "features before stress testing.",
            context={"offending_columns": offending},
        )
    out = np.empty((df.shape[0], df.shape[1]), dtype=np.float64)
    for position, column in enumerate(df.columns):
        out[:, position] = df[column].to_numpy(dtype=np.float64, copy=False)
    return out


def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a feature frame to a ``float64`` frame, naming any offending columns.

    A thin wrapper over :func:`coerce_numeric_array` for callers that want a
    frame back rather than an array.  The array is the primitive: this function
    exists so that a second, frame-shaped coercion path cannot grow its own
    idea of what is coercible (ADR-016.2).

    Args:
        df: Frame whose columns must all be numeric and uniquely named.

    Returns:
        pd.DataFrame: A new ``float64`` frame with the original index, columns
        and column order preserved.

    Raises:
        ValidationError: Propagated verbatim from :func:`coerce_numeric_array`.
    """
    data = coerce_numeric_array(df)
    return pd.DataFrame(
        data,
        index=df.index.copy(),
        columns=df.columns.copy(),
        dtype=np.float64,
    )
