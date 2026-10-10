"""Perturbation-kernel tests (architecture.md §4.4, §4.4.1, §6.1, §10.2).

**Tier T1** — requires numpy/pandas.

The headline regression is the mixed-dtype frame.  The pre-remediation operators
wrote through ``DataFrame.values``, which returns a *view* only when the 2-D
block dtype already equals the frame's storage dtype.  Measured on an
``int64 + float64`` frame — the realistic shape straight out of
``pd.read_csv`` — **five of six operators silently no-op'd**: the arithmetic ran
on a discarded upcast temporary and the returned frame was byte-identical to the
input.  ``feature_dropout`` was the lone exception, because multiplying by a
``float`` mask happens to keep the block dtype ``int64``.

So the test that matters most here is
:func:`test_mixed_dtype_frame_changes_under_every_operator`: for every operator's
nonzero parameters the output is never byte-identical to the input, which is the
invariant the five silent no-ops violated.

Bit-fidelity with the old numbers is explicitly **not** asserted (ADR-010.5): a
vectorised implementation draws randomness in a different order, so parity means
the same formula, the same distributional parameters and seeded self-consistency.

There *is* one bit-exact freeze, and it was added later and is a different claim:
:mod:`tests.test_reliability_parity` freezes reliability scoring, and
:func:`test_perturbation_output_is_bit_identical` freezes the **post-Wave-2**
perturbation kernel itself (ADR-016.7).  The latter is not a parity claim about
the old implementation — it is a claim about the *current* one, so that a
refactor asserting "behaviour-preserving" can be proven to be exactly that.  It
is what makes the C-contiguity contract in
:func:`test_coercion_is_c_contiguous_so_reductions_are_bit_stable` load-bearing
rather than advisory.

The last two sections gate the two ADR-016 claims that the bit-identity freeze
cannot reach on its own, because both are properties of *how* the kernel spends
memory rather than of *what* it returns:

* **Structural purity (ADR-016.5).**  Purity is enforced rather than detected —
  :func:`test_the_operator_is_handed_a_read_only_buffer`,
  :func:`test_a_read_only_buffer_is_what_rejects_an_in_place_operator` and
  :func:`test_the_callers_own_array_keeps_its_writeable_flag`.  These replaced a
  full-size ``values.copy()`` that existed only to be compared afterwards, so
  they are what makes the 2.13x peak reachable at all.
* **Bounded memory (ADR-016.2/.3/.6).**
  :func:`test_apply_perturbation_peak_allocation_stays_bounded` and
  :func:`test_coercion_peak_allocation_stays_bounded` measure the peak traced
  allocation of the two call frames, and
  :func:`test_the_entry_point_peak_decomposes_into_the_documented_buffers` pins
  the bound to named terms so a breach is attributable.  A bit-identical output
  is fully compatible with having silently gone back to a 4.00x kernel, which is
  why the freeze above would not have caught that regression on its own.
"""

from __future__ import annotations

import copy
import tracemalloc
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="perturbation tests need the scientific stack")
pytest.importorskip("pandas", reason="perturbation tests need the scientific stack")

import numpy as np  # noqa: E402 - after importorskip by design
import pandas as pd  # noqa: E402 - after importorskip by design

from core.config import StressBounds
from core.errors import (
    ParameterOutOfRangeError,
    UnsupportedStressTypeError,
    ValidationError,
)
from core.perturbations import (
    CORRUPTION_TYPES,
    OPERATION_LABELS,
    OPERATIONS,
    SHIFT_TYPES,
    apply_perturbation,
    available_operations,
    operation_key,
)
from core.validation import coerce_numeric_array

SEED = 20260904

#: Non-default parameters per operator: strong enough that "unchanged" cannot be
#: coincidence.  Read by every behavioural test.
PARAMS: dict[str, dict[str, object]] = {
    "gaussian_noise": {"noise_level": 0.5},
    "uniform_noise": {"noise_range": 0.5},
    "feature_dropout": {"dropout_rate": 0.5},
    "feature_corruption": {"corruption_rate": 0.4, "corruption_type": "zero"},
    "scale_perturbation": {"scale_factor": 3.0},
    "distribution_shift": {"shift_amount": 1.5, "shift_type": "mean"},
}

ALL_OPS = tuple(OPERATIONS)


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """The regression frame: ``int64`` alongside ``float64``, no NaNs."""
    return pd.DataFrame(
        {
            "int_col": np.arange(30, dtype="int64"),
            "float_col": np.linspace(0.1, 3.0, 30, dtype="float64"),
            "third_col": np.arange(30, 60, dtype="int64"),
        }
    )


@pytest.fixture
def float_frame() -> pd.DataFrame:
    """An all-``float64`` frame — the shape the old operators did work on."""
    rng = _rng(1)
    return pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])


#: Column count for :func:`wide_frame`.  Chosen so that a per-column coin flip
#: cannot land the same way twice in practice: ``scale_perturbation`` draws
#: exactly one value per column and applies either ``factor`` or ``1 / factor``,
#: so on the three-column ``float_frame`` two independent runs produce identical
#: output one time in four.  That is a property of the operator, not a defect —
#: but it makes "two runs differ" an unsuitable assertion at that width.  At 64
#: columns the chance of agreement is ``2 ** -63``.
WIDE_NCOLS = 64


@pytest.fixture
def wide_frame() -> pd.DataFrame:
    """A frame wide enough for "two runs of a coin-flip operator differ" to hold.

    Use this, not :func:`float_frame`, for every assertion of the form *the two
    results are not equal*.  :func:`float_frame` is right for everything else.
    """
    rng = _rng(1)
    return pd.DataFrame(
        rng.normal(size=(40, WIDE_NCOLS)),
        columns=[f"c{i}" for i in range(WIDE_NCOLS)],
    )


# ── §6.1 the regression ───────────────────────────────────────────────────────


def test_the_regression_frame_is_actually_mixed(mixed_frame: pd.DataFrame):
    """Guard: if this frame ever becomes homogeneous the regression is void."""
    dtypes = {str(dtype) for dtype in mixed_frame.dtypes}
    assert dtypes == {"int64", "float64"}


@pytest.mark.parametrize("op", ALL_OPS)
def test_mixed_dtype_frame_changes_under_every_operator(op: str, mixed_frame: pd.DataFrame):
    """§6.1 / ADR-010.3: the five silent no-ops are gone.

    This is the single most important test in the file.  Pre-remediation, five of
    these six assertions failed on exactly this frame.
    """
    out = apply_perturbation(mixed_frame, op, PARAMS[op], rng=_rng())
    before = mixed_frame.to_numpy(dtype=np.float64)
    after = out.to_numpy(dtype=np.float64)
    assert out.shape == mixed_frame.shape
    assert not np.array_equal(before, after), f"{op!r} silently no-op'd on an int64 + float64 frame"


def test_the_regression_needed_more_than_one_int_column(mixed_frame: pd.DataFrame):
    """Why two ``int64`` columns: the old failure needed an *upcast* block.

    ``DataFrame.values`` returns a view only when the 2-D block dtype already
    matches.  ``int64 + float64`` forces a temporary, which is what the write was
    discarded into.  A frame of only ``int64`` columns happened to keep the view
    and worked — so a single-dtype ``int64`` frame would *not* have caught this.
    """
    only_ints = mixed_frame[["int_col", "third_col"]]
    assert all(str(dtype) == "int64" for dtype in only_ints.dtypes)
    # It must still change — the point is that it always did, so the mixed frame
    # is the discriminating case, not this one.
    out = apply_perturbation(only_ints, "gaussian_noise", PARAMS["gaussian_noise"], rng=_rng())
    assert not np.array_equal(out.to_numpy(dtype=np.float64), only_ints.to_numpy(dtype=np.float64))


@pytest.mark.parametrize("op", ALL_OPS)
def test_int64_input_comes_back_float64(op: str):
    """§4.4.1: integer input is widened to ``float64``, never cast back to ``int``.

    This is the dtype half of the contract and it holds for all six operators.
    """
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype="int64")
    out = apply_perturbation(values, op, PARAMS[op], rng=_rng())
    assert out.dtype == np.float64


#: Operators whose arithmetic can produce a fractional value at all.  The other
#: three cannot, by construction rather than by truncation:
#:
#: * ``feature_dropout`` multiplies by a 0/1 mask, so an integer cell stays an
#:   integer value;
#: * ``feature_corruption`` with ``corruption_type="zero"`` writes ``0.0``;
#: * ``scale_perturbation`` multiplies by ``factor`` or ``1/factor``, which is
#:   fractional only when the drawn multiplier is the reciprocal.
#:
#: Pinning *why* matters: a test that demanded fractions from all six would be
#: asserting something false, and a test that demanded none would miss the
#: truncation bug the old ``astype(int)`` caused.
FRACTIONAL_OPS = ("gaussian_noise", "uniform_noise", "distribution_shift")


@pytest.mark.parametrize("op", FRACTIONAL_OPS)
def test_fractional_results_are_not_truncated(op: str):
    """§4.4.1: the pre-remediation ``astype(int)`` destroyed exactly this.

    These three add a continuous offset, so an integer input must come back with
    a fractional part intact.
    """
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype="int64")
    out = apply_perturbation(values, op, PARAMS[op], rng=_rng())
    assert np.any(out != np.floor(out)), (
        f"{op!r} returned only whole numbers; fractional perturbation was truncated"
    )


def test_scale_perturbation_reciprocal_multiplier_is_fractional():
    """The reciprocal branch is where ``scale_perturbation`` yields fractions.

    Searched across seeds because which branch a column takes is a draw; the
    requirement is that *some* column can be fractional, not that a given seed
    is.
    """
    values = np.arange(1, 13, dtype="int64").reshape(6, 2)
    saw_fraction = False
    for seed in range(20):
        out = apply_perturbation(
            values, "scale_perturbation", {"scale_factor": 3.0}, rng=_rng(seed)
        )
        assert out.dtype == np.float64
        saw_fraction |= bool(np.any(out != np.floor(out)))
    assert saw_fraction, "scale_perturbation never produced a fractional value"


def test_feature_corruption_random_type_is_fractional():
    """``corruption_type="random"`` draws from ``U(min, max)`` — continuous."""
    values = np.arange(100, dtype="int64").reshape(20, 5)
    out = apply_perturbation(
        values,
        "feature_corruption",
        {"corruption_rate": 0.5, "corruption_type": "random"},
        rng=_rng(),
    )
    assert out.dtype == np.float64
    assert np.any(out != np.floor(out))


def test_dataframe_output_is_always_float64(mixed_frame: pd.DataFrame):
    """§4.4.1: the output dtype contract holds for a frame regardless of input."""
    for op in ALL_OPS:
        out = apply_perturbation(mixed_frame, op, PARAMS[op], rng=_rng())
        assert all(dtype == np.dtype("float64") for dtype in out.dtypes), op


# ── Purity ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("container", ["frame", "array"])
def test_input_is_never_mutated(op: str, container: str, mixed_frame: pd.DataFrame):
    """Purity is the kernel's core contract: it must not write through its input.

    Checked by byte-comparing a copy, not by identity — a mutation that returns
    the same object would pass a weaker check.
    """
    source = (
        mixed_frame.copy(deep=True)
        if container == "frame"
        else mixed_frame.to_numpy(dtype=np.float64)
    )
    before = (
        source.to_numpy(dtype=np.float64).tobytes() if container == "frame" else source.tobytes()
    )
    apply_perturbation(source, op, PARAMS[op], rng=_rng())
    after = (
        source.to_numpy(dtype=np.float64).tobytes() if container == "frame" else source.tobytes()
    )
    assert before == after, f"{op!r} mutated its {container} input"


@pytest.mark.parametrize("op", ALL_OPS)
def test_result_is_a_new_object_not_an_alias(op: str, float_frame: pd.DataFrame):
    """An in-place result would pass the byte-compare above and still be wrong."""
    out = apply_perturbation(float_frame, op, PARAMS[op], rng=_rng())
    assert out is not float_frame


@pytest.mark.parametrize("op", ALL_OPS)
def test_metadata_is_preserved_for_frames(op: str, float_frame: pd.DataFrame):
    """Index, columns and column order must survive, or X/y alignment breaks."""
    shuffled = float_frame.iloc[::-1]  # a non-trivial index
    out = apply_perturbation(shuffled, op, PARAMS[op], rng=_rng())
    assert out.index.equals(shuffled.index)
    assert list(out.columns) == list(shuffled.columns)


@pytest.mark.parametrize("op", ALL_OPS)
def test_ndarray_input_returns_ndarray(op: str):
    """Container type is preserved for arrays too."""
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = apply_perturbation(values, op, PARAMS[op], rng=_rng())
    assert isinstance(out, np.ndarray)
    assert out.shape == values.shape


# ── Reproducibility ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
def test_seeded_runs_are_identical(op: str, float_frame: pd.DataFrame):
    """Seeded self-consistency is the parity claim (ADR-010.5)."""
    first = apply_perturbation(float_frame, op, PARAMS[op], rng=_rng(11))
    second = apply_perturbation(float_frame, op, PARAMS[op], rng=_rng(11))
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


#: The five operators that draw randomness.  ``distribution_shift`` is
#: deterministic by construction — it adds ``std * amount`` or scales deviations,
#: with no random component — so it is excluded from the seed-sensitivity tests
#: and pinned separately by
#: :func:`test_distribution_shift_is_deterministic_by_construction`.
STOCHASTIC_OPS = tuple(op for op in ALL_OPS if op != "distribution_shift")


@pytest.mark.parametrize("op", STOCHASTIC_OPS)
def test_different_seeds_give_different_results(op: str, wide_frame: pd.DataFrame):
    """Otherwise the RNG parameter would be decorative."""
    first = apply_perturbation(wide_frame, op, PARAMS[op], rng=_rng(1))
    second = apply_perturbation(wide_frame, op, PARAMS[op], rng=_rng(2))
    assert not np.array_equal(first.to_numpy(), second.to_numpy())


def test_distribution_shift_is_deterministic_by_construction():
    """Pinned as a *fact*, not skipped silently.

    Both branches are pure arithmetic on the column statistics, so two seeds
    agree.  If this ever starts failing, a random draw has been added to the
    operator and the seeded-reproducibility claims above need re-checking.
    """
    frame = pd.DataFrame(np.linspace(1.0, 9.0, 30).reshape(10, 3))
    for shift_type in ("mean", "variance"):
        first = apply_perturbation(
            frame,
            "distribution_shift",
            {"shift_amount": 1.5, "shift_type": shift_type},
            rng=_rng(1),
        )
        second = apply_perturbation(
            frame,
            "distribution_shift",
            {"shift_amount": 1.5, "shift_type": shift_type},
            rng=_rng(2),
        )
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


@pytest.mark.parametrize("op", STOCHASTIC_OPS)
def test_injected_generator_state_advances(op: str, wide_frame: pd.DataFrame):
    """§4.4.4: the *injected* generator is the one consumed — exactly.

    This is the claim the neighbouring "results differ" tests can only support
    probabilistically, so it is asserted directly: a module-global
    ``np.random`` call would leave the injected bit generator untouched.  Unlike
    an output comparison this holds for every operator at every frame width.
    """
    rng = _rng(5)
    start = copy.deepcopy(rng.bit_generator.state)
    apply_perturbation(wide_frame, op, PARAMS[op], rng=rng)
    assert rng.bit_generator.state != start, f"{op!r} did not consume the injected generator"


@pytest.mark.parametrize("op", STOCHASTIC_OPS)
def test_explicit_rng_is_consumed(op: str, wide_frame: pd.DataFrame):
    """Two calls sharing one generator must not produce the same frame.

    The state-advance half of this claim is exact and lives in
    :func:`test_injected_generator_state_advances`; what remains here is that
    the draw actually reaches the data.  See :func:`wide_frame` for why this
    needs 64 columns.
    """
    rng = _rng(5)
    first = apply_perturbation(wide_frame, op, PARAMS[op], rng=rng)
    second = apply_perturbation(wide_frame, op, PARAMS[op], rng=rng)
    assert not np.array_equal(first.to_numpy(), second.to_numpy()), (
        f"{op!r} ignored the injected generator"
    )


@pytest.mark.parametrize("op", STOCHASTIC_OPS)
def test_global_numpy_random_state_is_not_touched(op: str, wide_frame: pd.DataFrame):
    """§4.4.5: seeding ``np.random.seed`` must not make runs reproducible."""
    np.random.seed(7)
    before = np.random.random()
    np.random.seed(7)
    first = apply_perturbation(wide_frame, op, PARAMS[op])
    mid = np.random.random()
    np.random.seed(7)
    second = apply_perturbation(wide_frame, op, PARAMS[op])
    after = np.random.random()

    assert before == mid == after, "the global RNG state was consumed"
    assert not np.array_equal(first.to_numpy(), second.to_numpy())


# ── §4.4.2 zero-variance guard ───────────────────────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
def test_zero_variance_column_yields_no_nan(op: str):
    """§4.4.2: a constant column gets zero-width noise, never ``0/0``."""
    frame = pd.DataFrame(
        {
            "constant": np.full(30, 7.0),
            "varying": np.linspace(0.0, 1.0, 30),
        }
    )
    out = apply_perturbation(frame, op, PARAMS[op], rng=_rng())
    assert np.isfinite(out.to_numpy()).all(), f"{op!r} produced NaN/inf"


@pytest.mark.parametrize("op", ALL_OPS)
def test_all_columns_constant_yields_no_nan(op: str):
    """The degenerate extreme of the same guard."""
    frame = pd.DataFrame(np.full((10, 3), 2.0), columns=list("abc"))
    out = apply_perturbation(frame, op, PARAMS[op], rng=_rng())
    assert np.isfinite(out.to_numpy()).all()


def test_gaussian_noise_on_a_constant_column_is_a_noop():
    """Zero std × any level = zero noise, which is the guard working."""
    frame = pd.DataFrame({"c": np.full(10, 5.0)})
    out = apply_perturbation(frame, "gaussian_noise", {"noise_level": 1.0}, rng=_rng())
    np.testing.assert_allclose(out.to_numpy(), 5.0)


# ── §4.4.7 scale_factor: the old non-deterministic ZeroDivisionError ─────────


def test_scale_factor_zero_is_rejected_for_every_seed():
    """Validation precedes every RNG draw, so this cannot be seed-dependent.

    The pre-remediation code drew the ``factor if rng.random() > 0.5`` coin flip
    *before* dividing, so ``scale_factor=0`` raised ``ZeroDivisionError`` roughly
    half the time and passed silently the other half.  Sweeping the seeds is what
    makes the fix falsifiable — a seed-hunt would pass both implementations.
    """
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3))
    for seed in range(40):
        with pytest.raises(ParameterOutOfRangeError):
            apply_perturbation(frame, "scale_perturbation", {"scale_factor": 0.0}, rng=_rng(seed))


def test_scale_factor_zero_never_produces_zero_division_error():
    """Explicitly: the old exception type must not appear at all."""
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3))
    for seed in range(40):
        try:
            apply_perturbation(frame, "scale_perturbation", {"scale_factor": 0.0}, rng=_rng(seed))
        except ParameterOutOfRangeError:
            continue
        except ZeroDivisionError:  # pragma: no cover - the regression
            pytest.fail(f"seed {seed} still raises ZeroDivisionError")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"seed {seed} raised unexpected {type(exc).__name__}: {exc}")
        else:  # pragma: no cover - the regression
            pytest.fail(f"seed {seed} silently accepted scale_factor=0")


def test_dropout_rate_one_is_rejected_deterministically():
    """The other zero-everything case, same reasoning."""
    frame = pd.DataFrame(np.ones((10, 3)))
    for seed in range(10):
        with pytest.raises(ParameterOutOfRangeError):
            apply_perturbation(frame, "feature_dropout", {"dropout_rate": 1.0}, rng=_rng(seed))


# ── Non-numeric input ────────────────────────────────────────────────────────


def test_str_bearing_frame_raises_naming_the_column():
    """ADR-010.4: a ``str`` frame must raise, not silently no-op.

    The architecture explicitly rejects this as a *regression frame* — a
    categorical column is a modelling bug to surface, not something to coerce.
    """
    frame = pd.DataFrame({"ok": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]})
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.3}, rng=_rng())
    assert "label" in excinfo.value.context["offending_columns"]
    assert "label" in str(excinfo.value), "the message must name the column"


def test_all_column_names_are_reported_not_just_the_first():
    """A user with five bad columns should learn about five."""
    frame = pd.DataFrame({"a": ["p", "q"], "b": ["r", "s"], "n": [1.0, 2.0]})
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(frame, "uniform_noise", {"noise_range": 0.3}, rng=_rng())
    assert set(excinfo.value.context["offending_columns"]) == {"a", "b"}


def test_numeric_strings_that_parse_are_accepted():
    """A column of parseable text *is* numeric to pandas; coercion is correct."""
    frame = pd.DataFrame({"a": ["1.0", "2.0", "3.0"]})
    out = apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())
    assert out["a"].dtype == np.float64


def test_series_input_is_rejected():
    """A 1-D Series is a shape bug, not a container to guess about."""
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(pd.Series([1.0, 2.0]), "gaussian_noise", {"noise_level": 0.1})
    assert excinfo.value.context["received_type"] == "pandas.Series"


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros((2, 3, 4)), [1, 2, 3], "frame", 5, None])
def test_unsupported_containers_are_rejected(bad):
    """Anything that is not a frame or a 2-D array is a typed error."""
    with pytest.raises(ValidationError):
        apply_perturbation(bad, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())


def test_one_dimensional_array_names_the_ndim():
    frame = np.zeros((2, 3, 4))
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.1})
    assert excinfo.value.context["ndim"] == 3


def test_non_numeric_ndarray_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(
            np.array([["a", "b"], ["c", "d"]]), "gaussian_noise", {"noise_level": 0.1}
        )
    assert excinfo.value.context["dtype"]


def test_zero_column_frame_is_rejected():
    """Perturbing nothing is a caller bug, not a silent identity."""
    with pytest.raises(ValidationError):
        apply_perturbation(pd.DataFrame(index=[0, 1, 2]), "gaussian_noise", {"noise_level": 0.1})


# ── Zero rows: the container-asymmetric raw ``ValueError`` regression ─────────


def test_zero_row_ndarray_and_frame_behave_identically():
    """Run-014 M7: an empty ndarray leaked a raw ``numpy`` reduction error while
    the empty frame was already typed.

    The asymmetry was the defect: ``apply_perturbation`` coerced the ndarray
    straight to floats without the row guard ``validate_frame`` applied to a
    frame, so the same "no samples" bug surfaced as ``builtins.ValueError`` from
    deep inside the operator for one container and as a
    :class:`ValidationError` for the other.  Both must reject *before* any
    operator runs (ADR-025).
    """
    ndarray = np.empty((0, 3), dtype=np.float64)
    frame = pd.DataFrame(np.empty((0, 3), dtype=np.float64), columns=["a", "b", "c"])
    with pytest.raises(ValidationError):
        apply_perturbation(ndarray, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())
    with pytest.raises(ValidationError):
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())


@pytest.mark.parametrize("op", ALL_OPS)
def test_zero_row_ndarray_is_rejected_for_every_operator(op: str):
    """The guard sits before dispatch, so no operator may reach an empty buffer."""
    empty = np.empty((0, 3), dtype=np.float64)
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(empty, op, PARAMS[op], rng=_rng())
    assert excinfo.value.context["shape"] == [0, 3]


@pytest.mark.parametrize("op", ALL_OPS)
def test_zero_row_frame_is_rejected_for_every_operator(op: str):
    """The frame path must stay typed for the same six operators."""
    empty = pd.DataFrame(np.empty((0, 3), dtype=np.float64), columns=["a", "b", "c"])
    with pytest.raises(ValidationError):
        apply_perturbation(empty, op, PARAMS[op], rng=_rng())


# ── Repeated column labels: the untyped ``ValueError`` regression ─────────────


def test_duplicate_column_labels_raise_a_typed_error_not_a_builtin_valueerror():
    """Run-002 MAJOR-2: a repeated label leaked ``builtins.ValueError``.

    ``coerce_numeric_frame`` addresses columns by label, and ``df["a"]`` on a
    frame with two ``"a"`` columns returns a *frame*, not a ``Series``.  Each
    duplicate therefore contributed a 2-D block to ``np.column_stack``, and the
    reconstruction failed with::

        builtins.ValueError: Shape of passed values is (4, 5), indices imply (4, 3)

    which is not a type from :mod:`core.errors` — a direct breach of
    conventions §3 ("raise typed errors from ``core.errors``").  The failure was
    also unactionable: it named shapes, not the column the user has to rename.
    """
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3), columns=["a", "b", "a"])
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())

    assert type(excinfo.value) is ValidationError, (
        "a duplicate label must not surface as a more specific error type"
    )
    assert excinfo.value.context["duplicate_columns"] == ["a"]
    assert "a" in str(excinfo.value), "the message must name the column to rename"


def test_every_duplicate_label_is_reported_not_just_the_first():
    """A frame with two repeated labels must name both in one pass."""
    frame = pd.DataFrame(
        np.arange(20, dtype=np.float64).reshape(5, 4),
        columns=["a", "b", "a", "b"],
    )
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(frame, "uniform_noise", {"noise_range": 0.2}, rng=_rng())
    assert set(excinfo.value.context["duplicate_columns"]) == {"a", "b"}


def test_a_unique_label_frame_is_still_accepted():
    """The guard must not reject a frame whose labels merely look similar."""
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3), columns=["a", "b", "c"])
    out = apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.1}, rng=_rng())
    assert list(out.columns) == ["a", "b", "c"]


@pytest.mark.parametrize("op", ALL_OPS)
def test_duplicate_labels_are_rejected_for_every_operator(op: str, float_frame: pd.DataFrame):
    """The guard sits in ``validate_frame``, so no operator may slip past it."""
    duplicated = float_frame.copy()
    duplicated.columns = ["dup", *duplicated.columns[1:-1], "dup"]
    assert duplicated.columns.duplicated().any(), "the fixture lost its duplicate"
    with pytest.raises(ValidationError) as excinfo:
        apply_perturbation(duplicated, op, PARAMS[op], rng=_rng())
    assert excinfo.value.context["duplicate_columns"] == ["dup"]


def test_a_duplicate_column_survives_the_rejection_intact(float_frame: pd.DataFrame):
    """The input is not mutated on the rejection path either (conventions §5)."""
    duplicated = float_frame.copy()
    duplicated.columns = ["dup", *duplicated.columns[1:-1], "dup"]
    before = duplicated.copy()
    with pytest.raises(ValidationError):
        apply_perturbation(duplicated, "feature_dropout", {"dropout_rate": 0.3}, rng=_rng())
    pd.testing.assert_frame_equal(duplicated, before)


def test_stress_tester_delegations_report_the_same_typed_error():
    """The six thin delegations must not re-wrap or swallow the guard either."""
    from modules.stress_module import StressTester

    tester = StressTester(rng=_rng())
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3), columns=["a", "b", "a"])
    with pytest.raises(ValidationError) as excinfo:
        tester.add_gaussian_noise(frame, noise_level=0.1)
    assert excinfo.value.context["duplicate_columns"] == ["a"]


@pytest.mark.parametrize("bad_config", [{}, {"params": {"noise_level": 0.1}}, "gaussian_noise", None])
def test_batch_config_without_a_type_key_is_a_typed_error(bad_config: object):
    """Run-014 M11: a batch entry without ``type`` leaked a raw ``KeyError``.

    The loop indexed ``config["type"]`` before validating the entry, so a
    malformed batch surfaced as ``builtins.KeyError`` rather than a typed
    :class:`ValidationError` that names the defect (ADR-025).  A non-mapping
    entry is the same bug by another route and is rejected identically.
    """
    from modules.stress_module import StressTester

    tester = StressTester(rng=_rng())
    with pytest.raises(ValidationError):
        tester.batch_stress_test(None, pd.DataFrame(), None, [bad_config])


# ── Registry / label contract ────────────────────────────────────────────────


def test_operations_keys_are_exactly_the_six_binding_strings():
    """ADR-010.6: renaming a key silently disables every batch stress test."""
    assert set(OPERATIONS) == {
        "gaussian_noise",
        "uniform_noise",
        "feature_dropout",
        "feature_corruption",
        "scale_perturbation",
        "distribution_shift",
    }
    assert available_operations() == tuple(OPERATIONS)


def test_operation_labels_has_exactly_six_entries():
    assert len(OPERATION_LABELS) == 6
    assert len(set(OPERATION_LABELS.values())) == 6


def test_operation_labels_values_are_the_registry_keys():
    """The translation point must be total in both directions."""
    assert set(OPERATION_LABELS.values()) == set(OPERATIONS)


def test_operation_labels_keys_are_the_ui_literals():
    """§4.4: the labels are the six literals the UI selectbox offers.

    Read out of ``views/module_04_stress.py`` rather than transcribed, so a UI
    label added or renamed without a matching registry entry fails here.
    """
    import ast

    view = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "views"
        / "module_04_stress.py"
    )
    tree = ast.parse(view.read_text(encoding="utf-8"))
    offered: list[str] | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "selectbox"
            and node.args
            and ast.literal_eval(node.args[0]) == "Stress Test Type"
        ):
            offered = list(ast.literal_eval(node.args[1]))
    assert offered is not None, "the Stress Test Type selectbox was not found"
    assert set(offered) == set(OPERATION_LABELS), (
        f"UI offers {offered}; OPERATION_LABELS offers {list(OPERATION_LABELS)}"
    )


@pytest.mark.parametrize("label,key", sorted(OPERATION_LABELS.items()))
def test_operation_key_translates_both_directions(label: str, key: str):
    assert operation_key(label) == key


def test_operation_key_rejects_an_unknown_label():
    with pytest.raises(UnsupportedStressTypeError) as excinfo:
        operation_key("nope")
    assert excinfo.value.context["accepted"] == list(OPERATION_LABELS)
    for label in OPERATION_LABELS:
        assert label in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 5, ["Gaussian Noise"], "gaussian_noise"])
def test_operation_key_rejects_non_label_input(bad):
    """Notably ``"gaussian_noise"`` — the *key* is not a label."""
    with pytest.raises(UnsupportedStressTypeError):
        operation_key(bad)


def test_apply_perturbation_rejects_an_unknown_key():
    """The registry lookup happens before any array work."""
    with pytest.raises(UnsupportedStressTypeError) as excinfo:
        apply_perturbation(np.zeros((3, 2)), "not_an_operator", {}, rng=_rng())
    assert excinfo.value.context["accepted"] == list(OPERATIONS)


def test_a_label_is_not_accepted_as_an_operator_key():
    """The UI label chain and the registry chain must not be interchangeable."""
    with pytest.raises(UnsupportedStressTypeError):
        apply_perturbation(np.zeros((3, 2)), "Gaussian Noise", {}, rng=_rng())


def test_registry_and_labels_agree_in_both_directions():
    """The module-level assertion exists; assert it observably instead."""
    assert tuple(OPERATIONS) == tuple(OPERATION_LABELS.values())


def test_each_operator_declares_its_allowed_params():
    for op, spec in OPERATIONS.items():
        assert spec.name == op
        assert spec.allowed_params, op
        assert set(spec.defaults) <= set(spec.allowed_params), op


# ── Enum plumbing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("corruption_type", CORRUPTION_TYPES)
def test_each_corruption_type_changes_the_frame(corruption_type: str):
    """Not merely accepted by validation — every enum member must do something."""
    frame = pd.DataFrame(np.linspace(1.0, 5.0, 30).reshape(10, 3), columns=list("abc"))
    out = apply_perturbation(
        frame,
        "feature_corruption",
        {"corruption_rate": 0.4, "corruption_type": corruption_type},
        rng=_rng(),
    )
    assert not np.array_equal(out.to_numpy(), frame.to_numpy())


@pytest.mark.parametrize("shift_type", SHIFT_TYPES)
def test_each_shift_type_changes_the_frame(shift_type: str):
    frame = pd.DataFrame(np.linspace(1.0, 5.0, 30).reshape(10, 3), columns=list("abc"))
    out = apply_perturbation(
        frame,
        "distribution_shift",
        {"shift_amount": 1.5, "shift_type": shift_type},
        rng=_rng(),
    )
    assert not np.array_equal(out.to_numpy(), frame.to_numpy())


def test_corruption_zero_actually_writes_zeros():
    """``corruption_type="zero"`` must produce zeros, not merely different values."""
    frame = pd.DataFrame({"a": np.linspace(1.0, 10.0, 20)})
    out = apply_perturbation(
        frame, "feature_corruption", {"corruption_rate": 0.5, "corruption_type": "zero"}, rng=_rng()
    )
    assert (out["a"] == 0.0).sum() == 10


def test_corruption_respects_the_cell_budget_per_column():
    """``int(n_rows * rate)`` cells per column, not a per-column percentage.

    With 20 rows and ``rate=0.5`` exactly 10 cells per column must be replaced.
    Pinned because a per-column *fraction* implementation would coincidentally
    agree at some rates and silently differ at others.
    """
    frame = pd.DataFrame({"a": np.arange(1, 21, dtype="float64")})
    for rate, expected in ((0.5, 10), (0.25, 5), (0.05, 1)):
        out = apply_perturbation(
            frame,
            "feature_corruption",
            {"corruption_rate": rate, "corruption_type": "zero"},
            rng=_rng(),
        )
        # Values are 1..20 with no zeros, so every zeroed cell is a real change.
        changed = int((out["a"] == 0.0).sum())
        assert changed == expected, f"rate={rate}: {changed} cells changed, expected {expected}"


def test_enum_defaults_apply_when_omitted():
    """§4.4: omitting the enum uses the pre-remediation keyword default.

    Compared against an explicit request for that same default, so the assertion
    is about *which* default is chosen rather than about any particular mask.
    """
    frame = pd.DataFrame(np.linspace(1.0, 5.0, 20).reshape(10, 2))
    omitted = apply_perturbation(frame, "feature_corruption", {"corruption_rate": 0.5}, rng=_rng(3))
    explicit = apply_perturbation(
        frame,
        "feature_corruption",
        {"corruption_rate": 0.5, "corruption_type": "zero"},
        rng=_rng(3),
    )
    other = apply_perturbation(
        frame,
        "feature_corruption",
        {"corruption_rate": 0.5, "corruption_type": "random"},
        rng=_rng(3),
    )
    np.testing.assert_array_equal(omitted.to_numpy(), explicit.to_numpy())
    assert not np.array_equal(omitted.to_numpy(), other.to_numpy())


def test_distribution_shift_enum_defaults_to_mean():
    """The same requirement on the other enum, compared the same way."""
    frame = pd.DataFrame(np.linspace(1.0, 5.0, 20).reshape(10, 2))
    omitted = apply_perturbation(frame, "distribution_shift", {"shift_amount": 1.0}, rng=_rng(3))
    mean = apply_perturbation(
        frame,
        "distribution_shift",
        {"shift_amount": 1.0, "shift_type": "mean"},
        rng=_rng(3),
    )
    variance = apply_perturbation(
        frame,
        "distribution_shift",
        {"shift_amount": 1.0, "shift_type": "variance"},
        rng=_rng(3),
    )
    np.testing.assert_array_equal(omitted.to_numpy(), mean.to_numpy())
    assert not np.array_equal(omitted.to_numpy(), variance.to_numpy())


# ── Documented naming debt (ADR-010.7) ───────────────────────────────────────


def test_feature_dropout_masks_cells_not_features():
    """ADR-010.7: the recorded naming debt, pinned as the *actual* behaviour.

    The name and the UI label say "features"; the arithmetic masks individual
    cells.  The arithmetic is preserved for published-results comparability and
    only the docstring was corrected — so this test pins what it does, not what it
    is called.
    """
    frame = pd.DataFrame(np.ones((20, 4)))
    out = apply_perturbation(frame, "feature_dropout", {"dropout_rate": 0.5}, rng=_rng(4))
    zeros = out.to_numpy() == 0.0
    # Per-*column* masking would zero whole columns; per-cell masking zeros a
    # scattered subset of every column.
    assert zeros.any(axis=0).all(), "some column had no zeroed cell"
    assert not (zeros.all(axis=1)).all(), "a whole row was zeroed; that is row masking"


def test_distribution_shift_variance_contracts_toward_the_mean():
    """ADR-010.7: the ``variance`` branch scales deviations, not variance."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = apply_perturbation(
        frame,
        "distribution_shift",
        {"shift_amount": 0.5, "shift_type": "variance"},
        rng=_rng(),
    )
    before_std = frame["a"].std()
    after_std = out["a"].std()
    assert after_std == pytest.approx(before_std * 0.5)
    assert out["a"].mean() == pytest.approx(frame["a"].mean())


def test_distribution_shift_mean_adds_std_scaled_offset():
    """The other branch: a pure translation by ``std * amount``."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = apply_perturbation(
        frame, "distribution_shift", {"shift_amount": 1.0, "shift_type": "mean"}, rng=_rng()
    )
    offset = out["a"].to_numpy() - frame["a"].to_numpy()
    np.testing.assert_allclose(offset, offset[0])
    assert offset[0] == pytest.approx(frame["a"].std(ddof=0))


# ── Explicit bounds ──────────────────────────────────────────────────────────


def test_explicit_bounds_override_the_configured_ones():
    """A per-run policy must actually be enforced by the kernel, not just by
    the validator it happens to call."""
    frame = pd.DataFrame(np.linspace(0.0, 1.0, 20).reshape(10, 2))
    tight = StressBounds(noise_level=(0.0, 0.2))
    with pytest.raises(ParameterOutOfRangeError):
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 0.9}, rng=_rng(), bounds=tight)


def test_bounds_none_uses_the_configured_singleton():
    """``bounds=None`` must route through ``get_config()`` (§4.3)."""
    from core.config import get_config

    frame = pd.DataFrame(np.linspace(0.0, 1.0, 20).reshape(10, 2))
    with pytest.raises(ParameterOutOfRangeError):
        apply_perturbation(frame, "gaussian_noise", {"noise_level": 99.0}, rng=_rng(), bounds=None)
    assert get_config().stress_bounds.noise_level[1] == 5.0


# ── Output finiteness ────────────────────────────────────────────────────────


#: Each operator pushed to the edge of its own declared bounds.  Written as a
#: table rather than a cross-product because validation rejects a parameter the
#: operator does not accept — the point of this test is the *arithmetic* at the
#: boundary, not the cross-product of every parameter with every operator.
EXTREME_PARAMS: dict[str, dict[str, object]] = {
    "gaussian_noise": {"noise_level": 5.0},
    "uniform_noise": {"noise_range": 1.0},
    "feature_dropout": {"dropout_rate": 0.999},
    "feature_corruption": {"corruption_rate": 0.999, "corruption_type": "extreme"},
    "scale_perturbation": {"scale_factor": 10.0},
    "distribution_shift": {"shift_amount": 10.0, "shift_type": "variance"},
}


@pytest.mark.parametrize("op", ALL_OPS)
def test_extreme_but_legal_parameters_stay_finite(op: str):
    """An extreme-but-valid parameter must not produce inf or NaN.

    ``ensure_finite`` is the last gate before return, so this is the test that
    would catch it being removed or moved after the operators run.
    """
    frame = pd.DataFrame(np.linspace(-100.0, 100.0, 60).reshape(20, 3))
    out = apply_perturbation(frame, op, EXTREME_PARAMS[op], rng=_rng())
    assert np.isfinite(out.to_numpy()).all(), f"{op!r} produced non-finite output"


@pytest.mark.parametrize("op", ALL_OPS)
def test_extreme_parameters_on_a_zero_variance_frame_stay_finite(op: str):
    """The two gates combined: max parameter *and* no spread to scale it by."""
    frame = pd.DataFrame(np.full((20, 3), 3.0))
    out = apply_perturbation(frame, op, EXTREME_PARAMS[op], rng=_rng())
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize("op", ALL_OPS)
def test_single_row_input_is_handled(op: str):
    """One row means no per-column spread; every operator must still work."""
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = apply_perturbation(frame, op, PARAMS[op], rng=_rng())
    assert out.shape == (1, 2)
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize("op", ALL_OPS)
def test_wide_frame_is_handled(op: str):
    """A wide frame (many features) must not transpose or misalign."""
    frame = pd.DataFrame(np.linspace(0.0, 1.0, 4 * 60).reshape(4, 60))
    out = apply_perturbation(frame, op, PARAMS[op], rng=_rng())
    assert out.shape == (4, 60)
    assert list(out.columns) == list(frame.columns)


# ── ADR-016.4 the C-contiguity contract ──────────────────────────────────────
#
# `core/validation.py` and the module docstring above both name this test as the
# gate that makes the C-contiguity contract load-bearing.  It did not exist, so
# the contract had a citation and no enforcement — the "gate that is not a gate"
# shape this project has now hit five times.

#: Operators whose output **depends on the memory layout** of the buffer handed
#: to them, measured across every parameter branch in
#: :data:`tests.perturbation_golden.CASES`.
#:
#: These are exactly the two that reduce over ``axis=0`` with floating-point
#: summation — ``_column_stats`` -> ``mean``/``std`` — so the result depends on
#: the order the rows are accumulated in, and that order is what the layout
#: decides.
LAYOUT_SENSITIVE_OPS: tuple[str, ...] = ("gaussian_noise", "distribution_shift")

#: The four that do **not**, each for a structural reason worth freezing so a
#: future operator is not mistaken for evidence that the contract is decorative:
#:
#: * :func:`~core.perturbations._uniform_noise` reduces with ``min``/``max``,
#:   which are exact — reordering cannot change them.
#: * :func:`~core.perturbations._feature_dropout` and
#:   :func:`~core.perturbations._scale_perturbation` are elementwise and take no
#:   reduction at all.
#: * :func:`~core.perturbations._feature_corruption` does reduce, but over the
#:   1-D slice ``values[:, column]``, which numpy always makes contiguous
#:   regardless of the parent's layout.
LAYOUT_INVARIANT_OPS: tuple[str, ...] = (
    "uniform_noise",
    "feature_dropout",
    "feature_corruption",
    "scale_perturbation",
)


def _cancellation_array() -> np.ndarray:
    """A frame whose column statistics depend on summation **order**.

    The first column alternates ``1e16`` with ``1.0``, with a single ``-1e16``.
    Accumulated sequentially the small terms are absorbed by the large ones; any
    other order keeps some of them.  Since one ULP of ``1e16`` is ``2.0``, the
    resulting difference is on the order of whole units rather than the ~1e-16
    drift a benign array produces — so this demonstrates the effect by
    construction instead of hoping a particular numpy's reduction blocking
    happens to expose it.
    """
    n_rows = 64
    rng = _rng(11)
    values = np.empty((n_rows, 3), dtype=np.float64)
    values[:, 0] = 1e16
    values[1::2, 0] = 1.0
    values[1, 0] = -1e16
    values[:, 1] = rng.normal(size=n_rows)
    values[:, 2] = np.arange(n_rows, dtype=np.float64)
    return values


def test_coercion_is_c_contiguous_so_reductions_are_bit_stable():
    """ADR-016.4: the coercion buffer's layout is part of the kernel's output.

    Two halves, because the contract is only worth stating if both hold: the
    buffer **is** C-contiguous, and the layout genuinely **would** change the
    result.  The first alone is a one-line flag check that would stay green if
    the reason evaporated; the second alone would be a curiosity about numpy.

    The buffer is asserted C-contiguous for every shape in the golden fixture,
    including the multi-block ones pandas hands back F-contiguous — which is the
    whole reason the explicit fill loop in :func:`coerce_numeric_array` exists
    instead of ``df.to_numpy(dtype=np.float64)``.

    Returns:
        None.
    """
    from tests import perturbation_golden

    for name, frame in perturbation_golden._shapes().items():
        if isinstance(frame, pd.DataFrame):
            coerced = coerce_numeric_array(frame)
            assert coerced.flags.c_contiguous, (
                f"coerce_numeric_array({name!r}) returned a non-C-contiguous "
                f"buffer (C={coerced.flags.c_contiguous}, "
                f"F={coerced.flags.f_contiguous}). ADR-016.4: the layout of this "
                "buffer changes the output of every operator that reduces over "
                "axis=0."
            )
        else:  # ndarray input is passed through, not coerced
            coerced = np.asarray(frame, dtype=np.float64)

    # ...and the consequence: identical values, different layout, different
    # answer — for the two operators that reduce, and for no others.
    base = _cancellation_array()
    c_ordered = base.copy(order="C")
    f_ordered = base.copy(order="F")

    for op, param_sets in perturbation_golden.CASES.items():
        differs = False
        for params in param_sets:
            spec = OPERATIONS[op]
            pruned = {k: v for k, v in params.items() if k in spec.allowed_params}
            from_c = apply_perturbation(
                c_ordered.copy(order="C"), op, dict(pruned), rng=_rng(0)
            )
            from_f = apply_perturbation(
                f_ordered.copy(order="F"), op, dict(pruned), rng=_rng(0)
            )
            if not np.array_equal(from_c, from_f):
                differs = True

        if op in LAYOUT_SENSITIVE_OPS:
            assert differs, (
                f"{op!r} is recorded as layout-sensitive but produced identical "
                "output from a C-ordered and an F-ordered copy of the same "
                "values. Either the operator stopped reducing over axis=0 — in "
                "which case move it to LAYOUT_INVARIANT_OPS and say why — or "
                "numpy's reduction stopped depending on layout, which weakens "
                "the reason this contract exists."
            )
        else:
            assert not differs, (
                f"{op!r} is recorded as layout-invariant but its output changed "
                "with the buffer's layout. It now depends on memory order, so "
                "move it to LAYOUT_SENSITIVE_OPS."
            )


def test_pandas_hands_back_f_contiguous_for_the_multiblock_regression_frame(
    mixed_frame: pd.DataFrame,
):
    """The justification, asserted separately from the contract.

    :func:`coerce_numeric_array`'s explicit fill loop exists because
    ``DataFrame.to_numpy(dtype=float64)`` returns an **F-contiguous** array for a
    multi-block frame — and the ``int64 + float64`` frame is both the
    ADR-010.4 regression shape and the realistic post-``read_csv`` shape.  With
    or without ``copy=False``, on the pinned pandas.

    This is a **third-party** property, so it is not asserted inside the
    contract test above: a future pandas that returns C-contiguous would make
    the explicit loop redundant, not wrong, and it should surface as a review of
    *this* test rather than as a failure of the kernel's guarantee.  If pandas
    changes, re-examine the loop in :func:`coerce_numeric_array` — do not delete
    this test silently.

    Returns:
        None.
    """
    for copy_flag in (True, False):
        native = mixed_frame.to_numpy(dtype=np.float64, copy=copy_flag)
        assert native.flags.f_contiguous and not native.flags.c_contiguous, (
            f"to_numpy(dtype=float64, copy={copy_flag}) is no longer F-contiguous for "
            "a mixed int64+float64 frame. The explicit C-contiguous fill loop in "
            "coerce_numeric_array may now be unnecessary; verify that before "
            "relaxing anything."
        )


def test_the_contiguity_gate_would_catch_an_f_contiguous_coercion():
    """Falsify :func:`test_coercion_is_c_contiguous_so_reductions_are_bit_stable`.

    Two ways that gate could be decorative, each closed here:

    1. **A flag check that cannot fail.**  The contract half is re-checked
       against an array built to be F-contiguous, so the assertion is shown to
       distinguish the two layouts rather than accepting both.
    2. **A partition that names nothing.**  The two halves of
       :data:`LAYOUT_SENSITIVE_OPS` / :data:`LAYOUT_INVARIANT_OPS` together must
       be exactly the six registry keys, so an operator cannot be dropped from
       both lists and quietly exempted from the layout contract.

    Returns:
        None.
    """
    assert set(LAYOUT_SENSITIVE_OPS) | set(LAYOUT_INVARIANT_OPS) == set(ALL_OPS), (
        f"the layout partition covers {sorted(set(LAYOUT_SENSITIVE_OPS) | set(LAYOUT_INVARIANT_OPS))} "
        f"but the registry holds {sorted(ALL_OPS)}"
    )
    assert not set(LAYOUT_SENSITIVE_OPS) & set(LAYOUT_INVARIANT_OPS), (
        "an operator is recorded as both layout-sensitive and layout-invariant"
    )

    # 1. Mechanism: the contract assertion distinguishes the layouts.
    coerced = coerce_numeric_array(pd.DataFrame(_cancellation_array(), columns=list("abc")))
    assert coerced.flags.c_contiguous
    f_ordered = np.asfortranarray(coerced)
    assert f_ordered.flags.f_contiguous and not f_ordered.flags.c_contiguous, (
        "could not build an F-contiguous buffer to falsify the contract check "
        "against; the mechanism is untested"
    )
    assert f_ordered.flags.c_contiguous is not coerced.flags.c_contiguous or (
        coerced.shape[1] == 1
    ), "the two layouts were not actually distinguished"


# ── ADR-016.7 the bit-exact freeze ────────────────────────────────────────────


def test_perturbation_output_is_bit_identical():
    """ADR-016.7: the kernel's output bytes are frozen, and this runs on every suite.

    This is **not** the ADR-010.5 parity freeze.  Those pre-remediation numbers
    are deliberately *not* frozen and nothing asserts them, because a vectorised
    implementation cannot reproduce them.  What is frozen here is the
    **post-Wave-2** kernel, so that the bounded-memory redesign of
    :func:`apply_perturbation` could be *proven* to be behaviour-preserving
    rather than merely asserted to be — which is the only thing that made a
    4.00x → 2.13x memory reduction safe to land.

    The fixture covers every operator against every branch-selecting parameter
    set, at two seeds, across eight input shapes, and additionally freezes the
    container contract (column labels, index labels, dtype) rather than only the
    values.  A refactor that preserved the numbers while changing, say, the index
    it hands back would still be caught.

    Before this test existed the fixture was written by ``--capture`` and read by
    ``--check``, and nothing in the suite called either — so the "freeze" was a
    claim a maintainer had to know to type.  That is the gate-that-is-not-a-gate
    shape this project has now hit four times (roadmap deviations 0a, 8 and 11;
    ADR-016.8).

    Returns:
        None.
    """
    from tests import perturbation_golden

    frozen = perturbation_golden.load_fixture()
    live = perturbation_golden.snapshot()
    problems = perturbation_golden.diff(frozen, live)
    assert not problems, (
        "apply_perturbation's output is no longer bit-identical to the frozen "
        "kernel (ADR-016.7). If the change is deliberate, regenerate with "
        "`python -m tests.perturbation_golden --capture` and justify it; if it "
        "is not, this is the bug.\n" + "\n".join(problems)
    )


def test_the_golden_gate_would_catch_a_changed_kernel():
    """Falsify :func:`test_perturbation_output_is_bit_identical`.

    Three ways the freeze could be decorative, each closed here:

    1. **A fixture that compares nothing.**  The comparison lives in
       :func:`tests.perturbation_golden.diff`; this injects a corrupted digest
       and a dropped shape and requires both to be reported.
    2. **A fixture that was never compared against the real kernel.**  The live
       control below re-requires that the *unmodified* fixture is clean, so the
       injected-copy check cannot pass because the gate is broken for everyone.
    3. **A fixture too small to notice a change.**  The counts are asserted, and
       in particular every registry key and every input shape must be present —
       otherwise an operator removed from :data:`CASES` would shrink the fixture
       and stay green.

    Returns:
        None.
    """
    from tests import perturbation_golden

    frozen = perturbation_golden.load_fixture()

    # 1. Mechanism: a changed digest is reported, and named.
    poisoned = copy.deepcopy(frozen)
    victim = sorted(poisoned["shapes"]["mixed_frame"])[0]
    poisoned["shapes"]["mixed_frame"][victim] = "0" * 64
    problems = perturbation_golden.diff(poisoned, frozen)
    assert any(victim in line for line in problems), (
        f"a corrupted digest was not reported; the freeze would not notice a "
        f"changed value at {victim!r}: {problems}"
    )

    # ...and a *removed* shape is reported too, so a refactor that stops
    # producing an output for some input cannot hide behind an absent entry.
    shrunk = copy.deepcopy(frozen)
    del shrunk["shapes"]["fortran_array"]
    assert any("fortran_array" in line for line in perturbation_golden.diff(shrunk, frozen)), (
        "a dropped input shape was not reported; the freeze would silently "
        "shrink instead of failing"
    )

    # 2. Positive control against the live fixture: unmodified, it is clean.
    assert not perturbation_golden.diff(frozen, frozen), (
        "diff() reports a difference between a record and itself; it is not a "
        "comparison and every use of it above is vacuous"
    )

    # 3. Coverage: the freeze is wide enough for a change to land in.
    live = perturbation_golden.snapshot()
    assert set(live["shapes"]) == set(perturbation_golden._shapes()), (
        "every input shape in tests/perturbation_golden._shapes() must be "
        f"frozen; got {sorted(live['shapes'])}"
    )
    assert set(live["shapes"]) == set(frozen["shapes"]), (
        f"the fixture covers {sorted(frozen['shapes'])} but the live snapshot "
        f"produces {sorted(live['shapes'])}"
    )
    frozen_operators = {
        key.split("|")[0] for shape in frozen["shapes"].values() for key in shape
    }
    for op in ALL_OPS:
        assert op in frozen_operators, f"{op!r} is not frozen against any input shape"
    total = sum(len(v) for v in live["shapes"].values())
    assert total > 500, (
        f"only {total} digests are frozen; a freeze that small would not notice "
        "a change to one parameter branch"
    )
    assert set(live["rejections"]) == {name for name, _ in perturbation_golden.REJECTIONS}, (
        "every documented rejection must stay frozen, including the ones whose "
        "error *type* is part of the contract"
    )


# ── ADR-016.5 structural purity: the operator's input is read-only ───────────
#
# ADR-016.5 claims two mechanisms and asserts neither.  Deleting
# `core.perturbations._freeze_input` outright — removing the whole of the
# structural-purity redesign — left all 727 tests green when this section was
# written, because the guarantee is a *flag on an array nobody inspects* and every
# behavioural test asks a different question ("did the caller's bytes change?").
#
# That is the same defect class as the missing C-contiguity gate immediately
# above: a contract stated in prose, cited by a test that does not check it, and
# therefore unverifiable.  The three tests below make it checkable.


@pytest.mark.parametrize("op", ALL_OPS)
def test_the_operator_is_handed_a_read_only_buffer(
    op: str, mixed_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-016.5: purity is *structural*, so the operator's input is read-only.

    ``apply_perturbation`` clears ``writeable`` before calling into the registry.
    This observes that flag from inside the operator, through the public entry
    point and the real registry — no private helper is imported, because the
    contract is about what an *operator* receives, not about which function sets
    it.

    The flag is what makes :func:`test_input_is_never_mutated` load-bearing
    rather than decorative.  That test compares bytes, so it passes today only
    because every operator happens to allocate its own output.  An operator that
    wrote in place would still pass it — the write would land in a buffer the
    caller no longer holds, so the caller's bytes are untouched while the
    operator's contract is already broken.  With the flag set, such an operator
    raises ``ValueError`` from numpy at the moment of the write instead.

    Args:
        op: Registry key under test.
        mixed_frame: The regression frame (``int64`` + ``float64``).
        monkeypatch: Used to swap in a probe that observes the flag; the original
            registry entry is restored afterwards.

    Returns:
        None
    """
    observed: list[bool] = []
    spec = OPERATIONS[op]

    def probe(values: np.ndarray, params, rng):
        observed.append(bool(values.flags.writeable))
        return spec.fn(values, params, rng)

    monkeypatch.setitem(OPERATIONS, op, replace(spec, fn=probe))

    apply_perturbation(mixed_frame, op, PARAMS[op], rng=_rng())

    assert observed == [False], (
        f"{op!r} was handed a {'writeable' if observed[0] else 'read-only'} "
        "buffer. ADR-016.5 requires read-only: it is what turns operator purity "
        "from a convention into something numpy enforces, instead of a post-hoc "
        "byte comparison that an in-place operator passes anyway."
    )


def test_a_read_only_buffer_is_what_rejects_an_in_place_operator() -> None:
    """Falsify :func:`test_the_operator_is_handed_a_read_only_buffer`.

    A gate that asserts a flag is set can pass for the wrong reason: the flag
    might be set but not actually enforced, or the assertion might be satisfied
    by a buffer the operator never saw.  So the *same write* is attempted twice,
    once on a writeable buffer and once on a read-only one, and must succeed and
    fail respectively.

    Asserting only the negative would be the vacuous half — ``pytest.raises``
    wrapped around a line that raises for an unrelated reason is a green gate.

    Returns:
        None
    """

    def writer(values: np.ndarray, _params, _rng) -> np.ndarray:
        values[:] = 0.0  # the exact defect ADR-016.5 exists to make impossible
        return values

    writeable = np.full((4, 3), 7.0)
    assert writeable.flags.writeable
    writer(writeable, {}, _rng())  # must NOT raise
    assert not writeable.any(), "the control write did not happen, so the check below is vacuous"

    frozen = np.full((4, 3), 7.0)
    frozen.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        writer(frozen, {}, _rng())
    assert np.full((4, 3), 7.0).tobytes() == frozen.tobytes(), (
        "the rejected write still modified the caller's data"
    )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["float64", "int64"])
def test_the_callers_own_array_keeps_its_writeable_flag(order: str, dtype: str) -> None:
    """ADR-016.5: freezing is done on a *view*, never on the caller's buffer.

    ``np.asarray`` hands back the caller's own array when no conversion is
    needed — which is exactly the ``float64`` input case.  Clearing ``writeable``
    on *that* object would silently make the caller's array read-only: a visible,
    permanent side effect on a value the caller owns, introduced by a function
    documented to return a new object and never mutate its argument.

    ``_to_float_array`` takes a ``.view()`` precisely to avoid it: the view shares
    memory but has independent flags.  Deleting that ``.view()`` also left the
    whole suite green when this test was written, because every purity assertion
    asks about *contents* and none asks about *flags on the caller's object*.

    The ``F`` order is load-bearing rather than decorative: an F-ordered
    ``float64`` array is the other input where ``asarray`` preserves layout, so
    it reaches the same aliasing path as the C-ordered case.

    Args:
        order: Memory order the caller's array is built with.
        dtype: ``float64`` reaches the aliasing path; ``int64`` does not, because
            ``asarray`` must convert and therefore allocates a fresh buffer.

    Returns:
        None
    """
    values = np.ones((6, 4), dtype=dtype, order=order)
    assert values.flags.writeable, "the caller's array started out read-only"

    apply_perturbation(values, "gaussian_noise", PARAMS["gaussian_noise"], rng=_rng())

    assert values.flags.writeable, (
        f"apply_perturbation made the caller's own {order}-ordered {dtype} array "
        "read-only. That is a side effect on an object the caller owns, from a "
        "function whose contract is to return a new object and mutate nothing."
    )
    assert np.ones((6, 4), dtype=dtype).tobytes() == values.tobytes(), (
        f"apply_perturbation modified the caller's own {order}-ordered {dtype} array"
    )


# ── ADR-016.2/.3/.6 the bounded-memory contract ──────────────────────────────
#
# ADR-016's headline claim is "4.00x -> 2.13x".  It was *measured*, and the
# measurement is reproduced below, but **no test asserted it**: re-introducing
# the per-column dict plus ``column_stack`` that ADR-016.3 removed, or the
# full-size ``values.copy()`` purity guard that ADR-016.5 removed, or the double
# boolean mask that ADR-016.6 removed, each left all 727 tests green.
#
# Every other ADR-016 claim has a gate — bit-identity, C-contiguity, annotations,
# `__future__`, the uploader, ``run_batch_stress`` — and the bit-identity freeze in
# particular is why a refactor this size was safe to land at all.  The memory
# claim is the one that was *not* gated, which is the worst place for it to be
# missing: it is the entire justification for the refactor, and nothing in the
# suite distinguishes a 2.13x kernel from a 4.00x one.

#: Rows in the measured frame.  The ratio is scale-invariant (verified at 20k,
#: 40k and 200k: 2.134x / 2.129x / 2.126x), so this size buys *margin* against
#: interference rather than signal.  200k keeps the measurement two orders of
#: magnitude above any fixed-size allocator noise.
MEMORY_ROWS = 200_000

#: Half ``int64``, half ``float64`` — the ADR-010.4 regression shape, and the
#: shape that makes every buffer in the call frame-sized.
MEMORY_INT_COLUMNS = 25
MEMORY_FLOAT_COLUMNS = 26

#: Peak allocation allowed for one ``apply_perturbation`` call, as a multiple of
#: the input frame's own bytes.  The decomposition is exact and is checked, not
#: assumed, by
#: :func:`test_the_entry_point_peak_decomposes_into_the_documented_buffers`:
#:
#: * 1.00x — the coercion buffer (:func:`coerce_numeric_array`);
#: * 1.00x — the operator's own output;
#: * 0.125x — the single boolean ``isfinite`` mask (ADR-016.6).
#:
#: i.e. 2.125x measured against a 2.20x bound.  The removed implementations
#: measured 3.13x (purity guard) and 2.25x (double mask), so the margin is a whole
#: buffer against the first and exactly one boolean mask against the second.  Both
#: figures are reproducible from the current tree — add the buffer back to
#: ``apply_perturbation`` and measure — and both are re-measured against the bound
#: by :func:`test_the_memory_gate_would_catch_each_removed_buffer`.  An earlier
#: draft of this comment said "3.00x (purity guard)", which does not reproduce and
#: has been corrected rather than propagated; see roadmap deviation 15.
ENTRY_POINT_PEAK_BOUND = 2.20

#: Peak allocation allowed for :func:`coerce_numeric_array` on its own.  Its
#: output buffer *is* the 1.00x; anything beyond 0.10x is a second frame-sized
#: buffer, which is precisely what the pre-ADR-016 dict + ``column_stack`` did
#: (measured 1.49x: the converted ``int64`` columns were all held in the dict at
#: once while ``column_stack`` built the result alongside them).
COERCION_PEAK_BOUND = 1.10

#: What one ``bool`` mask over a frame of ``float64`` costs, as a multiple of that
#: frame: a mask is one byte per value and a ``float64`` is eight.  This is the
#: residue :func:`test_the_entry_point_peak_decomposes_into_the_documented_buffers`
#: expects above the two unavoidable ``float64`` buffers.
_BOOL_MASK_MULTIPLE = 1.0 / 8.0

#: Slack on that decomposition.  It is not zero because ``tracemalloc`` traces
#: *everything*, and the call frame also allocates a few fixed-size objects (the
#: parameter normalisation dict, the RNG's internal state) that do not scale with
#: the frame — about 65KB against an 81.6MB frame, or 0.08%.  Measured overshoot is
#: 0.64% and identical across repeated runs, because :func:`_peak_bytes` takes the
#: minimum of three.  The bound is two orders of magnitude above the noise and an
#: order of magnitude below the next real buffer (0.125x), so it discriminates
#: rather than merely cushioning.
RESIDUE_TOLERANCE = 0.02


def _memory_frame(rows: int = MEMORY_ROWS) -> pd.DataFrame:
    """The measured frame: ``int64`` alongside ``float64``, no NaNs.

    Args:
        rows: Row count.  Smaller values are used by the falsification controls,
            which need the same *ratio* and not the same absolute cost.

    Returns:
        pd.DataFrame: ``rows x 51``, half of each dtype.
    """
    rng = np.random.default_rng(20260906)
    return pd.DataFrame(
        {
            **{f"i{i}": rng.integers(0, 50, size=rows) for i in range(MEMORY_INT_COLUMNS)},
            **{f"f{i}": rng.normal(size=rows) for i in range(MEMORY_FLOAT_COLUMNS)},
        }
    )


def _frame_bytes(frame: pd.DataFrame) -> int:
    """The frame's own data footprint, which every measured buffer is a multiple of.

    ``index=False`` on purpose.  ``deep=True`` alone also counts the column
    ``Index`` (132 bytes here), which is neither a buffer any measured code path
    allocates nor part of the "one frame's worth of ``float64``" the bound is
    derived from.  Counting it would inflate the denominator and make every
    ratio below look slightly *better*, so this is the stricter of the two
    spellings — and it makes ``base == rows * columns * 8`` exact rather than
    approximate, which is what lets
    :func:`test_the_entry_point_peak_decomposes_into_the_documented_buffers`
    assert the decomposition instead of assuming it.

    Args:
        frame: The frame to measure.

    Returns:
        int: Total bytes held by the frame's column data.
    """
    return int(frame.memory_usage(deep=True, index=False).sum())


def _peak_bytes(action, *, repeats: int = 1) -> int:
    """Peak traced allocation during ``action()``, taking the best of ``repeats``.

    ``tracemalloc`` is process-wide, so a background thread allocating during the
    measured window inflates the peak.  Taking the **minimum** over a few runs
    discards that interference: a genuine extra frame-sized buffer is present on
    every run and so survives the minimum, whereas a transient allocation from
    another thread usually does not.  This is the conservative choice — it makes
    the gate harder to pass spuriously, not easier.

    Args:
        action: Zero-argument callable to measure.
        repeats: How many times to measure; the smallest peak wins.

    Returns:
        int: Peak bytes allocated beyond the baseline taken at entry.
    """
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        # A pre-existing session carries a peak from someone else's work, so
        # measuring inside it would report their high-water mark, not this call's.
        tracemalloc.stop()
    best: int | None = None
    try:
        for _ in range(repeats):
            tracemalloc.start()
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            try:
                action()
            finally:
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            best = peak - baseline if best is None else min(best, peak - baseline)
    finally:
        if was_tracing:
            tracemalloc.start()
    assert best is not None
    return best


def _entry_point_peak_ratio(frame: pd.DataFrame, op: str, params: dict) -> float:
    """Peak allocation of one ``apply_perturbation`` call, in frame-multiples."""
    base = _frame_bytes(frame)
    return _peak_bytes(
        lambda: apply_perturbation(frame, op, dict(params), rng=_rng(1)), repeats=3
    ) / base


@pytest.mark.parametrize("op", ALL_OPS)
def test_apply_perturbation_peak_allocation_stays_bounded(op: str) -> None:
    """ADR-016: the headline 4.00x -> 2.13x claim, as an executed gate.

    Measures the peak traced allocation of a single ``apply_perturbation`` call on
    the ADR-010.4 regression frame and requires it to stay within
    :data:`ENTRY_POINT_PEAK_BOUND` multiples of the frame.

    Every buffer this allows for is real and unavoidable: the coercion buffer, the
    operator's output, and one boolean finiteness mask.  All three removed
    implementations sit outside the bound, which
    :func:`test_the_memory_gate_would_catch_each_removed_buffer` proves rather
    than asserts.

    Args:
        op: Registry key under test.

    Returns:
        None
    """
    frame = _memory_frame()
    ratio = _entry_point_peak_ratio(frame, op, PARAMS[op])
    assert ratio <= ENTRY_POINT_PEAK_BOUND, (
        f"{op!r} peaked at {ratio:.3f}x the input frame, above the "
        f"{ENTRY_POINT_PEAK_BOUND}x bound (ADR-016). The bounded-memory redesign "
        "exists so that a frame this size does not need four times itself in "
        "transient buffers; re-measure before raising this bound rather than "
        "assuming the extra buffer is free."
    )


def test_coercion_peak_allocation_stays_bounded() -> None:
    """ADR-016.3: the coercion is one preallocated buffer, not a dict plus a stack.

    A separate bound from the entry point because the two regressions are not the
    same regression.  The per-column dict + ``column_stack`` spelling measured
    1.49x here while leaving the entry-point peak *unchanged* — the extra buffer is
    freed before the operator allocates its output, so it never dominates the
    total.  A gate on the entry point alone would not have seen it.

    Returns:
        None
    """
    frame = _memory_frame()
    ratio = _peak_bytes(lambda: coerce_numeric_array(frame), repeats=3) / _frame_bytes(frame)
    assert ratio <= COERCION_PEAK_BOUND, (
        f"coerce_numeric_array peaked at {ratio:.3f}x the input frame, above the "
        f"{COERCION_PEAK_BOUND}x bound (ADR-016.3). One pass into one preallocated "
        "buffer is the contract; a per-column dict, a list of columns, or a "
        "stacked copy all hold a second frame-sized buffer."
    )


def test_the_entry_point_peak_decomposes_into_the_documented_buffers() -> None:
    """The bound is a sum of named terms, so a future breach can be diagnosed.

    A bare "ratio <= 2.20" tells a maintainer only that something got bigger.
    This asserts the *terms*, each a fixed fraction of the frame and therefore
    checkable: the coercion buffer is exactly one frame's worth of ``float64``,
    and the operator's output is another.  What remains above ``2.00x`` should be
    :data:`_BOOL_MASK_MULTIPLE` — the single ``isfinite`` mask of ADR-016.6 — which
    makes the bound a sum a reader can check by hand, and makes a breach
    attributable to a named buffer instead of to "memory".

    Returns:
        None
    """
    frame = _memory_frame()
    base = _frame_bytes(frame)
    n_values = MEMORY_ROWS * (MEMORY_INT_COLUMNS + MEMORY_FLOAT_COLUMNS)

    assert base == n_values * 8, (
        f"the measured frame is {base} bytes for {n_values} values; every "
        "documented multiple below is derived from all columns being 8 bytes wide, "
        "and that assumption no longer holds"
    )

    coercion = _peak_bytes(lambda: coerce_numeric_array(frame), repeats=3) / base
    assert 1.0 <= coercion <= COERCION_PEAK_BOUND, (
        f"the coercion buffer should be exactly one frame's worth of float64 "
        f"(1.00x); measured {coercion:.3f}x"
    )

    worst = max(_entry_point_peak_ratio(frame, op, PARAMS[op]) for op in ALL_OPS)
    residue = worst - 2.0
    error = abs(residue - _BOOL_MASK_MULTIPLE) / _BOOL_MASK_MULTIPLE
    assert error <= RESIDUE_TOLERANCE, (
        f"the peak across all six operators is {worst:.4f}x, leaving {residue:.4f}x "
        f"above the two float64 buffers. That residue should be one bool mask "
        f"({_BOOL_MASK_MULTIPLE:.4f}x) to within {RESIDUE_TOLERANCE:.0%}; measured "
        f"{error:.2%} off. A larger residue means a transient frame-sized buffer "
        "has come back, and a smaller one means ensure_finite has stopped checking."
    )


def test_the_memory_gate_would_catch_each_removed_buffer() -> None:
    """Falsify both memory gates by re-measuring the implementations ADR-016 removed.

    A bound is only as good as its ability to fail.  Each spelling below is a
    real, previously-shipped implementation of this project, reproduced here in a
    few lines, and each must measure **above** the bound that
    :func:`test_apply_perturbation_peak_allocation_stays_bounded` and
    :func:`test_coercion_peak_allocation_stays_bounded` enforce.  If any of them
    now measures below, those bounds have stopped being able to detect the
    regression they were written for.

    Run on a 40k-row frame: the ratios are scale-invariant, and the point is the
    *separation* between the numbers, not their size.

    Returns:
        None
    """
    frame = _memory_frame(rows=40_000)
    base = _frame_bytes(frame)

    # 1. ADR-016.3's removed coercion: every converted column held in a dict, then
    #    stacked — two frame-sized allocations live at once.
    def removed_coercion(df: pd.DataFrame) -> np.ndarray:
        columns = {name: df[name].to_numpy(dtype=np.float64) for name in df.columns}
        return np.column_stack([columns[name] for name in df.columns])

    coercion_ratio = _peak_bytes(lambda: removed_coercion(frame)) / base
    assert coercion_ratio > COERCION_PEAK_BOUND, (
        f"the removed per-column dict + column_stack coercion measured "
        f"{coercion_ratio:.3f}x, at or below the {COERCION_PEAK_BOUND}x bound. The "
        "bound can no longer see the regression it was written for."
    )

    # 2. ADR-016.5's removed purity guard: a full-size copy kept only to compare
    #    afterwards.  Reproduced by copying the coercion buffer, which is exactly
    #    the buffer `_freeze_input` replaced.
    def removed_copy_guard(df: pd.DataFrame) -> np.ndarray:
        return coerce_numeric_array(df).copy()

    guard_ratio = _peak_bytes(lambda: removed_copy_guard(frame)) / base
    assert guard_ratio + 1.0 > ENTRY_POINT_PEAK_BOUND, (
        f"the removed full-size purity copy measured {guard_ratio:.3f}x, which "
        f"with an operator's own output does not exceed {ENTRY_POINT_PEAK_BOUND}x"
    )

    # 3. ADR-016.6's removed double mask: `~np.isfinite(array)` builds the mask
    #    and a second full-size array to invert it, both live at once.
    buffer_ = coerce_numeric_array(frame)
    single = _peak_bytes(lambda: bool(np.isfinite(buffer_).all())) / base
    double = _peak_bytes(lambda: bool((~np.isfinite(buffer_)).any())) / base
    assert double - single > 1.0 / 8.0 - 0.01, (
        f"the removed double mask measured {double - single:.3f}x above the "
        "single-mask spelling, which does not account for a second frame-sized "
        "boolean array; the ADR-016.6 claim is no longer measurable"
    )
    # ...and the live kernel must sit on the cheap side of that same edge.
    assert 2.0 + single <= ENTRY_POINT_PEAK_BOUND, (
        f"one boolean mask plus the two float64 buffers is {2.0 + single:.3f}x, "
        f"already above the {ENTRY_POINT_PEAK_BOUND}x bound"
    )
