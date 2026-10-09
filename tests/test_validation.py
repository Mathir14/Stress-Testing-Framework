"""Parameter-validation tests (architecture.md §4.3, §10.2).

**Tier T1** — requires numpy/pandas.

The binding invariant here is the **single source of truth**: the accepted ranges
live in :class:`core.config.StressBounds` and every decision is made by
:func:`core.validation.validate_stress_params`, so the UI can never offer a value
the kernel rejects.  That is asserted three ways:

1. Every slider minimum and maximum in ``views/module_04_stress.py`` is
   **parsed out of the view's own AST**, not copied into a list here.  A copied
   list is a comment; a parsed one cannot drift.  Every *position* a stepped
   slider can produce is then swept against the kernel, so the UI can never
   offer a value ``validate_stress_params`` rejects (ADR-015).
2. Every ``corruption_type`` / ``shift_type`` option the UI offers is parsed the
   same way and must be accepted.
3. ``bounds=None`` and an explicit ``get_config().stress_bounds`` must agree on
   every accept/reject decision.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from core.config import StressBounds, get_config
from core.errors import (
    ParameterOutOfRangeError,
    UnsupportedStressTypeError,
    ValidationError,
)
from core.validation import (
    CORRUPTION_TYPES,
    ENUM_PARAMETERS,
    NUMERIC_PARAMETERS,
    SHIFT_TYPES,
    accepted_keys,
    coerce_numeric_frame,
    ensure_finite,
    numeric_parameter_table,
    resolve_bounds,
    validate_frame,
    validate_stress_params,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRESS_VIEW = PROJECT_ROOT / "views" / "module_04_stress.py"


# ── The UI surface, read from the view's AST ─────────────────────────────────


def _view_tree() -> ast.Module:
    return ast.parse(STRESS_VIEW.read_text(encoding="utf-8"))


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _slider_params() -> dict[str, tuple[float, float, float, float]]:
    """Return ``label -> (min_value, max_value, default, step)`` from the view.

    ``st.slider(label, min_value, max_value, value, step, ...)`` passes the
    bounds positionally, so ``args[1]`` / ``args[2]`` / ``args[3]`` / ``args[4]``
    are read directly.  Reading them rather than transcribing them is the point:
    a transcribed list is a comment that stops being true when the UI moves.
    """
    found: dict[str, tuple[float, float, float, float]] = {}
    for node in ast.walk(_view_tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "slider"):
            continue
        if len(node.args) < 5:
            continue
        try:
            label = _literal(node.args[0])
            low, high = float(_literal(node.args[1])), float(_literal(node.args[2]))
            default = float(_literal(node.args[3]))
            step = float(_literal(node.args[4]))
        except (ValueError, SyntaxError, TypeError):
            continue
        found[str(label)] = (low, high, default, step)
    return found


def _steps_between(low: float, high: float, step: float) -> list[float]:
    """Return every value a stepped slider between ``low`` and ``high`` produces.

    Mirrors Streamlit's inclusive upper endpoint, rounding the way the widget
    does so ``0.0 → 1.0`` by ``0.05`` terminates on ``1.0`` and not ``0.95``.
    """
    count = int(round((high - low) / step))
    return [round(low + index * step, 10) for index in range(count + 1)]


#: ``st.slider`` label -> stress-parameter name.  Kept explicit because the
#: mapping is a *design* fact (this slider drives this parameter), while the
#: numeric bounds above are read from the view.
SLIDER_LABELS: dict[str, str] = {
    "Noise Level (std multiplier)": "noise_level",
    "Noise Range (fraction of feature range)": "noise_range",
    "Dropout Rate": "dropout_rate",
    "Corruption Rate": "corruption_rate",
    "Scale Factor": "scale_factor",
    "Shift Amount": "shift_amount",
}


def _ui_enum_options() -> dict[str, list[str]]:
    """Return the enum option lists the UI offers for corruption and shift type."""
    found: dict[str, list[str]] = {}
    for node in ast.walk(_view_tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {
            "selectbox",
            "radio",
        }:
            continue
        if not node.args:
            continue
        label = _literal(node.args[0])
        if label in {"Corruption Type", "Shift Type"}:
            found[label] = list(_literal(node.args[1]))
    return found


#: Every parameter value the batch-stress tab hard-codes.  Read from the source
#: by :func:`_ui_batch_params` rather than transcribed.
def _ui_batch_params() -> dict[str, set[float]]:
    """Return ``parameter -> {values the batch tab uses}`` from the view AST."""
    values: dict[str, set[float]] = {name: set() for name in NUMERIC_PARAMETERS.values()}
    for node in ast.walk(_view_tree()):
        if not isinstance(node, ast.Dict):
            continue
        try:
            payload = _literal(node)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(payload, dict) or "type" not in payload:
            continue
        params = payload.get("params")
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if key in values and isinstance(value, (int, float)):
                values[key].add(float(value))
    return values


# ── §4.3 rule 1–2: every reachable UI value is accepted ──────────────────────


def test_the_view_actually_has_the_six_sliders():
    """Guard for the parser above: if the UI shape changes, fail loudly here
    rather than silently validating nothing."""
    sliders = _slider_params()
    assert set(sliders) == set(SLIDER_LABELS), f"stress slider labels changed: {sorted(sliders)}"


@pytest.mark.parametrize("label", sorted(SLIDER_LABELS))
def test_every_slider_position_is_accepted(label: str):
    """§4.3: **every value the slider can produce** is accepted — no exceptions.

    The steps are swept rather than the endpoints, because a slider is stepped:
    ``0.0, 0.05, 0.1, …`` up to its maximum.  This is the invariant the module
    docstring claims, and it used to have exactly one carve-out: the ``Scale
    Factor`` slider's floor of ``1.0`` **was** ``StressBounds.scale_factor``'s
    exclusive lower bound, so the UI could produce a value the kernel rejected.
    ADR-015 resolved that by raising the floor to ``1.1``, so the carve-out is
    gone.  Do not reintroduce one — if a slider needs a wider range, widen the
    slider; if it needs a *narrower* one, narrow the slider.  Publishing a
    position the kernel refuses is a dead control, and per conventions §7 it can
    only ever be reported as a rendered error the user cannot act on.
    """
    low, high, default, step = _slider_params()[label]
    parameter = SLIDER_LABELS[label]
    op = next(o for o, p in NUMERIC_PARAMETERS.items() if p == parameter)

    positions = _steps_between(low, high, step)
    assert low in positions and high in positions, positions

    rejected = []
    for value in positions:
        try:
            validate_stress_params(op, {parameter: value})
        except ParameterOutOfRangeError:
            rejected.append(value)
    assert rejected == [], f"{parameter}: slider positions rejected = {rejected}"


def test_scale_factor_slider_floor_clears_the_exclusive_bound():
    """ADR-015: the Scale Factor slider floor is ``1.1``, not the bound ``1.0``.

    The bound stays ``(1.0, 10.0)`` with an **exclusive** low.  ``scale_perturbation``
    perturbs a column by ``f`` or ``1/f``, so ``f == 1.0`` is the identity — a run
    that perturbs nothing and still reports a stress score — and ``f == 0.0`` is the
    ``1/f`` division-by-zero path that made this bound exclusive in the first place.
    Both ends are genuinely degenerate, so the kernel bound is the correct side of
    this reconciliation and the widget was the side that had to move.

    This test pins the *resolution* rather than the inconsistency: it asserts the
    slider sits strictly inside the exclusive bound, and that the neighbouring
    ``1.0`` — which the widget can no longer produce — is still rejected.  If the
    bound ever becomes inclusive, the ADR needs revisiting, and this fails first.
    """
    low, high, _default, step = _slider_params()["Scale Factor"]
    assert get_config().stress_bounds.scale_factor == (1.0, 10.0)

    # The published floor clears the exclusive bound...
    assert low == 1.1, f"Scale Factor slider floor moved to {low}; see ADR-015"
    validate_stress_params("scale_perturbation", {"scale_factor": low})

    # ...and the degenerate value just below it is still refused, typed.
    with pytest.raises(ParameterOutOfRangeError) as excinfo:
        validate_stress_params("scale_perturbation", {"scale_factor": 1.0})
    assert excinfo.value.context["exclusive_low"] is True

    # The slider's own step can express nothing between the bound and the floor,
    # so no position was traded away to make room.
    assert (low - step) <= 1.0, "the 0.1 step no longer lands on the bound"


@pytest.mark.parametrize("label", sorted(SLIDER_LABELS))
def test_every_slider_default_is_accepted(label: str):
    """A slider's default must never be rejected — that would break the UI on
    first click with an error the user cannot act on."""
    _low, _high, default, _step = _slider_params()[label]
    parameter = SLIDER_LABELS[label]
    op = next(o for o, p in NUMERIC_PARAMETERS.items() if p == parameter)
    result = validate_stress_params(op, {parameter: default})
    assert result[parameter] == pytest.approx(float(default))


def test_no_slider_default_is_a_rejected_boundary():
    """Each slider's default sits strictly inside the declared bounds."""
    table = numeric_parameter_table()
    for label, parameter in SLIDER_LABELS.items():
        _low, _high, default, _step = _slider_params()[label]
        (bound_low, bound_high), exclusive_low, exclusive_high = table[parameter]
        assert bound_low <= default <= bound_high
        assert not (exclusive_low and default <= bound_low)
        assert not (exclusive_high and default >= bound_high)


def test_every_batch_config_value_is_accepted():
    """§4.3: the 19 hard-coded batch configurations are all inside the bounds."""
    batch = _ui_batch_params()
    seen = {k: sorted(v) for k, v in batch.items() if v}
    assert len(seen) == 6, f"batch parameters not found in the view: {seen}"
    for parameter, values in batch.items():
        op = next(o for o, p in NUMERIC_PARAMETERS.items() if p == parameter)
        for value in values:
            result = validate_stress_params(op, {parameter: value})
            assert result[parameter] == pytest.approx(value)


def test_every_batch_configuration_from_the_view_runs():
    """The strongest form: replay each literal batch config the view can build."""
    configs: list[dict[str, Any]] = []
    for node in ast.walk(_view_tree()):
        if not isinstance(node, ast.Dict):
            continue
        try:
            payload = _literal(node)
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, dict) and {"type", "params", "name"} <= set(payload):
            configs.append(payload)
    assert len(configs) == 19, f"expected 19 batch configs, found {len(configs)}"
    for config in configs:
        validated = validate_stress_params(config["type"], config["params"])
        assert set(validated) == set(accepted_keys(config["type"]))


# ── §4.3 rule 3: enum options ─────────────────────────────────────────────────


def test_every_ui_corruption_type_is_accepted():
    """§4.3: the UI's corruption option list equals ``CORRUPTION_TYPES``."""
    assert _ui_enum_options()["Corruption Type"] == list(CORRUPTION_TYPES)


def test_every_ui_shift_type_is_accepted():
    """§4.3: the UI's shift option list equals ``SHIFT_TYPES``."""
    assert _ui_enum_options()["Shift Type"] == list(SHIFT_TYPES)


@pytest.mark.parametrize("corruption_type", CORRUPTION_TYPES)
def test_each_corruption_type_is_accepted(corruption_type: str):
    """Exercised, not just compared: each must survive validation."""
    result = validate_stress_params(
        "feature_corruption",
        {"corruption_rate": 0.1, "corruption_type": corruption_type},
    )
    assert result["corruption_type"] == corruption_type


@pytest.mark.parametrize("shift_type", SHIFT_TYPES)
def test_each_shift_type_is_accepted(shift_type: str):
    result = validate_stress_params(
        "distribution_shift",
        {"shift_amount": 0.5, "shift_type": shift_type},
    )
    assert result["shift_type"] == shift_type


@pytest.mark.parametrize("bad", ["", "ZER0", "median", None, 3, "zero "])
def test_unknown_enum_value_is_rejected_with_the_accepted_set(bad: Any):
    """§4.3 rule 3: the message must name the accepted values so the user can
    act on it without reading the source."""
    with pytest.raises(UnsupportedStressTypeError) as excinfo:
        validate_stress_params(
            "feature_corruption", {"corruption_rate": 0.1, "corruption_type": bad}
        )
    message = str(excinfo.value)
    for accepted in CORRUPTION_TYPES:
        assert accepted in message


def test_unknown_shift_type_lists_the_accepted_set():
    """The same requirement on the other enum."""
    with pytest.raises(UnsupportedStressTypeError) as excinfo:
        validate_stress_params("distribution_shift", {"shift_amount": 0.5, "shift_type": "median"})
    assert excinfo.value.context["accepted"] == list(SHIFT_TYPES)


def test_enum_defaults_match_the_pre_remediation_keywords():
    """Omitting the enum yields the old keyword default, not an arbitrary one."""
    assert validate_stress_params("feature_corruption", {})["corruption_type"] == "zero"
    assert validate_stress_params("distribution_shift", {})["shift_type"] == "mean"


# ── §4.3 rule 4: unknown parameter keys ──────────────────────────────────────


def test_unknown_parameter_key_names_the_operator_and_its_keys():
    """§4.3 rule 4: a typo is reported with the accepted set, not ignored."""
    with pytest.raises(ValidationError) as excinfo:
        validate_stress_params("gaussian_noise", {"noise_levl": 0.1})
    message = str(excinfo.value)
    assert "gaussian_noise" in message
    assert "noise_level" in message
    assert excinfo.value.context["unknown"] == ["noise_levl"]
    assert excinfo.value.context["accepted"] == ["noise_level"]


def test_unknown_parameter_is_not_silently_dropped():
    """A silently ignored key would mean the caller thinks it took effect."""
    with pytest.raises(ValidationError):
        validate_stress_params("distribution_shift", {"shift_amount": 0.5, "shift_typo": "mean"})


def test_enum_key_on_an_operator_without_one_is_rejected():
    """``corruption_type`` means nothing to ``gaussian_noise``."""
    with pytest.raises(ValidationError) as excinfo:
        validate_stress_params("gaussian_noise", {"noise_level": 0.1, "corruption_type": "zero"})
    assert excinfo.value.context["unknown"] == ["corruption_type"]


# ── §4.3 bounds ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op,value",
    [
        ("gaussian_noise", -0.1),
        ("gaussian_noise", 5.1),
        ("uniform_noise", -0.01),
        ("uniform_noise", 1.1),
        ("feature_dropout", 1.0),
        ("feature_dropout", -0.1),
        ("feature_corruption", 1.0),
        ("feature_corruption", 2.0),
        ("scale_perturbation", 1.0),
        ("scale_perturbation", 0.9),
        ("scale_perturbation", 10.1),
        ("distribution_shift", -0.5),
        ("distribution_shift", 10.5),
    ],
)
def test_out_of_range_values_are_rejected(op: str, value: float):
    """§4.3 rule 2 with the exclusive flags applied."""
    with pytest.raises(ParameterOutOfRangeError) as excinfo:
        validate_stress_params(op, {NUMERIC_PARAMETERS[op]: value})
    assert excinfo.value.context["parameter"] == NUMERIC_PARAMETERS[op]
    assert excinfo.value.context["accepted_range"] == list(
        numeric_parameter_table()[NUMERIC_PARAMETERS[op]][0]
    )


def test_scale_factor_zero_is_deterministically_rejected_for_every_seed():
    """§4.4.7: the old ~50/50 ``ZeroDivisionError`` is gone.

    Validation precedes every RNG draw, so this cannot depend on the seed.  The
    loop exists to make that claim falsifiable: the pre-remediation code passed
    for about half of these seeds.
    """
    for _seed in range(64):
        with pytest.raises(ParameterOutOfRangeError):
            validate_stress_params("scale_perturbation", {"scale_factor": 0.0})


def test_dropout_rate_one_is_rejected_for_every_seed():
    """The other zero-everything case; also must not depend on the seed."""
    for _seed in range(16):
        with pytest.raises(ParameterOutOfRangeError):
            validate_stress_params("feature_dropout", {"dropout_rate": 1.0})


def test_inclusive_bounds_are_accepted():
    """A bound that is *not* exclusive must accept its own endpoint."""
    assert validate_stress_params("gaussian_noise", {"noise_level": 0.0})["noise_level"] == 0.0
    assert validate_stress_params("gaussian_noise", {"noise_level": 5.0})["noise_level"] == 5.0
    assert validate_stress_params("feature_dropout", {"dropout_rate": 0.0})["dropout_rate"] == 0.0
    assert validate_stress_params("uniform_noise", {"noise_range": 1.0})["noise_range"] == 1.0
    assert validate_stress_params("uniform_noise", {"noise_range": 0.0})["noise_range"] == 0.0
    assert (
        validate_stress_params("distribution_shift", {"shift_amount": 0.0})["shift_amount"] == 0.0
    )
    assert (
        validate_stress_params("feature_corruption", {"corruption_rate": 0.0})["corruption_rate"]
        == 0.0
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(bad: float):
    """NaN would otherwise slip past both ``<`` and ``>`` comparisons."""
    with pytest.raises(ParameterOutOfRangeError):
        validate_stress_params("gaussian_noise", {"noise_level": bad})


@pytest.mark.parametrize("bad", [None, [0.5], {"value": 0.5}, object(), {1, 2}, (0.5,)])
def test_non_numeric_values_are_rejected(bad: Any):
    """A value that is not a number at all must be a typed error, not a crash."""
    with pytest.raises((ValidationError, ParameterOutOfRangeError)):
        validate_stress_params("gaussian_noise", {"noise_level": bad})


def test_unparseable_strings_are_rejected():
    """``float()`` accepts numeric text only; ``"0.5x"`` must not become ``0.0``."""
    with pytest.raises((ValidationError, ParameterOutOfRangeError)):
        validate_stress_params("gaussian_noise", {"noise_level": "0.5x"})


def test_boolean_is_not_accepted_as_a_number():
    """``True`` is an ``int`` in Python; treating it as ``1.0`` would be a
    silent coercion of an unrelated type into a stress severity."""
    with pytest.raises(ValidationError) as excinfo:
        validate_stress_params("gaussian_noise", {"noise_level": True})
    assert excinfo.value.field == "noise_level"


def test_numeric_strings_are_coerced():
    """A JSON-sourced batch config may carry strings; coercion is deliberate."""
    assert validate_stress_params("gaussian_noise", {"noise_level": "0.25"})["noise_level"] == 0.25


# ── Normalisation ────────────────────────────────────────────────────────────


def test_validation_returns_only_the_accepted_keys():
    """The normalised mapping is the contract between validation and the kernel."""
    for op in NUMERIC_PARAMETERS:
        result = validate_stress_params(op, {})
        assert set(result) == set(accepted_keys(op))
        assert isinstance(result, dict)


def test_validation_does_not_mutate_its_argument():
    """A caller's dict must survive validation unchanged."""
    params = {"noise_level": 0.1}
    validate_stress_params("gaussian_noise", params)
    assert params == {"noise_level": 0.1}


def test_missing_params_mapping_is_treated_as_defaults():
    """``None`` and ``{}`` both mean "all defaults"."""
    assert validate_stress_params("gaussian_noise", None) == {"noise_level": 0.1}


def test_non_mapping_params_is_rejected():
    with pytest.raises(ValidationError):
        validate_stress_params("gaussian_noise", [0.1])  # type: ignore[arg-type]


def test_numpy_scalar_params_are_accepted():
    """Streamlit and pandas both hand back numpy scalars in places."""
    numpy = pytest.importorskip("numpy")
    assert (
        validate_stress_params("gaussian_noise", {"noise_level": numpy.float64(0.3)})["noise_level"]
        == 0.3
    )
    assert (
        validate_stress_params("gaussian_noise", {"noise_level": numpy.int64(1)})["noise_level"]
        == 1.0
    )


# ── §4.3 single-source invariant ─────────────────────────────────────────────


def test_resolve_bounds_defaults_to_the_configured_singleton():
    """``None`` must route through ``get_config()``, not to a private copy."""
    assert resolve_bounds(None) is get_config().stress_bounds


def test_explicit_bounds_are_used_verbatim():
    """An explicitly passed bounds object wins, for per-run overrides."""
    custom = StressBounds(noise_level=(0.0, 0.5))
    assert resolve_bounds(custom) is custom
    with pytest.raises(ParameterOutOfRangeError):
        validate_stress_params("gaussian_noise", {"noise_level": 0.9}, custom)
    assert validate_stress_params("gaussian_noise", {"noise_level": 0.4}, custom)


@pytest.mark.parametrize(
    "op,value",
    [("gaussian_noise", v) for v in (-1.0, 0.0, 0.5, 5.0, 5.1, 100.0)]
    + [("scale_perturbation", v) for v in (0.0, 0.9, 1.0, 2.0, 10.0, 10.1)]
    + [("feature_dropout", v) for v in (-0.1, 0.0, 0.5, 0.99, 1.0)]
    + [("uniform_noise", v) for v in (-0.1, 0.0, 1.0, 1.0001)]
    + [("distribution_shift", v) for v in (-1.0, 0.0, 10.0, 10.01)]
    + [("feature_corruption", v) for v in (-1.0, 0.0, 0.99, 1.0)],
)
def test_none_and_explicit_bounds_agree_on_every_decision(op: str, value: float):
    """§4.3: passing ``bounds=None`` and passing ``get_config().stress_bounds``
    must never disagree.  Swept with ``itertools.product`` below for the full
    cross-operator matrix."""
    params = {NUMERIC_PARAMETERS[op]: value}
    implicit_error: Exception | None = None
    explicit_error: Exception | None = None
    try:
        validate_stress_params(op, params)
    except Exception as exc:  # noqa: BLE001 - comparing two outcomes
        implicit_error = exc
    try:
        validate_stress_params(op, params, get_config().stress_bounds)
    except Exception as exc:  # noqa: BLE001
        explicit_error = exc

    assert type(implicit_error) is type(explicit_error)
    assert str(implicit_error) == str(explicit_error)


def test_full_bounds_matrix_agrees():
    """The complete sweep: every operator crossed with every parameter value.

    Cheap because validation is pure arithmetic; the point is that no value can
    be accepted on one path and rejected on the other.
    """
    sample_values = [
        -1.0,
        -0.001,
        0.0,
        0.001,
        0.1,
        0.5,
        0.999,
        1.0,
        1.001,
        2.0,
        10.0,
        10.001,
        100.0,
    ]
    bounds = get_config().stress_bounds
    for op, parameter in NUMERIC_PARAMETERS.items():
        for value in sample_values:
            decisions = []
            for candidate in (None, bounds):
                try:
                    validate_stress_params(op, {parameter: value}, candidate)
                    decisions.append("accept")
                except Exception as exc:  # noqa: BLE001
                    decisions.append(type(exc).__name__)
            assert decisions[0] == decisions[1], (
                f"{op}/{parameter}={value}: {decisions[0]} vs {decisions[1]}"
            )


def test_numeric_parameter_table_is_derived_not_hardcoded():
    """A tightened bound must show up in the table immediately."""
    table = numeric_parameter_table(StressBounds(noise_level=(0.0, 2.0)))
    assert table["noise_level"] == ((0.0, 2.0), False, False)
    assert table["scale_factor"] == ((1.0, 10.0), True, False)
    assert table["dropout_rate"] == ((0.0, 1.0), False, True)


# ── Registry integrity ───────────────────────────────────────────────────────


def test_registry_covers_exactly_the_six_operators():
    """ADR-010.6: the keys are the strings ``batch_stress_test`` dispatches on."""
    assert set(NUMERIC_PARAMETERS) == {
        "gaussian_noise",
        "uniform_noise",
        "feature_dropout",
        "feature_corruption",
        "scale_perturbation",
        "distribution_shift",
    }


def test_enum_parameters_reference_the_owned_tuples():
    """§2.3: the enums are *owned* by ``core.validation`` and shared, not copied."""
    for _op, (name, values) in ENUM_PARAMETERS.items():
        assert values is CORRUPTION_TYPES or values is SHIFT_TYPES
        assert name in {"corruption_type", "shift_type"}


def test_perturbations_reexports_the_same_enum_objects():
    """ADR-011.1: the documented public path keeps working *by identity*, so a
    future divergence is an import error rather than a silent inequality."""
    from core import perturbations

    assert perturbations.CORRUPTION_TYPES is CORRUPTION_TYPES
    assert perturbations.SHIFT_TYPES is SHIFT_TYPES


# ── validate_frame / ensure_finite / coerce_numeric_frame ─────────────────────


def test_validate_frame_accepts_a_normal_frame():
    pd = pytest.importorskip("pandas")
    validate_frame(pd.DataFrame({"a": [1.0, 2.0]}))
    validate_frame(pd.DataFrame({"a": [1.0]}), min_rows=1)  # inclusive boundary


def test_validate_frame_rejects_a_non_frame():
    with pytest.raises(ValidationError):
        validate_frame([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_validate_frame_rejects_zero_columns():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValidationError):
        validate_frame(pd.DataFrame(index=[0, 1]))


def test_validate_frame_rejects_too_few_rows():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValidationError) as excinfo:
        validate_frame(pd.DataFrame({"a": [1.0]}), min_rows=2)
    assert excinfo.value.context["rows"] == 1


def test_validate_frame_rejects_duplicate_column_labels():
    """Run-002 MAJOR-2: a repeated label used to reach ``builtins.ValueError``.

    ``df[label]`` returns a *frame* for a repeated label, so every
    column-addressing consumer of a feature frame silently built a 2-D block.
    The shape mismatch was reported by ``builtins.ValueError`` deep inside the
    reconstruction — an untyped error (conventions §3) naming shapes rather than
    the column the user has to rename.  The check lives here, in the one place
    the "usable as a feature matrix" contract is stated, so every current and
    future consumer inherits it.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["a", "b", "a"])
    with pytest.raises(ValidationError) as excinfo:
        validate_frame(frame)
    assert excinfo.value.context["duplicate_columns"] == ["a"]
    assert excinfo.value.context["column_count"] == 3
    assert excinfo.value.context["unique_column_count"] == 2
    assert "a" in str(excinfo.value)


def test_validate_frame_accepts_unique_but_similar_labels():
    """Only true repetition is rejected; near-collisions are legitimate."""
    pd = pytest.importorskip("pandas")
    validate_frame(pd.DataFrame([[1.0, 2.0]], columns=["a", "A"]))
    validate_frame(pd.DataFrame([[1.0, 2.0]], columns=["a", "a_1"]))


def test_ensure_finite_passes_and_returns_the_input():
    numpy = pytest.importorskip("numpy")
    values = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    assert ensure_finite(values, context="test") is values


def test_ensure_finite_rejects_nan_and_reports_rows():
    numpy = pytest.importorskip("numpy")
    values = numpy.array([[1.0, numpy.nan], [3.0, numpy.inf]])
    with pytest.raises(ValidationError) as excinfo:
        ensure_finite(values, context="operator 'x'")
    assert excinfo.value.context["affected_rows"] == 2
    assert excinfo.value.context["non_finite_count"] == 2
    assert "operator 'x'" in str(excinfo.value)


def test_ensure_finite_ignores_integer_arrays():
    """An integer array cannot be non-finite; no dtype error may escape."""
    numpy = pytest.importorskip("numpy")
    ensure_finite(numpy.array([[1, 2], [3, 4]]), context="test")


def test_coerce_numeric_frame_widens_to_float64():
    """§4.4.1: the widening that makes 5 of the 6 operators work at all."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"i": [1, 2, 3], "f": [0.5, 1.5, 2.5]})
    out = coerce_numeric_frame(frame)
    assert list(out.dtypes) == [numpy_dtype_float64(), numpy_dtype_float64()]
    assert out.index.equals(frame.index)
    assert list(out.columns) == ["i", "f"]


def numpy_dtype_float64():
    import numpy

    return numpy.dtype("float64")


def test_coerce_numeric_frame_names_the_offending_columns():
    """§4.4.1: a non-numeric column must be named, not silently no-op'd."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"ok": [1.0, 2.0], "name": ["a", "b"], "mixed": ["1", "2"]})
    with pytest.raises(ValidationError) as excinfo:
        coerce_numeric_frame(frame)
    assert "name" in excinfo.value.context["offending_columns"]


def test_coerce_numeric_frame_rejects_boolean_columns():
    """Booleans are not perturbable features; silently casting is a trap."""
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValidationError) as excinfo:
        coerce_numeric_frame(pd.DataFrame({"flag": [True, False]}))
    assert excinfo.value.context["offending_columns"] == ["flag"]


def test_coerce_numeric_frame_rejects_duplicate_column_labels():
    """Run-002 MAJOR-2, at the function that raised: no ``builtins.ValueError``.

    The reported failure was ``Shape of passed values is (4, 5), indices imply
    (4, 3)`` from the ``pd.DataFrame(data, columns=df.columns)``
    reconstruction — because ``df["a"]`` on a frame with two ``"a"`` columns
    returns a 2-D frame, so ``np.column_stack`` widened the block.  The offending
    column names are now reported instead.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(numpy_array_4x3(), columns=["a", "b", "a"])
    with pytest.raises(ValidationError) as excinfo:
        coerce_numeric_frame(frame)
    assert excinfo.value.context["duplicate_columns"] == ["a"]


def numpy_array_4x3():
    import numpy

    return numpy.arange(12, dtype="float64").reshape(4, 3)


def test_coerce_numeric_frame_does_not_mutate_the_input():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"i": [1, 2, 3]})
    before = frame.copy()
    coerce_numeric_frame(frame)
    pd.testing.assert_frame_equal(frame, before)
