"""Configuration contract tests for ``core.config`` (architecture.md §10.2).

**Tier T0.**  No scientific imports — ``core.config`` is rung 1 of the §2.3
ladder and must stay that way (ADR-011.4).

The load-bearing behaviour is the *cache*: rule 1 says the zero-arg call returns
one process-wide singleton, rule 2 says an overrides mapping returns a new
uncached object and leaves the singleton untouched, and rule 3 says the
``STF_*`` environment variables are read only on the zero-arg path and only at
first call.  A regression in any of the three silently repoints every later test
and corrupts the parity fixtures with no visible failure, which is exactly why
they are pinned here rather than left to the parity tests to discover.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from core.config import (
    ENV_LOG_LEVEL,
    ENV_RANDOM_SEED,
    FAILING_GRADE,
    AppConfig,
    ArtifactPolicy,
    ReliabilityWeights,
    ScoringPolicy,
    StressBounds,
    get_config,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def clean_config_cache(monkeypatch):
    """Give every test a pristine singleton and no inherited ``STF_*`` values."""
    for name in (ENV_LOG_LEVEL, ENV_RANDOM_SEED):
        monkeypatch.delenv(name, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


# ── §4.2 defaults reproduce pre-remediation behaviour ────────────────────────


def test_stress_bounds_defaults_are_unchanged():
    """§4.3: every range is a superset of the UI sliders and batch configs.

    The numbers are the contract, not an implementation detail — a widening or
    narrowing is a behaviour change requiring a parity update and a new ADR.
    """
    bounds = StressBounds()
    assert bounds.noise_level == (0.0, 5.0)
    assert bounds.noise_range == (0.0, 1.0)
    assert bounds.dropout_rate == (0.0, 1.0)
    assert bounds.corruption_rate == (0.0, 1.0)
    assert bounds.scale_factor == (1.0, 10.0)
    assert bounds.shift_amount == (0.0, 10.0)


def test_reliability_weights_defaults_are_unchanged():
    """ADR-005: the four component caps and the five grade cutoffs."""
    weights = ReliabilityWeights()
    assert weights.max_perf == 25.0
    assert weights.max_cal == 25.0
    assert weights.max_rob == 25.0
    assert weights.max_conf == 25.0
    assert weights.ece_cap == 0.5
    assert weights.drop_cap == 0.5
    assert weights.grade_cutoffs == (
        (90.0, "A+"),
        (80.0, "A"),
        (70.0, "B"),
        (60.0, "C"),
        (50.0, "D"),
    )


def test_failing_grade_fallthrough_is_declared():
    """ADR-009 §6: below the last cutoff there is an explicit ``F``, not an
    ``IndexError`` and not a silent absence."""
    assert FAILING_GRADE == "F"
    assert ReliabilityWeights().grade_cutoffs[-1][0] == 50.0


def test_artifact_policy_defaults_are_unchanged():
    """§4.5: ``.pkl`` only, 128-char cap, 256 MiB size cap, attestation on."""
    policy = ArtifactPolicy()
    assert policy.root_dir == Path("saved_models")
    assert policy.allowed_suffixes == (".pkl",)
    assert policy.max_filename_length == 128
    assert policy.max_size_bytes == 256 * 1024 * 1024
    assert policy.require_attestation is True


def test_filename_pattern_is_z_anchored():
    """ADR-009 §1: ``\\Z``, not ``$``.

    ``$`` also matches immediately before a trailing newline, which would admit
    ``"model.pkl\\n"`` — a bypass in a boundary whose value is that it cannot be
    bypassed.
    """
    pattern = ArtifactPolicy().filename_pattern
    assert pattern.endswith(r"\.pkl\Z")
    assert "$" not in pattern


# ── Immutability ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "instance",
    [
        StressBounds(),
        ReliabilityWeights(),
        ArtifactPolicy(),
        AppConfig(
            stress_bounds=StressBounds(),
            reliability=ReliabilityWeights(),
            artifacts=ArtifactPolicy(),
        ),
    ],
    ids=["StressBounds", "ReliabilityWeights", "ArtifactPolicy", "AppConfig"],
)
def test_config_dataclasses_are_frozen(instance):
    """§4.2: every config object is immutable, so no layer can mutate it."""
    assert dataclasses.is_dataclass(instance)
    field = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, getattr(instance, field))


def test_app_config_requires_its_three_sections():
    """The sections have no defaults: an incomplete config cannot be built."""
    with pytest.raises(TypeError):
        AppConfig()  # type: ignore[call-arg]


# ── Rule 1: the cached singleton ──────────────────────────────────────────────


def test_zero_arg_call_returns_the_identical_object():
    """ADR-011.4 rule 1: two zero-arg calls return *the same* object."""
    assert get_config() is get_config()


def test_singleton_reflects_the_defaults():
    """A pristine environment yields the documented defaults."""
    config = get_config()
    assert config.log_level == "INFO"
    assert config.random_seed is None
    assert config.stress_bounds == StressBounds()
    assert config.reliability == ReliabilityWeights()
    assert config.artifacts == ArtifactPolicy()


# ── Rule 3: environment overrides ─────────────────────────────────────────────


def test_environment_overrides_apply_on_the_zero_arg_path(monkeypatch):
    """ADR-011.4 rule 3: ``STF_*`` are folded in on first call."""
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")
    monkeypatch.setenv(ENV_RANDOM_SEED, "1234")
    reset_config_cache()
    config = get_config()
    assert config.log_level == "DEBUG"
    assert config.random_seed == 1234


def test_environment_is_read_only_at_first_call(monkeypatch):
    """A later ``os.environ`` change cannot repoint the live singleton."""
    assert get_config().log_level == "INFO"
    monkeypatch.setenv(ENV_LOG_LEVEL, "ERROR")
    assert get_config().log_level == "INFO", (
        "the zero-arg path must not re-read the environment once cached"
    )


def test_non_integer_seed_is_ignored_with_a_warning(monkeypatch, caplog):
    """A malformed seed degrades to the default rather than raising at import."""
    import logging

    monkeypatch.setenv(ENV_RANDOM_SEED, "not-a-number")
    reset_config_cache()
    with caplog.at_level(logging.WARNING):
        config = get_config()
    assert config.random_seed is None
    assert any("random" in record.getMessage().lower() for record in caplog.records)


def test_seed_environment_variable_must_be_an_integer(monkeypatch):
    """The coercion is explicit, not a silent ``int`` of ``None``."""
    monkeypatch.setenv(ENV_RANDOM_SEED, "42")
    reset_config_cache()
    assert get_config().random_seed == 42


# ── Rule 2: overrides return a new, uncached object ───────────────────────────


def test_overrides_return_a_new_object():
    """Rule 2: an overrides mapping never hands back the singleton."""
    override = get_config({"log_level": "DEBUG"})
    assert override is not get_config()
    assert override is not get_config({"log_level": "DEBUG"})
    assert override.log_level == "DEBUG"


def test_overrides_do_not_mutate_the_singleton():
    """The failure mode this rule exists to prevent."""
    singleton = get_config()
    override = get_config({"log_level": "DEBUG", "random_seed": 7})
    assert override.log_level == "DEBUG"
    assert override.random_seed == 7
    assert singleton.log_level == "INFO"
    assert singleton.random_seed is None
    assert get_config() is singleton


def test_overrides_do_not_pollute_the_cache():
    """A second zero-arg call must still see the singleton, not the override."""
    get_config({"log_level": "ERROR"})
    assert get_config().log_level == "INFO"


def test_overrides_preserve_every_unspecified_section():
    """Only the named fields change; the rest are the defaults, not garbage."""
    override = get_config({"random_seed": 3})
    assert override.stress_bounds == StressBounds()
    assert override.reliability == ReliabilityWeights()
    assert override.artifacts == ArtifactPolicy()
    assert override.log_level == "INFO"


def test_empty_overrides_mapping_behaves_like_the_defaults():
    """``{}`` is falsy: it is not an unknown key and not a nested mapping."""
    override = get_config({})
    assert override == get_config()


def test_explicit_none_seed_is_allowed():
    """``random_seed=None`` is a meaningful value, not an omitted one."""
    assert get_config({"random_seed": None}).random_seed is None


def test_numeric_strings_are_coerced():
    """The declared coercer runs, so ``"42"`` is accepted for an int field."""
    assert get_config({"random_seed": "42"}).random_seed == 42


def test_numeric_string_log_level_is_rejected():
    """``"5"`` coerces to a string but is not a recognised logging level name,
    so the ``__post_init__`` gate (ADR-023) rejects it."""
    from core.errors import ConfigurationError

    for value in ("5", 5, "BANANA"):
        with pytest.raises(ConfigurationError) as excinfo:
            get_config({"log_level": value})
        assert excinfo.value.context["field"] == "log_level"


# ── Rejected overrides ────────────────────────────────────────────────────────


def test_unknown_override_key_is_rejected():
    """An unknown key is an error, never a silently ignored typo."""
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        get_config({"log_leveld": "DEBUG"})
    assert "log_leveld" in str(excinfo.value)
    assert excinfo.value.context["accepted"] == ["log_level", "random_seed"]


def test_nested_override_path_is_rejected():
    """§10.2 requires one pinned answer for nested paths; the pinned answer is
    *rejected*, never silently ignored — ignoring it would leave the caller
    believing a section had been replaced when it had not.

    A nested value under a **known** key hits the dedicated nested-mapping
    branch; a nested value under a **section** name is caught one step earlier by
    the unknown-key check, which is the better message.  Both are pinned so
    neither route can start succeeding silently.
    """
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        get_config({"log_level": {"level": "DEBUG"}})
    assert "nested" in str(excinfo.value).lower()
    assert excinfo.value.context["key"] == "log_level"
    assert excinfo.value.context["accepted"] == ["log_level", "random_seed"]


def test_section_names_are_not_settable_as_scalars():
    """The three section fields are not reachable through ``overrides``.

    Reached as *unknown keys* rather than as nested mappings: the accepted-key
    set is enumerated, so a section name can never be mistaken for a scalar.
    """
    from core.errors import ConfigurationError

    for section in ("stress_bounds", "reliability", "artifacts"):
        with pytest.raises(ConfigurationError) as excinfo:
            get_config({section: {"max_perf": 10.0}})
        assert section in str(excinfo.value)
        assert excinfo.value.context["unknown"] == [section]

    for section, value in (
        ("stress_bounds", StressBounds()),
        ("reliability", ReliabilityWeights()),
        ("artifacts", ArtifactPolicy()),
    ):
        with pytest.raises(ConfigurationError):
            get_config({section: value})


def test_uncoercible_value_is_rejected():
    """A value the declared coercer cannot handle raises, with the expectation."""
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        get_config({"random_seed": object()})
    assert excinfo.value.context["expected"] == "int"


# ── ADR-023: self-validation on every path ────────────────────────────────────


def test_negative_seed_override_is_rejected():
    """ADR-023 rule 1: a negative seed crashes ``np.random.default_rng`` on
    every run, so the override path refuses it."""
    from core.errors import ConfigurationError

    for value in (-1, "-1", -42.0):
        with pytest.raises(ConfigurationError) as excinfo:
            get_config({"random_seed": value})
        assert excinfo.value.context["field"] == "random_seed"
        assert "non-negative" in str(excinfo.value)


def test_negative_seed_environment_is_rejected(monkeypatch):
    """ADR-023 rule 2: the same negative seed is refused on the ``STF_*`` path."""
    from core.errors import ConfigurationError

    monkeypatch.setenv(ENV_RANDOM_SEED, "-1")
    reset_config_cache()
    with pytest.raises(ConfigurationError) as excinfo:
        get_config()
    assert excinfo.value.context["field"] == "random_seed"
    assert excinfo.value.context["value"] == -1


def test_unknown_log_level_environment_is_rejected(monkeypatch):
    """A bad ``STF_LOG_LEVEL`` raises instead of silently falling back to INFO."""
    from core.errors import ConfigurationError

    monkeypatch.setenv(ENV_LOG_LEVEL, "verbose")
    reset_config_cache()
    with pytest.raises(ConfigurationError) as excinfo:
        get_config()
    assert excinfo.value.context["field"] == "log_level"


def test_bounds_validate_ordering_sign_and_fullness():
    """ADR-023: a reversed, negative, or wrong-shaped stress range is refused
    at construction, before any listener can trust it."""
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        StressBounds(noise_level=(5.0, 0.0))  # reversed
    with pytest.raises(ConfigurationError):
        StressBounds(noise_level=(-1.0, 5.0))  # negative lower bound
    with pytest.raises(ConfigurationError):
        StressBounds(noise_level=(0.0,))  # wrong arity
    with pytest.raises(ConfigurationError):
        StressBounds(noise_level=("low", "high"))  # not numbers


def test_reliability_weights_validate_caps_and_tables():
    """ADR-023: a non-positive cap or a malformed grade table is refused."""
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        ReliabilityWeights(max_perf=0.0)
    with pytest.raises(ConfigurationError):
        ReliabilityWeights(ece_cap=-0.1)
    with pytest.raises(ConfigurationError):
        ReliabilityWeights(grade_cutoffs=())
    with pytest.raises(ConfigurationError):
        ReliabilityWeights(grade_cutoffs=((50.0, "D"), (90.0, "A+")))  # ascending
    with pytest.raises(ConfigurationError):
        ReliabilityWeights(grade_colours=((90.0, "green"),))  # not hex


def test_scoring_policy_validate_weights_and_bands():
    """ADR-023: weights that do not sum to 1.0 or a non-positive gate is refused."""
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        ScoringPolicy(composite_weights=())
    with pytest.raises(ConfigurationError):
        ScoringPolicy(
            composite_weights=(("performance", 0.4), ("robustness", 0.4)),
        )  # sums to 0.8
    with pytest.raises(ConfigurationError):
        ScoringPolicy(min_measured_components=0)
    with pytest.raises(ConfigurationError):
        ScoringPolicy(ece_quality=((0.07, "Good"), (0.03, "Excellent")))  # descending
    with pytest.raises(ConfigurationError):
        ScoringPolicy(quality_fallthrough="")


def test_valid_custom_instances_build():
    """The validators must not reject legitimate customisations."""
    assert StressBounds(noise_level=(0.0, 0.2)).noise_level == (0.0, 0.2)
    assert ReliabilityWeights(max_perf=20.0).max_perf == 20.0
    policy = ScoringPolicy(
        composite_weights=(
            ("performance", 1.0),
            ("robustness", 0.0),
            ("calibration", 0.0),
        )
    )
    assert policy.weight_for("performance") == 1.0


# ── Cache reset ───────────────────────────────────────────────────────────────


def test_reset_config_cache_rebuilds_the_singleton():
    """The documented test seam actually rebuilds rather than returning stale."""
    first = get_config()
    reset_config_cache()
    second = get_config()
    assert first is not second
    assert first == second, "a rebuild from the same defaults must be equal"


def test_reset_config_cache_rereads_the_environment(monkeypatch):
    """Combined with rule 3, this is how the env path becomes observable."""
    assert get_config().log_level == "INFO"
    monkeypatch.setenv(ENV_LOG_LEVEL, "WARNING")
    reset_config_cache()
    assert get_config().log_level == "WARNING"


# ── §2.3 rung-1 purity ────────────────────────────────────────────────────────


def test_config_imports_nothing_project_local_at_module_scope():
    """§2.3: rung 1 imports no project-local module; the one ``core.errors``
    import must stay function-local.  See
    ``test_architecture.py::test_config_deferred_import_is_cycle_free``
    for the full assertion."""
    path = Path(__file__).resolve().parent.parent / "core" / "config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        offenders = [
            n for n in names if n.split(".", 1)[0] in {"core", "modules", "utils", "views"}
        ]
        assert not offenders, f"core/config.py imports {offenders} at module scope"


def test_config_declares_exactly_two_env_variables():
    """conventions §6 bans dynamic lookup: the env surface is enumerated."""
    path = Path(__file__).resolve().parent.parent / "core" / "config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("STF_")
    }
    assert found == {ENV_LOG_LEVEL, ENV_RANDOM_SEED}


# ── ADR-034 ScoringPolicy recommendation and robustness bands ─────────────────


def test_scoring_policy_resolve_recommendation_at_defaults():
    """ADR-034: resolve_recommendation matches the 80/60/40/Low scale at defaults."""
    from core.config import ScoringPolicy

    policy = ScoringPolicy()
    assert policy.resolve_recommendation(95.0) == "Excellent"
    assert policy.resolve_recommendation(80.0) == "Excellent"
    assert policy.resolve_recommendation(79.9) == "Good"
    assert policy.resolve_recommendation(60.0) == "Good"
    assert policy.resolve_recommendation(59.9) == "Moderate"
    assert policy.resolve_recommendation(40.0) == "Moderate"
    assert policy.resolve_recommendation(39.9) == "Low"
    assert policy.resolve_recommendation(0.0) == "Low"


def test_scoring_policy_resolve_robustness_at_defaults():
    """ADR-034: resolve_robustness matches the <10/<20/Low scale at defaults."""
    from core.config import ScoringPolicy

    policy = ScoringPolicy()
    assert policy.resolve_robustness(5.0) == "High"
    assert policy.resolve_robustness(9.99) == "High"
    assert policy.resolve_robustness(10.0) == "Medium"
    assert policy.resolve_robustness(19.99) == "Medium"
    assert policy.resolve_robustness(20.0) == "Low"
    assert policy.resolve_robustness(50.0) == "Low"


def test_scoring_policy_rejects_invalid_recommendation_bands():
    """ADR-034: recommendation bands must be non-empty and strictly descending."""
    from core.config import ScoringPolicy
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="recommendation_bands"):
        ScoringPolicy(recommendation_bands=())

    with pytest.raises(ConfigurationError, match="strictly descending"):
        ScoringPolicy(recommendation_bands=((60.0, "Good"), (80.0, "Excellent")))

    with pytest.raises(ConfigurationError, match="non-empty string"):
        ScoringPolicy(recommendation_fallthrough="")


def test_scoring_policy_rejects_invalid_robustness_bands():
    """ADR-034: robustness bands must be strictly ascending."""
    from core.config import ScoringPolicy
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="robustness_bands"):
        ScoringPolicy(robustness_bands=())

    with pytest.raises(ConfigurationError, match="strictly ascending"):
        ScoringPolicy(robustness_bands=((20.0, "Medium"), (10.0, "High")))

    with pytest.raises(ConfigurationError, match="non-empty string"):
        ScoringPolicy(robustness_fallthrough="")


def test_scoring_policy_rejects_invalid_thresholds():
    """ADR-034: vulnerability and stability thresholds must be bounded numbers."""
    from core.config import ScoringPolicy
    from core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="key_vulnerability_drop_threshold"):
        ScoringPolicy(key_vulnerability_drop_threshold=-1.0)

    with pytest.raises(ConfigurationError, match="low_prediction_stability_threshold"):
        ScoringPolicy(low_prediction_stability_threshold=1.5)

