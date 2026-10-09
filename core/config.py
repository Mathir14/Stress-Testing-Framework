"""Central configuration for every tunable in the framework.

Architecture reference: ``architecture.md`` §4.2, §2.3, ADR-005, ADR-009,
ADR-011.4.

Every default here reproduces the pre-remediation behaviour **exactly**.  These
numbers are the published results of the tool; changing one is a behaviour change
that requires a matching parity-test update *and* a new ADR.

Three rules govern :func:`get_config`, and each exists because its absence caused
a concrete failure mode:

1. The zero-arg call returns one process-wide cached singleton.
2. A non-``None`` ``overrides`` mapping returns a **new, uncached** config and
   never touches the singleton.  If overrides mutated the cache, a single
   ``get_config({"random_seed": 42})`` in one test would silently repoint every
   later test — including the parity fixtures — with no visible failure.
3. ``STF_*`` environment variables are read **only** on the zero-arg path, only
   at first call.

:func:`reset_config_cache` is the documented test seam that makes rule 3
observable; production code never calls it.

Import rules (architecture.md §2.3): rung 1 imports **no** project-local module.
``core.errors`` is therefore imported inside the function that needs it, so the
rung-1 assertion in ``tests/test_architecture.py`` holds at module scope.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ENV_LOG_LEVEL",
    "ENV_RANDOM_SEED",
    "FAILING_GRADE",
    "OVERRIDABLE_FIELDS",
    "StressBounds",
    "ReliabilityWeights",
    "ScoringPolicy",
    "ArtifactPolicy",
    "AppConfig",
    "get_config",
    "reset_config_cache",
]

logger = logging.getLogger(__name__)

#: The complete, closed set of environment variables this module reads.
#: Enumerated rather than resolved dynamically (conventions §6) so the surface
#: is auditable and asserted by ``tests/test_config.py``.
ENV_LOG_LEVEL = "STF_LOG_LEVEL"
ENV_RANDOM_SEED = "STF_RANDOM_SEED"

#: The implicit fallthrough grade below the lowest cutoff (ADR-009 §6).  Declared
#: rather than inlined so the "score < 50 -> F" rule cannot be lost during a
#: refactor of ``grade_cutoffs``.
FAILING_GRADE = "F"

#: Fields ``get_config(overrides=...)`` may replace.  The three sections are
#: absent on purpose: an override that replaced a whole section would silently
#: disagree with ``dataclasses.replace``'s nested-dataclass semantics, so the
#: accepted surface is scalars only and a section name is reported as an unknown
#: key (which is the better message).
OVERRIDABLE_FIELDS: tuple[str, ...] = ("log_level", "random_seed")


@dataclass(frozen=True)
class StressBounds:
    """Accepted range for each numeric stress parameter.

    Exclusive ends are declared in :mod:`core.validation` (single-source rule,
    architecture.md §2.3), not here, so this class stays a pure value object.

    Attributes:
        noise_level: Gaussian noise scale multiplier on the column std.
        noise_range: Uniform noise half-width as a fraction of the column range.
        dropout_rate: Per-cell Bernoulli masking rate.
        corruption_rate: Fraction of cells replaced per column.
        scale_factor: Per-column multiply/divide factor (lower bound exclusive).
        shift_amount: Mean/variance shift multiplier on the column statistics.
    """

    noise_level: tuple[float, float] = (0.0, 5.0)
    noise_range: tuple[float, float] = (0.0, 1.0)
    dropout_rate: tuple[float, float] = (0.0, 1.0)
    corruption_rate: tuple[float, float] = (0.0, 1.0)
    scale_factor: tuple[float, float] = (1.0, 10.0)
    shift_amount: tuple[float, float] = (0.0, 10.0)


@dataclass(frozen=True)
class ReliabilityWeights:
    """Component caps, grade boundaries and grade colours.

    Mirrors ``modules/reliability_module.py`` pre-remediation exactly: four
    components of 25 points, an ECE cap of 0.5, a relative-drop cap of 0.5, and
    grades at 90/80/70/60/50 with an implicit ``F`` below the last cutoff
    (ADR-005, ADR-009 §6).

    Attributes:
        max_perf: Maximum performance component points.
        max_cal: Maximum calibration component points.
        max_rob: Maximum robustness component points.
        max_conf: Maximum confidence component points.
        ece_cap: Expected-calibration-error value that scores zero.
        drop_cap: Relative performance drop that scores zero.
        grade_cutoffs: ``(minimum score, grade)`` pairs, highest first.
        grade_colours: ``(minimum score, hex colour)`` pairs, highest first.
    """

    max_perf: float = 25.0
    max_cal: float = 25.0
    max_rob: float = 25.0
    max_conf: float = 25.0
    ece_cap: float = 0.5
    drop_cap: float = 0.5
    grade_cutoffs: tuple[tuple[float, str], ...] = (
        (90.0, "A+"),
        (80.0, "A"),
        (70.0, "B"),
        (60.0, "C"),
        (50.0, "D"),
    )
    grade_colours: tuple[tuple[float, str], ...] = (
        (90.0, "#4CAF50"),
        (80.0, "#8BC34A"),
        (70.0, "#FFEB3B"),
        (60.0, "#FF9800"),
        (50.0, "#FF5722"),
        (0.0, "#C0392B"),
    )

    def resolve_grade(self, score: float) -> tuple[str, str]:
        """Resolve a score to its ``(grade, colour)``.

        Args:
            score: Overall reliability score on the 0-100 scale.

        Returns:
            tuple: The letter grade and its hex colour.  A score below the
            lowest cutoff yields ``FAILING_GRADE`` and the last colour.
        """
        value = float(score)
        grade = FAILING_GRADE
        for threshold, letter in self.grade_cutoffs:
            if value >= threshold:
                grade = letter
                break
        return grade, self._colour_for_threshold(self._threshold_for(grade))

    def colour_for(self, grade: str) -> str:
        """Return the colour for a letter grade.

        Args:
            grade: A letter grade, e.g. ``"B"``.

        Returns:
            str: The hex colour, falling back to the failing colour for an
            unknown grade.
        """
        return self._colour_for_threshold(self._threshold_for(grade))

    def _threshold_for(self, grade: str) -> float:
        for threshold, letter in self.grade_cutoffs:
            if letter == grade:
                return threshold
        return 0.0

    def _colour_for_threshold(self, threshold: float) -> str:
        for bound, colour in self.grade_colours:
            if threshold >= bound:
                return colour
        return self.grade_colours[-1][1]


@dataclass(frozen=True)
class ScoringPolicy:
    """The one owner of "how good is this model" policy (ADR-018).

    Reliability weights (``ReliabilityWeights``) own the component caps and the
    letter-grade bands.  This section owns everything that must not be
    restated per module: the fixed-denominator composite weights, the
    zero-credit-for-missing policy, the minimum measured-components gate, the
    ECE quality bands and the robustness severity bands.

    Attributes:
        composite_weights: ``(component, weight)`` pairs for the comparison
            composite; weights sum to ``1.0``.  Components are ``performance``,
            ``robustness`` and ``calibration``.
        missing_component_points: Points awarded to a component with no
            evidence.  Binding ADR-018 value is ``0.0`` — a missing component
            scores zero with **no redistribution**.
        min_measured_components: Fewest measured components a model needs
            before it is ``rated`` and eligible for ranking.
        ece_quality: Ascending ``(strict_upper_bound, label)`` pairs.
        severity_bands: Ascending ``(strict_upper_bound, label)`` pairs applied
            to a percentage performance drop.
        quality_fallthrough: Label when an ECE exceeds every band.
        severity_fallthrough: Label when a drop exceeds every band.
    """

    composite_weights: tuple[tuple[str, float], ...] = (
        ("performance", 0.5),
        ("robustness", 0.25),
        ("calibration", 0.25),
    )
    missing_component_points: float = 0.0
    min_measured_components: int = 2
    ece_quality: tuple[tuple[float, str], ...] = (
        (0.03, "Excellent"),
        (0.07, "Good"),
        (0.15, "Moderate"),
    )
    severity_bands: tuple[tuple[float, str], ...] = (
        (5.0, "Low"),
        (15.0, "Medium"),
        (30.0, "High"),
    )
    quality_fallthrough: str = "Poor"
    severity_fallthrough: str = "Critical"

    def resolve_ece_quality(self, ece: float) -> str:
        """Classify an expected-calibration-error value.

        Args:
            ece: Expected calibration error (lower is better).

        Returns:
            str: The first band label whose strict upper bound the value is
            below, else :attr:`quality_fallthrough`.
        """
        value = float(ece)
        for threshold, label in self.ece_quality:
            if value < threshold:
                return label
        return self.quality_fallthrough

    def resolve_severity(self, drop_pct: float) -> str:
        """Classify a percentage performance drop.

        Args:
            drop_pct: Performance drop as a percentage (higher is worse).

        Returns:
            str: The first band label whose strict upper bound the value is
            below, else :attr:`severity_fallthrough`.
        """
        value = float(drop_pct)
        for threshold, label in self.severity_bands:
            if value < threshold:
                return label
        return self.severity_fallthrough

    def weight_for(self, component: str) -> float:
        """Return the composite weight for ``component`` (0.0 when absent).

        Args:
            component: One of ``performance``, ``robustness`` or ``calibration``.

        Returns:
            float: The configured weight, or ``0.0`` for an unknown component.
        """
        for name, weight in self.composite_weights:
            if name == component:
                return float(weight)
        return 0.0


@dataclass(frozen=True)
class ArtifactPolicy:
    """Filename, containment and attestation policy for model artifacts.

    Attributes:
        root_dir: Directory the store owns exclusively.
        allowed_suffixes: Suffixes the store will write or read.
        filename_pattern: Regex applied with ``re.fullmatch``.  ``\\Z``-anchored
            on purpose: ``$`` also matches *before* a trailing newline, so an
            unanchored pattern would admit ``"model.pkl\\n"`` (ADR-009 §1).
        max_filename_length: Cap that keeps ``ENAMETOOLONG`` out of the trust
            path (ADR-009 §2).
        max_size_bytes: Size cap enforced on save and on load (ADR-010.2).
        require_attestation: Whether loading requires a manifest entry.
    """

    root_dir: Path = Path("saved_models")
    allowed_suffixes: tuple[str, ...] = (".pkl",)
    filename_pattern: str = r"[A-Za-z0-9][A-Za-z0-9._-]*\.pkl\Z"
    max_filename_length: int = 128
    max_size_bytes: int = 256 * 1024 * 1024
    require_attestation: bool = True


@dataclass(frozen=True)
class AppConfig:
    """The composed configuration.

    The three sections have **no** defaults: an ``AppConfig`` that could be
    built half-configured is exactly the silent-divergence failure this class
    exists to prevent.

    Attributes:
        stress_bounds: Accepted parameter ranges for the stress kernel.
        reliability: Component caps and grade boundaries.
        artifacts: Model-artifact policy.
        scoring: The unified scoring policy (ADR-018).  Defaulted so an
            ``AppConfig`` built before ADR-018 keeps its published scores for
            every field the policy did not change.
        log_level: Root log level name.
        random_seed: Seed for the per-run generator, or ``None`` for unseeded.
    """

    stress_bounds: StressBounds
    reliability: ReliabilityWeights
    artifacts: ArtifactPolicy
    scoring: ScoringPolicy = ScoringPolicy()
    log_level: str = "INFO"
    random_seed: int | None = None


_singleton: AppConfig | None = None


def get_config(overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Return the process configuration.

    Args:
        overrides: Optional scalar overrides.  A non-``None`` mapping returns a
            **new** config and never mutates, populates or invalidates the
            singleton.  Accepted keys are exactly :data:`OVERRIDABLE_FIELDS`;
            an unknown key is an error rather than a silently ignored typo.

    Returns:
        AppConfig: The cached singleton when ``overrides`` is ``None``,
        otherwise a freshly built config carrying the overrides.

    Raises:
        ConfigurationError: An override key is unknown, a value is a nested
            mapping, or a value cannot be coerced to the field's declared type.
    """
    if overrides is not None:
        return _apply_overrides(overrides)
    global _singleton
    if _singleton is None:
        _singleton = _from_environment()
    return _singleton


def reset_config_cache() -> None:
    """Discard the cached singleton.

    Test seam only (ADR-011.4 rule 4).  Without it the ``STF_*`` path is
    unobservable, because the environment is read exactly once per process.
    Production code must never call this.
    """
    global _singleton
    _singleton = None


def _apply_overrides(overrides: Mapping[str, Any]) -> AppConfig:
    from core.errors import ConfigurationError

    if not isinstance(overrides, Mapping):
        raise ConfigurationError(
            f"Configuration overrides must be a mapping, got {type(overrides).__name__}.",
            context={"received_type": type(overrides).__name__},
        )

    unknown = [key for key in overrides if key not in OVERRIDABLE_FIELDS]
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration override(s) {', '.join(sorted(unknown))}. "
            f"Accepted overrides are: {', '.join(OVERRIDABLE_FIELDS)}. Section "
            "names are not overridable.",
            context={"unknown": sorted(unknown), "accepted": list(OVERRIDABLE_FIELDS)},
        )

    nested = sorted(key for key, value in overrides.items() if isinstance(value, Mapping))
    if nested:
        raise ConfigurationError(
            f"Nested configuration overrides are not supported for "
            f"{', '.join(nested)}. Pass scalar values; sections are built by "
            "dataclasses.replace, which does not merge nested dataclasses.",
            context={"key": nested[0], "accepted": list(OVERRIDABLE_FIELDS)},
        )

    coerced: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "log_level":
            coerced[key] = _coerce_str(key, value)
        else:
            coerced[key] = _coerce_optional_int(key, value)
    return replace(
        AppConfig(
            stress_bounds=StressBounds(),
            reliability=ReliabilityWeights(),
            artifacts=ArtifactPolicy(),
        ),
        **coerced,
    )


def _coerce_str(key: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        from core.errors import ConfigurationError

        raise ConfigurationError(
            f"Configuration override {key!r} must be a string, got bool.",
            context={"key": key, "expected": "str"},
        )
    try:
        return str(value)
    except Exception as exc:  # pragma: no cover - str() is total for our inputs
        from core.errors import ConfigurationError

        raise ConfigurationError(
            f"Configuration override {key!r} must be a string.",
            context={"key": key, "expected": "str"},
        ) from exc


def _coerce_optional_int(key: str, value: Any) -> int | None:
    from core.errors import ConfigurationError

    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigurationError(
            f"Configuration override {key!r} must be an int or None, got bool.",
            context={"key": key, "expected": "int", "value": repr(value)},
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ConfigurationError(
                f"Configuration override {key!r} must be a whole number, got {value!r}.",
                context={"key": key, "expected": "int", "value": repr(value)},
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            raise ConfigurationError(
                f"Configuration override {key!r} must be an integer string, got {value!r}.",
                context={"key": key, "expected": "int", "value": repr(value)},
            ) from None
    raise ConfigurationError(
        f"Configuration override {key!r} must be an int or None, got {type(value).__name__}.",
        context={"key": key, "expected": "int", "value": repr(value)},
    )


def _from_environment() -> AppConfig:
    log_level = os.environ.get(ENV_LOG_LEVEL, "INFO")
    raw_seed = os.environ.get(ENV_RANDOM_SEED)
    seed: int | None = None
    if raw_seed is not None:
        try:
            seed = int(raw_seed)
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: the random seed must be an integer; using an "
                "unseeded generator instead.",
                ENV_RANDOM_SEED,
                raw_seed,
            )
    logger.info("Resolved configuration: log_level=%s random_seed=%s", log_level, seed)
    return AppConfig(
        stress_bounds=StressBounds(),
        reliability=ReliabilityWeights(),
        artifacts=ArtifactPolicy(),
        log_level=str(log_level),
        random_seed=seed,
    )
