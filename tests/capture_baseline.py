#!/usr/bin/env python3
"""Capture and verify the frozen reliability baseline.

Architecture reference: ``roadmap.md`` Wave 0 (capture) and Wave 6 (verify),
``architecture.md`` §10.2.

Usage::

    python -m tests.capture_baseline           # (re)capture the fixture
    python -m tests.capture_baseline --check   # verify, exit 1 on any drift

This is the milestone's **only bit-exact numeric freeze** (ADR-010.5).
Reliability scoring is pure arithmetic on caller-supplied floats — no RNG, no
clock, no library-dependent accumulation — so an exact freeze is both possible
and worth having.  Perturbations are deliberately *not* frozen: a vectorised
implementation draws randomness in a different order from the old per-column
loop, so there parity means "same formula, same distributional parameters, seeded
self-consistency".

The interpreter that produced the fixture is recorded **inside** it
(ADR-010.9): a fixture captured on 3.14 and consumed on 3.11 is not comparable,
and the devcontainer runs 3.11 while the remediation host runs 3.14.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.reliability_module import ReliabilityScorer  # noqa: E402

#: Bumped only when the fixture layout changes incompatibly.  The manifest
#: schema is frozen the same way (architecture.md §4.5.12).
SCHEMA_VERSION = 1

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reliability_baseline.json"

#: Class arities swept for every case.  ``n_classes`` feeds ``log2(n)``, the
#: entropy normaliser, so a change to the arity handling moves the confidence
#: component — and nothing else in the pipeline would notice.
N_CLASS_CASES: tuple[int, ...] = (2, 3, 5)

#: Representative inputs, one per reachable code path:
#:
#: * ``all_components``    — every input supplied, the ordinary case;
#: * ``perfect``           — every component at its maximum (the 100.0 ceiling);
#: * ``worst``             — every component clipped to zero (``ece == ece_cap``,
#:                           ``avg_drop == drop_cap``, maximum entropy/HCE);
#: * ``zero_performance``  — a falsy ``0.0``, which must score 0 rather than
#:                           falling back to a default;
#: * ``missing_all``       — no inputs, so every component is *missing* and takes
#:                           ``missing_component_points`` (``0.0``) — never a
#:                           midpoint (ADR-018 fixed denominator);
#: * ``partial``           — only accuracy, which is *not* enough for
#:                           Performance (both accuracy and f1 are required);
#: * ``confidence_only``   — entropy without hce_rate, i.e. one missing input
#:                           to a two-input component.
#:
#: The four ``*_drop`` cases freeze the run-015 C1/C1b deltas and were added when
#: the fixture was regenerated for ADR-021: a negative drop must clamp to the
#: robustness *cap* (not inflate past it), a drop beyond the cap must floor at
#: zero, a missing drop must score the missing-component floor rather than a
#: full-capacity zero-drop, and a non-finite drop must floor at zero rather than
#: escape as ``NaN``.
SCORING_CASES: dict[str, dict[str, Any]] = {
    "all_components": {
        "accuracy": 0.85,
        "f1": 0.83,
        "ece": 0.05,
        "avg_drop": 0.10,
        "avg_entropy": 0.4,
        "hce_rate": 0.05,
    },
    "perfect": {
        "accuracy": 1.0,
        "f1": 1.0,
        "ece": 0.0,
        "avg_drop": 0.0,
        "avg_entropy": 0.0,
        "hce_rate": 0.0,
    },
    "worst": {
        "accuracy": 0.0,
        "f1": 0.0,
        "ece": 0.5,
        "avg_drop": 0.5,
        "avg_entropy": 8.0,
        "hce_rate": 1.0,
    },
    "zero_performance": {
        "accuracy": 0.0,
        "f1": 0.0,
    },
    "missing_all": {},
    "partial": {
        "accuracy": 0.83,
    },
    "confidence_only": {
        "avg_entropy": 0.4,
    },
    # ── run-015 C1/C1b deltas (ADR-021) ──────────────────────────────────────
    "negative_drop": {
        "accuracy": 0.8,
        "f1": 0.8,
        "ece": 0.05,
        "avg_drop": -0.4,  # looks like an improvement; robustness clamps to cap
        "avg_entropy": 0.3,
        "hce_rate": 0.0,
    },
    "drop_over_cap": {
        "accuracy": 0.8,
        "f1": 0.8,
        "ece": 0.05,
        "avg_drop": 5.0,  # beyond drop_cap; robustness floors at zero
        "avg_entropy": 0.3,
        "hce_rate": 0.0,
    },
    "missing_drop": {
        "accuracy": 0.8,
        "f1": 0.8,
        "ece": 0.05,
        # avg_drop deliberately omitted: missing evidence is None -> 0 pts,
        # NOT a zero-drop full-capacity robustness score.
        "avg_entropy": 0.3,
        "hce_rate": 0.0,
    },
    "nan_drop": {
        "accuracy": 0.8,
        "f1": 0.8,
        "ece": 0.05,
        "avg_drop": float("nan"),  # non-finite floors to zero, never escapes as NaN
        "avg_entropy": 0.3,
        "hce_rate": 0.0,
    },
}

#: Scores probed for the grade table.  Each cutoff, each side of it, the
#: ceiling, the floor and a negative value so the ``< 50 -> "F"`` fallthrough is
#: exercised rather than assumed.
GRADE_PROBES: tuple[float, ...] = (
    100.0,
    95.0,
    90.0,
    89.999,
    85.0,
    80.0,
    79.999,
    75.0,
    70.0,
    69.999,
    65.0,
    60.0,
    59.999,
    55.0,
    50.0,
    49.999,
    25.0,
    0.0,
    -10.0,
)

#: Fields of a scoring case that must match bit-exactly.
_FROZEN_FIELDS = (
    "performance",
    "calibration",
    "robustness",
    "confidence",
    "total",
    "grade",
    "colour",
    "available_components",
    "missing_components",
)


def _interpreter_record() -> dict[str, str]:
    """Describe the interpreter the fixture was produced on."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system().lower(),
    }


def capture() -> dict[str, Any]:
    """Build the baseline payload from the current implementation.

    Returns:
        dict: The payload, in the shape written to
        ``tests/fixtures/reliability_baseline.json``.
    """
    scorer = ReliabilityScorer()

    grades: dict[str, str] = {}
    grade_colours: dict[str, str] = {}
    for probe in GRADE_PROBES:
        label = repr(probe)
        grade, colour = scorer._grade(probe)
        grades[label] = grade
        grade_colours[label] = colour

    cases: dict[str, Any] = {}
    for name, kwargs in SCORING_CASES.items():
        for n_classes in N_CLASS_CASES:
            key = name if n_classes == 2 else f"{name}__nc{n_classes}"
            result = scorer.score_model(name, n_classes=n_classes, **kwargs)
            cases[key] = {field: result[field] for field in _FROZEN_FIELDS}

    return {
        "schema_version": SCHEMA_VERSION,
        "interpreter": _interpreter_record(),
        "maxima": {
            "MAX_PERF": scorer.MAX_PERF,
            "MAX_CAL": scorer.MAX_CAL,
            "MAX_ROB": scorer.MAX_ROB,
            "MAX_CONF": scorer.MAX_CONF,
        },
        "grades": grades,
        "grade_colours": grade_colours,
        "cases": cases,
    }


def write(payload: dict[str, Any]) -> Path:
    """Persist the payload to the fixture path.

    Args:
        payload: The baseline payload.

    Returns:
        Path: Where it was written.
    """
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ``allow_nan=False`` makes a non-finite score a hard capture failure rather
    # than a fixture carrying the non-standard ``NaN`` token: the parity freeze
    # must never bless a value Python's own JSON dialect accepts but the format
    # does not (run-015 review).
    FIXTURE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return FIXTURE_PATH


def compare(payload: dict[str, Any]) -> list[str]:
    """Diff the recomputed payload against the recorded fixture.

    ``interpreter`` is compared too, but as provenance rather than as a gate on
    the *running* machine: a devcontainer on 3.11 must be able to verify a
    fixture captured on 3.14, so a mismatch is reported as a warning line rather
    than a failure.

    Args:
        payload: The freshly computed payload.

    Returns:
        list[str]: Human-readable differences; empty when there are none.
    """
    if not FIXTURE_PATH.is_file():
        return [f"fixture {FIXTURE_PATH} does not exist"]

    recorded = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    differences: list[str] = []

    if recorded.get("schema_version") != payload["schema_version"]:
        differences.append(
            f"schema_version: recorded {recorded.get('schema_version')!r}, "
            f"current {payload['schema_version']!r}"
        )

    if recorded.get("interpreter") != payload["interpreter"]:
        differences.append(
            f"NOTE interpreter differs — recorded {recorded.get('interpreter')}, "
            f"current {payload['interpreter']}; fixture was captured on a "
            "different interpreter"
        )

    for section in ("maxima", "grades", "grade_colours"):
        recorded_section = recorded.get(section, {})
        current_section = payload[section]
        if set(recorded_section) != set(current_section):
            differences.append(
                f"{section}: keys differ — recorded "
                f"{sorted(recorded_section)}, current {sorted(current_section)}"
            )
            continue
        for key in sorted(current_section):
            if recorded_section[key] != current_section[key]:
                differences.append(
                    f"{section}[{key}]: recorded {recorded_section[key]!r}, "
                    f"current {current_section[key]!r}"
                )

    recorded_cases = recorded.get("cases", {})
    if set(recorded_cases) != set(payload["cases"]):
        differences.append(
            f"cases: recorded {sorted(recorded_cases)}, current {sorted(payload['cases'])}"
        )
    for key in sorted(set(recorded_cases) & set(payload["cases"])):
        for field in _FROZEN_FIELDS:
            if recorded_cases[key].get(field) != payload["cases"][key][field]:
                differences.append(
                    f"cases[{key}].{field}: recorded "
                    f"{recorded_cases[key].get(field)!r}, current "
                    f"{payload['cases'][key][field]!r}"
                )
    return differences


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: ``0`` on success, ``1`` when ``--check`` finds drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the recorded fixture instead of rewriting it",
    )
    args = parser.parse_args(argv)

    payload = capture()

    if args.check:
        differences = compare(payload)
        if differences:
            for line in differences:
                print(line)
            print(f"BASELINE DRIFT: {len(differences)} difference(s)", file=sys.stderr)
            return 1
        print(f"Baseline matches {FIXTURE_PATH} bit-exactly.")
        return 0

    path = write(payload)
    print(f"Wrote {path}")
    print(
        f"  interpreter: {payload['interpreter']['python_implementation']} "
        f"{payload['interpreter']['python_version']}"
    )
    print(f"  cases: {len(payload['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
