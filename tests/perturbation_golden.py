"""Capture and verify bit-exact perturbation outputs across a refactor.

Architecture reference: ADR-016.4.  This is **not** the pre-remediation parity
freeze of ADR-010.5 — those numbers are deliberately *not* frozen and no test
asserts them.  This freezes the **post-Wave-2** kernel so that a refactor of
``core/perturbations.py`` / ``core/validation.py`` that claims to be
behaviour-preserving can be proven to be exactly that.

Run ``python -m tests.perturbation_golden --capture`` to rewrite the fixture,
``--check`` (the default) to verify against it.

The fixture is also read by
:func:`tests.test_perturbations.test_perturbation_output_is_bit_identical`, so
``--check`` is not a separate chore that can drift from the suite: both spellings
call :func:`snapshot` and :func:`diff` here, and a maintainer who regenerates the
fixture has regenerated the gate too (ADR-016.7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

FIXTURE = pathlib.Path(__file__).with_name("fixtures") / "perturbation_golden.json"

#: Every operator with the parameter sets that exercise every branch.
CASES: dict[str, list[dict[str, Any]]] = {
    "gaussian_noise": [{}, {"noise_level": 0.75}],
    "uniform_noise": [{}, {"noise_range": 0.5}],
    "feature_dropout": [{}, {"dropout_rate": 0.4}],
    "feature_corruption": [
        {},
        {"corruption_rate": 0.25, "corruption_type": "zero"},
        {"corruption_rate": 0.25, "corruption_type": "mean"},
        {"corruption_rate": 0.25, "corruption_type": "random"},
        {"corruption_rate": 0.25, "corruption_type": "extreme"},
    ],
    "scale_perturbation": [{}, {"scale_factor": 2.5}],
    "distribution_shift": [
        {},
        {"shift_amount": 1.5, "shift_type": "mean"},
        {"shift_amount": 1.5, "shift_type": "variance"},
    ],
}

SEEDS = (0, 12345)


def _shapes() -> dict[str, Any]:
    """The five input shapes the refactor must preserve, plus two rejections."""
    rng = np.random.default_rng(20260906)
    return {
        "float_frame": pd.DataFrame(rng.normal(size=(37, 9)), columns=[f"f{i}" for i in range(9)]),
        "mixed_frame": pd.DataFrame(
            {**{f"i{i}": rng.integers(0, 50, size=37) for i in range(4)},
             **{f"f{i}": rng.normal(size=37) for i in range(5)}}
        ),
        "int_frame": pd.DataFrame(rng.integers(0, 50, size=(37, 6)), columns=list("abcdef")),
        "single_col_frame": pd.DataFrame({"only": rng.normal(size=41)}),
        "zero_variance_frame": pd.DataFrame(
            {"const": np.full(30, 7.0), "vary": rng.normal(size=30)}
        ),
        "float_array": rng.normal(size=(37, 9)),
        "int_array": rng.integers(0, 50, size=(37, 9)),
        "fortran_array": np.asfortranarray(rng.normal(size=(37, 9))),
    }


REJECTIONS: tuple[tuple[str, Any], ...] = (
    ("str_frame", pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})),
    ("bool_frame", pd.DataFrame({"flag": [True, False]})),
    ("duplicate_labels", pd.DataFrame(np.ones((4, 3)), columns=["a", "a", "b"])),
    ("series", pd.Series([1.0, 2.0])),
    ("one_dimensional", np.ones(4)),
)


def _digest(value: Any) -> str:
    """Hash the raw bytes of an output array in a layout-stable way."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot() -> dict[str, Any]:
    """Compute the full digest table from the live implementation."""
    from core.perturbations import apply_perturbation  # noqa: PLC0415

    record: dict[str, Any] = {"shapes": {}, "rejections": {}}
    for name, frame in _shapes().items():
        entries: dict[str, str] = {}
        for op, param_sets in CASES.items():
            for index, params in enumerate(param_sets):
                for seed in SEEDS:
                    key = f"{op}|{index}|{seed}"
                    out = apply_perturbation(
                        frame, op, dict(params), rng=np.random.default_rng(seed)
                    )
                    entries[key] = _digest(out)
                    # The container contract is part of the freeze too.
                    if isinstance(frame, pd.DataFrame):
                        entries[key + "|cols"] = "|".join(map(str, out.columns))
                        entries[key + "|idx"] = "|".join(map(str, out.index))
                    entries[key + "|dtype"] = str(np.asarray(out.values if hasattr(out, "values") else out).dtype)
        record["shapes"][name] = entries

    from core.errors import ValidationError  # noqa: PLC0415

    for name, bad in REJECTIONS:
        try:
            apply_perturbation(bad, "gaussian_noise", {}, rng=np.random.default_rng(0))
        except ValidationError as exc:
            record["rejections"][name] = type(exc).__name__
        else:  # pragma: no cover - would be a regression
            record["rejections"][name] = "NO ERROR RAISED"
    return record


def diff(frozen: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Describe every way ``live`` departs from ``frozen``, one string per fact.

    Extracted so the CLI and :func:`tests.test_perturbations.
    test_perturbation_output_is_bit_identical` share **one** comparison.  The
    alternative — a second ``==`` spelled out in the test — is how a fixture
    stops being a gate: the test asserts something slightly weaker than the tool
    a maintainer actually runs, and neither one catches a kernel change.

    Compared in both directions and section by section, so a digest that
    *disappears* is reported as loudly as one that changes.  A plain
    ``frozen == live`` catches both too, but says nothing about *which*, and a
    900-line boolean diff in a CI log is a gate people learn to skip.

    Args:
        frozen: The record read from :data:`FIXTURE`.
        live: The record computed by :func:`snapshot`.

    Returns:
        list[str]: Human-readable differences; empty when the two agree.
    """
    problems: list[str] = []
    for section in ("shapes", "rejections"):
        names = sorted(set(frozen[section]) | set(live[section]))
        for name in names:
            before = frozen[section].get(name)
            after = live[section].get(name)
            if before == after:
                continue
            if section == "rejections":
                problems.append(f"  CHANGED {name}: {before} -> {after}")
                continue
            if before is None or after is None:
                problems.append(f"  {'REMOVED' if before else 'ADDED'} shape {name}")
                continue
            changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
            problems.append(f"  CHANGED {name}: {len(changed)} digest(s)")
            problems.extend(f"    {key}" for key in changed[:8])
            if len(changed) > 8:
                problems.append(f"    ... and {len(changed) - 8} more")
    return problems


def load_fixture() -> dict[str, Any]:
    """Return the frozen record.

    Raises:
        AssertionError: if the fixture file is missing — which would otherwise
            turn the gate green for the wrong reason, since ``snapshot()``
            alone can never disagree with a file that is not read.
    """
    if not FIXTURE.exists():
        raise AssertionError(
            f"missing fixture {FIXTURE}; regenerate it deliberately with "
            "`python -m tests.perturbation_golden --capture` and say in the "
            "commit message why the kernel's output legitimately changed"
        )
    return json.loads(FIXTURE.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="rewrite the fixture")
    parser.add_argument("--check", action="store_true", help="verify against it (default)")
    args = parser.parse_args(argv)

    live = snapshot()
    if args.capture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(live, indent=1, sort_keys=True) + "\n")
        total = sum(len(v) for v in live["shapes"].values())
        print(f"captured {total} digests + {len(live['rejections'])} rejections -> {FIXTURE}")
        return 0

    frozen = load_fixture()
    problems = diff(frozen, live)
    if not problems:
        total = sum(len(v) for v in live["shapes"].values())
        print(f"OK: {total} digests + {len(live['rejections'])} rejections match the frozen kernel")
        return 0
    for line in problems:
        print(line)
    print("FAIL: perturbation output is no longer bit-identical")
    return 1


if __name__ == "__main__":
    sys.exit(main())
