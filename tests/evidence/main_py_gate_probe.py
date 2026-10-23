"""Mutation probe for the §2.2 top-level-module gate (run-003, reviewer C1–C3/M4).

Records that the gates are load-bearing rather than decorative.  A gate that
cannot fail is the failure mode ADR-010's rationale warns about, and this gate
was previously self-certifying: widening the test's own ``ROOT_MODULES`` constant
was enough to admit a new root-level module while architecture.md §2.2 was left
untouched.

Updated for the two-module tree (ADR-012): the CLI entry point is *declared*, so
probes 1–4 use a third, **undeclared** root module (``extra.py``) — the shape that
must still be rejected — and probe 5 covers what §2.2 cannot judge, namely that
the declared ``main.py`` is a real CLI rather than the no-op stub.

Probes 3 and 4 declare the third module in **all three** places the gate family
requires (``ROOT_MODULES``, the §2.2 layout block, and the ADR-citation scan
list).  Two findings from the run-003 verification pass, both in this file rather
than in the tree, are recorded in ``_declare_in_citation_scan``:

1. Probe 3 reported a false FAIL because it simulated a two-edit declaration,
   which no longer satisfies reviewer M2's fourth guard.
2. Probe 4 had become **vacuous**: its red came from the citation-scan guard, not
   from the third-party dependency matrix it names.  Verified by deleting that
   matrix and finding the scenario green.  Probe 6 now pins the attribution, so
   probe 4 cannot silently pass for the wrong reason again.

Run against a scratch copy of the tree (it writes files):

    python tests/evidence/main_py_gate_probe.py /tmp/probe
    python tests/evidence/main_py_gate_probe.py --check-anchors

``--check-anchors`` (ADR-032) is the side-effect-free self-check: it asserts that
every declared ``(rel, old, new)`` source edit still resolves against the live
tree and still differs from its replacement, then exits non-zero on any miss.  It
reads files only, so the T0 gate
``test_evidence_probe_anchors_resolve_against_the_tree`` can invoke it on a bare
interpreter; the scratch-copy mutation runs above stay the on-demand evidence
path.

Each probe prints PASS/FAIL for the expected outcome.  A FAIL here means the gate
stopped protecting something it claims to protect.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# tests/evidence/<this file> -> repo root is two levels up.
SOURCE = Path(__file__).resolve().parents[2]

#: The historical no-op stub: answers 0 to every argument, including invalid ones.
STUB = '''#!/usr/bin/env python3
"""CLI stub."""
import sys


def main(argv=None):
    """Main CLI entry point."""
    argv = argv or sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print("Stress Testing Framework")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

UNDECLARED = "extra.py"

# ── Declared source edits (ADR-032) ──────────────────────────────────────────
#
# The four tree-editing helpers below each perform a single ``old → new`` text
# replacement against a scratch copy.  Declaring them here lets ``--check-anchors``
# verify each ``old`` still resolves against the live tree without writing
# anything, so a reword in ``test_architecture.py`` / ``architecture.md`` fails
# the T0 freshness gate instead of silently disabling a probe.

ROOT_MODULES_ANCHOR: str = 'ROOT_MODULES: tuple[str, ...] = (COMPOSITION_ROOT, "main.py")'

SPEC_LAYOUT_ANCHOR: str = "app.py                       # composition root, <=200 lines\n"

CITATION_SCAN_ANCHOR: str = "_ADR_CITATION_SOURCES: tuple[tuple[str, ...], ...] = (\n"

DEPENDENCY_GATE_ANCHOR: str = "def test_root_module_respects_the_matrix("

#: The ADR-032 anchor-freshness gate validates *this probe's* declared anchors
#: against the live tree.  Every scenario below deliberately mutates the very
#: files those anchors name (``tests/test_architecture.py`` and
#: ``.ai/project/architecture.md``), so the gate is red *by construction* in
#: every scratch copy and would misattribute the probe's reds — probe 3 would
#: report the legitimate declaration path as broken, and probe 6 would lose its
#: proof that probe 4's red comes from the dependency matrix.  The anchor gate is
#: therefore deselected from the scratch-suite runs; the real-tree invocation is
#: the normal suite's job (``test_evidence_probe_anchors_resolve_against_the_tree``).
DESELECTED_NODEIDS: tuple[str, ...] = (
    "tests/test_architecture.py::test_evidence_probe_anchors_resolve_against_the_tree",
    "tests/test_architecture.py::test_gate_evidence_bundle_is_tracked",
)

DECLARED_EDITS: list[tuple[str, str, str]] = [
    (
        "tests/test_architecture.py",
        ROOT_MODULES_ANCHOR,
        f'ROOT_MODULES: tuple[str, ...] = (COMPOSITION_ROOT, "main.py", "{UNDECLARED}")',
    ),
    (
        ".ai/project/architecture.md",
        SPEC_LAYOUT_ANCHOR,
        "app.py                       # composition root, <=200 lines\n"
        f"{UNDECLARED:<31}# declared third root module (probe)\n",
    ),
    (
        "tests/test_architecture.py",
        CITATION_SCAN_ANCHOR,
        CITATION_SCAN_ANCHOR + f'    ("{UNDECLARED}",),\n',
    ),
    (
        "tests/test_architecture.py",
        DEPENDENCY_GATE_ANCHOR,
        "def _disabled_test_root_module_respects_the_matrix(",
    ),
]


def check_anchors() -> int:
    """Verify every declared edit still resolves against the live tree.

    Side-effect-free: reads files only.  Exits non-zero (via the return value) if
    any ``old`` is absent from its file or is identical to its ``new``.

    Returns:
        int: ``0`` when every anchor resolves and differs; ``1`` otherwise.
    """
    problems: list[str] = []
    for rel, old, new in DECLARED_EDITS:
        path = SOURCE / rel
        if not path.is_file():
            problems.append(f"{rel}: file is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if old not in text:
            problems.append(f"{rel}: anchor not found")
        if old == new:
            problems.append(f"{rel}: anchor is identical to its replacement")
    if problems:
        print("ANCHOR CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"ALL {len(DECLARED_EDITS)} ANCHORS RESOLVE")
    return 0


def _run(work: Path, target: str = "tests/test_architecture.py") -> tuple[int, str]:
    args = [sys.executable, "-m", "pytest", target, "-q"]
    if target == "tests/test_architecture.py":
        for node in DESELECTED_NODEIDS:
            args += ["--deselect", node]
    proc = subprocess.run(
        args,
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _widen_constant(work: Path) -> None:
    path = work / "tests" / "test_architecture.py"
    text = path.read_text()
    text = text.replace(
        ROOT_MODULES_ANCHOR,
        f'ROOT_MODULES: tuple[str, ...] = (COMPOSITION_ROOT, "main.py", "{UNDECLARED}")',
        1,
    )
    path.write_text(text)


def _declare_in_spec(work: Path) -> None:
    path = work / ".ai" / "project" / "architecture.md"
    text = path.read_text()
    text = text.replace(
        SPEC_LAYOUT_ANCHOR,
        "app.py                       # composition root, <=200 lines\n"
        f"{UNDECLARED:<31}# declared third root module (probe)\n",
        1,
    )
    path.write_text(text)


def _declare_in_citation_scan(work: Path) -> None:
    """Add the probe's root module to ``_ADR_CITATION_SOURCES``.

    Reviewer M2's fourth guard in the same family (``test_the_adr_citation_scan_
    covers_every_root_module``) asserts the ADR-citation scan and ``ROOT_MODULES``
    agree in **both** directions, so declaring a root module properly is now a
    *three*-edit operation, not a two-edit one: the test constant, the §2.2
    layout block, and the citation-scan list.  Probe 3 still simulated only the
    first two, so it reported a false FAIL against a correct tree.

    This edit is also what keeps probe 4 honest.  Without it, probe 4's red came
    from the citation-scan guard rather than from the third-party dependency
    matrix it claims to exercise -- verified: with the scan list updated and
    ``test_root_module_respects_the_matrix`` deleted, probe 4's scenario is green.
    A probe that passes when the control it names is removed is the failure mode
    ADR-010's rationale warns about, so the scan list is updated first and the
    dependency gate is left to be the only remaining control.
    """
    path = work / "tests" / "test_architecture.py"
    text = path.read_text()
    anchor = CITATION_SCAN_ANCHOR
    assert anchor in text, "the citation-scan anchor moved; this probe needs updating"
    text = text.replace(anchor, anchor + f'    ("{UNDECLARED}",),\n', 1)
    path.write_text(text)


def _delete_dependency_gate(work: Path) -> None:
    """Remove the third-party root-module dependency gate (reviewer M4's gate).

    Renaming the test so pytest never collects it simulates the gate being
    absent, which is the only way to show that probe 4's red is attributable to
    the dependency matrix rather than to some other guard in the same file.
    """
    path = work / "tests" / "test_architecture.py"
    text = path.read_text()
    marker = DEPENDENCY_GATE_ANCHOR
    assert marker in text, "the dependency-gate test was renamed; update this probe"
    path.write_text(text.replace(marker, "def _disabled_test_root_module_respects_the_matrix(", 1))


def probe(
    label: str,
    mutate,
    *,
    expect_fail: bool,
    scratch: Path,
    target: str = "tests/test_architecture.py",
) -> bool:
    work = scratch / re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(
        SOURCE,
        work,
        ignore=shutil.ignore_patterns(
            ".venv", ".git", "__pycache__", ".forge", ".pytest_cache", ".ruff_cache",
        ),
    )
    mutate(work)
    code, out = _run(work, target)
    failed = code != 0
    ok = failed is expect_fail
    verdict = "PASS" if ok else "FAIL"
    detail = "suite red" if failed else "suite green"
    print(f"[{verdict}] {label}: {detail} (expected {'red' if expect_fail else 'green'})")
    if not ok:
        print(out[-2000:])
    return ok


def main(argv: list[str]) -> int:
    if "--check-anchors" in argv:
        return check_anchors()
    base = Path(argv[0]) if argv else Path(tempfile.mkdtemp(prefix="stf-probe-"))
    base.mkdir(parents=True, exist_ok=True)
    results = [
        probe(
            "1 undeclared root module added, gate untouched",
            lambda w: (w / UNDECLARED).write_text(STUB),
            expect_fail=True,
            scratch=base,
        ),
        probe(
            "2 constant widened, spec untouched (reviewer C2/C3)",
            lambda w: ((w / UNDECLARED).write_text(STUB), _widen_constant(w)),
            expect_fail=True,
            scratch=base,
        ),
        probe(
            "3 constant, spec and citation scan all updated",
            lambda w: ((w / UNDECLARED).write_text(STUB), _widen_constant(w),
                       _declare_in_spec(w), _declare_in_citation_scan(w)),
            expect_fail=False,
            scratch=base,
        ),
        probe(
            "4 declared module imports third-party (M4 dead-gate probe)",
            lambda w: ((w / UNDECLARED).write_text(
                STUB.replace("import sys", "import sys\nimport numpy as np")),
                _widen_constant(w), _declare_in_spec(w), _declare_in_citation_scan(w)),
            expect_fail=True,
            scratch=base,
        ),
        probe(
            "5 declared main.py reverted to the no-op stub",
            lambda w: (w / "main.py").write_text(STUB),
            expect_fail=True,
            scratch=base,
            target="tests/test_cli.py",
        ),
        probe(
            "6 probe 4 bites the dependency matrix, not the citation guard",
            lambda w: ((w / UNDECLARED).write_text(
                STUB.replace("import sys", "import sys\nimport numpy as np")),
                _widen_constant(w), _declare_in_spec(w), _declare_in_citation_scan(w),
                _delete_dependency_gate(w)),
            expect_fail=False,
            scratch=base,
        ),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} probes behaved as expected")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
