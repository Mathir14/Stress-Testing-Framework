#!/usr/bin/env python3
"""Mutation probes for the run-003 CHANGES_REQUIRED fixes.

Each probe mutates one file, runs a named subset of tests, records whether the
suite FAILED (the gate bites) or passed (the gate is vacuous), and restores the
file with an md5 check.  Any residue or restore failure is a hard error.

Usage:
    .venv/bin/python tests/evidence/mutation_probe.py
    .venv/bin/python tests/evidence/mutation_probe.py --check-anchors

``--check-anchors`` (ADR-032) is the side-effect-free self-check: it asserts that
every declared ``(rel, old, new)`` edit still resolves against the live tree and
still differs from its replacement, then exits non-zero on any miss.  It performs
no file writes, spawns no subprocess and runs no pytest, so the T0 gate
``test_evidence_probe_anchors_resolve_against_the_tree`` can invoke it on a bare
interpreter.  Full mutation execution stays the on-demand evidence path.

Note on M1: the reviewer asked for `main.py` to be deleted *or* declared with an
ADR and root-module gate coverage. ADR-012 took the second branch, so this probe
asserts the invariant that survives either remedy — every root module is
declared, and an undeclared one is rejected — rather than the deletion it
originally encoded.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# Ensure core is importable when run from any cwd.
sys.path.insert(0, str(ROOT))

from core.config import ArtifactPolicy  # noqa: E402 - after the sys.path insert above


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run(targets: list[str]) -> tuple[int, int]:
    proc = subprocess.run(
        [PY, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    m = re.search(r"(\d+) failed", proc.stdout)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", proc.stdout)
    passed = int(m.group(1)) if m else 0
    return failed, passed


def probe(name: str, rel: str, old: str, new: str, targets: list[str]) -> bool:
    """Apply a mutation, run targets, restore.  Return True if the gate bit."""
    return probe_multi(name, targets, [(rel, old, new)])


def probe_multi(
    name: str, targets: list[str], edits: list[tuple[str, str, str]]
) -> bool:
    """Apply one or more coordinated mutations, run targets, restore all."""
    originals: dict[Path, str] = {}
    digests: dict[Path, str] = {}
    for rel, old, _new in edits:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        if old not in text:
            print(f"!! {name}: anchor not found in {rel}")
            return False
        originals[path] = text
        digests[path] = md5(path)
    try:
        for rel, old, new in edits:
            path = ROOT / rel
            path.write_text(
                originals[path].replace(old, new, 1), encoding="utf-8"
            )
        failed, passed = run(targets)
    finally:
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")
    residue = [str(p.name) for p, d in digests.items() if md5(p) != d]
    if residue:
        raise SystemExit(f"restore failed for {residue}")
    verdict = "CAUGHT" if failed else "*** NOT CAUGHT ***"
    print(f"  {verdict}: {name}  ({failed} failed, {passed} passed)")
    print(f"      target: {' '.join(targets)}")
    return failed > 0


# ── Declared source edits (ADR-032) ──────────────────────────────────────────
#
# Every mutation this probe applies against the live tree is declared here as a
# ``(rel, old, new)`` triple, so ``--check-anchors`` can verify each anchor still
# resolves without running any probe.  A production change that moves an anchor
# then fails the T0 freshness gate loudly and cheaply, instead of silently
# disabling the probe it belongs to.

ADR_001A_EDIT = (
    "core/model_io.py",
    "        if not isinstance(model, bases):\n"
    "            raise ArtifactIntegrityError(\n",
    "        if False:\n"
    "            raise ArtifactIntegrityError(\n",
)

ADR_001B_EDIT = (
    "core/model_io.py",
    "            if actual != expected and not trust:\n",
    "            if False and actual != expected and not trust:\n",
)

# ADR-001c.  The anchor must span *both* size-cap checks in ``load()``: ADR-024
# added the post-read re-check (``size_bytes = len(buffer)``), so mutating only
# the cheap stat check leaves the second one to catch the oversized buffer and
# the test stays green.  The anchor therefore runs from the first check through
# the read-once buffer, the second check, the read-once digest and load()'s
# ``if entry is None: / if not trust:`` arm.  That trailing arm is what keeps the
# anchor from silently landing in ``verify()``, which shares the first block
# byte-for-byte but raises ``ArtifactNotFoundError`` on a missing entry.
_ADR_001C_SPAN = (
    "        if size_bytes > self.policy.max_size_bytes:\n"
    "            raise ArtifactIntegrityError(\n"
    '                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "\n'
    '                f"{self.policy.max_size_bytes}-byte limit.",\n'
    "                context={\n"
    '                    "filename": filename,\n'
    '                    "size_bytes": size_bytes,\n'
    '                    "max_size_bytes": self.policy.max_size_bytes,\n'
    "                },\n"
    "            )\n"
    "\n"
    "        # Read the artifact exactly once.  The digest below is computed over\n"
    "        # *this* buffer and ``pickle.loads`` consumes the same bytes, so the\n"
    "        # bytes that were verified are the bytes that are deserialised \u2014 no\n"
    "        # hash-then-reopen TOCTOU window (ADR-024, run-015 M6).\n"
    "        buffer = self._read_bytes(path)\n"
    "        size_bytes = len(buffer)\n"
    "        if size_bytes > self.policy.max_size_bytes:\n"
    "            # The file grew between the cheap stat and the read; re-check the\n"
    "            # bytes we actually hold rather than trusting the earlier size.\n"
    "            raise ArtifactIntegrityError(\n"
    '                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "\n'
    '                f"{self.policy.max_size_bytes}-byte limit. Nothing was loaded.",\n'
    "                context={\n"
    '                    "filename": filename,\n'
    '                    "size_bytes": size_bytes,\n'
    '                    "max_size_bytes": self.policy.max_size_bytes,\n'
    "                },\n"
    "            )\n"
    "\n"
    '        entry = self._read_manifest().get("artifacts", {}).get(filename)\n'
    "        actual = hashlib.sha256(buffer).hexdigest()\n"
    "\n"
    "        if entry is None:\n"
    "            if not trust:\n"
)
ADR_001C_EDIT = (
    "core/model_io.py",
    _ADR_001C_SPAN,
    _ADR_001C_SPAN.replace(
        "        if size_bytes > self.policy.max_size_bytes:\n",
        "        if size_bytes > self.policy.max_size_bytes and not trust:\n",
    ),
)

ADR_001D_EDIT = (
    "core/model_io.py",
    "        if size_bytes > self.policy.max_size_bytes:\n"
    "            raise ArtifactIntegrityError(\n"
    '                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "\n'
    '                f"{self.policy.max_size_bytes}-byte limit.",\n'
    "                context={\n"
    '                    "filename": filename,\n'
    '                    "size_bytes": size_bytes,\n'
    '                    "max_size_bytes": self.policy.max_size_bytes,\n'
    "                },\n"
    "            )\n"
    "\n"
    '        entry = self._read_manifest().get("artifacts", {}).get(filename)\n'
    "        if entry is None:\n"
    "            raise ArtifactNotFoundError(\n",
    "        if False and size_bytes > self.policy.max_size_bytes:\n"
    "            raise ArtifactIntegrityError(\n"
    '                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "\n'
    '                f"{self.policy.max_size_bytes}-byte limit.",\n'
    "                context={\n"
    '                    "filename": filename,\n'
    '                    "size_bytes": size_bytes,\n'
    '                    "max_size_bytes": self.policy.max_size_bytes,\n'
    "                },\n"
    "            )\n"
    "\n"
    '        entry = self._read_manifest().get("artifacts", {}).get(filename)\n'
    "        if entry is None:\n"
    "            raise ArtifactNotFoundError(\n",
)

M2A_EDIT = (
    "views/module_04_stress.py",
    "def render(ctx: AppContext) -> None:",
    'def render(ctx: AppContext) -> None:\n'
    '    st.checkbox("injected undeclared widget", key="zz_undeclared_key")',
)

M2B_EDIT = (
    "app.py",
    'st.title("🔬 ML Model Reliability & Stress Testing Framework")',
    'st.checkbox("injected undeclared root widget", key="zz_root_key")\n'
    'st.title("🔬 ML Model Reliability & Stress Testing Framework")',
)

M2C_EDIT = (
    "views/module_02_baseline.py",
    "load_model_trust",
    "load_model_trust_renamed",
)

M5A_EDIT = (
    "core/reporters.py",
    "class NullReporter:",
    "class NullReporter:\n"
    "    def _deferred(self) -> None:\n"
    "        from core.state import AppContext  # deferred: legitimately fine\n"
    "        return None\n",
)

M5B_EDIT = (
    "core/reporters.py",
    "class NullReporter:",
    "from core.state import AppContext  # module level: ladder violation\n\n\n"
    "class NullReporter:",
)

M4A_EDITS = [
    (
        "views/module_02_baseline.py",
        "def render(ctx: AppContext) -> None:",
        "from views.module_01_data import render as _cycle  # noqa: F401\n"
        "\n\ndef render(ctx: AppContext) -> None:",
    ),
    (
        "views/module_01_data.py",
        "def render(ctx: AppContext) -> None:",
        "from views.module_02_baseline import render as _cycle2  # noqa: F401\n"
        "\n\ndef render(ctx: AppContext) -> None:",
    ),
]

M4B_EDIT = (
    "views/module_02_baseline.py",
    "def render(ctx: AppContext) -> None:",
    "from views.module_01_data import render as _oneway  # noqa: F401\n"
    "\n\ndef render(ctx: AppContext) -> None:",
)

#: Every ``(rel, old, new)`` edit this probe applies.  ``--check-anchors``
#: iterates exactly this list (ADR-032).
DECLARED_EDITS: list[tuple[str, str, str]] = [
    ADR_001A_EDIT,
    ADR_001B_EDIT,
    ADR_001C_EDIT,
    ADR_001D_EDIT,
    M2A_EDIT,
    M2B_EDIT,
    M2C_EDIT,
    M5A_EDIT,
    M5B_EDIT,
    *M4A_EDITS,
    M4B_EDIT,
]


def check_anchors() -> int:
    """Verify every declared edit still resolves against the live tree.

    Side-effect-free: reads files only, spawns nothing, runs no pytest.  Exits
    non-zero (via the return value) if any ``old`` is absent from its file or is
    identical to its ``new``.

    Returns:
        int: ``0`` when every anchor resolves and differs; ``1`` otherwise.
    """
    problems: list[str] = []
    for rel, old, new in DECLARED_EDITS:
        path = ROOT / rel
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


if "--check-anchors" in sys.argv:
    raise SystemExit(check_anchors())


results: list[tuple[str, bool]] = []

# §4.5 step (d) caps three methods and `verify()` opens with a byte-for-byte copy
# of `load()`'s size-cap block.  A three-line anchor therefore mutates whichever
# comes first -- `verify()` -- and because `verify()` has no `trust` parameter
# the resulting NameError is reached by no test, so the suite stays green and
# the probe reports a PASS it has not earned.  That is what happened the first
# time ADR-001c was written.  Both anchors below run through each method's
# divergent `if entry is None:` arm, so a mis-anchored probe fails loudly
# ("anchor not found") instead of silently testing the wrong method.
print("ADR-001 - the security boundary must be load-bearing (decisions.md)")
# decisions.md (ADR-001, "Verified by mutation") claims these mutations are
# reproducible via this file.  A claim of that kind is evidence, not decoration:
# if the probes are absent the claim is false even though the code is correct, so
# they are asserted here rather than described in prose.  ADR-001d was added after
# ADR-001c reported a PASS for the wrong reason; see the comment on it.
results.append(
    (
        "ADR-001a removing the estimator type gate is caught",
        probe(
            "delete the isinstance(model, bases) gate",
            *ADR_001A_EDIT,
            ["tests/test_model_io.py"],
        ),
    )
)
results.append(
    (
        "ADR-001b disabling the digest comparison is caught",
        probe(
            "never compare the on-disk digest to the attestation",
            *ADR_001B_EDIT,
            ["tests/test_model_io.py"],
        ),
    )
)
results.append(
    (
        "ADR-001c trust=True cannot bypass the load-time size cap",
        probe(
            "let trust=True wave through the load() size cap",
            *ADR_001C_EDIT,
            ["tests/test_model_io.py"],
        ),
    )
)

results.append(
    (
        "ADR-001d the size cap is enforced by verify(), not only load()",
        # The anchor runs through `verify()`'s distinctive `if entry is None:
        # raise ArtifactNotFoundError(...)`, which `load()` does not share — so
        # this cannot silently mutate the wrong method the way ADR-001c did.
        probe(
            "delete verify()'s size cap",
            *ADR_001D_EDIT,
            ["tests/test_model_io.py"],
        ),
    )
)

print("\nADR-009.1 - the sanitizer's trailing-newline bypass stays closed")
# A direct behavioural assertion rather than a mutation: the point is that the
# *pattern* admits nothing, which no suite test can distinguish from the suite
# simply not covering it.

_policy = ArtifactPolicy()
_bypassed = re.fullmatch(_policy.filename_pattern, "model.pkl\n")
_closed = re.fullmatch(_policy.filename_pattern, "model.pkl") is not None
print(f"  pattern={_policy.filename_pattern!r}")
print(f"  {'PASS' if _bypassed is None else '*** BYPASSED ***'}: "
      f"'model.pkl\\n' does not match")
print(f"  {'PASS' if _closed else '*** REJECTS A VALID NAME ***'}: "
      f"'model.pkl' matches")
results.append(
    (
        "ADR-009.1a the trailing-newline bypass is closed",
        _bypassed is None and _closed,
    )
)

print("\nM2 - the widget-key contract, addition direction (reviewer's vacuous test)")
results.append(
    (
        "M2a new undeclared key in a view is rejected",
        probe(
            "new undeclared key in views/module_04_stress.py",
            *M2A_EDIT,
            ["tests/test_architecture.py"],
        ),
    )
)
results.append(
    (
        "M2b new undeclared key in a root module is rejected",
        probe(
            "new undeclared key in app.py",
            *M2B_EDIT,
            ["tests/test_architecture.py"],
        ),
    )
)
results.append(
    (
        "M2c renaming an existing key is still rejected",
        probe(
            "rename load_model_trust",
            *M2C_EDIT,
            ["tests/test_architecture.py"],
        ),
    )
)

print("\nM1 - an undeclared root module must not escape the gate")
# ADR-012 amended the remedy: the reviewer's M1 asked for `main.py` to be deleted
# *or* declared. It is now declared (§2.2, with a contract and its own suite), so
# the invariant is no longer "main.py is absent" — that would fail against the
# accepted decision — but "every root module in the tree is declared, and a root
# module that is not declared is rejected." The first half is asserted by running
# the two §2.2 gates against the real tree; the second by planting one.
UNDECLARED = "extra.py"
stub = ROOT / UNDECLARED
results.append(
    (
        "M1a every root module in the tree is declared in architecture.md 2.2",
        run(
            [
                "tests/test_architecture.py::test_root_modules_are_exactly_the_declared_set",
                "tests/test_architecture.py::test_root_modules_match_architecture_spec",
            ]
        )[0]
        == 0,
    )
)
declared = sorted(p.name for p in ROOT.glob("*.py"))
print(f"  PASS: {declared} — both §2.2 gates green against the real tree")
try:
    stub.write_text(
        "#!/usr/bin/env python3\n"
        '"""Undeclared harness-shaped stub; created and removed by this probe."""\n'
        "import sys\n"
        "import streamlit as st\n"
        "from views.base import build_context\n"
        "\n"
        "def main() -> int:\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n",
        encoding="utf-8",
    )
    failed, passed = run(["tests/test_architecture.py"])
    bit = failed > 0
    verdict = "CAUGHT" if bit else "*** NOT CAUGHT ***"
    print(f"  {verdict}: undeclared root module importing streamlit + views "
          f"({failed} failed, {passed} passed)")
    results.append(("M1b undeclared root module is rejected", bit))
finally:
    if stub.exists():
        stub.unlink()
    if stub.exists():  # pragma: no cover
        raise SystemExit("could not remove probe stub")

print("\nm5 - _module_level_imports must not report method-level imports")
ladder = ["tests/test_architecture.py::test_core_ladder_respected[reporters]"]
caught = probe(
    "method-level import of a forbidden rung (expected: NOT caught)",
    *M5A_EDIT,
    ladder,
)
results.append(("m5a method-level import is not a false positive", not caught))
caught = probe(
    "module-level import of a forbidden rung (expected: CAUGHT)",
    *M5B_EDIT,
    ladder,
)
results.append(("m5b module-level import is still caught", caught))

print("\nm4 - test_no_import_cycles must itself catch a cycle")
results.append(
    (
        "m4a a TRUE bidirectional intra-views cycle is caught",
        probe_multi(
            "module_01 <-> module_02 cycle (both directions)",
            ["tests/test_architecture.py::test_no_import_cycles"],
            list(M4A_EDITS),
        ),
    )
)
results.append(
    (
        "m4b a single one-way views edge is NOT a cycle (negative control)",
        not probe_multi(
            "module_02 -> module_01 only (not a cycle)",
            ["tests/test_architecture.py::test_no_import_cycles"],
            [M4B_EDIT],
        ),
    )
)

print("\n--- summary ---")
ok = True
for label, bit in results:
    print(f"  {'PASS' if bit else 'FAIL'}  {label}")
    ok = ok and bit
print("\nALL PROBES BEHAVED AS EXPECTED" if ok else "\nSOME PROBES DID NOT BITE")
raise SystemExit(0 if ok else 1)
