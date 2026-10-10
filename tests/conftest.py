"""Shared pytest bootstrap.

**Tier T0 constraint (architecture.md §10.1).**  This file is loaded by pytest
for *every* test in the suite, including the security and layering gates, so it
is itself held to the bare-interpreter rule: **stdlib + pytest only, no numpy,
no pandas, no project imports at module level.**  A single module-level
``import numpy`` here fails collection of
``tests/test_architecture.py`` and ``tests/test_model_io.py`` with
``ModuleNotFoundError`` on an interpreter that has only pytest installed — which
is exactly the environment those two gates exist to be runnable in.  A security
gate that cannot start is not a gate.

The T1 suites do **not** rely on that being noticed: each declares its own
scientific dependency with an explicit ``pytest.importorskip(...)`` placed
**before** the import that needs it, so a bare (pytest-only) full-suite run
collects the T0 gates and skips T1 with zero collection errors.  That ordering
is the whole point — a bare ``import numpy`` at module scope is a *collection
error*, not a skip, and a collection error aborts the run before any T0 gate
executes.  Which file skips on what is stated in each file's own header; this
file asserts nothing about it, because conftest cannot import anything to check.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make ``core``, ``modules``, ``utils`` and ``views`` importable when pytest is
# invoked as ``python -m pytest`` from an arbitrary working directory.  Purely
# stdlib, so it costs the T0 tier nothing.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
