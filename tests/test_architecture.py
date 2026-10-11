"""Executable import-layer contract (architecture.md §2.1, §2.3, §10.1).

**Tier T0** — stdlib + pytest only, by construction: this file imports ``ast``
and ``pathlib`` and nothing else.  It must stay runnable on an interpreter that
has *only* pytest installed, because a layering gate that cannot start is not a
gate (ADR-010.1).  Any temptation to ``import core.errors`` to compare exception
classes here has to be rejected — compare dotted name strings instead.

What is enforced:

* §2.1  the cross-layer import matrix (``core``/``modules``/``utils``/``views``/
  ``app.py``) in both directions: forbidden *and* permitted edges;
* §2.1  no module imports ``app``, and no import cycle exists;
* §2.2  the top-level modules are exactly the declared set (``app.py``,
  ``main.py``), so no file can sit outside the matrix by virtue of where it
  lives — and that allowlist is cross-checked against the §2.2 layout block in
  the specification, so it cannot be widened here without being declared there;
* §2.3  the intra-``core`` ladder, so the ``CORRUPTION_TYPES`` ownership cycle
  that §2.3 was written to prevent cannot reappear;
* §4.8  the nine sidebar labels and the 64 frozen widget ``key=`` literals, read
  from :file:`tests/fixtures/widget_keys.json` — asserted as exact set equality
  per view, in both directions.

Every check is a source-level ``ast`` walk rather than an import of the module
under test.  Importing would both violate the tier rule and defeat the purpose:
a module that violates the matrix is precisely the one that must still be
*readable* by the checker.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

PACKAGES = ("core", "modules", "utils", "views")


def _source_files(pkg: str) -> list[Path]:
    pkg_dir = ROOT / pkg
    if not pkg_dir.is_dir():
        return []
    return sorted(p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _root_modules() -> list[Path]:
    """Every top-level ``*.py`` file in the repository.

    Enumerated rather than hard-coded to ``app.py`` on purpose.  Hard-coding
    meant that any *new* root-level module was invisible to every check below —
    the cycle detector, ``test_no_module_imports_app`` and the widget-key
    contract all consumed :func:`_project_files`, so a file that was not
    literally named ``app.py`` was exempt from all of them while still being
    importable and shippable.  :func:`test_root_modules_are_exactly_the_declared_set`
    turns that exemption into a loud failure instead of a silent one, and
    :func:`test_root_modules_match_architecture_spec` keeps that failure's
    remedy honest by cross-checking the allowlist against architecture.md §2.2.
    """
    return sorted(p for p in ROOT.glob("*.py") if p.is_file())


def _project_files() -> list[Path]:
    files = _root_modules()
    for pkg in PACKAGES:
        files.extend(_source_files(pkg))
    return [f for f in files if f.is_file()]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> list[str]:
    """Return every imported top-level dotted name, absolute or relative.

    Relative imports are resolved to their absolute form (``from . import x``
    inside ``core`` yields ``core``) so the matrix can be applied uniformly
    without special-casing the package under inspection.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                found.append(f"{'.' * node.level}{mod}".lstrip("."))
            elif mod:
                found.append(mod)
    return found


def _top(name: str) -> str:
    return name.split(".")[0]


# ── §2.1 cross-layer matrix ───────────────────────────────────────────────────

#: module that MUST NOT be imported, per layer.
FORBIDDEN: dict[str, set[str]] = {
    "core": {"streamlit", "plotly", "views", "app", "modules"},
    "modules": {"streamlit", "views", "app"},
    "utils": {"streamlit", "views", "app", "modules", "core"},
    "views": {"app"},
}

#: stdlib and third-party roots each layer is *expected* to be able to use.
#: An import outside this set is not automatically a failure — it is reported
#: separately by :func:`test_no_unexpected_third_party_imports` so an unvetted new
#: dependency cannot slip in behind a passing matrix check.
ALLOWED_THIRD_PARTY: dict[str, set[str]] = {
    "core": {
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "xgboost",
        "typing_extensions",
    },
    "modules": {
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "xgboost",
        "plotly",
        "matplotlib",
        "seaborn",
        "reportlab",
        "joblib",
    },
    "utils": {"numpy", "pandas", "plotly", "matplotlib", "seaborn"},
    "views": {
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "plotly",
        "matplotlib",
        "seaborn",
        "streamlit",
        "altair",
        "jinja2",
    },
}

STDLIB = set(sys.stdlib_module_names)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_layer_forbids_matrix(pkg: str) -> None:
    """§2.1: no layer may import a module the matrix forbids."""
    banned = FORBIDDEN[pkg]
    offenders: list[str] = []
    for path in _source_files(pkg):
        for name in _imports(_parse(path)):
            root = _top(name)
            if root in banned:
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "forbidden imports for §2.1 layer:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_layer_allows_only_declared_dependencies(pkg: str) -> None:
    """§2.1: project-local imports may only point at the layers it MAY import.

    ``core`` MAY import nothing project-local (``core.errors``/``core.config``
    only, plus the intra-ladder entries checked separately), ``utils`` MAY import
    nothing project-local at all, ``modules`` MAY import ``utils``/``core``, and
    ``views`` MAY import ``core``/``modules``/``utils``.
    """
    permitted: dict[str, set[str]] = {
        "core": set(),  # §2.3 handles intra-core
        "modules": {"core", "utils"},
        "utils": set(),
        "views": {"core", "modules", "utils"},
    }
    allowed = permitted[pkg]
    offenders: list[str] = []
    for path in _source_files(pkg):
        for name in _imports(_parse(path)):
            root = _top(name)
            if root in PACKAGES and root != pkg and root not in allowed:
                offenders.append(
                    f"{path.relative_to(ROOT)} imports {name} "
                    f"(allowed project deps for {pkg}: {sorted(allowed) or 'none'})"
                )
    assert not offenders, "undeclared project-local imports:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_no_unexpected_third_party_imports(pkg: str) -> None:
    """No layer may pull in a third-party root outside its declared set.

    This is the dependency-hygiene half of §2.1: the FORBIDDEN matrix alone would
    happily accept a brand-new library that nothing has reviewed (conventions §9
    requires an ADR for a new runtime dependency).
    """
    allowed = ALLOWED_THIRD_PARTY[pkg]
    offenders: list[str] = []
    for path in _source_files(pkg):
        for name in _imports(_parse(path)):
            root = _top(name)
            if root in STDLIB or root in PACKAGES or root == "app":
                continue
            if root not in allowed:
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "undeclared third-party imports:\n  " + "\n  ".join(offenders)


def test_no_module_imports_app() -> None:
    """§2.1: ``app.py`` is the composition root; nothing may import it."""
    offenders: list[str] = []
    for path in _project_files():
        if path.name == "app.py":
            continue
        for name in _imports(_parse(path)):
            if _top(name) == "app":
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "app.py imported by:\n  " + "\n  ".join(offenders)


# ── §2.3 intra-``core`` ladder ────────────────────────────────────────────────

#: project-local modules each ``core`` module is permitted to import (§2.3).
#:
#: ``config`` is rung 1 and may import nothing project-local at module scope.  It
#: does need ``ConfigurationError`` to report bad overrides, so it defers that one
#: import into the function bodies that raise.  The edge is recorded explicitly
#: below rather than left implicit, because "config imports nothing" and "config
#: lazily imports errors" are different claims and only the first is what §2.3
#: protects: the ladder exists to keep the *module graph* acyclic at import time.
#: ``core.errors`` imports no project-local module, so even the deferred edge
#: cannot close a cycle — and :func:`test_config_deferred_import_is_cycle_free`
#: asserts exactly that rather than trusting it.
CORE_LADDER: dict[str, set[str]] = {
    "config": set(),
    "errors": set(),
    "validation": {"config", "errors"},
    "perturbations": {"config", "errors", "validation"},
    "model_io": {"config", "errors"},
    "reporters": set(),
    "state": {"config", "errors"},
    "logging_config": {"config"},
    "scoring": set(),
}

#: The only project-local import ``core.config`` may contain, and only inside a
#: function body.
CONFIG_DEFERRED_ALLOWED = "errors"


class _ImportTimeCollector(ast.NodeVisitor):
    """Collect the imports that execute when the module is *imported*.

    Descent stops at every ``FunctionDef``/``AsyncFunctionDef`` — module-level
    function, method, or nested closure alike — because a body only runs when it
    is called and is therefore not part of the import-time graph.  That
    distinction is exactly what §2.3's ladder turns on: ``core.config`` defers
    one import into the bodies that raise, and it must not be reported as an
    import-time edge.

    An earlier version of this walked the module and subtracted function bodies
    afterwards, which meant an import inside a *method* was reported as
    module-level.  That fails in the conservative direction (false positives, not
    false negatives), but a gate that cries wolf gets disabled, so the collector
    is now written so that it cannot walk into a function at all.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Intentionally empty: the body is deferred until call time."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Intentionally empty: the body is deferred until call time."""

    def visit_Import(self, node: ast.Import) -> None:
        self.names.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            self.names.append(f"{'.' * node.level}{module}".lstrip("."))
        elif module:
            self.names.append(module)


def _module_level_imports(tree: ast.Module) -> list[str]:
    """Imports that execute at module import time (i.e. not inside a function)."""
    collector = _ImportTimeCollector()
    collector.visit(tree)
    return collector.names


@pytest.mark.parametrize("module_name", sorted(CORE_LADDER))
def test_core_ladder_respected(module_name: str) -> None:
    """§2.3: each ``core`` module imports only from its permitted rungs.

    This is the check that makes ADR-011.1's placement of ``CORRUPTION_TYPES`` in
    ``core.validation`` load-bearing: if the enums ever moved back to
    ``core.perturbations`` without moving ``validate_stress_params``, this fails
    instead of producing an import cycle.
    """
    path = ROOT / "core" / f"{module_name}.py"
    # Reviewer m2: this used to ``pytest.skip`` on a missing file, which makes the
    # gate pass vacuously on exactly the structural loss it exists to detect.  It
    # fails instead; ``test_core_kernel_modules_exist`` already asserts the file
    # set, so the two agree on what "missing" means.
    assert path.is_file(), (
        f"core/{module_name}.py is absent, so the §2.3 ladder check for it "
        "cannot run — a gate that skips is not a gate"
    )
    permitted = CORE_LADDER[module_name]
    tree = _parse(path)
    offenders: list[str] = []
    for name in _module_level_imports(tree):
        root = _top(name)
        if root == "core":
            target = name.split(".")[1] if "." in name else ""
            if target and target != module_name and target not in permitted:
                offenders.append(
                    f"core/{module_name}.py imports core.{target} at module level "
                    f"(permitted: {sorted(permitted) or 'none'})"
                )
    assert not offenders, "intra-core ladder violations:\n  " + "\n  ".join(offenders)


def test_config_deferred_import_is_cycle_free() -> None:
    """``core.config``'s one deferred project-local import must be ``errors``.

    §2.3 puts ``config`` at rung 1 ("imports nothing project-local").  It needs
    ``ConfigurationError`` to report malformed overrides, so it defers the import
    into the raising functions.  That is only safe because ``core.errors`` itself
    imports nothing project-local; this test pins both halves of that argument
    rather than leaving it as a comment.
    """
    tree = _parse(ROOT / "core" / "config.py")
    project_local = [
        n for n in _imports(tree) if n.split(".")[0] in {"core", "modules", "utils", "views"}
    ]
    unexpected = [n for n in project_local if n.split(".")[1:] != [CONFIG_DEFERRED_ALLOWED]]
    assert not unexpected, (
        "core/config.py may only defer-import core."
        f"{CONFIG_DEFERRED_ALLOWED}; found {sorted(set(unexpected))}"
    )
    # And the deferred target must itself be cycle-free.
    errors_tree = _parse(ROOT / "core" / "errors.py")
    assert not [
        n for n in _imports(errors_tree) if n.split(".")[0] in {"core", "modules", "utils", "views"}
    ], "core/errors.py must import nothing project-local, or the ladder is invalid"


def test_core_has_no_streamlit_or_plotly() -> None:
    """§2.1/§2.2: ``core`` is the framework-agnostic kernel."""
    offenders = [
        f"{p.relative_to(ROOT)} imports {n}"
        for p in _source_files("core")
        for n in _imports(_parse(p))
        if _top(n) in {"streamlit", "plotly"}
    ]
    assert not offenders, "\n  ".join(offenders)


# ── §2.1 no import cycles ─────────────────────────────────────────────────────


def _project_import_graph() -> dict[str, set[str]]:
    """Map each project module to the project modules it imports."""
    graph: dict[str, set[str]] = {}
    for path in _project_files():
        if path.name == "app.py":
            stem = "app"
        elif len(path.relative_to(ROOT).parts) == 1:
            stem = path.stem
        else:
            parts = path.relative_to(ROOT).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            stem = ".".join(parts)
        deps: set[str] = set()
        for name in _imports(_parse(path)):
            if name in ("app", "core", "modules", "utils", "views"):
                deps.add(name)
            elif name.startswith(("core.", "modules.", "utils.", "views.")):
                deps.add(".".join(name.split(".")[:2]))
            elif name in {p.stem for p in _source_files("core")}:
                # legacy flat import of a core sibling, e.g. ``import config``
                deps.add(f"core.{name}")
        graph[stem] = deps
    return graph


def test_no_import_cycles() -> None:
    """§2.1: circular imports are a test failure.

    Implemented as an iterative depth-first search with an explicit colour map
    (white = unvisited, grey = on the current path, black = finished).  A grey
    node reached again is a back edge, i.e. a cycle, and is reported with the
    path that produced it.  Iterative rather than recursive so a deep or
    pathological graph cannot hit the recursion limit.
    """
    graph = _project_import_graph()
    colour: dict[str, int] = {}
    for start in sorted(graph):
        if colour.get(start, 0) != 0:
            continue
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        colour[start] = 1
        while stack:
            node, path = stack[-1]
            unvisited = [d for d in sorted(graph.get(node, ())) if colour.get(d, 0) == 0]
            if unvisited:
                nxt = unvisited[0]
                colour[nxt] = 1
                stack.append((nxt, [*path, nxt]))
                continue
            colour[node] = 2
            stack.pop()
            for dep in sorted(graph.get(node, ())):
                if colour.get(dep, 0) == 1:
                    cycle = " -> ".join([*path[path.index(dep) :], dep])
                    pytest.fail(f"import cycle detected: {cycle}")


# ── §4.8 persisted session contract ───────────────────────────────────────────


def _contract() -> dict:
    return json.loads((FIXTURES / "widget_keys.json").read_text(encoding="utf-8"))


def _attr_chain(node: ast.AST) -> list[str]:
    """Flatten ``module_01_data.render`` into ``["module_01_data", "render"]``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _fstring_template(node: ast.JoinedStr) -> str:
    """Turn ``f"missing_{col}"`` into the stable template ``missing_{col}``.

    Constant parts keep their text verbatim; each interpolation hole is
    unparsed, so ``{col}`` and ``{model_sel}`` round-trip exactly.  A hole that
    carries a conversion or a format spec (``{x!r:.2f}``) cannot be declared
    deterministically, so it is folded to ``{...}`` — the resulting template
    matches nothing in the fixture and the gate fails loudly instead of
    guessing.
    """
    parts: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            parts.append(part.value)
        elif (
            isinstance(part, ast.FormattedValue)
            and part.conversion == -1
            and part.format_spec is None
        ):
            parts.append("{" + ast.unparse(part.value) + "}")
        else:
            parts.append("{...}")
    return "".join(parts)


def _widget_keys(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Literals and f-string key templates in a module: ``(keys, templates)``.

    F-strings are kept apart from literals because a single
    ``st.selectbox(..., key=f"missing_{col}")`` writes one distinct persisted
    session entry per looped column — the template *family* is the unit of
    review, not any single rendered value (T12).
    """
    literals: set[str] = set()
    templates: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "key":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                literals.add(kw.value.value)
            elif isinstance(kw.value, ast.JoinedStr):
                templates.add(_fstring_template(kw.value))
    return literals, templates


def _widget_key_owners() -> list[tuple[Path, set[str], set[str]]]:
    """Every file that could own a widget key, with literals and templates.

    Root-level modules are included as well as the view package: ``app.py`` used
    to own all 64 keys before Wave 5, so a key re-added there must be caught
    rather than ignored because of where it was pasted.
    """
    paths = _root_modules() + _source_files("views")
    return [(p, *_widget_keys(_parse(p))) for p in paths if p.is_file()]


def test_nine_page_labels_are_frozen() -> None:
    """§4.8 / conventions §7: the nine sidebar labels may not drift."""
    expected = _contract()["page_labels"]
    assert len(expected) == 9, f"expected nine pages, fixture has {len(expected)}"
    base = ROOT / "views" / "base.py"
    # Reviewer m2: was a ``pytest.skip``, so deleting views/base.py turned the
    # frozen-label contract green instead of red.
    assert base.is_file(), (
        "views/base.py is absent, so the frozen sidebar-label contract cannot be "
        "checked — deleting the file would otherwise pass this gate"
    )
    tree = _parse(base)
    declared = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    missing = [label for label in expected if label not in declared]
    assert not missing, f"sidebar labels absent from views/base.py: {missing}"


def test_widget_key_contract_is_unchanged() -> None:
    """§4.8: the 64 ``key=`` literals are the persisted session contract.

    Asserted as **exact set equality per file, in both directions**.  The frozen
    fixture therefore catches all four ways the contract can break:

    * a key is **renamed**  → it is declared but absent;
    * a key is **dropped**   → it is declared but absent;
    * a key **moves to a different page** → absent from one file, undeclared in
      another;
    * a **new, undeclared key is added** → present but undeclared.

    The fourth case is the one an ``expected ⊆ actual`` check cannot see, and it
    is the case that matters most: an undeclared key silently writes a new
    session-state entry that nothing has reviewed.  It is what makes
    ``load_model_trust`` (Wave 1, ADR-001) the *only* sanctioned addition —
    architecture.md §4.8 records 63 + 1, so 63 would mean the trust gate had been
    dropped and 65 would mean an unreviewed widget had appeared.

    Root-level modules and the whole ``views`` package are both scanned, so the
    check cannot be defeated by relocating the widget.
    """
    contract = _contract()
    expected_total = contract["widget_keys_total"]
    assert expected_total == contract["widget_keys_unique"], (
        "fixture is internally inconsistent: "
        f"{expected_total} total vs {contract['widget_keys_unique']} unique"
    )
    by_view: dict[str, list[str]] = contract["widget_keys_by_view"]

    owners = _widget_key_owners()
    present_names = {p.name for p, _, _ in owners}
    stale = sorted(set(by_view) - present_names)
    assert not stale, f"fixture lists view files that no longer exist: {stale}"

    missing: list[str] = []
    extra: list[str] = []
    for path, actual, _ in owners:
        declared = set(by_view.get(path.name, ()))
        missing += [f"{path.name}:{key}" for key in sorted(declared - actual)]
        extra += [f"{path.name}:{key}" for key in sorted(actual - declared)]
    assert not missing, "widget keys declared but absent:\n  " + "\n  ".join(missing)
    assert not extra, (
        "undeclared widget keys found — add them to "
        "tests/fixtures/widget_keys.json and record why in "
        "added_during_remediation:\n  " + "\n  ".join(extra)
    )

    seen: set[str] = set()
    for keys in by_view.values():
        seen |= set(keys)
    assert len(seen) == expected_total, (
        f"widget key drift: fixture expects {expected_total} unique keys, "
        f"the union across views is {len(seen)}"
    )


def test_widget_key_templates_are_unchanged() -> None:
    """§4.8: the interpolated ``key=f"..."`` families are sanctioned as well.

    ``key="foo"`` gates one exact session key, but ``key=f"bar_{name}"`` gates a
    *family* — one persisted entry per looped column, gauge, or model.  A
    hand-written literal gate sees none of them, so without this check a
    developer could add a new f-string key, or repoint an existing one at a
    different ``{col}``/``{mname}``, and every rendered value would drift from
    the reviewed contract with the suite still green (T12).

    The declared families live in ``widget_key_templates_by_view`` inside
    :file:`tests/fixtures/widget_keys.json`:

    * ``module_01_data.py``        →
      ``missing_{col}`` (the per-column imputation: one selectbox for the
      detection strategy and one for the fill strategy);
    * ``module_08_reliability.py`` →
      ``rel_gauge_overview_{mname}`` and ``rel_gauge_detail_{model_sel}`` (the
      overview gauge row and the per-``model_sel`` compare gauge).

    Asserted as exact set equality per file, in both directions, exactly like
    the literal gate above — renaming the template renames the session key for
    every value the family has ever produced.
    """
    contract = _contract()
    declared: dict[str, list[str]] = contract["widget_key_templates_by_view"]
    owners = _widget_key_owners()
    missing: list[str] = []
    extra: list[str] = []
    for path, _, actual in owners:
        expected = set(declared.get(path.name, ()))
        missing += [f"{path.name}:{tpl}" for tpl in sorted(expected - actual)]
        extra += [f"{path.name}:{tpl}" for tpl in sorted(actual - expected)]
    assert not missing, (
        "widget key templates declared but absent (renamed or dropped?):\n  "
        + "\n  ".join(missing)
    )
    assert not extra, (
        "undeclared interpolated widget keys found — declare the family in "
        "tests/fixtures/widget_keys.json and record why, or the key must be a "
        "literal:\n  " + "\n  ".join(extra)
    )


def test_added_widget_keys_are_declared() -> None:
    """Every key added since the pre-remediation baseline is declared *and* real.

    Two halves, both of which have to hold for this test to mean anything:

    1. **Arithmetic** — the fixture total is 63 pre-remediation keys plus exactly
       one entry per declared addition.  A key can therefore never be absorbed
       into the 63 baseline silently.
    2. **Existence** — each declared addition is actually implemented in a view.
       Without this half the test is a statement about the fixture's own internal
       consistency and can be satisfied by a justification for a key that does
       not exist, which is ADR-010's "gate that passes for the wrong reason" in
       miniature.
    """
    contract = _contract()
    added = contract["added_during_remediation"]
    expected_total = contract["widget_keys_total"]
    assert expected_total == 63 + len(added), (
        f"fixture total {expected_total} != 63 pre-remediation keys + "
        f"{len(added)} declared addition(s): {sorted(added)}"
    )
    implemented: set[str] = set()
    for _, keys, _ in _widget_key_owners():
        implemented |= keys
    phantom = sorted(set(added) - implemented)
    assert not phantom, (
        f"declared as added during remediation but not implemented in any view: {phantom}"
    )


# ── §2.2 layout ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pkg", (*PACKAGES, "tests"))
def test_package_has_init(pkg: str) -> None:
    """§2.2: every package directory has an ``__init__.py``."""
    assert (ROOT / pkg / "__init__.py").is_file(), f"{pkg}/__init__.py is missing"


def test_view_render_callables_are_annotated() -> None:
    """Conventions §2: the nine page modules annotate ``ctx: AppContext``.

    ``PAGES`` in ``app.py`` is typed ``dict[str, Callable[[AppContext], None]]``,
    so an unannotated ``def render(ctx)`` is not merely a missing hint — it is a
    declared signature the implementation silently fails to match, and with no
    type checker in the project nothing else would notice.  All nine views
    shipped unannotated through Wave 5, which is the shape of Critic MIN-004.

    Asserted structurally (``ast``) rather than with ``typing.get_type_hints`` so
    the check stays in the T0 tier (§10.1) and never imports ``core.state``.
    """
    offenders: list[str] = []
    found = 0
    for path in _source_files("views"):
        if not path.name.startswith("module_"):
            continue
        for node in _parse(path).body:
            if not isinstance(node, ast.FunctionDef) or node.name != "render":
                continue
            found += 1
            if len(node.args.args) != 1 or node.args.args[0].arg != "ctx":
                offenders.append(f"{path.name}::render must take exactly one positional arg, 'ctx'")
                continue
            annotation = node.args.args[0].annotation
            if not (isinstance(annotation, ast.Name) and annotation.id == "AppContext"):
                offenders.append(
                    f"{path.name}::render(ctx) is not annotated AppContext "
                    f"(got {ast.unparse(annotation) if annotation else 'nothing'})"
                )
            if ast.unparse(node.returns) != "None":
                offenders.append(f"{path.name}::render must return None")
    assert found == 9, f"expected nine views/module_0N_*.py render() callables, found {found}"
    assert not offenders, "conventions §2 typing violations:\n  " + "\n  ".join(offenders)


# ── conventions §7 / §3 error rendering in ``views/`` ────────────────────────────

#: The two renderer entry points ``views/`` may use, per conventions §3 ("views
#: own all ``st.error`` rendering") and §7 ("every mutating action reports
#: success **or** a rendered ``FrameworkError``").
RENDERERS = ("render_error", "render_errors")


def _called_names(node: ast.AST) -> set[str]:
    """Every bare function name *called* anywhere under ``node``."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _button_handlers(tree: ast.Module) -> list[ast.If]:
    """Return every ``if st.button(…)`` handler in ``tree``, nested ones included.

    Read structurally because the call is the *test* of an ``if``, and that is the
    only place a button can be pressed in this codebase.  Walking the whole tree
    rather than one nesting level matters: a button handler legitimately contains
    nested conditionals, and a top-level-only scan would miss those entirely.
    """
    handlers: list[ast.If] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "button" in _called_names(node.test):
            handlers.append(node)
    return handlers


def test_renderer_symbols_are_bound_where_they_are_used() -> None:
    """A view that *calls* a renderer must have *imported* it.

    This is the ``NameError`` class, and it is not hypothetical: when
    ``views/module_06_calibration.py`` shipped in run-003, all four of its button
    handlers were wrapped in ``with render_errors():`` while the file never
    imported the name.  Every page-rendering test passed, 673 tests were green,
    and the defect was live in the tree — because ``render_errors`` is only bound
    when a button is *pressed*, and no test pressed one.  ``ast`` cannot see a
    runtime ``NameError``, so this check exists to see it statically instead.
    """
    unbound: list[str] = []
    for path in _source_files("views"):
        tree = _parse(path)
        used = {name for name in RENDERERS if name in _called_names(tree)}
        if not used:
            continue
        # The name is satisfied by an import *or* by a local definition:
        # ``views/reporters.py`` is where both renderers are defined.
        bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("reporters"):
                bound.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
        for name in sorted(used - bound):
            unbound.append(
                f"{path.name} calls {name}() but neither imports nor defines it — "
                "NameError the first time that code path runs"
            )
    assert not unbound, "unbound renderer symbols:\n  " + "\n  ".join(unbound)


def test_views_owning_mutating_buttons_import_a_renderer() -> None:
    """Reviewer MINOR-3: a view with a button must be able to render an error.

    MINOR-3 found the error-handling coverage across ``views/`` asymmetric:
    ``module_01``/``module_02`` rendered, while ``module_04``/``module_05``/
    ``module_06`` had no ``try``/``except`` at all.  ``module_07``/``08``/``09``
    were *not* at fault — they own no buttons, so they have no mutating action to
    report on.

    The rule this test makes explicit is therefore the per-view one the reviewer
    asked for, and it keys off button ownership rather than off the file list, so
    it cannot rot as pages are added or renamed: **a view that presents a mutating
    action can render a ``FrameworkError``**.  It checks the import only; whether
    the guard is actually *applied* to each handler is
    :func:`test_every_button_handler_body_is_guarded`.
    """
    offenders: list[str] = []
    found = 0
    for path in _source_files("views"):
        if not path.name.startswith("module_"):
            continue
        tree = _parse(path)
        if not _button_handlers(tree):
            continue
        found += 1
        if not any(name in _called_names(tree) for name in RENDERERS):
            offenders.append(
                f"{path.name} owns st.button( but imports neither render_error "
                "nor render_errors — conventions §7 has nothing to render with"
            )
    assert found, "no view owns a button; this gate is silently validating nothing"
    assert not offenders, "conventions §7 violations:\n  " + "\n  ".join(offenders)


def test_every_button_handler_body_is_guarded() -> None:
    """Reviewer MINOR-3: every button handler's body actually *applies* a guard.

    The import check above is necessary but not sufficient — a view can import
    ``render_errors`` and then leave one handler unwrapped.  This asserts that
    every ``if st.button(…)`` body contains, somewhere beneath it, either

    * a ``with render_errors():`` block, or
    * a ``try``/``except FrameworkError`` whose handler calls ``render_error``.

    Both spellings are accepted because both are idiomatic here:
    :func:`views.reporters.render_errors` exists precisely so a view can wrap a
    whole handler rather than hand-copy a ``try`` at each call site, and the
    explicit form predates it.  What is *not* accepted is a handler with neither,
    because that is exactly how MAJOR-1's ``ParameterOutOfRangeError`` reached
    the user as a traceback.

    Static by necessity (§10.1 keeps this file stdlib + pytest), so it proves the
    guard is *present*, not that it fires.  ``tests/test_app_smoke.py`` presses
    every button for the other half of the claim.
    """
    offenders: list[str] = []
    total = 0
    framework_errors = _framework_error_closure()
    for path in _source_files("views"):
        if not path.name.startswith("module_"):
            continue
        tree = _parse(path)
        for handler in _button_handlers(tree):
            total += 1
            guarded = False
            for node in ast.walk(handler):
                # with render_errors(): ...
                if isinstance(node, (ast.With, ast.AsyncWith)) and any(
                    name in RENDERERS for item in node.items for name in _called_names(item.context_expr)
                ):
                    guarded = True
                    break
                # try: ... except FrameworkError: render_error(exc)
                if isinstance(node, ast.Try):
                    for except_handler in node.handlers:
                        caught: set[str] = set()
                        for expr in _exception_types(except_handler):
                            if isinstance(expr, ast.Name):
                                caught.add(expr.id)
                            elif isinstance(expr, ast.Attribute):
                                caught.add(expr.attr)
                            elif isinstance(expr, ast.Tuple):
                                caught.update(
                                    elt.id if isinstance(elt, ast.Name) else ast.unparse(elt)
                                    for elt in expr.elts
                                )
                        # Any subclass counts, not just the base: see
                        # :func:`_framework_error_closure`.
                        renders = any(
                            name in RENDERERS
                            for name in _called_names(
                                ast.Module(body=except_handler.body, type_ignores=[])
                            )
                        )
                        if caught & framework_errors and renders:
                            guarded = True
                            break
            if not guarded:
                offenders.append(
                    f"{path.name}: a button handler body has no render_errors() "
                    "context and no except FrameworkError that renders — a domain "
                    "error here escapes as a traceback (conventions §7)"
                )
    assert total, "no button handlers found; this gate is silently validating nothing"
    assert not offenders, "conventions §7 violations:\n  " + "\n  ".join(offenders)


def _exception_types(handler: ast.ExceptHandler) -> list[ast.AST]:
    """Return the exception expression of an ``except`` clause, if any."""
    return [] if handler.type is None else [handler.type]


def _framework_error_closure() -> frozenset[str]:
    """Every class name that is ``FrameworkError`` or inherits from it.

    Resolved by parsing :file:`core/errors.py` rather than importing it, because
    §10.1 puts this file in the T0 tier: stdlib and pytest only, no project
    imports.  The alternative — matching the literal ``FrameworkError`` — would be
    wrong in the way that matters most here.  ``module_01``'s load button catches
    ``DatasetLoadError``, a *subclass*, and that handler is correctly guarded; a
    literal-name gate would flag it and train the reader to ignore the gate.

    Small fixed-point over the class table: a class joins the closure as soon as
    any base is already in it, repeated until nothing changes.
    """
    errors_py = ROOT / "core" / "errors.py"
    bases: dict[str, list[str]] = {}
    for node in _parse(errors_py).body:
        if isinstance(node, ast.ClassDef):
            bases[node.name] = [ast.unparse(base) for base in node.bases]

    closure = {"FrameworkError"}
    grew = True
    while grew:
        grew = False
        for name, parents in bases.items():
            if name not in closure and any(parent in closure for parent in parents):
                closure.add(name)
                grew = True
    return frozenset(closure)


# ── conventions §7 press coverage: every ``st.button`` has a press ────────────

#: Non-vacuity floor for
#: :func:`test_every_mutating_button_in_the_views_is_pressed_here`.
#:
#: Measured on the run-005 tree: ``views/`` owns exactly 20 ``st.button`` call
#: sites.  This is a floor on *what the extractor may find*, not an inventory —
#: no button identity is repeated here, so adding a button does not require
#: touching this file and both halves of the reconciliation stay derived from
#: the tree (roadmap deviation 0b.1).
#:
#: Three independent causes make a measured count land below the floor, and the
#: assertion message names all three so a failure is diagnosable from the
#: pytest output alone.  The gate deliberately cannot tell them apart — that
#: judgement belongs to whoever reads the failure:
#:
#: 1. Extractor regression — a renamed helper, a new call idiom, a scan that
#:    stops descending — silently stops finding sites, so the suite reconciles
#:    a shrunken set and could pass for the wrong reason, the "green for the
#:    wrong reason" shape ADR-010's rationale warns about.  Remedy: fix the
#:    extractor; the floor stays where it is.
#: 2. Legitimate app change — a real ``st.button`` was deleted from ``views/``.
#:    The walk found fewer sites because there *are* fewer sites.  Remedy: lower
#:    ``_BUTTON_SITE_FLOOR`` to the new count in the same change that removed
#:    the button.
#: 3. Deliberate narrowing of this claim — the press inventory or the claim's
#:    doc mirrors were trimmed on purpose.  Remedy: lower the floor with them,
#:    or retire the claim (and this constant) outright.
#:
#: The floor never moves on growth — finding *more* sites than the floor is
#: ordinary app growth and needs no edit at all.  The unsafe directions are
#: raising it to accommodate growth, or lowering it to silence cause 1, so the
#: message tells the reader to establish which cause applies before editing.
_BUTTON_SITE_FLOOR = 20

#: One ``st.button`` call site as ``(page, key, label, where, interpolated)``.
#: ``key`` and ``label`` may both be present: a keyed button still renders a
#: label, and the suite may legitimately press it either way.  ``where`` is
#: ``file:line``.  ``interpolated`` records how the label expression was
#: written: ``True`` for an f-string (``st.button(f"Run {job}")``), whose
#: rendered label *extends* the literal prefix the press list carries, so a
#: press may match it by prefix; ``False`` for a constant label, which is
#: matched by equality only — a press may not claim a button whose label it
#: only extends, that is a different button (ADR-008 / M1r).
ButtonSite = tuple[str | None, str | None, str | None, str, bool]

#: One press declaration as ``(page, kind, value, function, origin)``.  ``page``
#: is ``None`` when it cannot be resolved (an inline key press inside a helper
#: that navigates on the callee's behalf), which makes the press page-agnostic
#: rather than silently wrong; widget ``key=`` literals are unique across the
#: app by the §4.8 contract, so a key identifies its button regardless.
Press = tuple[str | None, str, str, str, str]


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    """Every module-level name in ``tree`` and the expression bound to it.

    Both ``x = …`` and ``x: T = …`` count, because the identifiers this gate
    reads — ``PAGES``, ``PAGE_KEYS``, the ``*BUTTONS`` inventories — are
    declared in whichever style the declaring file happens to use.

    Args:
        tree: Parsed module.

    Returns:
        dict: Name -> bound expression.
    """
    constants: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if node.value is None:
            continue
        for name in targets:
            constants[name] = node.value
    return constants


def _page_view_files() -> dict[str, str]:
    """Sidebar label -> ``views/<module>.py``, read from ``app.py``'s ``PAGES``.

    The nine labels are the only bridge between a *page* (how the smoke suite
    navigates, and how ``PAGE_KEYS[n]`` resolves) and a *file* (where ``ast``
    sees the button).  Read from the composition root rather than restated here
    so the mapping cannot drift from what the app actually dispatches.

    Returns:
        dict: The nine sidebar labels mapped to their view module filenames.

    Raises:
        AssertionError: If ``app.py`` no longer declares ``PAGES`` as a dict or
            one of its values does not name a ``views/module_*`` target.
    """
    pages = _module_constants(_parse(ROOT / "app.py")).get("PAGES")
    assert isinstance(pages, ast.Dict), "app.py no longer declares PAGES as a dict"
    files: dict[str, str] = {}
    for key, value in zip(pages.keys, pages.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        view = next(
            (part for part in ast.unparse(value).split(".") if part.startswith("module_")),
            None,
        )
        assert view, f"PAGES[{key.value!r}] does not name a views/module_* target"
        files[key.value] = f"{view}.py"
    return files


def _sidebar_page_keys() -> tuple[str, ...]:
    """The nine sidebar labels in display order, from ``views/base.py``.

    Returns:
        tuple[str, ...]: ``PAGE_KEYS`` as the view layer declares it.

    Raises:
        AssertionError: If ``views/base.py`` no longer declares ``PAGE_KEYS``.
    """
    value = _module_constants(_parse(ROOT / "views" / "base.py")).get("PAGE_KEYS")
    assert isinstance(value, (ast.Tuple, ast.List)), (
        "views/base.py no longer declares PAGE_KEYS as a tuple"
    )
    return tuple(
        element.value
        for element in value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )


def _static_button_label(node: ast.expr) -> tuple[str | None, bool]:
    """The label a button renders, and how its expression was written.

    An f-string contributes only its leading literal run — ``f"🚀 Train {name}"``
    yields ``"🚀 Train "`` — because that is the prefix a test can match with
    ``str.startswith`` without knowing the interpolated value; the returned flag
    marks it as *interpolated*, so callers know the rendered label may extend
    that prefix.  A plain constant label is returned with ``interpolated=False``
    and must be matched by equality: ``st.button("💾 Save Model Deluxe")`` is a
    different button from one that merely renders a label starting with
    ``"💾 Save Model"``, and a prefix match would let a press claim both
    (Reviewer M1r).  Anything that is neither returns ``(None, False)`` so the
    caller can report the site as unrecognised rather than quietly skip a
    button nobody can match.

    Args:
        node: The label expression of a ``st.button`` call.

    Returns:
        tuple: ``(statically known label text or None, interpolated)``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for segment in node.values:
            if isinstance(segment, ast.Constant) and isinstance(segment.value, str):
                parts.append(segment.value)
            else:
                break
        return "".join(parts), True
    return None, False


def _button_sites(view_sources: dict[str, str] | None = None) -> list[ButtonSite]:
    """Every ``st.button`` call in ``views/``, with the identities it can be pressed by.

    ``st.download_button`` is out of scope by construction: an export does not
    mutate anything, so conventions §7's "mutating action" does not reach it,
    and the scan requires the attribute to be exactly ``button``.

    Args:
        view_sources: filename -> source, used by the falsification controls.
            ``None`` reads the real ``views/`` package.

    Returns:
        list: One entry per button call site, in source order.
    """
    if view_sources is None:
        view_sources = {
            path.name: path.read_text(encoding="utf-8") for path in _source_files("views")
        }
    page_of_file = {filename: label for label, filename in _page_view_files().items()}
    sites: list[ButtonSite] = []
    for filename, source in view_sources.items():
        page = page_of_file.get(filename)
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "button"):
                continue
            key_node = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
            key = (
                key_node.value
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
                else None
            )
            label_text, interpolated = (
                _static_button_label(node.args[0])
                if node.args
                else (None, False)
            )
            sites.append(
                (page, key, label_text, f"{filename}:{node.lineno}", interpolated)
            )
    return sites


def _mentions_button(node: ast.AST) -> bool:
    """True if ``node`` reaches for ``.button`` — a call or the bare attribute.

    ``app.button(key=…)`` is a call; ``b for b in app.button`` is the attribute
    alone, and both spellings are how the smoke suite selects a widget.  Keying
    off the attribute rather than the call is what lets the ``b.label == "…"``
    comparisons be attributed to *buttons* specifically: this file also contains
    ``m.label.startswith(…)`` on a multiselect and ``s.label == …`` on a slider,
    and neither is a press.

    Args:
        node: Any AST node.

    Returns:
        bool: Whether a ``.button`` attribute appears anywhere beneath it.
    """
    return any(
        isinstance(child, ast.Attribute) and child.attr == "button"
        for child in ast.walk(node)
    )


def _button_bound_names(node: ast.AST) -> set[str]:
    """Local names bound to a button widget somewhere inside ``node``.

    Args:
        node: A function (or module-level statement) to scan.

    Returns:
        set[str]: Names whose ``.label`` reads are button labels.
    """
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and _mentions_button(child.value):
            bound.update(
                target.id for target in child.targets if isinstance(target, ast.Name)
            )
        elif isinstance(child, ast.comprehension) and _mentions_button(child.iter):
            bound.update(
                name.id for name in ast.walk(child.target) if isinstance(name, ast.Name)
            )
    return bound


def _page_of_node(node: ast.expr | None, page_keys: tuple[str, ...]) -> str | None:
    """Resolve a ``PAGE_KEYS[n]`` subscript — or a bare label — to a sidebar label.

    Anything else resolves to ``None``: guessing a page from an unrelated
    integer subscript would pair a press with the wrong view, and a wrong page
    is worse than an unknown one because it fails a correct press.

    Args:
        node: Expression to resolve.
        page_keys: The nine sidebar labels in display order.

    Returns:
        str | None: The sidebar label, or ``None`` if it cannot be resolved.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("PAGE_KEYS"):
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            if 0 <= index.value < len(page_keys):
                return page_keys[index.value]
    return None


def _parametrize_argnames(node: ast.expr) -> list[str]:
    """Names a ``pytest.mark.parametrize`` decorator binds, in order.

    Args:
        node: The decorator's first argument.

    Returns:
        list: The bound names.

    Raises:
        AssertionError: If the decorator spells its names in an unrecognised way.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [part.strip() for part in node.value.split(",") if part.strip()]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    raise AssertionError(f"unhandled parametrize argnames: {ast.unparse(node)}")


def _parametrize_rows(
    node: ast.expr, constants: dict[str, ast.expr]
) -> list[list[ast.expr]] | None:
    """Row values behind a ``parametrize`` ``argvalues`` expression.

    ``None`` means the sequence is *imported* rather than declared in the smoke
    file: ``pytest.mark.parametrize("label", PAGE_KEYS)`` sweeps the nine
    navigation labels, and reading those as button presses would credit nine
    presses no test performs.  Only inventories the file defines itself are
    press declarations.

    Args:
        node: The decorator's second argument.
        constants: Module-level bindings of the file being parsed.

    Returns:
        list | None: Rows of cells, or ``None`` for an imported sequence.

    Raises:
        AssertionError: If ``argvalues`` is spelled in an unrecognised way.
    """
    if isinstance(node, ast.Name):
        if node.id not in constants:
            return None
        node = constants[node.id]
    if isinstance(node, ast.Call):  # sorted(DIRTY_FRAME_BUTTONS.items())
        assert isinstance(node.func, ast.Name) and node.func.id == "sorted", (
            f"unhandled argvalues call: {ast.unparse(node)}"
        )
        assert len(node.args) == 1 and isinstance(node.args[0], ast.Call), (
            f"unhandled argvalues call: {ast.unparse(node)}"
        )
        items = node.args[0]
        assert isinstance(items.func, ast.Attribute) and items.func.attr == "items", (
            f"unhandled argvalues call: {ast.unparse(node)}"
        )
        assert isinstance(items.func.value, ast.Name), (
            f"unhandled argvalues call: {ast.unparse(node)}"
        )
        mapping = constants[items.func.value.id]
        assert isinstance(mapping, ast.Dict), (
            f"sorted() is applied to something other than a declared dict: "
            f"{ast.unparse(node)}"
        )
        return [
            [key, value]
            for key, value in zip(mapping.keys, mapping.values, strict=True)
            if key is not None and value is not None
        ]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            list(element.elts) if isinstance(element, (ast.Tuple, ast.List)) else [element]
            for element in node.elts
        ]
    raise AssertionError(f"unhandled parametrize argvalues: {ast.unparse(node)}")


def _presses_in(
    node: ast.stmt, constants: dict[str, ast.expr], page_keys: tuple[str, ...], function: str
) -> list[Press]:
    """Every press declared inside one function or module-level statement.

    Five idioms are recognised, and they are all present in the live file:
    ``parametrize`` tables keyed and labelled, a label dict passed through
    ``sorted(…​.items())``, an inline ``app.button(key=…)`` literal, a
    ``b.label == "…"`` comparison, and ``b.label.startswith("…")`` for the
    f-string Train label.  An idiom this does not recognise cannot credit
    coverage, so it fails as an *uncovered button* — the gate degrades to red
    rather than to silent acceptance.

    Args:
        node: The function or statement to scan.
        constants: Module-level bindings of the file being parsed.
        page_keys: The nine sidebar labels in display order.
        function: Name to attribute the presses to, for the wiring check.

    Returns:
        list: Press declarations found under ``node``.
    """
    presses: list[Press] = []

    # The function's own page, when exactly one PAGE_KEYS subscript identifies it.
    seen_pages: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            resolved = _page_of_node(child, page_keys)
            if resolved is not None:
                seen_pages.add(resolved)
    page = seen_pages.pop() if len(seen_pages) == 1 else None

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and ast.unparse(decorator.func).endswith("parametrize")
            ):
                continue
            if len(decorator.args) < 2:
                continue
            argnames = _parametrize_argnames(decorator.args[0])
            rows = _parametrize_rows(decorator.args[1], constants)
            if rows is None:
                continue
            for row in rows:
                row_page = page
                for index, name in enumerate(argnames):
                    if index >= len(row):
                        continue
                    cell = row[index]
                    if name == "page_key":
                        row_page = _page_of_node(cell, page_keys)
                    elif name == "label" and isinstance(cell, ast.Constant):
                        if isinstance(cell.value, str):
                            presses.append(
                                (row_page, "label", cell.value, function, "parametrized table")
                            )
                    elif name.endswith("key") and isinstance(cell, ast.Constant):
                        if isinstance(cell.value, str):
                            presses.append(
                                (row_page, "key", cell.value, function, "parametrized table")
                            )

    for child in ast.walk(node):
        # Inline app.button(key="…") — a press written out at the call site.
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "button"
        ):
            for keyword in child.keywords:
                value = keyword.value
                if keyword.arg == "key" and isinstance(value, ast.Constant):
                    if isinstance(value.value, str):
                        presses.append(
                            (page, "key", value.value, function, "inline key press")
                        )

    bound = _button_bound_names(node)
    for child in ast.walk(node):
        # b.label == "…" on a name bound to a button widget.
        if (
            isinstance(child, ast.Compare)
            and isinstance(child.left, ast.Attribute)
            and child.left.attr == "label"
            and isinstance(child.left.value, ast.Name)
            and child.left.value.id in bound
        ):
            for comparator in child.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    presses.append(
                        (page, "label", comparator.value, function, "label comparison")
                    )
        # b.label.startswith("…") — the idiom the f-string Train label needs.
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "startswith"
            and isinstance(child.func.value, ast.Attribute)
            and child.func.value.attr == "label"
            and isinstance(child.func.value.value, ast.Name)
            and child.func.value.value.id in bound
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ):
            presses.append((page, "label", child.args[0].value, function, "label prefix"))
    return presses


def _pressed_buttons(source: str | None = None) -> list[Press]:
    """Every press declared in ``tests/test_app_smoke.py``.

    Args:
        source: File text, for the falsification controls. ``None`` reads the
            real file.

    Returns:
        list: Press declarations in source order.
    """
    if source is None:
        source = (ROOT / "tests" / "test_app_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="test_app_smoke.py")
    constants = _module_constants(tree)
    page_keys = _sidebar_page_keys()
    presses: list[Press] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        elif _mentions_button(node):
            function = "<module>"
        else:
            continue
        presses.extend(_presses_in(node, constants, page_keys, function))
    return presses


def _press_matches_site(
    press_page: str | None, kind: str, value: str, site: ButtonSite
) -> bool:
    """Whether one press reaches one button site.

    Matching follows the rule :func:`_static_button_label` records (reviewer
    M1r): a **constant** label matches by equality only, because
    ``st.button("💾 Save Model Deluxe")`` is a different button from one that
    merely renders a label starting with ``"💾 Save Model"``, and a prefix match
    would let a press claim both.  An **interpolated** (f-string) label matches
    by prefix, since its rendered text extends the literal run the extractor
    captured and a press cannot know the interpolated value.  Keys match by
    equality — widget ``key=`` literals are unique across the app by the §4.8
    contract.  An empty press value matches nothing: equality and ``startswith``
    against ``""`` are both trivially true, which would make every button look
    covered.

    Args:
        press_page: Page the press was declared on, or ``None`` if unresolved.
        kind: ``"key"`` or ``"label"``.
        value: The pressed key or label.
        site: The button site to test against.

    Returns:
        bool: Whether the press reaches the site.
    """
    site_page, key, label, _where, interpolated = site
    if not value:
        return False
    if press_page is not None and site_page is not None and press_page != site_page:
        return False
    if kind == "key":
        return key == value
    if label is None:
        return False
    if interpolated:
        return label.startswith(value)
    return label == value


def _uncovered_buttons(sites: list[ButtonSite], presses: list[Press]) -> list[str]:
    """Button sites no declared press reaches — one direction of the contract.

    Args:
        sites: Button sites from :func:`_button_sites`.
        presses: Press declarations from :func:`_pressed_buttons`.

    Returns:
        list: One message per uncovered site.
    """
    problems: list[str] = []
    for site in sites:
        page, key, label, where, _interpolated = site
        if key is None and label is None:
            problems.append(
                f"{where}: the button has neither a literal key nor a literal label, "
                "so no test can name it — teach this gate the new idiom"
            )
            continue
        if any(
            _press_matches_site(press_page, kind, value, site)
            for press_page, kind, value, _function, _origin in presses
        ):
            continue
        described = f"key={key!r}" if key is not None else f"label={label!r}"
        problems.append(f"{where}: {page} button ({described}) is never pressed")
    return problems


def _dead_button_presses(sites: list[ButtonSite], presses: list[Press]) -> list[str]:
    """Press declarations that reach no button — the other direction.

    A press naming a button that does not exist is dead coverage: it reads as a
    protection the suite does not provide, which is the exact defect class
    reviewer M1 raised.

    Args:
        sites: Button sites from :func:`_button_sites`.
        presses: Press declarations from :func:`_pressed_buttons`.

    Returns:
        list: One message per press that matches nothing.
    """
    problems: list[str] = []
    for page, kind, value, function, origin in presses:
        if any(
            _press_matches_site(page, kind, value, site) for site in sites
        ):
            continue
        problems.append(
            f"{function} [{origin}]: press {kind}={value!r} on {page} matches no "
            "st.button in views/"
        )
    return problems


def _unwired_presses(source: str | None = None) -> list[str]:
    """Press declarations whose body never calls ``.click()``.

    The wiring half: an inventory can list twenty correct identifiers and still
    protect nothing if the test that consumes it renders a page and stops.
    That is the same "declared but not performed" shape as the missing gate
    itself, one level down.

    Args:
        source: File text, for the falsification controls. ``None`` reads the
            real file.

    Returns:
        list: One message per declaration with no click behind it.
    """
    if source is None:
        source = (ROOT / "tests" / "test_app_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="test_app_smoke.py")
    bodies = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    problems: list[str] = []
    for function in sorted({press[3] for press in _pressed_buttons(source)}):
        body = bodies.get(function)
        if body is None:
            problems.append(
                f"presses are attributed to {function!r}, which is not a def in "
                "tests/test_app_smoke.py — a press must live in a test body"
            )
            continue
        if not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "click"
            for child in ast.walk(body)
        ):
            problems.append(
                f"{function} declares presses but never calls .click() — the "
                "declaration credits a coverage no test performs"
            )
    return problems


def test_every_mutating_button_in_the_views_is_pressed_here() -> None:
    """conventions §7: every ``st.button`` in ``views/`` is pressed by this suite.

    The claim has two halves, stated separately because run-005 reviewer M1
    found them asserted as one while neither existed: the docstring of
    ``tests/test_app_smoke.py`` promised that a button "cannot slip in
    unpressed", and no test anywhere enforced it.

    * **Structural (this test, tier T0).**  Every ``st.button`` call in
      ``views/`` is accounted for by a press declared in
      ``tests/test_app_smoke.py``, matched in **both** directions: a button
      nobody presses, a press naming no button, and a press whose test body
      never calls ``.click()`` each fail.  Both sides are derived from the tree
      — no button inventory is duplicated here — so neither can be widened
      alone (roadmap deviation 0b.1).  ``ast`` only, so §10.1's stdlib-plus-
      pytest tier holds.
    * **Runtime (``tests/test_app_smoke.py``, tier T1).**  Those declared
      presses are actually performed with ``AppTest``, each asserted to report
      success or a rendered ``FrameworkError``.  This test cannot see a handler
      execute; that half belongs to the action tier, and neither half implies
      the other.

    Out of scope by construction: ``st.download_button`` (an export mutates
    nothing) and pages that own no button.

    Returns:
        None

    Raises:
        AssertionError: If any button is unpressed, any press is dead, any
            press is unwired, or the extractor finds fewer sites than the
            measured floor.
    """
    sites = _button_sites()
    presses = _pressed_buttons()

    assert len(sites) >= _BUTTON_SITE_FLOOR, (
        f"the walk found {len(sites)} st.button sites in views/, below the measured "
        f"floor of {_BUTTON_SITE_FLOOR}: the extractor has regressed, so this gate "
        "would now reconcile less than the tree contains"
    )
    uncovered = _uncovered_buttons(sites, presses)
    assert not uncovered, (
        "a mutating action with no pressed button (conventions §7; reviewer "
        "MINOR-1, run-005 M1):\n  " + "\n  ".join(uncovered)
    )
    dead = _dead_button_presses(sites, presses)
    assert not dead, (
        "a press declaration that matches no button — dead coverage claiming a "
        "protection it does not provide:\n  " + "\n  ".join(dead)
    )
    unwired = _unwired_presses()
    assert not unwired, (
        "a press declaration whose test body never clicks it:\n  "
        + "\n  ".join(unwired)
    )


def test_the_button_press_gate_would_catch_an_undeclared_button() -> None:
    """Falsify the structural half: a button with no press must fail the gate.

    A control on the reconciliation itself, not on the live tree: a gate that
    cannot fail is indistinguishable from one that is not wired up.  The
    synthetic button is appended to a real view, so the extractor has to find it
    in realistic surroundings, and the reverse direction is asserted to stay
    clean so the control cannot pass by tripping both halves.

    Returns:
        None

    Raises:
        AssertionError: If the synthetic button is reported as covered.
    """
    sources = {
        path.name: path.read_text(encoding="utf-8") for path in _source_files("views")
    }
    sources["module_01_data.py"] += (
        '\n\nif st.button("Synthetic Undeclared Button"):\n    pass\n'
    )

    sites = _button_sites(sources)
    presses = _pressed_buttons()
    uncovered = _uncovered_buttons(sites, presses)
    assert any("Synthetic Undeclared Button" in line for line in uncovered), (
        "an st.button that no test presses was reported as covered; the gate "
        "cannot detect a newly added unpressed button"
    )
    assert not _dead_button_presses(sites, presses), (
        "the control tripped the reverse direction too, so it does not isolate "
        "the undeclared-button failure"
    )


def test_the_button_press_gate_would_catch_a_dead_press() -> None:
    """Falsify the reverse half: a press naming no button must fail the gate.

    Both idioms that carry a *label* rather than a key are exercised, because a
    key-only control would leave the label matching — the half that has to cope
    with f-strings and page attribution — unproven.  The live tree's own
    coverage is asserted to stay intact, so the control isolates the reverse
    direction instead of failing everything.

    Returns:
        None

    Raises:
        AssertionError: If either synthetic press is accepted as real coverage.
    """
    source = (ROOT / "tests" / "test_app_smoke.py").read_text(encoding="utf-8")
    source += (
        "\n\ndef test_synthetic_presses_matching_nothing() -> None:\n"
        '    app.button(key="synthetic_key_that_matches_nothing").click().run()\n'
        "    button = next(\n"
        "        (b for b in app.button if b.label == "
        '"Synthetic Label That Matches Nothing"),\n'
        "        None,\n"
        "    )\n"
        "    button.click().run()\n"
    )

    presses = _pressed_buttons(source)
    dead = _dead_button_presses(_button_sites(), presses)
    assert any("synthetic_key_that_matches_nothing" in line for line in dead), (
        "a key press matching no button was accepted; dead coverage cannot be detected"
    )
    assert any("Synthetic Label That Matches Nothing" in line for line in dead), (
        "a label press matching no button was accepted; dead coverage cannot be detected"
    )
    assert not _uncovered_buttons(_button_sites(), presses), (
        "adding a synthetic press disturbed the live tree's coverage, so this "
        "control does not isolate the reverse direction"
    )


def test_the_button_press_gate_would_catch_an_unwired_press() -> None:
    """Falsify the wiring half: a declared press that never clicks must fail.

    The declaration is otherwise perfect — right page, right key, real button —
    which is precisely the shape that would slip through a coverage-only
    cross-reference: it reads as protection while performing nothing.  This is
    the defect class of the run-003 ``render_errors`` symbol that was only ever
    bound inside a handler no test entered.

    Returns:
        None

    Raises:
        AssertionError: If the never-clicked declaration is accepted.
    """
    source = (ROOT / "tests" / "test_app_smoke.py").read_text(encoding="utf-8")
    source += (
        "\n\nSYNTHETIC_UNCLICKED_BUTTONS = (\n"
        '    ("1️⃣ Data Management", "run_single_stress"),\n'
        ")\n\n"
        '@pytest.mark.parametrize(\n'
        '    ("page_key", "button_key"),\n'
        "    SYNTHETIC_UNCLICKED_BUTTONS,\n"
        ")\n"
        "def test_synthetic_press_declared_but_never_clicked(page_key, button_key) -> None:\n"
        "    assert page_key and button_key\n"
    )

    unwired = _unwired_presses(source)
    assert any(
        "test_synthetic_press_declared_but_never_clicked" in line for line in unwired
    ), "a parametrized press whose body never calls .click() was accepted as coverage"


def test_the_smoke_docstring_cites_gates_that_exist() -> None:
    """run-005 M1: a docstring may not claim protection from an absent test.

    ``tests/test_app_smoke.py`` once stated that
    ``test_every_mutating_button_in_the_views_is_pressed_here`` enforced its
    central rule "structurally", while that identifier occurred exactly once in
    the repository — in the sentence itself.  A claim reading as a closed
    question is the failure mode ADR-011 item 5 records, so the backticked
    ``test_*`` names in that module docstring are now checked against every
    ``def test_*`` under ``tests/``.

    Returns:
        None

    Raises:
        AssertionError: If the docstring cites no gate at all, or cites one that
            does not exist.
    """
    doc = ast.get_docstring(_parse(ROOT / "tests" / "test_app_smoke.py")) or ""
    cited = set(re.findall(r"`+(test_[A-Za-z0-9_]+)`+", doc))
    assert cited, (
        "tests/test_app_smoke.py's module docstring cites no test by name; it is "
        "where the two-half press claim lives, so it must keep naming its gate"
    )
    known = {
        child.name
        for path in sorted((ROOT / "tests").rglob("*.py"))
        if "__pycache__" not in path.parts
        for child in ast.walk(_parse(path))
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(cited - known)
    assert not missing, (
        "tests/test_app_smoke.py's module docstring cites tests that do not exist "
        f"(run-005 M1's defect): {missing}"
    )


def test_core_kernel_modules_exist() -> None:
    """§2.2: the nine ``core`` kernel modules are all present."""
    missing = [
        name
        for name in (
            "config",
            "errors",
            "logging_config",
            "model_io",
            "perturbations",
            "reporters",
            "scoring",
            "state",
            "validation",
        )
        if not (ROOT / "core" / f"{name}.py").is_file()
    ]
    assert not missing, f"core modules missing: {missing}"


#: The composition root.  §2.1 grants this file "everything", so it is the one
#: root module exempt from the third-party hygiene check in
#: :func:`test_root_module_respects_the_matrix`.
COMPOSITION_ROOT = "app.py"

#: Top-level modules architecture.md §2.2 declares: the composition root and the
#: command-line entry point.  A root-level module belongs to no §2.1 layer, so it
#: escapes the matrix, the cycle detector, the app-import check and the widget-key
#: contract simultaneously — which is why the list is explicit, cross-checked
#: against the specification, and small.
#:
#: This constant is *not* the authority on its own —
#: :func:`test_root_modules_match_architecture_spec` requires it to equal what
#: §2.2's layout block actually lists, so widening it here cannot pass without the
#: specification changing too.  That cross-check exists because a CLI stub was
#: once added to this tree with this constant widened to admit it, while §2.2 was
#: left untouched: the gate went green and its own failure message then asserted
#: something the specification did not say.
#:
#: ``main.py`` is declared (ADR-012) and is a real CLI with its own contract
#: (§4.9) and its own suite (``tests/test_cli.py``).  What was rejected twice was
#: not a second entry point — it was a stub that printed nothing and answered ``0``
#: to every input, including invalid ones.  ``tests/test_cli.py`` fails that stub.
ROOT_MODULES: tuple[str, ...] = (COMPOSITION_ROOT, "main.py")


def _architecture_declared_root_modules() -> tuple[str, ...]:
    """Parse architecture.md §2.2's layout block for declared top-level modules.

    Returns the unindented ``*.py`` entries of the §2.2 fenced layout, so the
    allowlist above can be checked against the document it claims to quote rather
    than trusting the constant's own comment.  Directory entries (``core/``) are
    skipped: they end in ``/``, not ``.py``.

    Raises:
        AssertionError: if §2.2 or its layout block cannot be located — a parser
            that silently found nothing would turn the cross-check into a gate that
            passes vacuously, which is the failure mode ADR-010 warns against.
    """
    spec = ROOT / ".ai" / "project" / "architecture.md"
    assert spec.is_file(), f"missing specification: {spec}"
    text = spec.read_text(encoding="utf-8")

    start = text.find("### 2.2")
    assert start != -1, "architecture.md has no '### 2.2' layout section"
    fence_start = text.find("```", start)
    assert fence_start != -1, "architecture.md §2.2 has no fenced layout block"
    fence_end = text.find("```", fence_start + 3)
    assert fence_end != -1, "architecture.md §2.2 layout fence is unterminated"

    declared: list[str] = []
    for line in text[fence_start + 3 : fence_end].splitlines():
        if line[:1].isspace() or not line.strip():
            continue  # indented entry = a file inside a package, not a root module
        entry = line.split("#", 1)[0].strip()
        if entry.endswith(".py"):
            declared.append(entry)

    assert declared, (
        "no top-level '*.py' entry found in the architecture.md §2.2 layout block; "
        "the parser would let test_root_modules_match_architecture_spec pass "
        "vacuously"
    )
    return tuple(sorted(declared))


def test_root_modules_are_exactly_the_declared_set() -> None:
    """§2.2: the top-level modules are the declared set — nothing more.

    §2.1's matrix is defined *per layer* and the root modules are its only
    members, so an **undeclared** root-level module would have no layer to be
    checked against: importing ``streamlit`` and ``views`` from it would satisfy
    every rule below while shipping a file nothing had reviewed.  That is not
    hypothetical — an undeclared CLI stub of exactly that shape was added to this
    tree and passed the whole gate.

    This test converts that blind spot into a loud failure, so a new root-level
    module has to be declared in §2.2 (and given a layer) before it can be added.

    **Renamed** from ``test_only_app_py_is_a_top_level_module`` when ADR-012
    declared ``main.py`` as the second root module.  The assertion is unchanged;
    the old name asserted something the specification stopped saying, and a gate
    whose name misdescribes its invariant is a maintenance trap.

    Compared as **sets**.  This was a tuple comparison until the mutation probe
    (``evidence/main_py_gate_probe.py``, probe 3) declared a root module whose
    name sorts *before* ``main.py`` and turned the gate red for two fully
    consistent sources: ``found`` is sorted, ``ROOT_MODULES`` is written in
    declaration order, and alphabetical ordering of the constant is not part of
    the architecture.  ``test_root_modules_match_architecture_spec`` already
    compared sets, so this also makes the two halves of the §2.2 gate agree.
    """
    found = tuple(sorted(p.name for p in _root_modules()))
    assert set(found) == set(ROOT_MODULES), (
        "architecture.md §2.2 declares "
        f"{sorted(ROOT_MODULES)} as the only top-level module(s); found {list(found)}. "
        "A root-level module is outside the §2.1 import matrix (it belongs to no "
        "layer), so adding one silently removes it from every gate in this file. "
        "Declare it in architecture.md §2.2 with its layer and extend "
        "ROOT_MODULES here, or move it under an existing package."
    )


def test_root_modules_match_architecture_spec() -> None:
    """§2.2: :data:`ROOT_MODULES` must equal what the specification declares.

    This is the second half of :func:`test_root_modules_are_exactly_the_declared_set`,
    and the half that was missing.  That test compares the tree against a
    constant defined in this file, so widening the constant alone made it green —
    while its failure message went on claiming "architecture.md §2.2 declares
    …", a sentence the document did not contain.  A gate whose remediation is
    half-completed is worse than no gate, because it certifies a declaration that
    does not exist.

    Asserted in both directions, so neither the constant nor the specification can
    drift alone:

    * every declared module in §2.2 must be in :data:`ROOT_MODULES`;
    * every module in :data:`ROOT_MODULES` must be declared in §2.2.

    The first direction also means adding a real second entry point is possible —
    it just requires a specification change, a matrix row for it, and the hygiene
    check in :func:`test_root_module_respects_the_matrix`.  The point is to make
    the addition deliberate, not to forbid it forever.
    """
    declared = _architecture_declared_root_modules()
    assert set(ROOT_MODULES) == set(declared), (
        f"ROOT_MODULES={list(ROOT_MODULES)} but architecture.md §2.2 declares "
        f"{list(declared)}. Both must change together: a root-level module that is "
        "not declared in §2.2 has no layer, and a declared one that is not in "
        "ROOT_MODULES escapes every gate in this file."
    )


def test_root_module_respects_the_matrix() -> None:
    """§2.1/§2.2: a root module other than ``app.py`` may not pull in third-party code.

    ``app.py`` is exempt because §2.1 grants it "everything" — it is the
    composition root and importing ``streamlit`` is its job.  Any *other*
    root-level module belongs to no layer, so it gets no dependency grant: it
    may import stdlib and project packages only.

    This is the companion to :func:`test_root_modules_are_exactly_the_declared_set`,
    and it deliberately keys off :data:`COMPOSITION_ROOT` rather than off
    :data:`ROOT_MODULES`.  Skipping the declared set instead would make this test
    unsatisfiable-by-failure: every declared module would be skipped, so widening
    the declaration would quietly exempt the new module from hygiene too.  As
    written, a declared second root module has to pass the same third-party check
    as the four packages before it ships — which is the whole point of widening
    the declaration on purpose.

    The declared ``main.py`` is held to a *tighter* rule still — stdlib and
    ``core`` only, at any scope, so ``--help`` works on a bare interpreter — and
    that is asserted by ``tests/test_cli.py``.  This check remains the coarse
    backstop that applies to every root module.
    """
    offenders: list[str] = []
    for path in _root_modules():
        if path.name == COMPOSITION_ROOT:
            continue
        for name in _imports(_parse(path)):
            root = _top(name)
            if root in STDLIB or root in PACKAGES or root == "app":
                continue
            offenders.append(f"{path.name} imports {name}")
    assert not offenders, (
        f"only {COMPOSITION_ROOT} may import third-party roots; every other root "
        "module may import stdlib and project packages only:\n  " + "\n  ".join(offenders)
    )


def test_app_is_a_thin_composition_root() -> None:
    """Conventions §1 / §4.8: ``app.py`` is wiring only, target <= 200 lines.

    "No page bodies" is the substantive half.  ``app.py`` legitimately defines
    small wiring helpers (``_service_factories``, ``_assert_page_contract``), so
    the check is not "defines no functions" — it is that no function in the
    composition root renders UI or computes domain results.  A page body would
    necessarily call ``st.*`` beyond the five startup calls, or reach into
    ``modules`` for a computation, and both are asserted below.
    """
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    lines = len(text.splitlines())
    assert lines <= 200, f"app.py is {lines} lines; the target is <= 200"

    tree = ast.parse(text)
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    offenders: list[str] = []
    for fn in funcs:
        for node in ast.walk(fn):
            # A page body renders through streamlit.
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "st"
            ):
                offenders.append(
                    f"app.py::{fn.name} calls st.{node.attr}; UI bodies live in views/"
                )
    assert not offenders, "page bodies leaked into the composition root:\n  " + "\n  ".join(
        offenders
    )


def test_app_dispatches_all_nine_pages() -> None:
    """§4.8: dispatch is an explicit ordered mapping covering all nine pages."""
    tree = _parse(ROOT / "app.py")
    pages: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PAGES"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        # An `ast.Dict` always has keys/values of equal length, so `strict=True`
        # documents that as an assumption rather than silently truncating.
        for key, val in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                pages[key.value] = _attr_chain(val)
    expected = _contract()["page_labels"]
    assert list(pages) == expected, (
        f"PAGES mapping must list the nine sidebar labels in order; got {list(pages)}"
    )
    for label, chain in pages.items():
        assert len(chain) == 2 and chain[1] == "render", (
            f"PAGES[{label!r}] must point at a view render() callable, got {chain}"
        )
        assert re.fullmatch(r"module_0[1-9]_\w+", chain[0]), (
            f"PAGES[{label!r}] must come from a views.module_0N_* import, got {chain}"
        )


# ── Wave 6 gate: README.md matches the final tree ─────────────────────────────


def test_readme_reproduces_the_frozen_page_labels_verbatim() -> None:
    """Wave 6's gate is "README.md matches the final tree"; this is its teeth.

    The nine sidebar labels are the **persisted session contract** (§4.8), and
    ``README.md`` is where a reader is most likely to copy one from.  Measured
    before this gate existed, the README published a paraphrase of the contract:
    every label had lost its emoji prefix, and ``3️⃣ Prediction & Confidence``
    and ``9️⃣ Visualization & Reports`` were written as "Prediction and
    Confidence" / "Visualization and Reports" — "and" where the contract has
    ``&``.

    Nothing caught it, and nothing would have: no roadmap assertion covers the
    README's accuracy, so the Wave 6 gate was satisfied by reading it once.  This
    is the same shape as the two gates added above — a convention that reads as
    enforced and is not — so it is enforced instead of merely recorded
    (roadmap deviation 13).

    Deliberately scoped to the label strings rather than a prose diff.  Asserting
    free text would make the gate rot on every reword, and a gate that fails for
    a reason unrelated to what it protects is a gate people disable.

    Returns:
        None.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [label for label in _contract()["page_labels"] if label not in readme]
    assert not missing, (
        "README.md must reproduce the nine sidebar labels verbatim -- they are the "
        "persisted session contract (architecture.md §4.8) and conventions §7 "
        "forbids reworded emoji-prefixed strings. Missing from README.md:\n  "
        + "\n  ".join(missing)
        + "\nExpected exactly (from views.base.PAGE_KEYS via "
        "tests/fixtures/widget_keys.json)."
    )


def test_the_readme_label_gate_would_catch_a_paraphrase() -> None:
    """Falsify :func:`test_readme_reproduces_the_frozen_page_labels_verbatim`.

    The gate is a plain substring search, so its risk is not that the pattern is
    wrong but that it matches too loosely -- ``"Data Management"`` is a substring
    of the real label, so a de-emoji'd README would pass on a naive reading of
    "does the label appear".  These controls require the **full** string, emoji
    prefix and ``&`` included, and confirm the substring relationship that makes
    the naive version vacuous.

    Returns:
        None.
    """
    labels = _contract()["page_labels"]

    # Positive control against the live README: every full label is present.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert all(label in readme for label in labels), (
        "the README is missing at least one frozen label; this falsification test "
        "cannot run until it is corrected"
    )

    # The paraphrase that was actually published must NOT satisfy the gate.
    paraphrase = "1. Prediction and Confidence\n2. Visualization and Reports\n"
    assert not any(label in paraphrase for label in labels), (
        "the de-emoji'd, 'and'-for-'&' paraphrase satisfied the substring search; "
        "the gate is matching too loosely to be a gate"
    )

    # And the substrings the paraphrase is built from are genuinely present in
    # the real labels -- which is what made the naive check vacuous.
    assert "Data Management" in labels[0] and "Data Management" != labels[0], (
        "labels[0] is expected to be an emoji-prefixed form of its own tail; if "
        "this no longer holds, re-derive the falsification control above"
    )


# ── §10.1 tier T0 self-check ──────────────────────────────────────────────────

#: §10.1 declares four files as tier **T0** — stdlib + pytest only. They are the
#: gates that must stay runnable in a bare environment; a security or layering
#: gate that cannot run is not a gate (ADR-010.1).
T0_TEST_FILES: tuple[str, ...] = (
    "tests/test_architecture.py",
    "tests/test_model_io.py",
    "tests/test_config.py",
    "tests/test_cli.py",
)

#: Every project module a tier-T0 file is *allowed* to reach (ADR-026.2) —
#: ``core/*`` and ``main.py`` only, never ``modules/``, ``utils/`` or ``views/``.
#:
#: The declaration is **complete over ``core/``** on purpose: before this, only
#: the three modules the T0 files happened to import were listed, so the first
#: import of a fourth one escaped the transitive check entirely.  That is the
#: Critic M9 defect — the file-level gate keyed off :data:`PACKAGES`, i.e. all
#: four packages, so ``from core.validation import ...`` in a T0 file was
#: admitted by name *and* unexamined for what it drags in.
#:
#: :func:`test_every_core_module_is_declared_in_the_t0_module_set` keeps this
#: list in step with the directory, so a ninth or tenth ``core`` module cannot
#: arrive ungoverned.
T0_PROJECT_MODULES: tuple[str, ...] = (
    "core/__init__.py",
    "core/config.py",
    "core/errors.py",
    "core/logging_config.py",
    "core/model_io.py",
    "core/perturbations.py",
    "core/reporters.py",
    "core/scoring.py",
    "core/state.py",
    "core/validation.py",
    "main.py",
)

#: The subset of :data:`T0_PROJECT_MODULES` importable with **nothing but the
#: standard library**.  A T0 test file may import these and nothing else project-
#: local, transitively (``core.model_io`` reaches ``core.config`` and
#: ``core.errors``; both are in the set).
#:
#: ``core/scoring.py`` is here because ADR-021 made it stdlib-only by design —
#: that is what lets the scoring arithmetic be gated without dragging numpy in.
#: ``core/validation.py``, ``core/perturbations.py`` and ``core/state.py``
#: are deliberately **absent**: they import numpy/pandas at module scope, which
#: is perfectly legal under §2.1 and precisely why a T0 file may not touch them.
T0_STACK_FREE_MODULES: frozenset[str] = frozenset(
    {
        "core/__init__.py",
        "core/config.py",
        "core/errors.py",
        "core/logging_config.py",
        "core/model_io.py",
        "core/reporters.py",
        "core/scoring.py",
        "main.py",
    }
)

#: Project modules that legitimately need the scientific stack at import time.
#: Every entry is in :data:`T0_PROJECT_MODULES` but not in
#: :data:`T0_STACK_FREE_MODULES`, and :func:`test_the_stack_needing_class_is_
#: really_stack_needing` asserts each one genuinely needs it — so the two classes
#: cannot drift apart unnoticed.
T0_STACK_NEEDING_MODULES: frozenset[str] = frozenset(T0_PROJECT_MODULES) - T0_STACK_FREE_MODULES

#: The roots a T0 file must never need. Asserted against the *import-time* graph
#: for project modules, and against the whole file for test files (a deferred
#: ``import numpy`` inside a T0 test body would still run under pytest).
SCIENTIFIC_STACK: frozenset[str] = frozenset(
    {
        "numpy", "pandas", "sklearn", "scipy", "xgboost", "plotly",
        "streamlit", "matplotlib", "seaborn", "altair", "jinja2",
        "reportlab", "joblib",
    }
)


def _module_closure(relpath: str) -> set[str]:
    """Project modules reachable from ``relpath``'s module-level imports.

    Follows ``core``/``modules``/``utils``/``views``/``main`` imports that run at
    *import* time — :func:`_module_level_imports` stops at every ``FunctionDef``,
    so ``core.config``'s deliberately deferred ``core.errors`` edge is treated
    the same way it is everywhere else in this file.

    Args:
        relpath: Repository-relative path of the file to start from.

    Returns:
        Repository-relative paths of every project module the file can reach,
        including itself if it is a project module.
    """
    start = ROOT / relpath
    seen: set[str] = set()
    frontier = [start] if start.is_file() else []
    while frontier:
        path = frontier.pop()
        key = str(path.relative_to(ROOT))
        if key in seen:
            continue
        seen.add(key)
        for name in _module_level_imports(_parse(path)):
            if _top(name) not in PACKAGES and _top(name) != "main":
                continue
            parts = name.split(".")
            candidate = ROOT / parts[0] / f"{parts[1]}.py" if len(parts) > 1 else (
                ROOT / parts[0] / "__init__.py"
            )
            if candidate.is_file():
                frontier.append(candidate)
    # The starting file is a *test* module, not a project module; dropping it
    # (and anything else outside the four packages) keeps the result a set of
    # project modules rather than a set that includes ``tests/…`` itself.
    return {key for key in seen if _top(key) in PACKAGES or _top(key) == "main"}


@pytest.mark.parametrize("relpath", T0_TEST_FILES)
def test_t0_test_file_needs_only_stdlib_and_pytest(relpath: str) -> None:
    """Guard the guard: every tier-T0 test file must stay stdlib + pytest only.

    Previously only this file checked itself, which left the other three T0
    files unguarded: adding ``import numpy`` to ``test_model_io.py`` would have
    promoted the *security* suite to T1 silently, on an interpreter where the
    scientific stack is absent, and the suite would fail to collect — the exact
    failure ADR-010.1 was written to prevent, discovered by reading the tier
    list instead of by running it.

    Project-local imports are permitted only from ``core`` and ``main``
    (ADR-026.2).  It used to read ``not in PACKAGES``, i.e. all four packages,
    while this docstring claimed "``core``, ``main``" — so the gate and its own
    stated rule disagreed, and importing ``modules.x`` in a T0 file was admitted
    by name.  Which *specific* ``core`` modules are reachable is the second half,
    in :func:`test_no_t0_test_file_reaches_a_stack_needing_module`.
    """
    path = ROOT / relpath
    assert path.is_file(), f"§10.1 declares {relpath} as tier T0 but it is missing"
    offenders = [
        f"{relpath} imports {name}"
        for name in _imports(_parse(path))
        if _top(name) not in STDLIB
        and _top(name) not in {"pytest", "__future__"}
        and _top(name) not in {"core", "main"}
    ]
    assert not offenders, (
        "tier T0 must need only the standard library and pytest "
        "(architecture.md §10.1, ADR-010.1):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("relpath", sorted(T0_STACK_FREE_MODULES))
def test_t0_project_module_needs_no_scientific_stack(relpath: str) -> None:
    """The modules T0 tests import must import without the scientific stack.

    This is the *load-bearing* half of the T0 tier, and it is what ADR-010.1
    (§4.5.6, lazy estimator gate) and ADR-011.3 (§4.5.11, ``library_versions``
    via ``importlib.metadata``) exist to preserve. A module-level
    ``from sklearn.base import BaseEstimator`` in ``core/model_io.py`` would
    transitively import sklearn + numpy + scipy and drag the security suite to
    T1 with no visible failure anywhere.

    Import-time graph only: a function-body import is deferred until call time
    and is explicitly permitted by §4.5.6, so this walks with
    :func:`_module_level_imports` rather than :func:`_imports`. Asserting the
    stricter rule here would reject the very laziness ADR-010.1 mandates.

    Parametrised over :data:`T0_STACK_FREE_MODULES`, not over the full
    :data:`T0_PROJECT_MODULES`: ``core/validation.py`` and its siblings import
    numpy at module scope legally (§2.1), so they are governed by the *other*
    half — :func:`test_no_t0_test_file_reaches_a_stack_needing_module` — rather
    than being asserted clean here and failing forever.
    """
    path = ROOT / relpath
    assert path.is_file(), (
        f"{relpath} is declared T0-stack-free but is missing; a declaration that "
        "names a file the tree does not have is a gate that cannot fail"
    )
    offenders = [
        f"{relpath} imports {name} at module level"
        for name in _module_level_imports(_parse(path))
        if _top(name) in SCIENTIFIC_STACK
    ]
    assert not offenders, (
        "a module reached by a tier-T0 test imports the scientific stack at "
        "import time (architecture.md §10.1, ADR-010.1):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("relpath", T0_TEST_FILES)
def test_no_t0_test_file_reaches_a_stack_needing_module(relpath: str) -> None:
    """ADR-026.2: a T0 file may only reach modules the standard library suffices for.

    The file-level gate above checks *root package names*; this one checks
    *which* modules, transitively.  It is what fails when a T0 test adds
    ``from core.validation import validate_stress_params`` — a name the file gate
    admits (``core`` is permitted) but which pulls numpy and pandas in at import
    time, promoting the suite to T1 with no collection error on a full
    environment and a hard abort on a bare one.

    The difference between those two environments is the point: on a machine
    with the stack installed the defect is invisible, so it has to be caught
    structurally rather than by running the tests.
    """
    reached = _module_closure(relpath)
    offenders = sorted(reached - T0_STACK_FREE_MODULES)
    assert not offenders, (
        f"{relpath} reaches project modules a tier-T0 test may not import "
        "(architecture.md §10.1, ADR-026.2):\n  " + "\n  ".join(offenders)
    )


def test_every_core_module_is_declared_in_the_t0_module_set() -> None:
    """ADR-026.2: the T0 module declaration is complete over ``core/`` and ``main.py``.

    Before the completeness rule the list held only the three modules the T0
    files happened to import, so a fourth import added next month was checked by
    neither the file gate (which only looks at roots) nor this one.  The two
    directions are asserted separately so neither list can move alone.
    """
    on_disk = {f"core/{p.name}" for p in (ROOT / "core").glob("*.py")}
    on_disk.add("main.py")
    # ``core/__init__.py`` is empty but is still a module a T0 file may import
    # (``import core``), so it belongs in the declaration too.
    declared = set(T0_PROJECT_MODULES)
    assert declared == on_disk, (
        "T0_PROJECT_MODULES and the tree disagree "
        f"(only in declaration: {sorted(declared - on_disk)}; "
        f"only on disk: {sorted(on_disk - declared)})"
    )
    assert T0_STACK_FREE_MODULES | T0_STACK_NEEDING_MODULES == declared
    assert not (T0_STACK_FREE_MODULES & T0_STACK_NEEDING_MODULES)


def test_the_stack_needing_class_is_really_stack_needing() -> None:
    """Falsification control: each excluded module genuinely imports the stack.

    Without this, the ``T0_STACK_NEEDING_MODULES`` half of the declaration could
    rot silently — someone could make ``core.validation`` stdlib-only and it would
    *stay* forbidden in T0 files forever, costing the tier a module it could have
    used.  The classification has to be a measured property of the file, not a
    comment that was once true.
    """
    for relpath in sorted(T0_STACK_NEEDING_MODULES):
        offenders = [
            name
            for name in _module_level_imports(_parse(ROOT / relpath))
            if _top(name) in SCIENTIFIC_STACK
        ]
        assert offenders, (
            f"{relpath} is declared stack-needing but imports no scientific root at "
            "module level; move it into T0_STACK_FREE_MODULES so a T0 test can use it"
        )


# ── §10.1 T1 test modules must skip before they import (ADR-026.5) ───────────


def _import_time_stack_roots(path: Path, seen: set[str] | None = None) -> set[str]:
    """Scientific roots ``path`` needs at *import* time, transitively.

    Both halves of the "a bare run must skip T1 cleanly" promise depend on this:
    a test file that imports ``modules.reporting_module`` needs pandas and
    plotly without ever naming them, so an ``importorskip`` for numpy alone does
    not make the file safe.

    Args:
        path: File to start from.
        seen: Paths already walked; mutated across the recursion.

    Returns:
        The subset of :data:`SCIENTIFIC_STACK` this file transitively imports at
        module scope.
    """
    seen = seen if seen is not None else set()
    if not path.is_file():
        return set()
    key = str(path.relative_to(ROOT))
    if key in seen:
        return set()
    seen.add(key)
    roots: set[str] = set()
    for name in _module_level_imports(_parse(path)):
        top = _top(name)
        if top in SCIENTIFIC_STACK:
            roots.add(top)
        elif top in PACKAGES or top == "main":
            parts = name.split(".")
            child = ROOT / parts[0] / (f"{parts[1]}.py" if len(parts) > 1 else "__init__.py")
            roots |= _import_time_stack_roots(child, seen)
    return roots


def _import_site(tree: ast.Module) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """Module-level import sites and ``importorskip`` sites, with line numbers.

    Args:
        tree: Parsed test module.

    Returns:
        A pair ``(imports, skips)`` where ``imports`` maps every module-level
        import to its line and ``skips`` maps each argument passed to a
        module-level ``pytest.importorskip(...)`` to the line it appears on.
        ``np = pytest.importorskip("numpy")`` counts as a skip because the
        assignment executes at import time exactly like the bare call.
    """
    imports: list[tuple[int, str]] = []
    skips: dict[str, int] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        imports.extend((node.lineno, name) for name in names)
        if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            for sub in ast.walk(node):
                func = sub.func if isinstance(sub, ast.Call) else None
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "importorskip"
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)
                ):
                    skips.setdefault(str(sub.args[0].value), sub.lineno)
    return imports, skips


def _t1_test_files() -> list[str]:
    """Every ``tests/test_*.py`` that is not one of the four tier-T0 files."""
    t0 = set(T0_TEST_FILES)
    return [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "tests").glob("test_*.py"))
        if str(p.relative_to(ROOT)) not in t0
    ]


def test_t1_test_file_list_is_neither_empty_nor_missing_files() -> None:
    """Guards the parametrisation below from going vacuous.

    If ``glob("test_*.py")`` ever matched nothing — the directory renamed, the
    tests moved — the parametrised gate would simply not run, which reads
    identically to a gate that passes.
    """
    found = set(_t1_test_files())
    assert found, "no tier-T1 test files found; the skip gate would then run zero cases"
    declared_t0 = set(T0_TEST_FILES)
    assert found & declared_t0 == set(), "a T0 file leaked into the T1 list"


@pytest.mark.parametrize("relpath", _t1_test_files())
def test_t1_test_modules_skip_before_they_import(relpath: str) -> None:
    """ADR-026.5: every T1 module skips on its stack *before* it needs it.

    A bare ``import numpy`` at module scope is a **collection error**, not a
    skip: pytest aborts the whole run and no tier-T0 gate executes at all — so
    "the security suite runs in a bare environment" was false while any T1 file
    imported the stack without a guard.  The requirement is that every root the
    file *transitively* needs is declared with ``pytest.importorskip`` on a line
    that precedes the first import needing it.

    Naming only the roots the file happens to write down would be too weak:
    ``test_reporting.py`` never mentions pandas, but ``modules.reporting_module``
    imports it, and a partial environment (numpy present, plotly absent) would
    collect-error all the same.
    """
    path = ROOT / relpath
    assert path.is_file(), f"{relpath} disappeared from the T1 list"
    needed = _import_time_stack_roots(path)
    if not needed:
        return
    imports, skips = _import_site(_parse(path))

    def _introduces(name: str) -> bool:
        top = _top(name)
        if top in SCIENTIFIC_STACK:
            return top in needed
        if top in PACKAGES:
            parts = name.split(".")
            child = ROOT / parts[0] / (f"{parts[1]}.py" if len(parts) > 1 else "__init__.py")
            return bool(_import_time_stack_roots(child) & needed)
        return False

    first_needed = next((line for line, name in imports if _introduces(name)), None)
    assert first_needed is not None, (
        f"{relpath} needs {sorted(needed)} but no module-level import was found that "
        "introduces one — the skip gate cannot place itself"
    )
    unguarded = [
        root
        for root in sorted(needed)
        if root not in skips or skips[root] > first_needed
    ]
    assert not unguarded, (
        f"{relpath} needs {unguarded} but does not pytest.importorskip them before "
        f"line {first_needed}, where it starts importing them (ADR-026.5); on a bare "
        "interpreter this is a collection error, which aborts the run before any "
        "tier-T0 gate can execute"
    )


#: Directories and files whose ``ADR-NNN`` citations must resolve to a real,
#: recorded decision.  Deliberately **excludes** ``.ai/roles`` and
#: ``.ai/templates``: those are the agent-harness protocol documents, and ADR-014
#: (the Common Agent Protocol) legitimately lives there rather than in this
#: repository's ``decisions.md``.  A gate that swept them in would fail on a
#: correct tree.
_ADR_CITATION_SOURCES: tuple[tuple[str, ...], ...] = (
    ("core",),
    ("modules",),
    ("utils",),
    ("views",),
    ("tests",),
    (".ai", "project"),
    # The two root modules are scanned explicitly.  Reviewer M2: the first
    # version of this list omitted them, so the gate was green against an
    # unrecorded citation injected into ``main.py`` — even though ``app.py`` and
    # ``main.py`` both cite ADRs as binding authority (``app.py`` at the module
    # docstring and the ``ADR-015`` dispatch comment; ``main.py`` at its
    # docstring and the ``doctor``/``info`` help text).  That is the fourth
    # instance of this gate family scanning *most* of the tree and silently
    # exempting the part added last, after deviations 0b.1, 0b.2 and 0d.
    #
    # Listed as files rather than via ``ROOT_MODULES`` so the list stays a plain
    # tuple of paths; ``test_the_adr_citation_scan_covers_every_root_module``
    # asserts it stays in step with the declared root-module set in both
    # directions, so a third root module cannot be added without being scanned.
    ("app.py",),
    ("main.py",),
)


def _declared_adr_numbers() -> set[str]:
    """Return every ``ADR-NNN`` defined as a heading in ``decisions.md``.

    A *definition* is a Markdown heading line (``## ADR-012 — ...``), not a
    mention.  Matching on mentions instead would make the gate vacuous: every ADR
    is mentioned by the later ADRs that reference it, so "mentioned somewhere"
    cannot distinguish "recorded" from "cited but never written".

    Raises:
        AssertionError: if ``decisions.md`` is missing or declares no ADRs at
            all — either would make every citation below pass vacuously, which
            is the failure mode ADR-010 keeps warning against.
    """
    decisions = ROOT / ".ai" / "project" / "decisions.md"
    assert decisions.is_file(), f"missing decision record: {decisions}"
    text = decisions.read_text(encoding="utf-8")
    declared = set(re.findall(r"^##\s+(ADR-\d{3})\b", text, re.MULTILINE))
    assert declared, (
        "decisions.md declares no '## ADR-NNN' headings, so every citation "
        "would resolve vacuously (ADR-010: a gate that passes for the wrong "
        "reason)"
    )
    return declared


def _cited_adr_numbers() -> dict[str, set[str]]:
    """Map every ``ADR-NNN`` cited in source or spec to the files citing it.

    :data:`tests/test_architecture.py` is skipped, because this gate's own prose
    and falsification test necessarily name hypothetical ADR numbers. Skipping it
    at the *scan* is deliberate: an earlier draft excluded it with a post-filter
    (``this_file not in where``), which silently failed. This gate's own
    falsification test contained the literal ``ADR-099``, so any real citation of
    ``ADR-099`` was co-claimed by this file and filtered out — the gate reported
    green against an injected unrecorded citation, which is the exact failure it
    exists to prevent.
    """
    pattern = re.compile(r"\bADR-\d{3}\b")
    this_file = "tests/test_architecture.py"
    cited: dict[str, set[str]] = {}
    for parts in _ADR_CITATION_SOURCES:
        base = ROOT.joinpath(*parts)
        # A single ``*.py`` root module (reviewer M2's addition) is a file, not a
        # directory: ``rglob``/``glob`` would yield nothing and the source would be
        # scanned as an empty set, which is precisely the silent exemption this
        # gate keeps falling into.
        if base.is_file():
            paths = [base]
        elif not base.is_dir():
            continue
        elif parts[0] != ".ai":
            paths = sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
        else:
            paths = sorted(base.glob("*.md"))
        for path in paths:
            rel = str(path.relative_to(ROOT))
            if rel == this_file:
                continue
            for number in pattern.findall(path.read_text(encoding="utf-8")):
                cited.setdefault(number, set()).add(rel)
    return cited


def test_the_adr_citation_scan_covers_every_root_module() -> None:
    """Every declared root module must be scanned for ADR citations.

    Reviewer M2 found the citation gate reading ``_ADR_CITATION_SOURCES``, finding
    no ``app.py`` or ``main.py`` in it, and passing anyway — while both files cite
    ADRs. The fix (listing them) is only durable if a *third* root module cannot
    be added without being scanned, so this asserts the two lists agree in both
    directions. Without the second direction, deleting a root module from the scan
    list alone would re-open exactly the hole M2 found.

    This is the fourth member of the gate family whose defect was "silently
    exempts the part that arrived last" (deviations 0b.1 self-certifying constant,
    0b.2 dead-gate skip, 0d order-sensitive comparison, M2 unscanned root module).
    The pattern is recurring, so the guard is structural rather than a one-off
    edit.

    Returns:
        None
    """
    scanned = {parts[0] for parts in _ADR_CITATION_SOURCES if parts[0].endswith(".py")}
    assert scanned == set(ROOT_MODULES), (
        "the ADR-citation scan and ROOT_MODULES disagree about the root modules: "
        f"scan={sorted(scanned)} declared={sorted(ROOT_MODULES)}. A root module "
        "added to one list but not the other would be exempt from the ADR gate "
        "(reviewer M2)."
    )
    for name in sorted(scanned):
        path = ROOT / name
        assert path.is_file(), f"{name} is listed for citation scanning but absent"
        # Prove the scan really reads the file rather than skipping it: the file
        # must contribute its own citations to the result.
        cited = _cited_adr_numbers()
        assert any(name in where for where in cited.values()) or not re.search(
            r"\bADR-\d{3}\b", path.read_text(encoding="utf-8")
        ), f"{name} cites ADRs but contributes none to the scan"


def test_every_cited_adr_is_recorded_in_decisions_md() -> None:
    """A cited ADR must exist as a heading in ``decisions.md``.

    Found by executing this milestone's own evidence rather than by review: the
    ninth sub-decision of ADR-011, ADR-013 and ADR-015 were each cited as binding
    authority across the tree — ADR-011 in 17 files including all of ``core/``,
    ADR-015 as the fix for the reviewer's MAJOR-1 — and **none of them was
    recorded in ``decisions.md``**. The code was correct and the decision was
    real; only the record was missing.

    That matters mechanically, not cosmetically. ADR-008 makes these records
    binding project input that every subsequent stage reads before implementing,
    so a cited-but-unrecorded decision is a decision the next reader cannot find,
    and the citation reads as a reference to nothing. It is also the exact shape
    of the three earlier gate defects this milestone recorded (deviation 8's stale
    probes, deviation 10's self-certifying gate), so it recurred until a check
    existed for it.

    Sub-decision references (``ADR-011.4``) resolve against the base number, which
    is deliberate: ``ADR-011.4`` is a clause *of* ADR-011, so requiring a separate
    heading per clause would push prose into the heading space.
    """
    declared = _declared_adr_numbers()
    offenders = [
        f"{number} cited by {', '.join(sorted(where))}"
        for number, where in sorted(_cited_adr_numbers().items())
        if number not in declared
    ]
    assert not offenders, (
        "these ADRs are cited as binding authority but have no '## ADR-NNN' "
        "heading in .ai/project/decisions.md — record the decision or drop the "
        "citation (ADR-008):\n  " + "\n  ".join(offenders)
    )


def test_the_adr_citation_gate_would_catch_an_unrecorded_citation() -> None:
    """Falsify :func:`test_every_cited_adr_is_recorded_in_decisions_md`.

    A gate that cannot fail is not a gate, and this one is regex over two
    documents — the failure mode is a pattern that quietly matches nothing and
    passes vacuously. This asserts the detection mechanism flags an unrecorded
    citation and accepts a recorded one, using a synthetic number built by
    concatenation so this file's own text can never collide with a real
    citation. Verified end-to-end against the live tree by injecting a phantom
    citation into ``core/state.py`` and requiring this gate to go red.
    """
    phantom = "ADR-0" + "99"  # never appears literally in this file
    declared = {"ADR-001", "ADR-012"}
    cited = {"ADR-001": {"a.py"}, "ADR-012": {"b.py"}, phantom: {"c.py"}}
    offenders = [
        number for number, where in sorted(cited.items())
        if number not in declared and "tests/test_architecture.py" not in where
    ]
    assert offenders == [phantom], (
        f"the detection mechanism reported {offenders}, expected ['{phantom}']"
    )
    # A recorded citation must not be flagged, or the gate is pure noise.
    assert "ADR-001" not in offenders and "ADR-012" not in offenders

    # And the scan must genuinely see the source tree: every declared ADR this
    # milestone relies on has to be resolvable, which is the positive control.
    real = _declared_adr_numbers()
    assert phantom not in real, f"{phantom} must not be a real heading"
    for required in ("ADR-001", "ADR-008", "ADR-011", "ADR-013", "ADR-015"):
        assert required in real, f"{required} is cited but has no heading"

    # The scope exclusion is load-bearing: ADR-014 is the harness protocol ADR and
    # legitimately lives in .ai/roles and .ai/templates, not in this repository.
    assert "ADR-014" not in real, (
        "ADR-014 is the harness protocol ADR and must stay out of decisions.md; "
        "if it has been recorded here, .ai/roles and .ai/templates need "
        "reconciling with this gate's scope"
    )


# ── §4.5 / ADR-001 deserialization containment ──────────────────────────────────

#: The single module permitted to turn bytes back into objects.  Named in
#: architecture.md §4.5 ("re-hashes the file before ``pickle.load``") and in
#: conventions §6 ("Never load a model artifact without ``ModelArtifactStore``").
DESERIALIZATION_BOUNDARY = "core/model_io.py"

#: Deserializers reached as ``<root>.<name>``.  Each is a bypass of the
#: attestation boundary if called from anywhere else: every one executes whatever
#: the file contains.  ``dumps``/``dump`` are deliberately absent -- *writing* a
#: pickle executes nothing, and ``test_model_io.py`` legitimately serialises
#: fixtures, so flagging the safe half would only make the gate noise.
_DESERIALIZERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pickle", "load"),
        ("pickle", "loads"),
        ("pickle", "Unpickler"),
        ("cPickle", "load"),
        ("cPickle", "loads"),
        ("dill", "load"),
        ("dill", "loads"),
        ("joblib", "load"),
        ("cloudpickle", "load"),
        ("cloudpickle", "loads"),
        ("torch", "load"),
    }
)

#: ``pandas.read_pickle`` reads *and executes* the archive, so it belongs with the
#: rest despite the friendly name.  Matched on the attribute name against a root
#: set, because the alias differs across files.
_PICKLE_READERS: frozenset[str] = frozenset({"read_pickle"})
_PICKLE_READER_ROOTS: frozenset[str] = frozenset({"pandas", "pd"})

#: ``numpy.load`` is safe by default and only becomes an execution primitive when
#: ``allow_pickle=True`` is passed, so it is flagged on that keyword rather than
#: unconditionally -- an unconditional flag would fire on the legitimate
#: ``np.load`` of a plain array and teach the reader to ignore the gate.
_ALLOW_PICKLE_KEYWORD = "allow_pickle"
_OPTIONAL_DESERIALIZERS: frozenset[tuple[str, str]] = frozenset(
    {("numpy", "load"), ("np", "load")}
)

#: ``from pickle import load`` must not slip past a qualified-call scan, so the
#: ``from``-import spelling is bound and the bare name checked separately.
_FROM_IMPORT_BINDS: frozenset[tuple[str, str]] = _DESERIALIZERS | frozenset(
    {("pandas", "read_pickle")}
)


def _dotted_name(node: ast.expr) -> str | None:
    """Render an attribute/name chain as ``a.b.c``.

    Args:
        node: An expression node.

    Returns:
        The dotted string, or ``None`` when the node is a call, subscript or any
        other expression that is not a plain name chain.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _deserialization_sites(tree: ast.Module) -> list[tuple[int, str]]:
    """Every call in ``tree`` that executes the contents of its argument.

    Walks calls rather than matching text, which is what makes the gate
    trustworthy in both directions: this file's own docstrings, the ADR prose and
    ``core/model_io.py``'s module header all *mention* ``pickle.load`` while no
    call exists outside the boundary.  A text match would either flag those -- and
    so teach the reader to ignore the gate -- or need a pile of special cases.

    Args:
        tree: A parsed module.

    Returns:
        ``(lineno, rendered_call)`` pairs, ascending by line number.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (node.module, alias.name) in _FROM_IMPORT_BINDS:
                    bound.add(alias.asname or alias.name)

    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        hit = False
        if isinstance(node.func, ast.Name):
            hit = node.func.id in bound
        else:
            rendered = _dotted_name(node.func)
            if rendered is not None and "." in rendered:
                root, attr = rendered.split(".", 1)
                pair = (root, attr.rsplit(".", 1)[-1])
                hit = pair in _DESERIALIZERS
                hit = hit or (pair in _OPTIONAL_DESERIALIZERS) and any(
                    kw.arg == _ALLOW_PICKLE_KEYWORD
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                hit = hit or (
                    attr.rsplit(".", 1)[-1] in _PICKLE_READERS
                    and root in _PICKLE_READER_ROOTS
                )
            else:
                rendered = None
        if hit:
            sites.append((node.lineno, _dotted_name(node.func) or ast.unparse(node.func)))
    return sorted(sites)


def _audited_deserialization_files() -> list[Path]:
    """Every ``*.py`` file the containment rule applies to, tests included.

    ``tests/`` is in scope because the rule reads "outside
    ``core/model_io.py``": a test that unpickles an artifact directly bypasses
    the very attestation the suite exists to prove, so that is a defect in the
    thing under test rather than an exemption from it.

    Returns:
        Sorted source paths, caches excluded.
    """
    files = list(_project_files())
    tests = ROOT / "tests"
    if tests.is_dir():
        files.extend(
            sorted(p for p in tests.rglob("*.py") if "__pycache__" not in p.parts)
        )
    return sorted({p for p in files if p.is_file()})


def test_deserialization_is_confined_to_model_io() -> None:
    """No file outside ``core/model_io.py`` may execute a pickle.

    This is the Wave 1 gate the roadmap states verbatim -- "no ``pickle.load``
    outside ``core/model_io.py`` (grep assertion)" -- and it was **absent**. The
    invariant happened to hold (a scan finds exactly one deserialization call in
    the whole tree, at ``core/model_io.py:719``), but nothing enforced it, so a
    passing result was indistinguishable from the absence of a check.  A future
    ``pickle.load`` / ``joblib.load`` / ``pd.read_pickle`` outside the boundary
    would have re-opened CRIT-001/CRIT-002 -- arbitrary code execution from a
    file -- with a fully green suite, because the tests that matter here assert
    the boundary's *own* behaviour and cannot see a second caller skipping it.

    Recorded because it is the same defect class as deviation 0b.1's
    self-certifying constant, deviation 0b.2's dead-gate skip and deviation 0d's
    order-sensitive comparison: a security check that does not exist is read as a
    security check that passes.

    Returns:
        None.
    """
    offenders: list[str] = []
    for path in _audited_deserialization_files():
        if path == ROOT / DESERIALIZATION_BOUNDARY:
            continue
        for lineno, rendered in _deserialization_sites(_parse(path)):
            rel = path.relative_to(ROOT)
            offenders.append(f"{rel}:{lineno}: {rendered}(...) executes the file")
    assert not offenders, (
        f"deserialization is confined to {DESERIALIZATION_BOUNDARY}, where ADR-001 "
        "enforces attestation, containment and consent. These calls bypass all "
        "three:\n  " + "\n  ".join(offenders)
    )


def test_the_deserialization_gate_would_catch_a_bypass() -> None:
    """Falsify :func:`test_deserialization_is_confined_to_model_io`.

    A gate that cannot fail is not a gate.  This one is a hand-written AST walk,
    so the risk is not "the regex matches nothing" -- it is that the walk misses
    the spellings a developer actually reaches for.  Every form below is parsed
    and required to be flagged, alongside two negative controls that must stay
    unflagged or the gate would be noise a reader learns to ignore.

    Returns:
        None.
    """
    flagged = {
        "pickle.load": "import pickle\nwith open(p, 'rb') as fh:\n    m = pickle.load(fh)\n",
        "pickle.loads": "import pickle\nm = pickle.loads(raw)\n",
        "pickle.Unpickler": "import pickle\nm = pickle.Unpickler(fh).load()\n",
        "from pickle import load": "from pickle import load\nm = load(fh)\n",
        "joblib.load": "import joblib\nm = joblib.load(p)\n",
        "pandas.read_pickle": "import pandas as pd\ndf = pd.read_pickle(p)\n",
        "numpy.load(allow_pickle=True)": (
            "import numpy as np\narr = np.load(p, allow_pickle=True)\n"
        ),
    }
    for label, source in flagged.items():
        assert _deserialization_sites(ast.parse(source)), (
            f"the containment walk did not flag {label}:\n{source}"
        )

    # Negative control 1: serialising executes nothing, and test_model_io.py
    # builds its fixtures with pickle.dumps.
    assert not _deserialization_sites(
        ast.parse("import pickle\nblob = pickle.dumps(obj)\n")
    ), "pickle.dumps must not be flagged: writing a pickle executes nothing"

    # Negative control 2: numpy.load is safe unless allow_pickle=True.
    assert not _deserialization_sites(
        ast.parse("import numpy as np\narr = np.load('scores.npy')\n")
    ), (
        "numpy.load without allow_pickle=True must not be flagged; flagging it "
        "would cry wolf on every array read"
    )

    # Positive control against the live tree: the boundary must actually contain
    # a call, or the allowlist is exempting a file with nothing to exempt and the
    # gate would be green without protecting anything.
    assert _deserialization_sites(_parse(ROOT / DESERIALIZATION_BOUNDARY)), (
        f"{DESERIALIZATION_BOUNDARY} contains no deserialization call, so the "
        "allowlist exempts a file with nothing to exempt -- this gate would pass "
        "without protecting anything. Either the boundary moved or the walk broke."
    )


# ── conventions §5 banned pandas patterns ──────────────────────────────────────

_DEPRECATED_FILL_KEYWORD = "method"
_INPLACE_KEYWORD = "inplace"
_FILL_METHOD_OWNERS: frozenset[str] = frozenset({"fillna"})


def _banned_pandas_sites(tree: ast.Module) -> list[tuple[int, str]]:
    """Calls in ``tree`` that use a pandas pattern conventions §5 forbids.

    Two rules, matching the Wave 3 gate's wording verbatim -- "no
    ``fillna(method=`` and no ``inplace=True`` remaining in ``modules/``":

    1. any call passing ``inplace=True``; and
    2. any ``fillna`` passing the removed ``method=`` keyword.

    AST rather than text, for the same reason as
    :func:`_deserialization_sites`: ``modules/data_module.py`` *documents* the
    banned form in its own docstring in order to explain why it is banned, so a
    text match would report a compliant file as a violation and force an
    exemption list -- the exact shape that makes a gate decorative.

    Args:
        tree: A parsed module.

    Returns:
        ``(lineno, rendered_reason)`` pairs, ascending by line number.
    """
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        rendered = _dotted_name(node.func) or ast.unparse(node.func)
        owner = rendered.rsplit(".", 1)[-1]
        for kw in node.keywords:
            if kw.arg is None:  # ``**kwargs`` splat: cannot be resolved statically
                continue
            value = kw.value
            if (
                kw.arg == _INPLACE_KEYWORD
                and isinstance(value, ast.Constant)
                and value.value is True
            ):
                sites.append((node.lineno, f"{rendered}(..., inplace=True)"))
            elif (
                kw.arg == _DEPRECATED_FILL_KEYWORD
                and owner in _FILL_METHOD_OWNERS
            ):
                sites.append((node.lineno, f"{rendered}(..., method=...)"))
    return sorted(sites)


def test_banned_pandas_patterns_are_absent_from_the_tree() -> None:
    """No project file may pass ``inplace=True`` or ``fillna(method=)``.

    The second half of the Wave 3 gate, also mandated and also **absent**. Its
    first half exists and is thorough -- ``test_data_preprocessing.py`` proves
    every strategy changes the frame and emits no ``FutureWarning`` -- but that
    half provably cannot see this one. Two independent reasons, both measured on
    the pinned ``pandas==2.3.3``:

    - ADR-011.9 records that ``fillna(<stat>, inplace=True)`` emits **no**
      warning and mutates correctly under Copy-on-Write off, so a chained
      ``df[col].fillna(v, inplace=True)`` passes the behavioural gate while
      staying exactly as latent as ADR-010.8 describes it. "No warning after the
      fix" was the correct assertion and it is blind here by construction.
    - The behavioural gate only reaches the strategies it parametrizes, so a
      banned call in any other function of ``modules/`` is simply never executed.

    Verified before fixing: appending either banned form to
    ``modules/data_module.py`` as an uncalled helper leaves the suite fully green.
    That is the same defect as the missing deserialization gate above -- a
    convention nothing enforces, which reads as a convention that holds.

    Scanned across every project file rather than only ``modules/``, because
    conventions §5 states both rules generally; the roadmap's ``modules/`` is
    the floor, not the ceiling.

    Returns:
        None.
    """
    offenders: list[str] = []
    for path in _project_files():
        for lineno, reason in _banned_pandas_sites(_parse(path)):
            rel = path.relative_to(ROOT)
            offenders.append(f"{rel}:{lineno}: {reason}")
    assert not offenders, (
        "conventions §5 bans chained assignment with inplace=True and the "
        "removed fillna(method=). Assign the result instead "
        "(df[col] = df[col].ffill()); see the handle_missing_values docstring in "
        "modules/data_module.py:\n  "
        + "\n  ".join(offenders)
    )


def test_the_pandas_pattern_gate_would_catch_a_reintroduction() -> None:
    """Falsify :func:`test_banned_pandas_patterns_are_absent_from_the_tree`.

    Asserts each banned spelling is detected **and**, just as importantly, that
    the compliant rewrites and the explanatory prose are not. A gate that flags
    the documented rationale in ``modules/data_module.py`` would be reported as a
    false positive, and the first response to that is to weaken the gate.

    Returns:
        None.
    """
    flagged = {
        "chained inplace fill": (
            "def f(df, col):\n    df[col].fillna(0.0, inplace=True)\n"
        ),
        "frame-level dropna inplace": (
            "def f(df):\n    df.dropna(inplace=True)\n"
        ),
        "chained ffill inplace": "def f(df, col):\n    df[col].ffill(inplace=True)\n",
        "fillna(method=)": "def f(df):\n    df.fillna(method='ffill')\n",
    }
    for label, source in flagged.items():
        assert _banned_pandas_sites(ast.parse(source)), (
            f"the conventions §5 walk did not flag {label}:\n{source}"
        )

    # The compliant rewrite the conventions prescribe must pass, including the
    # ffill/bfill spellings that replace the removed method= keyword.
    for source in (
        "def f(df, col):\n    df[col] = df[col].fillna(0.0)\n",
        "def f(df, col):\n    df[col] = df[col].ffill()\n",
        "def f(df, col):\n    df[col] = df[col].bfill()\n",
        "def f(df, method):\n    df = getattr(df, method)()\n",
    ):
        assert not _banned_pandas_sites(ast.parse(source)), (
            f"the compliant form was flagged:\n{source}"
        )

    # inplace=False is a no-op keyword, not the banned mutation, and flagging it
    # would be noise.
    assert not _banned_pandas_sites(
        ast.parse("def f(df, col):\n    df[col].fillna(0.0, inplace=False)\n")
    ), "inplace=False must not be flagged: it is the documented no-op, not a mutation"

    # Prose must not be flagged. This is the real file's own shape: the banned
    # form is quoted in a docstring to explain why it is banned.
    documented = (
        'def f(df, col):\n'
        '    """Replace ``df[col].fillna(v, inplace=True)`` with an assignment."""\n'
        "    return df\n"
    )
    assert not _banned_pandas_sites(ast.parse(documented)), (
        "a docstring quoting the banned form was flagged; only calls may match"
    )


# ── conventions §2 typing / ADR-016.1-016.2 ─────────────────────────────────────

#: Packages whose **public** definitions conventions §2 binds.  ADR-016.1 lifts
#: §9's signature freeze for an annotations-only change across all three, which is
#: why ``modules/`` and ``utils/`` are in scope here and were not before.
TYPED_PACKAGES: tuple[str, ...] = ("core", "modules", "utils")

#: Modules that annotate a name they do not import at module scope, and therefore
#: need ``from __future__ import annotations`` to survive the devcontainer's
#: CPython 3.11.  Kept as an explicit list rather than derived, because the
#: requirement is not "is the name imported" — it is "would evaluating this
#: annotation on 3.11 raise", which cannot be decided statically and which the
#: suite running on 3.14 can never observe (ADR-016.2).
LAZY_EVALUATION_REQUIRED: frozenset[str] = frozenset(
    {
        "modules/model_module.py",
        "modules/post_stress_module.py",
        "utils/metrics.py",
        "utils/plotting.py",
    }
)


def _public_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    """Public, module-level and class-level ``def``s in ``tree``.

    Two exclusions are load-bearing and both were arrived at by measurement:

    * **Nested functions are not public API.**  ``ast.walk`` descends into them,
      so the ``default=`` closure ``modules/reporting_module.py`` passes to
      ``json.dumps`` counted as an unannotated public definition.  That closure
      has ``col_offset == 8`` and is not part of any surface a caller can reach;
      counting it is how roadmap.md deviation 0a published 42 where the real
      figure is 41.
    * **``self`` / ``cls`` are never annotated.**  The implicit receiver is not a
      parameter a caller passes, so requiring an annotation on it makes *every*
      method look unannotated -- the error that produced the third and fourth
      published count.

    Args:
        tree: A parsed module.

    Returns:
        list[ast.FunctionDef]: Definitions whose name is not underscore-prefixed.
    """
    found: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                    found.append(sub)
    return found


def test_public_definitions_carry_type_hints() -> None:
    """conventions §2: every public ``def`` annotates its parameters and return.

    Critic MIN-004, closed by ADR-016.1.  Asserted over ``core/``, ``modules/``
    and ``utils/`` -- the three packages §9's freeze covers, which ADR-016 lifts
    for annotations only.  ``views/`` is excluded because ADR-011.5's
    ``color_severity`` closure is local, and ``tests/`` because it is not a
    package under §9.

    Structural (``ast``) rather than ``typing.get_type_hints``, so the check stays
    in the T0 tier (§10.1) and never imports a single module.
    """
    offenders: list[str] = []
    for package in TYPED_PACKAGES:
        for path in _source_files(package):
            for node in _public_defs(_parse(path)):
                args = node.args
                named = [
                    p
                    for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)
                    if p.arg not in ("self", "cls")
                ]
                missing = [p.arg for p in named if p.annotation is None]
                if missing or node.returns is None:
                    detail = []
                    if missing:
                        detail.append(f"params {missing}")
                    if node.returns is None:
                        detail.append("return")
                    offenders.append(
                        f"{package}/{path.name}:{node.lineno} {node.name}() "
                        f"missing {', '.join(detail)}"
                    )
    assert not offenders, (
        "conventions §2 typing violations — public definitions must annotate "
        "parameters and return (ADR-016.1):\n  " + "\n  ".join(offenders)
    )


def test_future_annotations_is_present_where_lazy_evaluation_is_required() -> None:
    """ADR-016.2: the four eagerly-evaluated modules must defer annotations.

    The devcontainer targets CPython 3.11; this gate runs on 3.14.  PEP 649 makes
    3.14 defer annotation evaluation **entirely**, so on the interpreter the suite
    runs on, a bare undefined annotation in a signature imports cleanly.  On 3.11
    the identical line raises ``NameError`` **at import time** — the app does not
    start.  No test written in this suite can catch that, because the suite runs
    where the bug is invisible; the only defence is to assert the guard is present.

    This is also the reason the annotation sweep above cannot be the only new gate:
    it would pass on 3.14 for code that is broken on 3.11.

    Returns:
        None
    """
    offenders = [
        relpath
        for relpath in sorted(LAZY_EVALUATION_REQUIRED)
        if "from __future__ import annotations"
        not in (ROOT / relpath).read_text()
    ]
    assert not offenders, (
        "these modules annotate names they do not import at module scope, so they "
        "need 'from __future__ import annotations' to import on the 3.11 "
        "devcontainer target (ADR-016.2):\n  " + "\n  ".join(offenders)
    )


def test_the_annotation_sweep_could_still_fail() -> None:
    """Falsify :func:`test_public_definitions_carry_type_hints`.

    A gate that reports nothing is indistinguishable from a gate that passes for
    the wrong reason, which is the failure mode this project has now hit three
    times (roadmap deviations 0b, 8 and 11).  Both directions are checked here:
    an unannotated parameter, a missing return, a ``self``-only method must be
    flagged; and the two shapes that must **not** be flagged -- a nested closure
    (ADR-011.5's ``color_severity``) and a method whose only unannotated argument
    is the implicit receiver -- must pass.

    Returns:
        None
    """
    module = ast.parse(
        "def good(a: int) -> int:\n"
        "    return a\n"
        "def missing_param(a) -> int:\n"
        "    return a\n"
        "def missing_return(a: int):\n"
        "    return a\n"
        "class C:\n"
        "    def method(self, a: int) -> int:\n"
        "        return a\n"
        "    def closure_holder(self) -> None:\n"
        "        def local(x):\n"
        "            return x\n"
        "        local(1)\n"
    )
    flagged = {
        node.name
        for node in _public_defs(module)
        if node.returns is None
        or any(p.annotation is None for p in node.args.args if p.arg not in ("self", "cls"))
    }
    assert flagged == {"missing_param", "missing_return"}, (
        f"the sweep flagged {sorted(flagged)}; it must flag exactly the two "
        "offenders, must not flag a self-only method, and must not descend into "
        "a nested closure (ADR-011.5)"
    )


# ── ADR-027: one normaliser at the comparison reading boundary ─────────────────

#: The module owning the boundary.  Named (not derived) because the gate is about
#: *this* module's structure; a rename there is a deliberate architecture move.
COMPARISON_MODULE: Path = ROOT / "modules" / "comparison_module.py"

#: The optional reading dicts ``ModelComparator._component_scores`` must normalise,
#: identified by the component label each passes to ``_normalise_reading``.
_COMPARISON_READING_COMPONENTS: frozenset[str] = frozenset(
    {"robustness", "calibration"}
)

#: The one private helper that owns reading normalisation (run-015 MIN-1,
#: ADR-027).  Both optional readings route through it so a non-numeric reading
#: raises a typed ``ValidationError`` and a numeric-but-non-finite reading becomes
#: *missing* rather than a floored zero that still counts as measured.
_COMPARISON_NORMALISER: str = "_normalise_reading"


def _named_defs(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    """Return every ``def <name>`` in ``tree``, module level or class level.

    Args:
        tree: A parsed module.
        name: The function name to collect.

    Returns:
        list[ast.FunctionDef]: Matching definitions, in walk order.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def _bare_float_calls(node: ast.AST) -> list[int]:
    """Line numbers of any ``float(...)`` call reachable from ``node``.

    ``_normalise_reading`` already calls ``float`` once; the gate is about a
    **second** coercion happening in ``_component_scores`` and bypassing the
    helper, so it scans that function's subtree only.

    Args:
        node: The function body to scan.

    Returns:
        list[int]: Ascending line numbers of bare ``float()`` calls.
    """
    lines: list[int] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "float"
        ):
            lines.append(child.lineno)
    return sorted(lines)


def _normaliser_component_args(node: ast.AST) -> set[str]:
    """Component labels passed as the first argument to the normaliser.

    Args:
        node: The function body to scan.

    Returns:
        set[str]: The string-literal first arguments, e.g. ``{"robustness"}``.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == _COMPARISON_NORMALISER
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ):
            found.add(child.args[0].value)
    return found


def test_comparison_readings_flow_through_the_single_normaliser() -> None:
    """ADR-027: one normaliser, and ``_component_scores`` has no bare ``float()``.

    Run-015 MIN-1: ``_component_scores`` called ``float()`` directly on the
    caller-supplied robustness and calibration readings, so a non-numeric value
    raised a raw ``ValueError``/``TypeError`` across the **public**
    ``ModelComparator.rate_models`` boundary — which conventions §3 forbids.  The
    two conversions had also drifted: a non-finite reading floored to ``0.0`` and
    stayed *available*, while an absent entry was ``None``/missing.

    This gate keeps the two fixes from regressing **together**: the helper must
    exist exactly once, ``_component_scores`` must contain no bare ``float()``
    coercion, and both optional readings must be routed through the helper.

    Returns:
        None.
    """
    assert COMPARISON_MODULE.is_file(), f"missing {COMPARISON_MODULE}"
    tree = _parse(COMPARISON_MODULE)

    definitions = _named_defs(tree, _COMPARISON_NORMALISER)
    assert len(definitions) == 1, (
        f"{_COMPARISON_NORMALISER} must be defined exactly once; found "
        f"{len(definitions)} at lines {[d.lineno for d in definitions]} — a "
        "second copy is a divergence path (ADR-027)"
    )

    component_scores = _named_defs(tree, "_component_scores")
    assert len(component_scores) == 1, (
        f"_component_scores must be defined exactly once; found "
        f"{len(component_scores)} (ADR-027)"
    )
    body = component_scores[0]

    bare = _bare_float_calls(body)
    assert not bare, (
        "modules/comparison_module.py:_component_scores coerces a reading with a "
        f"bare float() at line(s) {bare}; route it through {_COMPARISON_NORMALISER} "
        "so a non-numeric reading raises a typed ValidationError and a non-finite "
        "reading is missing, not a floored component (ADR-027)"
    )

    unrouted = _COMPARISON_READING_COMPONENTS - _normaliser_component_args(body)
    assert not unrouted, (
        f"_component_scores does not route {sorted(unrouted)} through "
        f"{_COMPARISON_NORMALISER}; both optional readings must share the one "
        "translation point (ADR-027)"
    )


def test_the_comparison_normaliser_gate_would_catch_a_bypass() -> None:
    """Falsify :func:`test_comparison_readings_flow_through_the_single_normaliser`.

    A gate that reports nothing is indistinguishable from one that passes for the
    wrong reason (ADR-010).  This asserts each detection mechanism fires on the
    bypass it exists to catch — a bare ``float()`` coercion, an unrouted reading,
    a duplicated helper — and, just as importantly, that the compliant spelling
    passes.

    Returns:
        None.
    """
    bypass = ast.parse(
        "class C:\n"
        "    def _component_scores(self, model, vals, robustness_dict, calibration_ece):\n"
        "        robustness = float(robustness_dict[model])\n"
        "        return robustness\n"
    )
    bypass_body = _named_defs(bypass, "_component_scores")[0]
    assert _bare_float_calls(bypass_body), (
        "the gate did not flag a bare float() coercion of a reading"
    )
    assert not _normaliser_component_args(bypass_body), (
        "the gate thought a bare float() was a normaliser call"
    )

    compliant = ast.parse(
        "class C:\n"
        "    def _component_scores(self, model, vals, robustness_dict, calibration_ece):\n"
        "        r = _normalise_reading('robustness', model, robustness_dict[model])\n"
        "        e = _normalise_reading('calibration', model, calibration_ece[model])\n"
        "        return r, e\n"
    )
    compliant_body = _named_defs(compliant, "_component_scores")[0]
    assert not _bare_float_calls(compliant_body), (
        "the gate flagged the compliant routings as bare float() calls"
    )
    assert _normaliser_component_args(compliant_body) == _COMPARISON_READING_COMPONENTS

    partial = ast.parse(
        "class C:\n"
        "    def _component_scores(self, model, vals, robustness_dict, calibration_ece):\n"
        "        r = _normalise_reading('robustness', model, robustness_dict[model])\n"
        "        return r\n"
    )
    partial_body = _named_defs(partial, "_component_scores")[0]
    assert _COMPARISON_READING_COMPONENTS - _normaliser_component_args(partial_body) == {
        "calibration"
    }, "the gate did not notice a reading that bypassed the normaliser"

    duplicated = ast.parse(
        "def _normalise_reading(component, model, raw):\n"
        "    return None\n"
        "def _normalise_reading(component, model, raw):\n"
        "    return None\n"
    )
    assert len(_named_defs(duplicated, _COMPARISON_NORMALISER)) == 2, (
        "the definition count did not see the duplicate helper"
    )
