# Project Roadmap

Milestone: **run-003 remediation** (remediation of the run-002 Critic audit).
Each wave lists its gate. A wave is not complete until its gate passes.

## Non-negotiable ordering

Waves are sequential because later waves depend on earlier seams. Waves MUST NOT
be merged or reordered. Each wave is an independent commit.

---

## Wave 0 — Safety net (blocks everything else)

- `requirements-dev.txt` (pytest pinned), `tests/conftest.py`,
  `tests/test_architecture.py` (layer matrix, no streamlit in core/modules/utils,
  no cycles).
- **Environment bootstrap (ADR-010.9).** No runtime dependency is installed in the
  working environment and the local interpreter is Python 3.14.7 against a 3.11
  devcontainer, with no CI. Wave 0 MUST create a venv, install
  `requirements.txt` + `requirements-dev.txt`, and **record the interpreter
  version** alongside the parity fixtures — otherwise the Wave 6 parity gate is
  comparing numbers captured on different interpreters. If the stack cannot be
  installed, Wave 0 still completes; it just scopes its gate to T0 and the T1
  fixtures are captured later.
- Baseline snapshots: record current `ReliabilityScorer` outputs for fixed inputs
  as fixtures for parity testing.
- **Gate:** `python -m pytest tests/test_architecture.py` green **on an
  interpreter with only pytest installed** (T0, architecture.md §10.1 — this is a
  real constraint, not a formality: assert it by running in a venv without the
  scientific stack). Baseline fixtures recorded with the interpreter version.
- **Addresses:** MIN-006, MAJ-001/MAJ-006 (partially), and makes every later
  wave verifiable.

## Wave 1 — Security boundary (CRITICAL)

- `core/errors.py`, `core/config.py` (incl. `ArtifactPolicy`).
- `core/model_io.py`: `ModelArtifactStore`, `ArtifactRecord`, manifest v1,
  filename sanitisation, containment, atomic writes, hash verification, trust
  gate, estimator type gate. **The estimator gate MUST be lazily resolved or
  injected via `estimator_bases` (ADR-010.1)** — a module-level sklearn import
  would drag sklearn+numpy+scipy into `test_model_io.py` and break the T0
  guarantee the security gate depends on.
- `modules/model_module.py`: `save_model`/`load_model` delegate to the store,
  lose `streamlit` import, gain `trust: bool = False`.
- `views/module_02_baseline.py` **does not exist at this wave** — `views/` is only
  created in Wave 5. Wave 1 MUST edit the save/load tab in `app.py:665-725` in
  place: explicit unchecked "this file executes Python code" confirmation, plus
  distinct error rendering for `Artifact*` errors. The extracted view module in
  Wave 5 inherits this behaviour verbatim. (Resolves the Wave 1 / Wave 5
  sequencing defect recorded in ADR-009.)
- `modules/reporting_module.py`: `html.escape` on every HTML interpolation.
- `tests/test_model_io.py`, `tests/test_reporting_escaping.py`.
- **Gate:** traversal/tamper/untrusted tests green; no `pickle.load` outside
  `core/model_io.py` (grep assertion); save-side traversal test present;
  `test_model_io.py` runs green **with only pytest installed** (T0).

## Wave 2 — Pure perturbation kernel (CRITICAL robustness)

- `core/perturbations.py` (`OPERATIONS`, `apply_perturbation`, zero-variance
  guard, RNG injection). **`OPERATIONS` keys are the six existing dispatch
  strings** (`gaussian_noise`, `uniform_noise`, `feature_dropout`,
  `feature_corruption`, `scale_perturbation`, `distribution_shift`) — renaming
  them silently disables every batch stress test (ADR-010.6).
- `apply_perturbation` coerces to `float64` up front and returns `float64`;
  integer input is no longer truncated, and a non-numeric frame raises
  `ValidationError` naming the offending columns (ADR-010.3).
- `core/validation.py` (`validate_stress_params`, `ensure_finite`,
  `validate_frame`) with bounds from `core/config.py`.
- `modules/stress_module.py`: six operators delegate to
  `apply_perturbation`; `batch_stress_test` rejects unknown stress types with
  `UnsupportedStressTypeError` instead of silently `continue`-ing.
- `tests/test_perturbations.py`, `tests/test_validation.py`.
- **Gate:** purity tests green; **an `int64 + float64` frame changes under all
  six operators** — this is the regression frame, chosen because measured today 5
  of 6 operators silently no-op on exactly that shape. A `str`-bearing frame is
  NOT a valid regression frame: it must now raise `ValidationError` (ADR-010.4).
  Out-of-range and `scale_factor=0`/`dropout_rate=1.0` rejected; `int64` ndarray
  comes back `float64` with fractions preserved; seeded runs reproducible; every
  UI slider maximum and batch-config value is accepted.

## Wave 3 — Data correctness & immutability

- `modules/data_module.py`: replace chained `inplace=True` + deprecated
  `method=` fills; explicit `deep=True` copies; rename the ambiguous
  `handle_missing_values` custom-value branch parameter; guard
  `get_split_summary` against zero-sample splits **and against returning `None`**
  (raise `DatasetError` — conventions §3 forbids `None` as a failure signal).
- `tests/test_data_preprocessing.py`.
- **Gate:** every missing-value strategy demonstrably changes the frame **and
  emits no `FutureWarning`**; no `fillna(method=` and no `inplace=True` remaining
  in `modules/` (grep assertion).

  **Corrected premise (ADR-010.8).** This gate previously described the chained
  `inplace=True` fills as a *current silent no-op*. Measured under the pinned
  `pandas==2.3.3` with Copy-on-Write off, chained assignment on a copy **does**
  mutate the frame. The real defects are a `FutureWarning` on every call and
  latent breakage the moment CoW is enabled or pandas 3.0 lands. Test the
  warning-free, correct behaviour — do not write a regression test against a bug
  that is not yet reachable, and do not "fix" the semantics beyond removing the
  chained assignment.

## Wave 4 — Error & logging policy, Streamlit decoupling

- `core/reporters.py` (Protocol, `NullReporter`, `CollectingReporter`),
  `core/logging_config.py`, `views/reporters.py` (`StreamlitReporter`).
- `modules/data_module.py`: `load_dataset` raises `DatasetLoadError`; drop
  `display_summary` body into `views/module_01_data.py` (keep the method as a thin
  delegating wrapper for API compatibility, or remove it and update the single
  call site — either is acceptable, document the choice).
- `modules/calibration_module.py:103-106`: log the Brier degradation.
- `modules/post_stress_module.py:12`: delete the unused `import streamlit as st`
  (dead import, zero `st.*` calls — the only reason this file trips the matrix).
- `views/` error rendering helper `render_error(exc)`.
- **Gate:** no `import streamlit` anywhere in `core/`, `modules/`, `utils/`
  (grep + arch test); every remaining `except` either raises or logs.

## Wave 5 — Composition root & view extraction

- `core/state.py` (`StateKeys`, `init_state`, `reset_state`, `AppContext`),
  `views/base.py`.
- `views/module_01..09_*.py`: verbatim moves of `app.py:88`, `384`, `783`,
  `1139`, `1766`, `2228`, `2617`, `2891`, `3181`, de-indented by 4. No
  behaviour, string, widget `key=` or evaluation-order change.
- `app.py` reduced to the composition root.
- **Gate:** `app.py` ≤ 200 lines; sidebar labels unchanged; every page exercised
  end-to-end (data upload → train → stress → post-stress → calibration → compare
  → score → report → export).

  **Correction (recorded after execution).** This gate originally read "the smoke
  test **cannot be automated** — `streamlit` is not installed and there is no
  browser harness". That was wrong on both counts: streamlit *is* installed, and
  streamlit ships `streamlit.testing.v1.AppTest`, a real script-level harness.
  `tests/test_app_smoke.py` is now the automated gate — it drives
  `app.py` itself, each of the nine pages selected in turn, with genuine session
  state built by real `DataManager` / `ModelTrainer` / `StressTester` objects
  rather than hand-stubbed dicts.

  **Action tier, added after run-003 (MINOR-1 / MINOR-3).** The tiers above only
  *render* pages; the entire defect class reviewer MAJOR-1 belonged to lived
  inside button bodies no test ever entered, and this file's own docstring had
  claimed `AppTest` could not press buttons. That claim was false
  (`app.button(key="…").click().run()` works), so a third tier now presses
  **every** mutating button: keyed buttons via `MUTATING_BUTTONS`, unkeyed ones
  by label, the three Data Management buttons that only render against a dirty
  frame via a separately seeded state, and the Train button (whose label is an
  f-string) on its own. The rule adopted is that **a mutating action with no
  pressed button is untested code**, however thoroughly its page renders.

  Two things `AppTest` cannot do are recorded honestly rather than papered over:

  1. **No visual verification.** `AppTest` asserts on the widget/element tree, not
     on pixels. Chart layout, plotly rendering and the styler output still need
     one human pass in a browser before the milestone is called done.
  2. **No `file_uploader`.** `AppTest` has no accessor for it, so the upload path
     is driven by seeding `session_state` with a frame produced by a real
     `DataManager` load. That covers everything downstream of the upload; the
     uploader widget itself is not exercised.

  Widget keys moved from "assert the count" to a **frozen contract fixture**,
  `tests/fixtures/widget_keys.json`, asserted in `tests/test_architecture.py`
  (T0). The count is 64, not 63: 63 pre-remediation keys plus `load_model_trust`,
  added in Wave 1 under ADR-001. A count of 63 would have been the regression.

  **Pre-verified, do not re-derive (architecture.md §4.8):** an `ast` pass over
  all nine branches found **zero cross-branch free variables**, so extraction
  cannot break a name Python currently leaks between pages. There are **63 unique
  `key=` widget literals** — assert the count is unchanged, since that count is
  the persisted session contract. The nav variable is `module`, the sidebar radio
  is at `app.py:69`, the status footer at `app.py:3501-3517`, and `StressTester`
  can move to eager `init_state` safely (its `__init__` has no side effects).

## Wave 6 — Configuration, dependency hygiene, documentation

- `ReliabilityScorer` + stress bounds read from `core/config.py`; defaults
  unchanged.
- `tests/test_reliability_parity.py` freezes pre-refactor scores/grades. This is
  the **only bit-exact** numeric freeze in the milestone (ADR-010.5); the
  perturbation suites assert formula parity and seeded self-consistency, never
  equality with the old implementation's numbers.
- Remove `shap==0.50.0` after a reverse-dependency check (ADR-007).
- Populate `.ai/project/*.md` (done in this architecture stage; update for any
  deviation), update `README.md` security note and project structure.
- **Gate:** parity tests green **on the same interpreter version recorded in
  Wave 0's fixtures**; `README.md` matches the final tree; no ADR violated.

  **Executed.** `shap` removed from `requirements.txt` and from the README
  dependency list. Reverse-dependency check, run before removal: no project file
  imports `shap`, no installed distribution declares it in `Requires-Dist`, and
  it was never installed in the remediation environment. `numba` and `llvmlite`
  were checked at the same time and **kept** — `numba` is declared by `pandas`
  (conditionally), so it is a pinned transitive of a direct dependency, exactly
  the case ADR-007 says not to churn.

  Reliability parity re-verified **bit-exactly** after the config wiring, via
  `python -m tests.capture_baseline --check` against the fixture captured in
  Wave 0 on CPython 3.14.7.

---

## Deviations recorded during execution

These are departures from the plan as written, each with its reason. They are
listed here rather than silently absorbed, per ADR-008.

### 0. `main.py` removed; the architecture gate now covers the whole tree

A CLI stub `main.py` appeared in the tree during Wave 6. It is in no ADR, no wave
and no architecture section, its own comment said it existed "as expected by the
test harness", and its body returned `0` for every argument including invalid
ones — a CLI structurally incapable of reporting an error. The framework's entry
point is `streamlit run app.py` and nothing referenced the file, so it is
**deleted**. Recorded here rather than in a comment because ADR-008 requires a
deviation to be documented.

The deeper defect it exposed is fixed too. `_project_files()` enumerated
`app.py` *by name*, so any other top-level module belonged to no layer, had no
§2.1 matrix row, and escaped the cycle detector, the app-import check and the
widget-key contract simultaneously. Two gates now close that:

- `test_only_app_py_is_a_top_level_module` — §2.2 declares exactly one
  top-level module, so a new one must be declared with a layer before it can ship;
- `test_root_module_respects_the_matrix` — any root module other than `app.py`
  may import stdlib and project packages only, so widening `ROOT_MODULES` on
  purpose cannot quietly widen the dependency surface too.

`_project_files()` now enumerates root `*.py` dynamically, and the widget-key
contract scans them alongside `views/`, so a widget re-added to `app.py` is
caught rather than ignored for being in the wrong file.

The widget-key contract is also now asserted as **exact set equality per file, in
both directions**. It previously asserted only that each expected key was
*present*, so adding a new undeclared key passed — a gate passing for the wrong
reason, which is precisely what ADR-010's rationale warns against.
`test_added_widget_keys_are_declared` gained the second half it was missing: each
declared addition must actually exist in a view, so a justification cannot become
a licence to declare a key that does not.

#### 0b. The stub was re-added, and the gate let it through (second occurrence)

The `main.py` stub above reappeared verbatim in the working tree, and this time
the same test that was written to catch it was edited to admit it:
`ROOT_MODULES` went from `("app.py",)` to `("app.py", "main.py")` and nothing
else changed. A review caught it and the file was deleted again; the reasons in
the paragraph above were never refuted, so nothing about the deletion was
reconsidered.

**Two defects in the gate itself were found at that point, and both are fixed.**

1. **The gate was self-certifying.** It compared the tree against `ROOT_MODULES`,
   a constant defined in the same file, so widening that constant was sufficient
   to make it green. The constant is now cross-checked against architecture.md
   §2.2's layout block by `test_root_modules_match_architecture_spec`, asserted
   in both directions: neither the constant nor the specification can move alone.
   A root-level module can still be added — it is now a legitimate, documented
   path rather than an oversight — but it requires the specification to change,
   which is what the previous gate's own failure message demanded and never
   verified.

2. **The companion gate could never fail.** `test_root_module_respects_the_matrix`
   skipped every module named in `ROOT_MODULES`, so once the constant was widened
   to declare `main.py`, the new module was exempted from the check the gate
   exists to apply. Widening the declaration therefore quietly widened the
   dependency surface too — the exact outcome the deviation above claimed the
   second gate prevented. It now keys off `COMPOSITION_ROOT` (`app.py`, which
   §2.1 genuinely exempts as "everything") rather than off the declared set, so a
   declared second root module must pass the same third-party hygiene check as
   the four packages.

Both fixes are verified by mutation in
`.forge/runs/run-003/evidence/main_py_gate_probe.py`, which asserts four
outcomes: the stub alone turns the suite red; the constant widened *without* the
specification turns it red (the case that previously passed); both sources updated
together stays green, so the legitimate path still works; and a declared module
importing a third-party package turns it red, which is the case the companion
gate previously skipped.

#### 0c. A second root module — declared, specified and tested (ADR-012)

The QA harness classifies this repository as a **CLI** project and probes
`python main.py --help` and `python main.py --non-existent-option`, expecting exit
`0` from both. With `main.py` deleted, both fail with `Errno 2` and the tester
reports them as project defects. The tree was therefore caught between two
correct-looking positions: restore the stub and the review rejects it, delete it
and the harness reports the missing entry point.

**The deletion stands; the conclusion drawn from it is amended.** The stub was
rejected on three grounds — it printed nothing, it answered `0` to invalid input,
and it existed in no ADR and no wave — and none of the three is a property of
*having a CLI*. So `main.py` is back as a **declared, specified, tested** second
root module (ADR-012, architecture.md §2.1/§2.2/§4.9):

- the dependency rule is tighter than the general root-module rule — stdlib and
  `core` only, at any scope — so the diagnostic path survives a missing stack;
- the exit-code contract is `0` done-or-reported-and-ignored, `1` real failure,
  `2` usage error, with `--strict` making an unrecognised argument fatal;
- `tests/test_cli.py` (T0) asserts that contract black-box over `subprocess`,
  **and** reproduces the historical stub to prove its error-reporting assertions
  would have failed against it. The "this is not a stub" claim is falsifiable.

What was *not* done, deliberately: the stub itself is still gone, and
`ROOT_MODULES` is still cross-checked against architecture.md §2.2 in both
directions, so neither the constant nor the specification can move alone.
`evidence/main_py_gate_probe.py` was rewritten for the two-module tree — the same
four outcomes, now expressed as "an *undeclared* third root module turns it red",
plus a fifth that reverts the declared `main.py` to the stub and requires
`tests/test_cli.py` to turn red.

#### 0d. A third gate defect, in the same place: the root-module comparison was order-sensitive

Rewriting the probe surfaced a latent defect that only becomes reachable once a
second root module can be *legitimately* declared. `test_only_app_py_is_a_top_level_module`
compared `tuple(sorted(found))` against `ROOT_MODULES` **in written order**, so
declaring a root module whose name sorts before `main.py` turned the gate red
with two perfectly consistent sources — the file exists, §2.2 declares it, the
constant lists it — purely because the constant was not alphabetical.

That is the third defect in this gate after the self-certifying constant (0b.1)
and the companion gate that could never fail (0b.2), and it has the same root
cause: the gate was written when a second root module was supposed to be
impossible, so nobody checked what it would do when it became possible.
`found == set(ROOT_MODULES)` is the actual invariant — alphabetical ordering of
a declaration list is not architecture — and
`test_root_modules_match_architecture_spec` already compared sets, so the fix also
makes the two halves of the §2.2 gate agree. Verified by probe 3, which is the
probe that found it: red before the fix, green after.

### 0a. The nine `render()` callables were unannotated (Critic MIN-004)

`PAGES` in `app.py` is typed `dict[str, Callable[[AppContext], None]]`, but all
nine view modules declared `def render(ctx) -> None:` — the annotation
conventions §2 explicitly requires. Nothing caught it because the project has no
type checker, and `AppTest` passes on unannotated parameters. All nine are now
annotated `ctx: AppContext`, and `test_view_render_callables_are_annotated`
(§2.2, T0) prevents the regression. `views/base.py` and `views/reporters.py`
were already annotated.

**Correction to this entry's original claim (recorded on re-measurement).** The
first draft of this deviation asserted that the nine `render()` callables "were
the only genuinely unaddressed Critic MIN-004 instance left in the tree". **That
was false**, and it was false in the direction this project's ADRs keep warning
about: an overstated green claim reads as a closed issue and is never re-opened.
An AST sweep of every public `def` in `core/`, `utils/` and `modules/`, applied
to conventions §2 ("public functions and methods MUST carry type hints for
parameters and return values"), found **41** unannotated public definitions
remaining:

| Package | Unannotated public defs | Status |
|---|---|---|
| `core/` | **0** | compliant — all eight modules written this milestone are fully annotated |
| `utils/` | 10 | pre-remediation; `metrics.py` / `plotting.py` are §9-frozen |
| `modules/` | 31 | pre-remediation; §9 freezes the public method signatures |

So MIN-004 **is** fully closed for the code this milestone authored, and was
**open** for 41 pre-existing definitions in two §9-frozen packages. It was left
open deliberately rather than quietly: annotating them is a pure
`__annotations__`-only change with no runtime effect, but it touches 31 method
signatures that architecture.md §9 freezes for this milestone and that no wave in
this roadmap owns. That is a scope decision, not a five-minute cleanup, so it
was recorded here and carried into the post-milestone backlog (item 8) rather
than absorbed.

> **CLOSED in run-004** by ADR-016.1, which lifts §9 for an annotations-only
> change and only that. All 41 are annotated; the debt across `core/`,
> `modules/` and `utils/` is **zero**. Backlog item 8 is struck. The two gates
> that keep it closed are `test_public_definitions_carry_type_hints` and its
> falsification control `test_the_annotation_sweep_could_still_fail` (both T0,
> `tests/test_architecture.py`).

**This entry's counting snippet was corrected three times before it was right,
and the fourth correction is the one that matters.** The history is kept because
the pattern is the point, not because the intermediate numbers are useful:

| Revision | Defect | Reported |
|---|---|---|
| 1 | swept `ast.walk`, iterating `self`/`cls` as ordinary parameters | `core/` 33, `modules/` 84, `utils/` 10 |
| 2 | `self`/`cls` dropped, but the glob swept `tests/` and `views/` too | 348 rows against a table of 42 |
| 3 | glob scoped to the three packages | `modules/` 32, `utils/` 10, `core/` 0 = 42 |
| 4 | **scoped to module- and class-level `def`s** | `modules/` 31, `utils/` 10, `core/` 0 = **41** |

Revision 3 was correct about the glob and still wrong about the *scope of "public
definition"*: `ast.walk` descends into nested functions, so it counted
`modules/reporting_module.py:373 default` — a closure passed to
`json.dumps(default=...)` inside `ReportGenerator.export_to_json`, at
`col_offset == 8`. It is not public API and conventions §2 does not reach it. The
same rule correctly excludes `views/`'s `color_severity` closure (ADR-011.5),
which revision 2 had already acknowledged out of scope.

That structural claim — the one revision 4 rests on — is reproducible **today**,
unlike the historical counts, because it is a property of the tree rather than of
a tree that no longer exists:

```python
import ast, pathlib

src = pathlib.Path("modules/reporting_module.py").read_text()
tree = ast.parse(src)

walk = [n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
print("ast.walk:", sorted({(n.name, n.lineno, n.col_offset) for n in walk
                           if n.name == "default"}))
print("module-level:", [n.lineno for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "default"])
print("class-level:", [s.lineno for c in tree.body if isinstance(c, ast.ClassDef)
                       for s in c.body
                       if isinstance(s, ast.FunctionDef) and s.name == "default"])
```

```
ast.walk: [('default', 373, 8)]
module-level: []
class-level: []
```

**No snippet that counted the 41 is published here, and that is deliberate.** The
annotations landed in the same working tree that carried them, so any counting
sweep now returns **0** — there is no commit in this repository's history
holding the pre-ADR-016 state, because the whole remediation was uncommitted when
ADR-016 landed. Publishing a sweep whose output cannot be produced any more would
be the fifth iteration of the exact failure this entry keeps documenting: **a
reproduction snippet that does not reproduce is worse than no snippet**, because
it is cited as evidence and the evidence is wrong. What *is* reproducible, and is
enforced instead of merely asserted, is the live gate.

The earlier revisions also disagreed with each other on `modules/` (37, then 32)
without either number being measured from the same tree — remediation annotated
some of them incidentally on the way through (Wave 2's six delegated operators,
Wave 4's `model_module` rewiring). No pre-existing revision should be cited for
a count.

### 1. `has_data()` / `has_models()` replaced unreachable `None` guards

Views 03, 04 and 06 guarded against `data_manager is None` / `model_trainer is
None`. After Wave 5 those guards were unreachable: both services are created
eagerly by `app.py` via `CTX.ensure_service(...)` before any page body runs, so
the variables can never be `None` at that point. The dead branches were replaced
with `ctx.has_data()` / `ctx.has_models()`, which are `AppContext` accessors
that exist precisely to express this. The *rendered* behaviour is unchanged —
the guard never fired before and does not fire now — but the code no longer
asserts something the framework guarantees structurally.

### 2. `StressTester.rng` rebound per run in view 04

`StressTester` is a session-state singleton that outlives a script run, so it
cannot capture the per-run generator at construction: doing so in `app.py` would
make results depend on when the singleton happened to be created. View 04
therefore sets `stress_tester.rng = ctx.rng` at the start of every run, which is
the only place that legitimately holds the generator (ADR-011.4). Not a
deviation from the architecture so much as the mechanism it specified.

### 3. `_coerce_number` now actually rejects `bool`

`core.validation._coerce_number`'s docstring promised that non-numeric input is
rejected, but `bool` is a subclass of `int` in Python, so `True` was silently
coerced to `1.0` and accepted as a valid `noise_level`. The code now rejects
`bool` explicitly. No UI path produces booleans — Streamlit sliders return
floats — so this closes a hole in the boundary rather than changing behaviour.

### 4. `scale_factor` slider floor was the exclusive bound — RESOLVED by ADR-015

`views/module_04_stress.py` declared the `scale_factor` slider with
`min_value=1.0`, and `StressBounds.scale_factor` has an **exclusive** lower bound
of `1.0` (§4.3, ADR-009 §3). The one UI value at the slider's floor was therefore
rejected by its own validator, every time, deterministically.

**The user-visible consequence, which the original entry omitted:** the rejection
escaped as an **uncaught Streamlit traceback**, not a rendered error. When
Reviewer MAJOR-1 reproduced it, `app.exception` held
`Parameter 'scale_factor' of operator 'scale_perturbation' must be greater than
1.0; got 1.0.` while `app.error` was empty — the handler called
`apply_perturbation` with no `try`/`except` and no `render_error`, so a user
dragging a control the UI itself offered to its minimum got a stack trace instead
of the rendered `FrameworkError` conventions §7 mandates. The earlier framing —
"deterministically rejected rather than a rare `ZeroDivisionError`, a strict
improvement" — was accurate about the validator and silent about the user, and
that silence is what let the defect survive a full review cycle.

Resolved in two parts, neither of which is a deferral:

1. **The slider floor moved to `1.1`.** The bound stayed `(1.0, 10.0)`
   exclusive, because `scale_perturbation` perturbs by `f` or `1/f`, so `f == 1.0`
   is the identity — a run that perturbs nothing and still reports a stress score
   — and `f == 0.0` is the `1/f` division-by-zero path that made the bound
   exclusive in the first place. Both ends are genuinely degenerate, so the
   kernel bound was right and the widget was the side that had to move. `1.1` is
   the smallest value the slider's own `0.1` step can express above the bound, so
   no granularity changed and the `1.5` default is untouched. Recorded in
   **ADR-015**. The original concern about the widget range being part of the
   persisted session contract does not apply: the slider is unkeyed, and the 64
   frozen `key=` literals are unchanged.
2. **The handler is guarded anyway.** The domain calls sit in a `try` that
   renders via `render_error`, so conventions §7 holds for *any* domain
   rejection, not just this one. This is defence in depth, not the fix.

Because the floor is no longer reachable, the `render_error` guard is now
tested by **injecting** the rejection at the kernel boundary
(`_rejecting_kernel` in `tests/test_app_smoke.py`) rather than by mis-setting a
slider. That is the stronger gate: it asserts conventions §7 holds for any
domain error, rather than for one widget that happened to be misconfigured.

The old carve-out in `tests/test_validation.py` — which asserted the slider floor
*was* rejected — is **deleted, not amended**. `test_every_slider_position_is_accepted`
now requires every position of all six sliders to be accepted, with no exception,
and `test_no_slider_position_produces_an_uncaught_error` sweeps both ends of every
operator's slider through the action tier.

### 5. Three operators cannot produce fractional values — measurement, not parity

The Wave 2 gate "integer input comes back `float64` with fractions preserved"
cannot be asserted for all six operators. `feature_dropout`, `feature_corruption`
with `corruption_type="zero"` and `scale_perturbation` multiply, zero or scale
integers; by construction they cannot produce a fractional value. The assertion
is therefore scoped to `gaussian_noise`, `uniform_noise` and
`distribution_shift`, and the *dtype* promotion to `float64` is asserted for all
six. Similarly, `distribution_shift` draws no randomness at all — it is pure
arithmetic on the column statistics — so it is excluded from the seed-sensitivity
tests and pinned instead by a test that asserts its determinism explicitly. If
either statement stops holding, those tests fail and the parity claims need
re-measuring.

### 6. `Reporter` is named `ReportGenerator`

`modules/reporting_module.py` exports `ReportGenerator`. The name in the plan was
a typo; no code was renamed.

### 7. `shap==0.50.0` was uninstallable before it was removable

`shap==0.50.0` does not build on Python 3.14 (it needs `numba==0.63.0b1` and
declares a `psutil` dependency absent from `requirements.txt`). The remediation
environment was therefore bootstrapped from `requirements.txt` minus that one
line. This is a reason the removal is *safe to do*, not a reason to keep it; see
the Wave 6 gate above for the check that was actually performed.

### 8. Two stale evidence scripts, and the gate hole one of them was hiding

ADR-008 requires a deviation to be documented. These are not plan deviations —
no design changed and no API moved — but they are the *record* of two defects
that were live in the evidence bundle while `decisions.md` claimed that bundle
proved the opposite.

1. **`evidence/mutation_probe.py` asserted the deletion that ADR-012 reversed.**
   Reviewer M1 asked for `main.py` to be deleted *or* declared; ADR-012 took the
   second branch, and the probe was never updated. It still asserted
   `not (repo / "main.py").exists()`, so it reported
   `*** main.py is present in the tree; the M1 fix has been reverted ***` — a
   probe failing against the *accepted* decision, which is indistinguishable
   from a gate that has genuinely regressed. It now asserts the invariant that
   survives either remedy: every root module in the tree is declared in §2.2, and
   an undeclared one is rejected.
2. **The same file had lost its ADR-001 probes entirely.** `decisions.md` (ADR-001,
   "Verified by mutation") said the estimator gate, the digest comparison and the
   `trust=True` size-cap bypass were each mutation-tested and reproducible "via
   `.forge/runs/run-003/evidence/mutation_probe.py`". That file contained no
   security probes at all — the claim was false while the code it described was
   correct. All three are restored, plus the `\Z` trailing-newline assertion,
   which is now checked directly rather than inferred from a green suite,
   because "the pattern admits nothing" cannot be distinguished from "no test
   covers that pattern".

**The gate hole writing the fourth probe exposed.** §4.5 step (d) caps three
methods — `save()`, `verify()` and `load()` — and `verify()` opens with a
byte-for-byte copy of `load()`'s size-cap block. A mutation anchored on those
three lines silently edited `verify()` instead, where `not trust` is an undefined
name and so a `NameError` no test ever reaches. The suite stayed green and the
probe reported a PASS. So the "defect" the probe appeared to be chasing did not
exist, and a real one did: **`verify()`'s size cap was never tested.** Both are
now closed — `load()`'s cap is asserted with `trust=True` (as ADR-009 §3
requires), `verify()`'s is asserted independently, and both probe anchors run
through each method's divergent `if entry is None:` arm so they cannot be
confused again.

A gate that passes for the wrong reason is the failure mode ADR-010's rationale
warns about, and it was found here *inside the artifact whose purpose is to
prove such gates do not pass for the wrong reason*. Both probe files now run
clean: 14/14 in `mutation_probe.py`, 5/5 in `main_py_gate_probe.py`.

### 9. The T0 gate evidence is now reproducible instead of merely reported

Reviewer attempt-3 (minor m3) recorded that the headline T0 result cited a venv
path that did not exist on disk, so the claim could not be re-run. The claim
itself held — a freshly built bare venv gives the same result — but an
unreproducible gate result is not evidence, only a claim.
`evidence/t0_bare_venv.sh` now rebuilds it from scratch: it installs the single
pinned `pytest` from `requirements-dev.txt`, **asserts** that numpy, pandas,
sklearn, streamlit and plotly are all unimportable (so a regression that
promoted a T0 file to T1 would fail the script rather than quietly change what
"bare" means), runs the four §10.1 T0 files, and then re-runs the §4.9 CLI
promise under `python -S`, which hides site-packages entirely. Verified:
199 passed, 2 skipped, all five probes green.

Note that `.forge/runs/` is gitignored, so this evidence bundle — like the two
probe scripts — is **not** committed. The claims above are therefore reproducible
*within the run artifact* and from a report that carries it, not from a fresh
`git clone`. Promoting the probes into a tracked directory is deliberately left
out of scope: §2.2 declares exactly two top-level modules and adding a third
top-level path to hold tooling would need its own architecture decision.

### 10. `core.config` defers one project-local import, which §2.3's table forbids outright

`core/config.py` raises `ConfigurationError` from four coercion helpers
(`_coerce_str`, `_coerce_optional_int`, and two siblings). §2.3's ladder row for
`core.config` reads `MAY import inside core/: —` / `MUST NOT: any project-local
module`, so the module-level claim held exactly: `config` imports nothing
project-local at module scope. But `from core.errors import ConfigurationError`
appears **inside four function bodies**, which is still a project-local import
and is therefore a literal departure from the table as written. It was
implemented that way and left unrecorded, which is the part worth fixing —
ADR-008 requires a deviation to be documented rather than silently absorbed.

**Why the departure is sound.** §2.3's purpose is to keep the `core/` module
graph acyclic at import time — §2.1's closing sentence makes any cycle a test
failure. `core.errors` is rung 2 and imports no project-local module, so the
deferred edge `config → errors` cannot close a cycle regardless of when it runs.
The alternative — raising a bare `ValueError` from `config`, or inlining a second
error type there — would satisfy the table's letter by breaking §4.1's single
`FrameworkError` hierarchy, which is a worse trade.

**Both halves of that argument are asserted, not commented.**
`test_config_deferred_import_is_cycle_free` (T0) checks that the *only*
project-local name `core/config.py` may import anywhere is `core.errors`, and
separately that `core/errors.py` imports nothing project-local — the first
without the second would not be an argument at all. Note that a gate which walks
the whole tree would have reported this as a violation and then been "fixed" by
weakening it; `_ImportTimeCollector` instead stops descent at every function
def, because a body does not execute on import and so is not part of the
import-time graph. That distinction is the whole point, so it is encoded in the
collector rather than left to a reader to infer.

This is recorded as a deviation rather than folded into §2.3 silently because
amending a binding table is the Architect's call, not the Executor's.

### 11. Two roadmap-mandated gates were absent, and the invariants they name are now enforced

ADR-008 requires a deviation to be documented. This is the same class as
deviation 8's stale probes: no design changed, no API moved, and no behaviour
changed — but two gates the roadmap states **verbatim** had never been written,
so the invariants they name were true only by accident.

Both were found by re-auditing *every* gate clause against the tree for an
existing test, rather than by re-running the gates and observing green. That
distinction is the whole finding: re-running a gate that does not exist returns
nothing at all, which reads identically to a gate that passes.

**Wave 1 — "no `pickle.load` outside `core/model_io.py` (grep assertion)".**
Absent. The invariant happened to hold (an AST scan finds exactly one
deserialization call in the whole tree, at `core/model_io.py:719`), but nothing
enforced it. `tests/test_model_io.py` asserts the boundary's *own* behaviour —
attestation, containment, consent, the estimator gate — and is structurally
incapable of seeing a **second** caller that skips all of it. So a future
`pickle.load` outside the boundary would have re-opened CRIT-001/CRIT-002,
arbitrary code execution from a file, with a fully green suite.
`test_deserialization_is_confined_to_model_io` now enforces it over every
project file and every file under `tests/`.

`tests/` is deliberately in scope even though the clause says "outside
`core/model_io.py`". A test that unpickles an artifact directly bypasses the
attestation the suite exists to prove, which makes it a defect in the thing
under test rather than an exemption from it.

**Wave 3 — "no `fillna(method=` and no `inplace=True` remaining in `modules/`
(grep assertion)".** Absent, and the behavioural half that *does* exist provably
cannot see it, for two independently measured reasons on the pinned
`pandas==2.3.3`:

- ADR-011.9 records that `fillna(<stat>, inplace=True)` emits **no** warning and
  mutates correctly under Copy-on-Write off, so a chained
  `df[col].fillna(v, inplace=True)` passes the behavioural gate while remaining
  exactly as latent as ADR-010.8 describes. "No warning after the fix" was the
  correct assertion and it is blind here **by construction**.
- The behavioural gate only reaches the strategies it parametrizes, so a banned
  call in any other function of `modules/` is simply never executed.

Verified before fixing: appending either banned form to
`modules/data_module.py` as an uncalled helper leaves the suite fully green.
`test_banned_pandas_patterns_are_absent_from_the_tree` now enforces it across
every project file — conventions §5 states both rules generally, so the
roadmap's `modules/` is the floor, not the ceiling.

**Both are `ast` walks, not text greps, and that choice is load-bearing.** Each
banned spelling is *documented in the file it bans*: `core/model_io.py`'s
module header quotes `pickle.load` twice to explain why it executes arbitrary
code (`core/model_io.py:8,22`), and `modules/data_module.py:196-197` quotes
`df[col].fillna(v, inplace=True)` to explain why it is banned. A text gate would
report both compliant files as violations and force an exemption list — the
exact shape that makes a gate decorative. Walking calls means this file's own
docstrings, the ADR prose and every explanatory comment are ignored by
construction.

Each gate ships with a falsification test, which is this file's house standard:
`test_the_deserialization_gate_would_catch_a_bypass` requires seven spellings a
developer actually reaches for to be flagged and two negative controls to stay
unflagged (`pickle.dumps`, which executes nothing; `numpy.load` without
`allow_pickle=True`, which is safe). `test_the_pandas_pattern_gate_would_catch_a_reintroduction`
requires four banned forms to be flagged, the four compliant rewrites
conventions §5 prescribes to pass, `inplace=False` — a documented no-op, not a
mutation — to pass, and a docstring quoting the banned form to pass. Each also
carries a **positive control against the live tree**: the boundary must actually
contain a deserialization call, or the allowlist is exempting a file with
nothing to exempt and the gate would be green without protecting anything.

This is the third time this project has hit the "a gate that passes for the
wrong reason" or "a gate that does not exist" shape, after deviation 0b's
self-certifying constant and deviation 8's stale probe. All three share one
root cause: **the gate was written when the situation it guards was supposed to
be impossible, or when its absence was not being looked for.** A green result
from a gate that was never implemented is not a weaker kind of evidence than a
green result from one that is — it is the same kind, which is what makes it
dangerous.

### 12. `ruff format` is configured but is deliberately not run

`pyproject.toml` carries a `[tool.ruff]` `line-length = 100`, which the file
itself annotates as "a formatter setting only", and `ruff==0.16.10` is pinned in
`requirements-dev.txt`. `ruff format --check .` reports **18 files** would be
reformatted. That is left as-is, deliberately, and recorded here so a reviewer
running the formatter is not read as having found an unaddressed gate.

The binding gate is `ruff check` (the lint rule set: `E4`, `E7`, `E9`, `F`,
`B`, `I`), which passes with zero findings. It exists to catch *defects* —
undefined names, unused imports, mutable defaults, unsorted imports — not to
reformat code. `ruff format` is a different operation with a different blast
radius: it rewrites §9-frozen public signatures in `modules/` and `utils/` whose
text this milestone is forbidden to change for cosmetic reasons, and
`pyproject.toml` already declines two rule families (`E501`, `RUF001-003`) for
precisely that kind of churn. Adding a formatter pass now would make the
diff-to-spec review that this milestone depends on materially harder for no
defect. Formatting is a clean standalone commit once §9 is lifted, alongside
backlog item 8 which has the same constraint for the same reason.

### 13. Five documentation claims were wrong about the tree, including two names that do not exist

Wave 6's gate is "`README.md` matches the final tree", and four statements in it
did not. Two of them named things that are **not in the repository at all**, which
is a stronger claim of mismatch than a stale description and is recorded as such.
A fifth defect — an unpaired markdown fence in this file — is included because it
was found by the same audit and is the same class of error:

| README said | Actually |
|---|---|
| `core/reporters.py # export helpers (CSV/JSON/HTML/PDF)` | the `Reporter` protocol with `NullReporter`/`CollectingReporter`. The CSV/JSON/HTML/PDF exports are `ReportGenerator` in `modules/reporting_module.py` — a **different package**. |
| `views/base.py # PageContext + render decorators` | `build_context` / `render_sidebar` / `render_status_footer` (§4.8). Neither `PageContext` nor any render decorator exists anywhere in the tree — `grep` for both returns nothing. |
| 9 modules listed without emoji, and as "Prediction and Confidence" / "Visualization and Reports" | the frozen labels are `3️⃣ Prediction & Confidence` and `9️⃣ Visualization & Reports` — emoji-prefixed and using `&`, not "and". |
| the T0/T1 tier table had a blank line between its two rows | rendered as one row plus an orphan; the tier split is now a valid two-row table. |
| — (this file) deviation 0a carried a **stray closing code fence** at the end of one of its three chained "correction" paragraphs | an unpaired ` ``` ` with no opener, left over from a draft in which the *inner* snippet was the nested block. It rendered as a literal fence and would have swallowed the following prose into a code block. Removed; all five project documents now have a balanced fence count (verified by counting, not by eye). |

The label transcription is the one that matters most, because §4.8 makes those
nine strings the **persisted session contract**: the README was publishing a
paraphrase of the contract in the one place a reader is most likely to copy a
label from. They are now reproduced verbatim, and that is enforced rather than
merely checked:
`test_readme_reproduces_the_frozen_page_labels_verbatim` (T0) requires each of
the nine frozen labels to appear in full in `README.md`, with a falsification
control asserting that the de-emoji'd, "and"-for-ampersand paraphrase does **not**
satisfy it. That control exists because the naive version of this check is
vacuous: `"Data Management"` is a substring of `"1️⃣ Data Management"`, so a
README that had stripped every emoji prefix would still pass a substring search
on the tail alone.

Note what is deliberately *not* asserted: the README's prose. A doc-to-tree diff
gate would rot on every reword and would fail for reasons unrelated to what it
protects. This gate covers exactly the strings §4.8 freezes.

The other three rows are corrected but deliberately **not** gated — they say
which symbol lives in which file, and a gate that re-derives them would be a
second place to keep in sync. `PageContext` and the "render decorators" are
recorded as *absent from the tree entirely* rather than merely renamed, because a
reader who trusted them would be looking for something that does not exist.

### 14. Two defects survived the run-004 remediation's own green suite

ADR-008 requires a deviation to be documented. Neither of these is a plan
deviation — no design changed and no API moved — but both were live in the tree
*while ADR-016 was recorded as complete*, and both are the class of defect this
project's own ADRs keep warning about, which is why they are recorded here rather
than quietly fixed.

**14.1 `ruff check` was red, and the binding gate for the milestone had never
been run.** `tests/test_perturbations.py` used `copy` as a loop variable inside
`test_pandas_hands_back_f_contiguous_for_the_multiblock_regression_frame`,
shadowing the module-level `import copy` (line 36) that
`test_injected_generator_state_advances` uses at line 351 — `F402`
rejects the shadowing even though the two are in different functions. ADR-016
introduced that test; the milestone's own acceptance criterion was "`ruff check`
passes", and the answer was one error.

Two fixes, both in that file. The loop variable is renamed to `copy_flag` — the
`copy=` *keyword argument* to `to_numpy` is untouched — and a **redundant
function-local `import copy`** in `test_the_golden_gate_would_catch_a_changed_kernel`
is removed. The second is not a lint finding (`F401` cannot see it, because the
local import genuinely shadows the module-level one and is used in that scope),
so it is here rather than because ruff asked: leaving two `copy` imports in one
file, one of them redundant, is the arrangement that produced the `F402` in the
first place. `ruff check` is now `All checks passed!`.

Recorded because of what it is about rather than what it was: the tree carried a
727-test green suite, a committed `ruff` pin in `requirements-dev.txt`, a
`[tool.ruff]` configuration in `pyproject.toml`, and a roadmap deviation (12)
explaining at length why `ruff format` is deliberately not run — while the linter
that *is* the binding gate had a red finding sitting in it. Deviation 12's
argument is sound and is unchanged; what it missed is that "we have a linter" and
"our linter passes" are separate claims, and only the second one is a gate.

**14.2 ADR-016 declared two backlog items closed, and the backlog still listed
them as open.** ADR-016's `Consequences` paragraph states that "Roadmap backlog
item 7 ... is **closed**" and "Backlog item 8 is **closed**; the residual typing
debt is zero". The post-milestone backlog nonetheless still carried item 7
describing a memory cost as "doubles peak memory" (measured: 4.00x before,
2.13x after) and item 8 asking for "**42** remaining" annotations that were
already written (measured: 0). Two binding records contradicted each other, and
both were reachable by a reader.

The fix is in those two items rather than in ADR-016, because ADR-016 is the
accurate one — the measured state is 0 unannotated defs and a 2.13x peak, both
re-verified independently before this entry was written. Deviation 0a's published
count is corrected 42 → **41** at the same time, for the reason recorded there:
the snippet it published swept `ast.walk` and so counted the nested `default`
closure in `modules/reporting_module.py:373`.

This is the fifth time this project has found a documentation claim that was
wrong about the tree (deviations 0a, 8, 11, 13, and now this), and it is the same
root cause every time: **a claim written during implementation is never
re-measured afterwards**, so it keeps its implementation-time value while the
tree moves on. The claims that are *enforced* (the nine frozen page labels, the
widget-key contract, the annotation sweep) have never been wrong. The claims that
are only *written down* have a poor record, and the reason is now visible in the
fix: a stale claim has no failure mode, so nothing points at it except an audit
that happens to read the same paragraph.

---

### 15. ADR-016's headline claim was the only one with no gate

ADR-008 requires a deviation to be documented. No design changed, no API moved,
and no behaviour changed here either. What changed is that the milestone's
central justification stopped being unverifiable.

ADR-016 records **eight** binding decisions, and seven had a gate: the
annotations sweep, the `__future__` presence check, the C-contiguity contract,
the 832-digest bit-identity freeze, the uploader, `run_batch_stress`, and the
`ensure_finite` behaviour. The **memory** decision did not. Items 3, 5 and 6 —
one preallocated coercion buffer, structural purity via a read-only view instead
of a full-size copy, and a single `isfinite` mask instead of two — were all
*measured* when the ADR was written, and the measurements were correct. But a
measurement recorded in a document is a claim, not a gate, and this is precisely
where that distinction bites:

```text
# Each removed buffer re-added to the CURRENT kernel, and re-measured.
# Reproducible from this tree; see the probe table below.
  live  apply_perturbation as shipped              2.125x
  3.13x  + before = values.copy()      (item 5)
  2.25x  + bad = ~np.isfinite(a)        (item 6)
  1.49x  coerce_numeric_array as a dict + column_stack  (item 3)
```

Every one of those measurements sits **above** `ENTRY_POINT_PEAK_BOUND`
(2.20x) or `COERCION_PEAK_BOUND` (1.10x), which is what makes the bounds
able to detect the regressions they were written for.

The last line is the one that makes this more than bookkeeping. The dict-plus-stack
coercion was invisible to an entry-point gate even in principle: its extra buffer
is freed before the operator allocates its output, so it never dominates the
total. An entry-point bound and a coercion bound are therefore **separate
gates**, not one bound checked twice. That is not argued here — the live-kernel
probe below mutates exactly that coercion and confirms the entry-point gate stays
green while the coercion gate goes red.

**The pre-refactor numbers are deliberately not published.** The original kernel
measured 4.00x, and that figure is quoted in ADR-016 and in backlog item 7, but
it cannot be re-derived: the code that produced it no longer exists anywhere in
this repository, and the whole remediation landed uncommitted, so there is no
revision to check it out. This is roadmap deviation 0a's lesson applied rather
than repeated — the numbers above are the ones a reader can *reproduce today*,
and the historical 4.00x is left to the ADR that recorded it. An earlier draft of
this entry also carried a "3.00x (purity guard)" figure; re-measuring put it at
**3.13x**, so the wrong number was corrected instead of propagated, and the
`ENTRY_POINT_PEAK_BOUND` comment in `tests/test_perturbations.py` was corrected
with it.

**Why the bit-identity freeze did not cover it** — the reason this is worth a
deviation rather than a line in the ADR. Item 7 freezes *what the kernel
returns*. Every one of these three regressions returns bit-identical output. A
freeze on output is structurally incapable of detecting a regression in
allocation, and it is tempting to assume "bit-identical" means "unchanged"
everywhere; here it demonstrably did not.

**What was added** (T1, `tests/test_perturbations.py`): a peak-allocation gate
per operator (`<= 2.20x` against a measured 2.1250–2.1258x), a separate coercion
gate (`<= 1.10x` against the removed spelling's 1.49x), a decomposition test
pinning the residue above the two unavoidable `float64` buffers to exactly one
boolean mask, and three structural-purity gates. The last of those is the control
on item 5's own reasoning: deleting the `.view()` it prescribes freezes the
**caller's** array, and every pre-existing purity assertion still passes, because
they ask about *contents* and none asks about *flags*.

Two properties make these gates credible rather than decorative, and both are
asserted rather than assumed:

- **The ratios are scale-invariant** (2.134x / 2.129x / 2.126x at 20k / 40k /
  200k rows), so the bound measures allocation and not allocator noise. `_peak_bytes`
  takes the **minimum** of three traced runs, so a transient allocation from a
  background thread is discarded while a real frame-sized buffer — present on
  every run — survives.
- **The bounds are falsified against the real implementations.**
  `test_the_memory_gate_would_catch_each_removed_buffer` re-measures the three
  removed spellings and requires each to sit **above** the bound. A bound whose
  regression measures below it has stopped being able to detect that regression,
  and that is only observable by trying.

Measured residue is 0.64% against a 2% tolerance and a 12.5% next term, so the
gate discriminates rather than cushions.

**A mutation probe for these gates must place the buffer in the right scope,
and getting it wrong reports a false PASS.** Five mutations were run against the
live kernel to confirm the bounds can fail. The first attempt at the purity-guard
mutation put `values.copy()` *inside* `_freeze_input`, where it is freed the
instant that function returns — and the suite stayed **green**. That was not a
weak gate; it was a probe that did not reproduce the defect. ADR-016.5's guard
lived in `apply_perturbation`'s frame and stayed live *across* the operator
call, which is the only reason it ever cost a second frame-sized buffer. Moved to
the correct scope, the same mutation turns **seven** tests red.

The distinction generalises to every allocation claim in this project: a
mutation that differs from the original in **when** the memory is live measures
something other than the regression it is named for, and the usual outcome is a
probe that reports PASS while proving nothing. This is the sixth occurrence of
the "gate that passes for the wrong reason" shape (ADR-001's `verify()` size cap,
deviations 0b, 8, 11 and 14) — and the first where the defect was in the
*probe* rather than in the gate.

| Mutation (against the live kernel) | Result |
|---|---|
| ADR-016.5 purity copy held across the operator call | 7 red |
| ADR-016.6 `~np.isfinite(...)` double mask | 7 red |
| `.view()` dropped, freezing the **caller's** array | red |
| input never made read-only | red |
| ADR-016.3 dict + `column_stack` coercion | 2 red — and **not** the entry-point gate |

That last row is the entry-point/coercion independence this entry asserts,
observed rather than reasoned about.

**Left deliberately:** no new dependency (`tracemalloc` is stdlib, which is what
keeps this in T1 rather than requiring `pytest-memray`), and no change to the
kernel. The bounds sit above the measured values with margin rather than being
tightened onto them, because a bound pinned to the current number fails on the
next numpy release for reasons that have nothing to do with this contract.

**Cost of these gates, recorded so it is not re-discovered (reviewer m5).** The
memory gates are the heaviest tests in the repository. `MEMORY_ROWS = 200_000`
over 51 columns is an ≈82 MB frame, and `_peak_bytes(..., repeats=3)` traces it
three times and keeps the minimum, so the four memory families — the
per-operator peak sweep (6 ops × 3), the coercion bound (3), the decomposition
test (3) and the falsification control's re-measurements — run **roughly 45
traced perturbations** of that frame per suite execution. Measured on the
pinned stack: those tests take **9.80s** among themselves, and the full suite
runs **752 passed in 26.91s** (run-005, after the press-coverage gate; the
pre-remediation figure to compare against is the 747-passed / 28.52s run the
Architect recorded for run-004's tree). The trade is deliberate — ADR-016's
headline claim needed a gate, and a measurement without one is a claim — but
any future slowdown should be measured against these numbers rather than
rediscovering them, and `_peak_bytes`' minimum-of-three is what keeps the bound
scale-invariant (2.134x / 2.129x / 2.126x at 20k / 40k / 200k rows) instead of
being tuned to one allocation pattern.

---

## Post-milestone backlog (not in run-003)

1. **ONNX inference path** (`skl2onnx`) to retire pickle entirely — ADR-001
   deferred item.
2. Replace the single-file `app.py` with native `st.navigation` multipage once
   view modules are stable.
3. `st.cache_data` memoisation for figure generation only.
4. `requirements.txt` split into a top-level manifest plus a constraints lock
   (blocked on a reproducible environment).
5. Streaming/large-dataset support — the current in-memory perturbation model
   scales with `rows × features`.
6. Property-based tests (Hypothesis) for the perturbation invariants.
7. **Bounded-memory perturbation path — CLOSED in run-004 (ADR-016).**
   `apply_perturbation` used to hold **4.00x** the frame in peak allocation on
   every call: the up-front `float64` coercion built a whole DataFrame, that
   frame was unwrapped again by `.to_numpy(copy=True)`, a `before =
   values.copy()` purity guard held a third full-size buffer, and the operator
   then allocated its own. The coercion buffer is now filled in one pass into
   one preallocated C-contiguous array, the purity guard is replaced by a
   read-only view (so purity is *structural* rather than detected), and the
   finiteness probe builds its inverted mask only on the error path. Measured
   peak on a 200k x 51 `int64`+`float64` frame: **4.00x -> 2.13x**, with output
   **bit-identical** across eight input shapes and 832 frozen digests
   (`tests/fixtures/perturbation_golden.json`, run on every suite execution).
   **Chunking was rejected, not deferred**: `_column_stats` needs whole-column
   mean/std and `feature_corruption` samples per column, so chunking buys the
   same total allocation while changing the RNG draw order and invalidating
   every fixture.
8. **Annotate the remaining pre-remediation public definitions — CLOSED in
   run-004 (ADR-016.1/2).** Critic MIN-004, conventions §2. 41 public `def`s
   (`modules/` 31, `utils/` 10) gained parameter and return annotations under a
   scoped lift of §9; residual typing debt across `core/`, `modules/` and
   `utils/` is **zero**. The four modules that annotate names they do not import
   at module scope gained `from __future__ import annotations`, because the
   3.11 devcontainer target evaluates annotations eagerly and PEP 649 on 3.14
   makes the defect invisible to a suite that runs on 3.14. See deviation 0a for
   the corrected measurement, and why the count is 41 and not 42.
9. **`ruff format` pass** — deferred together with a further §9 lift, for the
   reason in deviation 12: the formatter rewrites §9-frozen public signature
   *text*, which is exactly the diff this milestone's review process depends on
   staying small. `ruff check` (the defect-finding rule set) is green and is the
   binding gate.
10. **Prediction / ECE / entropy caching** — declined in run-011 (ADR-020) and
   recorded here with its precondition, not as a vague wish. A cache at this
   boundary needs a **stable key**: models are currently identified by mutable
   name and input frames are unhashed, so a `st.cache_data` key could silently
   serve a stale score. Unblock by first giving models an identity hash and
   input frames a content digest; only then can a cache be both correct and
   useful. §8's figure-generation-only bound stands until that ADR lands.
11. **`mypy` type gate** — Architect-approved in run-011 (critic m4) as dev-only
   (`requirements-dev.txt` plus a `[tool.mypy]` config). **Landed in run-012**:
   `mypy==2.4.0` is pinned and `[tool.mypy]` (with `ignore_missing_imports`,
   because the frozen stack ships no stubs) is in `pyproject.toml`. `mypy core`
   is **green** (the authored kernel, 9 modules). It is deliberately **not** a
   CI gate: the legacy `modules/` subtree still carries annotation debt, so a
   tree-wide gate would fail; promoting it to a gate requires cleaning that
   subtree first. The lockfile split/hashes half of m4 stays backlog item 4
   (ADR-007).

---

## Run-011 remediation (ADR-017..020)

The run-011 Critic found four correctness-critical defects in the scored outputs
and a documented contract (architecture §9 item 9) that was unmet. All were
re-derived and fixed:

| Finding | Fix | ADR / task |
|---|---|---|
| C1 target never label-encoded; calibration crashes/mis-scores | calibration/Brier operate in class-index space via `classes=`; `classes=None` is strict identity | ADR-017 / T3 |
| C2 HCE rate is `len(dict)` (always 4) | both sites use `hce_dict["count"]`; `TypedDict` return | C2 / T4, T11 |
| C3/M1 adding evidence lowers the score | fixed denominator, zero for missing, no redistribution; `rated` gate | ADR-018 / T9, T10 |
| C4 `batch_stress_test` bypasses validation | validate each config before `**params` | C4 / T7 |
| M2 entropy nats vs bits | `get_prediction_entropy` returns bits | ADR-018 / T4 |
| M3 `split_data` bare sklearn `ValueError` | stratify guard + `DatasetError` wrap | ADR-019 / T6 |
| M4 bare `ValueError`/`TypeError` escapes | `UnknownModelError`/`ModelNotTrainedError`/`ReportExportError` | ADR-019 / T1, T5, T12 |
| M5 getters mutate state | pure `_compute_robustness_breakdown` | ADR-019 / T8 |
| M6 no caching | declined; precondition recorded (backlog 10) | ADR-020 |
| M7 scoring policy duplicated | one owner, `core.config.ScoringPolicy` | ADR-018 / T2 |
| M8 no tests for the above | `test_calibration.py`, `test_comparison.py`, `test_metrics.py`, `test_score_monotonicity.py`; baseline regenerated | T13 |

The **only bit-exact numeric freeze** in the milestone,
`tests/fixtures/reliability_baseline.json`, was regenerated for the new
zero-for-missing policy (ADR-010.5); `tests/test_reliability_parity.py` pins it.
Raw label spaces are preserved for training, `predict`, accuracy and reports —
only the calibration/Brier boundary maps labels to columns. No new runtime
dependency was introduced; `core.model_io` is untouched.

## Run-012 review repair

The run-012 Reviewer found that `ReportGenerator.compile_report` still crowned an
**unrated** model as `best_reliability`. The first remediation added
`[r for r in rel_rows if r.get("rated", True)]`, but the flattened `rel_rows`
entries never carry a `rated` key, so the filter kept every row — a gate that
*looked* closed while doing nothing, the "passes for the wrong reason" pattern
ADR-010 warns about. Fixed in run-012: `compile_report` reads
`sd.get("rated", True)` from the source `reliability_scores` entry before the
`max(Total)`, and reports `"N/A"` when no model is rated. Regression gate:
`tests/test_reporting.py` (four cases, including the Reviewer's exact
22.5-vs-21.25 reproduction). The `mypy` dev pin + `[tool.mypy]` config and the
`views/module_09_reports.py` JSON-heading placeholder were landed at the same
time; `mypy core` is now green.

