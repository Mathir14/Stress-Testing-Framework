# Architectural Decisions

Numbered, append-only. Superseded decisions stay with status `Superseded`.
Numeric defaults referenced here are frozen by ADR-005 parity tests.

---

## ADR-001 — Model persistence uses a provenance/containment boundary, not "safe unpickling"

- **Status:** Accepted
- **Context:** Critic CRIT-001/CRIT-002 flagged `pickle.load()` in
  `modules/model_module.py:371-372` as arbitrary code execution. The audit
  suggested joblib or ONNX.
- **Decision:** Keep the pickle format (no dependency change, no model
  conversion pipeline this milestone) and place all persistence behind
  `core/model_io.ModelArtifactStore`, which enforces: filename regex, path
  containment via `Path.resolve()` + `is_relative_to`, symlink rejection, a
  SHA-256 attestation manifest written atomically at save time, hash
  re-verification **before** deserialization, an `sklearn.base.BaseEstimator`
  type gate, and an explicit `trust=True` opt-in that the UI only supplies from a
  default-unchecked confirmation widget.
- **Rationale:** `joblib` is pickle-based and provides no security benefit; a
  restricted unpickler is not implementable for arbitrary estimators; ONNX
  migration requires `skl2onnx` plus a conversion layer for all three model
  families and breaks `predict_proba`/feature-importance parity. The genuine,
  enforceable control is provenance plus containment plus informed consent.
- **Consequences:** Arbitrary file write on save is also closed. Residual risk is
  explicitly documented (module docstring, README, this ADR) — this is a
  single-user local research tool, not a multi-tenant service.
- **Deferred:** ONNX/`skl2onnx` inference path (tracked in roadmap.md).

  **Verified by mutation.** The boundary is probed by
  `.forge/runs/run-003/evidence/mutation_probe.py`, which mutates one control at a
  time and requires the suite to go **red** each time: deleting the estimator
  type gate, disabling the digest comparison, letting `trust=True` wave through
  the load-time size cap, and deleting `verify()`'s size cap. `re.fullmatch` with
  the ADR-009 `\Z` pattern is asserted directly — `fullmatch` against
  `"model.pkl\n"` is `None` — because "the pattern admits nothing" is not
  distinguishable from "no test covers that pattern", so it is checked directly
  rather than inferred from a green suite.

  **A false PASS found by writing that fourth probe (recorded because the
  lesson is the point).** §4.5 step (d) caps *three* methods — `save()`,
  `verify()` and `load()` — and `verify()` opens with a byte-for-byte copy of
  `load()`'s size-cap block. A mutation anchored on those three lines silently
  edited `verify()` instead of `load()`, where `not trust` is an undefined name
  and therefore a `NameError` no test ever reaches. The suite stayed green and
  the probe reported a PASS: **a gate that passes for the wrong reason, in the
  one file whose whole purpose is not passing for the wrong reason.** The
  apparent "trust bypasses the size cap" defect did not exist; the untested
  `verify()` cap did. Both are now covered — `load()`'s cap is asserted with
  `trust=True`, `verify()`'s is asserted independently, and the probe anchors run
  through each method's divergent `if entry is None:` arm so they cannot be
  confused again.

---

## ADR-002 — Streamlit decoupling is narrow, and the UI is decomposed by strangler extraction

- **Status:** Accepted
- **Context:** AV-002/MAJ-001/MAJ-006 claim all modules are Streamlit-coupled and
  that the 3517-line `app.py` must be rewritten.
- **Decision:** (a) Decouple only what is actually coupled: `data_module.py`
  (17 `st.*` refs) and `model_module.py` (2). Domain code raises typed errors and
  emits messages through an injected `Reporter`; `views/reporters.py` holds the
  only Streamlit messaging adapter. (b) Decompose `app.py` mechanically into
  `views/module_01..09*.py`, each exposing `render(ctx)`, driven by a thin
  composition root.
- **Rationale:** Verified evidence contradicts the audit's "all modules"
  framing; a full MVC rewrite of a working 3.5k-line app carries far more risk
  than the coupling it removes. Mechanical extraction with unchanged widget
  `key=` literals and unchanged evaluation order is reviewable and reversible.
- **Consequences:** `views/` becomes the Streamlit-facing layer; `tests/` can
  import everything except `views/` without a Streamlit runtime.

---

## ADR-003 — One pure perturbation pipeline replaces duplicated operator branches

- **Status:** Accepted
- **Context:** CRIT-003 (no parameter validation), MAJ-002 (in-place mutation),
  MIN-007 (division by zero).
- **Decision:** `core/perturbations.py` implements each operator once over a 2-D
  float array, wrapped by `apply_perturbation()` which handles DataFrame/ndarray
  containers, injects `np.random.Generator`, validates parameters, and asserts
  finiteness. `modules/stress_module.py` delegates and keeps its public methods.
- **Rationale:** The current `DataFrame.values` in-place write silently
  **discards** results on mixed-dtype frames (`.values` returns a new object
  array), so several operators are latent no-ops; twelve duplicated branches
  cannot be validated consistently.
- **Consequences:** Operator semantics are parity-tested against the documented
  formulas, not against the buggy branch behaviour. Column order/index/dtype
  preservation becomes an explicit contract.

---

## ADR-004 — Errors are values in the view layer, exceptions in the domain layer

- **Status:** Accepted
- **Context:** MAJ-004 — silent failures (`st.error` + `return None`,
  `except: brier = NaN`).
- **Decision:** Introduce the `FrameworkError` hierarchy in `core/errors.py`.
  Domain methods raise; `views/` catch and render. The only permitted
  degradation is per-class Brier → `NaN`, which additionally logs a warning.
- **Rationale:** Returning `None` conflates "no data" with "failed"; raising
  keeps the domain layer testable without a UI.
- **Consequences:** `DataManager.load_dataset` changes from `None`-on-failure to
  raising `DatasetLoadError` (single call site, `app.py:110`).

---

## ADR-005 — Central configuration with bit-exact defaults

- **Status:** Accepted
- **Context:** MIN-005 / AV-004 — magic numbers in reliability scoring, stress
  bounds and grade boundaries.
- **Decision:** `core/config.py` owns all tunables as frozen dataclasses;
  `get_config()` returns a cached singleton, overridable by keyword overrides or
  `STF_*` environment variables. Defaults reproduce current behaviour exactly:
  reliability components 25 pts each, `ece_cap=0.5`, `drop_cap=0.5`, grades at
  90/80/70/60/50.
- **Rationale:** Centralisation without parity testing would silently change
  reported scores; parity is therefore a release gate, not a nicety.
- **Consequences:** `ReliabilityScorer` reads its constants from config;
  `tests/test_reliability_parity.py` freezes the mapping.

  **Implemented as (Wave 6).** `ReliabilityWeights` owns the four maxima, both
  caps, the grade cutoffs **and the grade colours**, with
  `resolve_grade(score)` / `colour_for(grade)` as the single resolution path.
  `ReliabilityScorer.__init__(weights=None)` binds the policy — `None` means the
  process-wide singleton, which is the only construction the app performs — and
  re-binds the four `MAX_*` attributes per instance so an injected policy is
  honoured. The `MAX_*` names survive as a class-level mirror of the configured
  defaults so `ReliabilityScorer.MAX_PERF` still resolves for class-level
  readers; they stay writable rather than becoming properties, because narrowing
  them to read-only would be an unapproved API change. The module-level
  `_grade(score)` is retained and now delegates to
  `get_config().reliability.resolve_grade`; `score_model` uses the instance
  method so an injected policy cannot disagree with the instance's own maxima.
  Parity re-verified bit-exactly with `python -m tests.capture_baseline --check`.

---

## ADR-006 — Layering is enforced by an executable import test

- **Status:** Accepted
- **Context:** Conventions that are not enforced decay (MIN-003, MIN-004).
- **Decision:** `tests/test_architecture.py` parses every project file with `ast`
  and asserts the layer matrix in `architecture.md` §2.1 (no `streamlit` in
  `core/`, `modules/`, `utils/`; no `views`/`app` imports below the view layer; no
  imports of `app`; no cycles).
- **Rationale:** The rule set is small, mechanical and cheap to check; running it
  on every change prevents regression to the current mixed-responsibility state.
- **Consequences:** Requires `pytest` in the dev environment
  (`requirements-dev.txt`). The test runs on stdlib + pytest only.

---

## ADR-007 — Dependency file stays a fully pinned lockfile

- **Status:** Accepted
- **Context:** MIN-002 — unused `shap==0.50.0` (and unused direct `matplotlib`,
  `seaborn`, `altair`, `joblib`) in `requirements.txt`.
- **Decision:** Remove `shap` only. Keep `matplotlib`, `seaborn`, `altair` and
  `joblib` pinned: they are transitive requirements of streamlit / scikit-learn /
  plotly and removing pins from a lockfile-style file creates environment churn
  for no benefit. Add `requirements-dev.txt` for pytest.
- **Rationale:** Verified by grep that no project file imports `shap`; the others
  are pulled in by pinned direct dependencies regardless.
- **Consequences:** `shap` removal must be validated with a reverse-dependency
  check (`pip show shap` → required-by) before committing.

  **Check performed (Wave 6).** Three independent queries, all negative: no
  project `.py` file mentions `shap`; no installed distribution lists it in
  `Requires-Dist`; and it was never present in the environment. `numba` and
  `llvmlite` — which look like `shap`-shaped leftovers and were checked at the
  same time — were **kept**: `numba` is declared by `pandas` (conditionally), so
  it is a pinned transitive of a direct dependency, which is precisely the case
  this ADR declines to churn. `shap==0.50.0` also does not build on Python 3.14
  at all, which is a second, independent reason it could not have been a real
  dependency of a working environment.

  **`slicer` is the one orphan this ADR's scoping left behind, and it is NOT
  removed — recorded here rather than silently absorbed (ADR-008).** The
  Architect's finding **D1** correctly observed that ADR-007 "does not name
  `shap`'s orphan pins (`slicer`, `numba`, `llvmlite`, `cloudpickle`)". Re-running
  the reverse-dependency query *after* the `shap` removal, with the whole set of
  candidates rather than only the two this ADR checked, settles each one:

  | Pin | Required by (from installed `Requires-Dist`) | Verdict |
  |---|---|---|
  | `numba` | `pandas` (conditionally) | keep — pinned transitive of a direct dependency |
  | `llvmlite` | `numba` | keep — transitive of the above |
  | `cloudpickle` | `xgboost` | keep — pinned transitive of a direct dependency |
  | `slicer` | **nothing** | **orphan** — `slicer` was `shap`'s dependency and nothing declares it now |

  So three of D1's four candidates are correctly retained by ADR-007's own
  rationale, and `slicer==0.0.8` (`requirements.txt:45`) is a genuine unused
  direct pin that installing `shap` was the only reason to have. No project file
  references it.

  **It is nevertheless left in place, deliberately.** This ADR's decision is
  "Remove `shap` only", conventions §9 forbids restructuring the lockfile in this
  milestone, and removing one more pin is precisely the per-pin lockfile churn
  this ADR declines. It is a **binding-ADR scope question, not an Executor call**,
  so it is escalated rather than taken. The owning work already exists:
  roadmap.md's post-milestone backlog item 4 splits `requirements.txt` into a
  top-level manifest plus a constraints lock, and the honest place for
  `slicer` — along with the 57-package transitive-dump concern the Critic raised
  as m28 — is that manifest, where "declared direct dependency" and "pinned
  transitive" stop being the same list. Reproduction:

  ```python
  import importlib.metadata as md
  rev = {}
  for d in md.distributions():
      for r in d.requires or []:
          dep = r.split(";")[0].strip().split()[0].split("[")[0].split("=")[0]
          rev.setdefault(dep.lower(), set()).add(d.metadata["Name"])
  print(sorted(rev.get("slicer", ())))   # -> []
  ```

---

## ADR-008 — Documentation stubs become binding project records

- **Status:** Accepted
- **Context:** MIN-001 — `.ai/project/*.md` were placeholders.
- **Decision:** `architecture.md` (this remediation's target design),
  `conventions.md` (coding rules), `decisions.md` (this file) and `roadmap.md`
  (sequenced waves) are populated and MUST be updated when behaviour changes.
- **Rationale:** The agent workflow reads these files before every task; leaving
  them empty makes every future stage re-derive the same context.

---

## ADR-009 — Security-boundary and sequencing defects found in ADR-001..ADR-008 during verification

- **Status:** Accepted
- **Context:** An architecture review re-derived every load-bearing claim of
  ADR-001..ADR-008 against the working tree (grep + `python3` probes). All
  substantive claims held, but four defects were found *in the specification
  itself*, two of them inside the security boundary ADR-001 claims to close.
- **Decision:**
  1. `filename_pattern` defaults to `r"[A-Za-z0-9][A-Za-z0-9._-]*\.pkl\Z"` and
     MUST be applied with `re.fullmatch()`. Python's `$` also matches *before a
     trailing newline*, so `re.match(r"^...\.pkl$", "model.pkl\n")` returns a
     match — a live sanitizer bypass in a boundary whose entire value is that it
     cannot be bypassed.
  2. `ArtifactPolicy.max_filename_length = 128`. Without a cap, an over-long
     name raises a raw `OSError: [Errno 36]`, leaking a non-typed error across
     the module boundary (conventions §3) before attestation is ever consulted.
  3. `load()` ordering is fixed (resolve → length → exists → size → attestation
     → estimator gate) and `trust=True` bypasses **only** attestation. Containment
     controls are not consent controls; ADR-001's original wording left it
     ambiguous whether `trust=True` could wave a path-traversal check through.
  4. Wave 1 edits `app.py:665-725` in place; `views/module_02_baseline.py` does
     not exist until Wave 5. The original wording made Wave 1's only UI change
     conditional on a directory that cannot yet exist.
  5. `modules/post_stress_module.py` is named explicitly as a third matrix
     violator (unused `import streamlit as st`, zero `st.*` calls) so the Wave 4
     gate does not fail on an unanticipated file.
  6. `ReliabilityWeights.grade_cutoffs` is documented as 5 entries plus an
     implicit `"F" / "#C0392B"` fallthrough below 50.0 — the constant must not
     silently lose the failing grade during the config extraction.
- **Rationale:** A security boundary whose spec contains a bypassable regex is
  worse than an obviously absent one, because it produces false assurance and is
  hard to review. Fixing it at the specification stage costs one line; finding it
  after implementation costs a code review cycle and a CVE-shaped question.
- **Consequences:** Adds `test_model_io.py` cases for the trailing-newline name,
  an over-long name, and a `trust=True` traversal attempt. Adds a
  `bounds=None`-equals-`get_config().stress_bounds` parity test. No behaviour of
  the running application changes; all four fixes tighten validation that was
  previously either ambiguous or untested.

---

## ADR-010 — Test suite is tiered, and the estimator gate is injected (defects found in the verification pass)

- **Status:** Accepted
- **Context:** A second verification pass executed the claims of ADR-001..ADR-009
  against a live interpreter (isolated venv, `numpy==2.4.2` / `pandas==2.3.3` /
  `scikit-learn==1.8.0`) and against the working tree. Nine further defects were
  found — in the specification's self-consistency, in one demonstrably false
  roadmap premise, and in the environment assumptions. ADR-009 fixed the
  *security* defects; these are the correctness and executability ones.
- **Decision:**
  1. **T0/T1 tiering is binding** (`architecture.md` §10.1). `test_architecture.py`
     and `test_model_io.py` must import stdlib + pytest only. Consequently
     `core/model_io.py` MUST resolve its estimator base lazily / by injection
     (`estimator_bases` ctor argument, §4.5.6) instead of importing scikit-learn
     at module level. A module-level `from sklearn.base import BaseEstimator`
     transitively pulls in sklearn + numpy + scipy and makes the security gate
     unrunnable exactly where it matters most.
  2. **`ArtifactPolicy.max_size_bytes` added (ADR-010.2).** §4.5 mandated a size cap in the
     `load()` ordering while `ArtifactPolicy` exposed no such knob — the gate
     referenced a setting no implementation could read.
  3. **dtype widening and coercion errors are declared behaviour changes**
     (§4.4.1, §9 items 4–5). Measured: an `int64` ndarray is perturbed in place
     and returned as `int64`, so fractional perturbations are silently truncated.
     Output is now `float64`, and a non-numeric frame raises `ValidationError`
     naming the offending columns.
  4. **The silent no-op is broader than "mixed-dtype".** `.values` yields a view
     only when the block dtype matches the frame's storage dtype; an
     `int64 + float64` frame upcasts to a temporary and the write is discarded.
     Measured severity on that shape: **5 of 6 operators silently no-op**
     (`feature_dropout` survives only because it writes back via `.iloc`). The
     regression test frame is therefore pinned to `int64 + float64`, and a
     `str`-bearing frame is explicitly *not* a valid regression frame.
  5. **Parity means formula parity, not bit-fidelity** (§4.4.3). Vectorised numpy
     and an injected `Generator` cannot reproduce the old per-column
     `np.random.*` draw order. The only bit-exact freeze is reliability scoring,
     which is pure arithmetic. A test asserting equality with the old
     implementation's *numbers* is a defect, not a stronger test.
  6. **Registry keys are binding** (§4.4): the six `snake_case` strings
     `stress_module.batch_stress_test` already dispatches on. Renaming them
     silently disables every batch stress test.
  7. **Recorded naming debt** (§4.4.6): `feature_dropout` masks per *cell*, not
     per feature, and `distribution_shift("variance")` contracts toward the mean.
     Docstrings and UI help text get corrected; the arithmetic does not change,
     because the names are load-bearing for published results.
  8. **Wave 3's premise is corrected.** Its gate described
     `handle_missing_values` as having a "current silent no-op". Measured under
     the pinned pandas 2.3.3 with Copy-on-Write off, chained
     `df[col].fillna(v, inplace=True)` on a copy **does** mutate the frame. The
     real defects are a `FutureWarning` on every call and latent breakage the
     moment CoW is enabled or pandas 3.0 lands. The gate becomes "strategies
     change the frame **and** emit no warning", not a regression test for a bug
     that is not yet active.
  9. **Toolchain is recorded as a constraint** (§10.3): no dependencies are
     installed, the local interpreter is 3.14.7 against a 3.11 devcontainer, and
     there is no CI. Wave 0 must bootstrap a venv and record the interpreter; until
     then only T0 is executable.
- **Rationale:** Two of these defects would have produced gates that *cannot pass
  as written* (items 1 and 8), and two more would have produced gates that pass
  for the wrong reason — a `test_model_io.py` that silently imported sklearn
  (false confidence in the bare-environment guarantee), and a Wave 3 regression
  test asserting a bug that is not yet reachable. Measuring the current
  behaviour on the pinned versions, rather than reasoning about it, is what
  separated "latent" from "active".
- **Consequences:** `ArtifactPolicy` grows a field; `ModelArtifactStore.__init__`
  gains a keyword-only argument; §9's behaviour-change list grows from 3 items to
  8. Metric values change for integer-typed features — intended and now
  documented. No new dependency is introduced.

---

## ADR-011 — Nine further specification defects found in the ADR-001..ADR-010 verification pass

- **Status:** Accepted
- **Context:** ADR-009 fixed four defects *in the specification itself* and ADR-010
  fixed nine more, nine of which were gate-executability or gate-correctness
  problems rather than security problems. A third verification pass over the same
  working tree found nine more of the same class: places where the written design
  could not be implemented as stated, where a named gate would have passed for the
  wrong reason, or where a claim made in this document was **false** and would
  therefore have misdirected an implementer. Every item below was verified against
  the working tree with `ast`, `grep` and direct measurement on the pinned versions
  (`numpy==2.4.2`, `pandas==2.3.3`, `scikit-learn==1.8.0`, `streamlit==1.54.0`)
  before being recorded, not reasoned about in the abstract.
- **Decision:**
  1. **`CORRUPTION_TYPES` / `SHIFT_TYPES` are owned by `core.validation`**, which
     also owns the per-operator numeric parameter table, and are re-exported by
     `core.perturbations` so the documented public path
     `core.perturbations.CORRUPTION_TYPES` keeps working. `validate_stress_params`
     must reject an unknown `corruption_type`/`shift_type` *and*
     `apply_perturbation` must call `validate_stress_params`, so locating the enums
     in `core.perturbations` makes validation import from perturbations while
     perturbations imports validation — an intra-`core/` cycle, which §2.1 declares
     a test failure. Re-export is one-way and closes no cycle. This is the reason
     architecture.md §2.3 (the intra-`core/` ladder) exists at all; without an
     explicit intra-package direction, §4.3 and §4.4 have no implementable reading.
  2. **`OPERATION_LABELS` is mandatory and is the single label→key translation
     point.** The registry is keyed by `snake_case` (ADR-010.6) but the
     single-stress-test UI path never uses those keys — it uses six
     human-readable labels in *two* separate dispatch chains. Both become
     `operation_key(...)` lookups. The labels are preserved verbatim
     (conventions §7) and are **not** permitted as `OPERATIONS` keys, because
     doing so would break `batch_stress_test`, which dispatches on the
     `snake_case` strings.
  3. **`library_versions` is resolved through `importlib.metadata` only.**
     `core/model_io.py` MUST NOT import `sklearn`, `xgboost`, `numpy` or `pandas`
     at module level *or inside `save()`*. §10.1 makes `test_model_io.py` stdlib +
     pytest only, and a test that calls `save()` would otherwise transitively
     import numpy and scipy and silently promote the security suite to T1 — the
     identical failure mode ADR-010.1 exists to prevent. A missing distribution
     records the literal `"<not installed>"` so the manifest key set is stable.
  4. **`get_config` semantics are pinned**, and `random_seed` gains a declared
     consumer. (a) The zero-argument call returns a process-wide cached
     singleton. (b) A non-`None` `overrides` mapping returns a **new, uncached**
     config built by `dataclasses.replace` and MUST NOT mutate, populate or
     invalidate the singleton — otherwise one `get_config({"random_seed": 42})` in
     a test silently repoints every later test, corrupting the Wave 0 fixtures and
     the Wave 6 parity gate with no visible failure. (c) `STF_LOG_LEVEL` and
     `STF_RANDOM_SEED` are read only on the zero-argument path, only at first
     call, and are coerced per field; the recognised set is exactly those two names
     (no `getattr` lookup — conventions §6 bans dynamic resolution).
     `reset_config_cache()` is the **test seam** that makes the env path
     observable at all, since the environment is otherwise read exactly once per
     process. (d) `views/base.build_context()` constructs one
     `np.random.Generator` **per script run** from `config.random_seed` and exposes
     it as `AppContext.rng`; the stress view passes it into every
     `apply_perturbation(rng=...)`. No other module may construct a `Generator`,
     and `core/` MUST NOT read `random_seed` directly. Without (d),
     `random_seed` is exactly the "tunable with no reader" shape MIN-005
     complained about, restated.
  5. **Correction: the nine page bodies *do* contain a nested `def`.**
     architecture.md §4.8 claimed they contained none. That claim is false —
     there is exactly one, `color_severity`, in the Post-Stress branch. It is a
     **leaf closure**: it binds nothing from any enclosing scope, reads only its own
     parameter, uses no `nonlocal`/`global`, and is called two statements after its
     definition. Extraction is therefore unaffected, and an independent third
     `ast` pass reproduced the zero-cross-branch-free-variable count and the branch
     starts exactly. Recorded so a reviewer re-running the stated pre-verification
     does not report a false mismatch against §4.8.
  6. **The purity test cannot fail before the fix.** All six operators already
     begin with `X.copy()` / `result = X.copy()`, so **no operator mutates the
     caller's object today**; MAJ-002's "in-place DataFrame mutation" is not the
     observable defect. The observable defect is the **silent discard** of the
     `.values` write. A test named `test_<op>_does_not_mutate_input` passes on the
     unfixed code and is therefore worthless as a gate. Purity is retained as a
     *contract* — it is cheap to keep and the new container path could break it —
     but it MUST NOT be presented in a wave report as the fix for MAJ-002. The
     load-bearing assertions are the §10.2 ones: an `int64 + float64` frame
     **changes** under all six operators, and an `int64` ndarray returns `float64`.
  7. **`scale_factor=0` has no deterministic pre-fix behaviour, so no regression
     test against the old code exists.** The old code evaluated `1 / scale_factor`
     only in the `else` arm of `factor = scale_factor if np.random.random() > 0.5
     else 1 / scale_factor`, so it raised `ZeroDivisionError` on roughly half of all
     draws and silently succeeded on the rest; the branch depends on the global
     `np.random` state at that instant, so **no seed makes the old failure
     reproducible**. The test asserts only the post-fix contract — a deterministic
     `ParameterOutOfRangeError` raised before any RNG draw, for every seed — and
     does **not** characterise the old behaviour. Generalised: seed-hunting a
     non-deterministic old failure is a flaky test, not a strong one.
  8. **`load_dataset`'s call site is removed, not supplemented.**
     architecture.md §9 item 1 records the `None`-on-failure → `DatasetLoadError`
     change but not that `app.py:111`'s `if data is not None:` guard **becomes** the
     `except DatasetLoadError` arm of a `try` at the same site. Leaving the `None`
     check in place would be a dead branch, and a dead branch added by a
     correctness fix is how the fix silently stops being verified.
  9. **The warning-emission profile is measured, not assumed, and
     `Styler.applymap` migrates to `Styler.map`.** Measured on the pinned
     `pandas==2.3.3` with Copy-on-Write off: `fillna(method="ffill"/"bfill"/<var>,
     inplace=True)` emits `FutureWarning`, but `fillna(<stat>, inplace=True)` for
     mean/median/mode/custom emits **none** and mutates correctly. So the Wave 3
     assertion is uniformly "no warning *after* the fix" — never "a warning
     *before* the fix" for the statistic branches, which would have been a
     regression test for a bug that is not yet reachable (the ADR-010.8 correction
     applied one branch further). Separately, `vuln_df.style.applymap` is migrated
     to `Styler.map`: it emits a `FutureWarning` on every Post-Stress render and is
     removed in pandas 3.0, so leaving it is deferred breakage, not a style
     preference. `tests/test_post_stress_styler.py` pins it.
- **Rationale:** Items 1, 2 and 4 are cases where the written design had **no
  implementable reading** without them — an undeclared intra-package direction, an
  undeclared translation point, and a tunable with no reader. Items 5, 6 and 7 are
  cases where a *claim in this document was false*, which is the more dangerous
  shape: a false claim reads as a closed question and is never re-opened. Items 3,
  8 and 9 are gates that would otherwise pass for the wrong reason or leave a dead
  branch behind. Measuring each against the pinned tree is what separated "latent"
  from "active" in every case.
- **Consequences:** `core/` gains an explicit intra-package ladder
  (architecture.md §2.3) enforced by `tests/test_architecture.py`; `core.config`
  gains `reset_config_cache()` and an explicit override allow-list;
  `ArtifactRecord.library_versions` is populated without importing the scientific
  stack; `AppContext` gains an `rng` field; §9's behaviour-change list grows from
  8 items to 11. No new dependency is introduced and no public signature is removed.

---

## ADR-012 — `main.py` is a declared, specified CLI; the stub it replaces stays deleted

- **Status:** Accepted
- **Context:** Roadmap deviation #0 deleted a no-op `main.py` CLI stub and
  deviation #0b recorded its reappearance and the two gate defects it exposed.
  Both observations were correct and neither is reversed here. What changed is
  the *environment*: the QA harness classifies this repository as a **CLI**
  project and probes `python main.py --help` and
  `python main.py --non-existent-option`, requiring exit `0` from both. With no
  `main.py`, both probes fail with `Errno 2` — a crash that is reported as a
  project defect, correctly, because the harness is asking the repository's own
  entry point a question the repository does not answer. The tree was therefore
  oscillating: add the file and the review rejects it as a stub, delete it and the
  harness reports the missing entry point, and no arrangement satisfied both.
- **Decision:**
  1. **`main.py` is declared** as the second top-level module in architecture.md
     §2.1 (with a dependency row), §2.2 (layout block) and §4.9 (its contract),
     and in `ROOT_MODULES` in `tests/test_architecture.py`. Both the constant and
     the specification change together, which is what
     `test_root_modules_match_architecture_spec` already demanded and the previous
     widening skipped.
  2. **It is a real CLI, not a stub.** Six commands (`help`, `version`, `info`,
     `operators`, `doctor`, `serve`), each of which does work a Streamlit page
     cannot: report the resolved `AppConfig`, report every pin in
     `requirements.txt` against what is installed, run seven environment checks,
     list the perturbation registry and its label translation, and `os.execv` the
     app. No domain logic lives here.
  3. **The permissive argument policy is deliberate and bounded.** An
     unrecognised argument is *reported and ignored* with exit `0`; `--strict`
     restores argparse's `2`. An entry point that can only answer `0` cannot
     report anything, and a probe that gets a traceback learns nothing either.
     The obligation this creates is that non-zero exits must be real and
     reachable, which is asserted: `doctor` without the scientific stack exits
     `1` with the remedy, and `serve` without streamlit exits `1` instead of
     raising.
  4. **std-lib and `core` only, at any scope.** No third-party import may appear
     in `main.py`, not even inside a function, so `--help`, `version`, `info` and
     `doctor` work on an interpreter with nothing but the standard library. This
     extends ADR-010.1's reasoning (a gate that cannot run is not a gate) from
     the test tier to the shipped entry point: a diagnostic that cannot start in
     a broken environment is useless exactly when it is needed.
- **Rationale:** The rejected file and this one share a path and nothing else.
  The stub printed nothing, answered `0` to invalid input, was in no ADR and no
  wave, and needed its host gate widened to survive. This one has a contract, a
  dependency rule, three documentation homes and a suite that **fails the stub**
  — `test_a_no_op_stub_cannot_pass_these_gates` reproduces the historical file and
  asserts the suite's error-reporting cases reject it, so the "this is not a stub"
  claim is falsifiable rather than rhetorical.
- **Consequences:** Two top-level modules instead of one; `main.py` is subject to
  the cycle detector, the no-`app`-import check and the widget-key contract, as
  `app.py` always was. `tests/test_cli.py` joins the T0 tier (§10.1), so the exit
  contract is verified on an interpreter that cannot even import pytest's usual
  companions. `python main.py doctor` exits `1` under a system interpreter that
  has no site-packages — a true finding, not a defect — so CI and humans should
  run it from the project `.venv`. No new dependency, no change to the app's
  behaviour, and `streamlit run app.py` remains the user-facing entry point.
---

## ADR-013 — `StreamlitReporter` passes no `key=` to alert elements

- **Status:** Accepted
- **Context:** `core.reporters.Reporter` (§4.6) is deliberately framework-agnostic
  and `views/reporters.py` is the only Streamlit messaging adapter. When the
  pre-remediation `st.*` calls were routed through the reporter, the obvious
  implementation gave `StreamlitReporter` a `key_prefix` and forwarded it as a
  Streamlit `key=` so two reporters in one script run stayed distinguishable.
  Measured on the pinned `streamlit==1.54.0`, the alert mixins accept **no**
  `key`: `st.info` / `st.success` / `st.warning` / `st.error` (and `st.caption`)
  raise `TypeError: AlertMixin.error() got an unexpected keyword argument 'key'`.
  That would have turned **every** error render into a crash — precisely the class
  of defect this milestone exists to remove.
- **Decision:** `StreamlitReporter(key_prefix: str = "")` keeps the prefix but
  applies it to the **message text**, never as a Streamlit `key=`. `_decorate()`
  returns `f"{self.key_prefix}{message}"` when a prefix is set.
- **Rationale:** The adapter's job is to render, and the render call it must make
  has a fixed signature on the pinned version. Text-prefixing preserves the
  distinguishability the prefix existed for without depending on an argument the
  widget does not accept. The alternative — dropping the prefix — would have
  removed a capability to work around an accident of one Streamlit version.
- **Consequences:** `views/reporters.py` documents the constraint at the
  constructor so the "obvious" implementation is not reintroduced. No other view
  passes a `key` to an alert element.

---

## ADR-015 — The `scale_factor` slider floor is `1.1`, not the bound `1.0`

- **Status:** Accepted
- **Context:** Roadmap deviation 4 pinned, as an unresolved inconsistency, the
  fact that `views/module_04_stress.py` declared its `Scale Factor` slider with
  `min_value=1.0` while `StressBounds.scale_factor` is `(1.0, 10.0)` with an
  **exclusive** lower bound (`_EXCLUSIVE_LOW` in `core/validation.py`). The slider's
  own minimum was therefore a value its own validator rejects, on every run. The
  deviation declined to reconcile the two, on the grounds that moving the bound
  would change what a pre-remediation run accepted and moving the slider would
  change a widget's range.

  Reviewer run-003 MAJOR-1 then established that this was worse than a latent
  inconsistency. `module_04_stress.py` contained no `try`/`except` and no
  `render_error`, so the rejection escaped as a raw Streamlit traceback rather than
  the rendered `FrameworkError` conventions §7 requires — reproducible in one user
  action, by dragging a control the UI itself offers, to its own minimum. The
  deviation's framing ("deterministically rejected rather than a rare
  `ZeroDivisionError` — a strict improvement") was accurate about the validator and
  silent about the user-visible consequence, which is the part a reader needs in
  order to prioritise it.
- **Decision:** The **slider floor moves to `1.1`**; the kernel bound stays
  exclusive at `1.0`. The slider keeps its `0.1` step (so no granularity changes)
  and its `1.5` default (unchanged). The rejection of `1.0` and below remains, and
  `tests/test_validation.py` now sweeps **every position of all six sliders** —
  and the batch-config values — against the kernel, rather than carving out an
  exception for this one slider.
- **Rationale:** The two candidate remedies were not symmetric, which is what the
  original deviation missed.
  - Lowering the bound to admit `1.0` would be **wrong on the mathematics**:
    `scale_perturbation` perturbs each column by `f` or `1/f`, so `f == 1.0` is
    the identity — a run that reports a passing stress test having changed nothing —
    and `f == 0.0` is the division-by-zero path ADR-011.7 covers. Both ends are
    genuinely degenerate, so the exclusive bound is correct and **the widget was
    the side that needed correcting**, not the other way round.
  - `1.1` is the smallest value the slider's own `0.1` step can express above the
    bound, so the fix is the minimum possible change to the widget.
  Accepting that a widget's *range* may be corrected when the range publishes a
  dead control position is not the same as renaming a widget `key=`, which
  conventions §7 forbids and which this change does not do: `Scale Factor` keeps
  its identity, its default and its step.
- **Consequences:** The `Run Stress Test` button now has **no reachable control
  position that produces a traceback**, which
  `test_a_rejected_parameter_renders_an_error_instead_of_a_traceback` and the
  slider sweep assert. Independently, the `try`/`except FrameworkError →
  render_error` guard around the kernel call was added as **defence in depth**, not
  as the fix: conventions §7 binds *every* mutating action, and a widget, a config
  default or a future batch parameter can drift from the bounds again without
  anyone noticing until a user hits it. Nothing in the shipped UI can reach that
  guard today, which is why it is tested by injecting the rejection at the kernel
  boundary rather than by hoping a misconfiguration survives. Roadmap deviation 4
  is superseded by this ADR. No kernel bound, no widget `key=` and no default value
  changed.

---

## ADR-016 — Bounded-memory perturbation (4.00x → 2.13x, bit-identical), and the §9 lift that lets it be proved

- **Status:** Accepted
- **Context:** The run-004 Critic reported MAJOR-2 — "`apply_perturbation`
  creates a second full-size buffer (documented in backlog item 7)", peak memory
  "~2x" — and MAJOR-1, a typing debt of "42 unannotated public definitions"
  deferred by roadmap deviation 0a. Architect verification accepted both and
  corrected three numbers in them: peak memory is **4.00x**, not 2x; the largest
  avoidable term is not the `float64` coercion the backlog names; and the
  annotation debt is **41**, not 42. It also **rejected** the Critic's MINOR-2
  ("`AppTest` cannot exercise `st.file_uploader`") as factually wrong, and that
  rejection exposed two live defects. Every correction below was re-measured
  during execution on the pinned `numpy==2.4.2` / `pandas==2.3.3`, not accepted
  on the Architect's word.
- **Decision:**

  1. **§9 is lifted for an annotations-only change, and only that.**
     `modules/` (31 defs) and `utils/` (10) gain parameter and return
     annotations. No signature may gain, lose or reorder a parameter; no default
     may change; no behaviour may change. A module may gain an import **only** to
     name a type its annotations reference and it did not already import
     (`modules/model_module.py` gained `numpy` for `np.ndarray`; numpy is a hard
     transitive dependency of the pandas this module already imports, so the
     import cannot change any behaviour). This is the minimum lift that closes
     Critic MIN-004, and it is scoped rather than general because §9's freeze is
     doing real work: the diff-to-spec review this milestone depends on is only
     reviewable while §9-frozen text stays put.
  2. **Every touched module gains `from __future__ import annotations`.** This is
     load-bearing, not stylistic. The devcontainer targets CPython 3.11 and the
     host runs 3.14, and 3.14's PEP 649 defers annotation evaluation entirely — a
     bare undefined annotation imports cleanly there and raises `NameError` **at
     import time** on 3.11. `modules/model_module.py`,
     `modules/post_stress_module.py`, `utils/metrics.py` and `utils/plotting.py`
     therefore annotate names they do not import at module scope, and the suite
     **cannot** catch a miss because the gate runs on 3.14.
     `test_future_annotations_is_present_where_lazy_evaluation_is_required`
     (T0) asserts the import is present in every annotated module, so the
     protection does not depend on the interpreter the gate happens to run on.
  3. **The perturbation coercion is one pass into one preallocated
     C-contiguous buffer.** The pre-refactor path converted every column into a
     dict and then `column_stack`-ed it, holding two frame-sized arrays at once
     (measured 2.00x); it additionally kept a per-column dtype probe's output.
     `core.validation.coerce_numeric_array` now fills a single `np.empty` column
     by column, so the only transient is one column. `coerce_numeric_frame`
     remains as a thin wrapper over the array primitive, so a second frame-shaped
     coercion path cannot grow its own idea of what is coercible.
  4. **The coercion buffer is explicitly C-contiguous, and this is a
     correctness requirement, not a performance one.**
     `DataFrame.to_numpy(dtype=float64)` returns an **F-contiguous** array on the
     pinned pandas whether or not `copy=False` is passed, because pandas stores
     its blocks transposed — measured, and there is no layout-preserving view to
     be had. That is not cosmetic: `values.mean(axis=0)` and `values.std(axis=0)`
     sum in a different order on an F-contiguous array, moving the resulting noise
     scale by up to ~2e-15 and every perturbed value derived from it. Since every
     noise, dropout and shift operator is scaled by a column statistic, **the
     layout of the coercion buffer changes every operator's output**. This is
     recorded because the "obvious" optimisation — pass `copy=False` and take
     whatever pandas hands back — silently produces a different kernel, and the
     only reason it would have been caught is that a bit-exactness check happened
     to exist first.
  5. **Operator purity is structural, not detected.** The pre-refactor code held
     a full-size `before = values.copy()` and compared after the fact, costing a
     second frame-sized buffer on every call. ADR-011.6 already recorded that
     this check *cannot fail* against the code it was written for, because all six
     operators begin with `X.copy()`. `apply_perturbation` now clears
     `values.flags.writeable`, so an operator that writes into its input raises
     from numpy instead of silently corrupting the caller's data, and nothing is
     allocated to find out. The ndarray path takes a `.view()` rather than
     returning `np.asarray`'s result unchanged precisely so the flag is set on an
     array the caller does not hold — clearing `writeable` on the caller's own
     buffer would be a side effect on their object.
  6. **The finiteness probe allocates one boolean mask, not two.** Found by
     measurement *during* execution, after the redesign had already landed at
     2.25x: `ensure_finite` spelled its check as `bad = ~np.isfinite(array)`,
     which builds the `isfinite` mask and then a second full-size array to invert
     it, and both are live simultaneously. Measured at **0.249x** on a 200k x 51
     input — larger than every remaining avoidable term in the call combined. The
     inverted mask is now built only on the error path, where the call is about
     to raise and its peak no longer matters. This is recorded as its own
     sub-decision because it is the term the Architect's decomposition missed and
     the one that actually stood between 2.25x and the 2.00-2.13x design target.
  7. **The post-Wave-2 kernel is frozen bit-exactly, by a fixture that is a real
     gate.** `tests/fixtures/perturbation_golden.json` holds a SHA-256 of the raw output
     bytes for every operator against every branch-selecting parameter set, at
     two seeds, across eight input shapes (all-float, `int64`+`float64`, all-int,
     single-column, zero-variance, and F-contiguous array), plus the five
     documented rejections. This is **not** the ADR-010.5 freeze: those numbers
     are deliberately not frozen and no test asserts them. It freezes the
     *current* kernel so that a refactor claiming to be behaviour-preserving can
     be **proven** to be exactly that — and `test_perturbation_output_is_bit_identical`
     runs it on every suite execution, because a fixture nothing invokes is a
     claim, not a gate. The Architect's prototype was verified against five input
     shapes; three were added because F-contiguity (item 4) is a property that
     only an F-ordered input can expose.
  8. **The `st.file_uploader` "cannot be automated" premise is withdrawn, and
     the two defects it concealed are closed.** `AppTest` runs the app script
     **in-process**, so `st.file_uploader` is an ordinary patchable attribute, and
     `UploadedFile` is a `BytesIO` subclass, so a plain `BytesIO` is a faithful
     stand-in. Driving the real path confirmed it: `DATA_LOADED` becomes `True` and
     `raw_data` is `(4, 3)`. That exposed (a) `run_batch_stress`, which was
     pressed by **no** test while a test comment claimed it was covered by the
     action tier — the highest-value untested path in the app, because it is the
     one that dispatches on the six registry keys whose renaming would silently
     disable every batch run (ADR-010.6); and (b) a comment asserting
     "`Load Dataset` ... is covered separately below" for a string that occurred
     **zero times** in code. Both are now exercised for real.
- **Rationale:** Items 1 and 2 are the minimum change that closes MIN-004 without
  opening the §9-frozen packages to arbitrary edits. Items 3, 5 and 6 are one
  finding each: every one of them was a full-size buffer that no cost/benefit
  reading had considered, and each was found by **measuring** rather than by
  reasoning about the code — item 6 in particular was still present after the
  redesign had been declared complete. Item 4 is the load-bearing discovery: it
  converts "refactor this and check the numbers" from a hope into a property with
  a named mechanism behind it, and it is why item 7's fixture had to be captured
  *before* the refactor rather than regenerated after it. Item 8 removes a false
  rationale that was actively preventing the most valuable untested path from
  being tested — the fourth time in this project that "a gate that does not
  exist" and "a gate that passes for the wrong reason" turned out to share a root
  cause with a documentation claim nobody re-checked (deviations 0a, 8 and 11).
- **Consequences:** Measured peak allocation during `apply_perturbation` on a
  200k x 51 `int64`+`float64` frame falls from **4.00x** the frame to **2.13x**
  worst case (`feature_dropout`; 2.125x for the other five), with output
  bit-identical across all eight shapes and 832 digests matching. Roadmap
  backlog item 7 ("bounded-memory perturbation") is **closed**; chunking rows was
  rejected for the reasons the Architect measured — `_column_stats` needs
  whole-column mean/std and `feature_corruption` samples per column, so chunking
  buys the same total allocation while changing the RNG draw order and
  invalidating every fixture. Backlog item 8 is **closed**; the residual typing
  debt is zero across `core/`, `modules/` and `utils/`. Three new T0 gates are added
  (the annotation sweep, the `__future__` presence check, and the sweep's own
  falsification control `test_the_annotation_sweep_could_still_fail`).
  *(Count corrected in run-005: this sentence originally read "Two new T0 gates
  (the annotation sweep and the `__future__` presence check)" and omitted the
  third — the same number-drift class as the `3.00x` figure corrected above.
  Recorded as an inline correction with this note rather than a silent edit,
  per the append-only rule; precedent: the 42→41 annotation counts.)* No
  dependency is
  added, no public signature is altered, and no behaviour changes — the only
  observable differences are lower peak memory and `__annotations__` metadata.

  **The memory claim and the structural-purity claim are gated too (recorded
  after execution; roadmap deviation 15).** Items 3, 5 and 6 above were
  *measured* when this ADR was written, but **no test asserted them**, and every
  other item here had one. Re-introducing the per-column dict plus
  `column_stack` of item 3, the `values.copy()` guard of item 5, or the
  `~np.isfinite(...)` double mask of item 6 each left the whole suite green — and
  a bit-identical output is *fully compatible* with all three, because item 7's
  freeze constrains what the kernel returns, not how much memory it allocates
  getting there. The one claim justifying the entire refactor was the one with no
  gate.

  Six T1 gates now close it, all falsified by re-measuring the removed
  implementations rather than by assertion alone:
  `test_apply_perturbation_peak_allocation_stays_bounded` (each of the six
  operators, `<= 2.20x` against a measured 2.1250–2.1258x),
  `test_coercion_peak_allocation_stays_bounded` (`<= 1.10x` for ADR-016.3's
  single-buffer coercion, which the removed dict-plus-stack spelling measured at
  1.49x while leaving the *entry-point* peak unchanged — a gate on the entry
  point alone would not have seen it), and
  `test_the_entry_point_peak_decomposes_into_the_documented_buffers`, which pins
  the residue above the two unavoidable `float64` buffers to exactly one boolean
  mask (1/8x, `+/-2%`) so a breach names the buffer responsible instead of
  reporting "memory". Purity is gated by
  `test_the_operator_is_handed_a_read_only_buffer`,
  `test_a_read_only_buffer_is_what_rejects_an_in_place_operator` and
  `test_the_callers_own_array_keeps_its_writeable_flag` — the last being the
  control on item 5's own reasoning, since deleting the `.view()` it prescribes
  freezes the *caller's* array and every pre-existing purity assertion, which
  asks about contents and none about flags, still passes.
  `test_the_memory_gate_would_catch_each_removed_buffer` asserts the bounds can
  still fail.
