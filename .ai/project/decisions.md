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

---

## ADR-008 — Documentation stubs become binding project records

- **Status:** Accepted
- **Context:** MIN-001 — `.ai/project/*.md` were placeholders.
- **Decision:** `architecture.md` (this remediation's target design),
  `conventions.md` (coding rules), `decisions.md` (this file) and `roadmap.md`
  (sequenced waves) are populated and MUST be updated when behaviour changes.
- **Rationale:** The agent workflow reads these files before every task; leaving
  them empty makes every future stage re-derive the same context.
