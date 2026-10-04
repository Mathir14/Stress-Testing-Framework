# Project Roadmap

Milestone: **run-003 remediation** (remediation of the run-002 Critic audit).
Each wave lists its gate. A wave is not complete until its gate passes.

## Non-negotiable ordering

Waves are sequential because later waves depend on earlier seams. Waves MUST NOT
be merged or reordered. Each wave is an independent commit.

---

## Wave 0 — Safety net (blocks everything else)

- `requirements-dev.txt` (pytest pinned), `tests/conftest.py`,
  `tests/test_architecture.py` (layer matrix, no streamlit in core/modules/utils).
- Baseline snapshots: record current `ReliabilityScorer` outputs for fixed inputs
  as fixtures for parity testing.
- **Gate:** `pytest tests/test_architecture.py` green (runs with stdlib+pytest
  only). Baseline fixtures recorded.
- **Addresses:** MIN-006, MAJ-001/MAJ-006 (partially), and makes every later
  wave verifiable.

## Wave 1 — Security boundary (CRITICAL)

- `core/errors.py`, `core/config.py` (incl. `ArtifactPolicy`).
- `core/model_io.py`: `ModelArtifactStore`, `ArtifactRecord`, manifest v1,
  filename sanitisation, containment, atomic writes, hash verification, trust
  gate, estimator type gate.
- `modules/model_module.py`: `save_model`/`load_model` delegate to the store,
  lose `streamlit` import, gain `trust: bool = False`.
- `views/module_02_baseline.py` (or the current `app.py` save/load tab, if Wave 5
  has not landed): explicit unchecked "this file executes Python code"
  confirmation, plus distinct error rendering for `Artifact*` errors.
- `modules/reporting_module.py`: `html.escape` on every HTML interpolation.
- `tests/test_model_io.py`, `tests/test_reporting_escaping.py`.
- **Gate:** traversal/tamper/untrusted tests green; no `pickle.load` outside
  `core/model_io.py` (grep assertion); save-side traversal test present.

## Wave 2 — Pure perturbation kernel (CRITICAL robustness)

- `core/perturbations.py` (`OPERATIONS`, `apply_perturbation`, zero-variance
  guard, RNG injection).
- `core/validation.py` (`validate_stress_params`, `ensure_finite`,
  `validate_frame`) with bounds from `core/config.py`.
- `modules/stress_module.py`: six operators delegate to
  `apply_perturbation`; `batch_stress_test` rejects unknown stress types with
  `UnsupportedStressTypeError` instead of silently `continue`-ing.
- `tests/test_perturbations.py`, `tests/test_validation.py`.
- **Gate:** purity tests green; mixed-dtype DataFrame regression test green;
  out-of-range and `scale_factor=0`/`dropout_rate=1.0` rejected; seeded runs
  reproducible.

## Wave 3 — Data correctness & immutability

- `modules/data_module.py`: replace chained `inplace=True` + deprecated
  `method=` fills; explicit `deep=True` copies; rename the ambiguous
  `handle_missing_values` custom-value branch parameter; guard
  `get_split_summary` against zero-sample splits.
- `tests/test_data_preprocessing.py`.
- **Gate:** missing-value strategies demonstrably change the frame (regression
  test for the current silent no-op); no `fillna(method=` and no
  `inplace=True` remaining in `modules/` (grep assertion).

## Wave 4 — Error & logging policy, Streamlit decoupling

- `core/reporters.py` (Protocol, `NullReporter`, `CollectingReporter`),
  `core/logging_config.py`, `views/reporters.py` (`StreamlitReporter`).
- `modules/data_module.py`: `load_dataset` raises `DatasetLoadError`; drop
  `display_summary` body into `views/module_01_data.py` (keep the method as a thin
  delegating wrapper for API compatibility, or remove it and update the single
  call site — either is acceptable, document the choice).
- `modules/calibration_module.py:104-107`: log the Brier degradation.
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
- **Gate:** `app.py` ≤ 200 lines; sidebar labels unchanged; manual smoke test of
  all nine pages (data upload → train → stress → post-stress → calibration →
  compare → score → report → export) recorded in the wave report.

## Wave 6 — Configuration, dependency hygiene, documentation

- `ReliabilityScorer` + stress bounds read from `core/config.py`; defaults
  unchanged.
- `tests/test_reliability_parity.py` freezes pre-refactor scores/grades.
- Remove `shap==0.50.0` after a reverse-dependency check (ADR-007).
- Populate `.ai/project/*.md` (done in this architecture stage; update for any
  deviation), update `README.md` security note and project structure.
- **Gate:** parity tests green; `README.md` matches the final tree; no ADR
  violated.

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
