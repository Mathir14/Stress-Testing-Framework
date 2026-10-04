# Project Conventions

Binding rules for this repository. Enforcement points are named where a rule is
machine-checkable (`tests/test_architecture.py`).

## 1. Layout & Layering

- Layer order and the import matrix are defined in `architecture.md` §2.1 and are
  test-enforced.
- `core/` = framework-agnostic domain kernel (no `streamlit`, no `plotly`).
- `views/` = presentation only. No domain computation, no metric formulas.
- `app.py` = composition root, target ≤ 200 lines: page config, logging,
  `init_state`, sidebar, dispatch, status footer. No page bodies.
- `modules/` = domain services. Public API frozen (see `architecture.md` §9).
- `utils/` = stateless helpers; must not depend on `modules` or `core`.
- Every package directory has an `__init__.py` (empty is fine).
- Module filenames use `snake_case`; classes `PascalCase`; functions and
  variables `snake_case`; module-level constants `UPPER_SNAKE_CASE`.

## 2. Typing

- Public functions and methods MUST carry type hints for parameters and return
  values. `app.py` page modules annotate their single `ctx: AppContext`
  parameter; procedural Streamlit bodies are exempt.
- Domain containers are typed (`Mapping[str, Any]`, `np.ndarray`,
  `pd.DataFrame`); avoid bare `dict`/`list` in new signatures.
- Use `from __future__ import annotations` where PEP 604 unions are used in
  signatures evaluated at runtime (mirror `modules/reliability_module.py:12`).

## 3. Errors

- Raise typed errors from `core.errors`; never return `None` to signal failure.
- Never `except:` bare, never `except Exception: pass`. Catching
  `Exception` requires a `logger.exception(...)` or re-raise.
- Error messages are user-actionable sentences naming the offending field and the
  accepted range. Machine context goes in `err.context`, never in the message.
- Views own all `st.error`/`st.warning` rendering.

## 4. Logging

- `logger = logging.getLogger(__name__)` per module; no `print()` outside
  `views/`.
- Level policy: domain `warning`/`exception` for recoverable degradations,
  `info` for lifecycle events (model saved, stress test completed), `debug` for
  per-column perturbation diagnostics.
- Logging is configured once in `app.py` via `core.logging_config.configure_logging`.

## 5. Purity & Data

- Functions in `core/` MUST NOT mutate their arguments. DataFrame-returning
  helpers return new frames; document the copy strategy explicitly
  (`deep=True` when followed by mutation).
- Chained assignment with `inplace=True` is banned:
  `df[col].fillna(v, inplace=True)` → `df[col] = df[col].fillna(v)`;
  `df[col].fillna(method="ffill")` → `df[col] = df[col].ffill()`.
- Randomness is injected: functions accept `rng: np.random.Generator` (or
  `random_state: int`) and MUST NOT call the global `np.random`/`random` state.
- All tunables come from `core.config`; magic numbers in domain code require an
  inline justification comment.
- Values interpolated into HTML MUST pass through `html.escape(str(v), quote=True)`.

## 6. Security

- No `eval`, `exec`, `shell=True`, or dynamic imports.
- Filesystem access goes through `core.model_io.ModelArtifactStore` (or a future
  `core.paths` helper): validate the name, resolve, assert containment, refuse
  symlinks. No raw `os.path.join` with user input.
- Deserialization is a trust boundary: attestation + explicit confirmation, per
  ADR-001. Never load a model artifact without `ModelArtifactStore`.

## 7. UI Conventions

- Emoji-prefixed headings/labels are the established voice — preserve existing
  strings verbatim when extracting code into `views/` (including the nine sidebar
  radio labels).
- Widget `key=` literals are part of the persisted session contract; never rename
  or drop them during extraction.
- `use_container_width=True` on `st.dataframe`/`st.plotly_chart` calls.
- Long or blocking work runs inside `st.spinner(...)`.
- Every mutating action reports success **or** a rendered `FrameworkError`.

## 8. Docstrings

- Google style, matching the existing modules: summary line, blank line,
  `Args:`, `Returns:`, `Raises:` (the latter mandatory for new public functions
  that raise).
- Every non-obvious numeric formula states the formula in plain text
  (see `modules/reliability_module.py:59-77`).

## 9. Dependencies

- `requirements.txt` is a fully pinned lockfile; do not restructure it in this
  milestone (ADR-007). Test tooling goes in `requirements-dev.txt`.
- New runtime dependencies require an ADR justifying the need and the removal of
  an alternative.
- Prefer stdlib: `hashlib`, `json`, `pathlib`, `html`, `logging`, `dataclasses`.

## 10. Testing

- `tests/` mirrors source layout: `tests/test_<module>.py` for
  `core/<module>.py`.
- Test names state the invariant: `test_gaussian_noise_does_not_mutate_input`.
- Every bug fixed in this milestone gets a regression test named after the defect.
- Security tests use `tmp_path`; never touch the real `saved_models/`.
- Tests MUST be deterministic (seed every RNG; no wall-clock or network).
