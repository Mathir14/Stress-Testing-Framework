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
- Views own all `st.error`/`st.warning` rendering, via `views.reporters`.
- A public kernel API MUST NOT leak a raw builtin exception. Every documented
  failure mode is a typed `FrameworkError`; if a function's docstring declares
  `ValidationError` as its only raise, then every input — including degenerate
  ones like duplicate column labels — raises it (reviewer MAJOR-2).

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
- `width="stretch"` on `st.dataframe`/`st.plotly_chart` calls — **not**
  `use_container_width=True`. Measured on the pinned `streamlit==1.54.0`:
  `use_container_width` prints *"Please replace `use_container_width` with
  `width`. `use_container_width` will be removed after 2025-12-31. For
  `use_container_width=True`, use `width='stretch'`."* The 76 pre-remediation
  `use_container_width=True` call sites were migrated to `width="stretch"` when
  the page bodies were extracted, so this rule previously contradicted every
  `st.dataframe`/`st.plotly_chart` call in the tree. `use_container_width=False`
  maps to `width="content"`.
- Long or blocking work runs inside `st.spinner(...)`.
- Every mutating action reports success **or** a rendered `FrameworkError`.
  Concretely: any `views/module_*.py` containing an `if st.button(…)` handler
  MUST import `render_error` / `render_errors` from `views.reporters`, and each
  such handler body MUST apply one. Wrap the **whole** handler with
  `with render_errors():` rather than hand-copying a `try` per call site — a
  copied `try` is easy to forget at the next button, and the context manager
  additionally guarantees a rejected call cannot fall through into display code
  reading a variable it was going to bind. Catching a specific
  `FrameworkError` subclass is equally acceptable. Enforced by
  `tests/test_architecture.py`; the defect class is reviewer MAJOR-1.
- A UI control MUST NOT offer a value the kernel rejects. Every position of every
  slider and selectbox is validated against `core.validation`, with no carve-outs
  (ADR-015).
- **A mutating action with no pressed button is untested code**, however
  thoroughly its page renders. `tests/test_app_smoke.py` presses every one; this
  is reviewer MINOR-1's lesson, and the false rationale that once excused the gap
  ("`AppTest` cannot reach buttons") must not be reintroduced. **Press coverage
  is enforced, not asserted:**
  `test_every_mutating_button_in_the_views_is_pressed_here`
  (`tests/test_architecture.py`, tier T0) derives every `st.button` site in
  `views/` *and* every press declared in `tests/test_app_smoke.py` from the
  tree, and requires the two to match in both directions — an unpressed button,
  a press that names no button, and a declaration whose test body never calls
  `.click()` each fail it (run-005 M1). The runtime half, that those presses
  actually execute and report success or a rendered `FrameworkError`, stays with
  the smoke suite's action tier; neither half implies the other.

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
