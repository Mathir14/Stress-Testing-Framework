# Architecture Overview

Status: **authoritative** — this document defines the target architecture for the
remediation of the run-002 Critic audit. Executors MUST treat the interfaces,
dependency rules and parity constraints below as binding.

## 1. System Summary

ML Model Reliability & Stress Testing Framework — a local, single-user Streamlit
application for evaluating binary/multiclass classifiers across performance,
confidence, calibration, perturbation robustness, reliability scoring and report
export. Python 3.11+ (devcontainer), fully pinned dependencies
(`requirements.txt`), no network services, no database, no CI.

Current shape (verified at remediation time):

| Component | Lines | Streamlit refs | Role |
|---|---|---|---|
| `app.py` | 3517 | 854 | UI + orchestration + session state (9 pages) |
| `modules/*.py` (8 files) | 3333 | 19 | Domain services (mostly pure) |
| `utils/*.py` (2 files) | 394 | 0 | Metrics + plotting helpers |
| `tests/` | 0 | – | **absent** |

Notable correction to the Critic report: the claim that "most modules directly
import and call Streamlit" (AV-002) is **not** supported by the code. Only
`modules/data_module.py` (17 refs) and `modules/model_module.py` (2 refs) call
`st.*`. Seven of the nine domain modules make no Streamlit call at all. The
decoupling work is therefore *narrow* (2 files), not repository-wide.

Streamlit coupling has two distinct severities and both must be cleared by Wave 4:

| File | Import | `st.*` calls | Severity |
|---|---|---|---|
| `modules/data_module.py` | yes | 17 | real coupling — 17 `st.success`/`st.error`/`st.metric` calls to rewrite as `Reporter` messages |
| `modules/model_module.py` | yes | 2 | real coupling — `st.success` in `save_model`/`load_model` |
| `modules/post_stress_module.py` | **yes** | **0** | dead import only — delete the `import streamlit as st` line; no behaviour change |

`post_stress_module.py` is a *third* file that violates the §2.1 matrix
(`modules/` MUST NOT import `streamlit`) purely through an unused import. It is
trivially fixable but must be named explicitly or the Wave 4 grep gate will fail
on a file no one expected to touch.

## 2. Target Layering

```
┌──────────────────────────────────────────────────────────────┐
│ app.py  – composition root: page config, logging, state      │
│           bootstrap, sidebar nav, page dispatch, footer      │
├──────────────────────────────────────────────────────────────┤
│ views/  – presentation only. One module per Streamlit page.  │
│           StreamlitReporter lives here. No domain logic.      │
├──────────────────────────────────────────────────────────────┤
│ modules/ – domain services (existing public API preserved).  │
│ utils/   – stateless numeric/plot helpers.                    │
├──────────────────────────────────────────────────────────────┤
│ core/    – framework-agnostic domain kernel: config, errors, │
│           validation, perturbations, model_io, reporters     │
│           (protocol), state schema. No streamlit, no plotly. │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Binding dependency rules (enforced by `tests/test_architecture.py`)

| Layer | MUST NOT import | MAY import |
|---|---|---|
| `core/` | `streamlit`, `plotly`, `views`, `app`, `modules` | stdlib, `numpy`, `pandas`, `sklearn` |
| `modules/` | `streamlit`, `views`, `app` | stdlib, `numpy`, `pandas`, `sklearn`, `xgboost`, `plotly`, `utils`, `core` |
| `utils/` | `streamlit`, `views`, `app`, `modules`, `core` | stdlib, `numpy`, `pandas`, `plotly` |
| `views/` | `app` | `core`, `modules`, `utils`, `streamlit`, `plotly` |
| `app.py` | – | everything except `views` importing `app` |
| `main.py` | `app`, `views`, `modules`, `utils`, **any third-party root** | stdlib, `core` |

No module may import `app.py`. Circular imports are a test failure.

`main.py` is the command-line entry point (ADR-012, §4.9). It is held to a
rule **tighter** than the general root-module rule, and the tighter rule is the
one that matters: stdlib and `core` only, **at any scope** — no third-party
import may appear inside a function either. That is what keeps `--help`,
`version`, `info` and `doctor` runnable on an interpreter with nothing but the
standard library, which is precisely where a "your environment is broken" tool
is needed. Heavier dependencies are reached *by name* through
`importlib.util.find_spec` or in a child process, never by importing them.

### 2.2 Target directory layout

```
app.py                       # composition root, <=200 lines
main.py                      # CLI entry point: doctor/info/version/operators/serve
core/
  __init__.py
  config.py                  # AppConfig, StressBounds, ReliabilityWeights, ArtifactPolicy
  errors.py                  # FrameworkError hierarchy
  logging_config.py          # configure_logging()
  model_io.py                # ModelArtifactStore, ArtifactRecord  (security boundary)
  perturbations.py           # pure perturbation ops + registry    (CRIT-003, MAJ-002)
  reporters.py               # Reporter Protocol, NullReporter, CollectingReporter
  state.py                   # StateKeys, init_state, reset_state, AppContext
  validation.py              # validate_stress_params, ensure_finite, validate_frame
modules/                     # existing services, public API unchanged
utils/                       # metrics.py, plotting.py
views/
  __init__.py
  base.py                    # build_context, render_sidebar, render_status_footer
  reporters.py               # StreamlitReporter  (only streamlit messaging adapter)
  module_01_data.py .. module_09_reports.py   # render(ctx) -> None
tests/
  conftest.py
  test_architecture.py       # import-layer rules (stdlib + pytest only)
  test_cli.py                # main.py exit-code contract (stdlib + pytest only)
  test_perturbations.py
  test_validation.py
  test_model_io.py
  test_data_preprocessing.py
  test_reliability_parity.py
requirements-dev.txt          # pytest (+ pytest-cov), pinned
```

`__init__.py` MUST be added to `core/`, `views/`, `tests/`, `modules/`, `utils/`.
Empty content is acceptable.

§2.2 declares **exactly two** top-level modules, and both are load-bearing:
`app.py` (the Streamlit composition root) and `main.py` (the CLI, ADR-012). A
root-level module belongs to no §2.1 layer, so every one of them has to be
declared here *and* cross-checked against `ROOT_MODULES` in
`tests/test_architecture.py`, in both directions — a constant widened alone
cannot make that gate green, and a specification edited alone cannot either.
`main.py` was added only after the CLI it declares had a contract (§4.9), a
dependency rule (§2.1) and a suite that fails the no-op stub it replaced
(`tests/test_cli.py`).

### 2.3 Intra-`core/` layering (binding — added by ADR-011.1)

§2.1 constrains imports *between* layers. It does not constrain imports *within*
`core/`, and §2.1's closing sentence makes any cycle a test failure. Without an
explicit intra-package direction, the natural reading of §4.3 + §4.4 produces a
cycle that cannot be implemented. `core/` is therefore a strict ladder; each
module may import only from the rungs above it.

```
   rung 4   core.perturbations   (depends on 1,2,3)
   rung 3   core.validation      (depends on 1,2)   <- numeric table + enums live HERE
   rung 2   core.errors          (depends on 1)     imports nothing project-local
   rung 1   core.config          (depends on nothing project-local)
            (+ core.logging_config, core.model_io, core.reporters, core.state
              are siblings: they may import rungs 1-2 only, never each other
              except model_io -> config/errors)
```

| Module | MAY import inside `core/` | MUST NOT |
|---|---|---|
| `core.config` | — | any project-local module |
| `core.errors` | — | any project-local module |
| `core.validation` | `config`, `errors` | `perturbations`, `model_io`, `state`, `reporters` |
| `core.perturbations` | `config`, `errors`, `validation` | `model_io`, `state`, `reporters` |
| `core.model_io` | `config`, `errors` | `validation`, `perturbations`, `state`, `reporters` |
| `core.reporters` | — | any project-local module |
| `core.state` | `config`, `errors` | `validation`, `perturbations`, `model_io` |
| `core.logging_config` | `config` | everything else in `core/` |

**Why `CORRUPTION_TYPES` / `SHIFT_TYPES` live in `core.validation` and not in
`core.perturbations`.** §4.4 originally published them from `core.perturbations`.
But `validate_stress_params` (§4.3) must reject an unknown `corruption_type` /
`shift_type`, and `apply_perturbation` (§4.4) must call `validate_stress_params`.
If the enums are owned by `perturbations`, validation must import them from
`perturbations` while `perturbations` imports validation — an intra-`core/`
cycle, which §2.1 declares a test failure. There is no placement that satisfies
both sections as originally written.

Resolution: `core.validation` owns the canonical tuples **and** the per-operator
numeric parameter table; `core.perturbations` re-exports them
(`from core.validation import CORRUPTION_TYPES, SHIFT_TYPES`), so the documented
public path `core.perturbations.CORRUPTION_TYPES` keeps working unchanged for
callers and tests. Re-export is a one-way import and creates no cycle.

The numeric parameter table (`op -> {param -> (low, high, exclusive_low,
exclusive_high)}`) is derived from `StressBounds` at call time; it is not a
second hard-coded copy of the ranges.

## 3. Module Boundaries

| Module | Owns | Must not |
|---|---|---|
| `core.config` | All tunable constants (bounds, score weights/caps, grade cutoffs, artifact policy, log level, seed) | Contain business logic or read `st.*` |
| `core.errors` | Typed exception hierarchy + error `context` payloads | Import anything project-local |
| `core.validation` | Parameter normalisation/bounds checks, finiteness assertions, frame sanity checks | Emit UI messages |
| `core.perturbations` | The single implementation of every stress operator; purity + RNG injection; registry | Mutate inputs, import `modules`, emit UI |
| `core.model_io` | Filename sanitisation, path containment, SHA-256 manifest, atomic writes, load gating | Import `modules`, emit UI |
| `core.reporters` | `Reporter` protocol + null/collecting implementations | Import `streamlit` |
| `core.state` | Session key constants, idempotent bootstrap, reset, `AppContext` façade over an injected mapping | Import `streamlit` (mapping is injected) |
| `modules.*` | Domain computations and figures | Import `streamlit` after Wave 4 |
| `views.*` | Widget layout, orchestration, error rendering | Contain domain computations |
| `app.py` | Wiring only | Contain page bodies |

## 4. Interfaces (binding signatures)

### 4.1 `core.errors`

```python
class FrameworkError(Exception):
    context: dict[str, Any]

class ConfigurationError(FrameworkError): ...
class ValidationError(FrameworkError):
    def __init__(self, message: str, *, field: str | None = None,
                 value: Any = None) -> None: ...
class ParameterOutOfRangeError(ValidationError): ...
class UnsupportedStressTypeError(ValidationError): ...
class DatasetError(FrameworkError): ...
class DatasetLoadError(DatasetError): ...
class ArtifactError(FrameworkError): ...
class ArtifactPathError(ArtifactError): ...      # traversal / containment violation
class ArtifactNotFoundError(ArtifactError): ...   # missing file or manifest entry
class ArtifactIntegrityError(ArtifactError): ...  # sha256 mismatch
class ArtifactUntrustedError(ArtifactError): ...  # not attested; trust not granted
class UnknownModelError(FrameworkError): ...       # model name not in the trainer (ADR-019)
class ModelNotTrainedError(FrameworkError): ...    # predict/save before training (ADR-019)
class ReportExportError(FrameworkError): ...       # JSON serialisation failure (ADR-019)
```

`UnknownModelError`, `ModelNotTrainedError` and `ReportExportError` close the
porous error boundary of run-011 M4: `ModelTrainer.get_model` / `predict` /
`save_model` and `ReportGenerator.export_to_json` previously raised bare
`ValueError`/`TypeError`, which `views.reporters.render_errors` (which catches
only `FrameworkError`) could not render — so ordinary UI misuse surfaced as a raw
traceback instead of the framework's error UI (conventions §3).

### 4.2 `core.config`

```python
@dataclass(frozen=True)
class StressBounds:
    noise_level: tuple[float, float]      # (0.0, 5.0)
    noise_range: tuple[float, float]      # (0.0, 1.0)
    dropout_rate: tuple[float, float]     # (0.0, 1.0) exclusive upper
    corruption_rate: tuple[float, float]  # (0.0, 1.0) exclusive upper
    scale_factor: tuple[float, float]     # (1.0, 10.0) exclusive lower (ADR-015)
    shift_amount: tuple[float, float]     # (0.0, 10.0)

@dataclass(frozen=True)
class ReliabilityWeights:
    max_perf: float = 25.0
    max_cal: float = 25.0
    max_rob: float = 25.0
    max_conf: float = 25.0
    ece_cap: float = 0.5
    drop_cap: float = 0.5
    grade_cutoffs: tuple[tuple[float, str], ...] = (
        (90.0, "A+"), (80.0, "A"), (70.0, "B"),
        (60.0, "C"), (50.0, "D"),
    )   # score < 50.0 => "F" / "#C0392B" (implicit fallthrough, see ADR-009)

@dataclass(frozen=True)
class ScoringPolicy:
    """Single owner of "how good is this model" policy (ADR-018, run-011 M7).

    ``ReliabilityWeights`` keeps the component caps and letter grades;
    ``ScoringPolicy`` owns the values that run-011 found duplicated and
    self-contradictory: the composite weights, the missing-component policy, the
    minimum measured-components gate, the ECE quality bands and the robustness
    severity bands.
    """

    composite_weights: tuple[tuple[str, float], ...] = (
        ("performance", 0.5), ("robustness", 0.25), ("calibration", 0.25),
    )
    missing_component_points: float = 0.0   # zero credit; NO redistribution
    min_measured_components: int = 2         # below this a model is unrated
    ece_quality: tuple[tuple[float, str], ...] = (
        (0.03, "Excellent"), (0.07, "Good"), (0.15, "Moderate"),
    )   # higher ECE => quality_fallthrough ("Poor")
    severity_bands: tuple[tuple[float, str], ...] = (
        (5.0, "Low"), (15.0, "Medium"), (30.0, "High"),
    )   # higher drop => severity_fallthrough ("Critical")

    def resolve_ece_quality(self, ece: float) -> str: ...
    def resolve_severity(self, drop_pct: float) -> str: ...
    def weight_for(self, component: str) -> float: ...

@dataclass(frozen=True)
class ArtifactPolicy:
    root_dir: Path = Path("saved_models")
    allowed_suffixes: tuple[str, ...] = (".pkl",)
    # MUST be applied with re.fullmatch(), or anchored with \Z — never re.match()
    # with a `$` anchor, because `$` also matches before a trailing newline and
    # would admit "model.pkl\n". See ADR-009.
    filename_pattern: str = r"[A-Za-z0-9][A-Za-z0-9._-]*\.pkl\Z"
    max_filename_length: int = 128   # keeps ENAMETOOLONG out of the trust path
    # §4.5 step (d) mandates a size cap (ADR-010.2); the knob MUST exist here or the gate
    # references a setting no implementation can read. 256 MiB is far above any
    # pickled sklearn/xgboost classifier produced by this app (typically <10 MiB)
    # and bounds the hashing/DoS surface of `verify()`.
    max_size_bytes: int = 256 * 1024 * 1024
    require_attestation: bool = True

@dataclass(frozen=True)
class AppConfig:
    stress_bounds: StressBounds
    reliability: ReliabilityWeights
    artifacts: ArtifactPolicy
    scoring: ScoringPolicy = ScoringPolicy()   # ADR-018; defaulted
    log_level: str = "INFO"
    random_seed: int | None = None

def get_config(overrides: Mapping[str, Any] | None = None) -> AppConfig: ...
```

**Monotone scoring policy (ADR-018, run-011 C3/M1/M7).** Both scored outputs —
`ReliabilityScorer.score_model` and `ModelComparator.compute_composite_score` —
use a **fixed denominator with zero credit for a missing component and no
redistribution**. Every component score is `>= 0`, so measuring additional
evidence can never lower a total (the pre-remediation behaviour inverted rankings:
`45/50x100 = 90.0` fell to `75/100x100 = 75.0` when robustness/calibration were
added). A model with fewer than `min_measured_components` measured components is
`rated=False` and is surfaced separately rather than ranked. Entropy is measured
in **bits** (`np.log2`) at the `utils.metrics` boundary so the confidence
component's `log2(n_classes)` normalisation is dimensionally consistent.

**Parity rule:** every default above reproduces current behaviour exactly. The
reliability constants mirror `modules/reliability_module.py:25-51,66,75`. Changing
a default is a behaviour change and requires a matching test update plus an ADR.

**`get_config` semantics (pinned by ADR-011.4).** ADR-005 said "cached singleton,
overridable by keyword overrides or `STF_*` environment variables" while the §4.2
signature showed neither caching nor environment handling. A cached singleton
cannot honour per-call overrides, and if overrides mutate the cache then a single
`get_config({"random_seed": 42})` in one test silently repoints every later test —
which would corrupt the Wave 0 baseline fixtures and the Wave 6 parity gate
without any visible failure. Binding rules:

1. `get_config()` with **no** argument returns the process-wide cached singleton
   (`functools.lru_cache` over a zero-arg private function).
2. `get_config(overrides)` with a **non-None** `overrides` returns a **new,
   uncached** `AppConfig` built by `dataclasses.replace` over the defaults. It
   MUST NOT mutate, populate or invalidate the singleton.
3. `STF_*` environment variables are read **only** on the zero-arg path, only at
   first call, and are `int`/`float`/`str` coerced per field. The resolved values
   are logged at `INFO` by `core.logging_config`. Recognised names are exactly
   `STF_LOG_LEVEL` and `STF_RANDOM_SEED`; anything else is ignored (no
   `getattr`-driven dynamic lookup — conventions §6 bans dynamic resolution).
4. A `monkeypatch.setenv`-based test MUST therefore exercise the env path only if
   the singleton can be reset; otherwise the env path is tested by calling
   `get_config(overrides=...)` and asserting only the override semantics. Wave 0's
   fixture-capture script MUST use `get_config()` (singleton) or explicit
   `overrides`, never a mutated cache.

**`random_seed` MUST have a declared consumer (ADR-011.4).** `AppConfig.random_seed`
was specified but no component read it, which is precisely the "tunable with no
reader" shape MIN-005 complained of. Binding: `views/base.build_context()`
constructs one `np.random.Generator` per script run from
`config.random_seed` (`np.random.default_rng(seed)`; `None` → unseeded) and
exposes it as `AppContext.rng` (§4.7). The stress-testing view passes
`ctx.rng` into every `apply_perturbation(rng=...)` call. No other module may
construct a `Generator`, and `core/` MUST NOT read `random_seed` directly.

### 4.3 `core.validation`

```python
def validate_stress_params(op: str, params: Mapping[str, Any],
                           bounds: StressBounds | None = None) -> dict[str, Any]:
    """Return normalised params; raise UnsupportedStressTypeError / ParameterOutOfRangeError."""

def validate_frame(df: pd.DataFrame, *, min_rows: int = 1) -> None: ...
def ensure_finite(values: np.ndarray, *, context: str) -> np.ndarray:
    """Raise ValidationError if a perturbation produced NaN/inf."""
```

Validation bounds MUST be a superset of the UI slider ranges
(`app.py:1207-1281`) and of every batch config (`app.py:1420-1583`), so no
existing UI path can be rejected. Verified superset: UI maxima are
`noise_level<=1.0`, `noise_range<=0.5`, `dropout_rate<=0.8`,
`corruption_rate<=0.5`, `scale_factor<=3.0`, `shift_amount<=2.0`; batch maxima
are `0.3/0.3/0.3/0.25/2.5/1.5`. Both sit inside the §4.2 defaults.

**Single-source invariant:** `apply_perturbation` and `validate_stress_params`
both accept an optional `bounds`. When it is `None` they MUST both fall back to
`get_config().stress_bounds` — never to a second hard-coded literal. Two
independent default sources would let validation and execution disagree, which is
precisely the class of bug this kernel exists to remove. A parity test must assert
that passing `bounds=None` and passing `get_config().stress_bounds` produce
identical accept/reject decisions for the six operators.

**Complete validated parameter set (added by ADR-011.1).** The original §4.3
listed only the six *numeric* parameters and was silent on the two *enum*
parameters and on unknown keys, which left three of the eight live UI/batch
parameter shapes unspecified. `validate_stress_params` MUST handle all three:

| `op` | numeric params (bounds from `StressBounds`) | enum params (source of truth) |
|---|---|---|
| `gaussian_noise` | `noise_level` | — |
| `uniform_noise` | `noise_range` | — |
| `feature_dropout` | `dropout_rate` | — |
| `feature_corruption` | `corruption_rate` | `corruption_type` → `CORRUPTION_TYPES` |
| `scale_perturbation` | `scale_factor` | — |
| `distribution_shift` | `shift_amount` | `shift_type` → `SHIFT_TYPES` |

`CORRUPTION_TYPES = ("zero", "mean", "random", "extreme")` and
`SHIFT_TYPES = ("mean", "variance")` are owned by `core.validation` per §2.3 and
re-exported by `core.perturbations`. Both tuples were verified to equal the UI
option lists exactly: `app.py:1248-1252` offers
`["zero", "mean", "random", "extreme"]` and `app.py:1270-1271` offers
`["mean", "variance"]`. No UI option is missing from the tuples and no tuple
member is unreachable from the UI.

1. **Unknown `op`** → `UnsupportedStressTypeError`.
2. **Numeric out of bounds**, including the exclusive bounds
   (`scale_factor <= 1.0`, `dropout_rate >= 1`, `corruption_rate >= 1`) →
   `ParameterOutOfRangeError` naming the field and the accepted range.
   **Every position of every UI slider MUST be accepted** — the widget and the
   bound cannot disagree in either direction (ADR-015). A slider that publishes a
   position its own validator rejects is a dead control, and per conventions §7 it
   could only ever be reported as an error the user cannot act on.
3. **Enum not a member of its tuple** → `UnsupportedStressTypeError` (the type is
   the unsupported thing, not the value's magnitude).
4. **Unknown parameter key** → `ValidationError` listing the accepted keys for
   that `op`. This closes a behaviour change that §9 did not list: today
   `batch_stress_test` calls `self.add_gaussian_noise(X, **params)`
   (`modules/stress_module.py:316-329`), so a stray key raises a bare
   `TypeError: add_gaussian_noise() got an unexpected keyword argument` from
   deep inside the batch loop, with no operator name in the message. This is
   added to §9 as item 9.

### 4.4 `core.perturbations`

```python
OpFn = Callable[[np.ndarray, Mapping[str, Any], np.random.Generator], np.ndarray]

@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    fn: OpFn
    allowed_params: tuple[str, ...]
    defaults: Mapping[str, Any]

OPERATIONS: Mapping[str, PerturbationSpec]        # 6 entries, frozen
CORRUPTION_TYPES: tuple[str, ...] = ("zero", "mean", "random", "extreme")  # re-export
SHIFT_TYPES: tuple[str, ...] = ("mean", "variance")                          # re-export
OPERATION_LABELS: Mapping[str, str]        # UI label -> registry key, 6 entries

def available_operations() -> tuple[str, ...]: ...
def operation_key(label: str) -> str:
    """Map a Streamlit selectbox/radio label to its registry key;
    raise UnsupportedStressTypeError on an unknown label."""
def apply_perturbation(X, op: str, params: Mapping[str, Any] | None = None, *,
                       rng: np.random.Generator | None = None,
                       bounds: StressBounds | None = None):
    """Pure. Never mutates X. DataFrame in -> DataFrame out (index, columns and
    column order preserved); ndarray in -> ndarray out. Non-finite output raises."""
```

**Registry keys are binding.** `OPERATIONS` MUST be keyed by exactly the six
strings that `StressTester.batch_stress_test` already dispatches on
(`modules/stress_module.py:316-329`) and that `app.py:1434-1583` already writes
into `stress_configs`:

```
gaussian_noise | uniform_noise | feature_dropout
feature_corruption | scale_perturbation | distribution_shift
```

`snake_case`, not camelCase: inventing `addGaussianNoise`/`gaussianNoise` keys
silently disables every batch stress test, because the current unknown-type
branch is a `continue` (Wave 2 replaces it with a raise, which is how the mistake
would surface — as a crash on every batch run).

**`OPERATION_LABELS` is mandatory (added by ADR-011.2).** The registry is keyed
by `snake_case`, but the *single*-stress-test UI path never uses those keys. It
uses six human-readable labels, in two separate dispatch chains:

| UI label (`app.py:1190-1198`) | registry key | second dispatch site |
|---|---|---|
| `"Gaussian Noise"` | `gaussian_noise` | `app.py:1300` |
| `"Uniform Noise"` | `uniform_noise` | `app.py:1302` |
| `"Feature Dropout"` | `feature_dropout` | `app.py:1304` |
| `"Feature Corruption"` | `feature_corruption` | `app.py:1306` |
| `"Scale Perturbation"` | `scale_perturbation` | `app.py:1308` |
| `"Distribution Shift"` | `distribution_shift` | `app.py:1310` |

Both chains are `if/elif` on the label today, so both must become
`operation_key(...)` lookups in Wave 2 — the labels are preserved verbatim
(conventions §7) and are **not** permitted as `OPERATIONS` keys. Without a
declared mapping, an implementer either pollutes the registry with labels (which
breaks `batch_stress_test`, ADR-010.6) or writes an ad-hoc mapping in a view
module that no test can reach. `OPERATION_LABELS` is the single declared
translation point; both chains and any future caller go through
`operation_key()`.

Contract details (must be honoured by every operator):

1. Operators receive a 2-D float ndarray of values only; container handling lives
   in `apply_perturbation`. This removes the 12 duplicated DataFrame/ndarray
   branches in `modules/stress_module.py:20-238`.
2. Zero-variance columns: when `std == 0` the noise scale is `0`, never `NaN`.
   `1 / scale_factor` must never be evaluated with `scale_factor == 0`.
3. **Formula parity, not bit-exidelity.** The formulas below MUST be preserved,
   but output values are **not** required to equal today's numbers: vectorised
   numpy and an injected `Generator` necessarily draw randomness in a different
   order from the current per-column `np.random.*` loop. Parity is therefore
   defined as (a) same formula, (b) same distributional parameters, (c) seeded
   self-consistency across runs. The only **bit-exact** freeze in this milestone
   is `core/reliability` scoring, which is pure arithmetic with no RNG. Any test
   asserting equality with the *old implementation's* numbers is a defect, not a
   stronger test.
4. Current semantics, verified against the source: dropout multiplies by a
   **per-cell** Bernoulli mask (`rng.random(shape) > rate`); corruption replaces
   `int(len * rate)` sampled cells with zero / column mean / uniform(min,max) /
   random choice of (min,max); scale picks `factor` or `1/factor` **once per
   column** via `rng.random() > 0.5`; mean shift adds `std * amount`; variance
   shift maps `mean + (x - mean) * amount`.
5. Every operator receives an explicit `np.random.Generator`; `np.random.*`
   module-level calls are banned inside `core/` (test-enforced).
6. **Recorded naming debt (ADR-010.7) — do not "fix" it in this milestone.** `feature_dropout`
   masks individual *cells*, not features per sample as its docstring and the UI
   label ("Fraction of features to randomly set to zero") claim; and
   `distribution_shift("variance")` contracts values toward the mean rather than
   scaling variance. Both names are load-bearing for published results. Correct
   the docstrings and UI help text, keep the arithmetic.
7. **`scale_factor=0` has no deterministic pre-fix behaviour (ADR-011.7).** Today
   `modules/stress_module.py:180` evaluates `1 / scale_factor` *only* in the
   `else` arm of `factor = scale_factor if np.random.random() > 0.5 else
   1 / scale_factor`, so `scale_perturbation(X, scale_factor=0.0)` raises
   `ZeroDivisionError` on roughly half of all draws and silently succeeds on the
   rest. Consequently §9 item 8 has **no stable regression test against the old
   code** — there is no seed that makes the old failure reproducible, because the
   branch taken depends on the global `np.random` state at that instant. The Wave 2
   test MUST therefore assert only the *post-fix* contract (a deterministic
   `ParameterOutOfRangeError` raised before any RNG draw, for every seed) and MUST
   NOT attempt to characterise the old behaviour. Same reasoning applies to any
   future zero-division defect: seed-hunting a non-deterministic old failure is a
   flaky test, not a strong one.
8. **The purity test cannot fail before the fix (ADR-011.6) — correct §6.1's
   framing.** All six operators already begin with `X.copy()` / `result =
   X.copy()` (`modules/stress_module.py:26,60,102,126,180,214`), so **no operator
   mutates the caller's object today**; MAJ-002's "in-place DataFrame mutation" is
   not the observable defect. The observable defect is the *silent discard* of the
   `.values` write (§6.1 table, reproduced exactly in the third verification
   pass). A test named `test_<op>_does_not_mutate_input` passes on the unfixed
   code and is therefore worthless as a gate. The load-bearing Wave 2 assertions
   are the §10.2 ones: the `int64 + float64` frame **changes** under all six
   operators, and an `int64` ndarray returns `float64`. Purity is retained as a
   *contract* (it is cheap to keep and the new container path could break it), but
   it MUST NOT be presented in the wave report as the fix for MAJ-002.

### 4.4.1 Numeric container contract (dtype widening — behaviour change)

`apply_perturbation` coerces its input once, up front:

- `DataFrame` → `X.to_numpy(dtype=np.float64)`, then reconstruct with the
  **original index, columns and column order**. Perturbed columns are returned as
  `float64`; integer columns are NOT cast back down.
- `ndarray` → returned as `float64`.

Rationale, measured against `modules/stress_module.py` today: an `int64` ndarray
input is perturbed in place and returned as `int64`, so every fractional
perturbation is **truncated back to an integer** — verified `dtype=int64` with the
noise silently discarded. Widening is the only way to make the operators
meaningful on integer data.

If coercion fails (a frame carrying genuinely non-numeric columns, e.g. `str` or
`bool`), `apply_perturbation` MUST raise `ValidationError` naming the offending
column names in `exc.context` — never leak the raw `ValueError`/`TypeError` from
`to_numpy`, and never silently no-op. This is the counterpart to §6.1: the
current code does neither — it silently returns an unchanged frame.

### 4.5 `core.model_io` (security boundary)

```python
@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: Path
    sha256: str
    size_bytes: int
    created_at: str            # ISO-8601 UTC
    model_name: str
    format_version: int        # 1
    library_versions: Mapping[str, str]   # {"scikit-learn": ..., "xgboost": ...}

class ModelArtifactStore:
    def __init__(self, policy: ArtifactPolicy | None = None) -> None: ...
    @property
    def manifest_path(self) -> Path: ...
    def resolve(self, filename: str) -> Path:
        """Raise ArtifactPathError unless filename matches filename_pattern and
        the resolved path stays inside root_dir (symlinks rejected)."""
    def save(self, model: Any, filename: str, *, model_name: str) -> ArtifactRecord: ...
    def list_artifacts(self) -> list[ArtifactRecord]: ...   # manifest order
    def verify(self, filename: str) -> ArtifactRecord: ...   # sha256 re-hash
    def load(self, filename: str, *, trust: bool = False) -> tuple[str, Any]: ...
    def delete(self, filename: str) -> None: ...
```

Security model — **the honest position**: pickle cannot be made safe in-process.
`joblib` does not fix this (it uses pickle under the hood) and a restricted
unpickler is not a viable boundary here. The mitigation is therefore a
**provenance and containment boundary**, not "safe deserialization":

1. Filename sanitisation on **both** save and load (`^[A-Za-z0-9][A-Za-z0-9._-]*\.pkl$`,
   no separators, no absolute paths, no `..`, no NUL, no symlink).
2. Path containment: resolved candidate must be `is_relative_to(root_dir)`;
   anything else raises `ArtifactPathError`. This closes the **save-side**
   arbitrary-write hole at `app.py:680-686` (raw `st.text_input` → `os.path.join`),
   which the Critic audit did not report.
3. Attestation: `save()` writes `saved_models/manifest.json`
   (`{"format_version": 1, "artifacts": {...}}`) with a SHA-256 digest, size,
   timestamp, model name and library versions. Writes are atomic
   (temp file + `os.replace`).
4. `load()` re-hashes the file and compares against the manifest **before**
   `pickle.load`. Mismatch or absent entry raises `ArtifactIntegrityError` /
   `ArtifactNotFoundError` unless the caller passes `trust=True`, which the UI
   only sets from an explicit, default-unchecked confirmation widget that states
   the file will execute arbitrary Python code.
5. `load()` must apply, in this exact order: (a) `resolve()` → `ArtifactPathError`,
   (b) name-length cap (`policy.max_filename_length`) → `ArtifactPathError`,
   (c) existence → `ArtifactNotFoundError`,
   (d) size cap (`policy.max_size_bytes`) → `ArtifactIntegrityError`,
   (e) hash/manifest attestation, (f) estimator type gate.
   `trust=True` bypasses **only** (e); it never bypasses (a)–(d), because
   those are containment controls rather than provenance controls.
6. **The estimator type gate MUST NOT make this module unimportable without
   scikit-learn** (ADR-010.1). `load()` enforces an estimator type check, but the check MUST
   be resolved *lazily* (import inside the function body, or inject a base tuple)
   so that `import core.model_io` succeeds on a bare interpreter. Signature:

   ```python
   class ModelArtifactStore:
       def __init__(self, policy: ArtifactPolicy | None = None, *,
                    estimator_bases: tuple[type, ...] | None = None) -> None: ...
   ```

   `estimator_bases=None` resolves to `(sklearn.base.BaseEstimator,)` on first
   use. Tests inject a locally-defined stand-in class. This is what makes §10's
   "stdlib + pytest only" requirement satisfiable — a module-level
   `from sklearn.base import BaseEstimator` would transitively import
   scikit-learn, numpy and scipy into every test that touches this file, and the
   security gate would become unrunnable in the bare environment where it matters
   most. The *production* default must still be `BaseEstimator`: the relaxation
   exists for test injection only, and a caller passing `estimator_bases=()`
   disables the gate by explicit opt-in.
7. Loaded objects must satisfy `isinstance(model, estimator_bases)`; otherwise
   `ArtifactIntegrityError`.
8. Every `OSError` raised while opening, hashing or replacing a file MUST be wrapped
   into the matching `ArtifactError` subclass. Conventions §3 forbids leaking a raw
   `OSError` (e.g. `ENAMETOOLONG`, `EACCES`, `EISDIR`) across the module boundary.
9. A manifest-miss is a distinct, actionable condition from a tamper: the error
   message MUST tell the user the file is unattested and can be loaded only by
   re-saving it through the app (to write a manifest) or by explicitly trusting it.
   Residual risk is documented in the ADR, the module docstring and README.
10. The store MUST `root_dir.mkdir(parents=True, exist_ok=True)` before writing.
    Today `save_model` calls `os.makedirs(os.path.dirname(filepath))`
    (`modules/model_module.py:357`), which raises `FileNotFoundError` when the
    caller passes a bare filename; `saved_models/` does not exist in a fresh
    checkout, so this is the first-save path.
11. **`library_versions` MUST be resolved without importing the scientific stack
    (ADR-011.3).** §4.5's `ArtifactRecord.library_versions` is populated by
    `save()`, and §10.1 makes `test_model_io.py` stdlib + pytest only — a test
    that calls `save()`. `import sklearn; sklearn.__version__` inside `save()`
    transitively imports numpy and scipy and silently promotes the security suite
    to T1, which is the identical failure mode ADR-010.1 was written to prevent.
    Binding: resolve each distribution through the **stdlib** only —
    `importlib.metadata.version("scikit-learn")` / `("xgboost")`. On
    `importlib.metadata.PackageNotFoundError` record the literal
    `"<not installed>"`; the key is always present so the manifest schema is
    stable. `core/model_io.py` MUST NOT `import sklearn`, `xgboost`, `numpy` or
    `pandas` at module level or inside `save()`.
12. **Manifest schema is frozen at `format_version = 1`** and gains no field in
    this milestone. `load()` MUST tolerate a manifest whose recorded
    `library_versions` values differ from the running environment (that is
    provenance metadata, not an integrity check) and MUST NOT reject on it.

`ModelTrainer.save_model(model_name, filepath)` and
`ModelTrainer.load_model(model_name, filepath, *, trust=False)` keep their
signatures and delegate to the store (`filepath` is reduced to
`Path(filepath).name`). Both stop importing `streamlit`.

### 4.6 `core.reporters`

```python
class Reporter(Protocol):
    def info(self, message: str) -> None: ...
    def success(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...

class NullReporter: ...          # default for domain services
class CollectingReporter: ...    # records messages for tests
```

`views/reporters.py: class StreamlitReporter(Reporter)` is the **only**
streamlit messaging adapter. Domain services accept `reporter: Reporter | None`
in `__init__` defaulting to `NullReporter`.

Two view-layer helpers are part of this contract:

- `render_error(exc, reporter=None)` — renders one exception plus its
  `context` and type as captions.
- `render_errors(reporter=None)` — a context manager wrapping a **whole** button
  handler. It suppresses `FrameworkError` and renders it, and lets everything
  else propagate. It exists because conventions §7 binds *every* mutating action,
  and a hand-copied `try` at each call site is easy to forget at the next button
  — the run-003 defect where four calibration handlers referenced `render_errors`
  without importing it was live in the tree while 673 tests were green.

**Binding (added by the run-003 MINOR-3 remediation):** any `views/module_*.py`
that presents a mutating action — i.e. contains an `if st.button(…)` handler —
MUST render `FrameworkError`, and `tests/test_architecture.py` asserts both that
the renderer is *imported* and that each handler body actually *applies* it. The
gate keys off button ownership, not a file list, so it cannot rot as pages are
added. Catching any `FrameworkError` **subclass** counts, since the natural guard
for a specific call site is the specific type it raises.

### 4.7 `core.state`

```python
class StateKeys:                      # string values MUST equal today's literals
    DATA_MANAGER = "data_manager"
    MODEL_TRAINER = "model_trainer"
    POST_STRESS_ANALYZER = "post_stress_analyzer"
    CALIBRATION_ANALYZER = "calibration_analyzer"
    MODEL_COMPARATOR = "model_comparator"
    RELIABILITY_SCORER = "reliability_scorer"
    REPORT_GENERATOR = "report_generator"
    STRESS_TESTER = "stress_tester"
    CURRENT_STEP = "current_step"
    DATA_LOADED = "data_loaded"
    DATA_PREPARED = "data_prepared"
    MODEL_TRAINED = "model_trained"
    # + derived keys: "predictions", "single_stress_result",
    #   "batch_stress_results", "batch_stress_results_by_model"

def init_state(store: MutableMapping[str, Any]) -> None: ...    # idempotent
def reset_state(store: MutableMapping[str, Any], *, keep_services: bool = False) -> None: ...

@dataclass(frozen=True)
class AppContext:
    store: MutableMapping[str, Any]
    rng: np.random.Generator            # added by ADR-011.4; see §4.2
    def ensure_service(self, key: str, factory: Callable[[], Any]) -> Any: ...
    def has_data(self) -> bool: ...
    def has_models(self) -> bool: ...
    def invalidate_from(self, step: int) -> None: ...   # clears derived results
```

`AppContext` is constructed **per script run** from `st.session_state`
(`views/base.build_context()`), never cached in session state, because Streamlit
re-executes the script on every interaction. All `hasattr(st.session_state, ...)`
checks (e.g. `app.py:1155`, `app.py:1608`) and the ad-hoc init block
(`app.py:44-59`) are replaced by `init_state` / `ensure_service`.

`AppContext` is `frozen=True` and holds exactly two references (`store`, `rng`),
so it stays a cheap façade per §8: constructing one per rerun allocates two
objects and copies two references. `rng` is a fresh
`np.random.default_rng(config.random_seed)` per script run — **not** stored in
`session_state`, because persisting a `Generator` across reruns would make results
depend on how many times the user clicked, and `session_state` is not required to
be deep-copyable. `core.state` MUST NOT import `streamlit` (§2.1); the `Generator`
is injected by `views/base.py`, which is the only place allowed to build one.

### 4.8 `views`

```python
# views/base.py
def build_context() -> AppContext: ...
def render_sidebar(config: AppConfig) -> str: ...            # returns selected page key
def render_status_footer(ctx: AppContext) -> None: ...

# views/module_0N_*.py
def render(ctx: AppContext) -> None: ...
```

`app.py` dispatches through an explicit ordered mapping of the nine existing
sidebar labels (emoji strings preserved verbatim) to the `render` callables.

**Extraction pre-verified (do not re-derive).** The nine pages are currently one
flat module-level `if/elif module == "…" /` chain at `app.py:90-3500`, with
bodies indented exactly one level. Verified:

- The navigation variable is `module` (not `page`); the sidebar radio is at
  `app.py:69` and the status footer at `app.py:3501-3517`.
- **Zero cross-branch free variables.** An `ast` pass over all nine branches
  (assignments vs. loads, excluding imports, module-level pre-assignments and
  builtins) found no name that is loaded in one page and assigned only in a
  different page. Extraction into nine functions therefore cannot break a name
  that Python currently leaks between branches. This is the single biggest
  mechanical risk in Wave 5 and it is now retired.
- **63 `key=` widget literals, all unique.** This is the persisted session
  contract; the count is recorded so a reviewer can assert the count is
  unchanged after extraction, not just eyeball the diff.
  **After execution the contract is 64, not 63.** The 63 pre-remediation keys are
  unchanged; Wave 1 added `load_model_trust` — the explicitly-unchecked "this file
  executes Python code" confirmation required by ADR-001. A post-remediation count
  of 63 would mean the trust gate had been dropped, so the invariant is written
  as `63 + len(added_during_remediation)` in `tests/test_architecture.py` rather
  than as a bare number, and the full key-to-view mapping is frozen in
  `tests/fixtures/widget_keys.json` so the check survives without the original
  single-file `app.py`.
- 13 `hasattr(st.session_state, …)` sites (`app.py:115, 390, 870, 916, 990,
  1068, 1155, 1329, 1608, 1635, 1725, 3503, 3508, 3513`) are the §4.7
  `init_state`/`ensure_service` migration surface.
- `StressTester` is currently created lazily at `app.py:1156`, only when the
  Stress Testing page is visited. Moving it into eager `init_state` is safe: its
  `__init__` assigns three attributes and has no side effects.

**Correction to "no nested function defs" (ADR-011.5).** §4.8 previously claimed
the nine branch bodies contain no nested `def`. That claim is **false**: there is
exactly one, at `app.py:1984` inside branch 5 (5️⃣ Post-Stress Evaluation) —

```python
def color_severity(val):        # app.py:1984, used once at app.py:1994
    if val == "Critical":  return "background-color: #ff4444; color: white"
    elif val == "High":    return "background-color: #ff9944; color: white"
    elif val == "Medium":  return "background-color: #ffcc44"
    else:                  return "background-color: #44ff44"
```

It is a **leaf closure**: its only name binding from an enclosing scope is
nothing — it reads only its own parameter `val` and returns string literals, uses
no `nonlocal`/`global`, and is invoked two statements after its definition at
`app.py:1994`. Extraction into `render(ctx)` is therefore safe: it simply remains
a nested `def` inside `views/module_05_post_stress.py`, unchanged and in place.
The zero-cross-branch-free-variable conclusion is unaffected — a third,
independent `ast` pass over the same nine branches reproduced the count of 0, and
the branch starts (`88, 384, 783, 1139, 1766, 2228, 2617, 2891, 3181`) exactly.
This correction is recorded so that a reviewer re-running the stated
pre-verification does not report a false mismatch against §4.8.

### 4.9 `main.py` — the CLI contract (added by ADR-012)

```python
# main.py
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

`main.py` exists for the three things a Streamlit page cannot do: **report** a
resolved configuration or a broken environment, **answer** a script with a
machine-readable exit code, and **launch** the app. It contains no domain logic;
each command prints what `core.config` resolved, delegates to
`core.perturbations`, reads the attestation manifest, or execs `app.py`.

| Command | Needs | Does |
|---|---|---|
| `help` (`-h`, `--help`, no args) | stdlib | top-level help |
| `version` (`--version`) | stdlib | interpreter, platform, every pin in `requirements.txt` vs what is installed |
| `info` | stdlib | the resolved `AppConfig`: log level, seed, stress bounds, reliability policy, artifact policy |
| `operators` | numpy | the six registry keys and the `OPERATION_LABELS` → key translation, proving `operation_key` |
| `doctor` | stdlib | 7 checks: interpreter floor, `app.py`, venv, pins, pytest, artifact root, manifest |
| `serve` | streamlit | `os.execv` of `streamlit run app.py` with `--port/--address/--headless` |

**Exit-code contract.** `0` = the work was done, *or* an unrecognised argument
was reported and ignored. `1` = the work failed (missing dependency, unreadable
manifest, unmet pin) and the message names the remedy. `2` = a usage error inside
a recognised command (argparse), or any unrecognised argument when `--strict` is
passed. `main()` is total: it returns an int and never raises `SystemExit`, so a
caller or a test asserts on the value the shell would observe.

**The permissive default is the decision, not an oversight.** An unrecognised
argument produces a stdout note naming it, a usage line, the supported commands
and the `--strict` hint — then exit `0`. `--strict` restores argparse's
convention (message and usage on stderr, exit `2`), and is what CI should pass.
The reason is recorded in ADR-012: a probe of an entry point must receive an
*answer*, and an entry point that can only answer `0` cannot report anything.
The counterpart obligation is that non-zero exits must be real and reachable, so
`doctor` in a stackless environment and `serve` without streamlit both exit `1`
with an actionable message — asserted in `tests/test_cli.py`, together with a
stub-falsification test proving those two assertions can fail.

`main.py` MUST import nothing but stdlib and `core`, **at any scope**
(§2.1), so the diagnostic path survives a missing scientific stack. That is
checked by AST walk in `tests/test_cli.py` and behaviourally by running
`python -S main.py …`, where `-S` hides every site-packages directory.

## 5. Error & Logging Policy

- Domain code **raises** typed `FrameworkError` subclasses; **views catch and
  render**. No `st.*` call inside `core/`, `modules/`, `utils/`.
- No bare `except:`; no `except Exception` that swallows. The single documented
  exception is per-class Brier degradation in
  `modules/calibration_module.py:104-107`, which must additionally emit
  `logger.warning`.
- Every module obtains `logger = logging.getLogger(__name__)`; `app.py` calls
  `configure_logging(config.log_level)` once. Silent `NaN`/no-op fallbacks are
  replaced by raised errors or logged warnings.
- UI error rendering pattern (uniform, so users always get an actionable path):
  `except FrameworkError as exc: reporter.error(str(exc))` plus
  `st.caption(exc.context)` when context is present. `views.reporters.render_errors()`
  is the equivalent context-manager form for a whole button handler; §4.6 makes
  its application per handler a tested requirement, not a style preference.

## 6. Data-Flow Invariants

1. **Purity of stress operators** — `add_gaussian_noise`, `add_uniform_noise`,
   `feature_dropout`, `feature_corruption`, `scale_perturbation`,
   `distribution_shift` never mutate their argument.

   **Framing correction (ADR-011.6):** this invariant does **not** fix MAJ-002,
   because purity already holds today. Every operator opens with `X.copy()` or
   `result = X.copy()` (`modules/stress_module.py:26, 60, 102, 126, 180, 214`),
   so the caller's object is never written to. Purity is retained as a *contract*
   the new container path could plausibly break, and a purity test is still worth
   having — but it passes on the unfixed code and MUST NOT be cited as the
   remediation or used as the wave gate. MAJ-002's real, observable defect is the
   silent result discard documented immediately below.

   The silent no-op this invariant removes is **wider than "mixed-dtype"**.
   `.values` returns a *view* only when the resulting 2-D block dtype already
   equals the frame's storage dtype; whenever pandas has to build a new array —
   because columns upcast, or because dtypes are heterogeneous — the operators
   write into a temporary and the result is discarded. Measured on the current
   implementation:

   | Input frame | `.values` dtype | write reaches the frame? |
   |---|---|---|
   | `int64` only | `int64` | yes |
   | `float64` only | `float64` | yes |
   | `int64` + `float64` | `float64` (upcast copy) | **no — silent no-op** |
   | `int64` + `float64` + `str` | `object` | **no — silent no-op** |
   | `float64` + `str` | `object` | **no — silent no-op** |
   | `bool` + `float64` | `object` | **no — silent no-op** |

   Per-operator severity on an `int64 + float64` frame (the realistic
   post-CSV-upload shape, since each column is typed independently on read):
   **5 of 6 operators silently no-op** — `add_gaussian_noise`,
   `add_uniform_noise`, `feature_corruption`, `scale_perturbation` and both
   `distribution_shift` modes. Only `feature_dropout` survives, because its
   DataFrame branch writes back through `.iloc` rather than `.values`.

   Consequence for the test suite: the regression frame MUST be
   `int64 + float64` (numeric, coercible, currently broken). A frame containing
   `str`/`bool` is *not* a usable regression frame — under §4.4.1 it now raises
   `ValidationError` instead of changing, which is the intended behaviour, not a
   failure.
2. **Split hygiene** — `DataManager.split_data` keeps scaler fitting on train
   only; stress tests operate on the *same* column space as training.
3. **No mutation of `st.session_state` service internals from views** — views
   call service methods; services mutate their own state.
4. **Label alignment** — `evaluate_stress_test` converts a `Series` target to
   `ndarray` before scoring (already correct at
   `modules/stress_module.py:256-259`); preserve this.
5. **Report export escaping** — every value interpolated into
   `ReportGenerator.export_to_html` MUST pass through
   `html.escape(str(value), quote=True)` (`modules/reporting_module.py:419-429`
   and the KPI/summary block at lines 456-479). Dataset names, model names and
   metric strings are user-controlled; unescaped interpolation is a stored-XSS
   vector in the exported report.

## 7. Security Model (summary)

| Threat | Control | Location |
|---|---|---|
| RCE via model deserialization | Containment + SHA-256 attestation + explicit trust gate + documented residual risk | `core/model_io.py` |
| Path traversal on model **save** | Filename regex + `is_relative_to` | `core/model_io.py`, `app.py:680-686` |
| Arbitrary file write via filename | Same as above | `app.py:680-686` |
| Out-of-range / non-finite stress params | `validate_stress_params` + `ensure_finite` + zero-variance guard | `core/validation.py`, `core/perturbations.py` |
| Integer truncation / silent no-op in perturbations | `float64` coercion up front, purity contract (§4.4.1, §6.1) | `core/perturbations.py` |
| XSS in exported HTML | `html.escape` at every interpolation | `modules/reporting_module.py` |
| Silent corruption of user data | `df[col] = df[col].method()` instead of chained `inplace=True` | `modules/data_module.py:136-149` |

Explicit non-goal: sandboxed/ONNX model loading (see ADR-001).

## 8. Performance Envelope

No caching layer, no background jobs, no async. Streamlit reruns the whole
script per interaction; `AppContext` must stay a cheap façade (reference copies
only). Perturbations stay vectorised numpy. `@st.cache_data` MAY be introduced
for figure generation only, keyed on explicit primitive arguments — not on
service objects.

## 9. Preserved Public Contracts

The following signatures are frozen for this milestone (tests assert them):

- `DataManager.load_dataset / validate_dataset / display_summary / handle_missing_values /
  encode_categorical / scale_features / split_data / get_split_summary / get_data`
- `ModelTrainer.get_model / train_model / predict / evaluate_model /
  get_classification_report / plot_confusion_matrix / plot_metrics_comparison /
  get_feature_importance / plot_feature_importance / save_model / load_model /
  get_best_model / get_probability_distribution / get_model_summary`
- `StressTester.add_gaussian_noise / add_uniform_noise / feature_dropout /
  feature_corruption / scale_perturbation / distribution_shift /
  evaluate_stress_test / batch_stress_test`
- `CalibrationAnalyzer`, `PostStressAnalyzer`, `ModelComparator`,
  `ReliabilityScorer`, `ReportGenerator` public methods
- `utils.metrics.*`, `utils.plotting.*`

Intentional behaviour changes — the **complete** list; anything not here is a bug:

1. `DataManager.load_dataset` raises `DatasetLoadError` instead of returning
   `None` after `st.error` (single call site, `app.py:110`). **Call-site detail
   (ADR-011.8):** `app.py:111` is currently `if data is not None:` guarding the
   `data_loaded = True` + `display_summary(data)` pair at lines 112-113. That
   `if` becomes the `except DatasetLoadError` arm of a `try` at the same call
   site — the `None` check is removed, not merely supplemented.
2. All six perturbation methods are pure (never mutate their argument) and
   reject out-of-range parameters. **The purity half is not a change** — it
   already holds today (ADR-011.6); only the rejection half is new.
3. Model I/O is attested, contained, and gated behind explicit trust.
4. **dtype widening** — perturbed output is `float64`; integer input is no longer
   truncated back to `int` (§4.4.1). This changes reported metric values for
   integer-typed features and is the intended consequence of fixes 2 and 5.
5. **Non-numeric feature frames raise `ValidationError`** instead of silently
   returning an unchanged frame (§4.4.1).
6. `batch_stress_test` raises `UnsupportedStressTypeError` on an unknown
   `type` instead of `continue`-ing (§4.3).
7. `DataManager.get_split_summary` raises `DatasetError` when data is not split
   instead of returning `None`, and no longer divides by zero on empty splits
   (conventions §3 forbids `None` as a failure signal).
8. A zero-variance column yields a zero-scale perturbation, and
   `scale_factor=0` raises `ParameterOutOfRangeError` instead of
   `ZeroDivisionError` — deterministically, for every seed, because validation
   runs before any RNG draw (§4.4.7).
9. **An unrecognised parameter key raises `ValidationError` naming the operator
   and its accepted keys** (§4.3), instead of the bare
   `TypeError: <op>() got an unexpected keyword argument '…'` that
   `batch_stress_test`'s `self.<op>(X, **params)` expansion
   (`modules/stress_module.py:316-329`) raises from inside the batch loop.
10. **An unrecognised enum value** (`corruption_type`, `shift_type`) raises
    `UnsupportedStressTypeError` naming the accepted values, instead of being
    silently ignored — today `feature_corruption(X, corruption_type="typo")` and
    `distribution_shift(X, shift_type="typo")` fall through every `elif`
    (`modules/stress_module.py:135-138, 226-231`) and return the frame
    **completely unperturbed**, with no error and no log.
11. **`vuln_df.style.applymap` → `Styler.map`** (`app.py:1994`, migrating to
    `views/module_05_post_stress.py` in Wave 5). Measured on the pinned
    `pandas==2.3.3`: `FutureWarning: Styler.applymap has been deprecated. Use
    Styler.map instead.` Purely an API migration — the returned styling is
    identical — but `applymap` is removed in pandas 3.0, so leaving it is
    deferred breakage, not a style preference.

**Run-011 remediation behaviour changes (ADR-017..020).** These extend the list
above; the raw-link target, training, `predict`, accuracy, confusion-matrix and
report paths stay in the estimator's native label space.

12. **Calibration/Brier operate in the model's class-index space** (ADR-017,
    run-011 C1). `align_target_indices` / `multiclass_brier` and every
    `CalibrationAnalyzer` entry point take a keyword-only `classes`. Passing the
    estimator's `classes_` maps arbitrary labels (`"cat"`, `[5, 9]`) to
    probability columns; `classes=None` means **strict identity** — labels must
    be exactly `0..K-1` or a typed `ValidationError` is raised, so a call site
    that forgets `classes` fails loudly instead of mis-scoring.
13. **`utils.calculate_brier_score` and `utils.get_confidence_bins` are removed**
    (ADR-017). `modules.calibration_module.multiclass_brier` is the framework's
    single Brier implementation. The §9 `utils.metrics.*` freeze is therefore
    narrowed by exactly these two symbols; everything else in `utils.metrics` is
    frozen unchanged except `get_prediction_entropy`'s units (item 15).
14. **Monotone scoring** (ADR-018, run-011 C3/M1/M7). `score_model` awards a
    missing component `ScoringPolicy.missing_component_points` (`0.0`, formerly
    the `12.5` midpoint) and adds a `rated` flag; `compute_composite_score` uses
    the fixed denominator with no redistribution and a `rated` gate. Ranking and
    recommendation helpers never rank an unrated model.
15. **`utils.metrics.get_prediction_entropy` returns bits**, not nats (ADR-018,
    run-011 M2): `np.log2` instead of `np.log`. The Reliability view's displayed
    "Avg Entropy" changes units; the confidence component is now dimensionally
    correct (uniform binary confidence `= 12.5`, was `~16.34`).
16. **HCE rate uses the error count, not `len(dict)`** (run-011 C2). Both
    `views/module_08_reliability.py` and `views/module_09_reports.py` read
    `hce_dict["count"] / n`; `identify_high_confidence_errors` returns a
    `TypedDict`. The previous `len(hce_dict)` was always `4`.
17. **Complete error boundary and pure getters** (ADR-019, run-011 M3/M4/M5/m2/m6).
    `UnknownModelError` / `ModelNotTrainedError` / `ReportExportError` replace
    bare exits; `PostStressAnalyzer._compute_robustness_breakdown` is the single
    arithmetic source and the getters never write `robustness_scores`;
    `calculate_robustness_score` raises `ValidationError` for an unknown model;
    zero-mean consistency is handled explicitly (no `nan`-masking clip);
    `split_data` guards `stratify` (fewer than two members in any class, or ten
    or more classes) and wraps any residual `ValueError` as `DatasetError`.
18. **Dead comparison weights removed** (run-011 m1). `ModelComparator`'s
    `METRIC_WEIGHT` / `ROBUSTNESS_WEIGHT` / `CALIBRATION_WEIGHT` are deleted;
    `ScoringPolicy` is the single owner.
19. **Prediction/ECE/entropy caching is declined and deferred** (ADR-020,
    run-011 M6). No cache key can be stable while models are name-identified and
    frames are unhashed, so a cache could silently serve stale scores. §8's
    figure-generation bound is reaffirmed.

Measured today, for reference — the defects items 1–11 remove: `int64 + float64`
frames silently no-op in 5 of 6 operators; `int64` ndarrays silently truncate;
`scale_perturbation(scale_factor=0.0)` raises `ZeroDivisionError` on roughly half
of all draws and silently succeeds on the rest; `feature_dropout(dropout_rate=1.0)`
is accepted and zeroes every cell; `get_split_summary()` returns `None` before
splitting and divides by zero when all splits are empty; an unknown
`corruption_type`/`shift_type` returns the input frame untouched; a stray
parameter key raises a bare `TypeError` naming no operator; `app.py:1994` emits a
`FutureWarning` on every Post-Stress render.

## 10. Test Strategy

`tests/` uses pytest. Priority order: security (`test_model_io.py`),
correctness of perturbations (`test_perturbations.py`, `test_validation.py`),
data preprocessing (`test_data_preprocessing.py`), numeric parity
(`test_reliability_parity.py`), and the import-layer contract
(`test_architecture.py`).

The tiering, dtype and regression-frame decisions below are recorded in ADR-010;
§10.2 case lists are normative, §10.3 is an environment constraint.

### 10.1 Two execution tiers (binding)

The suite is split by what it needs to import, because the security and layering
gates must stay runnable in an environment where the scientific stack is absent
or unbuildable (§10.3):

| Tier | Files | Imports allowed | Rationale |
|---|---|---|---|
| **T0 — bare** | `test_architecture.py`, `test_model_io.py`, `test_config.py`, `test_cli.py` | stdlib + pytest only | Uses `ast` parsing, `subprocess`, `hashlib`/`json`/`pathlib`; must never fail for an unrelated dependency problem. A security gate that cannot run is not a gate. |
| **T1 — numeric** | `test_perturbations.py`, `test_validation.py`, `test_data_preprocessing.py`, `test_reliability_parity.py` | the pinned scientific stack | Requires numpy/pandas/scikit-learn. |

Enforcing T0 is what §4.5.6's lazy estimator gate exists for: a module-level
`from sklearn.base import BaseEstimator` in `core/model_io.py` would drag
scikit-learn, numpy and scipy into `test_model_io.py` and silently promote the
security suite to T1.

T0 tests MUST be runnable as `python -m pytest tests/test_architecture.py
tests/test_model_io.py tests/test_config.py tests/test_cli.py` on an interpreter
with only pytest installed. `test_cli.py` earns its place in this tier for the
same reason the other three are here: it drives the entry point with
`python -S main.py …`, which is only meaningful if the suite itself needs nothing.

### 10.2 Minimum required cases

- `test_model_io.py` (T0): rejects `../evil.pkl`, `/etc/passwd.pkl`, `a/b.pkl`,
  `..%2f`-style names, `"model.pkl\n"` (the `$`-anchor bypass, ADR-009.1),
  non-`.pkl` suffixes, over-long names, symlinked targets; refuses an
  unmanifested file; detects single-byte tampering; `trust=True` path works;
  `trust=True` does **not** bypass a traversal attempt; oversized file rejected;
  non-estimator payload rejected (via injected `estimator_bases`); manifest is
  written atomically and is re-readable after a simulated crash (leftover `.tmp`
  ignored); `save()` creates a missing `root_dir`.
- `test_perturbations.py` (T1): input array/frame is byte-identical after each
  operator; **an `int64 + float64` frame actually changes under all six
  operators** (regression test for the silent no-op measured in §6.1 — note
  `feature_dropout` already passes today, so the five failures are the
  regression signal); an `int64` ndarray comes back `float64` with fractional
  values preserved, not truncated; a `str`-bearing frame raises
  `ValidationError` naming the column; zero-variance column yields no NaN;
  `scale_factor=0` and `dropout_rate=1.0` are rejected **deterministically, for
  every seed** (validation precedes any RNG draw — do not seed-hunt the old
  flaky `ZeroDivisionError`, §4.4.7); a fixed `np.random.default_rng(seed)`
  gives identical results across runs; all six `OPERATIONS` keys are exactly the
  strings in §4.4; **`OPERATION_LABELS` has exactly 6 entries, its values are
  exactly the six `OPERATIONS` keys, and its keys are exactly the six literals
  offered by `app.py:1190-1198`** (§4.4); `operation_key("Gaussian Noise") ==
  "gaussian_noise"` and `operation_key("nope")` raises
  `UnsupportedStressTypeError`.
- `test_validation.py` (T1): **every position of every UI slider** and every
  batch-config value in `app.py` is accepted, with **no carve-out** (ADR-015 —
  the `scale_factor` floor exception this file previously carried is deleted, not
  amended); out-of-range values are rejected with a
  user-actionable message; `bounds=None` and `get_config().stress_bounds` agree
  on every accept/reject decision (§4.3 single-source invariant); **every UI
  `corruption_type` / `shift_type` option is accepted and an unknown value is
  rejected with the accepted set in the message** (§4.3 table);
  **an unknown parameter key raises `ValidationError` naming the operator and its
  accepted keys** (§4.3 rule 4); `CORRUPTION_TYPES`/`SHIFT_TYPES` are importable
  from **both** `core.validation` and `core.perturbations` and are equal
  (§2.3 re-export).
- `test_architecture.py` (T0): additionally asserts the **§2.3 intra-`core/`
  ladder** — every import in `core/*.py` resolves to a permitted rung — so the
  enum-ownership cycle cannot reappear. Its §2.2 root-module gate is asserted
  against the declared set in **both** directions, so neither `ROOT_MODULES` nor
  this specification can move alone. It also carries the **§4.6 conventions-§7
  gates** (added by the run-003 MINOR-3 remediation): a view that uses
  `render_error`/`render_errors` has imported or defined it; a view owning an
  `if st.button(…)` imports a renderer; and **each** button handler body applies
  one, matching any `FrameworkError` subclass resolved from `core/errors.py`.
  Both are `ast` walks — importing `core.errors` would break §10.1's T0 tier, so
  the subclass closure is computed by parsing the class table instead.
  It also carries the **conventions-§7 press-coverage cross-reference** (run-005
  M1): `test_every_mutating_button_in_the_views_is_pressed_here` derives every
  `st.button` site in `views/` and every press declared in
  `tests/test_app_smoke.py` — across all five press idioms the smoke suite
  actually uses (keyed and labelled `parametrize` tables, a label dict passed
  through `sorted(…​.items())`, inline `app.button(key=…)` literals, `b.label ==
  …` and `b.label.startswith(…)` for the f-string Train label) — and asserts
  set equality **in both directions**, plus that each declaration sits in a body
  that calls `.click()`. No button inventory is duplicated inside the gate (both
  sides are tree-derived, roadmap deviation 0b.1), and three falsification
  controls cover an undeclared button, a dead press and an unwired declaration.
  `st.download_button` is out of scope: an export mutates nothing.
- `test_cli.py` (T0, new — ADR-012): the §4.9 exit-code contract, black-box over
  `subprocess`. `--help`, `-h`, `help` and an empty `argv` exit `0` with usage on
  stdout and **nothing on stderr**; `--non-existent-option` exits `0` but must
  *name* the token, print usage and point at `--strict`; `--strict` with the same
  token exits `2` with the message on stderr; a usage error inside a recognised
  command exits `2`; `--strict` is honoured in any position. `version` reports
  **every** pin parsed from `requirements.txt` (so a new pin cannot be silently
  omitted) and `info` reports the resolved bounds/policy and flags a `STF_*`
  override. `operators` lists the six registry keys and proves `operation_key`.
  **The error-reporting cases are the load-bearing ones:** `doctor` under
  `python -S` (no site-packages) must exit `1` and print the remedy, and `serve`
  must exit `1` with an actionable message instead of a traceback — asserted, plus
  a falsification test that reproduces the historical no-op stub and shows those
  two assertions reject it. A separate AST case pins `main.py` to stdlib + `core`
  at *any* scope, and the bare-interpreter cases (`python -S main.py --help /
  --version / info`) prove the T0 promise behaviourally.
- `test_reliability_parity.py` (T1, **bit-exact** — the one true numeric
  freeze): `_grade` boundaries (90/80/70/60/50 and the `<50 → "F"` fallthrough)
  and the four component scores at representative inputs match the
  pre-refactor values.
- `test_data_preprocessing.py` (T1): each missing-value strategy demonstrably
  changes the frame and emits **no** `FutureWarning`; no chained assignment.
  **Warning-emission profile measured on `pandas==2.3.3`, CoW off (ADR-011.9) —
  do not over-assert:** `fillna(method="ffill"/"bfill"/<var>, inplace=True)` emits
  `FutureWarning` today; `fillna(<stat>, inplace=True)` for
  mean/median/mode/custom emits **none** and mutates correctly. So the
  assertion is uniformly "no warning *after* the fix", never "a warning
  *before* the fix" for the statistic branches.
- `test_post_stress_styler.py` (T1, new — ADR-011.9): the Post-Stress
  vulnerability table renders with **no `FutureWarning`** on the pinned pandas.
  Requires `jinja2` (present transitively via `streamlit`, pinned in
  `requirements.txt`); skip cleanly if absent rather than erroring, so the suite
  stays runnable in a trimmed environment.
- `test_config.py` (T0 — no scientific imports needed, ADR-011.4):
  `get_config()` returns the identical object on two calls; `get_config({...})`
  returns a **new** object and does **not** mutate the singleton; nested override
  paths (`{"reliability": {...}}`) are rejected or explicitly supported —
  pin one; `AppConfig` is frozen.

### 10.3 Toolchain constraint (verified, must be honoured by Wave 0)

The repository's runtime dependencies are **not installed** in the working
environment, and the local interpreter is **Python 3.14.7** while the devcontainer
targets **3.11-bookworm** (`mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`).
There is no CI. Consequences:

- Wave 0 MUST establish a bootstrap step (a venv plus `pip install -r
  requirements.txt -r requirements-dev.txt`) and record the interpreter version
  used. PyPI is reachable and the pinned `numpy==2.4.2`, `pandas==2.3.3` and
  `scikit-learn==1.8.0` install cleanly on 3.14.
  **Re-confirmed in the third verification pass (ADR-011):** PyPI is reachable
  from this environment and `cp314` manylinux wheels exist for `pandas==2.3.3`;
  a throwaway venv on **Python 3.14.7** installed `numpy==2.4.2` + `pandas==2.3.3`
  + `jinja2` and ran every measurement cited in §4.4.1, §4.4.7, §6.1 and §10.2.
  The venv used for the parity fixtures should be the project's own `.venv`, not
  a throwaway.
- Until that bootstrap runs, **only T0 is executable**, so Wave 0's gate must be
  scoped to T0 plus the recorded fixtures, not the whole suite.
- The parity fixtures captured in Wave 0 are only comparable if produced and
  consumed on the same interpreter. Record the version alongside them.
- Nothing may assume `streamlit` is importable.
  **Corrected after execution:** this bullet asserted that "the app smoke test in
  Wave 5 is manual and cannot be automated in this environment". Both halves are
  false — streamlit installs cleanly from the pinned lockfile, and
  `streamlit.testing.v1.AppTest` ships with it. `tests/test_app_smoke.py`
  automates the nine-page script-level pass. What genuinely remains manual is
  *visual* verification, which `AppTest` cannot perform because it asserts on the
  element tree rather than on rendered pixels. See roadmap.md, Wave 5.

