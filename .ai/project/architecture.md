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
`modules/data_module.py` (17 refs) and `modules/model_module.py` (2 refs) touch
`st.*`. Seven of the nine domain modules are already Streamlit-free. The
decoupling work is therefore *narrow* (2 files), not repository-wide.

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

No module may import `app.py`. Circular imports are a test failure.

### 2.2 Target directory layout

```
app.py                       # composition root, <=200 lines
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
  test_perturbations.py
  test_validation.py
  test_model_io.py
  test_data_preprocessing.py
  test_reliability_parity.py
requirements-dev.txt          # pytest (+ pytest-cov), pinned
```

`__init__.py` MUST be added to `core/`, `views/`, `tests/`, `modules/`, `utils/`.
Empty content is acceptable.

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
```

### 4.2 `core.config`

```python
@dataclass(frozen=True)
class StressBounds:
    noise_level: tuple[float, float]      # (0.0, 5.0)
    noise_range: tuple[float, float]      # (0.0, 1.0)
    dropout_rate: tuple[float, float]     # (0.0, 1.0) exclusive upper
    corruption_rate: tuple[float, float]  # (0.0, 1.0) exclusive upper
    scale_factor: tuple[float, float]     # (1.0, 10.0) exclusive lower
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
    )

@dataclass(frozen=True)
class ArtifactPolicy:
    root_dir: Path = Path("saved_models")
    allowed_suffixes: tuple[str, ...] = (".pkl",)
    filename_pattern: str = r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pkl$"
    require_attestation: bool = True

@dataclass(frozen=True)
class AppConfig:
    stress_bounds: StressBounds
    reliability: ReliabilityWeights
    artifacts: ArtifactPolicy
    log_level: str = "INFO"
    random_seed: int | None = None

def get_config(overrides: Mapping[str, Any] | None = None) -> AppConfig: ...
```

**Parity rule:** every default above reproduces current behaviour exactly. The
reliability constants mirror `modules/reliability_module.py:25-51,66,75`. Changing
a default is a behaviour change and requires a matching test update plus an ADR.

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
existing UI path can be rejected.

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
CORRUPTION_TYPES: tuple[str, ...] = ("zero", "mean", "random", "extreme")
SHIFT_TYPES: tuple[str, ...] = ("mean", "variance")

def available_operations() -> tuple[str, ...]: ...
def apply_perturbation(X, op: str, params: Mapping[str, Any] | None = None, *,
                       rng: np.random.Generator | None = None,
                       bounds: StressBounds | None = None):
    """Pure. Never mutates X. DataFrame in -> DataFrame out (index, columns and
    column order preserved); ndarray in -> ndarray out. Non-finite output raises."""
```

Contract details (must be honoured by every operator):

1. Operators receive a 2-D float ndarray of values only; container handling lives
   in `apply_perturbation`. This removes the 12 duplicated DataFrame/ndarray
   branches in `modules/stress_module.py:20-238`.
2. Zero-variance columns: when `std == 0` the noise scale is `0`, never `NaN`.
   `1 / scale_factor` must never be evaluated with `scale_factor == 0`.
3. Semantics parity with today: dropout multiplies by a Bernoulli mask
   (`rng.random(shape) > rate`); corruption replaces selected cells with
   zero / column mean / uniform(min,max) / random choice of (min,max); scale
   picks `factor` or `1/factor` per column via `rng.random() > 0.5`; mean shift
   adds `std * amount`; variance shift maps `mean + (x - mean) * amount`.
4. Every operator receives an explicit `np.random.Generator`; `np.random.*`
   module-level calls are banned inside `core/` (test-enforced).

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
5. Loaded objects must satisfy `isinstance(model, sklearn.base.BaseEstimator)`;
   otherwise `ArtifactIntegrityError`.
6. Residual risk is documented in the ADR, the module docstring and README.

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
  `st.caption(exc.context)` when context is present.

## 6. Data-Flow Invariants

1. **Purity of stress operators** — `add_gaussian_noise`, `add_uniform_noise`,
   `feature_dropout`, `feature_corruption`, `scale_perturbation`,
   `distribution_shift` never mutate their argument (fixes MAJ-002 and the
   silent no-op in the current `DataFrame.values` branch, which discards writes
   on mixed-dtype frames).
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
| XSS in exported HTML | `html.escape` at every interpolation | `modules/reporting_module.py` |
| Silent corruption of user data | Purity contract + `df[col] = df[col].method()` instead of chained `inplace=True` | `core/perturbations.py`, `modules/data_module.py:136-149` |

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

Intentional behaviour changes (the only ones): `DataManager.load_dataset` raises
`DatasetLoadError` instead of returning `None` after `st.error`; all six
perturbation methods are pure and reject out-of-range parameters; model I/O is
attested.

## 10. Test Strategy

`tests/` uses pytest. Priority order: security (`test_model_io.py`),
correctness of perturbations (`test_perturbations.py`, `test_validation.py`),
data preprocessing (`test_data_preprocessing.py`), numeric parity
(`test_reliability_parity.py`), and the import-layer contract
(`test_architecture.py`).

`test_architecture.py` and `test_model_io.py` MUST run with **stdlib + pytest
only** (no numpy/pandas/sklearn import), so the security and layering gates stay
runnable in a bare environment.

Minimum required cases:

- `test_model_io.py`: rejects `../evil.pkl`, `/etc/passwd.pkl`, `a/b.pkl`,
  `..%2f`-style names, non-`.pkl` suffixes, symlinked targets; refuses an
  unmanifested file; detects single-byte tampering; `trust=True` path works;
  non-estimator payload rejected; manifest is written atomically and is
  re-readable after a simulated crash (leftover `.tmp` ignored).
- `test_perturbations.py`: input array/frame is byte-identical after each
  operator; mixed-dtype DataFrame actually changes (regression test for the
  current silent no-op); zero-variance column yields no NaN; `scale_factor=0`
  and `dropout_rate=1.0` rejected; identical results for a fixed
  `np.random.default_rng(seed)` across runs.
- `test_reliability_parity.py`: `_grade` boundaries (90/80/70/60/50) and the
  four component scores at representative inputs match the pre-refactor values.
