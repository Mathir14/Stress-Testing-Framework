# ML Model Reliability and Stress Testing Framework

A Streamlit-based framework for evaluating classification models across performance, confidence, calibration, stress robustness, reliability scoring, and report generation.

## What Is Implemented

The app currently exposes **9 modules** in the sidebar. These labels are the
persisted session contract and are reproduced here verbatim, emoji and `&`
included — the same strings `views.base.PAGE_KEYS` freezes:

1. 1️⃣ Data Management
2. 2️⃣ Baseline Modeling
3. 3️⃣ Prediction & Confidence
4. 4️⃣ Stress Testing
5. 5️⃣ Post-Stress Evaluation
6. 6️⃣ Calibration Analysis
7. 7️⃣ Model Comparison
8. 8️⃣ Reliability Scoring
9. 9️⃣ Visualization & Reports

## Module Status (Current)

| Module | Status | Core capabilities |
|---|---|---|
| 1️⃣ Data Management | Complete | CSV upload, validation, cleaning, encoding, scaling, train/val/test split |
| 2️⃣ Baseline Modeling | Complete | Logistic Regression, Random Forest, XGBoost, evaluation, confusion matrix, feature importance, attested save/load |
| 3️⃣ Prediction & Confidence | Complete | Predictions, confidence analysis, high-confidence errors, calibration/entropy analysis |
| 4️⃣ Stress Testing | Complete | Single and batch perturbation tests (noise, dropout, corruption, scaling, distribution shift) |
| 5️⃣ Post-Stress Evaluation | Complete | Robustness score, vulnerability analysis, stress-type summary, recommendations |
| 6️⃣ Calibration Analysis | Complete | Reliability diagrams, calibration metrics, per-class analysis, temperature scaling |
| 7️⃣ Model Comparison | Complete | Cross-model performance tables/charts, robustness and calibration comparison, best-model recommendation |
| 8️⃣ Reliability Scoring | Complete | Unified 0-100 reliability score with component breakdown, grades, per-model recommendations |
| 9️⃣ Visualization & Reports | Complete | Unified dashboard and export to CSV/JSON/HTML/PDF |

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py          # or: python main.py serve
```

Open the URL shown by Streamlit (usually http://localhost:8501).

### 3) Run the tests (optional, requires `-r requirements-dev.txt`)

```bash
python -m pytest tests/
```

## Command Line

`main.py` is the project's terminal entry point, for the three things a browser
page cannot do: report the resolved configuration, answer a script with an exit
code, and launch the app. It needs nothing but the standard library for
everything except `operators` (needs numpy) and `serve` (needs streamlit).

```bash
python main.py --help          # usage
python main.py doctor          # is this checkout runnable? exits 1 if not
python main.py info            # the resolved config: bounds, weights, artifact policy
python main.py version         # interpreter + every requirements.txt pin vs installed
python main.py operators       # the six perturbation registry keys and UI labels
python main.py serve --port 8501
```

| Exit | Meaning |
|---|---|
| `0` | the work was done, **or** an unrecognised argument was reported and ignored |
| `1` | the work failed (missing dependency, unmet pin, unreadable manifest) — the message names the remedy |
| `2` | usage error inside a command, or any unrecognised argument when `--strict` is passed |

Unrecognised arguments are non-fatal by default on purpose: a script probing the
entry point should get an answer, not a traceback. Pass `--strict` for the
conventional fatal treatment — recommended in CI. The contract is specified in
architecture.md §4.9, recorded in ADR-012, and enforced by `tests/test_cli.py`.

Run `doctor` from the project `.venv`. Under a bare system interpreter it
correctly reports the missing runtime pins and exits `1`.

## Recommended Workflow

1. Use Module 1 to prepare data.
2. Train and evaluate models in Module 2.
3. Inspect confidence behavior in Module 3.
4. Run perturbation tests in Module 4.
5. Review robustness findings in Module 5.
6. Analyze and improve calibration in Module 6.
7. Compare all trained models in Module 7.
8. Use reliability scoring in Module 8.
9. Export consolidated outputs in Module 9.

## Project Structure

```text
Mini Project/
|- app.py                        # composition root: sidebar + page routing
|- main.py                       # CLI entry point: doctor / info / version / operators / serve
|- requirements.txt              # fully pinned runtime lockfile
|- requirements-dev.txt          # test tooling (pytest); must stay T0-safe
|- core/                         # framework layer: no streamlit, no UI
|  |- errors.py                  # typed exception hierarchy
|  |- config.py                  # frozen dataclasses; single owner of all tunables
|  |- logging_config.py          # logging setup from config
|  |- state.py                   # session-state keys + AppContext accessors
|  |- model_io.py                # model-artifact security boundary (pickle)
|  |- validation.py              # parameter/frame validation, owns the bounds table
|  |- perturbations.py           # pure perturbation kernel, injected RNG
|  |- scoring.py                 # single owner of the scoring component transforms (ADR-021)
|  |- reporters.py               # Reporter protocol + null/collecting implementations
|- modules/                      # domain logic; no streamlit imports
|  |- data_module.py
|  |- model_module.py
|  |- stress_module.py
|  |- post_stress_module.py
|  |- calibration_module.py
|  |- comparison_module.py
|  |- reliability_module.py
|  |- evidence_module.py          # collect_evidence: the single stress/ECE evidence reader
|  |- reporting_module.py       # ReportGenerator: compile + CSV/JSON/HTML/PDF export
|- views/                        # one module per sidebar page; Streamlit only here
|  |- base.py                    # build_context / render_sidebar / render_status_footer
|  |- module_01_data.py ... module_09_reports.py
|  |- reporters.py               # StreamlitReporter; render_error / render_errors
|- utils/
|  |- metrics.py
|  |- plotting.py
|- tests/                        # pytest suite (see below)
`- saved_models/                 # local model artifacts
```

Layering is enforced by `tests/test_architecture.py`: `core` may not import
`streamlit`, `modules` may not import `streamlit` or `views`, and `core.config`
(rung 1) may not import any project-local module.

### Tests

```bash
python -m pytest tests/            # full suite
```

The suite is tiered (architecture.md §10.1):

| Tier | Needs | Files |
|---|---|---|
| **T0** | pytest only | `test_architecture.py`, `test_config.py`, `test_model_io.py`, `test_cli.py` |
| **T1** | numpy/pandas/plotly/streamlit | the remaining files, including the `AppTest` smoke test |

T1 modules declare each third-party root with `pytest.importorskip` **before**
they import it (ADR-026), so a bare `pytest`-only run collects T0 and skips T1
with zero collection errors rather than aborting on the first missing import.

The architecture gate also asserts the *tree* shape, not just the import edges:
the top-level modules are exactly the set architecture.md §2.2 declares
(`app.py` and `main.py`, cross-checked in both directions so neither the test
constant nor the specification can move alone), the nine page `render()`
callables are annotated `ctx: AppContext` (conventions §2), and the 64 widget
`key=` literals are frozen per view in **both** directions — a new, undeclared key
fails just as a renamed one does. Interpolated `key=f"..."` families are frozen
the same way from `widget_key_templates_by_view`, because a literal scan alone is
blind to them.

It also enforces two conventions that behavioural tests provably cannot see:
deserialization is confined to `core/model_io.py` (so a second `pickle.load`
caller cannot skip attestation, containment and consent), and no project file
passes `inplace=True` or the removed `fillna(method=)` (conventions §5). Both are
`ast` walks rather than text searches, because each banned spelling is documented
in the file that bans it.

The T0 tier is a real constraint, not a formality — it is the tier the security
and layering gates depend on, and it must be runnable in a virtualenv without the
scientific stack:

```bash
python -m venv /tmp/t0 && /tmp/t0/bin/pip install -r requirements-dev.txt
/tmp/t0/bin/python -m pytest tests/test_architecture.py tests/test_config.py tests/test_model_io.py tests/test_cli.py
```

`tests/fixtures/reliability_baseline.json` freezes the module 8 scores bit-exactly
against the interpreter version recorded inside the fixture; regenerate and review
it deliberately rather than as part of an unrelated change
(`python -m tests.capture_baseline`). To *verify* it — the check the suite's
parity test now fails closed on rather than skipping — run
`python -m tests.capture_baseline --check`, which exits `1` on any drift.

## Artifacts and Git Tracking

Generated/testing artifacts are intentionally excluded from version control:

- `saved_models/` (trained model binaries)
- `figures/` (paper/testing output images)
- personal/local datasets (for example `synthetic_mobile_sales_2025.csv`)
- paper-specific helper scripts and document files

These are controlled via `.gitignore`.

## Security Notes

- **Loading a model executes arbitrary Python.** `.pkl` files are pickled
  object graphs, so `pickle.load` runs whatever the file contains. Loading is
  gated in `core/model_io.py`: a file must be explicitly, separately attested as
  trusted by the operator, must sit inside `saved_models/` with a sanitised
  name, must be within the size cap, and must match its recorded SHA-256 before
  it is unpickled. The load is **read-once and hash-bound** — the buffer that is
  verified is the buffer that is deserialised, never a re-opened path — so there
  is no window between the check and the unpickle. Only load artifacts you
  produced yourself.
- **Escaping is not optional.** All HTML and PDF exports pass through
  `html.escape`, so model names and dataset contents cannot inject markup into a
  generated report.
- **Parameter validation happens before any work.** Stress parameters are range-
  checked against the single bounds table in `core/config.py` before a
  perturbation runs, and untrusted frames are rejected rather than coerced.

## Key Dependencies

- streamlit
- scikit-learn
- xgboost
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- reportlab

`matplotlib`, `seaborn` and `joblib` are pinned even though the app does not
import them directly: they are transitive requirements of streamlit / plotly /
scikit-learn and this file is a lockfile. Unused direct pins (`shap`) were
removed after a reverse-dependency check confirmed nothing imports or requires
them (ADR-007).

## Notes

- This project is focused on classification workflows with probability outputs.
- Some calibration/reliability views require models that support `predict_proba`.
- Robustness and reliability views are most informative after batch stress tests are run.
