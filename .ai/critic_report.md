# Codebase Critique — ML Model Reliability & Stress Testing Framework

**Date:** 2026-10-04
**Scope:** `app.py`, `modules/*.py` (8), `utils/*.py` (2), `requirements.txt`, `.devcontainer/`, `.gitignore`, `README.md`
**Method:** full manual read of all 7,244 LOC + dynamic verification in an isolated venv (numpy 2.5.3 / pandas 3.0.6 / sklearn 1.9.1 / plotly 7.1.0) + `pyflakes`.
**Note:** `.ai/project/{architecture,conventions,decisions,roadmap}.md` are unmodified placeholder templates. Architecture below is reverse-engineered from code.

---

## Executive Summary

The framework is a competent *prototype* and a dangerous *product*. The domain logic is well decomposed into eight readable classes, but a 3,517-line Streamlit monolith owns all orchestration, all state, and all business rules, and it silently swallows every failure it encounters.

Verification confirmed **eight defects that produce wrong numbers or silently do nothing** — the worst class of bug for a tool whose entire purpose is quantitative model assessment:

| # | Defect | Impact |
|---|---|---|
| 1 | `handle_missing_values` uses chained `inplace=True` — **imputation is a total no-op on pandas ≥ 3.0** while the UI prints "✅ Missing values handled successfully!" | Silent data corruption |
| 2 | `stress_module` mutates `df.values` in place — **all 6 perturbations are no-ops on mixed-dtype frames**, and **raise `ValueError` on pandas ≥ 3.0** | Silent stress-test fabrication |
| 3 | `len(hce_df)` on a 4-key dict used as an error count (`app.py:2945`, `3238`) | `hce_rate` = 4/n, measures dataset size not model quality |
| 4 | Reliability score assigns **12.5/25 neutral to missing components** | Unmeasured model (50.0/D) **outranks** a model measured as garbage (7.6/F) |
| 5 | Composite score redistributes missing weight to performance | Collecting **more** evidence *lowers* the score (90.00 → 86.25) |
| 6 | Target column is never label-encoded anywhere in the repo | String targets **crash** calibration; non-0..K-1 labels silently yield `avg_accuracy = 0.24` |
| 7 | Mean/median imputation runs on the **full dataset before** the split (`app.py:240` vs `:330`) | Textbook train/test leakage |
| 8 | `find_optimal_temperature` fits on the **test set** by default | Test-set overfitting; reported ECE is optimistic |

Security posture is weak: **arbitrary-path pickle write via path traversal** (`app.py:685`) and **arbitrary-code execution via `pickle.load`** (`model_module.py:372`), with `saved_models/` as a shared global directory and no locking. The devcontainer compounds this by launching Streamlit with `--server.enableCORS false --server.enableXsrfProtection false`.

Engineering discipline is absent: **zero tests, zero CI, zero lint config, zero logging**. `requirements.txt` is an unpinned-intent transitive dump (57 packages, including `altair`/`pydeck`/`numba`). The README's "Module Status: Complete" table for all 9 modules is not credible given the above.

---

## Overall Code Health Score: **3 / 10**

| Dimension | Score | Rationale |
|---|---|---|
| Correctness | 2/10 | 8 empirically confirmed wrong-result bugs, several silently so |
| Security | 2/10 | Path traversal + pickle RCE + XSRF/CORS disabled in devcontainer |
| Testability | 1/10 | 0 tests, 0 CI; core logic inseparable from `import streamlit` |
| Maintainability | 3/10 | Good docstrings, but 3,517-line monolith, 10-level nesting, 19 duplicated blocks |
| Architecture | 4/10 | Sensible class split below the UI; UI layer is a God-object |
| Readability | 6/10 | Consistent style, type hints present, genuinely clear method-level docs |

---

## Critical Issues

### C1. Silent arbitrary-path write via path traversal
`app.py:680-686` → `model_module.py:356-359`

```python
filename = st.text_input("Filename:", value=f"{save_model.lower().replace(' ','_')}.pkl")
if st.button("💾 Save Model"):
    filepath = os.path.join("saved_models", filename)   # raw free text
```
Verified:
```
input='../../../../tmp/pwned.pkl' -> 'saved_models/../../../../tmp/pwned.pkl'  escapes_dir=True
input='/etc/cron.d/pwn'           -> '/etc/cron.d/pwn'                        escapes_dir=True
```
`save_model` then calls `os.makedirs(os.path.dirname(filepath), exist_ok=True)` and `pickle.dump(...)`. No `os.path.basename`, no `werkzeug.utils.secure_filename`, no containment check. Any process can create directories and drop a pickle anywhere writable.

### C2. Arbitrary code execution via `pickle.load`
`app.py:702-719` → `model_module.py:371-372`

`model_files` is built from `os.listdir("saved_models")`, so the *name* is constrained — but the file *content* is not. `pickle.load(f)` executes on deserialization. No `weights_only=True`, no signature, no allowlist. The `try/except` at `app.py:716` catches the failure *after* the payload has already run.

### C3. Devcontainer disables Streamlit's XSRF and CORS protections
`.devcontainer/devcontainer.json`
```json
"postAttachCommand": {"server": "streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false"}
```
Combined with `"onAutoForward": "openPreview"` (Codespaces public tunnel) and C1/C2, this exposes CSRF-able arbitrary file write + RCE. The README's quick-start (`streamlit run app.py`) is safe by contrast — the two documented launch paths have opposite security postures.

### C4. Missing-value imputation is a silent total no-op (pandas ≥ 3.0)
`data_module.py:136-149`

```python
df_copy[column].fillna(df_copy[column].mean(), inplace=True)
```
Verified on pandas 3.0.6:
```
input : [1.0, nan, 3.0, nan, 5.0]
output: [1.0, nan, 3.0, nan, 5.0]   # nothing imputed
```
Yet `app.py:244` unconditionally renders `st.success("✅ Missing values handled successfully!")`. The user is affirmatively told the data was cleaned when it was not. Under the pinned `pandas==2.3.3` this works only via deprecated chained assignment (SettingWithCopyWarning); it is one pandas minor version away from silent failure, and the pin is the *only* thing preventing it.

### C5. All stress perturbations are silent no-ops on mixed-dtype frames
`stress_module.py:31-238`

Five of six methods do `result = X.copy(); values = result.values; values[:, i] = ...` and rely on `.values` aliasing the block. Verified: a numeric-only frame aliases (`shares_memory == True`), but a frame containing any object column yields a **detached object array**, so every perturbation is discarded and the *unmodified* frame is returned — the framework then reports "accuracy dropped 0.0%", a fabricated robustness result. Encoding is **optional** in Module 1, so this is reachable by normal use.

Only `feature_dropout` (`stress_module.py:105`) forces write-back via `result.iloc[:, :] = values * mask`.

On pandas ≥ 3.0 (Copy-on-Write) it is worse — a hard crash:
```
add_gaussian_noise    RAISES ValueError: assignment destination is read-only
add_uniform_noise     RAISES ValueError: assignment destination is read-only
scale_perturbation    RAISES ValueError: assignment destination is read-only
distribution_shift    RAISES ValueError: assignment destination is read-only
feature_corruption    RAISES ValueError: assignment destination is read-only
feature_dropout       OK
```

### C6. `len(dict)` used as an error count — reliability score is meaningless
`app.py:2941-2945` and `app.py:3235-3238`

`identify_high_confidence_errors` returns a **dict** (`utils/metrics.py:58-65`). Two call sites take `len()` of it:
```python
hce_df = identify_high_confidence_errors(dm.y_test, preds, probs, threshold=0.9)
hce_data[mn] = len(hce_df) / total_preds if total_preds > 0 else 0.0
```
Verified:
```
len(hce_df) = 4   (number of dict keys)
real count = 49
hce_rate   = 0.0125   (app.py)   vs   0.153125 (correct)
```
`hce_rate` is a function of dataset size, not model quality. It feeds `_confidence_score` → `score_all_models` (`app.py:2967`) → the Module 8 total, Module 9 gauges, and the exported PDF/JSON. The third call site (`app.py:1018`) uses `hce_df["count"]` correctly — the bug is a silent inconsistency between call sites.

### C7. Reliability scoring rewards missing data and inverts model ranking
`reliability_module.py:130-162`

Every unavailable component is assigned `MAX_x / 2` (12.5) "neutral". Verified:
```
well-measured + good      total= 90.12  grade=A+  missing=0
well-measured + GARBAGE   total=  7.62  grade=F   missing=0
NO DATA AT ALL            total= 50.00  grade=D   missing=4
```
A model with zero measurements outranks a model measured to be worthless. `build_summary_df` (`reliability_module.py:393`) and `plot_total_bar` (`:362`) both sort by `total`, so the ranking actively misleads. The `missing_components` field exists and *is* surfaced in the table, but the sort order and the gauge chart both present the inflated total as fact.

### C8. Composite score *decreases* as evidence accumulates
`comparison_module.py:278-283`

```python
missing = (0 if has_rob else 25) + (0 if has_cal else 25)
perf_bonus = perf * missing
total = perf_score + perf_bonus + rob_score + cal_score
```
Verified:
```
performance only              -> 90.00
performance + robustness + calibration -> 86.25
```
Running the stress tests and the calibration analysis — the framework's two headline features — *reduces* the score. `METRIC_WEIGHT`/`ROBUSTNESS_WEIGHT`/`CALIBRATION_WEIGHT` (`:17-19`) declare 0.5/0.25/0.25 but are dead; the weights are hardcoded as 50/25/25 literals.

### C9. Target column is never encoded — calibration crashes or silently misreports
`data_module.py:172` is the *only* `LabelEncoder` in the repo and it is applied to **features**, never to `target_column`. `split_data` (`data_module.py:250`) passes `df[target_column]` through raw.

`calibration_module.py:31-33, 79-81` assumes 0..K-1 integer labels:
```python
y_pred = np.argmax(probabilities, axis=1)   # yields 0..K-1
correct = (y_pred == y_true).astype(int)
...
if 0 <= int(label) < n_classes:            # int() on a str raises
```
Verified:
- `y_true = ['yes','no']*50` → `ValueError: invalid literal for int() with base 10: np.str_('yes')`
- `y_true = [1,2]*50` → **no exception**, `avg_accuracy = 0.24` (should be ~1.0); ECE/MCE/Brier all garbage

Because `app.py:2681` swallows that `ValueError` with `pass`, a string-labelled dataset produces a UI that reports "Calibration ECE not available" rather than "your target must be integer-encoded".

`utils/metrics.py:83` (`y_true_binary[i, label] = 1`) has no guard at all → `IndexError`.

---

## Major Issues

### M1. Seven silent `except Exception: pass` blocks, no logging anywhere
`app.py:2681, 2924, 2946, 2957, 3215, 3224, 3239`

```
except Exception:
    pass
```
No logger, no `st.warning`, no error accumulation. The repo has **zero** `import logging`, zero `logger`, zero `warnings.warn`. Every failure in the ECE/entropy/HCE gather loops is converted into "missing data", which the UI presents as a legitimate result. This is the mechanism that converts C6 and C9 from crashes into wrong numbers. `calibration_module.py:106` adds an eighth, substituting `float("nan")`.

### M2. Train/test leakage — imputation runs before the split
`app.py:240` (`handle_missing_values` on the full `raw_data`) precedes `app.py:330` (`split_data`). Mean/median/mode statistics are computed over train+val+test. For a framework whose purpose is trustworthy model evaluation, this invalidates every downstream metric. Scaling is correctly fit on train only (`data_module.py:198-221`) — the inconsistency shows the pattern was known and not applied throughout.

### M3. Test set used for calibration fitting and reporting
`app.py:2278`, `:2361`, `:2448`, `:2533` — Modules 6/7/8/9 default to `X_test`/`y_test`. `find_optimal_temperature` (`calibration_module.py:279`) grid-searches 50 temperatures to minimise ECE **on the test set**, then the same test set is used to report the resulting ECE. Optimistic bias baked into the headline calibration number.

### M4. 3,517-line monolith owns all state, all logic, and all rendering
`app.py` is 49% of all LOC. It contains 20 buttons, 27 selectboxes, 21 sliders, 41 `st.columns`, 38 `st.plotly_chart`, 60 `st.metric`, 14 download buttons. Max nesting depth is **10 compound levels** (deepest at `app.py:384→397→666→690→695→701→714→716→731→732`). Dispatch is a single flat `if/elif` chain of 9 branches keyed on **emoji-prefixed display strings** (`"4️⃣ Stress Testing"`) — renaming a nav label silently disables a module. There is no `else` fallback.

`st.tabs` bodies all execute on every rerun, so every widget and every computation in all 5 tabs of a module runs on any interaction anywhere in that module.

### M5. Zero caching; heavy work on every rerun
`st.cache_data` / `st.cache_resource`: **0 occurrences repo-wide**. Not button-gated, executed on every rerun:
- Full test-set inference in 6 separate loops: `app.py:2676, 2921, 2938, 3212, 3232` — `2921`/`2938` and `3212`/`3232` are **duplicate loops doing identical work twice per rerun**
- `add_batch_results` called 5× per rerun (`app.py:1809, 1934, 2020, 2125, 2184`)
- **`export_to_pdf` builds a full reportlab document on every rerun** (`app.py:3445`) even if the user never clicks Download; a reportlab failure renders a red error banner on every page load
- Module 9 recomputes the same 4 plots twice (`app.py:3286/3358, 3300/3365, 3292/3374, 3296/3383`)

`find_optimal_temperature` is a 50-iteration grid search, each iteration a full ECE computation (`calibration_module.py:285-291`).

### M6. Stale state is never invalidated
`data_loaded`, `data_prepared`, `model_trained`, `predictions`, `batch_stress_results*` are set to `True` and never reset. Loading a new CSV (`app.py:110`) overwrites only `data_manager.raw_data`, leaving `X_train/X_val/X_test`, `trained_models`, `metrics`, and all stress results from the **previous dataset** intact and still passing their gates. `PostStressAnalyzer.add_stress_result` (`post_stress_module.py:35-38`) only ever adds/overwrites — stale test names are never pruned. A user who re-splits or re-uploads silently gets metrics computed against the old split.

The prerequisite gates are booleans, not real preconditions: `app.py:1146` checks `data_manager is not None`, never `X_val is not None`, yet `app.py:1291` dereferences `dm.X_val`.

### M7. Results computed inside `if st.button(...)` vanish on the next rerun
`app.py:126-179` (validation report), `327-379` (split summary), `1814-1918` (robustness score + radar), `2119-2168`, `2188-2226`, `2274`, `2358`, `2444`, `2531`. Values are never written to `st.session_state`, so any widget interaction anywhere wipes the entire result panel. `calculate_robustness_score` returns early at `post_stress_module.py:64-65` **without** writing `robustness_scores[model_name]`, and `app.py:1826` then indexes it → `KeyError`. Reproduced directly:
```
return : 0.0
stored : {}
```

### M8. `UnboundLocalError` waiting to happen
`app.py:1206-1314` — `params` and `X_stressed` are assigned in six `if/elif` branches with **no `else`**, then consumed at `1301-1314`. Safe only because the selectbox currently offers exactly those six options. Any seventh option, or a desynced widget value, crashes.

### M9. Getters with write side-effects (SRP violation)
`post_stress_module.plot_robustness_radar:231` and `compare_model_robustness:332` both call `calculate_robustness_score`, mutating `self.robustness_scores`. Verified — calling the plot populates the cache. A rendering function that mutates shared analyzer state is untestable and order-dependent.

### M10. Scoring formulas and grade bands triplicated
- `_grade()` → grade→hex map: `reliability_module.py:23-36`
- `grade_color` dict: `reporting_module.py:254-261`
- Grade bands restated in user-facing prose: `app.py:2880-2884, 3039-3046, 3064-3069`
- ECE quality thresholds: `calibration_module.py:293-301` **and** inlined again at `reporting_module.py:92-100`
- Robustness severity bands: `post_stress_module.py:152-161` and `210-214` (two different band sets in the same file: 5/15/30 vs 10/20)
- Missing-data convention: 12.5 neutral (`reliability_module.py`) vs 0 + redistribution (`comparison_module.py`) — **mutually contradictory** (C7 vs C8)

Six independent definitions of "how good is this model". Any change requires finding all of them.

### M11. Label encoding applied to nominal features without warning
`data_module.py:169-174`. Verified: `['small','large','medium']` → `[2,0,1]`. Ordinal distance is fabricated, then fed to Logistic Regression, which assumes it is meaningful. The framework applies this to every `object` column the user selects and never distinguishes nominal from ordinal. It also does `astype(str)` (`:173`), converting `NaN` to the literal class `"nan"`, and silently discards the fitted encoders' consistency with `onehot` mode (`:177` records nothing).

### M12. `saved_models/` is a shared global directory with no isolation
Hardcoded relative literal `"saved_models"` at `app.py:685, 694, 748, 750` — CWD-relative, so it breaks when Streamlit is launched from anywhere but the repo root. No locking, no atomic write (no temp-file + rename), no collision handling: two concurrent sessions saving the same filename corrupt each other, and a crash mid-`pickle.dump` leaves a truncated `.pkl` that is offered in the load dropdown.

### M13. Unescaped model names interpolated into exported HTML
`reporting_module.py:431-498` builds HTML with f-strings; `_table` (`:419-429`) interpolates `row[k]` raw. `model_name` originates from a free-text `st.text_input` (`app.py:708`) and flows into `trained_models` keys → `report["performance"][i]["Model"]` → HTML and into `reportlab` `Paragraph` (`:569`), where `<` is markup. HTML/markup injection in the exported artifact.

---

## Minor Issues

| # | Issue | Location |
|---|---|---|
| m1 | `use_container_width=True` repeated **73×** instead of set once | `app.py` |
| m2 | Dead state `current_step` — assigned, never read | `app.py:58-59` |
| m3 | 19 distinct duplicated logic blocks (dataset radio+extract ×3, prereq gate ×3, `add_batch_results` ×5, ECE loop ×3, entropy loop ×2, model selectbox ×4, `predict_proba` guard ×4, …) | `app.py` |
| m4 | 20 unused imports (pyflakes) incl. `streamlit` in `post_stress_module.py` (imported, never used — yet it makes the module unimportable headless) | repo-wide |
| m5 | Dead local `drops` computed then discarded | `reporting_module.py:68-72` |
| m6 | Dead class attrs `ModelTrainer.models`, `ModelComparator.{METRIC,ROBUSTNESS,CALIBRATION}_WEIGHT` | `model_module.py:36`, `comparison_module.py:17-19` |
| m7 | Dead method `PostStressAnalyzer.compare_model_robustness` — never called | `post_stress_module.py:314` |
| m8 | Redundant inner imports of already-global `go` and `utils.metrics` symbols | `app.py:2039, 2931-2934` |
| m9 | `st.rerun()` **discards** the `st.success`/`st.write` it follows — user never sees the confirmation | `app.py:244-245, 281-283` |
| m10 | Broken literal `"### {} JSON"` (missing emoji) — renders as `### {} JSON` | `app.py:3416` |
| m11 | Markdown bullets indented 20 spaces inside a triple-quoted string → renders as a code block | `app.py:2206-2215` |
| m12 | `Styler.applymap` deprecated since pandas 2.1 (pinned 2.3.3) → `FutureWarning` each rerun | `app.py:1994` |
| m13 | Download filename built from the **live** widget (`dataset_choice`) not the stored value (`pred_data["dataset"]`) → mislabelled CSV after switching the radio | `app.py:858, 877, 907` |
| m14 | `current_step` in a 3-state mix: 2× `st.rerun()`, ~20× implicit rerun, no rule | `app.py` |
| m15 | Stale comments: `# OTHER MODULES (Placeholders)` (all implemented); `# shared model / dataset selector` (duplicated in all 4 tabs) | `app.py:383, 2251` |
| m16 | Return-type lies: `get_split_summary() -> Dict` returns `None`; `load_dataset` returns `None` on failure and `app.py:110` has no guard → uncaught traceback on malformed CSV | `data_module.py:283-291, 33-49` |
| m17 | `validate_dataset` divides by `len(df)` → `NaN` percentages on an empty frame, no error | `data_module.py:66` |
| m18 | `handle_missing_values` `drop` mutates the frame mid-loop → imputation of later columns depends on dict ordering | `data_module.py:131-151` |
| m19 | `calculate_robustness_score`: `np.std/np.mean` with `mean == 0` → `nan`; masked only by `max(0, min(nan, 1))` returning `0` by accident. Fragile, not correct. | `post_stress_module.py:95-99` |
| m20 | Stress-category classification is substring matching on display names (`"shift"`, `"scale"`, `"distribution"`) — a test named `rescale_x` is misfiled; keyword sets duplicated between `get_stress_type_summary:179-185` and `get_recommendations:433-468` | `post_stress_module.py` |
| m21 | `plot_robustness_heatmap` builds an O(models × types × rows) `next()` scan | `reporting_module.py:204-217` |
| m22 | `plot_reliability_gauge_row` puts one gauge subplot per model in a single row — unusable past ~4 models | `reporting_module.py:241-253` |
| m23 | Two brier-score implementations that disagree on guarding: `utils/metrics.py:68-85` (none) and `calibration_module.py:77-82` (range check) | — |
| m24 | Two confidence-binning implementations: `utils/metrics.get_confidence_bins` (unused) and `calibration_module.py:35-36` | — |
| m25 | Mixed `st.download_button` call conventions (positional vs keyword) in one file | `app.py:904` vs `:3406` |
| m26 | Identical widget label + different `min/max` + no `key` — latent `DuplicateWidgetID` | `app.py:453, 468` |
| m27 | Widget key built from raw CSV header: `key=f"missing_{col}"` | `app.py:225, 231` |
| m28 | `requirements.txt` is a 57-package transitive dump with `==` pins, no hashes, no lockfile, includes `altair`/`pydeck`/`slicer`/`numba`; no dev/test extras. One `pip install` away from an unresolvable set. | `requirements.txt` |
| m29 | No `pyproject.toml`/package metadata — modules are not importable as a package (no `__init__.py`), only as top-level dirs | repo-wide |
| m30 | README "Module Status: Complete" for all 9 modules is not credible | `README.md:19-31` |
| m31 | `.ai/project/*.md` are unmodified placeholders — no architecture, conventions, or ADRs exist to audit against | `.ai/project/` |

---

## Recommended Refactoring Targets

**Tier 1 — correctness & safety (do first, in this order)**

1. **`modules/stress_module.py`** — rewrite the DataFrame branch of all six perturbations to use `X.to_numpy(dtype=float).copy()` + `pd.DataFrame(...)` reconstruction. Removes both the silent no-op (C5) and the pandas-3 crash. Add a unit test asserting the output differs from the input for a mixed-dtype frame.
2. **`modules/data_module.py:117-151`** — replace chained `inplace=True` with `df_copy[column] = df_copy[column].fillna(...)`; resolve the strategy dict before mutating so `drop` ordering is deterministic; add an `if df.empty` guard to `validate_dataset`.
3. **`app.py:2945, 3238`** — `hce_df["count"]` instead of `len(hce_df)`; rename the variable to `hce_info`.
4. **`app.py:680-686` + `modules/model_module.py:345-359`** — `secure_filename()`, force `os.path.basename`, resolve and assert `Path(fp).resolve().is_relative_to(MODEL_DIR)`, write to a temp file then `os.replace`.
5. **`modules/model_module.py:363-375`** — replace `pickle` with `joblib`, or at minimum `weights_only=True`; add a documented trust boundary.
6. **`.devcontainer/devcontainer.json`** — drop `--server.enableCORS false --server.enableXsrfProtection false`.
7. **Label-encode the target in `DataManager.split_data`** (or in Module 1 tab 5) and thread `classes_` through `CalibrationAnalyzer`; replace `argmax == y_true` with a label→index map. Fixes C9.
8. **Reorder Module 1** so imputation is fit on train only — or explicitly document the leakage and rename the tab.

**Tier 2 — stop the bleeding on error handling and state**

9. **Introduce `logging`**; replace all 7 `except Exception: pass` with `except Exception: logger.exception(...)` + a visible `st.warning` that marks the value as unavailable. Do not present a swallowed failure as "no data".
10. **Create a single `AppState` dataclass** in one module; initialize every key once (removes `hasattr` soup, `current_step`, and M6). Add explicit `reset()` on new dataset / re-split / re-encode, and make gates assert real preconditions (`dm.X_val is not None`) rather than booleans.
11. **Persist every computed result to session state** at the point of computation, not inside the button branch (M7).
12. **Add `else: raise` to the `if/elif` stress dispatch** at `app.py:1200-1312` and `stress_module.py:316-329`; make `batch_stress_test` reject unknown `stress_type` instead of `continue`.

**Tier 3 — structural**

13. **Split `app.py` into `ui/module_01_data.py` … `ui/module_09_reports.py`** behind a registry. One `elif` branch → one function call. Target ≤ 400 LOC per module. This alone removes the 10-level nesting and makes 12 of the 19 duplicated blocks extractable.
14. **Extract the 6 duplicated helpers** the duplication audit found: dataset radio+extract, prerequisite gate, `add_batch_results` + select model, ECE gather loop, entropy/HCE gather loop, `predict_proba` capability guard.
15. **Consolidate scoring into one module.** `reliability_module` owns grade bands, ECE quality bands, severity bands, and the missing-data policy; `comparison_module` and `reporting_module` import them. Resolve the 12.5-vs-0 contradiction — recommend **renormalising over available components** and refusing to rank a model with < 2 components, which fixes C7 and C8 together.
16. **Add caching at the boundary:** `@st.cache_data` on ECE/entropy/HCE computation keyed on `(model_id, split_id)`, `@st.cache_resource` on fitted models. Move `export_to_pdf` behind its download button.
17. **Extract pure logic from the Streamlit layer.** `post_stress_module`, `reporting_module`, `comparison_module`, and `reliability_module` import `streamlit`/`plotly` for presentation only; the scoring/aggregation methods are pure and should be testable without either. (`post_stress_module.py:12` imports `streamlit` and never uses it — proof the coupling is accidental.)
18. **Establish the test baseline:** `pytest` covering `utils/metrics.py`, the six perturbations, ECE against a hand-computed fixture, Brier against `sklearn`, and score monotonicity (more/better data must never lower a score). Add `.github/workflows/ci.yml` running `pytest` + `ruff` + `mypy`.
19. **Replace `requirements.txt`** with a `pyproject.toml` declaring direct dependencies only, a `uv.lock`/`requirements.lock` for the resolved set, and a `pandas>=2.2` floor with an explicit CoW-safe implementation.
20. **Populate `.ai/project/*.md`** with the real architecture, conventions, and ADRs — there is currently no documented standard for this codebase to be held to.

---

## Verification Appendix

Environment: `numpy 2.5.3`, `pandas 3.0.6`, `scikit-learn 1.9.1`, `plotly 7.1.0`, `streamlit 1.65.0`. Repo pins `pandas==2.3.3`; findings C4/C5 are version-conditional and marked as such. `pyflakes` clean-run listed in m4. Path-traversal, no-op, crash, `len(dict)`, score-inversion, and label-encoding behaviours were each reproduced by direct execution, not inferred.
