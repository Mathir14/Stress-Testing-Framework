# Codebase Critique — Dashboard (Module 9: Visualization & Reports)

**Date:** 2026-10-04
**Scope:** `app.py:3181-3517` (Module 9 dispatch), `modules/reporting_module.py` (599 LOC, entire file)
**Method:** full static read of the dashboard slice + cross-module trace of every value it consumes
(`reliability_module.py`, `model_module.py`, `calibration_module.py`, `utils/metrics.py`, Module 8's
parallel computation block at `app.py:2915-2969`).
**Environment limitation:** runtime verification was **not** possible — the system `python3` has no
`numpy`/`pandas`/`plotly` and there is no venv in the repo. All findings below are derived from source
reading and cross-file call-graph tracing. Claims are marked `[static]` accordingly; none are marked
`[verified]` as in the prior repo-wide pass.
**Prior art:** `.ai/critic_report.md` (repo-wide, 2026-10-04) covers the whole codebase. This document
is the **dashboard-focused** audit and deliberately supersedes/refines items M5, M13, m5, m10, m21, m22
from that report. Findings here that are **new** are marked **[NEW]**.

---

## Executive Summary

Module 9 is the product's front door and its single source of exported truth. Structurally it is the
cleanest part of the codebase: a pure `compile_report()` that turns five module objects into one dict,
then five thin plot methods, then five exporters. That decomposition is correct and should be preserved.

The implementation, however, fails at the one job a dashboard must do: **tell the truth about the
data underneath it.** Three independent classes of defect compound here.

**1. It manufactures authoritative-looking numbers from partial data.** Three bare
`except Exception: pass` blocks (`app.py:3215, 3224, 3239`) convert any per-model failure into
"no data", which the scorer then fills with a neutral 12.5/25 and the dashboard renders as a confident
gauge and a filled radar polygon. There is no path by which a user learns a model was skipped. The
dashboard also ignores the user's own `error_threshold` setting and hardcodes `0.9`, so the Confidence
component on the dashboard is not the Confidence the user configured in Module 3.

**2. Its visual constants have silently forked from their source of truth.** `plot_radar_all` hardcodes
`maxvals = [25, 25, 25, 25]` (`reporting_module.py:337`) instead of reading `ReliabilityScorer.MAX_*`,
and `ReliabilityScorer.plot_component_radar` gets this *right* (`reliability_module.py:288`). The
dashboard holds the defective copy. Its radar palette is 5 entries against the scorer's 7, so from the
6th model onward two models share a colour. Its gauge background bands (`[0,50],[50,70],[70,90],[90,100]`)
do not match `_grade()`'s bands (`50/60/70/80/90`). **[NEW]** None of these fail loudly — they
misrepresent data.

**3. It does ~9 units of redundant heavy work on every single rerun.** Streamlit re-executes the whole
script per interaction. Module 9 alone builds **9 Plotly figures** (4 of them exact duplicates across
Tab 1 and Tab 3, because `st.tabs` bodies all run) and invokes **9 exporters** (including a full
reportlab PDF build) unconditionally, before the user has clicked a single download button. There are
**zero** `@st.cache_data`/`@st.cache_resource` calls in the entire repository.

On top of that, the HTML exporter interpolates untrusted model names into markup with **no escaping
anywhere in the repository**, producing a self-contained, browser-openable, shareable artifact — and
model names are auto-populated from **filenames on disk**, so a crafted `.pkl` filename propagates
script injection into a published report without the user typing a character. **[NEW]**

The dashboard also contradicts the modules it aggregates: its "🏆 Best Performance" (max F1) and
"🛡️ Most Robust" (min mean drop) are independently invented definitions that can name different
"winners" than Module 5 and Module 7 display on the same dataset. **[NEW]**

---

## Overall Code Health Score: **4 / 10**

| Dimension | Score | Rationale |
|---|---|---|
| Correctness / data integrity | 3/10 | Silent skips become neutral scores; forked visual constants; heatmap clips real values; KPI contradicts its own table |
| Security | 3/10 | Unescaped HTML export of disk-derived strings into a shareable artifact |
| Performance | 2/10 | 9 figures + 9 exporters (incl. PDF) per rerun, zero caching repo-wide |
| Maintainability | 4/10 | Clean compile/plot/export split, but ~40 LOC duplicated from Module 8 and two bad copies of scorer visuals |
| Testability | 1/10 | 0 tests, 0 CI; `reporting_module.py` is pure and trivially testable, yet untested |
| Readability | 7/10 | Genuinely the most readable file in the repo; typed, docstringed, consistent |

The score is one point above the repo-wide 3/10 solely because the module's *structure* is sound. The
*behaviour* is not.

---

## Critical Issues

### C1. Silent `except Exception: pass` converts model failures into fake mid-range scores **[NEW severity]**
`app.py:3211-3216`, `3222-3225`, `3231-3240`

```python
for mn in trainer.trained_models:
    try:
        _, probs = trainer.predict(mn, dm.X_test)
        cm_r = cal_an.compute_calibration_metrics(dm.y_test, probs)
        rep_cal_ece[mn] = cm_r["ece"]
    except Exception:
        pass
```

Three of these loops run on every visit to the dashboard. A model without `predict_proba`
(`model_module.predict` at `model_module.py:111` calls it unconditionally — an `SVC` without
`probability=True` raises `AttributeError`), a calibration `ValueError` from un-encoded string labels
(the prior report's C9), or a shape mismatch **all** vanish with no log, no warning, no UI marker.

The consequence is worse than a crash, because of what happens downstream:

- the model is missing from `rep_cal_ece` → `score_all_models` receives `ece=None`
  (`reliability_module.py:230`) → `_calibration_score` is never called → the component falls to the
  `MAX_CAL / 2` neutral branch (`reliability_module.py:142`) → **12.5/25**
- same for entropy and HCE → Confidence gets `MAX_CONF / 2` neutral (`reliability_module.py:161`)
- the dashboard then renders a full green-ish gauge and a filled radar polygon for a model that was
  **never measured**

The user's only signal is a number sitting mid-scale, which reads as "measured, mediocre". A tool
whose purpose is trustworthy assessment is reporting untested models as average. Contrast
`app.py:3454`, which *does* surface PDF failures — the inconsistency proves the pattern was known.

**Evidence chain `[static]`:** `app.py:3215` → `reliability_module.py:230` → `reliability_module.py:142`
→ `reporting_module.py:262-283` (gauge) and `reporting_module.py:341-359` (radar).

### C2. HTML export interpolates untrusted, disk-derived strings with zero escaping **[NEW escalation]**
`reporting_module.py:419-429`, `:456`, `:461-478`

```python
def _table(rows):
    keys = list(rows[0].keys())
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{row[k]}</td>" for k in keys) + "</tr>"
```

`grep -rn "html.escape" modules/ utils/ app.py` → **zero matches in the repository.** Raw f-string
interpolation into a `<td>`, into the `<title>`, and into the KPI grid.

Taint path `[static]`:
```
app.py:706-712   st.text_input("Model name:", value=selected_file.replace(".pkl","").replace("_"," ").title())
                 ^^^^^^^^^^^^^^ DEFAULT VALUE IS THE FILENAME ON DISK
app.py:717-718   model_trainer.load_model(model_name_input, filepath)
model_module.py  trained_models[model_name] = ...          # user string is the dict key
app.py:3210      for mn in trainer.trained_models           # dashboard iterates it
reporting_module.py:56   "Model": m                          # into report["performance"]
reporting_module.py:109  "Model": m                          # into report["reliability"]
reporting_module.py:419-429                                 # into the HTML, unescaped
```

This is worse than the prior report's M13 for three reasons:

1. **No user input is required.** The model name is *prefilled from a `.pkl` filename* in the
   `saved_models/` directory. A crafted filename is sufficient; the victim clicks "Load Model".
2. **The artifact is designed to be shared.** `app.py:3430` markets it as a "Styled, self-contained
   HTML page" and it is the deliverable a user attaches to a paper or publishes. It executes in the
   reader's browser with no sandboxing and same-origin access to nothing — but full script execution,
   credential-phishing overlays, and defacement are all available.
3. **`summary['best_performance']` is a second injection point** (`:469`,
   `max(perf_rows, key=...)["Model"]`) — the same tainted string, rendered outside any table, where a
   reviewer is even less likely to look.

The `reportlab` `Paragraph` path (`:544`, `:569`) is a related markup-injection vector; the prior
report covers it. `html.escape` on every interpolation, plus a filename allowlist, is the fix.

### C3. `plot_radar_all` hardcodes the scorer's weights instead of reading them **[NEW]**
`reporting_module.py:336-346` vs `reliability_module.py:47-50, 288`

```python
# reporting_module.py:336-337  — the DASHBOARD copy
cats   = ["Performance", "Calibration", "Robustness", "Confidence"]
maxvals = [25, 25, 25, 25]

# reliability_module.py:288     — the CORRECT copy
max_vals = [self.MAX_PERF, self.MAX_CAL, self.MAX_ROB, self.MAX_CONF]
```

`MAX_PERF/CAL/ROB/CONF` are the declared class attributes (`reliability_module.py:48-51`) and every
other consumer reads them. The dashboard holds a frozen copy of today's values. Re-tune the weights —
the single most likely change to this codebase — and the dashboard radar keeps dividing by 25 and
**silently plots the wrong shape** while Module 8's radar is correct. Two views of the same score
disagreeing, with no error.

The radar is also the only place the four components are directly comparable, so this is the chart a
user is most likely to screenshot into a report.

### C4. `use_container_width` deprecation + unbounded subplot count in the gauge row
`reporting_module.py:241-253`, all 8 `st.plotly_chart` calls in Module 9

`requirements.txt` pins `streamlit==1.54.0`, where `use_container_width` is deprecated in favour of
`width`. It appears **73×** in `app.py` (8× inside Module 9 alone), so every dashboard render carries
deprecation noise. Separately, `plot_reliability_gauge_row` places one `indicator` subplot per model
in a **single row**; at 6+ models each gauge is unreadable and Plotly's per-subplot `domain` maths
produces overlapping titles. There is no wrap, no cap, and no fallback to a bar chart.

---

## Major Issues

### M1. ~9 redundant heavyweight computations on every rerun, with zero caching **[NEW accounting]**
`app.py:3181-3499`

`grep -c "cache_resource\|@st.cache" app.py` → **0**. Streamlit re-runs the entire script on every
widget interaction — including the `rep_dataset` selectbox (`:3201`), every tab click, and every
download-button press. Within Module 9 alone, per rerun:

| Work | Count | Lines |
|---|---|---|
| `trainer.predict` (full `predict` + `predict_proba`) | 2× per model | `:3212`, `:3232` |
| Plotly figures built | 9 | 5 unique + 4 exact duplicates |
| `export_to_json` | 2 | `:3418`, `:3493` |
| Exporters invoked (all unconditional) | 9 | incl. **full reportlab `doc.build`** at `:3445` |
| `datetime.now()` | 1 | `reporting_module.py:140` |

The ECE loop (`:3210-3216`) and the entropy/HCE loop (`:3230-3240`) call `trainer.predict(mn, X_test)`
with **identical arguments** and discard all but different outputs — a textbook case for computing once.

The figure duplication is pure waste: Tab 3 "Charts Gallery" re-renders the same four charts Tab 1
already built, under different `key=` values (`gal_perf` vs `rep_perf`, etc.). Because `st.tabs`
executes all bodies, all 9 figures are constructed even though the user sees one tab. The exports are
worse: a complete PDF document is laid out on every page load whether or not anyone clicks Download.

The fix is mechanical and high-value: compute the ECE/entropy/HCE payload **once**, cache it with
`@st.cache_data` keyed on `(model_id, split_id)`, build each figure once and reuse the object in both
tabs, and move every exporter behind its `st.download_button`.

### M2. The dashboard re-derives Module 8's scores independently — the two can disagree **[NEW]**
`app.py:2915-2969` (Module 8) vs `app.py:3207-3250` (Module 9)

The two blocks are near-identical clones: same ECE loop, same entropy/HCE loop, same
`threshold=0.9`, same `n_classes` derivation, same three silent `except: pass`. Roughly 40 duplicated
LOC `[NEW detail — the prior report logged this only as "ECE loop ×3, entropy loop ×2"]`.

The hazard is not just maintenance. Because each module swallows exceptions **independently**, the
identical computation can succeed in Module 8 and fail in Module 9 (different local state, different
exception timing). The user then sees two different reliability totals, two different grades, and two
different gauges for the same model on the same data, with no indication that either is unreliable.
Extract one `compute_reliability_inputs()` and have both modules call it.

### M3. The dashboard ignores the user's configured `error_threshold` **[NEW]**
`app.py:3236` vs `app.py:1000-1016`

Module 3 exposes a real control:

```python
error_threshold = st.slider("Minimum Confidence for Errors", 0.5, 1.0, 0.8,
                             step=0.05, key="error_threshold")
hc_errors_info = identify_high_confidence_errors(..., threshold=error_threshold)
```

Module 9 reads `st.session_state.error_threshold` **nowhere** and hardcodes `threshold=0.9`. So a user
who sets 0.95 to inspect only near-certain errors sees a dashboard Confidence component computed at
0.9, a gauge they cannot reconcile with Module 3, and no indication the two disagree. This is a
straightforward violation of the single-source-of-truth principle the rest of the app is built on.

### M4. `plot_robustness_heatmap` hardcodes `zmin=0, zmax=0.5`, clipping real data **[NEW]**
`reporting_module.py:228-229`

`0.5` is not a data bound — it is the *scoring* constant from `reliability_module.py:75`
(`max(0.0, (1.0 - avg_drop / 0.5)) * MAX_ROB`), borrowed into a colormap as if it were a physical
limit. Two consequences:

- **`Drop < 0`** — noise injection and dropout routinely *improve* accuracy on small test sets. These
  clamp to `zmin=0` and render as pure green, indistinguishable from "no effect".
- **`Drop > 0.5`** — a model destroyed by a stress test renders identically to one that lost exactly
  half its accuracy, because both saturate at the top of `RdYlGn_r`.

The cells still carry the true number as `text`, so a user who reads the labels sees the truth — but
the *visual encoding*, the thing the chart exists to convey, is wrong at both ends. Use the actual
`min`/`max` of `z`, or symlog scaling.

### M5. The KPI cards contradict the tables directly beneath them **[NEW]**
`reporting_module.py:142` vs `:55-64`; `app.py:3265`

```python
"num_models": len(models),          # ALL trained models, regardless of dataset
```
```python
entry = trainer.metrics.get(m, {}).get(dataset_name)
if entry:                            # silently skipped when absent
    perf_rows.append({...})
```

`available_ds` (`:3198-3200`) is the **union** of dataset names across all models. Select a dataset
that only some models were evaluated on and the dashboard renders
`🧠 Models Trained: 5` directly above a 3-row performance table with no explanation of the two missing
models. Every other count in the summary (`num_stressed`, `num_calibrated`) is derived from actual
rows; only `num_models` is derived from `trained_models`. It should be
`len({r["Model"] for r in perf_rows})`, or the selector should be intersected so only
fully-evaluated datasets are offered.

### M6. "Best Performance" and "Most Robust" are definitions invented by this module **[NEW]**
`reporting_module.py:120-137` vs `comparison_module.py` and `post_stress_module.py`

```python
best_perf = max(perf_rows, key=lambda r: r["F1"])["Model"] ...   # F1 only
...
most_robust = min({m: np.mean([r["Drop"] for r in rob_rows if r["Model"] == m])
                   for m in models if any(...)}).items(), key=lambda x: x[1])[0]
```

`most_robust` additionally averages over **whatever stress types happen to have been run**, with no
coverage requirement — a model tested against one perturbation competes on equal footing with one
tested against five, and models with zero coverage are excluded entirely with no marker. `Drop` may
be negative, in which case `min` rewards the model that was *least* stressed rather than the most
robust.

Neither definition matches the ones the user already saw:
- Module 5 (`post_stress_module.calculate_robustness_score`) uses a σ-aware, drop-and-std composite
- Module 7 (`comparison_module`) uses a `perf`-weighted composite
- Module 8 uses `ReliabilityScorer`

So the dashboard can crown a different "most robust" model than Module 5 did, on the same screen
session, from the same stress results. For a comparative tool this is the most damaging class of
defect: it makes the framework's conclusions non-reproducible. Either delegate to the scorers or
label the cards with the exact criterion used.

### M7. Tab 1 has no empty-state guards; Tab 3 does — users get blank chart frames **[NEW]**
`app.py:3286-3301` vs `:3362-3388`

```python
# Tab 1 — unconditional
fig_perf = rgen.plot_performance_overview(report); st.plotly_chart(fig_perf, ...)
fig_cal  = rgen.plot_calibration_bar(report);     st.plotly_chart(fig_cal, ...)
fig_rad  = rgen.plot_radar_all(report);           st.plotly_chart(fig_rad, ...)
fig_rob  = rgen.plot_robustness_heatmap(report);  st.plotly_chart(fig_rob, ...)
```

Four methods return a bare `go.Figure()` when their input list is empty
(`reporting_module.py:166, 199, 294, 334`). Tab 1 renders all four unconditionally, so a user who
trained a model but never ran stress tests sees an **empty gridded frame with axes and a title** where
the heatmap should be, and an empty circular frame for the radar. Tab 3 gets this right:

```python
with st.expander("🛡️ Robustness Heatmap"):
    if report["robustness"]:
        st.plotly_chart(...)
    else:
        st.info("No stress data available.")
```

Tab 1's only guard is `if fig_gauges:` (`:3305`), which works solely because that one method returns
`None` rather than an empty figure — an inconsistency in the plotting API itself. Give all five
methods a `None` return for "no data" and guard uniformly, or have `plotly_chart` calls short-circuit
on an empty figure.

### M8. The exported HTML/PDF contain none of the dashboard's charts
`reporting_module.py:413-499`, `:501-599`; `app.py:3430`

Tab 1 renders five figures. `export_to_html` embeds **zero** — no Plotly, no static image, no SVG. The
PDF likewise contains only tables. Yet the UI describes the HTML as a "Styled, self-contained HTML
page" (`app.py:3430`) and the module header as "Aggregated dashboard across all modules with one-click
export" (`app.py:3183-3186`), and `README.md:31` marks Module 9 **Complete** with "Unified dashboard
and export".

The exported artifact is therefore strictly less informative than the screen. Either embed
`fig.to_html(include_plotlyjs="cdn")` / Kaleido PNGs, or stop claiming parity. This is a
specification-vs-implementation gap, not merely a missing feature.

### M9. No `export_calibration_csv`; `export_all_csv` can emit a zero-byte file
`app.py:3458-3498`, `reporting_module.py:399-411`

Individual downloads exist for performance, robustness, and reliability — but **not calibration**,
despite ECE being a headline metric with a full Tab 2 table and its own chart. The asymmetry is
unexplained.

Separately, `export_all_csv` returns `b""` (`:409`) when no section has rows, and the caller
(`app.py:3405`) passes it straight to `st.download_button` unconditionally. The user is offered a
0-byte download that silently produces an empty file on disk.

---

## Minor Issues

| # | Issue | Location |
|---|---|---|
| m1 | Broken literal `st.markdown("### {} JSON")` — missing `f` prefix, renders as `### {} JSON`; also reuses 📄 for both CSV and PDF while HTML uses 🌐 | `app.py:3416` |
| m2 | Dead local `drops` built at `:68-72` and never read; the loop at `:73-84` does the actual work | `reporting_module.py:68-72` |
| m3 | Heatmap `z` built by an O(models × types × rows) `next()` scan where a dict pivot would be O(n) | `reporting_module.py:204-217` |
| m4 | Radar palette has 5 entries vs the scorer's 7 (`reliability_module.py:289-297`); `palette[i % len(palette)]` makes the 6th model reuse colour #1, so two series are indistinguishable | `reporting_module.py:338` |
| m5 | Gauge `steps` bands `[0,50],[50,70],[70,90],[90,100]` contradict `_grade()`'s `50/60/70/80/90` — bar colour follows the grade, the background implies a different scale | `reporting_module.py:273-278` vs `reliability_module.py:25-36` |
| m6 | `n_cls = 2` hardcoded fallback: with models trained but `y_test is None`, entropy normalisation assumes binary and the Confidence component is silently wrong | `app.py:3220` |
| m7 | Divergent guards for the same value: `len(hce_df) / max(len(preds_r), 1)` vs Module 8's `... if total_preds > 0 else 0.0` | `app.py:3238` vs `:2945` |
| m8 | `report['models']` (user-controlled names) joined into markdown at `:3396` — markdown/image injection into the Streamlit render path | `app.py:3396` |
| m9 | `generated_at = datetime.now()` — naive local time, no timezone, and regenerated every rerun, so exports are never byte-reproducible and cannot be diffed across runs | `reporting_module.py:140` |
| m10 | `ReportGenerator` is stateless (no `__init__`, no fields) yet is stored in `st.session_state` — cargo-cult state management | `app.py:56-57` |
| m11 | Mixed `st.download_button` conventions in one file — keyword `label=` here (`:3406`), positional there (`:3463`) | `app.py:3406` vs `:3463` |
| m12 | All colour/threshold constants are inline literals in five plot methods (`#4C78A8`, `zmax=0.5`, `range=[0,1.15]`, …); no theme constants class, while `reliability_module._grade()` shows the correct pattern exists | `reporting_module.py` |
| m13 | `json.dumps(default=...)` handles `np.integer`/`np.floating`/`np.ndarray` but not `np.bool_`, which `json` rejects — a latent export crash if a bool ever reaches the report | `reporting_module.py:376-383` |

---

## Recommended Refactoring Targets

**Tier 1 — integrity (the dashboard must not lie)**

1. **`app.py:3211-3216, 3222-3225, 3231-3240`** — replace the three `except Exception: pass` blocks with
   an accumulating error list surfaced as `st.warning("N of M models could not be evaluated: …")`, and
   have `score_all_models` receive an explicit "not measured" marker rather than falling through to the
   12.5 neutral. Fixes C1 and stops the dashboard from presenting untested models as average.
2. **`reporting_module.py:336-337`** — `maxvals = [ReliabilityScorer.MAX_PERF, …]`, or delete
   `plot_radar_all`/`plot_reliability_gauge_row` and have Module 9 delegate to
   `ReliabilityScorer.plot_component_radar`/`plot_gauge`. Removes C3, m4 and m5 in one move.
3. **`reporting_module.py:419-429, 456, 461-478`** — `html.escape(str(row[k]))` on every interpolation,
   plus a filename/model-name allowlist in `model_module.load_model`. Fixes C2.
4. **`app.py:3236`** — `threshold=st.session_state.get("error_threshold", 0.9)`; better, thread it
   through `score_all_models` as an explicit parameter. Fixes M3.
5. **`reporting_module.py:228-229`** — derive `zmin`/`zmax` from `z` (or use symlog) instead of the
   hardcoded scoring constant. Fixes M4.

**Tier 2 — performance (cheap, mechanical, immediately felt)**

6. **`app.py:3207-3250`** — extract one `compute_reliability_inputs(trainer, dm, ...)` and `@st.cache_data`
   it on `(model_id, split_id, threshold)`; call it from **both** Module 8 and Module 9. Fixes M1
   (halves the inference cost) and M2 (one score, one truth) together.
7. **`app.py:3286-3301` + `:3356-3388`** — build each of the four shared figures once, reuse the same
   `go.Figure` object in Tab 1 and Tab 3. Removes 4 redundant constructions per rerun.
8. **`app.py:3405-3455, 3491-3498`** — move all nine exporter calls behind their `st.download_button`
   handlers. A PDF must never be built speculatively.

**Tier 3 — semantics & polish**

9. **`reporting_module.py:120-137`** — delete the local `best_perf`/`most_robust` definitions and take
   them from `ModelComparator` and `PostStressAnalyzer`, or rename the cards to state the criterion
   (e.g. "Highest F1"). Require full stress coverage before crowning a "Most Robust" model.
10. **`reporting_module.py:142`** — derive `num_models` from `perf_rows`, not `trained_models`; intersect
    `available_ds` so only fully-evaluated datasets are selectable. Fixes M5.
11. **`reporting_module.py:166, 199, 294, 334`** — return `None` for "no data" (matching
    `plot_reliability_gauge_row:246`) and guard every `st.plotly_chart` call in Tab 1 the way Tab 3
    already does. Fixes M7.
12. **`app.py:3416`** — `st.markdown("### 📄 JSON")`. Add `export_calibration_csv`. Guard the
    zero-byte `export_all_csv` case. Fixes m1, M9.
13. **`modules/reporting_module.py`** — add the test suite this file has been waiting for. It is pure,
    has no Streamlit dependency, and is trivially testable: escaping in `_table`, `compile_report` KPI
    derivation, `plot_*` empty-data returns, `export_all_csv` concatenation. Then gate CI on it.

---

## Verification Appendix

**Not dynamically verified.** The system `python3` lacks `numpy`/`pandas`/`plotly` and the repository
contains no virtualenv, so no code could be executed. Every claim above is derived from source reading
plus cross-file call-graph tracing; the specific line-level chains are cited inline to make each one
independently checkable. Findings requiring runtime confirmation before being actioned are C2 (payload
delivery), C1 (which specific exception fires per model type), and M4/M5 (actual `Drop` sign
distribution on real stress runs).