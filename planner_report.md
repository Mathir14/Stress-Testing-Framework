# PLANNER Report — run-011 remediation

## Summary

The Architect's spec is complete and binding. I decomposed the 4 critical, 8 major findings plus minor items into 13 concrete tasks. Key constraints: fixed-denominator monotone scoring (no redistribution, no midpoint), calibration/Brier in class-index space with `classes` threading and strict identity, HCE computed as `count/n`, entropy converted to bits, `batch_stress_test` validates before dispatch, and PostStress getters must be non-mutating. The reliability baseline must be regenerated.

## Assumptions

- ScoringPolicy must be centralized in `core.config` and consumed by reliability, comparison, reporting; dead weight attributes removed.
- `classes=None` means strict identity (0..K-1) for calibration/Brier; views must pass `estimator.classes_` when available.
- `identify_high_confidence_errors` returns a structure with `count` key; both view sites (module_08_reliability, module_09_reports) must use it.
- `utils/` remains free of `core/`/`modules/` imports; the calibration aligner lives in `modules/calibration_module`.
- Post-remediation locking: M6 (caching) is declined by ADR-020; no new runtime deps; mypy additions are dev-only.

## Dependencies

Tasks have clear ordering: T1 (errors) before T5; T2 (ScoringPolicy) before T9,T10,T12; T3 (calibration classes) before T11; T4 (entropy bits) before T9; T9,T10,T11 before T13 baseline/tests; ruff + architecture tests must pass.

## Tasks

| ID | Component | Description | Acceptance |
|---|---|---|---|
| T1 | core.errors | Add `UnknownModelError`, `ModelNotTrainedError`, `ReportExportError` subclasses | Errors exist, AST gate for FrameworkError subclasses unaffected, importable |
| T2 | core.config | Add `ScoringPolicy` dataclass with composite weights, missing component points=0.0, min_measured_components=2, ece/severity bands; add `scoring` field to `AppConfig`; expose `resolve_grade` still works | Config round-trips, frozen dataclass, used by downstream (no cycles) |
| T3 | modules.calibration_module | Add `align_target_indices(y_true, classes)` raising `ValidationError` if classes None and labels not 0..K-1; add `multiclass_brier` taking `classes=None` (strict identity); make `compute_calibration_metrics`, `_multiclass_brier` usage, `compute_per_class_calibration`, `plot_confidence_histogram`, `find_optimal_temperature` accept keyword-only `classes`; remove reliance on `sklearn.brier_score_loss` in a way consistent with consolidation | C1 fixed; strict identity enforced; single Brier owner path |
| T4 | utils.metrics | Change `get_prediction_entropy` to use `np.log2` (bits); `identify_high_confidence_errors` returns `TypedDict`-compatible dict with `count`/`percentage`/`avg_confidence`/`indices`; delete `calculate_brier_score`, `get_confidence_bins` | Units correct; no removed symbols imported elsewhere |
| T5 | modules.model_module | Replace bare `ValueError` at `get_model`, `predict`, `save_model`, `load_model` with typed errors from core.errors | No bare ValueError escapes model_module; load/save paths still respect trust/attestation |
| T6 | modules.data_module | Guard `stratify` in `split_data` (check class frequencies >= 2 for all classes) and wrap residual sklearn ValueError as `DatasetError` with context; log fallback to non-stratified if needed | M3 fixed; no raw sklearn ValueError propagates |
| T7 | modules.stress_module | In `batch_stress_test`, for each config validate with `validate_stress_params(op, params)` (normalize) before `**params` expansion; raise `ValidationError`/`ParameterOutOfRangeError`/`UnsupportedStressTypeError` as appropriate (C4) | Unknown param keys rejected with typed errors; existing valid configs unchanged |
| T8 | modules.post_stress_module | Extract pure `_compute_robustness_breakdown(model_name, ...)` returning dict; `calculate_robustness_score` uses it but doesn't have side effects that break purity; make `plot_robustness_radar` and `compare_model_robustness` non-mutating (read-only); handle zero-mean explicitly (no NaN from std/mean); raise `ValidationError`/typed error for unknown model per spec; severity bands from `ScoringPolicy` if available | M5 fixed; no mutation of internal state from getters |
| T9 | modules.reliability_module | Apply fixed-denominator zero-for-missing: missing components get 0 (not midpoint); add `rated` (bool) and require `min_measured_components>=2`? Set `rated = len(available) >= scorer.weights.min_measured_components` (default 2); don't award bonus; use log2-based max entropy (from bits); integrate `ScoringPolicy` via `weights`; ranking must treat `rated=False` separately | C2's entropy context; M1,M2 fixed; monotone |
| T10 | modules.comparison_module | Remove dead `METRIC_WEIGHT`/`ROBUSTNESS_WEIGHT`/`CALIBRATION_WEIGHT`; compute composite with fixed denominator (total = perf+rob+cal), zero for missing, no redistribution; cap appropriately; add `rated` gate in recommendation/ranking; read policy from config if injected or add parameter to pass ScoringPolicy | C3 fixed; monotone; no dead attrs |
| T11 | views | module_08_reliability: use `hce_df["count"]/total_preds` (not `len(hce_df)`) — C2; also module_09_reports same; thread `classes = getattr(trainer.trained_models[model].model, "classes_", None)` or from stored estimator when calling calibration; update UI to show "Unrated" when `rated=False` | Both HCE sites fixed; classes threading complete |
| T12 | modules.reporting_module | Wrap JSON export/serialization errors as `ReportExportError`; use `ScoringPolicy` for grade colours/ECE bands; ensure HCE values sourced correctly | M4 area addressed for reporting; consistent policy |
| T13 | tests | Regenerate `tests/fixtures/reliability_baseline.json` on recorded interpreter (per Wave 0); update `tests/test_reliability_parity.py` to new policy; add `tests/test_calibration.py`, `tests/test_comparison.py`; add batch-path validation test in `tests/test_validation.py`; add score-monotonicity property test; update any tests broken by entropy units/removals; run full suite | M8 fixed; parity green with new baseline |

## Acceptance Criteria

All listed in machine YAML above. Key points: monotone scoring (no 90→75 inversion), strict identity when `classes=None`, HCE uses `count`, entropy in bits, batch validates before dispatch, getters non-mutating, removed symbols gone.

## Validation Required

- `pytest tests/test_calibration.py -v`
- `pytest tests/test_comparison.py -v`  
- `pytest tests/test_reliability_parity.py -v`
- `pytest tests/test_validation.py -v`
- `pytest tests/test_architecture.py -v`
- `ruff check .`

## Risks

- Baseline regeneration must happen on the exact interpreter recorded in Wave 0; any interpreter skew breaks parity.
- Changing entropy units affects any code that assumed nats (module_03 shows "Avg Entropy" - label may need clarification but value is bits now).
- Score semantics change visibly (models lose midpoint bonus) — UI copy must reflect "Unrated" state.
- Removing `utils.calculate_brier_score`/`get_confidence_bins` will break any external references; grep to confirm none.

## Recommendation

Proceed to EXECUTOR. The plan is concrete and matches the Architect's binding spec.
