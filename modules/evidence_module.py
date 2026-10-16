"""Per-model evidence assembly — the single owner of "what did we measure?".

Architecture reference: ``architecture.md`` §3 (views are presentation only),
ADR-018, ADR-021.  This module was added by run-015 refactor R6 to close critic
M2/M3 and the "views contain domain metric formulas" finding: the ECE, entropy,
high-confidence-error and robustness figures were previously re-derived in three
separate pages (``views/module_07_comparison.py``, ``module_08_reliability.py``,
``module_09_reports.py``), so a fix to one silently missed the others.

The arithmetic itself lives in :mod:`core.scoring`; this module only *gathers*
the raw measurements (accuracy, F1, ECE, mean performance drop, entropy, HCE
rate, class count) into one :class:`ModelEvidence` per model.  The gathering is
pure over its inputs — no session state, no Streamlit — so it is unit-testable
without a UI.

Layer rules (``architecture.md`` §2.1): ``modules/`` may import ``core`` and
``utils``; this module imports both and never imports ``streamlit``/``views``/
``app``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from core.errors import (
    DatasetError,
    FrameworkError,
    ModelNotTrainedError,
    ModelPredictionError,
)
from core.scoring import mean_performance_drop, robustness_fraction
from utils.metrics import get_prediction_entropy, identify_high_confidence_errors

logger = logging.getLogger(__name__)

__all__ = [
    "ModelEvidence",
    "collect_evidence",
    "class_count",
    "calibration_ece",
    "mean_entropies",
    "hce_rates",
    "robustness_scores",
]

#: Confidence threshold above which a misclassification is a "high-confidence
#: error".  Matches the pre-remediation view calls (``threshold=0.9``).
_HCE_THRESHOLD = 0.9


@dataclass(frozen=True)
class ModelEvidence:
    """Everything measurable about one trained model on one dataset.

    A field is ``None`` exactly when the quantity could not be measured — it is
    never ``0.0`` standing in for "missing", because the scoring policy treats
    those two cases differently (ADR-018/ADR-021).

    Attributes:
        model_name: The trainer's model key.
        accuracy: Test accuracy, or ``None``.
        f1: Test F1 score, or ``None``.
        ece: Expected calibration error, or ``None``.
        avg_drop: Mean ``performance_drop`` across stress types, or ``None``.
        avg_entropy: Mean prediction entropy in **bits**, or ``None``.
        hce_rate: High-confidence-error rate in ``[0, 1]``, or ``None``.
        n_classes: Number of output classes (at least 2).
    """

    model_name: str
    accuracy: float | None
    f1: float | None
    ece: float | None
    avg_drop: float | None
    avg_entropy: float | None
    hce_rate: float | None
    n_classes: int

    def robustness_score(self, drop_cap: float) -> float | None:
        """Return the 0-100 robustness score, or ``None`` if unmeasured.

        Args:
            drop_cap: The drop that scores zero (``ReliabilityWeights.drop_cap``).

        Returns:
            float | None: ``robustness_fraction(avg_drop, drop_cap) * 100`` — the
            same transform the reliability scorer uses (ADR-021), so the two
            published robustness figures cannot disagree.
        """
        if self.avg_drop is None:
            return None
        return round(robustness_fraction(self.avg_drop, drop_cap=drop_cap) * 100.0, 2)


def _resolve_class_count(y_test: Any, fallback: int = 2) -> int:
    """Return ``max(2, unique labels)`` or the binary ``fallback`` on failure."""
    if y_test is None:
        return fallback
    try:
        return max(2, int(np.unique(np.asarray(y_test)).size))
    except Exception as exc:  # noqa: BLE001 - a degenerate target is not fatal
        logger.warning(
            "Could not determine the class count from the target; falling back to "
            "the binary default of %d, which will mis-grade a multiclass model. "
            "Cause: %s",
            fallback,
            exc,
        )
        return fallback


def _predict(
    trainer: Any, model_name: str, X_test: Any
) -> tuple[Any, Any] | None:
    """Return ``(predictions, probabilities)`` or ``None`` if prediction fails."""
    try:
        return trainer.predict(model_name, X_test)
    except ModelNotTrainedError as exc:
        logger.info(
            "Model %r has not been trained yet; skipping evidence gathering: %s",
            model_name,
            exc,
        )
        return None
    except (ModelPredictionError, FrameworkError) as exc:
        logger.warning(
            "Prediction failed for model %r (%s); its calibration and confidence "
            "evidence will be treated as missing. Cause: %s",
            model_name,
            type(exc).__name__,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - evidence gathering must not abort
        logger.exception(
            "Unexpected error while predicting for model %r; calibration and confidence "
            "evidence will be treated as missing: %s",
            model_name,
            exc,
        )
        return None


def collect_evidence(
    trainer: Any,
    X_test: Any,
    y_test: Any,
    *,
    stress_results: Mapping[str, Any] | None = None,
    dataset_name: str = "Test",
    calibration_analyzer: Any | None = None,
    n_classes: int | None = None,
) -> dict[str, ModelEvidence]:
    """Assemble one :class:`ModelEvidence` per trained model.

    Performs at most one ``predict`` per model and reuses the probabilities for
    both the calibration and the entropy/high-confidence-error evidence.

    Args:
        trainer: A ``ModelTrainer``-like object exposing ``trained_models``,
            ``metrics`` and ``predict``.
        X_test: Test features, or ``None`` when test data is not prepared.
        y_test: Test labels, or ``None``.
        stress_results: ``batch_stress_results_by_model`` mapping, or ``None``.
        dataset_name: Which metric split to read accuracy/F1 from.
        calibration_analyzer: A ``CalibrationAnalyzer``-like object; when
            ``None`` the ECE component is left unmeasured.
        n_classes: Explicit class count; when ``None`` it is derived from
            ``y_test``.

    Returns:
        dict: ``{model_name: ModelEvidence}`` for every trained model.
    """
    metrics_by_model = getattr(trainer, "metrics", {}) or {}
    trained = getattr(trainer, "trained_models", {}) or {}
    classes = _resolve_class_count(y_test) if n_classes is None else max(2, int(n_classes))
    have_test = X_test is not None and y_test is not None

    evidence: dict[str, ModelEvidence] = {}
    for model_name in trained:
        entry = (metrics_by_model.get(model_name, {}) or {}).get(dataset_name)

        accuracy = None
        f1 = None
        if entry:
            if entry.get("accuracy") is not None:
                accuracy = float(entry["accuracy"])
            if entry.get("f1") is not None:
                f1 = float(entry["f1"])

        ece: float | None = None
        avg_entropy: float | None = None
        hce_rate: float | None = None

        predicted = _predict(trainer, model_name, X_test) if have_test else None
        if predicted is not None:
            preds, probs = predicted
            if calibration_analyzer is not None:
                try:
                    model_obj = trained.get(model_name)
                    ece = float(
                        calibration_analyzer.compute_calibration_metrics(
                            y_test,
                            probs,
                            classes=getattr(model_obj, "classes_", None),
                        )["ece"]
                    )
                except Exception as exc:  # noqa: BLE001 - one bad model is not fatal
                    logger.warning(
                        "Calibration ECE unavailable for model %r; the model is "
                        "excluded from the calibration component. Cause: %s",
                        model_name,
                        exc,
                    )
            try:
                avg_entropy = float(np.mean(get_prediction_entropy(probs)))
                hce_info = identify_high_confidence_errors(
                    y_test, preds, probs, threshold=_HCE_THRESHOLD
                )
                total = len(preds)
                hce_rate = float(hce_info["count"]) / total if total > 0 else 0.0
            except Exception as exc:  # noqa: BLE001 - entropy/HCE share one failure
                logger.warning(
                    "Entropy / high-confidence-error rate unavailable for model "
                    "%r; both confidence figures will be missing. Cause: %s",
                    model_name,
                    exc,
                )

        avg_drop: float | None = None
        if stress_results and model_name in stress_results:
            avg_drop = mean_performance_drop(stress_results[model_name])

        evidence[model_name] = ModelEvidence(
            model_name=model_name,
            accuracy=accuracy,
            f1=f1,
            ece=ece,
            avg_drop=avg_drop,
            avg_entropy=avg_entropy,
            hce_rate=hce_rate,
            n_classes=classes,
        )

    return evidence


def class_count(
    evidence: Mapping[str, ModelEvidence],
    *,
    fallback: int = 2,
    strict: bool = False,
) -> int:
    """Return the shared class count held by the evidence, or ``fallback``.

    Args:
        evidence: Model evidence mapping.
        fallback: Fallback class count when evidence is empty or discordant.
        strict: When True, raise :class:`core.errors.DatasetError` on conflicting class counts.

    Raises:
        DatasetError: If ``strict=True`` and evidence contains conflicting class counts.
    """
    if not evidence:
        return fallback

    counts = {item.n_classes for item in evidence.values()}
    if len(counts) > 1:
        if strict:
            raise DatasetError(
                f"Conflicting class counts detected across models: {sorted(counts)}.",
                context={"class_counts": sorted(counts)},
            )
        logger.warning(
            "Discrepancy detected in model evidence class counts: %s; falling back to %d.",
            counts,
            fallback,
        )
        return fallback

    return next(iter(counts))


def calibration_ece(evidence: Mapping[str, ModelEvidence]) -> dict[str, float]:
    """Return ``{model: ece}`` for the models whose ECE was measured."""
    return {name: item.ece for name, item in evidence.items() if item.ece is not None}


def mean_entropies(evidence: Mapping[str, ModelEvidence]) -> dict[str, float]:
    """Return ``{model: avg_entropy}`` for the models whose entropy was measured."""
    return {
        name: item.avg_entropy
        for name, item in evidence.items()
        if item.avg_entropy is not None
    }


def hce_rates(evidence: Mapping[str, ModelEvidence]) -> dict[str, float]:
    """Return ``{model: hce_rate}`` for the models whose HCE rate was measured."""
    return {
        name: item.hce_rate
        for name, item in evidence.items()
        if item.hce_rate is not None
    }


def robustness_scores(
    evidence: Mapping[str, ModelEvidence], *, drop_cap: float
) -> dict[str, float]:
    """Return ``{model: 0-100 robustness score}`` for the measured models."""
    scores: dict[str, float] = {}
    for name, item in evidence.items():
        score = item.robustness_score(drop_cap)
        if score is not None:
            scores[name] = score
    return scores
