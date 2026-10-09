"""
Stress Testing Module
Functions for applying various stress tests to evaluate model robustness

The six operators are thin delegations to :mod:`core.perturbations`
(ADR-003).  Before Wave 2 each operator carried its own DataFrame/ndarray pair —
twelve duplicated branches in total — and the DataFrame branches wrote through
``DataFrame.values``, which silently discarded the result whenever pandas had
to build an upcast temporary.  Measured on an ``int64 + float64`` frame, five of
the six operators were silent no-ops.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.errors import UnsupportedStressTypeError
from core.perturbations import apply_perturbation, operation_key

# ``core.validation`` owns both vocabularies (ADR-011.1) and the parameter
# validators (ADR-019, C4); this module is the historical import site callers
# already use.  The two alias re-exports are declared in ``__all__`` so they are
# explicit to both readers and linters rather than looking like stray imports.
from core.validation import CORRUPTION_TYPES as CORRUPTION_TYPES
from core.validation import SHIFT_TYPES as SHIFT_TYPES
from core.validation import validate_stress_params

__all__ = ["StressTester", "CORRUPTION_TYPES", "SHIFT_TYPES"]

logger = logging.getLogger(__name__)

#: Frozen mapping from this module's public method names to registry keys, so
#: the delegation table is declarative and auditable.
_METHOD_TO_OPERATION: Mapping[str, str] = {
    "add_gaussian_noise": "gaussian_noise",
    "add_uniform_noise": "uniform_noise",
    "feature_dropout": "feature_dropout",
    "feature_corruption": "feature_corruption",
    "scale_perturbation": "scale_perturbation",
    "distribution_shift": "distribution_shift",
}


class StressTester:
    """
    Class for applying stress tests to datasets and evaluating model robustness
    """

    def __init__(self, rng: np.random.Generator | None = None):
        """
        Args:
            rng: Random generator injected by the caller.  ``None`` delegates
                the default to :func:`core.perturbations.apply_perturbation`,
                which is the only domain module permitted to create one when
                the UI has not supplied ``AppContext.rng`` (architecture.md
                §4.7, ADR-011.4).
        """
        self.original_data = None
        self.stressed_data = {}
        self.stress_results = {}
        self.rng = rng

    # ── Operators (thin delegations to the kernel) ──────────────────────────

    def _apply(
        self,
        X,
        method: str,
        params: Mapping[str, Any],
        rng: np.random.Generator | None = None,
    ):
        """Delegate to :func:`apply_perturbation` with this module's registry key."""
        return apply_perturbation(
            X,
            _METHOD_TO_OPERATION[method],
            params,
            rng=self.rng if rng is None else rng,
        )

    def add_gaussian_noise(
        self,
        X: Any,
        noise_level: float = 0.1,
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Add Gaussian noise to features

        Formula (preserved from the pre-refactor implementation): each column
        receives ``N(0, (std * noise_level)^2)`` noise.  A zero-variance column
        has ``std == 0`` and therefore receives zero-width noise rather than
        ``NaN``.

        Args:
            X: Input features (DataFrame or 2-D array)
            noise_level: Standard deviation of noise relative to feature std
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``
        """
        return self._apply(X, "add_gaussian_noise", {"noise_level": noise_level}, rng)

    def add_uniform_noise(
        self,
        X: Any,
        noise_range: float = 0.1,
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Add uniform noise to features

        Formula: each column receives ``U(-r, r)`` where
        ``r = (max - min) * noise_range``.

        Args:
            X: Input features (DataFrame or 2-D array)
            noise_range: Range of uniform noise as fraction of feature range
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``
        """
        return self._apply(X, "add_uniform_noise", {"noise_range": noise_range}, rng)

    def feature_dropout(
        self,
        X: Any,
        dropout_rate: float = 0.2,
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Randomly zero out cells (dropout)

        Formula: ``x * (rng.random(shape) > dropout_rate)`` — a per-**cell**
        Bernoulli mask.

        Naming debt, recorded in ADR-010.7 and deliberately **not** corrected in
        this milestone: this masks individual cells, not features-per-sample as
        the name and the UI label suggest.  The behaviour is load-bearing for
        published results, so only the documentation changes.

        Args:
            X: Input features (DataFrame or 2-D array)
            dropout_rate: Fraction of cells to zero out; must be < 1
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``

        Raises:
            ParameterOutOfRangeError: If ``dropout_rate >= 1``.
        """
        return self._apply(X, "feature_dropout", {"dropout_rate": dropout_rate}, rng)

    def feature_corruption(
        self,
        X: Any,
        corruption_rate: float = 0.1,
        corruption_type: str = "zero",
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Corrupt a fraction of cells in each column

        Formula: ``int(n_rows * corruption_rate)`` sampled cells per column are
        replaced with ``0.0`` / the column mean / ``U(min, max)`` / a random
        choice of ``min`` or ``max`` depending on ``corruption_type``.

        Args:
            X: Input features (DataFrame or 2-D array)
            corruption_rate: Fraction of values to corrupt; must be < 1
            corruption_type: One of ``CORRUPTION_TYPES``
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``

        Raises:
            UnsupportedStressTypeError: If ``corruption_type`` is unknown.  The
                pre-refactor code fell through every ``elif`` and returned the
                frame completely unperturbed with no error and no log.
        """
        return self._apply(
            X,
            "feature_corruption",
            {"corruption_rate": corruption_rate, "corruption_type": corruption_type},
            rng,
        )

    def scale_perturbation(
        self,
        X: Any,
        scale_factor: float = 1.5,
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Perturb feature scales

        Formula: each column is multiplied by ``factor`` when
        ``rng.random() > 0.5`` and by ``1 / factor`` otherwise — one draw per
        column.  ``scale_factor`` has an **exclusive** lower bound of 1.0 and is
        validated before any RNG draw, so ``scale_factor=0`` now raises
        deterministically instead of raising ``ZeroDivisionError`` on roughly
        half of all draws (ADR-011.7).

        Args:
            X: Input features (DataFrame or 2-D array)
            scale_factor: Scaling factor; must be > 1.0
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``

        Raises:
            ParameterOutOfRangeError: If ``scale_factor <= 1.0``.
        """
        return self._apply(X, "scale_perturbation", {"scale_factor": scale_factor}, rng)

    def distribution_shift(
        self,
        X: Any,
        shift_type: str = "mean",
        shift_amount: float = 0.5,
        *,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Shift feature distributions

        Formulas: ``mean`` adds ``std * shift_amount`` to each column;
        ``variance`` maps ``mean + (x - mean) * shift_amount``.

        Naming debt, recorded in ADR-010.7 and deliberately **not** corrected in
        this milestone: the ``variance`` branch contracts values toward the mean
        for ``shift_amount < 1``; it does not scale variance.

        Args:
            X: Input features (DataFrame or 2-D array)
            shift_type: One of ``SHIFT_TYPES``
            shift_amount: Amount of shift (for ``mean``) or multiplier (for
                ``variance``)
            rng: Optional generator override; defaults to ``self.rng``

        Returns:
            New features of the same container type, dtype ``float64``

        Raises:
            UnsupportedStressTypeError: If ``shift_type`` is unknown.
        """
        return self._apply(
            X,
            "distribution_shift",
            {"shift_type": shift_type, "shift_amount": shift_amount},
            rng,
        )

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate_stress_test(
        self,
        model: Any,
        X_original: pd.DataFrame,
        X_stressed: pd.DataFrame,
        y_true: Any,
    ) -> dict[str, Any]:
        """
        Evaluate model performance on stressed data

        Args:
            model: Trained model
            X_original: Original features
            X_stressed: Stressed features
            y_true: True labels

        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Convert y_true to numpy array if it's a pandas Series (to avoid index issues)
        if isinstance(y_true, pd.Series):
            y_true_array = y_true.values
        else:
            y_true_array = y_true

        # Predictions on original data
        y_pred_original = model.predict(X_original)
        acc_original = accuracy_score(y_true_array, y_pred_original)

        # Predictions on stressed data
        y_pred_stressed = model.predict(X_stressed)
        acc_stressed = accuracy_score(y_true_array, y_pred_stressed)

        # Performance drop
        performance_drop = acc_original - acc_stressed

        # Precision, Recall, F1 for stressed data
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_array, y_pred_stressed, average="weighted", zero_division=0
        )

        # Agreement between original and stressed predictions
        prediction_agreement = (y_pred_original == y_pred_stressed).mean()

        return {
            "accuracy_original": acc_original,
            "accuracy_stressed": acc_stressed,
            "performance_drop": performance_drop,
            "performance_drop_pct": (
                (performance_drop / acc_original * 100) if acc_original > 0 else 0
            ),
            "precision_stressed": precision,
            "recall_stressed": recall,
            "f1_stressed": f1,
            "prediction_agreement": prediction_agreement,
            "y_pred_original": y_pred_original,
            "y_pred_stressed": y_pred_stressed,
        }

    def batch_stress_test(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Any,
        stress_configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run multiple stress tests

        Args:
            model: Trained model
            X: Features
            y: Labels
            stress_configs: List of stress test configurations, each with a
                ``type`` key naming one of the six registry keys and an optional
                ``params`` mapping

        Returns:
            Dictionary of results for each stress test

        Raises:
            UnsupportedStressTypeError: If a configuration names an unknown
                stress type or an unsupported enum value.  The pre-refactor code
                silently ``continue``-d, which hid typos until a batch produced
                fewer rows than expected.
            ParameterOutOfRangeError: If a parameter is outside its bounds.
            ValidationError: If a configuration contains an unrecognised
                parameter key; the message names the operator and its accepted
                keys (architecture.md §9 item 9).
        """
        results = {}

        for config in stress_configs:
            stress_type = config["type"]
            raw_params = config.get("params", {})
            name = config.get("name", stress_type)

            # Resolve the registry key up front so an unknown type is a typed
            # error naming the operator, not a bare TypeError from deep inside
            # the parameter expansion below.
            if stress_type not in _METHOD_TO_OPERATION.values():
                raise UnsupportedStressTypeError(
                    f"Unknown stress type {stress_type!r} in batch configuration "
                    f"{name!r}. Accepted types are: "
                    f"{', '.join(_METHOD_TO_OPERATION.values())}.",
                    field="type",
                    value=stress_type,
                    context={
                        "stress_type": stress_type,
                        "accepted": list(_METHOD_TO_OPERATION.values()),
                    },
                )

            method_name = next(
                method
                for method, key in _METHOD_TO_OPERATION.items()
                if key == stress_type
            )
            # Validate before dispatch (C4 / architecture.md §9 item 9): an
            # unrecognised parameter key must be a typed ValidationError that
            # names the operator, not a bare TypeError from ``**params``.
            params = validate_stress_params(stress_type, raw_params)
            logger.info("Applying batch stress %s", name)
            X_stressed = getattr(self, method_name)(X, **params)

            # Evaluate
            result = self.evaluate_stress_test(model, X, X_stressed, y)
            result["config"] = config
            results[name] = result

        return results

    @staticmethod
    def label_to_operation(label: str) -> str:
        """Translate a UI stress-test label to its registry key.

        Thin wrapper over :func:`core.perturbations.operation_key` so callers in
        the view layer never build their own label map (ADR-011.2).

        Args:
            label: Human-readable label such as ``"Gaussian Noise"``.

        Returns:
            str: The registry key.

        Raises:
            UnsupportedStressTypeError: If the label is not recognised.
        """
        return operation_key(label)