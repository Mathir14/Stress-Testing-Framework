"""Session-state schema and the per-run application context.

Architecture reference: ``architecture.md`` §4.7, ADR-011.4.

Two separate jobs, deliberately kept apart:

* :class:`StateKeys` plus :func:`init_state` / :func:`reset_state` own the
  **schema** — which keys exist and what they mean.  Every literal value here
  equals the string the pre-remediation ``app.py`` used directly, because those
  strings are the persisted session contract.
* :class:`AppContext` is a **cheap per-script-run façade** over an injected
  mapping.  Streamlit re-executes the whole script on every interaction, so the
  context is constructed once per run in ``views/base.build_context()`` and never
  cached in the session state itself.

The RNG is injected rather than built here: ``core.state`` must not import
Streamlit, and persisting a ``Generator`` across reruns would make results depend
on how many times the user clicked (architecture.md §4.7).

Import rules (architecture.md §2.3): ``core.state`` may import ``core.config``
and ``core.errors`` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, MutableMapping

import numpy as np

from core.errors import ConfigurationError

__all__ = ["StateKeys", "init_state", "reset_state", "AppContext"]

logger = logging.getLogger(__name__)


class StateKeys:
    """Session-state key constants.

    Every value is the literal string the pre-remediation ``app.py`` used, so
    widgets and view code that reference them keep working unchanged
    (conventions §7).
    """

    # Service singletons.
    DATA_MANAGER = "data_manager"
    MODEL_TRAINER = "model_trainer"
    POST_STRESS_ANALYZER = "post_stress_analyzer"
    CALIBRATION_ANALYZER = "calibration_analyzer"
    MODEL_COMPARATOR = "model_comparator"
    RELIABILITY_SCORER = "reliability_scorer"
    REPORT_GENERATOR = "report_generator"
    STRESS_TESTER = "stress_tester"

    # Workflow flags.
    CURRENT_STEP = "current_step"
    DATA_LOADED = "data_loaded"
    DATA_PREPARED = "data_prepared"
    MODEL_TRAINED = "model_trained"

    # Derived results.
    PREDICTIONS = "predictions"
    SINGLE_STRESS_RESULT = "single_stress_result"
    BATCH_STRESS_RESULTS = "batch_stress_results"
    BATCH_STRESS_RESULTS_BY_MODEL = "batch_stress_results_by_model"

    @classmethod
    def services(cls) -> tuple[str, ...]:
        """Return every service-singleton key, in bootstrap order."""
        return (
            cls.DATA_MANAGER,
            cls.MODEL_TRAINER,
            cls.POST_STRESS_ANALYZER,
            cls.CALIBRATION_ANALYZER,
            cls.MODEL_COMPARATOR,
            cls.RELIABILITY_SCORER,
            cls.REPORT_GENERATOR,
            cls.STRESS_TESTER,
        )

    @classmethod
    def flags(cls) -> tuple[str, ...]:
        """Return every workflow-flag key."""
        return (cls.CURRENT_STEP, cls.DATA_LOADED, cls.DATA_PREPARED, cls.MODEL_TRAINED)

    @classmethod
    def derived(cls) -> tuple[str, ...]:
        """Return every derived-result key, in dependency order."""
        return (
            cls.PREDICTIONS,
            cls.SINGLE_STRESS_RESULT,
            cls.BATCH_STRESS_RESULTS,
            cls.BATCH_STRESS_RESULTS_BY_MODEL,
        )

    @classmethod
    def all_keys(cls) -> tuple[str, ...]:
        """Return every key this module manages."""
        return (*cls.services(), *cls.flags(), *cls.derived())


def _default_flag_value(key: str) -> Any:
    """Return the default value for a workflow flag."""
    if key == StateKeys.CURRENT_STEP:
        return 1
    return False


def init_state(store: MutableMapping[str, Any]) -> None:
    """Ensure every managed key exists in ``store``.

    Idempotent: keys already present are left untouched, so calling it at the top
    of every script run is safe and cannot clobber a user's progress.

    Args:
        store: Session-state mapping to initialise.

    Returns:
        None
    """
    for key in StateKeys.flags():
        if key not in store:
            store[key] = _default_flag_value(key)

    for key in StateKeys.derived():
        store.setdefault(key, None)


def reset_state(store: MutableMapping[str, Any], *, keep_services: bool = False) -> None:
    """Clear workflow flags and derived results.

    Args:
        store: Session-state mapping to reset.
        keep_services: When ``True``, service singletons are preserved so the
            user does not lose an expensive training run.

    Returns:
        None
    """
    for key in (*StateKeys.flags(), *StateKeys.derived()):
        store.pop(key, None)

    if not keep_services:
        for key in StateKeys.services():
            store.pop(key, None)

    init_state(store)
    logger.info("Session state reset (keep_services=%s)", keep_services)


@dataclass(frozen=True)
class AppContext:
    """A cheap façade over the session-state mapping for one script run.

    Attributes:
        store: The injected session-state mapping (``st.session_state``).
        rng: A fresh ``np.random.default_rng(config.random_seed)`` created once
            per script run by ``views/base.build_context()``.  It is deliberately
            **not** stored in the session state.
    """

    store: MutableMapping[str, Any]
    rng: np.random.Generator = field(repr=False)

    def ensure_service(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return the service stored under ``key``, creating it on first use.

        Args:
            key: Session-state key.
            factory: Zero-argument callable constructing the service.

        Returns:
            The stored service instance.

        Raises:
            ConfigurationError: If ``key`` is not a declared service key.
        """
        if key not in StateKeys.services():
            raise ConfigurationError(
                f"{key!r} is not a declared service key. Declared service keys "
                f"are: {', '.join(StateKeys.services())}.",
                context={"key": key, "accepted": list(StateKeys.services())},
            )
        existing = self.store.get(key)
        if existing is None:
            existing = factory()
            self.store[key] = existing
            logger.debug("Created service %s", key)
        return existing

    def has_data(self) -> bool:
        """Return whether a dataset has been loaded and prepared."""
        data_manager = self.store.get(StateKeys.DATA_MANAGER)
        if data_manager is None:
            return False
        raw = getattr(data_manager, "raw_data", None)
        return raw is not None and bool(self.store.get(StateKeys.DATA_LOADED, False))

    def has_models(self) -> bool:
        """Return whether at least one model has been trained."""
        if not self.store.get(StateKeys.MODEL_TRAINED, False):
            return False
        trainer = self.store.get(StateKeys.MODEL_TRAINER)
        trained = getattr(trainer, "trained_models", None)
        return bool(trained)

    def invalidate_from(self, step: int) -> None:
        """Clear every derived result downstream of ``step``.

        Args:
            step: 1-based workflow step.  ``1`` (data) invalidates every derived
                result; ``2`` (features/split) invalidates stress, calibration
                and reliability outputs; ``3`` (models) invalidates only the
                derived per-model results.

        Returns:
            None
        """
        if step <= 1:
            self.store[StateKeys.DATA_PREPARED] = False
            self.store[StateKeys.MODEL_TRAINED] = False
            cleared = StateKeys.derived()
        elif step == 2:
            self.store[StateKeys.MODEL_TRAINED] = False
            cleared = StateKeys.derived()
        else:
            cleared = StateKeys.derived()[1:]

        for key in cleared:
            self.store[key] = None
        logger.info("Invalidated derived state from step %s: %s", step, list(cleared))
