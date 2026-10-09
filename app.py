"""ML Model Reliability & Stress Testing Framework — composition root.

Architecture reference: ``architecture.md`` §2.2 (target layering) and §4.8
(``views``); conventions §1 ("``app.py`` = composition root … no page bodies").

This module is the only place that knows about *all* the layers at once.  It
performs exactly five jobs and contains no domain logic:

1. declare the Streamlit page configuration (must be the first ``st.*`` call);
2. resolve the configuration and install logging once
   (:func:`core.logging_config.configure_logging`, conventions §4);
3. bootstrap session state and build the per-script-run
   :class:`~core.state.AppContext` — including the service singletons and the
   single RNG the framework is allowed to build (ADR-011.4);
4. draw the sidebar navigation and dispatch to the selected page;
5. render the status footer.

Every page body lives in ``views/module_0N_*.py`` and is reached only through
:data:`PAGES`.  The nine sidebar labels are the persisted session contract and
are reproduced here verbatim from :data:`views.base.PAGE_KEYS` (conventions §7);
:func:`_assert_page_contract` fails loudly if the two ever drift apart.

Evaluation order is unchanged from the pre-refactor single-file application:
page config → session state → title → sidebar → page body → footer.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from core.config import get_config
from core.logging_config import configure_logging
from core.state import AppContext, StateKeys
from modules.calibration_module import CalibrationAnalyzer
from modules.comparison_module import ModelComparator
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from modules.post_stress_module import PostStressAnalyzer
from modules.reliability_module import ReliabilityScorer
from modules.reporting_module import ReportGenerator
from modules.stress_module import StressTester
from views import (
    module_01_data,
    module_02_baseline,
    module_03_confidence,
    module_04_stress,
    module_05_post_stress,
    module_06_calibration,
    module_07_comparison,
    module_08_reliability,
    module_09_reports,
)
from views.base import PAGE_KEYS, build_context, render_sidebar, render_status_footer

# ── 1. Page configuration ────────────────────────────────────────────────────
# Must remain the first Streamlit call in the script: Streamlit raises if any
# other st.* command (including st.write) runs first.
st.set_page_config(
    page_title="ML Reliability Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Configuration and logging ──────────────────────────────────────────────
CONFIG = get_config()
configure_logging(CONFIG)

# ── 3. Session state and per-run context ──────────────────────────────────────
CTX = build_context(CONFIG)


def _service_factories() -> dict[str, Callable[[], object]]:
    """Return the service-singleton factories, in bootstrap order.

    These are the seven services the pre-refactor ``app.py`` created eagerly in
    its init block, plus ``StressTester``, which architecture.md §4.8 records as
    safe to move to eager initialisation (its ``__init__`` only assigns three
    attributes).

    ``core.state`` cannot build these itself — §2.1 forbids ``core/`` from
    importing ``modules/`` — which is exactly why the wiring lives here.

    No factory captures the RNG.  ``StressTester`` is a session-state singleton
    that outlives a script run, so binding it to one run's generator here would
    make results depend on when it happened to be created.  The stress view
    rebinds ``stress_tester.rng = ctx.rng`` on every run instead (ADR-011.4),
    which is the only place that legitimately holds the generator.
    """
    return {
        StateKeys.DATA_MANAGER: DataManager,
        StateKeys.MODEL_TRAINER: ModelTrainer,
        StateKeys.POST_STRESS_ANALYZER: PostStressAnalyzer,
        StateKeys.CALIBRATION_ANALYZER: CalibrationAnalyzer,
        StateKeys.MODEL_COMPARATOR: ModelComparator,
        StateKeys.RELIABILITY_SCORER: ReliabilityScorer,
        StateKeys.REPORT_GENERATOR: ReportGenerator,
        StateKeys.STRESS_TESTER: StressTester,
    }


for _key, _factory in _service_factories().items():
    CTX.ensure_service(_key, _factory)

# ── 4. Navigation and dispatch ────────────────────────────────────────────────
#: The explicit, ordered page mapping required by architecture.md §4.8.  Keys are
#: spelled out rather than zipped against ``PAGE_KEYS`` so the contract is
#: reviewable in one place; ``_assert_page_contract`` keeps it in sync.
PAGES: dict[str, Callable[[AppContext], None]] = {
    "1️⃣ Data Management": module_01_data.render,
    "2️⃣ Baseline Modeling": module_02_baseline.render,
    "3️⃣ Prediction & Confidence": module_03_confidence.render,
    "4️⃣ Stress Testing": module_04_stress.render,
    "5️⃣ Post-Stress Evaluation": module_05_post_stress.render,
    "6️⃣ Calibration Analysis": module_06_calibration.render,
    "7️⃣ Model Comparison": module_07_comparison.render,
    "8️⃣ Reliability Scoring": module_08_reliability.render,
    "9️⃣ Visualization & Reports": module_09_reports.render,
}


def _assert_page_contract() -> None:
    """Fail loudly if :data:`PAGES` and :data:`PAGE_KEYS` have drifted apart."""
    if tuple(PAGES) != PAGE_KEYS:
        raise RuntimeError(
            "Composition-root page mapping is out of sync with "
            f"views.base.PAGE_KEYS. PAGES={list(PAGES)} PAGE_KEYS={list(PAGE_KEYS)}"
        )


_assert_page_contract()

st.title("🔬 ML Model Reliability & Stress Testing Framework")
st.markdown("### A comprehensive framework for testing ML model robustness and reliability")

MODULE = render_sidebar(CONFIG)
PAGES[MODULE](CTX)

# ── 5. Status footer ──────────────────────────────────────────────────────────
render_status_footer(CTX)
