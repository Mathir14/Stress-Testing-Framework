"""Composition helpers shared by every view module.

Architecture reference: ``architecture.md`` §4.8.

This module owns three responsibilities and nothing else:

* :func:`build_context` — construct a fresh :class:`~core.state.AppContext` for
  the current script run, including the one RNG the framework is allowed to
  build (ADR-011.4).
* :func:`render_sidebar` — draw the navigation radio and return the selected page
  key.  The nine labels are preserved verbatim (conventions §7).
* :func:`render_status_footer` — the three-line status block at the bottom of the
  sidebar.

No domain computation happens here.
"""

from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from core.config import AppConfig, get_config
from core.state import AppContext, StateKeys, init_state

__all__ = [
    "PAGE_KEYS",
    "build_context",
    "render_sidebar",
    "render_status_footer",
]

logger = logging.getLogger(__name__)

#: The nine sidebar labels, verbatim, in display order (conventions §7,
#: architecture.md §4.8).  ``app.py`` dispatches through this tuple.
PAGE_KEYS: tuple[str, ...] = (
    "1️⃣ Data Management",
    "2️⃣ Baseline Modeling",
    "3️⃣ Prediction & Confidence",
    "4️⃣ Stress Testing",
    "5️⃣ Post-Stress Evaluation",
    "6️⃣ Calibration Analysis",
    "7️⃣ Model Comparison",
    "8️⃣ Reliability Scoring",
    "9️⃣ Visualization & Reports",
)


def build_context(config: AppConfig | None = None) -> AppContext:
    """Build the per-script-run application context.

    A **new** ``AppContext`` and a **new** ``Generator`` are created on every
    run: persisting the generator would make results depend on how many times the
    user clicked (architecture.md §4.7).

    Args:
        config: Configuration supplying ``random_seed``; ``None`` uses
            ``get_config()``.

    Returns:
        AppContext: Context wrapping ``st.session_state`` and the fresh RNG.
    """
    active = config if config is not None else get_config()
    init_state(st.session_state)
    rng = np.random.default_rng(active.random_seed)
    return AppContext(store=st.session_state, rng=rng)


def render_sidebar(config: AppConfig | None = None) -> str:
    """Draw the navigation radio and return the selected page label.

    The radio deliberately carries **no** ``key=``: the pre-refactor ``app.py``
    radio at line 69 had none either, and conventions §7 makes the widget keys a
    persisted session contract.  Adding one would grow that contract from its
    frozen 64 literals (63 pre-remediation + ``load_model_trust``, ADR-001) to
    65 for no behavioural gain.

    Args:
        config: Unused configuration hook, kept so ``app.py`` can pass it
            uniformly to its view helpers.

    Returns:
        str: One of :data:`PAGE_KEYS`.
    """
    st.sidebar.title("📋 Navigation")
    selected = st.sidebar.radio(
        "Select Module:",
        list(PAGE_KEYS),
    )
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Complete modules in order for best results")
    return selected


def render_status_footer(ctx: AppContext) -> None:
    """Render the three-line workflow status block in the sidebar.

    Args:
        ctx: The per-run application context.

    Returns:
        None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Status")

    if ctx.has_data():
        st.sidebar.success("✅ Data loaded")
    else:
        st.sidebar.warning("⚠️ No data loaded")

    if ctx.store.get(StateKeys.DATA_PREPARED, False):
        st.sidebar.success("✅ Data prepared")
    else:
        st.sidebar.warning("⚠️ Data not prepared")

    if ctx.has_models():
        trained_count = len(ctx.store[StateKeys.MODEL_TRAINER].trained_models)
        st.sidebar.success(f"✅ {trained_count} model(s) trained")
    else:
        st.sidebar.warning("⚠️ No models trained")
