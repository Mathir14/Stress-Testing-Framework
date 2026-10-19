"""Nine-page smoke test driven by Streamlit's ``AppTest`` harness.

Architecture reference: ``architecture.md`` §10.3 and roadmap.md Wave 5 gate.

roadmap.md states the Wave 5 smoke test "cannot be automated — ``streamlit`` is
not installed and there is no browser harness".  The second half of that claim is
no longer true: ``streamlit.testing.v1.AppTest`` ships with Streamlit itself and
runs the real script against a real (headless) session, exercising widget
callbacks and reruns.  It cannot verify pixels, scroll position or CSS, so the
manual visual pass is still worth doing, but *script-level* coverage of all nine
pages — previously impossible — is now a gate rather than a hope.

Three tiers, because they need different fixtures:

* **Navigation tier (no fixture needed).**  Select each of the nine sidebar
  labels and assert the page rendered without raising.  This is the tier that
  catches Wave 5's real risk: an extraction that leaves a free variable unbound,
  a widget key collision, or a stale ``st.session_state`` reference.  The
  un-extracted pages legitimately show their "do Modules 1-2 first" warnings
  because no dataset is loaded.
* **State tier (fixture needed).**  Seed a genuinely prepared, trained and
  stressed session — built by the real services, not by clicking — then assert
  every page renders *and* that pages 3-9 no longer show their module-level
  prerequisite warnings.  This is the data upload → train → stress →
  post-stress → calibration → compare → score → report path from the Wave 5
  gate, driven end to end.
* **Upload tier (no fixture needed).**  ``AppTest`` runs the app script
  **in-process**, so ``st.file_uploader`` is an ordinary patchable attribute, and
  Streamlit's ``UploadedFile`` is a ``BytesIO`` subclass, so a plain ``BytesIO``
  is a faithful stand-in.  The genuine upload path is therefore driven for real.
  An earlier revision of this file — and roadmap.md Wave 5, and the run-004
  Critic audit that repeated both — claimed the uploader could not be reached.
  That premise was false, and it was doing real damage: it is what justified
  leaving ``run_batch_stress`` unpressed and what anchored a comment asserting
  ``Load Dataset`` "is covered separately below" for a string that occurred zero
  times in code.  Both are fixed here (ADR-016.8).
* **Action tier (fixture needed).**  Press the mutating buttons on the prepared
  session and assert each one either reports success or renders a
  ``FrameworkError``.  ``AppTest`` **can** press buttons —
  ``app.button(key="…").click().run()`` — and an earlier revision of this
  docstring claimed it could not.  That claim was false, and believing it is
  what left reviewer MAJOR-1 (an uncaught
  ``ParameterOutOfRangeError`` at the ``scale_factor`` slider floor) invisible
  to a fully green suite for an entire review cycle: the defect lived entirely
  inside a button body that no test ever clicked.  The rule this file now
  follows is that a mutating action with no pressed button is untested code, no
  matter how thoroughly its page renders — and the rule is enforced in **two
  halves**, which run-005 reviewer M1 found asserted as one while neither
  existed:

  * the **structural** half is
    ``test_every_mutating_button_in_the_views_is_pressed_here``, an ``ast``
    cross-reference in ``tests/test_architecture.py`` (tier T0) that derives
    every ``st.button`` site in ``views/`` *and* every press declared in this
    file, and requires the two to match in both directions — so a button added
    next month fails that gate until a press is declared for it, and a press
    naming nothing fails it too;
  * the **runtime** half is this file's action tier: the declared presses are
    actually performed with ``AppTest``, and each is asserted to report success
    or a rendered ``FrameworkError``.

  Neither half implies the other — the structural half cannot see a handler
  run, and this file cannot see a button it does not declare — which is why the
  claim is written as two.

Every third-party root this file needs is declared at its own head with
``pytest.importorskip`` **before** it is imported — numpy, pandas, then
streamlit — so a bare (pytest-only) run collects this file and skips it cleanly
rather than turning one missing dependency into a suite-wide collection error.
That ordering is enforced by ``test_t1_test_modules_skip_before_they_import``
in ``tests/test_architecture.py`` (ADR-026).
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the smoke test builds real session data")
pytest.importorskip("pandas", reason="the smoke test builds real session data")
pytest.importorskip("sklearn", reason="the smoke test trains real models")
pytest.importorskip("xgboost", reason="the smoke test trains real models")
pytest.importorskip("plotly", reason="the smoke test renders real figures")
pytest.importorskip("streamlit", reason="Wave 5 smoke test needs the Streamlit runtime")

import numpy as np  # noqa: E402 - after importorskip by design
import pandas as pd  # noqa: E402 - after importorskip by design
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.config import get_config  # noqa: E402
from core.errors import ParameterOutOfRangeError  # noqa: E402
from core.state import StateKeys  # noqa: E402
from views.base import PAGE_KEYS  # noqa: E402

APP = "app.py"

#: The repository root and the *real* artifact root, i.e. the one this module
#: must never write to (conventions §10).
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ARTIFACT_ROOT = REPO_ROOT / "saved_models"

#: A small, well-conditioned binary dataset: enough rows for a 60/20/20 split
#: to give every class representation, and two informative features so the
#: logistic-regression baseline is not degenerate.
SEED = 7

#: Literal fragments of the module-level prerequisite warnings, keyed to the
#: condition each one guards.  Used to assert they disappear once real state
#: exists, and (in the placeholder test) that they appear when it does not.
MODULE_PREREQUISITES = (
    "Please load and prepare data in Module 1 first.",
    "Please train at least one model in Module 2 first.",
    "No trained models found",
    "Complete Modules 1\u20132 first",
)


@pytest.fixture(autouse=True)
def _isolated_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the artifact store at ``tmp_path`` for every test in this module.

    conventions §10: *"Security tests use ``tmp_path``; never touch the real
    ``saved_models/``."*  This module previously honoured that only for
    ``test_model_io.py``; the action tier presses "Save Model", and
    ``ArtifactPolicy.root_dir`` is ``Path("saved_models")`` — **CWD-relative** —
    so every run silently overwrote the developer's real artifacts and rewrote
    their manifest entries.  The damage is unrecoverable and reproduced on every
    single suite run, so the isolation is applied here as an ``autouse``
    fixture rather than per test: a future button press cannot reintroduce it.

    ``core.config._singleton`` is patched rather than the environment, because
    ADR-011.4 rule 3 makes ``STF_*`` readable only at first call and
    ``root_dir`` is deliberately not an ``STF_*`` override.  ``monkeypatch``
    restores the real singleton afterwards.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's patcher.

    Returns:
        Path: The temporary artifact root the store will use.
    """
    root = tmp_path / "saved_models"
    config = get_config()
    monkeypatch.setattr(
        "core.config._singleton",
        replace(config, artifacts=replace(config.artifacts, root_dir=root)),
    )
    return root


def _run(label: str | None = None, **kwargs) -> AppTest:
    """Run the app fresh, optionally selecting a sidebar label."""
    app = AppTest.from_file(APP, default_timeout=180, **kwargs)
    app.run()
    if label is not None:
        app.sidebar.radio[0].set_value(label).run()
    return app


def _alert_counts(app: AppTest) -> tuple[int, int]:
    """Snapshot ``(success, error)`` counts so a press can be shown to add one.

    Both pages in the action tier already render page-level success messages as
    soon as ``_prepared_state`` is loaded ("dataset loaded", "model trained", …),
    so an outcome check that only asked *"is a success present"* would be
    satisfied before the button was ever pressed — a gate that passes for the
    wrong reason, which is the defect ADR-026.4 exists to remove.  The caller
    snapshots before the press; :func:`_assert_clean` compares after.
    """
    return len(app.success), len(app.error)


def _assert_clean(
    app: AppTest,
    context: str,
    *,
    baseline: tuple[int, int] | None = None,
) -> None:
    """Assert the run produced no uncaught exception.

    With ``baseline`` — a :func:`_alert_counts` snapshot taken **before** the
    press — this also enforces ADR-026.4's action-tier contract: the pressed
    button must *add* a success message or a rendered ``FrameworkError``
    (``render_error`` → ``st.error``), rather than merely not raising.
    ``_assert_clean`` alone is satisfied by a handler that silently did nothing,
    and a bare "is any alert present" check is satisfied by the page's own
    pre-press messages; the before/after delta is what makes the assertion
    load-bearing.  Rendering-only tests omit ``baseline``.

    Args:
        app: The finished :class:`AppTest`.
        context: Human-readable description of what was exercised.
        baseline: ``(success, error)`` counts from before the mutating action;
            omit for a rendering-only assertion.

    Returns:
        None
    """
    assert not app.exception, f"{context} raised: {[str(e.value) for e in app.exception]}"
    if baseline is not None:
        success, errors = _alert_counts(app)
        base_success, base_error = baseline
        assert success > base_success or errors > base_error, (
            f"{context} added neither a success message nor a rendered "
            f"FrameworkError (success {base_success}->{success}, "
            f"error {base_error}->{errors}); conventions §7 requires a mutating "
            "action to report one"
        )


@pytest.mark.parametrize("label", PAGE_KEYS)
def test_every_page_renders(label: str) -> None:
    """Each of the nine pages renders without raising from an empty session.

    This is the Wave 5 extraction gate.  Before extraction all nine bodies were
    one ``if/elif`` chain sharing module scope; the pre-verified claim was that
    no name leaked between branches.  This test exercises that claim at runtime
    rather than trusting the static ``ast`` pass alone.
    """
    app = _run(label)
    _assert_clean(app, f"rendering {label!r}")
    assert app.header, f"{label!r} rendered no st.header"


def test_sidebar_labels_are_verbatim() -> None:
    """The sidebar offers exactly the nine labels, in order, emoji intact."""
    app = _run()
    _assert_clean(app, "initial render")
    radio = app.sidebar.radio[0]
    assert radio.options == list(PAGE_KEYS)
    assert radio.value == PAGE_KEYS[0]


def test_footer_status_reflects_empty_session() -> None:
    """The status footer reports the three "not yet done" states on a fresh run."""
    app = _run()
    _assert_clean(app, "initial render")
    messages = [m.value for m in app.sidebar.success] + [m.value for m in app.sidebar.warning]
    assert any("No data loaded" in m for m in messages), messages
    assert any("Data not prepared" in m for m in messages), messages
    assert any("No models trained" in m for m in messages), messages


def _prepared_state() -> dict[str, object]:
    """Build genuine prepared-and-trained session state using the real services.

    ``AppTest`` exposes no *accessor* for ``st.file_uploader``, but it runs the
    app script in-process, so the widget is an ordinary patchable attribute — and
    the genuine upload path is driven for real by
    :func:`test_the_real_upload_path_loads_a_dataset`.  This helper exists for
    the pages whose fixtures need a trained, stressed session, and building that
    through the real :class:`DataManager`, :class:`ModelTrainer` and
    :class:`StressTester` rather than by clicking means a failure here points at
    the view or at the domain contract, not at a fiddly widget interaction.

    Returns:
        dict: Session-state keys and values, ready to assign to
        ``AppTest.session_state``.
    """
    from core.state import StateKeys
    from modules.data_module import DataManager
    from modules.model_module import ModelTrainer
    from modules.stress_module import StressTester

    rng = np.random.default_rng(SEED)
    n = 240
    signal = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "feature_signal": signal + 0.15 * rng.normal(0.0, 1.0, n),
            "feature_noise": noise + 0.15 * rng.normal(0.0, 1.0, n),
            "target": (signal + noise > 0).astype(int),
        }
    )

    manager = DataManager()
    manager.raw_data = frame
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(
        frame, "target", 0.2, 0.1, 42
    )
    manager.X_train, manager.X_val, manager.X_test = manager.scale_features(X_train, X_val, X_test)
    manager.y_train, manager.y_val, manager.y_test = y_train, y_val, y_test

    trainer = ModelTrainer()
    model_name = "Logistic Regression"
    trainer.train_model(model_name, manager.X_train, manager.y_train, random_state=42)
    # A second, stress-tested model so the Post-Stress "Compare Models" branch is
    # reachable: it renders only when two models carry batch results, so with one
    # model that button does not exist and pressing it cannot be tested.
    second_model = "Random Forest"
    trainer.train_model(
        second_model, manager.X_train, manager.y_train, n_estimators=20, random_state=42
    )
    for split, X, y in (
        ("Train", manager.X_train, manager.y_train),
        ("Validation", manager.X_val, manager.y_val),
        ("Test", manager.X_test, manager.y_test),
    ):
        trainer.evaluate_model(model_name, X, y, split)
        trainer.evaluate_model(second_model, X, y, split)

    tester = StressTester(rng=np.random.default_rng(SEED))
    configs = [
        {
            "type": "gaussian_noise",
            "params": {"noise_level": 0.15},
            "name": "Gaussian Noise (Medium)",
        },
        {
            "type": "feature_dropout",
            "params": {"dropout_rate": 0.2},
            "name": "Feature Dropout (20%)",
        },
        {
            "type": "distribution_shift",
            "params": {"shift_type": "mean", "shift_amount": 0.5},
            "name": "Mean Shift (0.5)",
        },
    ]
    results = tester.batch_stress_test(
        trainer.trained_models[model_name], manager.X_test, manager.y_test, configs
    )
    second_results = tester.batch_stress_test(
        trainer.trained_models[second_model], manager.X_test, manager.y_test, configs
    )

    return {
        StateKeys.DATA_MANAGER: manager,
        StateKeys.MODEL_TRAINER: trainer,
        StateKeys.STRESS_TESTER: tester,
        StateKeys.DATA_LOADED: True,
        StateKeys.DATA_PREPARED: True,
        StateKeys.MODEL_TRAINED: True,
        StateKeys.BATCH_STRESS_RESULTS_BY_MODEL: {
            model_name: results,
            second_model: second_results,
        },
        StateKeys.BATCH_STRESS_RESULTS: {
            "model": model_name,
            "dataset": "Test Set",
            "results": results,
        },
    }


def _prepared_state_with_a_saved_artifact() -> dict[str, object]:
    """Return :func:`_prepared_state` with one attested artifact on disk.

    The Baseline page renders the "Load Model" button only when the artifact
    store lists something (``views/module_02_baseline.py``: ``if saved_records:``),
    so reaching that button requires a real saved file.

    Before the reviewer-M1 fix this came from *whatever happened to be lying in
    the developer's working tree* — on a clean checkout the button was absent
    and the press asserted nothing, so the guard it exercises was untested in
    exactly the environment a fresh clone produces.  Seeding it explicitly makes
    the press deterministic, and writes only into the ``tmp_path`` root that
    :func:`_isolated_artifact_root` installed.

    Returns:
        dict: Session state, with one attested ``.pkl`` in the temporary root.
    """
    from core.state import StateKeys
    from modules.model_module import ModelTrainer

    state = _prepared_state()
    trainer = state[StateKeys.MODEL_TRAINER]
    assert isinstance(trainer, ModelTrainer)
    trainer.save_model("Logistic Regression", "seeded_for_load.pkl")
    return state


def test_pages_render_with_prepared_state() -> None:
    """Every page renders against real prepared, trained and stressed state.

    This is the data upload → train → stress → post-stress → calibration →
    compare → score → report path from the roadmap's Wave 5 gate, automated.  It
    is strict about two things: no page may raise, and no page may still show a
    *module-level* prerequisite warning once the state exists.

    Tab-level placeholders ("Run batch stress tests ... first") are out of scope
    *for this test*: this test asserts rendering only, and the button that clears
    them is exercised by the action tier below — ``run_batch_stress`` is a keyed
    button on :data:`MUTATING_BUTTONS`, so the claim is now enforced rather than
    merely made.  Claiming the harness could not press them would be wrong; it can.
    """
    seeded = _prepared_state()
    for index, label in enumerate(PAGE_KEYS):
        app = AppTest.from_file(APP, default_timeout=180)
        app.run()
        for key, value in seeded.items():
            app.session_state[key] = value
        app.run()
        if index:
            app.sidebar.radio[0].set_value(label).run()

        _assert_clean(app, f"rendering {label!r} with prepared state")
        assert app.header, f"{label!r} rendered no st.header"
        placeholders = [
            w.value
            for w in (*app.warning, *app.info)
            if any(phrase in w.value for phrase in MODULE_PREREQUISITES)
        ]
        assert not placeholders, f"{label!r} still shows a placeholder: {placeholders}"


def test_placeholder_warnings_appear_without_state() -> None:
    """Pages 3, 4 and 6 gate on prerequisites and say so when they are unmet.

    The counterpart to :func:`test_pages_render_with_prepared_state`: it proves
    the prerequisite warnings are real UI, not something the prepared-state test
    silently satisfied.  The expected substrings are the views' literal strings,
    so renaming one breaks this test on purpose.
    """
    expected = {
        PAGE_KEYS[2]: "Please load and prepare data in Module 1 first.",
        PAGE_KEYS[3]: "Please load and prepare data in Module 1 first.",
        PAGE_KEYS[5]: "Please train at least one model in Module 2 first.",
    }
    for label, message in expected.items():
        app = _run(label)
        _assert_clean(app, f"rendering {label!r} from an empty session")
        rendered = [w.value for w in (*app.warning, *app.info)]
        assert any(message in text for text in rendered), (
            f"{label!r} did not warn {message!r}; rendered={rendered}"
        )


def _on_page(page_key: str, seeded: dict[str, object]) -> AppTest:
    """Return a live ``AppTest`` sitting on ``page_key`` with ``seeded`` applied.

    Args:
        page_key: One entry of :data:`views.base.PAGE_KEYS`.
        seeded: Session state from :func:`_prepared_state`.

    Returns:
        AppTest: A run that has navigated to ``page_key``, so pressing a button
        on that page is the *next* interaction rather than two interactions.
    """
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    for key, value in seeded.items():
        app.session_state[key] = value
    app.run()
    app.sidebar.radio[0].set_value(page_key).run()
    _assert_clean(app, f"navigating to the {page_key!r} page")
    return app


def _on_stress_page(seeded: dict[str, object]) -> AppTest:
    """Return a live ``AppTest`` sitting on the Stress Testing page.

    Args:
        seeded: Session state from :func:`_prepared_state`.

    Returns:
        AppTest: A run that has navigated to ``PAGE_KEYS[3]`` with ``seeded``
        applied, so a button press is the *next* interaction.
    """
    return _on_page(PAGE_KEYS[3], seeded)


def _rejects_scale_factor(frame, operation, params, *, rng=None):
    """Stand in for ``apply_perturbation``, rejecting exactly as the kernel does."""
    raise ParameterOutOfRangeError(
        "Parameter 'scale_factor' of operator 'scale_perturbation' "
        "must be greater than 1.0; got 1.0.",
        context={
            "operation": operation,
            "parameter": "scale_factor",
            "value": 1.0,
            "exclusive_low": True,
            "params": dict(params),
        },
    )


def _install_rejecting_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the stress view's kernel reference at the rejecting stand-in."""
    monkeypatch.setattr(
        "views.module_04_stress.apply_perturbation", _rejects_scale_factor
    )


@pytest.fixture
def _rejecting_kernel(monkeypatch: pytest.MonkeyPatch):
    """Make ``apply_perturbation`` reject exactly as the kernel does.

    Reviewer MAJOR-1's traceback was reachable through the ``Scale Factor``
    slider, whose floor of ``1.0`` *was* the exclusive lower bound of
    ``StressBounds.scale_factor`` (§4.3).  ADR-015 fixed that at the source by
    raising the floor to ``1.1``, which means **no interaction the shipped UI
    offers can reach the guard any more** — deliberately.

    That makes the guard untestable by dragging a slider, and "the shipped UI
    cannot get there" is not a test.  So the rejection is injected at the kernel
    boundary instead, with the real typed error and the real context payload.  The
    result is a *stronger* gate than the original reproducer: it asserts
    conventions §7 holds for **any** domain rejection, rather than for one widget
    that happened to be misconfigured, and it keeps that coverage alive if a
    config default, a batch parameter or a future widget drifts out of bounds
    again.
    """
    _install_rejecting_kernel(monkeypatch)
    return monkeypatch


def test_a_rejected_parameter_renders_an_error_instead_of_a_traceback(
    _rejecting_kernel: None,
) -> None:
    """Reviewer MAJOR-1: a domain rejection renders, it does not traceback.

    ``views/module_04_stress.py`` calls ``apply_perturbation`` and
    ``stress_tester.evaluate_stress_test`` inside a ``try`` that hands a
    ``FrameworkError`` to ``render_error``.  Before that guard, the same
    interaction escaped as a raw Streamlit traceback with ``app.error`` empty —
    violating conventions §7 ("every mutating action reports success **or** a
    rendered ``FrameworkError``").

    This test exists because the whole defect class is invisible to a suite that
    never presses a button, which is what left it green for a full review cycle.
    See :func:`_rejecting_kernel` for why the rejection is injected rather than
    produced by a mis-set slider.
    """
    app = _on_stress_page(_prepared_state())
    app.selectbox(key="single_stress_type").set_value("Scale Perturbation").run()

    before = _alert_counts(app)
    app.button(key="run_single_stress").click().run()

    _assert_clean(
        app, "pressing Run Stress Test with a rejecting kernel", baseline=before
    )
    errors = [e.value for e in app.error]
    assert any("scale_factor" in text for text in errors), errors
    assert any("greater than 1.0" in text for text in errors), errors


def test_an_accepted_parameter_stores_a_result_and_writes_no_half_state() -> None:
    """The counterpart: the happy path still works after the MAJOR-1 guard.

    Without this, "renders an error" could be satisfied by a handler that simply
    rejects every input.
    """
    from core.state import StateKeys

    app = _on_stress_page(_prepared_state())
    app.selectbox(key="single_stress_type").set_value("Scale Perturbation").run()

    before = _alert_counts(app)
    app.button(key="run_single_stress").click().run()

    _assert_clean(
        app, "pressing Run Stress Test with the default scale_factor", baseline=before
    )
    assert not [e.value for e in app.error], [e.value for e in app.error]
    stored = app.session_state[StateKeys.SINGLE_STRESS_RESULT]
    assert stored["stress_type"] == "Scale Perturbation", stored["stress_type"]


def test_a_rejected_parameter_leaves_no_partial_result_behind(
    _rejecting_kernel: pytest.MonkeyPatch,
) -> None:
    """A rejected run must not overwrite a previously good result.

    ``apply_perturbation`` runs *before* the store write, so the handler stores
    only on success.  This pins the ordering, which is the whole reason the
    ``try`` block wraps the domain call rather than the assignment.

    The good run needs the real kernel, so this fixture-driven test undoes the
    injection for its first half and re-applies it for the second.
    """
    from core.state import StateKeys

    _rejecting_kernel.undo()

    app = _on_stress_page(_prepared_state())
    app.selectbox(key="single_stress_type").set_value("Scale Perturbation").run()
    app.button(key="run_single_stress").click().run()
    good = app.session_state[StateKeys.SINGLE_STRESS_RESULT]

    _install_rejecting_kernel(_rejecting_kernel)

    before = _alert_counts(app)
    app.button(key="run_single_stress").click().run()

    _assert_clean(app, "a rejecting run after a good run", baseline=before)
    assert app.session_state[StateKeys.SINGLE_STRESS_RESULT] == good, (
        "a rejected parameter overwrote the stored result"
    )


#: Every stress operator the single-run tab offers, with the label of the slider
#: whose floor is the most likely place for a bound to be violated.  Read from
#: ADR-015's audit: ``Scale Factor`` was the one slider whose minimum coincided
#: with an *exclusive* kernel bound, which is why the invariant needed a sweep
#: rather than an endpoint check.
STRESS_OPERATORS = (
    ("Gaussian Noise", "Noise Level (std multiplier)"),
    ("Uniform Noise", "Noise Range (fraction of feature range)"),
    ("Feature Dropout", "Dropout Rate"),
    ("Feature Corruption", "Corruption Rate"),
    ("Scale Perturbation", "Scale Factor"),
    ("Distribution Shift", "Shift Amount"),
)


@pytest.mark.parametrize(
    ("operator", "slider_label"),
    STRESS_OPERATORS,
    ids=[op for op, _ in STRESS_OPERATORS],
)
def test_no_slider_position_produces_an_uncaught_error(
    operator: str, slider_label: str
) -> None:
    """ADR-015: no reachable control position yields a traceback (MAJOR-1's class).

    For every operator the single-run tab offers, this drags that operator's
    parameter slider to **both** ends and presses Run Stress Test.  Before
    ADR-015 the ``Scale Perturbation`` / ``Scale Factor`` / floor case produced an
    uncaught ``ParameterOutOfRangeError``; the floor is now ``1.1`` and every
    position of all six sliders is accepted by the kernel, which
    ``tests/test_validation.py`` proves statically for every step.

    ``tests/test_validation.py`` proves the kernel accepts every position; this
    proves the *action tier* has no uncaught failure at any published extreme,
    which is the end-user promise and the thing the static sweep cannot see.
    """
    app = _on_stress_page(_prepared_state())
    app.selectbox(key="single_stress_type").set_value(operator).run()

    slider = next(s for s in app.slider if s.label == slider_label)
    for end in (slider.min, slider.max):
        slider.set_value(end).run()
        before = _alert_counts(app)
        app.button(key="run_single_stress").click().run()
        _assert_clean(
            app,
            f"pressing Run Stress Test for {operator!r} at {slider_label}={end}",
            baseline=before,
        )


#: Every remaining mutating action the harness can reach by ``key``, as
#: ``(page key, button key)``.  Reviewer MINOR-1 asked for "at least one AppTest
#: that presses a mutating button on each page that owns one"; the Calibration
#: page is where that ask became concrete, because ``views/module_06_calibration.py``
#: shipped a ``with render_errors():`` guard naming a symbol the file never
#: imported.  Nothing caught it: ``render_errors`` is only bound when a button is
#: pressed, every page-rendering test passes without pressing one, and the
#: ``ast``-based architecture gate cannot see a runtime ``NameError``.  The
#: defect was live in the tree while 673 tests were green.
MUTATING_BUTTONS = (
    (PAGE_KEYS[2], "generate_predictions"),
    (PAGE_KEYS[3], "run_batch_stress"),
    (PAGE_KEYS[5], "btn_cal_1"),
    (PAGE_KEYS[5], "btn_cal_2"),
    (PAGE_KEYS[5], "btn_cal_3"),
    (PAGE_KEYS[5], "btn_autotemp"),
    (PAGE_KEYS[5], "btn_applytemp"),
)


@pytest.mark.parametrize(
    ("page_key", "button_key"), MUTATING_BUTTONS, ids=[f"{k}@{p}" for p, k in MUTATING_BUTTONS]
)
def test_every_mutating_button_is_reachable_and_runs(page_key, button_key) -> None:
    """Press each remaining mutating action and require no uncaught exception.

    This is the generalisation of the MAJOR-1 tests, which covered one button on
    one page.  conventions §7 requires a mutating action to report success **or**
    a rendered ``FrameworkError``; a ``NameError`` from a half-applied guard
    violates that in the same way a missing ``except`` does, and it is invisible
    to every other tier in this file.

    Args:
        page_key: Sidebar label of the page owning the button.
        button_key: Widget key of the button to press.

    Returns:
        None
    """
    app = _on_page(page_key, _prepared_state())
    button = app.button(key=button_key)
    assert button is not None, f"{button_key!r} is absent from {page_key!r}"

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, f"pressing {button_key!r} on {page_key!r}", baseline=before)


@pytest.mark.parametrize(
    "label",
    (
        "🔄 Calculate Robustness Score",
        "📊 Compare Models",
        "🔍 Generate Recommendations",
    ),
)
def test_every_unkeyed_post_stress_button_runs(label: str) -> None:
    """The Post-Stress page's three buttons carry no ``key``; press them by label.

    They are unreachable by key, which is why the keyed table above cannot cover
    them.  They are the page with the most mutating surface in the app, so
    leaving them unpressed would preserve exactly the gap this tier exists to
    close.

    Args:
        label: The button's visible label.

    Returns:
        None
    """
    app = _on_page(PAGE_KEYS[4], _prepared_state())
    button = next((b for b in app.button if b.label == label), None)
    assert button is not None, f"{label!r} is absent from the Post-Stress page"

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, f"pressing {label!r}", baseline=before)


#: The buttons MINOR-3's remediation newly wrapped in ``render_errors()``, as
#: ``(page key, button label)``.  These are all **unkeyed**, so neither the keyed
#: table above nor the Post-Stress label table reaches them — and they are exactly
#: the handlers that had no guard at all when the reviewer looked, so they are the
#: ones where a bad wrap would be invisible.
#:
#: ``Load Dataset`` renders only when ``st.file_uploader`` returns a file, which
#: no other test in this file arranges, so it is pressed by
#: :func:`test_the_real_upload_path_loads_a_dataset` rather than here with no
#: file.  (This comment previously claimed ``Load Dataset`` "is covered
#: separately below"; the string occurred zero times in code, and
#: ``run_batch_stress`` — asserted by a neighbouring comment to be "exercised by
#: the action tier" — was pressed by nothing.  Both are now real.)
NEWLY_GUARDED_BUTTONS = (
    (PAGE_KEYS[0], "Run Validation"),
    (PAGE_KEYS[0], "Split Dataset"),
    (PAGE_KEYS[1], "💾 Save Model"),
    (PAGE_KEYS[1], "📂 Load Model"),
)

#: The three Data Management buttons that only render when ``raw_data`` is
#: *dirty*, as ``label -> the frame property that reveals it``.  They are absent
#: from a clean frame — the view shows an ``st.info`` instead — so they need their
#: own seeded state rather than a slot in :data:`NEWLY_GUARDED_BUTTONS`.
DIRTY_FRAME_BUTTONS = {
    "Remove Duplicate Rows": "duplicated rows",
    "Apply Missing Value Handling": "null cells",
    "Apply Encoding": "a categorical column",
}


def _with_dirty_raw_data(state: dict[str, object]) -> dict[str, object]:
    """Return ``state`` with a ``raw_data`` frame that reveals every dirty button.

    :func:`_prepared_state` builds a clean, well-conditioned numeric frame, so the
    duplicate / missing-value / encoding buttons never render against it and a
    press would assert nothing.  This adds the one property each needs, without
    disturbing the already-split ``X_train``/``y_train`` the rest of the state
    carries — the split ran before the frame was dirtied, exactly as it would if
    the user uploaded a messy file and had already split an earlier one.

    Args:
        state: Session state from :func:`_prepared_state`.

    Returns:
        dict: The same mapping with ``raw_data`` replaced by a messy frame.
    """
    from core.state import StateKeys

    manager = state[StateKeys.DATA_MANAGER]
    messy = manager.raw_data.copy()
    messy.loc[messy.index[:12], "feature_signal"] = np.nan
    messy["grade"] = pd.Series(["a", "b", "c"] * (len(messy) // 3 + 1))[: len(messy)]
    messy = pd.concat([messy, messy.iloc[[0]]], ignore_index=True)  # a duplicate row

    manager.raw_data = messy
    return state


@pytest.mark.parametrize(
    ("label", "revealed_by"),
    sorted(DIRTY_FRAME_BUTTONS.items()),
    ids=sorted(DIRTY_FRAME_BUTTONS),
)
def test_every_dirty_frame_button_runs(label: str, revealed_by: str) -> None:
    """Press the Data Management buttons that need a messy ``raw_data``.

    These three handlers gained a ``render_errors()`` guard from the MINOR-3
    remediation, and none of them renders at all against
    :func:`_prepared_state`'s clean frame — so without dirty state they would be
    guarded, statically checked, and never executed, which is precisely the hole
    the MAJOR-1 lesson describes.

    Args:
        label: The button's visible label.
        revealed_by: Human-readable frame property that makes the button appear.

    Returns:
        None
    """
    app = _on_page(PAGE_KEYS[0], _with_dirty_raw_data(_prepared_state()))
    # "Apply Encoding" is behind two further gates the frame alone cannot satisfy:
    # the column must be *selected* in the multiselect first.  Seeding the widget
    # value is the same move as seeding session state — there is no click path
    # from "column exists" to "column selected" without driving the multiselect.
    if label == "Apply Encoding":
        multiselect = next(
            (m for m in app.multiselect if m.label.startswith("Select columns to encode")), None
        )
        assert multiselect is not None, "the categorical-column multiselect is absent"
        multiselect.set_value(["grade"]).run()

    button = next((b for b in app.button if b.label == label), None)
    assert button is not None, (
        f"{label!r} is absent even with {revealed_by} in raw_data — the button's "
        "render condition changed, so this test is no longer covering it"
    )

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, f"pressing {label!r} with {revealed_by}", baseline=before)


@pytest.mark.parametrize(
    ("page_key", "label"),
    NEWLY_GUARDED_BUTTONS,
    ids=[f"{label}@{page_key}" for page_key, label in NEWLY_GUARDED_BUTTONS],
)
def test_every_newly_guarded_button_runs(page_key: str, label: str) -> None:
    """Press each handler MINOR-3's fix put behind ``render_errors()``.

    A guard added without a pressed button is the same untested code the MAJOR-1
    lesson was about: ``render_errors`` binds only when its block runs, so a
    mis-nested ``with``, a wrong indentation level or a stale variable reference
    inside the block is invisible to every other tier.  The architecture gate
    proves the guard is *present*; this proves the wrapped body still *executes*.

    Args:
        page_key: Sidebar label of the page owning the button.
        label: The button's visible label.

    Returns:
        None
    """
    # The Load Model button renders only when the artifact store lists a file, so
    # that one case needs a seeded artifact rather than a bare prepared session.
    seeded = (
        _prepared_state_with_a_saved_artifact()
        if label == "📂 Load Model"
        else _prepared_state()
    )
    app = _on_page(page_key, seeded)
    button = next((b for b in app.button if b.label == label), None)
    assert button is not None, f"{label!r} is absent from {page_key!r}"

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, f"pressing {label!r} on {page_key!r}", baseline=before)


def test_load_model_failure_during_evaluation_leaves_model_trained_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If evaluation fails after load_model, MODEL_TRAINED remains False (transactional)."""
    from core.errors import ModelPredictionError

    seeded = _prepared_state_with_a_saved_artifact()
    seeded[StateKeys.MODEL_TRAINED] = False

    def _fail_eval(*args: object, **kwargs: object) -> None:
        raise ModelPredictionError("Evaluation inference failed", context={"stage": "evaluation"})

    app = _on_page(PAGE_KEYS[1], seeded)
    trainer = app.session_state[StateKeys.MODEL_TRAINER]
    monkeypatch.setattr(trainer, "evaluate_model", _fail_eval)

    button = next((b for b in app.button if b.label == "📂 Load Model"), None)
    assert button is not None, "Load Model button is absent from Baseline page"

    button.click().run()

    assert app.session_state[StateKeys.MODEL_TRAINED] is False, (
        "MODEL_TRAINED was set to True even though evaluate_model raised ModelPredictionError"
    )
    assert app.error, "ModelPredictionError should have been rendered as an error alert"


def test_the_train_button_is_guarded_and_runs() -> None:
    """Press the Train button, whose label is an f-string and so not tableable.

    ``views/module_02_baseline.py`` builds its label as ``f"🚀 Train {model_name}"``,
    so it cannot go in :data:`NEWLY_GUARDED_BUTTONS` without hard-coding a model
    name that the seeded state may not use.  It is the most consequential newly
    guarded handler in the app — it sets ``MODEL_TRAINED``, which four later pages
    gate on — so it gets its own press rather than being skipped.

    Returns:
        None
    """
    app = _on_page(PAGE_KEYS[1], _prepared_state())
    button = next((b for b in app.button if b.label.startswith("🚀 Train")), None)
    assert button is not None, "no Train button on the Baseline page"

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, "pressing the Train button", baseline=before)


def test_pressing_save_model_writes_only_into_the_temporary_root(
    tmp_path: Path,
) -> None:
    """Reviewer M1: the Save press must not touch the real ``saved_models/``.

    The action tier above genuinely presses "Save Model", and
    ``ArtifactPolicy.root_dir`` is CWD-relative, so before the ``autouse``
    fixture each run overwrote the developer's artifacts and their manifest
    entries in place.  Reviewer M1 destroyed a real model that way and nothing
    caught it, because the test passed while corrupting the tree.

    This asserts the isolation *behaviourally* — the artifact exists where the
    fixture pointed, and the repository's own root is byte-for-byte unchanged —
    rather than trusting the fixture to still be wired up in a future edit.

    Returns:
        None
    """
    def snapshot() -> dict[str, tuple[int, bytes]]:
        if not REPO_ARTIFACT_ROOT.is_dir():
            return {}
        return {
            path.name: (path.stat().st_size, path.read_bytes())
            for path in sorted(REPO_ARTIFACT_ROOT.iterdir())
            if path.is_file()
        }

    before = snapshot()

    app = _on_page(PAGE_KEYS[1], _prepared_state())
    button = next((b for b in app.button if b.label == "💾 Save Model"), None)
    assert button is not None, "no Save Model button on the Baseline page"
    alerts = _alert_counts(app)
    button.click().run()
    _assert_clean(app, "pressing Save Model", baseline=alerts)

    isolated_root = tmp_path / "saved_models"
    saved = sorted(p.name for p in isolated_root.glob("*.pkl")) if isolated_root.is_dir() else []
    assert saved, (
        f"expected the save to land in {isolated_root}, found nothing; "
        "the artifact root is not isolated from the repository"
    )
    assert snapshot() == before, (
        f"the test run modified {REPO_ARTIFACT_ROOT}; conventions §10 forbids "
        "touching the real saved_models/"
    )


# ── Upload tier: the genuine st.file_uploader path ────────────────────────────

#: A tiny, well-formed CSV.  Four rows is enough for ``DataManager.load_dataset``
#: to succeed and populate ``raw_data``; the upload tier asserts the *path*, not
#: the data quality.
UPLOAD_CSV = (
    b"feature_signal,feature_noise,target\n"
    b"1.2,-0.4,1\n"
    b"0.8,0.3,1\n"
    b"-1.1,0.2,0\n"
    b"-0.6,-0.9,0\n"
)


def _patched_uploader(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[int]:
    """Replace ``st.file_uploader`` with one returning ``payload``; return a counter.

    ``AppTest`` runs ``app.py`` **in-process** — it is a Python object driving a
    real script run, not a subprocess — so ``streamlit.file_uploader`` is an
    ordinary module attribute and patching it is enough to make the view's
    ``uploaded_file`` non-``None``.  Streamlit's ``UploadedFile`` subclasses
    ``io.BytesIO`` and ``pandas.read_csv`` only needs ``read``, so a plain
    ``BytesIO`` is a faithful stand-in rather than a stub that would let a broken
    reader pass.

    roadmap.md Wave 5 and the run-004 Critic both recorded this as impossible.
    They were wrong, and the error was load-bearing: it is what justified leaving
    ``run_batch_stress`` unpressed and what anchored a comment asserting
    ``Load Dataset`` "is covered separately below" (ADR-016.8).

    Args:
        monkeypatch: pytest's patcher, which restores the real widget afterwards.
        payload: The bytes the fake uploader hands back.

    Returns:
        list[int]: A call counter the tests assert against, so they can prove the
        view actually *asked* for a file rather than receiving one some other way.
    """
    calls: list[int] = []

    def _fake(*args: object, **kwargs: object) -> io.BytesIO:
        calls.append(1)
        return io.BytesIO(payload)

    monkeypatch.setattr("streamlit.file_uploader", _fake)
    return calls


def test_the_real_upload_path_loads_a_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``st.file_uploader`` → "Load Dataset" → ``raw_data`` for real.

    This is the interaction roadmap.md Wave 5 and the run-004 Critic both
    declared unreachable.  It is not, and the whole path below the uploader was
    therefore only ever covered by seeding session state by hand — which means the
    ``uploaded_file is not None`` branch, the ``except DatasetLoadError`` arm
    ADR-011.8 turned the old ``if data is not None:`` guard into, and the
    ``data_loaded = True`` + ``display_summary`` pair beside it were all reached
    by nothing.

    Returns:
        None
    """
    calls = _patched_uploader(monkeypatch, UPLOAD_CSV)

    app = _run(PAGE_KEYS[0])
    assert calls, "the Data page never called st.file_uploader; the patch did not take"

    button = next((b for b in app.button if b.label == "Load Dataset"), None)
    assert button is not None, (
        "Load Dataset did not render while the uploader returned a file; the "
        "`uploaded_file is not None` branch is unreachable and its guard untested"
    )

    before = _alert_counts(app)
    button.click().run()
    _assert_clean(app, "pressing Load Dataset with a patched file_uploader", baseline=before)

    assert app.session_state[StateKeys.DATA_LOADED] is True, (
        "the upload handler did not set DATA_LOADED"
    )
    raw = app.session_state[StateKeys.DATA_MANAGER].raw_data
    assert raw is not None, "raw_data is still None after a successful upload"
    assert raw.shape == (4, 3), f"expected the uploaded CSV's (4, 3) shape, got {raw.shape}"
    assert len(app.dataframe) >= 3, "expected summary tables (preview, columns, statistics) to be rendered"


def test_a_malformed_upload_renders_a_dataset_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file ``read_csv`` cannot parse must render, not raise.

    conventions §7 and ADR-011.8: ``load_dataset`` raises ``DatasetLoadError``
    and the view catches it and calls ``render_error``.  That ``except`` arm is
    only reachable with a *bad* file, so it is the branch a good-file-only test
    can never enter — which is why it needs its own press rather than being
    assumed covered by the happy path.

    Returns:
        None
    """
    _patched_uploader(monkeypatch, b"\x00\x01\x02 not a csv at all\xff")

    app = _run(PAGE_KEYS[0])
    button = next((b for b in app.button if b.label == "Load Dataset"), None)
    assert button is not None, "Load Dataset did not render"

    before = _alert_counts(app)
    button.click().run()

    _assert_clean(app, "pressing Load Dataset with an unparseable file", baseline=before)
    assert app.error, (
        "a malformed upload produced no rendered error; the DatasetLoadError arm "
        "of views/module_01_data.py is unreachable or unguarded"
    )
    assert app.session_state[StateKeys.DATA_LOADED] is not True, (
        "DATA_LOADED was set despite the upload failing"
    )


def test_pressing_run_batch_stress_produces_results_for_every_registry_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Press "Run Batch Stress Tests" and require results for all six operators.

    This is the highest-value untested path in the app, and it was untested for a
    specific reason: it dispatches on the six ``snake_case`` registry keys that
    ADR-010.6 declares binding, and renaming any of them silently disables every
    batch run — the unknown-type branch was a ``continue`` until Wave 2 replaced it
    with a raise.  Every *other* stress test in the suite goes through the single
    operator path, so a registry key that only ``batch_stress_test`` uses would
    have had no coverage at all.

    Asserting the count rather than mere absence of an exception is deliberate: a
    handler that builds an empty ``stress_configs`` list and iterates zero times
    raises nothing and passes a bare "no exception" check.

    Returns:
        None
    """
    app = _on_stress_page(_prepared_state())
    # No checkbox setup is needed: all six batch operators in
    # views/module_04_stress.py ship with value=True.  (The loop that used to
    # sit here claimed to "enable every operator rather than only the default
    # selection", but it never ran: its condition — `checkbox.value is False`
    # on a label containing "Stress" — could not hold, since every default is
    # already True and none of the six labels contains that word.)  The
    # `expected <= operators` assertion below is what actually proves this
    # press covered all six registry keys.

    button = app.button(key="run_batch_stress")
    assert button is not None, "run_batch_stress is absent from the Stress page"

    before = _alert_counts(app)
    button.click().run()
    _assert_clean(app, "pressing Run Batch Stress Tests", baseline=before)

    by_model = app.session_state[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL]
    assert by_model, "pressing Run Batch Stress Tests stored no per-model results"

    operators = {
        config["type"]
        for results in by_model.values()
        for result in results.values()
        for config in [result.get("config", {})]
        if isinstance(config, dict) and "type" in config
    }
    expected = {
        "gaussian_noise",
        "uniform_noise",
        "feature_dropout",
        "feature_corruption",
        "scale_perturbation",
        "distribution_shift",
    }
    assert expected <= operators, (
        f"batch stress did not dispatch on every registry key; missing "
        f"{sorted(expected - operators)} from {sorted(operators)}"
    )


def test_dataset_with_missing_values_splits_and_trains_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset containing missing values is auto-cleaned on split and trains cleanly in Module 2."""
    csv_bytes = (
        b"feat_a,feat_b,target\n"
        b"1.0,,0\n"
        b",2.0,0\n"
        b"3.0,3.0,0\n"
        b"4.0,4.0,0\n"
        b"5.0,5.0,0\n"
        b",6.0,0\n"
        b"7.0,7.0,0\n"
        b"8.0,,0\n"
        b"9.0,9.0,0\n"
        b"10.0,10.0,0\n"
        b"11.0,11.0,1\n"
        b"12.0,,1\n"
        b",13.0,1\n"
        b"14.0,14.0,1\n"
        b"15.0,15.0,1\n"
        b"16.0,16.0,1\n"
        b"17.0,,1\n"
        b",18.0,1\n"
        b"19.0,19.0,1\n"
        b"20.0,20.0,1\n"
    )
    _patched_uploader(monkeypatch, csv_bytes)

    # 1. Load dataset on Module 1
    app = _run(PAGE_KEYS[0])
    load_btn = next((b for b in app.button if b.label == "Load Dataset"), None)
    assert load_btn is not None
    load_btn.click().run()
    assert app.session_state[StateKeys.DATA_LOADED] is True

    # 2. Split dataset on Module 1
    split_btn = next((b for b in app.button if b.label == "Split Dataset"), None)
    assert split_btn is not None
    split_btn.click().run()
    assert app.session_state[StateKeys.DATA_PREPARED] is True

    # 3. Navigate to Module 2 and train Logistic Regression
    app.sidebar.radio[0].set_value(PAGE_KEYS[1]).run()
    train_btn = next((b for b in app.button if "Train" in b.label), None)
    assert train_btn is not None
    train_btn.click().run()

    assert app.session_state[StateKeys.MODEL_TRAINED] is True
    success_texts = [s.value for s in app.success]
    assert any("trained successfully!" in t for t in success_texts)


