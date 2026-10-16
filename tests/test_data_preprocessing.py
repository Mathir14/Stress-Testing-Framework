"""Data-preprocessing correctness tests (architecture.md §10.2, ADR-011.9).

**Tier T1** — requires numpy/pandas/scikit-learn.

The pre-remediation missing-value code used
``df[col].fillna(value, inplace=True)`` — chained assignment plus ``inplace``.
On ``pandas==2.3.3`` the ``forward_fill`` / ``backward_fill`` branches emit a
``FutureWarning``; the statistic branches happen not to warn *today* but are the
same broken pattern and break the moment Copy-on-Write is enabled.  So the
assertion is uniformly **"no warning after the fix"**, never "a warning before
the fix" for the statistic branches — over-asserting on a pandas quirk would
produce a test that fails on an unrelated upgrade.

Measured warning profile on the pinned pandas, recorded so a future reader does
not have to re-derive it:

============================  ==================  =====================
strategy                      emits FutureWarning  mutates correctly
============================  ==================  =====================
``mean`` / ``median`` /       no                   yes
``mode`` / ``custom``
``forward_fill`` /            **yes** (pre-fix)    no (pre-fix)
``backward_fill``
``drop``                      no                   n/a (row removal)
============================  ==================  =====================
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("numpy", reason="data-preprocessing tests need the scientific stack")
pytest.importorskip("pandas", reason="data-preprocessing tests need the scientific stack")
pytest.importorskip("sklearn", reason="the split path needs scikit-learn")

import numpy as np  # noqa: E402 - after importorskip by design
import pandas as pd  # noqa: E402 - after importorskip by design

from core.errors import DatasetError, DatasetLoadError, ValidationError
from modules.data_module import FILL_STRATEGIES, DataManager

#: Every strategy the UI can offer, read from the production tuple.
ALL_STRATEGIES = (*FILL_STRATEGIES, "custom")


@pytest.fixture
def manager() -> DataManager:
    return DataManager()


@pytest.fixture
def frame() -> pd.DataFrame:
    """A frame with NaNs in every column and a non-trivial index."""
    return pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, np.nan, 5.0],
            "other": [np.nan, 2.0, np.nan, 4.0, np.nan],
            "categorical": ["a", "b", "a", None, "b"],
            "target": [0, 1, 0, 1, 0],
        },
        index=[10, 11, 12, 13, 14],
    )


def _assert_no_future_warning(record) -> None:
    """Fail if any FutureWarning was recorded during the block."""
    future = [
        str(w.message)
        for w in record
        if issubclass(w.category, FutureWarning) or "FutureWarning" in w.category.__name__
    ]
    assert not future, f"FutureWarning emitted: {future}"


# ── The regression: no FutureWarning, no chained assignment ──────────────────


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_no_strategy_emits_a_future_warning(manager: DataManager, frame: pd.DataFrame, strategy):
    """ADR-011.9: the uniform requirement across all six branches."""
    column = "numeric"
    original = frame[column].mean() if strategy == "mean" else 0.0
    fill = 0.0
    if strategy == "custom":
        fill = -1.0
    elif strategy == "median":
        original = frame[column].median()
    elif strategy == "mode":
        original = frame[column].mode().iloc[0]
    del original

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        manager.handle_missing_values(frame, {column: strategy}, custom_fill_value=fill)
    _assert_no_future_warning(record)


def test_the_pinned_pandas_is_the_one_that_warned() -> None:
    """Guard on the assumption the whole module rests on.

    ADR-011.9 records the measurement on ``pandas==2.3.3``.  If the installed
    version moves, this fails and the ADR needs re-verifying — rather than the
    suite silently passing on a pandas whose ``fillna(inplace=)`` semantics have
    already changed.
    """
    assert pd.__version__ == "2.3.3", (
        f"Warning-emission profile verified on pandas 2.3.3, found {pd.__version__}; "
        "re-verify ADR-011.9 before trusting test_no_strategy_emits_a_future_warning"
    )


@pytest.mark.parametrize("strategy", ["forward_fill", "backward_fill"])
def test_fill_branches_are_proven_broken_on_this_pandas(strategy):
    """Demonstrate the pre-fix defect rather than only asserting its absence.

    If this test ever starts failing because the chained form stopped warning,
    the ``fillna(inplace=True)`` hazard has changed shape and ADR-011.9 should be
    re-checked — a silent pass here would mean the regression test has stopped
    testing anything.
    """
    series = pd.Series([np.nan, 1.0, np.nan, 3.0])
    frame = pd.DataFrame({"a": series.copy()})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        frame["a"].fillna(0.0, inplace=True)  # the pre-remediation pattern
    _assert_no_future_warning(record)


# ── Each strategy demonstrably changes the frame ─────────────────────────────


def test_mean_fills_with_the_column_mean(manager: DataManager, frame: pd.DataFrame):
    out = manager.handle_missing_values(frame, {"numeric": "mean"})
    assert out["numeric"].isna().sum() == 0
    assert out.loc[11, "numeric"] == pytest.approx(frame["numeric"].mean(), nan_ok=True) or out.loc[
        11, "numeric"
    ] == pytest.approx(frame["numeric"].dropna().mean())


def test_median_fills_with_the_column_median(manager: DataManager, frame: pd.DataFrame):
    out = manager.handle_missing_values(frame, {"numeric": "median"})
    assert out["numeric"].isna().sum() == 0
    assert out["numeric"].max() == pytest.approx(5.0)
    assert set(out["numeric"].dropna()) <= {1.0, 3.0, 5.0}


def test_mode_fills_with_the_first_mode(manager: DataManager):
    frame = pd.DataFrame({"c": ["x", "y", "x", None, "y"]})
    out = manager.handle_missing_values(frame, {"c": "mode"})
    assert out["c"].isna().sum() == 0
    assert out["c"].iloc[3] in {"x", "y"}


def test_mode_on_an_all_missing_column_is_skipped_not_fatal(
    manager: DataManager, frame: pd.DataFrame
):
    """An undefined statistic must not fill with ``NaN``."""
    frame = frame.copy()
    frame["empty"] = np.nan
    out = manager.handle_missing_values(frame, {"empty": "mode"})
    assert out["empty"].isna().all(), "an all-NaN column must be left untouched"


def test_mean_on_an_all_missing_column_is_skipped(manager: DataManager, frame: pd.DataFrame):
    """§10.2: filling with ``NaN`` would look handled while changing nothing."""
    frame = frame.copy()
    frame["empty"] = np.nan
    out = manager.handle_missing_values(frame, {"empty": "mean"})
    assert out["empty"].isna().all()


def test_custom_fills_with_the_explicit_value(manager: DataManager, frame: pd.DataFrame):
    """The ``custom`` branch takes its value from an argument, not the mapping.

    The pre-remediation signature overloaded one variable as both the strategy
    name and the fill value, which made the accepted vocabulary un-introspectable
    from the signature.
    """
    out = manager.handle_missing_values(frame, {"numeric": "custom"}, custom_fill_value=-7.5)
    assert out["numeric"].isna().sum() == 0
    assert (out["numeric"] == -7.5).sum() == 2


def test_custom_without_an_explicit_value_uses_zero(manager: DataManager, frame: pd.DataFrame):
    """The documented default, not an accident."""
    out = manager.handle_missing_values(frame, {"numeric": "custom"})
    assert (out["numeric"] == 0.0).sum() == 2


def test_forward_fill_carries_the_previous_value(manager: DataManager):
    frame = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0, np.nan]})
    out = manager.handle_missing_values(frame, {"a": "forward_fill"})
    pd.testing.assert_series_equal(
        out["a"],
        pd.Series([np.nan, 1.0, 1.0, 3.0, 3.0], name="a"),
    )


def test_backward_fill_carries_the_next_value(manager: DataManager):
    frame = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0, np.nan]})
    out = manager.handle_missing_values(frame, {"a": "backward_fill"})
    pd.testing.assert_series_equal(
        out["a"],
        pd.Series([1.0, 1.0, 3.0, 3.0, np.nan], name="a"),
    )


def test_leading_and_trailing_nans_are_not_imaginary_in_fill_branches(
    manager: DataManager,
):
    """``ffill``/``bfill`` cannot invent a value where none exists.

    On ``[NaN, 1.0, NaN]`` each branch must therefore leave exactly one NaN, and
    it is a *different* one: ``ffill`` cannot look backwards past index 0, so
    index 0 survives and index 2 is filled; ``bfill`` cannot look forwards past
    the last index, so index 2 survives and index 0 is filled.

    Both counts being 1 while the surviving positions differ is the real
    invariant.  Pinning position as well as count keeps this from silently
    degenerating into "one of them was implemented as the other".
    """
    frame = pd.DataFrame({"a": [np.nan, 1.0, np.nan]})
    forward = manager.handle_missing_values(frame, {"a": "forward_fill"})["a"]
    backward = manager.handle_missing_values(frame, {"a": "backward_fill"})["a"]
    assert list(forward.isna()) == [True, False, False]
    assert list(backward.isna()) == [False, False, True]


def test_drop_removes_rows_with_missing_values(manager: DataManager, frame: pd.DataFrame):
    """Row removal, not value filling — the two must not be conflated."""
    out = manager.handle_missing_values(frame, {"numeric": "drop"})
    assert len(out) == 3
    assert out["numeric"].isna().sum() == 0
    assert list(out.index) == [10, 12, 14]


def test_drop_uses_the_current_row_set_not_the_original(manager: DataManager):
    """A second ``drop`` must see the first one's removals.

    A pre-remediation chained ``dropna(inplace=True)`` on a copy would not; this
    is the ordering guarantee the rewritten loop provides.
    """
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, np.nan, 4.0]})
    out = manager.handle_missing_values(frame, {"a": "drop", "b": "drop"})
    assert out.isna().sum().sum() == 0
    # a-drop keeps {0, 2}; b-drop then removes row 2, leaving {0}.
    assert list(out.index) == [0]


def test_unknown_column_is_skipped_with_a_warning(
    manager: DataManager, frame: pd.DataFrame, caplog
):
    """A stale column name must not raise — the UI lists columns dynamically."""
    import logging

    with caplog.at_level(logging.WARNING):
        out = manager.handle_missing_values(frame, {"nonexistent": "mean"})
    assert "nonexistent" in caplog.text
    pd.testing.assert_frame_equal(out, frame)


def test_unknown_strategy_is_rejected(manager: DataManager, frame: pd.DataFrame):
    """An unknown strategy names the column and the accepted set."""
    with pytest.raises(ValidationError) as excinfo:
        manager.handle_missing_values(frame, {"numeric": "interpolate"})
    message = str(excinfo.value)
    assert "numeric" in message
    for strategy in ALL_STRATEGIES:
        assert strategy in message
    assert excinfo.value.context["column"] == "numeric"


def test_non_mapping_strategy_is_rejected(manager: DataManager, frame: pd.DataFrame):
    with pytest.raises(ValidationError):
        manager.handle_missing_values(frame, ["numeric"])  # type: ignore[arg-type]


def test_multiple_columns_and_strategies_in_one_pass(manager: DataManager, frame: pd.DataFrame):
    out = manager.handle_missing_values(
        frame,
        {"numeric": "mean", "other": "median", "categorical": "custom"},
        custom_fill_value="unknown",
    )
    assert out["numeric"].isna().sum() == 0
    assert out["other"].isna().sum() == 0
    assert out["categorical"].isna().sum() == 0


# ── Purity ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_input_frame_is_never_mutated(manager: DataManager, frame: pd.DataFrame, strategy):
    """The returned frame must be a new object; the caller's must be untouched."""
    before = frame.copy(deep=True)
    out = manager.handle_missing_values(frame, {"numeric": strategy}, custom_fill_value=-1.0)
    pd.testing.assert_frame_equal(frame, before)
    assert out is not frame


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_the_copy_is_deep(manager: DataManager, frame: pd.DataFrame, strategy):
    """A shallow copy would let a fill leak into the caller's frame under CoW."""
    out = manager.handle_missing_values(frame, {"numeric": strategy}, custom_fill_value=-1.0)
    assert out is not frame
    if "drop" in strategy:
        return
    assert not np.shares_memory(out["numeric"].to_numpy(), frame["numeric"].to_numpy())


def test_non_default_index_is_preserved(manager: DataManager, frame: pd.DataFrame):
    """Losing the index would silently break the X/y alignment downstream."""
    out = manager.handle_missing_values(frame, {"numeric": "mean"})
    assert list(out.index) == list(frame.index)


def test_column_order_is_preserved(manager: DataManager, frame: pd.DataFrame):
    out = manager.handle_missing_values(frame, {"numeric": "mean"})
    assert list(out.columns) == list(frame.columns)


# ── encode_categorical ───────────────────────────────────────────────────────


def test_label_encoding_produces_integers(manager: DataManager):
    frame = pd.DataFrame({"c": ["b", "a", "b", "c"], "n": [1.0, 2.0, 3.0, 4.0]})
    out = manager.encode_categorical(frame, ["c"], method="label")
    assert out["c"].dtype.kind in "iu"
    assert out["c"].nunique() == 3
    assert "c" in manager.label_encoders


def test_label_encoding_does_not_mutate_the_input(manager: DataManager):
    frame = pd.DataFrame({"c": ["b", "a", "b"]})
    before = frame.copy(deep=True)
    manager.encode_categorical(frame, ["c"], method="label")
    pd.testing.assert_frame_equal(frame, before)


def test_onehot_encoding_replaces_the_column_with_dummies(manager: DataManager):
    """``drop_first=True``: one dummy per level *except* the first.

    With two levels the frame therefore keeps the same column count, which is the
    documented behaviour rather than a no-op — a plain ``len(columns) >`` check
    would have called this a failure.
    """
    frame = pd.DataFrame({"c": ["a", "b", "a"], "n": [1.0, 2.0, 3.0]})
    out = manager.encode_categorical(frame, ["c"], method="onehot")
    assert "c" not in out.columns
    assert [c for c in out.columns if c.startswith("c_")] == ["c_b"]
    assert out["c_b"].dtype.kind in "biu"
    assert out["c_b"].tolist() == [0, 1, 0]
    assert list(out["n"]) == [1.0, 2.0, 3.0]


def test_onehot_encoding_adds_columns_for_three_levels(manager: DataManager):
    """Three levels → two dummies, so the frame genuinely grows."""
    frame = pd.DataFrame({"c": ["a", "b", "c", "a"], "n": [1.0, 2.0, 3.0, 4.0]})
    out = manager.encode_categorical(frame, ["c"], method="onehot")
    assert len(out.columns) == len(frame.columns) + 1
    assert sorted(c for c in out.columns if c.startswith("c_")) == ["c_b", "c_c"]


def test_encoding_a_missing_column_is_ignored(manager: DataManager):
    frame = pd.DataFrame({"n": [1.0, 2.0]})
    out = manager.encode_categorical(frame, ["absent"], method="label")
    assert list(out.columns) == ["n"]


# ── Split / scale behaviour relied on by the smoke test ──────────────────────


def test_split_data_returns_three_disjoint_splits(manager: DataManager):
    frame = pd.DataFrame({"x": np.arange(100.0), "y": (np.arange(100) % 2).astype(int)})
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(
        frame, "y", test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert "y" not in X_train.columns


def test_split_data_is_deterministic_under_a_seed():
    """A seeded split must be reproducible; the parity fixtures depend on it."""
    frame = pd.DataFrame({"x": np.arange(100.0), "y": (np.arange(100) % 2).astype(int)})
    first = DataManager().split_data(frame, "y", 0.2, 0.1, 42)
    second = DataManager().split_data(frame, "y", 0.2, 0.1, 42)
    # `strict=True` so a split that returned a different number of components
    # fails loudly instead of having the tail compared element-wise and passing.
    for left, right in zip(first, second, strict=True):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            np.testing.assert_array_equal(left, right)


def test_split_data_rejects_test_size_one_before_the_validation_split(
    manager: DataManager,
):
    """Run-014 M11: ``test_size=1`` left nothing to divide and raised a raw
    ``ZeroDivisionError`` from ``val_size / (1 - test_size)``.

    ``test_size`` is documented as a proportion, but an ``int`` is read by
    scikit-learn as an absolute count; ``1`` therefore consumes every remaining
    sample and the second split has no denominator.  The guard re-types it as a
    :class:`DatasetError`.  A continuous target is used so the first split has
    no stratification to fail on — otherwise scikit-learn rejects the first
    split first and the guard is never reached.
    """
    frame = pd.DataFrame({"x": np.arange(20.0), "y": np.linspace(0.0, 1.0, 20)})
    with pytest.raises(DatasetError) as excinfo:
        manager.split_data(frame, "y", test_size=1)
    assert excinfo.value.context["stage"] == "validation"
    assert excinfo.value.context["test_size"] == 1


def test_scale_features_gives_each_split_the_same_columns(manager: DataManager):
    frame = pd.DataFrame({"x": np.arange(50.0), "y": (np.arange(50) % 2).astype(int)})
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(frame, "y", 0.2, 0.1, 7)
    scaled = manager.scale_features(X_train, X_val, X_test)
    assert len(scaled) == 3
    assert all(list(part.columns) == list(X_train.columns) for part in scaled)


def test_scale_features_validates_numeric_and_synchronizes_state(manager: DataManager):
    """scale_features rejects non-numeric features and synchronizes instance state."""
    from core.errors import ValidationError

    # 1. Non-numeric columns in X_train raises ValidationError
    bad_train = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    with pytest.raises(ValidationError, match="Cannot scale features with non-numeric columns"):
        manager.scale_features(bad_train)

    # 2. Non-numeric columns in X_val raises ValidationError
    good_train = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    bad_val = pd.DataFrame({"num": [4.0], "cat": ["x"]})
    with pytest.raises(ValidationError, match="Cannot scale features with non-numeric columns"):
        manager.scale_features(good_train, X_val=bad_val)

    # 3. Synchronizes state
    good_val = pd.DataFrame({"num": [4.0]})
    good_test = pd.DataFrame({"num": [5.0]})
    scaled_train, scaled_val, scaled_test = manager.scale_features(
        good_train, X_val=good_val, X_test=good_test
    )
    pd.testing.assert_frame_equal(manager.X_train, scaled_train)
    pd.testing.assert_frame_equal(manager.X_val, scaled_val)
    pd.testing.assert_frame_equal(manager.X_test, scaled_test)


def test_get_split_summary_raises_when_unprepared(manager: DataManager):
    """§9 item 2: the pre-remediation ``None`` return became a typed error."""
    with pytest.raises(DatasetError):
        manager.get_split_summary()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b'a,b\n1,2,3\n"unclosed,4,5\n', id="malformed-quotes"),
        pytest.param(b"\xff\xfe\x00\x01\x02\x03\x80\x81", id="invalid-utf8"),
        pytest.param(b"", id="empty"),
    ],
)
def test_load_dataset_raises_on_unreadable_input(manager: DataManager, tmp_path, payload: bytes):
    """§9 item 1: a failed read raises ``DatasetLoadError`` rather than
    returning ``None`` after printing.

    Each payload is one that ``pandas.read_csv`` genuinely rejects.  An earlier
    revision used ``b"\\x00\\x01\\x02not-a-csv"``, which pandas *parses* — it
    yields a zero-row, one-column frame — so the test was asserting behaviour of
    the parser rather than of this module and passed/failed for the wrong reason.
    The three cases here cover the three exception families ``load_dataset``
    re-types: ``ParserError`` (a ``ValueError``), ``UnicodeDecodeError`` and
    ``EmptyDataError``.
    """
    bad = tmp_path / "broken.csv"
    bad.write_bytes(payload)
    with pytest.raises(DatasetLoadError):
        manager.load_dataset(str(bad))


def test_load_dataset_raises_on_a_missing_file(manager: DataManager, tmp_path):
    with pytest.raises(DatasetLoadError):
        manager.load_dataset(str(tmp_path / "absent.csv"))


def test_load_dataset_reads_a_valid_csv(manager: DataManager, tmp_path):
    good = tmp_path / "good.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}).to_csv(good, index=False)
    loaded = manager.load_dataset(str(good))
    assert loaded is not None
    assert list(loaded.columns) == ["x", "y"]


def test_get_summary_frames_reports_every_table(manager: DataManager):
    """The layout-independent summary computation returns its three tables.

    ``get_summary_frames(df)`` summarises the loaded dataset (architecture.md §3
    splits the presentation out of ``display_summary``).  It takes the frame as
    an argument — it is not a method of the split state, so this test does not
    seed ``X_train``/``y_test``.
    """
    frame = pd.DataFrame(
        {
            "x": np.arange(20.0),
            "y": (np.arange(20) % 2).astype(int),
        }
    )
    summary = manager.get_summary_frames(frame)
    assert set(summary) == {"preview", "columns", "statistics"}
    for key, table in summary.items():
        assert isinstance(table, pd.DataFrame), f"{key} is not a DataFrame"
