"""Security boundary tests for ``core.model_io`` (architecture.md §10.2).

**Tier T0.**  This file imports only the standard library and pytest.  It must
never gain a ``numpy`` / ``pandas`` / ``scikit-learn`` / ``xgboost`` import,
directly or transitively — ``core/model_io.py`` is stdlib-only for the same
reason, and the estimator type gate is exercised here through **injected**
``estimator_bases`` rather than by importing ``sklearn.base.BaseEstimator``.  A
security gate that cannot run in a bare environment is not a gate.

``pickle.load`` executes arbitrary Python by construction; see the module
docstring of ``core/model_io.py`` and README ("Security") for why the control
here is containment plus attestation plus informed consent rather than
sandboxing.  These tests pin every containment and attestation control, and pin
the one gate that ``trust=True`` deliberately does *not* bypass.

Both sides of the gate are asserted for every containment case: ``save()`` must
refuse the name (so nothing is ever written outside ``root_dir``) *and* ``load()``
must refuse it (so an attacker-supplied name is never read).
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from core.config import ArtifactPolicy
from core.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPathError,
    ArtifactUntrustedError,
)
from core.model_io import MANIFEST_FORMAT_VERSION, ArtifactRecord, ModelArtifactStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Tier-0 self-check ─────────────────────────────────────────────────────────

#: Distribution names that must never appear in ``sys.modules`` as a result of
#: importing the T0 surface.
_BANNED = ("numpy", "pandas", "sklearn", "streamlit", "plotly", "xgboost")

#: The T0 import surface, re-imported in a clean subprocess.
_T0_IMPORTS = (
    "core.config",
    "core.errors",
    "core.model_io",
    "tests.test_model_io",
)


def _probe_isolated_imports() -> list[str]:
    """Import the T0 surface in a *fresh* interpreter and report leakage.

    The self-check must observe ``sys.modules`` before any T1 test module has
    imported numpy/pandas, so it cannot inspect the live interpreter: pytest
    collects the whole suite into one process, and ``tests/test_validation.py``
    alone is enough to put numpy and pandas in ``sys.modules``.  Asserting on the
    live process made this check pass or fail purely on test *ordering* — it
    failed in a full-suite run and passed in isolation, which is exactly the
    false-confidence failure mode ADR-010.1 was written to prevent.

    A subprocess is the only observation point that is independent of collection
    order, and it is also a stricter test: it proves the modules import cleanly on
    a bare interpreter, not merely that nothing extra is resident.

    Returns:
        list[str]: Dotted names of banned packages found in ``sys.modules``.
    """
    script = (
        "import sys\n"
        f"for _name in {list(_T0_IMPORTS)!r}:\n"
        "    __import__(_name)\n"
        f"banned = {_BANNED!r}\n"
        "print('\\n'.join(sorted("
        "n for n in sys.modules if n.split('.', 1)[0] in banned)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    assert completed.returncode == 0, (
        "T0 surface failed to import on a bare interpreter:\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def test_t0_surface_imports_no_scientific_stack():
    """§10.1: the T0 tier must not import the scientific stack, even transitively.

    Runs in a subprocess so the guarantee is independent of test ordering.  It is
    asserted on ``sys.modules`` rather than on import statements, so a transitive
    promotion (``core.model_io`` importing sklearn) is caught too.
    """
    leaked = _probe_isolated_imports()
    assert not leaked, f"T0 surface leaked third-party imports: {leaked}"


# ── Stand-in estimator ────────────────────────────────────────────────────────


class _FakeEstimator:
    """Minimal picklable stand-in for a fitted estimator.

    Defined at module scope so ``pickle`` can serialise it by reference.  Using a
    local stand-in rather than a real scikit-learn estimator is what keeps this
    file in T0 (ADR-010.1).
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight


@pytest.fixture
def policy(tmp_path: Path) -> ArtifactPolicy:
    """Return a policy rooted in a temporary directory."""
    return ArtifactPolicy(root_dir=tmp_path / "saved_models")


@pytest.fixture
def store(policy: ArtifactPolicy) -> ModelArtifactStore:
    """Return a store whose estimator gate accepts only ``_FakeEstimator``."""
    return ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))


def _write_raw(root: Path, filename: str, payload: bytes) -> Path:
    """Write raw bytes into ``root`` without any manifest entry."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_bytes(payload)
    return target


def _pickle_bytes(obj: object) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


# ── Filename sanitisation ─────────────────────────────────────────────────────

#: Every name the sanitiser must refuse, with the control it exercises.
#: ``"model.pkl\n"`` is the live ``$``-anchor bypass: Python's ``$`` also matches
#: *before* a trailing newline, so an unanchored pattern would admit it.
#: ``"..%2fevil.pkl"`` is the percent-encoded traversal, which is not a path
#: separator on POSIX but is a separator after any URL-decoding proxy.
TRAVERSAL_NAMES = [
    "../evil.pkl",
    "../../evil.pkl",
    "sub/model.pkl",
    "a/b.pkl",
    "a\\b.pkl",
    "..\\evil.pkl",
    "/etc/passwd.pkl",
    "..%2fevil.pkl",
    "evil%2f..pkl",
    "..%2f..%2fetc%2fpasswd.pkl",
    "model.pkl\n",
    "\nmodel.pkl",
    "model.pkl\x00",
    "",
    ".",
    "..",
    "model.txt",
    "model.pickle",
    "model",
    ".hidden.pkl",
    "-leading-dash.pkl",
    " model.pkl",
    "model .pkl",
]


@pytest.mark.parametrize("filename", TRAVERSAL_NAMES)
def test_save_rejects_unsafe_filename(store: ModelArtifactStore, filename: str):
    """§4.5.1: ``save()`` refuses every unsafe name, writing nothing."""
    with pytest.raises(ArtifactPathError) as excinfo:
        store.save(_FakeEstimator(1.0), filename, model_name="Fake")
    assert excinfo.value.context, "an ArtifactPathError must carry context"
    assert not (store.policy.root_dir / filename.lstrip("/")).exists()


@pytest.mark.parametrize("filename", TRAVERSAL_NAMES)
def test_load_rejects_unsafe_filename(store: ModelArtifactStore, filename: str):
    """§4.5.5(a)-(b): ``load()`` refuses the same names — no trust bypass."""
    with pytest.raises(ArtifactPathError):
        store.load(filename)
    with pytest.raises(ArtifactPathError):
        store.load(filename, trust=True)


@pytest.mark.parametrize("filename", TRAVERSAL_NAMES)
def test_resolve_rejects_unsafe_filename(store: ModelArtifactStore, filename: str):
    """§4.5.4: the public ``resolve()`` applies the same policy as ``load()``."""
    with pytest.raises(ArtifactPathError):
        store.resolve(filename)


def test_trust_true_does_not_bypass_traversal(store: ModelArtifactStore):
    """ADR-009 §3: consent skips attestation, never containment.

    ``trust=True`` bypasses the SHA-256 check *only*.  A traversal attempt must
    still raise ``ArtifactPathError`` with ``trust`` set, on a name that a real
    file exists for — so the refusal cannot be an artefact of the file being
    absent.
    """
    root = Path(store.policy.root_dir)
    _write_raw(root.parent, "evil.pkl", _pickle_bytes(_FakeEstimator(9.0)))

    for name in ("../evil.pkl", "/etc/passwd.pkl", "a/b.pkl", "model.pkl\n"):
        with pytest.raises(ArtifactPathError):
            store.load(name, trust=True)


def test_pattern_is_anchored_against_trailing_newline():
    """ADR-009 §1: ``re.fullmatch`` with ``\\Z``, never ``$``.

    Pinned directly against :mod:`re` so the guard cannot be quietly weakened by
    a future edit to ``filename_pattern``.
    """
    import re

    pattern = ArtifactPolicy().filename_pattern
    assert re.fullmatch(pattern, "model.pkl") is not None
    for name in ("model.pkl\n", "model.pkl\r\n", "model.pkl\t", "model.pkl "):
        assert re.fullmatch(pattern, name) is None, (
            f"{name!r} must not match the artifact filename pattern"
        )


def test_over_long_name_is_rejected_without_oserror(policy: ArtifactPolicy):
    """ADR-009 §2: the length cap keeps ``ENAMETOOLONG`` out of the trust path.

    A raw ``OSError`` would cross the module boundary as an untyped failure the
    view layer cannot render.
    """
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    long_name = "a" * (policy.max_filename_length + 1) + ".pkl"
    assert len(long_name) > policy.max_filename_length
    with pytest.raises(ArtifactPathError) as excinfo:
        store.load(long_name, trust=True)
    assert "too long" in str(excinfo.value).lower()
    assert excinfo.value.context["max_filename_length"] == policy.max_filename_length


def test_non_string_name_is_rejected(store: ModelArtifactStore):
    """A ``Path`` or ``None`` is refused: artifacts are addressed by bare name."""
    for bad in (None, 123, Path("model.pkl"), ["model.pkl"]):
        with pytest.raises(ArtifactPathError):
            store.resolve(bad)  # type: ignore[arg-type]


def test_symlinked_artifact_target_is_refused(policy: ArtifactPolicy, tmp_path: Path):
    """§4.5.3: a symlink inside ``root_dir`` is refused, where it points or not.

    Two distinct branches are covered.  A symlink escaping the root is caught by
    the containment check; a symlink whose *target* is also inside the root would
    pass containment and must be caught by the explicit symlink refusal, because
    resolving through it would let a name address bytes the store never attested.
    """
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    root = Path(policy.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    outside = tmp_path / "outside.pkl"
    outside.write_bytes(_pickle_bytes(_FakeEstimator(4.0)))
    (root / "escaping.pkl").symlink_to(outside)

    inside = root / "real.pkl"
    inside.write_bytes(_pickle_bytes(_FakeEstimator(4.0)))
    (root / "linked.pkl").symlink_to(inside)

    for name in ("escaping.pkl", "linked.pkl"):
        with pytest.raises(ArtifactPathError) as excinfo:
            store.load(name, trust=True)
        assert excinfo.value.context, f"{name!r} must raise with context"
    assert "escapes the store root" in str(
        pytest.raises(ArtifactPathError, store.load, "escaping.pkl", trust=True).value
    )
    assert (
        "symbolic link"
        in str(pytest.raises(ArtifactPathError, store.load, "linked.pkl", trust=True).value).lower()
    )


def test_save_refuses_symlinked_target(policy: ArtifactPolicy, tmp_path: Path):
    """The save side refuses a symlink too, so it cannot clobber an outside file."""
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    root = Path(policy.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    (root / "model.pkl").symlink_to(victim)

    with pytest.raises(ArtifactPathError):
        store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    assert victim.read_text(encoding="utf-8") == "original"


def test_symlinked_root_dir_is_refused(tmp_path: Path):
    """ADR-024: a symlinked store root is refused, not resolved through.

    A link as the root would retarget every containment check at a directory the
    store never attested, so ``resolve`` must reject it before it resolves the
    candidate against a followed path.
    """
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    linked_root = tmp_path / "linked_root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    policy = ArtifactPolicy(root_dir=linked_root)
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    with pytest.raises(ArtifactPathError) as excinfo:
        store.resolve("model.pkl")
    assert "symbolic link" in str(excinfo.value).lower()
    assert excinfo.value.context


def test_symlinked_manifest_is_refused(policy: ArtifactPolicy, tmp_path: Path):
    """ADR-024: the attestation manifest may not be a symbolic link."""
    root = Path(policy.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_manifest.json"
    outside.write_text(
        json.dumps({"format_version": MANIFEST_FORMAT_VERSION, "artifacts": {}}),
        encoding="utf-8",
    )
    (root / "manifest.json").symlink_to(outside)

    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.list_artifacts()
    assert "symbolic link" in str(excinfo.value).lower()


def test_load_reads_the_artifact_once(monkeypatch: pytest.MonkeyPatch, store: ModelArtifactStore):
    """ADR-024: ``load`` hashes and deserialises the *same* single read.

    The digest must be computed over the buffer that is deserialised, not by
    re-hashing the path.  ``_sha256`` (the path-based helper that ``verify``
    still uses) is made fatal here, so a return to hash-then-reopen fails loudly
    instead of silently reopening the TOCTOU window.
    """
    store.save(_FakeEstimator(2.0), "model.pkl", model_name="Fake")

    reads: list[Path] = []
    real_read = ModelArtifactStore._read_bytes

    def counting_read(self: ModelArtifactStore, path: Path) -> bytes:
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(ModelArtifactStore, "_read_bytes", counting_read)

    def forbidden(self: ModelArtifactStore, path: Path) -> str:  # pragma: no cover
        raise AssertionError("load() must not re-open the artifact to hash it")

    monkeypatch.setattr(ModelArtifactStore, "_sha256", forbidden)

    model, record = store.load("model.pkl")
    assert isinstance(model, _FakeEstimator)
    assert record.sha256
    assert len(reads) == 1, "the artifact must be read exactly once"


def test_load_binds_the_digest_to_the_deserialised_bytes(
    monkeypatch: pytest.MonkeyPatch, store: ModelArtifactStore
):
    """ADR-024: a buffer swapped after the read cannot slip past attestation.

    If the digest were taken from the path rather than the buffer, returning a
    different buffer from ``_read_bytes`` would still deserialise.  Because the
    digest is bound to that exact buffer, the substituted bytes fail the
    manifest check.
    """
    store.save(_FakeEstimator(2.0), "model.pkl", model_name="Fake")
    tampered = _pickle_bytes({"not": "an estimator"})

    monkeypatch.setattr(
        ModelArtifactStore, "_read_bytes", lambda self, path: tampered
    )

    with pytest.raises(ArtifactIntegrityError):
        store.load("model.pkl", trust=False)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_contained_name_resolves_inside_root(store: ModelArtifactStore):
    """The control is not merely rejecting: a valid name still resolves."""
    resolved = store.resolve("rf_model-1.pkl")
    assert resolved.is_relative_to(Path(store.policy.root_dir).resolve())
    assert resolved.name == "rf_model-1.pkl"


# ── Round trip ────────────────────────────────────────────────────────────────


def test_save_creates_missing_root_dir(policy: ArtifactPolicy):
    """§4.5.10: a fresh checkout has no ``saved_models/`` at all."""
    root = Path(policy.root_dir)
    assert not root.exists()
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    record = store.save(_FakeEstimator(1.5), "model.pkl", model_name="Fake")
    assert root.is_dir()
    assert Path(record.path).is_file()


def test_save_then_load_round_trips(store: ModelArtifactStore):
    """The attested happy path returns the object and its record."""
    store.save(_FakeEstimator(7.5), "model.pkl", model_name="Fake Estimator")
    model, record = store.load("model.pkl")
    assert isinstance(model, _FakeEstimator)
    assert model.weight == 7.5
    assert record.name == "model.pkl"
    assert record.model_name == "Fake Estimator"
    assert record.format_version == MANIFEST_FORMAT_VERSION
    assert record.size_bytes == Path(record.path).stat().st_size


def test_save_does_not_mutate_the_model(store: ModelArtifactStore):
    """Saving must not replace the caller's object with the unpickled copy."""
    original = _FakeEstimator(2.0)
    store.save(original, "model.pkl", model_name="Fake")
    assert original.weight == 2.0
    assert isinstance(original, _FakeEstimator)


# ── Attestation ───────────────────────────────────────────────────────────────


def test_unmanifested_file_is_refused(store: ModelArtifactStore):
    """§4.5.6: no manifest entry means unattested, regardless of file integrity."""
    _write_raw(
        Path(store.policy.root_dir),
        "dropped.pkl",
        _pickle_bytes(_FakeEstimator(5.0)),
    )
    with pytest.raises(ArtifactUntrustedError) as excinfo:
        store.load("dropped.pkl")
    assert "manifest" in str(excinfo.value).lower()


def test_trust_true_loads_an_unmanifested_file(store: ModelArtifactStore):
    """§4.5.6: the documented consent path works, and says so loudly."""
    _write_raw(
        Path(store.policy.root_dir),
        "dropped.pkl",
        _pickle_bytes(_FakeEstimator(5.0)),
    )
    model, record = store.load("dropped.pkl", trust=True)
    assert isinstance(model, _FakeEstimator)
    assert model.weight == 5.0
    # The record is honest about the missing provenance.
    assert record.created_at == ""
    assert record.model_name == ""


def test_trust_true_emits_a_warning(store: ModelArtifactStore, caplog):
    """Consent is logged, so a skipped attestation leaves an audit trail."""
    import logging

    _write_raw(
        Path(store.policy.root_dir),
        "dropped.pkl",
        _pickle_bytes(_FakeEstimator(5.0)),
    )
    with caplog.at_level(logging.WARNING):
        store.load("dropped.pkl", trust=True)
    assert any(
        "trust=True" in record.getMessage() and record.levelno >= logging.WARNING
        for record in caplog.records
    )


def test_single_byte_tampering_is_detected(store: ModelArtifactStore):
    """§4.5.6: the digest is re-checked on every load, not once at save time."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    target = Path(store.resolve("model.pkl"))
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01  # flip exactly one bit; the file stays a readable pickle
    target.write_bytes(bytes(raw))

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.load("model.pkl")
    assert "integrity" in str(excinfo.value).lower()
    assert excinfo.value.context["expected_sha256"] != excinfo.value.context["actual_sha256"]


def test_trust_true_loads_a_tampered_file(store: ModelArtifactStore):
    """A tampered file loads under ``trust=True`` — the consent is explicit.

    The tamper swaps in a *valid but different* pickle, so the only failing gate
    is the digest: this isolates attestation consent from deserialisation.
    """
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    target = Path(store.resolve("model.pkl"))
    original = target.read_bytes()
    target.write_bytes(_pickle_bytes(_FakeEstimator(99.0)))
    assert target.read_bytes() != original

    with pytest.raises(ArtifactIntegrityError):
        store.load("model.pkl")

    model, record = store.load("model.pkl", trust=True)
    assert isinstance(model, _FakeEstimator)
    assert model.weight == 99.0
    # The record reports the hash of what is actually on disk, not the stale
    # attested one — a trusting caller still sees the real provenance gap.
    assert (
        record.sha256
        != ArtifactRecord.from_dict(
            json.loads(store.manifest_path.read_text(encoding="utf-8"))["artifacts"]["model.pkl"]
        ).sha256
    )


def test_truncating_tampering_is_detected(store: ModelArtifactStore):
    """Truncation — the cheapest tamper — is caught before deserialisation."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    target = Path(store.resolve("model.pkl"))
    target.write_bytes(target.read_bytes()[:-4])
    with pytest.raises(ArtifactIntegrityError):
        store.load("model.pkl")


def test_verify_reports_the_manifest_record(store: ModelArtifactStore):
    """§4.5.7: ``verify()`` re-hashes without deserialising."""
    store.save(_FakeEstimator(3.0), "model.pkl", model_name="Fake")
    record = store.verify("model.pkl")
    assert isinstance(record, ArtifactRecord)
    assert record.path == Path(store.resolve("model.pkl"))


def test_missing_file_raises_not_found(store: ModelArtifactStore):
    """An absent file is "not found", not "untrusted" — different UI copy."""
    with pytest.raises(ArtifactNotFoundError):
        store.load("absent.pkl")
    with pytest.raises(ArtifactNotFoundError):
        store.load("absent.pkl", trust=True)


def test_corrupt_manifest_is_treated_as_empty(store: ModelArtifactStore):
    """§4.5.8: a corrupt manifest degrades to "unattested", never to "allowed"."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    store.manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactUntrustedError):
        store.load("model.pkl")


def test_manifest_holding_a_non_dict_is_treated_as_empty(store: ModelArtifactStore):
    """A manifest whose top level is a list is equally unusable."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    store.manifest_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ArtifactUntrustedError):
        store.load("model.pkl")


def test_malformed_entry_is_skipped_not_fatal(store: ModelArtifactStore):
    """One bad entry must not hide every good one."""
    store.save(_FakeEstimator(1.0), "good.pkl", model_name="Good")
    store.save(_FakeEstimator(2.0), "other.pkl", model_name="Other")
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["broken"] = {"name": "broken"}  # missing every other key
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    names = [record.name for record in store.list_artifacts()]
    assert names == ["good.pkl", "other.pkl"], names


# ── Size cap ──────────────────────────────────────────────────────────────────


def test_oversized_file_is_rejected(tmp_path: Path):
    """ADR-010 §2: the size cap the load ordering references actually exists."""
    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models", max_size_bytes=512)
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    _write_raw(Path(policy.root_dir), "big.pkl", b"\x00" * 4096)
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.load("big.pkl", trust=True)
    assert excinfo.value.context["size_bytes"] == 4096
    assert excinfo.value.context["max_size_bytes"] == 512


def test_oversized_save_is_rejected_before_writing(tmp_path: Path):
    """The same cap applies on save, so an oversized blob never lands on disk."""
    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models", max_size_bytes=64)
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.save(_FakeEstimator(), "model.pkl", model_name="Fake")
    assert "exceeding" in str(excinfo.value)
    assert not (Path(policy.root_dir) / "model.pkl").exists()


def test_oversized_file_is_rejected_by_verify(tmp_path: Path):
    """§4.5 step (d) caps three methods, and all three must enforce it.

    ``verify()`` opens with a byte-for-byte copy of ``load()``'s size-cap block,
    which is exactly why it can rot unnoticed: a mutation aimed at ``load()``
    that lands in ``verify()`` raises a ``NameError`` no test ever reaches, so
    the suite stays green and the probe reports a false PASS. Found that way —
    the anchor was ambiguous, not the gate. Asserting the cap here is what makes
    the ADR-001c probe's answer trustworthy rather than lucky.
    """
    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models", max_size_bytes=512)
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    _write_raw(Path(policy.root_dir), "big.pkl", b"\x00" * 4096)
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.verify("big.pkl")
    assert excinfo.value.context["max_size_bytes"] == 512


def test_unserialisable_model_is_typed(tmp_path: Path):
    """A pickle failure is re-typed at the boundary, not leaked as ``TypeError``."""
    from core.errors import ArtifactError

    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models")
    store = ModelArtifactStore(policy, estimator_bases=(_FakeEstimator,))
    with pytest.raises(ArtifactError):
        store.save(lambda: None, "model.pkl", model_name="Fake")  # type: ignore[arg-type]


def test_corrupt_pickle_payload_is_typed(store: ModelArtifactStore):
    """A garbage-but-attested payload raises ``ArtifactIntegrityError``.

    The file is attested by writing it through the store, then overwritten with
    non-pickle bytes and re-attested by a second ``save``-free path: the digest is
    recomputed by hand so the *only* failing gate is deserialisation.
    """
    _write_raw(Path(store.policy.root_dir), "junk.pkl", b"definitely not a pickle")
    manifest = json.loads(
        store.manifest_path.read_text(encoding="utf-8")
        if store.manifest_path.is_file()
        else '{"format_version": 1, "artifacts": {}}'
    )
    import hashlib

    payload = (Path(store.policy.root_dir) / "junk.pkl").read_bytes()
    manifest["artifacts"]["junk.pkl"] = {
        "name": "junk.pkl",
        "path": str(Path(store.policy.root_dir) / "junk.pkl"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "created_at": "2026-01-01T00:00:00+00:00",
        "model_name": "Junk",
        "format_version": MANIFEST_FORMAT_VERSION,
        "library_versions": {},
    }
    Path(store.policy.root_dir).mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        store.load("junk.pkl")


# ── Estimator type gate ───────────────────────────────────────────────────────


def test_non_estimator_payload_is_rejected(store: ModelArtifactStore):
    """§4.5.7(f): a valid pickle of the wrong type is refused after load."""
    _write_raw(Path(store.policy.root_dir), "dict.pkl", _pickle_bytes({"not": "a model"}))
    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "artifacts": {
            "dict.pkl": {
                "name": "dict.pkl",
                "path": str(Path(store.policy.root_dir) / "dict.pkl"),
                "sha256": __import__("hashlib")
                .sha256(_pickle_bytes({"not": "a model"}))
                .hexdigest(),
                "size_bytes": len(_pickle_bytes({"not": "a model"})),
                "created_at": "2026-01-01T00:00:00+00:00",
                "model_name": "Dict",
                "format_version": MANIFEST_FORMAT_VERSION,
                "library_versions": {},
            }
        },
    }
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.load("dict.pkl")
    assert "dict" in str(excinfo.value).lower()


def test_estimator_gate_cannot_be_bypassed_by_trust(store: ModelArtifactStore):
    """``trust=True`` skips attestation, not the type gate."""
    _write_raw(Path(store.policy.root_dir), "list.pkl", _pickle_bytes([1, 2, 3]))
    with pytest.raises(ArtifactIntegrityError):
        store.load("list.pkl", trust=True)


def test_estimator_gate_can_be_disabled_by_injection(tmp_path: Path):
    """``estimator_bases=()`` is the documented explicit opt-out."""
    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models")
    store = ModelArtifactStore(policy, estimator_bases=())
    store.save({"anything": True}, "model.pkl", model_name="Anything")
    model, _record = store.load("model.pkl")
    assert model == {"anything": True}


def test_estimator_gate_defaults_to_resolving_lazily(tmp_path: Path):
    """§4.5.6: the default base is resolved on first use, not at import time.

    Constructing the store with no injected bases must not import scikit-learn —
    that is the whole point of the lazy resolution (ADR-010.1).

    The check is a *before/after delta* on ``sys.modules`` rather than an absolute
    ``"sklearn" not in sys.modules``.  The absolute form is order-dependent: in a
    full-suite run some T1 test module has already imported scikit-learn, so the
    assertion fails for a reason that has nothing to do with
    ``core.model_io``.  The delta measures the property actually under test —
    that *constructing the store* pulls in nothing new.

    ``test_t0_surface_imports_no_scientific_stack`` above is the stronger,
    order-independent proof that the eager import was never introduced.
    """
    policy = ArtifactPolicy(root_dir=tmp_path / "saved_models")
    before = set(sys.modules)
    store = ModelArtifactStore(policy)
    pulled_in = sorted(
        name for name in set(sys.modules) - before if name.split(".", 1)[0] in _BANNED
    )
    assert pulled_in == [], f"constructing the store imported {pulled_in}"

    # The default base is still *correct*: a non-estimator payload is rejected.
    # This half needs scikit-learn, because the default base is resolved by
    # importing it — the very laziness under test.  In a bare T0 environment the
    # import is unavailable and there is nothing to assert about the gate; the
    # delta above is the part that runs everywhere.
    pytest.importorskip("sklearn", reason="resolving the default base requires it")
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    with pytest.raises(ArtifactIntegrityError):
        store.load("model.pkl")


# ── Manifest mechanics ────────────────────────────────────────────────────────


def test_manifest_is_written_atomically(store: ModelArtifactStore):
    """§4.5.11: no ``.tmp`` file survives a successful save."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    leftovers = [p.name for p in Path(store.policy.root_dir).iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_manifest_is_re_readable_from_disk(store: ModelArtifactStore):
    """The manifest on disk is valid JSON with the frozen schema."""
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == MANIFEST_FORMAT_VERSION
    entry = payload["artifacts"]["model.pkl"]
    assert set(entry) == {
        "name",
        "path",
        "sha256",
        "size_bytes",
        "created_at",
        "model_name",
        "format_version",
        "library_versions",
    }


def test_leftover_tmp_file_is_ignored(store: ModelArtifactStore):
    """A crash mid-write leaves a ``.tmp``; the real manifest must still win.

    Simulated by writing a truncated ``.tmp`` beside a complete manifest — the
    state a killed process would leave behind.
    """
    store.save(_FakeEstimator(1.0), "model.pkl", model_name="Fake")
    (Path(store.policy.root_dir) / ".manifest-crashed.tmp").write_text(
        '{"format_version": 1, "artifacts": {"garbage', encoding="utf-8"
    )
    model, _record = store.load("model.pkl")
    assert isinstance(model, _FakeEstimator)
    assert [r.name for r in store.list_artifacts()] == ["model.pkl"]


def test_save_preserves_other_manifest_entries(store: ModelArtifactStore):
    """Re-saving one artifact must not drop the others."""
    store.save(_FakeEstimator(1.0), "a.pkl", model_name="A")
    store.save(_FakeEstimator(2.0), "b.pkl", model_name="B")
    assert [r.name for r in store.list_artifacts()] == ["a.pkl", "b.pkl"]


def test_delete_removes_file_and_entry(store: ModelArtifactStore):
    """§4.5.13: delete drops both halves."""
    store.save(_FakeEstimator(1.0), "a.pkl", model_name="A")
    store.delete("a.pkl")
    assert not Path(store.resolve("a.pkl")).exists()
    assert store.list_artifacts() == []
    with pytest.raises(ArtifactNotFoundError):
        store.delete("a.pkl")


def test_list_artifacts_on_a_fresh_store_is_empty(store: ModelArtifactStore):
    """Absent manifest is an empty list, not an error."""
    assert store.list_artifacts() == []


def test_artifact_record_round_trips_through_dict():
    """§4.5.12: the record's dict form is a lossless schema mapping."""
    record = ArtifactRecord(
        name="model.pkl",
        path=Path("/tmp/model.pkl"),
        sha256="a" * 64,
        size_bytes=17,
        created_at="2026-01-01T00:00:00+00:00",
        model_name="Fake",
        library_versions={"scikit-learn": "1.8.0"},
    )
    assert ArtifactRecord.from_dict(record.to_dict()) == record
