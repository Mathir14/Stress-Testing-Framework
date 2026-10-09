"""Attested, contained model persistence — the framework's security boundary.

Architecture reference: ``architecture.md`` §4.5, ADR-001, ADR-009, ADR-010.1,
ADR-010.2, ADR-011.3.

The honest position on pickle
-----------------------------
``pickle.load`` executes arbitrary Python by construction.  ``joblib`` does not
fix this (it *is* pickle underneath) and a restricted unpickler is not a viable
boundary for arbitrary fitted estimators.  The control implemented here is
therefore **provenance plus containment plus informed consent**, not "safe
deserialisation":

1. Filename sanitisation with ``re.fullmatch`` against a ``\\Z``-anchored
   pattern, on **both** save and load.  ``$`` is deliberately avoided because it
   also matches before a trailing newline and would admit ``"model.pkl\\n"``.
2. Path containment: the resolved candidate must stay inside ``root_dir``, and
   symbolic links are refused outright — a link whose target is *also* inside the
   root would pass containment and still let one name address bytes the store
   never attested.
3. A SHA-256 attestation manifest written atomically at save time.
4. The digest is re-verified **before** ``pickle.load``; a skip requires the
   caller to pass ``trust=True``, which the UI only supplies from an explicit,
   default-unchecked confirmation.
5. An estimator type gate on the loaded object.

Residual risk: a user who ticks the confirmation on a file an attacker placed in
``saved_models/`` still executes that file's code.  This is a single-user local
research tool, not a multi-tenant service; the ADR and the README record the
same trade-off.

Import rules (architecture.md §2.3, ADR-010.1, ADR-011.3): this module is
**stdlib only** — no ``numpy``, ``pandas``, ``scikit-learn`` or ``xgboost``, at
module scope or inside any function.  The estimator base is resolved lazily or
injected, and distribution versions come from ``importlib.metadata``, so
``tests/test_model_io.py`` can exercise the whole security boundary on a bare
interpreter with only pytest installed (§10.1, tier T0).  A security gate that
cannot run in the environment where it matters is not a gate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.config import ArtifactPolicy
from core.errors import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPathError,
    ArtifactUntrustedError,
)

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "ArtifactRecord",
    "ModelArtifactStore",
]

logger = logging.getLogger(__name__)

#: Frozen manifest schema version (architecture.md §4.5.12).  Gaining a field is
#: a breaking change to the on-disk format and requires a new version plus an ADR.
MANIFEST_FORMAT_VERSION = 1

#: The manifest's filename inside ``root_dir``.  It is a fixed constant, never
#: caller-supplied, so it can never be routed through the filename sanitiser.
MANIFEST_FILENAME = "manifest.json"

#: Chunk size for hashing.  Bounded so the read is not a single large allocation.
_HASH_CHUNK_BYTES = 65536


@dataclass(frozen=True)
class ArtifactRecord:
    """Provenance record for one saved artifact.

    Attributes:
        name: The sanitised filename, which is also the manifest key.
        path: Absolute, resolved location on disk.
        sha256: Hex digest of the file contents.
        size_bytes: Size on disk at the time of the record.
        created_at: ISO-8601 UTC timestamp of the save.
        model_name: Human-readable model name supplied by the caller.
        format_version: Manifest schema version of this record.
        library_versions: Provenance metadata, resolved without importing the
            scientific stack.  Never used as an integrity check.
    """

    name: str
    path: Path
    sha256: str
    size_bytes: int
    created_at: str
    model_name: str
    format_version: int = MANIFEST_FORMAT_VERSION
    library_versions: Mapping[str, str] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable manifest entry for this record.

        Returns:
            dict: Exactly the eight manifest fields; ``path`` becomes a string.
        """
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "format_version": self.format_version,
            "library_versions": dict(self.library_versions or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRecord:
        """Rebuild a record from a manifest entry.

        Args:
            payload: A mapping carrying the eight manifest fields.

        Returns:
            ArtifactRecord: The reconstructed record.

        Raises:
            KeyError: If a required field is absent.  Callers iterating the
                manifest treat this as "skip this entry", so a single corrupt
                row cannot hide every good one.
            TypeError: If ``payload`` is not a mapping.
        """
        if not isinstance(payload, Mapping):
            raise TypeError(f"manifest entry must be a mapping, got {type(payload).__name__}")
        return cls(
            name=payload["name"],
            path=Path(payload["path"]),
            sha256=payload["sha256"],
            size_bytes=int(payload["size_bytes"]),
            created_at=payload["created_at"],
            model_name=payload["model_name"],
            format_version=int(payload.get("format_version", MANIFEST_FORMAT_VERSION)),
            library_versions=dict(payload.get("library_versions") or {}),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelArtifactStore:
    """The only sanctioned path for reading or writing a model artifact.

    Attributes:
        policy: The filename, containment and attestation policy in force.
    """

    def __init__(
        self,
        policy: ArtifactPolicy | None = None,
        *,
        estimator_bases: tuple[type, ...] | None = None,
    ) -> None:
        """Create a store.

        No directory is created here: construction must not have side effects,
        so a mere import cannot litter the working tree.  ``save()`` creates
        ``root_dir`` itself (§4.5.10) because ``os.makedirs(os.path.dirname(
        filepath))`` used to raise ``FileNotFoundError`` for a bare filename.

        Args:
            policy: Policy to enforce; the frozen defaults when omitted.
            estimator_bases: Tuple of accepted base types for the type gate.
                ``None`` resolves ``sklearn.base.BaseEstimator`` on first use —
                lazily, so importing this module never imports scikit-learn.
                An empty tuple disables the gate by explicit opt-in.
        """
        self.policy = policy if policy is not None else ArtifactPolicy()
        self._injected_bases = estimator_bases

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def manifest_path(self) -> Path:
        """Location of the attestation manifest inside ``root_dir``."""
        return self.policy.root_dir / MANIFEST_FILENAME

    def resolve(self, filename: str) -> Path:
        """Validate a filename and resolve it inside ``root_dir``.

        Args:
            filename: Bare filename.  Anything else — a ``Path``, an absolute
                path, a separator, a symlink — is refused.

        Returns:
            Path: The resolved absolute path, guaranteed inside ``root_dir``.

        Raises:
            ArtifactPathError: If the name is not a string, exceeds
                ``max_filename_length``, fails ``filename_pattern``, escapes the
                store root, or is a symbolic link.
        """
        if not isinstance(filename, str):
            raise ArtifactPathError(
                f"Model artifacts are addressed by bare filename (str); got "
                f"{type(filename).__name__}.",
                context={
                    "received_type": type(filename).__name__,
                    "max_filename_length": self.policy.max_filename_length,
                },
            )

        length = len(filename)
        if length > self.policy.max_filename_length:
            raise ArtifactPathError(
                f"Filename is too long: {length} characters, maximum is "
                f"{self.policy.max_filename_length}.",
                context={
                    "filename": filename,
                    "length": length,
                    "max_filename_length": self.policy.max_filename_length,
                },
            )

        if not re.fullmatch(self.policy.filename_pattern, filename):
            raise ArtifactPathError(
                f"Invalid model artifact filename {filename!r}. Names must match "
                f"{self.policy.filename_pattern!r}: start with a letter or digit "
                "and contain only letters, digits, dots, underscores and dashes.",
                context={
                    "filename": filename,
                    "pattern": self.policy.filename_pattern,
                    "allowed_suffixes": list(self.policy.allowed_suffixes),
                },
            )

        root = self.policy.root_dir.resolve()
        candidate = root / filename
        is_link = candidate.is_symlink()
        resolved = candidate.resolve()

        if not resolved.is_relative_to(root) or resolved.parent != root:
            raise ArtifactPathError(
                f"Model artifact {filename!r} escapes the store root ({root}).",
                context={"filename": filename, "root_dir": str(root), "resolved": str(resolved)},
            )
        if is_link:
            raise ArtifactPathError(
                f"Model artifact {filename!r} is a symbolic link. The store refuses "
                "to resolve through links so one name cannot address bytes it "
                "never attested.",
                context={"filename": filename, "resolved": str(resolved)},
            )
        return resolved

    # ── Integrity primitives ──────────────────────────────────────────────────

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ArtifactError(
                f"Could not read model artifact {path.name!r}: {exc.strerror or exc}.",
                context={"filename": path.name, "errno": getattr(exc, "errno", None)},
            ) from exc
        return digest.hexdigest()

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError as exc:
            raise ArtifactError(
                f"Could not stat model artifact {path.name!r}: {exc.strerror or exc}.",
                context={"filename": path.name, "errno": getattr(exc, "errno", None)},
            ) from exc

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _read_manifest(self) -> dict[str, Any]:
        """Read the manifest, degrading to an empty one when unusable.

        A corrupt or non-mapping manifest is *unattested*, never *allowed*:
        returning an empty mapping makes every load raise
        ``ArtifactUntrustedError`` instead of silently skipping verification.

        Returns:
            dict: ``{"format_version": int, "artifacts": dict}``.
        """
        path = self.manifest_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"format_version": MANIFEST_FORMAT_VERSION, "artifacts": {}}
        except OSError as exc:
            logger.warning(
                "Could not read %s (%s); treating every artifact as unattested.",
                path,
                exc.strerror or exc,
            )
            return {"format_version": MANIFEST_FORMAT_VERSION, "artifacts": {}}

        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            logger.warning("%s is not valid JSON; treating every artifact as unattested.", path)
            return {"format_version": MANIFEST_FORMAT_VERSION, "artifacts": {}}

        if not isinstance(payload, dict):
            logger.warning(
                "%s does not contain a JSON object; treating every artifact as unattested.",
                path,
            )
            return {"format_version": MANIFEST_FORMAT_VERSION, "artifacts": {}}

        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        payload["artifacts"] = artifacts
        payload.setdefault("format_version", MANIFEST_FORMAT_VERSION)
        return payload

    def _write_manifest(self, payload: Mapping[str, Any]) -> None:
        """Persist the manifest atomically.

        Args:
            payload: The manifest to write.

        Raises:
            ArtifactError: If the root directory or the temporary file cannot
                be created, or the atomic replace fails.
        """
        root = self.policy.root_dir
        tmp_path: str | None = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            handle, tmp_path = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=str(root))
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            os.replace(tmp_path, self.manifest_path)
            tmp_path = None
        except OSError as exc:
            raise ArtifactError(
                f"Could not write the attestation manifest {self.manifest_path}: "
                f"{exc.strerror or exc}.",
                context={"manifest": str(self.manifest_path), "errno": getattr(exc, "errno", None)},
            ) from exc
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:  # pragma: no cover - best-effort cleanup
                    logger.debug("Could not remove temporary manifest %s", tmp_path)

    def _library_versions(self) -> dict[str, str]:
        """Resolve distribution versions through the stdlib only.

        ``importlib.metadata`` reads installed metadata without importing the
        package, which is what keeps this module (and therefore the T0 security
        suite) free of the scientific stack (ADR-011.3).

        Returns:
            dict: Always carries both keys; an absent distribution is recorded
            as ``"<not installed>"`` so the manifest schema stays stable.
        """
        versions: dict[str, str] = {}
        for distribution in ("scikit-learn", "xgboost"):
            try:
                versions[distribution] = importlib_metadata.version(distribution)
            except Exception:  # noqa: BLE001 - metadata lookup must never fail the save
                versions[distribution] = "<not installed>"
        return versions

    # ── Estimator gate ────────────────────────────────────────────────────────

    def _estimator_bases(self) -> tuple[type, ...]:
        """Return the accepted base types, resolving scikit-learn lazily.

        Returns:
            tuple: The injected bases, or ``(BaseEstimator,)`` resolved on first
            use.

        Raises:
            ArtifactError: If scikit-learn is neither installed nor bypassed by
                injection.  Failing closed is the only safe option: silently
                degrading to "no gate" would turn an absent dependency into an
                absent security control.
        """
        if self._injected_bases is not None:
            return self._injected_bases
        try:
            from sklearn.base import BaseEstimator
        except ImportError as exc:
            raise ArtifactError(
                "Cannot verify the model type: scikit-learn is not installed. "
                "Install it, or construct the store with estimator_bases=() to "
                "opt out of the type gate explicitly.",
                context={"sklearn": "not installed"},
            ) from exc
        return (BaseEstimator,)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, model: Any, filename: str, *, model_name: str) -> ArtifactRecord:
        """Serialise a model, attest it and write it atomically.

        Args:
            model: The object to persist.  Not mutated.
            filename: Bare ``.pkl`` filename inside ``root_dir``.
            model_name: Human-readable name recorded in the manifest.

        Returns:
            ArtifactRecord: The attestation record just written.

        Raises:
            ArtifactPathError: The filename failed sanitisation or containment,
                including a symlinked target (which would let a save clobber a
                file outside the store).
            ArtifactIntegrityError: The serialised payload exceeds
                ``max_size_bytes``; nothing is written.
            ArtifactError: The model could not be serialised, or the filesystem
                refused the write.
        """
        path = self.resolve(filename)

        try:
            payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001 - any pickling failure is ours to type
            raise ArtifactError(
                f"Model for {model_name!r} could not be serialised: {type(exc).__name__}: {exc}",
                context={"filename": filename, "model_name": model_name},
            ) from exc

        size_bytes = len(payload)
        if size_bytes > self.policy.max_size_bytes:
            raise ArtifactIntegrityError(
                f"Serialised model is {size_bytes} bytes, exceeding the "
                f"{self.policy.max_size_bytes}-byte artifact limit. Nothing was "
                "written.",
                context={
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "max_size_bytes": self.policy.max_size_bytes,
                },
            )

        try:
            self.policy.root_dir.mkdir(parents=True, exist_ok=True)
            tmp_path: str | None
            handle, tmp_path = tempfile.mkstemp(
                prefix=".artifact-", suffix=".tmp", dir=str(self.policy.root_dir)
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(payload)
                os.replace(tmp_path, path)
                tmp_path = None
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:  # pragma: no cover - best-effort cleanup
                        logger.debug("Could not remove temporary artifact %s", tmp_path)
        except OSError as exc:
            raise ArtifactError(
                f"Could not write model artifact {filename!r}: {exc.strerror or exc}.",
                context={"filename": filename, "errno": getattr(exc, "errno", None)},
            ) from exc

        record = ArtifactRecord(
            name=filename,
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=size_bytes,
            created_at=_utc_now(),
            model_name=model_name,
            format_version=MANIFEST_FORMAT_VERSION,
            library_versions=self._library_versions(),
        )

        manifest = self._read_manifest()
        manifest["format_version"] = MANIFEST_FORMAT_VERSION
        manifest["artifacts"][filename] = record.to_dict()
        self._write_manifest(manifest)
        logger.info(
            "Attested model artifact %s (%d bytes, model_name=%s)",
            filename,
            size_bytes,
            model_name,
        )
        return record

    def list_artifacts(self) -> list[ArtifactRecord]:
        """Return every attested artifact in manifest insertion order.

        Entries missing a required field are skipped rather than fatal, so one
        corrupt row cannot hide every good one.

        Returns:
            list[ArtifactRecord]: The readable records.
        """
        artifacts = self._read_manifest().get("artifacts", {})
        records: list[ArtifactRecord] = []
        for name in artifacts:
            try:
                records.append(ArtifactRecord.from_dict(artifacts[name]))
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed manifest entry %r.", name)
        return records

    def verify(self, filename: str) -> ArtifactRecord:
        """Re-hash an artifact and return its attested record.

        Deserialisation is deliberately *not* performed: this is the check you
        run when you want provenance without executing anything.

        Args:
            filename: Bare ``.pkl`` filename.

        Returns:
            ArtifactRecord: The manifest record, with ``sha256`` and
            ``size_bytes`` refreshed from what is actually on disk.

        Raises:
            ArtifactPathError: Filename failed sanitisation or containment.
            ArtifactNotFoundError: No such file, or no manifest entry.
            ArtifactIntegrityError: The digest does not match, or the size cap
                is exceeded.
        """
        path = self.resolve(filename)
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"No model artifact named {filename!r} in {self.policy.root_dir}.",
                context={"filename": filename, "root_dir": str(self.policy.root_dir)},
            )
        size_bytes = self._file_size(path)
        if size_bytes > self.policy.max_size_bytes:
            raise ArtifactIntegrityError(
                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "
                f"{self.policy.max_size_bytes}-byte limit.",
                context={
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "max_size_bytes": self.policy.max_size_bytes,
                },
            )

        entry = self._read_manifest().get("artifacts", {}).get(filename)
        if entry is None:
            raise ArtifactNotFoundError(
                f"Model artifact {filename!r} has no entry in {MANIFEST_FILENAME}, "
                "so it is unattested and cannot be verified.",
                context={"filename": filename, "manifest": str(self.manifest_path)},
            )

        actual = self._sha256(path)
        expected = entry.get("sha256", "")
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Artifact integrity check failed for {filename!r}: the file on "
                f"disk does not match its attestation digest.",
                context={
                    "filename": filename,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                },
            )
        return self._record_from_entry(filename, path, actual, size_bytes, entry)

    def load(self, filename: str, *, trust: bool = False) -> tuple[Any, ArtifactRecord]:
        """Load an attested artifact, with the full gate chain in order.

        Order (architecture.md §4.5.5): resolve → name-length cap → existence →
        size cap → attestation → estimator type gate.  ``trust=True`` bypasses
        **only** the attestation step: containment controls are not consent
        controls, and neither is the type gate.

        Args:
            filename: Bare ``.pkl`` filename.
            trust: Skip the digest/manifest attestation.  The UI supplies this
                only from an explicit, default-unchecked confirmation stating
                that the file executes arbitrary Python code.

        Returns:
            tuple: ``(model, record)``.  ``record`` always reports the digest of
            what is actually on disk, so a trusting caller still sees the gap.

        Raises:
            ArtifactPathError: Filename failed sanitisation or containment.
            ArtifactNotFoundError: No such file.
            ArtifactIntegrityError: Size cap exceeded, digest mismatch, an
                undeserialisable payload, or a payload that is not an estimator.
            ArtifactUntrustedError: Unattested and ``trust`` not set.
            ArtifactError: The filesystem refused a read.
        """
        path = self.resolve(filename)

        if not path.is_file():
            raise ArtifactNotFoundError(
                f"No model artifact named {filename!r} in {self.policy.root_dir}.",
                context={"filename": filename, "root_dir": str(self.policy.root_dir)},
            )

        size_bytes = self._file_size(path)
        if size_bytes > self.policy.max_size_bytes:
            raise ArtifactIntegrityError(
                f"Model artifact {filename!r} is {size_bytes} bytes, exceeding the "
                f"{self.policy.max_size_bytes}-byte limit.",
                context={
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "max_size_bytes": self.policy.max_size_bytes,
                },
            )

        entry = self._read_manifest().get("artifacts", {}).get(filename)
        actual = self._sha256(path)

        if entry is None:
            if not trust:
                raise ArtifactUntrustedError(
                    f"Model artifact {filename!r} has no attestation record in "
                    f"{MANIFEST_FILENAME}, so it cannot be loaded. Re-save it "
                    "through the app to write a manifest entry, or load it with "
                    "an explicit trust confirmation.",
                    context={"filename": filename, "manifest": str(self.manifest_path)},
                )
            logger.warning(
                "Loading %s with trust=True: it has no attestation record in %s. "
                "Deserialising it executes arbitrary Python code.",
                filename,
                MANIFEST_FILENAME,
            )
        else:
            expected = entry.get("sha256", "")
            if actual != expected and not trust:
                raise ArtifactIntegrityError(
                    f"Artifact integrity check failed for {filename!r}: the file on "
                    f"disk does not match its attestation digest. It may have been "
                    "modified since it was saved.",
                    context={
                        "filename": filename,
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                    },
                )
            if actual != expected:
                logger.warning(
                    "Loading %s with trust=True despite a digest mismatch "
                    "(attested %s, on disk %s).",
                    filename,
                    expected[:12],
                    actual[:12],
                )

        model = self._deserialise(path)
        record = self._record_from_entry(filename, path, actual, size_bytes, entry)
        self._assert_estimator(model, filename)
        logger.info("Loaded model artifact %s (%d bytes)", filename, size_bytes)
        return model, record

    def delete(self, filename: str) -> None:
        """Delete an artifact and its manifest entry.

        Args:
            filename: Bare ``.pkl`` filename.

        Raises:
            ArtifactPathError: Filename failed sanitisation or containment.
            ArtifactNotFoundError: Neither a file nor a manifest entry exists.
            ArtifactError: The filesystem refused the delete.
        """
        path = self.resolve(filename)
        manifest = self._read_manifest()
        artifacts = manifest.get("artifacts", {})
        has_entry = filename in artifacts
        has_file = path.is_file()

        if not has_entry and not has_file:
            raise ArtifactNotFoundError(
                f"No model artifact named {filename!r} to delete.",
                context={"filename": filename, "root_dir": str(self.policy.root_dir)},
            )

        if has_entry:
            del artifacts[filename]
            manifest["artifacts"] = artifacts
            manifest["format_version"] = MANIFEST_FORMAT_VERSION
            self._write_manifest(manifest)

        if has_file:
            try:
                path.unlink()
            except OSError as exc:
                raise ArtifactError(
                    f"Could not delete model artifact {filename!r}: {exc.strerror or exc}.",
                    context={"filename": filename, "errno": getattr(exc, "errno", None)},
                ) from exc
        logger.info("Deleted model artifact %s", filename)

    # ── Internals used by the public methods ──────────────────────────────────

    def _deserialise(self, path: Path) -> Any:
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - a bad payload is ours to type
            raise ArtifactIntegrityError(
                f"Model artifact {path.name!r} could not be deserialised: "
                f"{type(exc).__name__}: {exc}",
                context={"filename": path.name},
            ) from exc

    def _assert_estimator(self, model: Any, filename: str) -> None:
        bases = self._estimator_bases()
        if not bases:
            return
        if not isinstance(model, bases):
            raise ArtifactIntegrityError(
                f"Model artifact {filename!r} loaded as "
                f"{type(model).__name__!r}, which is not a supported estimator. "
                f"Accepted types: "
                f"{', '.join(base.__name__ for base in bases)}.",
                context={
                    "filename": filename,
                    "loaded_type": type(model).__name__,
                    "accepted": [base.__name__ for base in bases],
                },
            )

    @staticmethod
    def _record_from_entry(
        filename: str,
        path: Path,
        actual_sha256: str,
        size_bytes: int,
        entry: Mapping[str, Any] | None,
    ) -> ArtifactRecord:
        payload = entry if isinstance(entry, Mapping) else {}
        return ArtifactRecord(
            name=filename,
            path=path,
            sha256=actual_sha256,
            size_bytes=size_bytes,
            created_at=payload.get("created_at", ""),
            model_name=payload.get("model_name", ""),
            format_version=int(payload.get("format_version", MANIFEST_FORMAT_VERSION)),
            library_versions=dict(payload.get("library_versions") or {}),
        )
