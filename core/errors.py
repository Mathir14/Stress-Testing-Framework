"""Typed exception hierarchy for the framework.

Architecture reference: ``architecture.md`` §4.1, conventions §3.

Every domain failure crosses a module boundary as one of these types.  Domain
code **raises**; the view layer catches and renders (ADR-004).  ``None`` is never
used as a failure signal.

Machine context lives in :attr:`FrameworkError.context`, never in the message:
the message is rendered to the user and must stay an actionable sentence, while
``context`` is for tests, logging and programmatic handling.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FrameworkError",
    "ConfigurationError",
    "ValidationError",
    "ParameterOutOfRangeError",
    "UnsupportedStressTypeError",
    "DatasetError",
    "DatasetLoadError",
    "ArtifactError",
    "ArtifactPathError",
    "ArtifactNotFoundError",
    "ArtifactIntegrityError",
    "ArtifactUntrustedError",
    "UnknownModelError",
    "ModelNotTrainedError",
    "ReportExportError",
]


class FrameworkError(Exception):
    """Base class for every error the framework raises deliberately.

    Attributes:
        context: Machine-readable payload describing the failure.  Always a
            ``dict`` (never ``None``) so callers can index it unconditionally.
    """

    context: dict[str, Any]

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        """Build the error.

        Args:
            message: User-actionable sentence naming the offending input and
                the accepted values.
            context: Optional machine context merged into :attr:`context`.
        """
        super().__init__(message)
        self.context = dict(context or {})


class ConfigurationError(FrameworkError):
    """A configuration value or override is unusable."""


class ValidationError(FrameworkError):
    """An input failed validation.

    ``field`` and ``value`` are convenience accessors for the two most common
    context entries; the full payload is still available through
    :attr:`context`.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Build the error.

        Args:
            message: User-actionable sentence.
            field: Name of the offending field, when there is exactly one.
            value: The offending value, as supplied by the caller.
            context: Additional machine context; ``field`` / ``value`` are
                merged in without overwriting explicitly supplied entries.
        """
        payload: dict[str, Any] = dict(context or {})
        if field is not None:
            payload.setdefault("field", field)
        if value is not None:
            payload.setdefault("value", value)
        super().__init__(message, context=payload)
        self.field = field


class ParameterOutOfRangeError(ValidationError):
    """A numeric parameter is outside its accepted range."""


class UnsupportedStressTypeError(ValidationError):
    """An operator key, UI label or enum value is not supported."""


class DatasetError(FrameworkError):
    """A dataset could not be loaded, split or summarised."""


class DatasetLoadError(DatasetError):
    """A dataset file could not be read into a frame."""


class ArtifactError(FrameworkError):
    """A model artifact could not be saved, loaded or attested."""


class ArtifactPathError(ArtifactError):
    """A filename failed sanitisation or path containment."""


class ArtifactNotFoundError(ArtifactError):
    """An artifact file or manifest entry does not exist."""


class ArtifactIntegrityError(ArtifactError):
    """An artifact failed its digest, size cap or estimator type gate."""


class ArtifactUntrustedError(ArtifactError):
    """An artifact has no attestation and consent was not granted."""


class UnknownModelError(FrameworkError):
    """An estimator name or key is not one this framework builds."""


class ModelNotTrainedError(FrameworkError):
    """An operation needs a trained estimator that has not been fitted yet."""


class ReportExportError(FrameworkError):
    """A report could not be serialised to its export format."""
