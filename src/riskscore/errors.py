"""Error types shared by the API and service layers.

Every error carries the HTTP status code that the transport layer should use
plus a stable machine-readable ``code`` so that clients can branch on the
failure without parsing prose.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RiskScoreError(Exception):
    """Base class for all errors raised deliberately by this service."""

    status = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def to_envelope(self) -> Dict[str, Any]:
        """Return the JSON body sent to the client for this error."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationError(RiskScoreError):
    """The request body or query string is malformed."""

    status = 400
    code = "validation_error"


class UnprocessableError(RiskScoreError):
    """The payload is well-formed JSON but semantically impossible."""

    status = 422
    code = "unprocessable_entity"


class NotFoundError(RiskScoreError):
    """The requested resource does not exist."""

    status = 404
    code = "not_found"


class MethodNotAllowedError(RiskScoreError):
    """The path exists but not for this HTTP method."""

    status = 405
    code = "method_not_allowed"


class PayloadTooLargeError(RiskScoreError):
    """The request body exceeds the configured maximum size."""

    status = 413
    code = "payload_too_large"


class ModelNotLoadedError(RiskScoreError):
    """No model artifact is available to serve predictions."""

    status = 503
    code = "model_unavailable"


__all__ = [
    "RiskScoreError",
    "ValidationError",
    "UnprocessableError",
    "NotFoundError",
    "MethodNotAllowedError",
    "PayloadTooLargeError",
    "ModelNotLoadedError",
]
