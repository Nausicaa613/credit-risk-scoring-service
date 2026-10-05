"""Application service: routing, validation and JSON serialisation.

This module is transport-agnostic. It receives a :class:`RequestContext` and
returns a :class:`Response`; it never touches a socket. That boundary is what
makes the API testable without binding a port, and it would let the same logic
be mounted behind a WSGI server later without a rewrite.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

from . import __version__
from .audit import EVENT_REJECTED, EVENT_SCORED, AuditLog
from .config import Settings
from .errors import (
    MethodNotAllowedError,
    ModelNotLoadedError,
    NotFoundError,
    RiskScoreError,
    ValidationError,
)
from .metrics import MetricsRegistry
from .model import ScorecardModel
from .scorecard import Scorer, band_table
from .storage import (
    ApplicationStore,
    new_application_id,
    new_request_id,
    record_decision,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

APPLICATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{5,63}$")
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 200


# --------------------------------------------------------------------------
# Transport-neutral request / response
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContext:
    """One inbound request, already decoded."""

    method: str
    path: str
    query: Mapping[str, List[str]] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    request_id: str = ""

    @property
    def segments(self) -> List[str]:
        return [unquote(segment) for segment in self.path.strip("/").split("/") if segment]

    def query_one(self, name: str, default: Optional[str] = None) -> Optional[str]:
        values = self.query.get(name)
        if not values:
            return default
        return values[0]

    def json_body(self) -> Any:
        """Decode the body as JSON, mapping syntax errors to a 400."""
        if not self.body:
            raise ValidationError("request body must not be empty")
        try:
            return json.loads(self.body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValidationError("request body must be valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(
                "request body is not valid JSON",
                details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
            ) from exc


@dataclass(frozen=True)
class Response:
    """One outbound response, ready to be written to a socket."""

    status: int
    payload: Any
    headers: Mapping[str, str] = field(default_factory=dict)

    def body_bytes(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=False).encode(
            "utf-8"
        )

    @classmethod
    def error(cls, error: RiskScoreError, *, request_id: str = "") -> "Response":
        envelope = error.to_envelope()
        if request_id:
            envelope["request_id"] = request_id
        headers: Dict[str, str] = {}
        if error.status == 405:
            headers["Allow"] = "GET, POST, HEAD"
        return cls(status=error.status, payload=envelope, headers=headers)


# --------------------------------------------------------------------------
# Query parameter helpers
# --------------------------------------------------------------------------


def _parse_int_param(raw: Optional[str], *, name: str, default: int, low: int, high: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"query parameter '{name}' must be an integer", details={"parameter": name}
        ) from exc
    if value < low or value > high:
        raise ValidationError(
            f"query parameter '{name}' must be within [{low}, {high}]",
            details={"parameter": name, "value": value},
        )
    return value


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class ScoringService:
    """Coordinates the model, storage, audit log and metrics."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: Optional[ScorecardModel] = None,
        store: Optional[ApplicationStore] = None,
        audit: Optional[AuditLog] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.store = store or ApplicationStore(settings.db_path)
        self.audit = audit or AuditLog(settings.db_path)
        self.metrics = metrics or MetricsRegistry()
        self.scorer: Optional[Scorer] = None
        self.model_error: Optional[str] = None

        self.store.init()
        if model is not None:
            self.load_model(model)

    # -- lifecycle ---------------------------------------------------------

    def load_model(self, model: ScorecardModel) -> None:
        """Install a model and rebuild the scorer."""
        self.model = model
        self.scorer = Scorer(model, threshold=self.settings.decision_threshold)
        self.model_error = None

    def load_model_from_path(self) -> bool:
        """Try to load the configured artifact. Returns success."""
        path = self.settings.model_path
        try:
            model = ScorecardModel.load(path)
        except FileNotFoundError:
            self.model_error = f"model artifact not found at {path}"
            logger.warning("%s; run 'make train' first", self.model_error)
            return False
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.model_error = f"model artifact at {path} is invalid: {exc}"
            logger.error(self.model_error)
            return False
        self.load_model(model)
        logger.info(
            "loaded model version %s (trained_at=%s) from %s",
            model.version,
            model.trained_at or "unknown",
            path,
        )
        return True

    def require_scorer(self) -> Scorer:
        if self.scorer is None:
            raise ModelNotLoadedError(
                "no model is loaded; run 'make train' and restart the service",
                details={"model_path": self.settings.model_path, "reason": self.model_error},
            )
        return self.scorer

    # -- routing -----------------------------------------------------------

    def handle(self, request: RequestContext) -> Response:
        """Route one request to its handler."""
        started = time.perf_counter()
        response: Response
        try:
            response = self._route(request)
        except RiskScoreError as error:
            response = Response.error(error, request_id=request.request_id)
        except Exception:  # noqa: BLE001 - last-resort guard, logged with a traceback
            logger.exception("unhandled error while serving %s %s", request.method, request.path)
            response = Response.error(
                RiskScoreError("internal server error"), request_id=request.request_id
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        route_label = _route_label(request.segments)
        self.metrics.observe_request(
            path=route_label,
            status=response.status,
            duration_ms=duration_ms,
            bytes_out=len(response.body_bytes()),
        )
        return response

    def _route(self, request: RequestContext) -> Response:
        method = request.method.upper()
        segments = request.segments

        if not segments:
            if method in ("GET", "HEAD"):
                return self._handle_index()
            raise MethodNotAllowedError(f"{method} is not allowed on /")

        head, *tail = segments

        if head == "healthz":
            return self._handle_health(method)
        if head == "metrics":
            return self._handle_metrics(method)

        if head == "v1":
            if not tail:
                return self._handle_index()
            section, *rest = tail
            if section == "score":
                if method != "POST":
                    raise MethodNotAllowedError(f"{method} is not allowed on /v1/score")
                return self._handle_score(request)
            if section == "bands":
                if method not in ("GET", "HEAD"):
                    raise MethodNotAllowedError(f"{method} is not allowed on /v1/bands")
                return Response(200, {"bands": band_table(), "model_version": self._model_version()})
            if section == "applications":
                if not rest:
                    if method not in ("GET", "HEAD"):
                        raise MethodNotAllowedError(
                            f"{method} is not allowed on /v1/applications"
                        )
                    return self._handle_list_applications(request)
                if len(rest) == 1:
                    if method not in ("GET", "HEAD"):
                        raise MethodNotAllowedError(
                            f"{method} is not allowed on /v1/applications/<id>"
                        )
                    return self._handle_get_application(request, rest[0])
            if section == "audit":
                if method not in ("GET", "HEAD"):
                    raise MethodNotAllowedError(f"{method} is not allowed on /v1/audit")
                return self._handle_audit(request)

        raise NotFoundError(
            f"no route for {method} {request.path}",
            details={"path": request.path},
        )

    # -- handlers ----------------------------------------------------------

    def _handle_index(self) -> Response:
        return Response(
            200,
            {
                "service": "credit-risk-scoring-service",
                "version": __version__,
                "model_version": self._model_version(),
                "endpoints": [
                    {"method": "POST", "path": "/v1/score", "description": "score one application"},
                    {"method": "GET", "path": "/v1/applications/<id>", "description": "fetch a stored decision"},
                    {"method": "GET", "path": "/v1/applications", "description": "list recent decisions"},
                    {"method": "GET", "path": "/v1/audit", "description": "read the audit trail"},
                    {"method": "GET", "path": "/v1/bands", "description": "risk band definitions"},
                    {"method": "GET", "path": "/healthz", "description": "liveness and readiness"},
                    {"method": "GET", "path": "/metrics", "description": "in-process metrics"},
                ],
            },
        )

    def _handle_health(self, method: str) -> Response:
        database_ok = True
        database_error: Optional[str] = None
        try:
            self.store.count()
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            database_ok = False
            database_error = str(exc)

        healthy = self.scorer is not None and database_ok
        payload: Dict[str, Any] = {
            "status": "ok" if healthy else "degraded",
            "service": "credit-risk-scoring-service",
            "version": __version__,
            "uptime_seconds": round(self.metrics.uptime_seconds, 3),
            "model": {
                "loaded": self.scorer is not None,
                "version": self._model_version(),
                "path": self.settings.model_path,
                "error": self.model_error,
            },
            "database": {
                "ok": database_ok,
                "path": self.settings.db_path,
                "applications": self.store.count() if database_ok else None,
                "error": database_error,
            },
            "checks": {
                "decision_threshold": self.settings.decision_threshold,
                "metrics_enabled": self.settings.metrics_enabled,
            },
        }
        if method == "HEAD":
            return Response(200 if healthy else 503, payload)
        return Response(200 if healthy else 503, payload)

    def _handle_metrics(self, method: str) -> Response:
        if not self.settings.metrics_enabled:
            raise NotFoundError("/metrics is disabled by configuration")
        snapshot = self.metrics.snapshot()
        try:
            snapshot["storage"] = {
                "applications_total": self.store.count(),
                "by_risk_band": self.store.score_distribution(),
                "by_decision": self.store.decision_counts(),
                "audit_events_total": self.audit.count(),
            }
        except Exception as exc:  # noqa: BLE001 - metrics must stay available
            snapshot["storage"] = {"error": str(exc)}
        return Response(200, snapshot)

    def _handle_score(self, request: RequestContext) -> Response:
        scorer = self.require_scorer()
        body = request.json_body()
        if not isinstance(body, Mapping):
            raise ValidationError("request body must be a JSON object")

        payload = body
        application_id = body.get("application_id")
        if application_id is not None:
            if not isinstance(application_id, str) or not APPLICATION_ID_PATTERN.match(
                application_id
            ):
                raise ValidationError(
                    "field 'application_id' must be 6-64 characters of "
                    "letters, digits, '_', '.', ':' or '-'",
                    details={"field": "application_id"},
                )
            payload = {key: value for key, value in body.items() if key != "application_id"}
        else:
            application_id = new_application_id()

        result = scorer.score(application_id, payload)
        serialised = result.as_dict()
        created_at = utc_now_iso()

        response_payload: Dict[str, Any] = {
            "request_id": request.request_id,
            "created_at": created_at,
            "result": serialised,
        }

        # The decision row and its audit row describe the same event, so they are
        # committed together: one fsync instead of two, and no window in which a
        # stored decision exists without a matching audit entry.
        audit_payload = {
            "credit_score": result.credit_score,
            "probability_of_default": round(result.probability_of_default, 6),
            "risk_band": result.band,
            "decision": result.decision,
            "model_version": result.model_version,
            "decision_threshold": result.decision_threshold,
            "reason_codes": serialised["reason_codes"],
        }
        try:
            record_decision(
                self.settings.db_path,
                application_id=application_id,
                credit_score=result.credit_score,
                probability_of_default=result.probability_of_default,
                risk_band=result.band,
                decision=result.decision,
                model_version=result.model_version,
                payload=dict(payload),
                response=serialised,
                audit_payload=audit_payload,
                audit_event_type=EVENT_SCORED,
                request_id=request.request_id,
                created_at=created_at,
            )
        except Exception:  # noqa: BLE001 - a failed write must not lose the score
            logger.exception(
                "failed to persist the decision for %s; the score was computed but not stored",
                application_id,
            )
            raise

        self.metrics.observe_score(
            credit_score=result.credit_score,
            probability_of_default=result.probability_of_default,
            band=result.band,
            decision=result.decision,
        )
        return Response(201, response_payload, {"Location": f"/v1/applications/{application_id}"})

    def _handle_get_application(self, request: RequestContext, application_id: str) -> Response:
        record = self.store.get(application_id)
        return Response(200, {"request_id": request.request_id, "application": record.as_dict()})

    def _handle_list_applications(self, request: RequestContext) -> Response:
        limit = _parse_int_param(
            request.query_one("limit"), name="limit", default=DEFAULT_LIST_LIMIT, low=1, high=MAX_LIST_LIMIT
        )
        offset = _parse_int_param(
            request.query_one("offset"), name="offset", default=0, low=0, high=1_000_000
        )
        records = self.store.list_recent(limit=limit, offset=offset)
        return Response(
            200,
            {
                "request_id": request.request_id,
                "count": len(records),
                "limit": limit,
                "offset": offset,
                "applications": [
                    {
                        "application_id": record.application_id,
                        "created_at": record.created_at,
                        "credit_score": record.credit_score,
                        "probability_of_default": record.probability_of_default,
                        "risk_band": record.risk_band,
                        "decision": record.decision,
                    }
                    for record in records
                ],
            },
        )

    def _handle_audit(self, request: RequestContext) -> Response:
        limit = _parse_int_param(
            request.query_one("limit"), name="limit", default=DEFAULT_LIST_LIMIT, low=1, high=MAX_LIST_LIMIT
        )
        application_id = request.query_one("application_id") or None
        events = self.audit.tail(limit=limit, application_id=application_id)
        return Response(
            200,
            {
                "request_id": request.request_id,
                "count": len(events),
                "limit": limit,
                "application_id": application_id,
                "events": events,
            },
        )

    # -- helpers -----------------------------------------------------------

    def _model_version(self) -> Optional[str]:
        return self.model.version if self.model is not None else None

    def record_rejection(
        self, *, request_id: str, path: str, error: RiskScoreError, raw_body: bytes
    ) -> None:
        """Audit a rejected request.

        Called by the transport layer, which is the only place that still has
        the raw body after :meth:`handle` has produced a response.
        """
        self.audit.record(
            EVENT_REJECTED,
            {
                "path": path,
                "status": error.status,
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "body_bytes": len(raw_body),
            },
            request_id=request_id,
        )


def _route_label(segments: Sequence[str]) -> str:
    """Collapse identifiers so metrics do not explode in cardinality."""
    if not segments:
        return "/"
    if segments[0] == "v1" and len(segments) >= 2:
        if segments[1] == "applications" and len(segments) >= 3:
            return "/v1/applications/<id>"
        return f"/v1/{segments[1]}"
    return f"/{segments[0]}"


__all__ = [
    "RequestContext",
    "Response",
    "ScoringService",
    "new_request_id",
    "utc_now_iso",
]
