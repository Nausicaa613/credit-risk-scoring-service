"""HTTP transport built on :mod:`http.server`.

A framework would be less code, but a hand-written transport makes the request
lifecycle explicit -- which is exactly the part an interviewer or a reviewer
asks about. The layer does five things and nothing else:

1. parse the request line, headers and body (with a hard body-size cap),
2. assign a request id,
3. delegate to :class:`~riskscore.api.ScoringService`,
4. write the JSON response with a small set of standard headers,
5. log the exchange and record the response time.

Run it with ``python -m riskscore.server`` or the ``make run`` target.
"""

from __future__ import annotations

import json
import logging
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from .api import RequestContext, Response, ScoringService, new_request_id
from .config import Settings
from .errors import PayloadTooLargeError, RiskScoreError, ValidationError
from .storage import utc_now_iso

logger = logging.getLogger("riskscore.server")

SERVER_NAME = "credit-risk-scoring-service"
READ_TIMEOUT_SECONDS = 30.0

#: How much of a rejected request body is read and discarded before the
#: connection is closed, and how long to wait for it. Both bounds matter: the
#: declared length comes from the client, so draining must not let a client make
#: the server read an unbounded amount or pin a worker thread indefinitely.
DRAIN_LIMIT_BYTES = 1 << 20
DRAIN_TIMEOUT_SECONDS = 0.5

#: Response headers applied to every reply. ``X-Content-Type-Options`` and
#: ``Cache-Control`` are the two that actually matter for a JSON API served
#: without a reverse proxy in front of it.
BASE_HEADERS: Dict[str, str] = {
    "Content-Type": "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "X-Service": SERVER_NAME,
}


def _error_response(
    status: int,
    code: str,
    message: str,
    *,
    request_id: str = "",
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    payload: Dict[str, Any] = {"error": {"code": code, "message": message, "details": {}}}
    if request_id:
        payload["request_id"] = request_id
    return Response(status=status, payload=payload, headers=dict(headers or {}))


class RiskScoreRequestHandler(BaseHTTPRequestHandler):
    """One handler instance per request; ``service`` comes from the server."""

    server_version = f"{SERVER_NAME}/0.1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    @property
    def service(self) -> ScoringService:
        return self.server.service  # type: ignore[attr-defined]

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        """Route the stdlib's stderr chatter through ``logging``."""
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def log_error(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        logger.warning("%s - %s", self.address_string(), fmt % args)

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("PUT")

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("OPTIONS")

    # -- core --------------------------------------------------------------

    def _discard_rejected_body(self, length: int) -> None:
        """Read and throw away a request body that is about to be rejected.

        Closing a socket that still holds unread data makes the OS send RST
        instead of FIN, and an RST can make the peer discard the response that
        was just written. So an honest client that sends a slightly oversized
        body could be told the connection was reset instead of receiving the 413
        it was owed. Draining first turns the close back into a clean FIN.

        This is bounded in both bytes and time on purpose: ``length`` is chosen
        by the client, so an unbounded drain would be a denial-of-service vector
        and an unbounded wait would pin a worker thread.
        """
        if length <= 0:
            return
        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(DRAIN_TIMEOUT_SECONDS)
            remaining = min(length, DRAIN_LIMIT_BYTES)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            # A client that stalls or vanishes is exactly the case we were
            # already handling by closing, so give up quietly and let the close
            # do its job.
            pass
        finally:
            try:
                self.connection.settimeout(previous_timeout)
            except OSError:
                pass

    def _read_body(self) -> Tuple[bytes, Optional[Response]]:
        """Read the body, enforcing the size cap.

        Returns ``(body, early_response)``; when ``early_response`` is not
        ``None`` the caller must send it and stop.

        Every early return refuses the request *before* its body has been read,
        so each one also tells the client the connection is finished. Without
        that, the unread body is still in the socket: the connection is either
        reused with the body parsed as the next request line, or closed abruptly
        with an RST that can destroy this response.
        """
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                return b"", _error_response(
                    411,
                    "length_required",
                    "chunked requests are not supported; send Content-Length",
                    headers={"Connection": "close"},
                )
            return b"", None

        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            return b"", _error_response(
                400,
                "validation_error",
                "invalid Content-Length header",
                headers={"Connection": "close"},
            )
        if length < 0:
            return b"", _error_response(
                400,
                "validation_error",
                "Content-Length must not be negative",
                headers={"Connection": "close"},
            )
        if length > self.settings.max_body_bytes:
            self._discard_rejected_body(length)
            return b"", _error_response(
                413,
                "payload_too_large",
                f"request body exceeds {self.settings.max_body_bytes} bytes",
                headers={"Connection": "close"},
            )
        if length == 0:
            return b"", None
        return self.rfile.read(length), None

    def _dispatch(self, method: str) -> None:
        started = time.perf_counter()
        request_id = new_request_id()
        split = urlsplit(self.path)
        path = split.path or "/"
        query = parse_qs(split.query, keep_blank_values=True)

        body, early = self._read_body()
        if early is not None:
            self._write(early, method=method, request_id=request_id, started=started)
            return

        context = RequestContext(
            method=method,
            path=path,
            query=query,
            headers={key.lower(): value for key, value in self.headers.items()},
            body=body,
            request_id=request_id,
        )

        try:
            response = self.service.handle(context)
        except RiskScoreError as error:  # defensive: handle() already catches these
            response = Response.error(error, request_id=request_id)
        except Exception:  # noqa: BLE001 - never leak a traceback to the client
            logger.exception("unhandled transport error on %s %s", method, path)
            response = _error_response(
                500, "internal_error", "internal server error", request_id=request_id
            )

        self._write(response, method=method, request_id=request_id, started=started)

    def _write(
        self,
        response: Response,
        *,
        method: str,
        request_id: str,
        started: float,
    ) -> None:
        body = response.body_bytes()
        try:
            self.send_response(response.status)
            for name, value in BASE_HEADERS.items():
                self.send_header(name, value)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-Id", request_id)
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("client disconnected before the response was written (%s)", request_id)
            return

        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s %s -> %d in %.2f ms request_id=%s",
            method,
            self.path,
            response.status,
            duration_ms,
            request_id,
        )


class RiskScoreHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that carries the service and settings."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], service: ScoringService, settings: Settings) -> None:
        self.service = service
        self.settings = settings
        super().__init__(address, RiskScoreRequestHandler)

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def build_service(settings: Optional[Settings] = None) -> ScoringService:
    """Create a service and try to load the configured model artifact."""
    resolved = settings or Settings.from_env()
    service = ScoringService(resolved)
    service.load_model_from_path()
    return service


def build_server(
    service: ScoringService, settings: Optional[Settings] = None
) -> RiskScoreHTTPServer:
    resolved = settings or service.settings
    return RiskScoreHTTPServer((resolved.host, resolved.port), service, resolved)


def serve_forever(server: RiskScoreHTTPServer) -> None:
    """Serve until SIGINT/SIGTERM, then shut down cleanly."""
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("received signal %s, shutting down", signum)
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), _handle_signal)
            except ValueError:  # not on the main thread
                pass

    host, port = server.server_address[:2]
    logger.info("listening on http://%s:%s", host, port)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        logger.info("server stopped")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
    )


def main(argv: Optional[list] = None) -> int:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    logger.info("starting %s with %s", SERVER_NAME, json.dumps(settings.as_dict(), sort_keys=True))
    logger.info("started_at=%s", utc_now_iso())

    service = build_service(settings)
    if service.scorer is None:
        logger.warning(
            "the service will start but /v1/score will return 503 until a model is available"
        )

    server = build_server(service, settings)
    serve_forever(server)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
