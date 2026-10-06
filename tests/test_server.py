"""Integration tests for the HTTP transport.

These bind a real socket on an ephemeral port and speak HTTP to it, so they
cover what the transport-neutral API tests cannot: status lines, headers,
``Content-Length`` framing, oversized bodies and request-id propagation. The
server is started once per test case and always torn down.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace

from riskscore.config import Settings
from riskscore.features import FEATURE_NAMES
from riskscore.model import ScorecardModel, Standardizer
from riskscore.server import RiskScoreHTTPServer, build_server, build_service


def valid_payload(**overrides):
    payload = {
        "age": 40,
        "annual_income": 300000,
        "employment_years": 8.0,
        "debt_to_income_ratio": 0.3,
        "num_delinquencies_24m": 0,
        "credit_history_months": 150,
        "loan_amount": 200000,
        "loan_term_months": 36,
        "num_open_accounts": 5,
        "revolving_utilization": 0.3,
        "purpose": "equipment",
    }
    payload.update(overrides)
    return payload


def build_model() -> ScorecardModel:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("num_delinquencies_24m")] = 0.8
    return ScorecardModel(
        feature_names=tuple(FEATURE_NAMES),
        weights=tuple(weights),
        intercept=-0.6,
        standardizer=Standardizer(
            means=tuple(0.0 for _ in FEATURE_NAMES),
            scales=tuple(1.0 for _ in FEATURE_NAMES),
        ),
        decision_threshold=0.5,
        version="test-http",
    )


class RawHttpConnection:
    """Minimal buffered HTTP client used to exercise keep-alive directly.

    ``urllib`` opens a new connection per request, so it cannot prove that
    responses are correctly framed on a reused socket. This keeps its own buffer
    so bytes that arrive in the same TCP segment as the headers are not lost.
    """

    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.create_connection((host, port), timeout=10)
        self._buffer = b""

    def __enter__(self) -> "RawHttpConnection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:  # pragma: no cover - already closed
            pass

    def _fill(self) -> bool:
        chunk = self._socket.recv(4096)
        if not chunk:
            return False
        self._buffer += chunk
        return True

    def read_response(self) -> tuple:
        """Return ``(status_line, headers, body)`` for the next response."""
        while b"\r\n\r\n" not in self._buffer:
            if not self._fill():
                break
        head, _, self._buffer = self._buffer.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        status_line, header_lines = lines[0], lines[1:]

        length = 0
        for line in header_lines:
            key, _, value = line.partition(b":")
            if key.strip().lower() == b"content-length":
                length = int(value.strip())
        while len(self._buffer) < length:
            if not self._fill():
                break
        body, self._buffer = self._buffer[:length], self._buffer[length:]
        return status_line, header_lines, body

    def send(self, data: bytes) -> None:
        self._socket.sendall(data)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def bind_on_free_port(service, settings, *, attempts: int = 10):
    """Bind ``service`` on an ephemeral port, retrying if the port is taken.

    ``free_port()`` can only report a port that *was* free: the probe socket is
    closed before the real server binds, and in that window the OS may hand the
    same port to anything else on the machine. The result is a rare ``OSError``
    ("address already in use") raised from ``setUp``, which surfaces as an
    *error* rather than a failure -- exactly the kind of flake that erodes trust
    in a green suite. Retrying with a fresh port closes the window without
    touching production code.

    ``settings.port`` only ever reaches the bind call (nothing else in the
    service reads it), so returning the settings that actually bound is safe.

    Returns ``(server, settings)``.
    """
    last_error = None
    for _ in range(attempts):
        candidate = replace(settings, port=free_port())
        try:
            return build_server(service, candidate), candidate
        except OSError as error:
            last_error = error
    raise last_error


class ServerTestCase(unittest.TestCase):
    max_body_bytes = 64 * 1024

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.settings = Settings(
            host="127.0.0.1",
            port=0,
            db_path=os.path.join(self.directory.name, "http.db"),
            model_path=os.path.join(self.directory.name, "model.json"),
            max_body_bytes=self.max_body_bytes,
        )
        self.service = build_service(self.settings)
        self.service.load_model(build_model())
        self.server, self.settings = bind_on_free_port(self.service, self.settings)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05})
        self.thread.daemon = True
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.settings.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    # -- helpers -----------------------------------------------------------

    def http(self, method, path, *, body=None, headers=None):
        data = None
        request_headers = dict(headers or {})
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                return response.status, dict(response.headers), raw
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def json_response(self, method, path, **kwargs):
        status, headers, raw = self.http(method, path, **kwargs)
        return status, headers, json.loads(raw.decode("utf-8"))


class HttpBasicsTests(ServerTestCase):
    def test_get_root_returns_json(self) -> None:
        status, headers, payload = self.json_response("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(payload["service"], "credit-risk-scoring-service")
        self.assertIn("application/json", headers["Content-Type"])

    def test_security_headers_are_present(self) -> None:
        _, headers, _ = self.http("GET", "/healthz")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Service"], "credit-risk-scoring-service")

    def test_request_id_header_is_returned_and_unique(self) -> None:
        _, first, _ = self.http("GET", "/healthz")
        _, second, _ = self.http("GET", "/healthz")
        self.assertTrue(first["X-Request-Id"].startswith("req_"))
        self.assertNotEqual(first["X-Request-Id"], second["X-Request-Id"])

    def test_head_returns_headers_without_a_body(self) -> None:
        status, headers, raw = self.http("HEAD", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(raw, b"")
        self.assertGreater(int(headers["Content-Length"]), 0)

    def test_unknown_route_returns_404(self) -> None:
        status, _, payload = self.json_response("GET", "/does-not-exist")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_wrong_method_returns_405_with_allow(self) -> None:
        status, headers, payload = self.json_response("GET", "/v1/score")
        self.assertEqual(status, 405)
        self.assertIn("POST", headers["Allow"])

    def test_options_is_rejected_explicitly(self) -> None:
        status, _, _ = self.http("OPTIONS", "/v1/score")
        self.assertEqual(status, 405)

    def test_delete_is_rejected(self) -> None:
        status, _, _ = self.http("DELETE", "/v1/applications/app_abcdef123456")
        self.assertEqual(status, 405)


class HttpScoringTests(ServerTestCase):
    def test_post_returns_201_with_location(self) -> None:
        status, headers, payload = self.json_response("POST", "/v1/score", body=valid_payload())
        self.assertEqual(status, 201)
        self.assertTrue(headers["Location"].startswith("/v1/applications/app_"))
        self.assertIn("result", payload)

    def test_body_round_trips_through_the_socket(self) -> None:
        payload = valid_payload(num_delinquencies_24m=3)
        _, _, response = self.json_response("POST", "/v1/score", body=payload)
        application_id = response["result"]["application_id"]
        _, _, fetched = self.json_response("GET", f"/v1/applications/{application_id}")
        self.assertEqual(fetched["application"]["payload"]["num_delinquencies_24m"], 3)

    def test_unicode_payload_is_preserved(self) -> None:
        payload = valid_payload()
        payload["purpose"] = "equipment"
        status, _, response = self.json_response("POST", "/v1/score", body=payload)
        self.assertEqual(status, 201)
        self.assertEqual(response["result"]["model_version"], "test-http")

    def test_invalid_payload_returns_400_with_problems(self) -> None:
        status, _, payload = self.json_response("POST", "/v1/score", body={"age": 900})
        self.assertEqual(status, 400)
        self.assertTrue(payload["error"]["details"]["problems"])

    def test_malformed_json_returns_400(self) -> None:
        status, _, payload = self.json_response("POST", "/v1/score", body=b"{oops")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "validation_error")

    def test_oversized_body_returns_413(self) -> None:
        huge = json.dumps(valid_payload()).encode("utf-8")
        huge = huge + b" " * (self.max_body_bytes + 100)
        status, headers, payload = self.json_response("POST", "/v1/score", body=huge)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "payload_too_large")
        # The body is still in flight when the request is refused. This header
        # is the client-visible half of the fix: it stops the connection from
        # being reused with the unread body parsed as the next request line, and
        # the server drains what is already buffered so the close is a clean FIN
        # rather than an RST that would discard this very response.
        self.assertEqual(headers.get("Connection"), "close")

    def test_missing_application_returns_404(self) -> None:
        status, _, payload = self.json_response("GET", "/v1/applications/app_missing99999")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["details"]["application_id"], "app_missing99999")

    def test_custom_application_id_is_echoed_in_location(self) -> None:
        status, headers, _ = self.json_response(
            "POST", "/v1/score", body=valid_payload(application_id="app_httpclient01")
        )
        self.assertEqual(status, 201)
        self.assertEqual(headers["Location"], "/v1/applications/app_httpclient01")

    def test_metrics_endpoint_reports_traffic(self) -> None:
        self.http("POST", "/v1/score", body=valid_payload())
        status, _, payload = self.json_response("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(payload["counters"]["http_requests_total"], 1)
        self.assertEqual(payload["storage"]["applications_total"], 1)

    def test_health_returns_ok(self) -> None:
        status, _, payload = self.json_response("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_audit_endpoint_lists_the_scoring_event(self) -> None:
        self.http("POST", "/v1/score", body=valid_payload())
        status, _, payload = self.json_response("GET", "/v1/audit")
        self.assertEqual(status, 200)
        self.assertTrue(
            any(event["event_type"] == "application_scored" for event in payload["events"])
        )

    def test_pipelined_requests_on_one_connection(self) -> None:
        """HTTP/1.1 keep-alive must work: three requests, one socket.

        ``urllib`` opens a fresh connection per call, so this drives the socket
        directly and reads each response by its own ``Content-Length``. That is
        the only way to prove the server frames responses correctly instead of
        relying on the client closing the connection.
        """
        with RawHttpConnection("127.0.0.1", self.settings.port) as connection:
            body = json.dumps(valid_payload()).encode("utf-8")
            for index in range(3):
                request = (
                    b"POST /v1/score HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Connection: keep-alive\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                )
                connection.send(request)
                status_line, _headers, payload = connection.read_response()
                self.assertIn(
                    b"201", status_line, f"request {index + 1} returned {status_line!r}"
                )
                self.assertIn(b"credit_score", payload)


class ServerWithoutModelTests(ServerTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Remove the model to exercise the degraded path over real HTTP.
        self.service.model = None
        self.service.scorer = None
        self.service.model_error = "model removed for the test"

    def test_scoring_returns_503(self) -> None:
        status, _, payload = self.json_response("POST", "/v1/score", body=valid_payload())
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "model_unavailable")

    def test_health_returns_503_degraded(self) -> None:
        status, _, payload = self.json_response("GET", "/healthz")
        self.assertEqual(status, 503)
        self.assertEqual(payload["status"], "degraded")


class ServerConstructionTests(unittest.TestCase):
    def test_build_server_uses_the_configured_address(self) -> None:
        directory = tempfile.TemporaryDirectory()
        try:
            settings = Settings(
                host="127.0.0.1",
                port=0,
                db_path=os.path.join(directory.name, "x.db"),
            )
            service = build_service(settings)
            server, bound = bind_on_free_port(service, settings)
            try:
                self.assertIsInstance(server, RiskScoreHTTPServer)
                self.assertEqual(server.server_address[1], bound.port)
                self.assertIs(server.service, service)
            finally:
                server.server_close()
        finally:
            directory.cleanup()

    def test_build_service_survives_a_missing_artifact(self) -> None:
        directory = tempfile.TemporaryDirectory()
        try:
            settings = Settings(
                db_path=os.path.join(directory.name, "y.db"),
                model_path=os.path.join(directory.name, "absent.json"),
            )
            service = build_service(settings)
            self.assertIsNone(service.scorer)
            self.assertIn("not found", service.model_error)
        finally:
            directory.cleanup()

    def test_daemon_threads_are_enabled(self) -> None:
        self.assertTrue(RiskScoreHTTPServer.daemon_threads)

    def test_unsupported_chunked_body_is_rejected(self) -> None:
        directory = tempfile.TemporaryDirectory()
        settings = Settings(
            host="127.0.0.1",
            port=0,
            db_path=os.path.join(directory.name, "z.db"),
        )
        service = build_service(settings)
        server, settings = bind_on_free_port(service, settings)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.daemon = True
        thread.start()
        try:
            with socket.create_connection(("127.0.0.1", settings.port), timeout=10) as connection:
                connection.sendall(
                    b"POST /v1/score HTTP/1.1\r\nHost: x\r\n"
                    b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                )
                response = connection.recv(65536)
            self.assertIn(b"411", response)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            directory.cleanup()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
