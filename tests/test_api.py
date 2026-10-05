"""Tests for the transport-neutral API layer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from riskscore.api import RequestContext, ScoringService
from riskscore.config import Settings
from riskscore.features import FEATURE_NAMES
from riskscore.model import ScorecardModel, Standardizer
from riskscore.scorecard import band_for_score


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


def build_model(**overrides) -> ScorecardModel:
    """A deterministic model with a non-zero weight on one interpretable feature."""
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("num_delinquencies_24m")] = 0.8
    weights[FEATURE_NAMES.index("revolving_utilization")] = 0.5
    weights[FEATURE_NAMES.index("employment_years")] = -0.4
    kwargs = {
        "feature_names": CLASSIFY_FEATURE_NAMES,
        "weights": tuple(weights),
        "intercept": -0.5,
        "standardizer": Standardizer(
            means=tuple(0.0 for _ in FEATURE_NAMES),
            scales=tuple(1.0 for _ in FEATURE_NAMES),
        ),
        "decision_threshold": 0.5,
        "version": "test-0.1.0",
    }
    kwargs.update(overrides)
    return ScorecardModel(**kwargs)


CLASSIFY_FEATURE_NAMES = tuple(FEATURE_NAMES)


def request(method, path, *, body=None, query=None, request_id="req_test"):
    raw = b"" if body is None else (
        body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    )
    return RequestContext(
        method=method,
        path=path,
        query=query or {},
        headers={"content-type": "application/json"},
        body=raw,
        request_id=request_id,
    )


class ServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.settings = Settings(
            db_path=os.path.join(self.directory.name, "api.db"),
            model_path=os.path.join(self.directory.name, "absent-model.json"),
            decision_threshold=0.5,
        )
        self.model = build_model()
        self.service = ScoringService(self.settings, model=self.model)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def score(self, payload=None, **kwargs):
        return self.service.handle(request("POST", "/v1/score", body=payload or valid_payload(), **kwargs))


class RoutingTests(ServiceTestCase):
    def test_root_returns_the_service_index(self) -> None:
        response = self.service.handle(request("GET", "/"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["service"], "credit-risk-scoring-service")
        self.assertTrue(response.payload["endpoints"])

    def test_v1_root_returns_the_index(self) -> None:
        self.assertEqual(self.service.handle(request("GET", "/v1")).status, 200)

    def test_unknown_path_returns_404(self) -> None:
        response = self.service.handle(request("GET", "/nope"))
        self.assertEqual(response.status, 404)
        self.assertEqual(response.payload["error"]["code"], "not_found")

    def test_unknown_v1_path_returns_404(self) -> None:
        self.assertEqual(self.service.handle(request("GET", "/v1/unknown")).status, 404)

    def test_wrong_method_returns_405_with_allow_header(self) -> None:
        response = self.service.handle(request("GET", "/v1/score"))
        self.assertEqual(response.status, 405)
        self.assertEqual(response.payload["error"]["code"], "method_not_allowed")
        self.assertIn("POST", response.headers["Allow"])

    def test_index_rejects_write_methods(self) -> None:
        self.assertEqual(self.service.handle(request("DELETE", "/")).status, 405)

    def test_request_id_is_echoed_on_errors(self) -> None:
        response = self.service.handle(request("GET", "/nope", request_id="req_abc123"))
        self.assertEqual(response.payload["request_id"], "req_abc123")


class ScoreEndpointTests(ServiceTestCase):
    def test_valid_payload_returns_201_with_location(self) -> None:
        response = self.score()
        self.assertEqual(response.status, 201)
        self.assertTrue(response.headers["Location"].startswith("/v1/applications/app_"))
        result = response.payload["result"]
        self.assertEqual(result["application_id"], response.headers["Location"].rsplit("/", 1)[-1])

    def test_result_contains_every_documented_field(self) -> None:
        result = self.score().payload["result"]
        for field in (
            "application_id",
            "credit_score",
            "probability_of_default",
            "risk_band",
            "risk_band_label",
            "decision",
            "decision_threshold",
            "model_version",
            "expected_default_rate",
            "log_odds",
            "intercept",
            "contributions",
            "reason_codes",
        ):
            self.assertIn(field, result)

    def test_band_matches_the_score(self) -> None:
        result = self.score().payload["result"]
        self.assertEqual(result["risk_band"], band_for_score(float(result["credit_score"])).name)

    def test_explainability_is_additive(self) -> None:
        result = self.score().payload["result"]
        total = result["intercept"] + sum(c["contribution"] for c in result["contributions"])
        self.assertAlmostEqual(total, result["log_odds"], places=4)

    def test_risky_payload_declines(self) -> None:
        response = self.score(
            valid_payload(num_delinquencies_24m=9, revolving_utilization=1.5, employment_years=0.0)
        )
        self.assertEqual(response.payload["result"]["decision"], "decline")

    def test_safe_payload_approves(self) -> None:
        response = self.score(
            valid_payload(num_delinquencies_24m=0, revolving_utilization=0.0, employment_years=20.0)
        )
        self.assertEqual(response.payload["result"]["decision"], "approve")

    def test_client_supplied_application_id_is_respected(self) -> None:
        response = self.score(valid_payload(application_id="app_client001"))
        self.assertEqual(response.payload["result"]["application_id"], "app_client001")

    def test_client_supplied_id_is_not_scored_as_a_feature(self) -> None:
        with_id = self.score(valid_payload(application_id="app_client002")).payload["result"]
        without_id = self.score().payload["result"]
        self.assertEqual(with_id["credit_score"], without_id["credit_score"])
        self.assertEqual(with_id["probability_of_default"], without_id["probability_of_default"])

    def test_malformed_application_id_is_rejected(self) -> None:
        response = self.score(valid_payload(application_id="has spaces"))
        self.assertEqual(response.status, 400)
        self.assertEqual(response.payload["error"]["details"]["field"], "application_id")

    def test_short_application_id_is_rejected(self) -> None:
        self.assertEqual(self.score(valid_payload(application_id="ab")).status, 400)

    def test_unknown_field_is_rejected(self) -> None:
        response = self.score(valid_payload(favourite_colour="blue"))
        self.assertEqual(response.status, 400)

    def test_all_validation_problems_are_reported(self) -> None:
        response = self.score({"age": 500, "annual_income": -10})
        self.assertEqual(response.status, 400)
        problems = response.payload["error"]["details"]["problems"]
        self.assertGreaterEqual(len(problems), 3)

    def test_missing_body_is_rejected(self) -> None:
        response = self.service.handle(
            RequestContext(method="POST", path="/v1/score", body=b"", request_id="req_1")
        )
        self.assertEqual(response.status, 400)
        self.assertIn("empty", response.payload["error"]["message"])

    def test_malformed_json_is_rejected_with_position(self) -> None:
        response = self.service.handle(
            RequestContext(method="POST", path="/v1/score", body=b"{not json", request_id="req_1")
        )
        self.assertEqual(response.status, 400)
        self.assertIn("line", response.payload["error"]["details"])

    def test_non_utf8_body_is_rejected(self) -> None:
        response = self.service.handle(
            RequestContext(method="POST", path="/v1/score", body=b"\xff\xfe\x00", request_id="req_1")
        )
        self.assertEqual(response.status, 400)

    def test_json_array_body_is_rejected(self) -> None:
        response = self.service.handle(
            RequestContext(
                method="POST", path="/v1/score", body=b"[1,2,3]", request_id="req_1"
            )
        )
        self.assertEqual(response.status, 400)

    def test_alias_fields_are_accepted(self) -> None:
        payload = valid_payload()
        payload["income"] = payload.pop("annual_income")
        payload["dti"] = payload.pop("debt_to_income_ratio")
        self.assertEqual(self.score(payload).status, 201)

    def test_score_is_deterministic(self) -> None:
        first = self.score().payload["result"]
        second = self.score().payload["result"]
        self.assertEqual(first["credit_score"], second["credit_score"])
        self.assertEqual(first["probability_of_default"], second["probability_of_default"])

    def test_reason_codes_are_limited_and_ranked(self) -> None:
        result = self.score(
            valid_payload(num_delinquencies_24m=6, revolving_utilization=1.4)
        ).payload["result"]
        self.assertEqual(len(result["reason_codes"]), 3)
        magnitudes = [abs(code["contribution"]) for code in result["reason_codes"]]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))


class MissingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        # Model artifacts live in their own directory so that a model written by
        # another test can never satisfy this one's "artifact absent" premise.
        self.model_directory = os.path.join(self.directory.name, "models")
        settings = Settings(
            db_path=os.path.join(self.directory.name, "empty.db"),
            model_path=os.path.join(self.model_directory, "never-trained.json"),
        )
        self.service = ScoringService(settings)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_scoring_without_a_model_returns_503(self) -> None:
        response = self.service.handle(
            request("POST", "/v1/score", body=valid_payload())
        )
        self.assertEqual(response.status, 503)
        self.assertEqual(response.payload["error"]["code"], "model_unavailable")

    def test_health_reports_degraded(self) -> None:
        response = self.service.handle(request("GET", "/healthz"))
        self.assertEqual(response.status, 503)
        self.assertEqual(response.payload["status"], "degraded")
        self.assertFalse(response.payload["model"]["loaded"])

    def test_load_model_from_path_reports_a_missing_artifact(self) -> None:
        self.assertFalse(self.service.load_model_from_path())
        self.assertIn("not found", self.service.model_error)

    def test_load_model_from_path_rejects_a_corrupt_artifact(self) -> None:
        path = os.path.join(self.directory.name, "broken.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ not a model }")
        service = ScoringService(Settings(db_path=self.service.settings.db_path, model_path=path))
        self.assertFalse(service.load_model_from_path())
        self.assertIn("invalid", service.model_error)

    def test_load_model_from_path_accepts_a_real_artifact(self) -> None:
        path = os.path.join(self.directory.name, "model.json")
        build_model().save(path)
        service = ScoringService(
            Settings(db_path=self.service.settings.db_path, model_path=path)
        )
        self.assertTrue(service.load_model_from_path())
        self.assertIsNotNone(service.scorer)


class ApplicationReadTests(ServiceTestCase):
    def test_get_returns_the_stored_record(self) -> None:
        created = self.score().payload["result"]
        response = self.service.handle(
            request("GET", f"/v1/applications/{created['application_id']}")
        )
        self.assertEqual(response.status, 200)
        application = response.payload["application"]
        self.assertEqual(application["credit_score"], created["credit_score"])
        self.assertEqual(application["response"]["credit_score"], created["credit_score"])
        self.assertIn("payload", application)

    def test_get_unknown_id_returns_404(self) -> None:
        response = self.service.handle(request("GET", "/v1/applications/app_missing001"))
        self.assertEqual(response.status, 404)
        self.assertEqual(response.payload["error"]["details"]["application_id"], "app_missing001")

    def test_list_returns_scored_applications(self) -> None:
        for _ in range(3):
            self.score()
        response = self.service.handle(request("GET", "/v1/applications"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["count"], 3)

    def test_list_honours_limit(self) -> None:
        for _ in range(4):
            self.score()
        response = self.service.handle(
            request("GET", "/v1/applications", query={"limit": ["2"]})
        )
        self.assertEqual(response.payload["count"], 2)

    def test_list_rejects_a_non_integer_limit(self) -> None:
        response = self.service.handle(
            request("GET", "/v1/applications", query={"limit": ["many"]})
        )
        self.assertEqual(response.status, 400)

    def test_list_rejects_an_out_of_range_limit(self) -> None:
        response = self.service.handle(
            request("GET", "/v1/applications", query={"limit": ["9999"]})
        )
        self.assertEqual(response.status, 400)

    def test_list_rejects_a_write_method(self) -> None:
        self.assertEqual(self.service.handle(request("POST", "/v1/applications")).status, 405)


class AuditEndpointTests(ServiceTestCase):
    def test_scoring_writes_an_audit_event(self) -> None:
        created = self.score().payload["result"]
        response = self.service.handle(request("GET", "/v1/audit"))
        self.assertEqual(response.status, 200)
        events = response.payload["events"]
        matching = [e for e in events if e["application_id"] == created["application_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["event_type"], "application_scored")
        self.assertIn("reason_codes", matching[0]["payload"])

    def test_audit_records_the_model_version_and_threshold(self) -> None:
        self.score()
        events = self.service.handle(request("GET", "/v1/audit")).payload["events"]
        scored = [e for e in events if e["event_type"] == "application_scored"][0]
        self.assertEqual(scored["payload"]["model_version"], "test-0.1.0")
        self.assertEqual(scored["payload"]["decision_threshold"], 0.5)

    def test_audit_can_be_filtered_by_application(self) -> None:
        first = self.score().payload["result"]["application_id"]
        self.score()
        response = self.service.handle(
            request("GET", "/v1/audit", query={"application_id": [first]})
        )
        self.assertEqual(response.payload["count"], 1)
        self.assertEqual(response.payload["events"][0]["application_id"], first)

    def test_audit_honours_limit(self) -> None:
        for _ in range(4):
            self.score()
        response = self.service.handle(request("GET", "/v1/audit", query={"limit": ["2"]}))
        self.assertEqual(response.payload["count"], 2)

    def test_rejections_can_be_audited_explicitly(self) -> None:
        from riskscore.errors import ValidationError

        self.service.record_rejection(
            request_id="req_rejected",
            path="/v1/score",
            error=ValidationError("bad payload", details={"problems": []}),
            raw_body=b"{}",
        )
        events = self.service.handle(request("GET", "/v1/audit")).payload["events"]
        self.assertTrue(any(event["event_type"] == "application_rejected" for event in events))


class AuxiliaryEndpointTests(ServiceTestCase):
    def test_bands_endpoint(self) -> None:
        response = self.service.handle(request("GET", "/v1/bands"))
        self.assertEqual(response.status, 200)
        bands = response.payload["bands"]
        self.assertEqual(len(bands), 5)
        self.assertEqual(bands[0]["band"], "A")

    def test_health_reports_ok_with_a_model(self) -> None:
        response = self.service.handle(request("GET", "/healthz"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["status"], "ok")
        self.assertTrue(response.payload["model"]["loaded"])
        self.assertTrue(response.payload["database"]["ok"])

    def test_metrics_include_storage_state(self) -> None:
        self.score()
        response = self.service.handle(request("GET", "/metrics"))
        self.assertEqual(response.status, 200)
        storage = response.payload["storage"]
        self.assertEqual(storage["applications_total"], 1)
        self.assertTrue(storage["by_risk_band"])
        self.assertTrue(storage["by_decision"])

    def test_metrics_can_be_disabled(self) -> None:
        service = ScoringService(
            Settings(
                db_path=self.settings.db_path,
                model_path=self.settings.model_path,
                metrics_enabled=False,
            ),
            model=self.model,
        )
        self.assertEqual(service.handle(request("GET", "/metrics")).status, 404)

    def test_metrics_count_requests_by_collapsed_route(self) -> None:
        self.score()
        self.score()
        self.service.handle(request("GET", "/metrics"))
        snapshot = self.service.metrics.snapshot()
        self.assertEqual(snapshot["http_path_counts"].get("/v1/score"), 2)
        # Identifiers are collapsed so cardinality stays bounded.
        created = self.score().payload["result"]["application_id"]
        self.service.handle(request("GET", f"/v1/applications/{created}"))
        self.assertEqual(
            self.service.metrics.snapshot()["http_path_counts"].get("/v1/applications/<id>"), 1
        )

    def test_metrics_counts_scores_and_bands(self) -> None:
        self.score()
        snapshot = self.service.metrics.snapshot()
        self.assertEqual(snapshot["counters"]["scores_total"], 1)

    def test_response_serialisation_is_utf8_json(self) -> None:
        body = self.score().body_bytes()
        self.assertIsInstance(body, bytes)
        self.assertEqual(json.loads(body.decode("utf-8"))["result"]["model_version"], "test-0.1.0")


class ModelFactory:
    """Rebuild a model through its JSON artifact, as a redeploy would."""

    @staticmethod
    def rebuild(model: ScorecardModel) -> ScorecardModel:
        return ScorecardModel.from_dict(model.to_dict())


class PersistenceIntegrationTests(ServiceTestCase):
    def test_decisions_survive_a_new_service_instance(self) -> None:
        created = self.score().payload["result"]
        rebuilt = ScoringService(self.settings, model=ModelFactory.rebuild(self.model))
        response = rebuilt.handle(request("GET", f"/v1/applications/{created['application_id']}"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["application"]["credit_score"], created["credit_score"])

    def test_two_services_share_the_database(self) -> None:
        self.score()
        other = ScoringService(self.settings, model=self.model)
        self.assertEqual(other.handle(request("GET", "/v1/applications")).payload["count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
