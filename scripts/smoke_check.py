#!/usr/bin/env python3
"""End-to-end smoke check: generate data, train, score, persist, audit.

This is the script referenced from the README as "verify a fresh clone". It
exercises the whole pipeline in a temporary directory so it never touches the
checked-in artifacts, and it fails loudly if any stage regresses.

Usage
-----
    python scripts/smoke_check.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import replace
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from generate_dataset import generate_dataset, write_jsonl  # noqa: E402

from riskscore.api import RequestContext, ScoringService  # noqa: E402
from riskscore.config import Settings  # noqa: E402
from riskscore.features import FEATURE_NAMES, build_features, normalise_payload  # noqa: E402
from riskscore.model import TrainingConfig, train_scorecard  # noqa: E402
from riskscore.server import build_server  # noqa: E402
from riskscore.webapp import SAMPLE_APPLICATION  # noqa: E402

GOOD_APPLICATION: Dict[str, Any] = {
    "age": 42,
    "annual_income": 420000,
    "employment_years": 12.0,
    "debt_to_income_ratio": 0.14,
    "num_delinquencies_24m": 0,
    "credit_history_months": 216,
    "loan_amount": 180000,
    "loan_term_months": 36,
    "num_open_accounts": 4,
    "revolving_utilization": 0.12,
    "purpose": "equipment",
}

RISKY_APPLICATION: Dict[str, Any] = {
    "age": 23,
    "annual_income": 62000,
    "employment_years": 0.4,
    "debt_to_income_ratio": 1.05,
    "num_delinquencies_24m": 4,
    "credit_history_months": 9,
    "loan_amount": 260000,
    "loan_term_months": 84,
    "num_open_accounts": 13,
    "revolving_utilization": 1.32,
    "purpose": "refinance",
}


def check(condition: bool, message: str) -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {message}")
    if not condition:
        raise SystemExit(f"smoke check failed: {message}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="riskscore-smoke-") as workdir:
        data_path = os.path.join(workdir, "applications.jsonl")
        model_path = os.path.join(workdir, "model.json")
        db_path = os.path.join(workdir, "riskscore.db")

        print("[1/7] generating synthetic dataset")
        dataset = generate_dataset(1500, seed=20260101)
        write_jsonl(dataset, data_path)
        default_rate = sum(row["defaulted"] for row in dataset) / len(dataset)
        check(os.path.exists(data_path), f"dataset written to {data_path}")
        check(0.10 <= default_rate <= 0.28, f"default rate is plausible ({default_rate:.4f})")

        print("[2/7] building the feature matrix")
        rows = []
        labels = []
        for row in dataset:
            payload = {k: v for k, v in row.items() if not k.startswith("_") and k != "defaulted"}
            rows.append(build_features(normalise_payload(payload)).values)
            labels.append(int(row["defaulted"]))
        check(len(rows[0]) == len(FEATURE_NAMES), f"feature width is {len(FEATURE_NAMES)}")
        check(len(set(labels)) == 2, "both classes are present")

        print("[3/7] training the scorecard")
        started = time.perf_counter()
        model, report = train_scorecard(
            rows[:1200],
            labels[:1200],
            feature_names=FEATURE_NAMES,
            config=TrainingConfig(epochs=300, threshold=0.35),
            trained_at="smoke",
            validation=(rows[1200:], labels[1200:]),
        )
        elapsed = time.perf_counter() - started
        auc = report["test"]["auc"]
        check(auc > 0.65, f"holdout AUC {auc:.4f} clears the 0.65 floor")
        check(elapsed < 120.0, f"training finished in {elapsed:.1f}s")
        model.save(model_path)
        check(os.path.exists(model_path), "model artifact written")

        print("[4/7] scoring two contrasting applicants")
        settings = Settings(db_path=db_path, model_path=model_path, decision_threshold=0.35)
        service = ScoringService(settings, model=model)

        good = service.handle(
            RequestContext(
                method="POST",
                path="/v1/score",
                body=json.dumps(GOOD_APPLICATION).encode(),
                request_id="req_smoke_good",
            )
        )
        risky = service.handle(
            RequestContext(
                method="POST",
                path="/v1/score",
                body=json.dumps(RISKY_APPLICATION).encode(),
                request_id="req_smoke_risky",
            )
        )
        check(good.status == 201, f"good applicant scored (HTTP {good.status})")
        check(risky.status == 201, f"risky applicant scored (HTTP {risky.status})")

        good_result = good.payload["result"]
        risky_result = risky.payload["result"]
        print(
            f"        strong profile -> score {good_result['credit_score']}, "
            f"PD {good_result['probability_of_default']:.4f}, band {good_result['risk_band']}, "
            f"{good_result['decision']}"
        )
        print(
            f"        weak profile   -> score {risky_result['credit_score']}, "
            f"PD {risky_result['probability_of_default']:.4f}, band {risky_result['risk_band']}, "
            f"{risky_result['decision']}"
        )
        check(
            good_result["credit_score"] > risky_result["credit_score"],
            "the stronger profile scores higher",
        )
        check(
            good_result["probability_of_default"] < risky_result["probability_of_default"],
            "the stronger profile has lower probability of default",
        )
        check(risky_result["decision"] == "decline", "the weak profile is declined")

        print("[5/7] checking explainability")
        contributions = risky_result["contributions"]
        log_odds = sum(item["contribution"] for item in contributions) + risky_result["intercept"]
        # The JSON response rounds contributions to 6 decimals, so 18 features can
        # drift by a few 1e-6. The exact-additivity guarantee is asserted against
        # the unrounded values in tests/test_scorecard.py.
        check(
            abs(log_odds - risky_result["log_odds"]) < 1e-4,
            "contributions plus intercept reconstruct the reported log-odds",
        )
        check(len(risky_result["reason_codes"]) == 3, "three reason codes are returned")
        check(
            any(code["direction"] == "increases_risk" for code in risky_result["reason_codes"]),
            "at least one reason code explains the decline",
        )

        print("[6/7] checking persistence, audit trail and read endpoints")
        application_id = risky_result["application_id"]
        fetched = service.handle(
            RequestContext(
                method="GET",
                path=f"/v1/applications/{application_id}",
                request_id="req_smoke_get",
            )
        )
        check(fetched.status == 200, "the stored application can be read back")
        check(
            fetched.payload["application"]["credit_score"] == risky_result["credit_score"],
            "the stored score matches the response",
        )

        audit = service.handle(
            RequestContext(method="GET", path="/v1/audit", request_id="req_smoke_audit")
        )
        check(audit.status == 200, "the audit trail is readable")
        check(audit.payload["count"] >= 2, f"audit trail holds {audit.payload['count']} events")

        health = service.handle(
            RequestContext(method="GET", path="/healthz", request_id="req_smoke_health")
        )
        check(health.status == 200, "health check reports ok")

        metrics = service.handle(
            RequestContext(method="GET", path="/metrics", request_id="req_smoke_metrics")
        )
        check(metrics.status == 200, "metrics are exposed")
        check(
            metrics.payload["storage"]["applications_total"] == 2,
            "both applications were persisted",
        )

        not_found = service.handle(
            RequestContext(
                method="GET", path="/v1/applications/app_doesnotexist", request_id="req_smoke_404"
            )
        )
        check(not_found.status == 404, "an unknown application returns 404")

        invalid = service.handle(
            RequestContext(
                method="POST",
                path="/v1/score",
                body=json.dumps({"age": 200, "annual_income": -5}).encode(),
                request_id="req_smoke_400",
            )
        )
        check(invalid.status == 400, "an invalid payload returns 400")
        check(
            "problems" in invalid.payload["error"]["details"],
            "the validation error lists every problem it found",
        )

        print("[7/7] checking the demo UI over a real socket")
        # Port 0 lets the OS pick a free port, so the check never races another
        # process for a probed one; the bound port is read back from the socket.
        server = build_server(service, replace(settings, port=0))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.daemon = True
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/app", timeout=10) as page:
                body = page.read().decode("utf-8")
                check(page.status == 200, "GET /app returns 200")
                check(
                    page.headers.get_content_type() == "text/html",
                    "the demo page is served as HTML",
                )
            check("<!doctype html>" in body.lower(), "the demo page is a full HTML document")
            check("/v1/score" in body, "the page drives the public scoring endpoint")
            check(
                all(f'name="{field}"' in body for field in ("age", "purpose")),
                "the page renders the application form",
            )
            external = re.findall(r'(?:src|href)="(https?://[^"]*)"', body)
            check(external == [], "the page loads no external resources")

            # The prefilled sample is the first thing a reviewer will submit, so
            # the page's own defaults have to survive the real API.
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/score",
                data=json.dumps(SAMPLE_APPLICATION).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as scored:
                scored_body = json.loads(scored.read().decode("utf-8"))
                check(scored.status == 201, "the prefilled sample scores over HTTP")
            check(
                300 <= scored_body["result"]["credit_score"] <= 850,
                "the sample returns a score inside the published range",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    print()
    print("smoke check passed: generate -> train -> score -> persist -> audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
