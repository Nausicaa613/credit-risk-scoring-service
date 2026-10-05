# Design: credit-risk-scoring-service

Technical design for a dependency-free Python credit risk scoring microservice (Python 3.9+, standard library only). Covers architecture, module layout, data model, scoring math, error contract, observability, testing, and the tradeoffs behind the current design. Model behavior and limitations are documented in `MODEL_CARD.md`.

## 1. Overview and goals

The service is a single-process HTTP server that scores a credit application with a linear logistic scorecard and returns an explainable result.

1. **Zero dependencies.** Nothing beyond the standard library and `make` is needed to generate data, train, run, or test.
2. **Reproducibility.** A fixed seed drives dataset generation, the train/evaluation split, and gradient-descent initialization, so identical inputs yield identical artifacts and metrics.
3. **Explainability by construction.** Every score response carries a per-feature contribution vector and derived reason codes, not just a probability.
4. **Auditability.** Every scored application and every rejected request lands in an append-only audit table.
5. **Inspectable artifacts and honest scope.** The model is plain JSON that can be read, diffed, and versioned in git, and this is a portfolio/educational model — the engineering around it is the deliverable.

## 2. Non-goals

- **No real credit decisions.** The service must not approve, deny, price, or limit credit for any person.
- **No authentication or authorization in v0.1.** No API keys, sessions, OAuth, or tenant isolation.
- **No distributed deployment.** No orchestration, service mesh, shared cache, or message broker.
- **No Postgres in v0.1.** Persistence is one local SQLite file.
- No feature store, streaming ingestion, batch scoring UI, model registry, automatic retraining, or drift monitoring; no TLS termination in-process (a reverse proxy would own that).

## 3. Architecture

```text
  HTTP request (JSON)
        |
        v
  server.py      ThreadingHTTPServer + hand-written router   <-->  metrics.py
        |                                                        (counters,
        v                                                         latency,
  api.py         parse, validate, orchestrate, shape response     /metrics)
        |
        +--> features.py   schema, validation, engineering, WoE encoding, imputation
        +--> model.py      JSON artifact: coefficients, intercept, feature order
        +--> scorecard.py  integer score, bands A-E, threshold, reasons, contributions
        +--> storage.py    SQLite: applications table, application lookup, audit query
        +--> audit.py      append-only audit_events; no update or delete path
                 |
                 v
  config.py      env + JSON settings: paths, port, threshold, band edges, body limit
  SQLite file    applications, audit_events
```

Offline build-time pipeline, outside the request path:

```text
scripts/generate_dataset.py --> data/dataset.csv   (fixed seed, configurable N)
scripts/train_model.py      --> models/model.json  (WoE bins, coefficients, intercept,
                                                     imputation values, threshold, bands)
```

### 3.1 Request lifecycle: `POST /v1/score`

1. **Accept.** `server.py` accepts on a `ThreadingHTTPServer` and reads `Content-Length`; a body larger than `MAX_BODY_BYTES` is rejected with 413 before buffering.
2. **Route and parse.** The hand-written router matches method and path: unknown paths yield 404, and a known path with the wrong method yields 405 with an `Allow` header. The body is then decoded as UTF-8 and parsed with `json.loads`; malformed JSON or a non-object top level yields 400.
3. **Validate shape.** `features.py` checks required keys, types, and ranges. Missing keys, wrong types, and unparseable values yield 400 listing the offending fields; unknown keys are ignored and logged.
4. **Validate semantics.** Cross-field rules run next: non-positive income or `loan_amount`, negative ratios, `loan_term_months` outside the supported set. Violations yield 422.
5. **Feature engineering.** Values are clipped to training-time ranges, ratio features are recomputed when absent, `purpose_code` is mapped through the stored WoE table, missing optional values are imputed from the artifact's frozen statistics, and the result is reordered into the artifact's fixed feature order.
6. **Model inference.** `model.py` computes the linear combination and applies the sigmoid link, producing the probability of default.
7. **Band mapping.** `scorecard.py` converts the probability to a monotone integer score, maps the score to a band (A–E) via the configured edges, and applies the decision threshold to produce `approve` / `decline` / `review`.
8. **Contributions and reason codes.** Per-feature contributions are computed from the same linear terms, flagged when imputed, ranked by magnitude, and mapped to reason-code templates capped at a configured top-N.
9. **Persist.** `storage.py` inserts one `applications` row inside a transaction and commits, yielding the application id.
10. **Audit append.** `audit.py` appends an `audit_events` row recording request id, method, path, status, decision, model version, and latency. The append shares the insert transaction so an application is never stored without its audit record.
11. **Observe.** `metrics.py` increments request and error counters and records the observed latency.
12. **Respond.** `api.py` assembles `application_id`, `score`, `probability_of_default`, `band`, `decision`, `model_version`, `contributions[]`, and `reason_codes[]`; `server.py` writes it as `application/json` with `Content-Length`, or writes the error envelope on any failure path above.

## 4. Module layout

| Module | Responsibility |
| --- | --- |
| `src/riskscore/config.py` | Load settings from environment variables and an optional JSON file (database path, model path, host/port, body size limit, decision threshold, band edges, log level) and expose one immutable settings object. |
| `src/riskscore/features.py` | Feature schema (names, types, ranges, required/optional), request validation, clipping, ratio recomputation, categorical WoE encoding, and frozen missing-value imputation. |
| `src/riskscore/model.py` | Load and validate the JSON artifact; hold coefficients, intercept, feature order, and imputation values; expose `predict_proba(vector)` and `linear_terms(vector)`. |
| `src/riskscore/scorecard.py` | Probability-to-score mapping, band assignment (A–E), decision thresholding, contribution ranking, and reason-code derivation. |
| `src/riskscore/storage.py` | SQLite connection ownership, schema and index creation, application insert and lookup, and paginated audit queries. |
| `src/riskscore/audit.py` | Build and append audit records; enforce append-only access (no update or delete helper exists); serialize events for `/v1/audit`. |
| `src/riskscore/metrics.py` | In-process counters and latency histograms, score and probability distributions, model-version gauge, and Prometheus-style text rendering for `/metrics`. |
| `src/riskscore/api.py` | Route handler functions: parse and validate input, orchestrate domain modules, and build the response or error envelope. Contains no transport concerns. |
| `src/riskscore/server.py` | `ThreadingHTTPServer` subclass, hand-written router, HTTP parsing limits, headers, wire serialization, graceful shutdown, and the module entry point. |
| `scripts/generate_dataset.py` | Generate the synthetic dataset from a fixed seed with configurable N; write CSV plus a metadata sidecar (seed, N, generator version). |
| `scripts/train_model.py` | Fit WoE bins on the train split, run batch gradient descent, compute metrics (AUC, KS, accuracy, precision, recall, F1, confusion matrix), and write the JSON artifact plus a metrics report. |
| `tests/` | Unit and integration tests for features, model math, scorecard mapping, storage, the API contract, and the metrics registry. See Section 9. |

## 5. Data model

```sql
CREATE TABLE IF NOT EXISTS applications (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at             TEXT    NOT NULL,  -- ISO-8601 UTC
    request_id             TEXT    NOT NULL,  -- correlates logs, response, audit
    model_version          TEXT    NOT NULL,
    age                    INTEGER,           -- raw inputs as received
    annual_income          REAL,
    employment_years       REAL,
    debt_to_income_ratio   REAL,
    num_delinquencies_24m  INTEGER,
    credit_history_months  INTEGER,
    loan_amount            REAL,
    loan_term_months       INTEGER,
    num_open_accounts      INTEGER,
    revolving_utilization  REAL,
    has_mortgage           INTEGER,           -- 0/1
    purpose_code           TEXT,
    probability_of_default REAL    NOT NULL,  -- derived outputs
    score                  INTEGER NOT NULL,
    band                   TEXT    NOT NULL,  -- 'A'..'E'
    decision               TEXT    NOT NULL,  -- 'approve'|'decline'|'review'
    contributions_json     TEXT    NOT NULL,
    reason_codes_json      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_applications_created_at ON applications (created_at);
CREATE INDEX IF NOT EXISTS idx_applications_band       ON applications (band);
CREATE INDEX IF NOT EXISTS idx_applications_request_id ON applications (request_id);

CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at    TEXT    NOT NULL,  -- ISO-8601 UTC
    request_id     TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,  -- score_request|validation_error|read|startup
    application_id INTEGER,           -- NULL when not tied to a scored row
    method         TEXT,
    path           TEXT,
    status_code    INTEGER,
    latency_ms     REAL,
    model_version  TEXT,
    detail_json    TEXT               -- structured free-form payload
);

CREATE INDEX IF NOT EXISTS idx_audit_events_occurred_at    ON audit_events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_application_id ON audit_events (application_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_request_id     ON audit_events (request_id);
```

**`audit_events` is append-only by design.** Only INSERT and SELECT paths exist for this table; no UPDATE or DELETE statement appears anywhere in the codebase and no endpoint accepts a mutating verb for audit data. This is a deliberate non-repudiation property: an audit trail that the API can rewrite is not an audit trail. Retention, rotation, and off-host shipping are operational concerns and must copy rows out rather than mutate them in place.

`applications` rows are never updated by the API either. Raw inputs are stored beside the derived outputs so any score can be reproduced from the stored row plus the pinned model version.

## 6. Scoring math

With feature vector `x` in the artifact's fixed order, coefficients `w`, and intercept `b`:

`z = b + sum_i ( w_i * x_i )`

`p = sigmoid(z) = 1 / (1 + exp(-z))`
`logit(p) = ln( p / (1 - p) ) = z`

Coefficients are estimated by batch gradient descent on the train split, minimizing mean binary cross-entropy with optional L2 regularization:

`L = -(1/N) * sum_n [ y_n * ln(p_n) + (1 - y_n) * ln(1 - p_n) ] + (lambda/2) * sum_i ( w_i^2 )`
`w_i := w_i - eta * dL/dw_i`, `b := b - eta * dL/db`

where `eta` is the learning rate and the loop runs for a configured number of epochs. Features are WoE-transformed before fitting, so each coefficient weights one unit of WoE and the sigmoid is applied in the WoE domain.

**Integer score.** The probability maps monotonically onto a fixed range (for example 300–850, illustrative only):

`score = round( SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (1 - p) )`

so lower default probability yields a higher score. The mapping is monotone and lossy; the probability remains the authoritative output.

**Bands and decision.** Band edges are configuration, not learned parameters: `band = A if score >= edge_A`, `B if score >= edge_B`, ... `E otherwise`, with `edge_A > edge_B > edge_C > edge_D`. The threshold applies to `p`, not to the score: `p < threshold_low` gives `approve`, `p > threshold_high` gives `decline`, and the interval between them gives `review`. Default edges and thresholds are illustrative.

**Missing values.** Imputation is deterministic and frozen at training time: the artifact stores a training-split median for continuous features and a training-split mode for binary and categorical features. At inference a missing optional feature is replaced by that value and the response records `imputed: true` for it. A missing required feature is a 400 validation error, never an imputation.

**Per-feature contributions.** `contribution_i = w_i * x_i_effective`, where `x_i_effective` is the observed value or, when imputed, the frozen imputation value. Contributions are exactly additive:

`z = b + sum_i ( contribution_i )`

Reason codes come from ranking contributions by magnitude, taking the largest positive (risk-increasing) and negative (risk-decreasing) terms, and mapping them to templates such as `HIGH_DEBT_TO_INCOME` or `LONG_CREDIT_HISTORY`, capped at a configured top-N.

## 7. Error handling contract

Every error response uses the same envelope and sets `Content-Type: application/json`. `details` is always present, an empty list when there is nothing field-specific to report. `request_id` is echoed from the request header when supplied and generated otherwise, and appears in logs, the response, and `audit_events`.

```json
{
  "error": {
    "code": "validation_error",
    "message": "Human-readable summary.",
    "details": [{ "field": "annual_income", "issue": "must be a number greater than 0" }],
    "request_id": "req-..."
  }
}
```

| Status | `code` | When it is returned |
| --- | --- | --- |
| 400 Bad Request | `bad_request` | Body is not valid UTF-8, not valid JSON, not a JSON object, or has missing or wrongly typed fields. |
| 404 Not Found | `not_found` | No route matches the path, or `/v1/applications/<id>` references an unknown id. |
| 405 Method Not Allowed | `method_not_allowed` | The path exists but the method is unsupported; the response includes an `Allow` header. |
| 413 Payload Too Large | `payload_too_large` | `Content-Length` exceeds `MAX_BODY_BYTES`; the body is not read. |
| 422 Unprocessable Entity | `unprocessable_entity` | Body is well formed and well typed but semantically invalid (negative income, `loan_amount <= 0`, contradictory ratios). |
| 500 Internal Server Error | `internal_error` | Unhandled exception, unloadable or corrupt model artifact, or database failure. Details are logged server-side; the client gets a generic message plus `request_id`. |

Validation failures (400 and 422) are also appended to `audit_events` with `event_type = 'validation_error'`, because rejected input is analytically interesting. Error responses never include stack traces, filesystem paths, or SQL.

## 8. Observability

`GET /metrics` returns `text/plain; version=0.0.4` in Prometheus exposition format, so it can be scraped without adding a dependency. Exposed series:

- `riskscore_requests_total{route,method,status}` — request counter.
- `riskscore_request_duration_seconds` — latency histogram with fixed buckets, plus `_sum` and `_count`.
- `riskscore_scores_total{band}` and `riskscore_probability_of_default` — scores per band and the distribution of returned probabilities, for stability checks.
- `riskscore_model_info{version}` — gauge fixed at 1, carrying the loaded model version.
- `riskscore_errors_total{code}` and `riskscore_audit_events_total{event_type}` — error envelopes by error code and appended audit events by type.

`GET /healthz` returns 200 with `{"status":"ok","model_version":"...","uptime_s":N}` when the artifact is loaded and SQLite answers a trivial query; otherwise 503 naming the failing component.

**Latency measurement.** `metrics.py` takes a monotonic start timestamp (`time.perf_counter()`) at the top of the request handler and the elapsed duration after the response is assembled. The interval therefore covers routing, parsing, validation, inference, persistence, and audit append, and excludes socket write time. The same value is stored in `audit_events.latency_ms`, so `/metrics` and the audit trail agree by construction. Counters are process-local and in-memory: a restart resets them, and nothing is aggregated across processes because v0.1 runs one process.

## 9. Testing strategy

| Test file | What it covers |
| --- | --- |
| `tests/test_features.py` | Schema validation, required versus optional fields, type and range checks, clipping, ratio recomputation, WoE encoding of unseen categories, and imputation with frozen statistics, including boundary and missing-value cases. |
| `tests/test_model.py` | Artifact loading and rejection of malformed artifacts, feature-order enforcement, sigmoid numerical stability at extreme inputs, probability monotonicity, and the invariant that contributions sum to `z - b`. |
| `tests/test_scorecard.py` | Score mapping bounds, band-edge boundary behavior, decision threshold logic including the review interval, contribution ranking, and reason-code selection with top-N capping. |
| `tests/test_storage.py` | Schema creation idempotency across restarts, insert/read round-trip, index presence, append-only enforcement, and rollback leaving no partial application row. |
| `tests/test_api.py` | The full HTTP contract against a live server on an ephemeral port: the happy path, every status in Section 7, the error envelope shape, `Allow` on 405, body-size rejection before buffering, `/healthz`, and `/v1/applications/<id>` including the unknown-id case. |
| `tests/test_metrics.py` | Counter and histogram arithmetic, exposition rendering, label cardinality, latency recording, and that `/metrics` reflects traffic produced by earlier requests. |

Tests use only `unittest` and `http.client` and run under `make test`. Model tests build tiny in-memory artifacts rather than depending on a trained file, so the suite does not require `make train` to have run first.

## 10. Security and privacy considerations

- **No auth in v0.1.** Intended for `127.0.0.1` development; the server warns at startup when bound to a non-loopback address and must not be network-exposed without an authenticating proxy.
- **No real personal data.** Inputs are synthetic. The schema still stores applicant-shaped fields, so adapting this to real data immediately creates a privacy-sensitive system needing encryption at rest, retention limits, and a lawful basis.
- **SQL injection.** All statements use parameter binding; no user input is interpolated into SQL.
- **Request handling.** Routes are a fixed table of literal patterns, so no filesystem path derives from user input; `Content-Length` is required and conflicting or duplicated length headers are rejected.
- **Resource exhaustion.** `MAX_BODY_BYTES` caps body size, a parse guard bounds JSON nesting depth, the thread budget bounds concurrency, and numeric inputs are range-checked before arithmetic.
- **Log and error hygiene.** Log fields are JSON-escaped and client-supplied strings such as `request_id` and `purpose_code` are truncated and sanitized; internal errors never surface stack traces, SQL, or file paths.
- **Audit and artifact integrity.** Audit events are append-only with no mutating endpoint and a `request_id` shared by logs, response, and audit row; the artifact path is configured, its schema and feature order are validated before use, and a checksum is returned in response metadata so a score ties to an exact model revision.

## 11. Tradeoffs and alternatives

| Decision | Alternative | Rationale |
| --- | --- | --- |
| stdlib `ThreadingHTTPServer` | FastAPI, Flask, Uvicorn | Keeps the dependency count at zero, which is the point of the project, and makes the HTTP layer fully inspectable. Costs hand-written routing, validation, and serialization, and no automatic OpenAPI or async I/O. Throughput is not a requirement. |
| SQLite | PostgreSQL | One file, no server process, hermetic tests, and `make run` works anywhere. Costs coarse write concurrency (single writer) and no networked data access. Adequate for a single-process educational service; a real deployment needs Postgres for concurrency, backups, and access control. |
| Linear logistic scorecard | Gradient boosting, random forest | A linear model on WoE features yields coefficients that can be read, diffed, and summed into exact per-feature contributions, which is the explainability property being demonstrated. Costs capacity: no interactions, and likely underfitting on real data. Interpretability was chosen over accuracy deliberately. |
| JSON model artifact | `pickle`, joblib, ONNX | JSON is readable, diffable in git, language-agnostic, and safe to load from an untrusted path. `pickle` is smaller and faster but executes arbitrary code on load and is opaque in review. Costs a larger file and explicit serialization code. |
| Hand-written router | A framework router | A literal route table is a few dozen lines with no hidden behavior, letting the Section 7 error contract be enforced in one place. Costs manual `Allow` headers, manual 404/405, and no automatic path-parameter binding. |
| Synthetic data | German Credit, Lending Club, or another public dataset | Synthetic data is redistributable without licence questions, contains no personal data, and is exactly reproducible from a seed, which a portfolio artifact needs. Costs real-world signal: metrics measure recovery of the generator, and no fairness analysis is possible. A public dataset would add licence, privacy, and preprocessing burdens plus its own fairness limits. |

## 12. Extension points

**New feature.** Add the field to the schema in `features.py` (name, type, range, required flag, imputation strategy), implement its validation and engineering, extend `scripts/generate_dataset.py` so the column exists, retrain with `make train` to produce new WoE bins and coefficients, and add cases to `tests/test_features.py`. Feature order is part of the artifact contract, so reordering existing features is a breaking change requiring retraining and a version bump. A new optional feature only works against an artifact that was trained with it, because the artifact pins the vector.

**New model backend.** `model.py` is the only module that knows the artifact's internal representation. Define a common interface — `load(path)`, `predict_proba(vector)`, `linear_terms(vector)`, `metadata()` — and implement it per backend. `scorecard.py` and the response contract rely on an additive decomposition, so a tree backend must supply SHAP-style attributions or declare contributions unsupported. Backend selection belongs in `config.py`, and the backend name is recorded in artifact metadata and in every response.

**New storage driver.** `storage.py` already exposes a narrow surface: initialize schema, insert application, fetch application by id, append audit event, query audit events. Extract that as a protocol, keep SQLite as the default implementation, and add Postgres or a remote store behind the same interface. Append-only audit is part of the interface, not the SQLite implementation: every driver exposes append and query only. Callers in `api.py` must not depend on SQLite specifics such as `lastrowid`, so insertion returns a driver-neutral identifier.

## 13. Milestones

**v0.1 (current).** Synthetic generator with a fixed seed; from-scratch logistic training with WoE features and a JSON artifact; `POST /v1/score` with contributions, reason codes, bands, and a configurable threshold; SQLite persistence with append-only audit; `/v1/applications/<id>` and `/v1/audit`; `/healthz` and `/metrics`; Makefile targets `data`, `train`, `run`, `test`, `bench`; GitHub Actions CI; Dockerfile and `docker-compose.yml`; README in Chinese and English. No authentication.

**v0.2 (validation and operations).** Slice reporting by band, income decile, and loan purpose emitted by the training script and written into `MODEL_CARD.md`; calibration curve and reliability table; PSI drift check against a reference score distribution; structured JSON logging with a request correlator; optional API-key authentication and per-key rate limiting; an audit export command that copies rows out without mutating them.

**v0.3 (extensibility and deployment realism).** Pluggable model backends behind the Section 12 interface with at least one comparison backend; pluggable storage driver with a Postgres implementation exercising the same interface; batch scoring CLI sharing `features.py` and `model.py` with the API; a maintained OpenAPI description matching the error contract; container hardening (non-root user, read-only root filesystem, pinned base digest) and a documented reverse-proxy topology. Any move toward human-affecting use additionally requires the fairness, calibration, and governance work described in `MODEL_CARD.md`.
