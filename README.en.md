# credit-risk-scoring-service

A dependency-free credit risk scoring microservice in Python. It trains a logistic-regression
scorecard from scratch (batch gradient descent, no numpy, no scikit-learn), serves it over a REST API
built on the standard library HTTP server, persists applications and an append-only audit trail in
SQLite, and ships health checks, metrics and model-governance artifacts alongside a full `unittest`
suite.

```text
Python 3.9+ | License: MIT | Runtime dependencies: none | Tests: 244 unittest cases, all passing
```

[![CI](https://github.com/Nausicaa613/credit-risk-scoring-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Nausicaa613/credit-risk-scoring-service/actions/workflows/ci.yml)

[中文 README](README.md) | [Design document](docs/DESIGN.md) | [Model card](MODEL_CARD.md) | [Contributing](CONTRIBUTING.md)

---

## Why this project exists

A credit scoring system carries a constraint most machine-learning projects do not: **it has to survive
review**. A regulator, an auditor or a credit committee needs to answer "why did this applicant score
612?", and the answer must match the arithmetic the model actually performed at the time.

A linear scorecard remains the industry-standard baseline precisely because that question has an exact
answer. In log-odds space the model is additive:

```text
logit(PD) = intercept + sum_i (weight_i * feature_i)
```

Every term `weight_i * feature_i` is that feature's contribution to the decision, so explainability is a
property of the model rather than an approximation bolted on afterwards. Gradient boosting would very
likely rank better on AUC (see the roadmap); that is the price of this auditability, and it is a
deliberate trade.

The second constraint is **reproducibility**. The same data and the same seed must produce the same
model, which is why the artifact is plain JSON — diffable and reviewable — rather than a pickle.

## Features

- **Explainable by construction.** Every response carries all 18 feature contributions, and the
  contributions plus the intercept reconstruct the model's log-odds exactly (asserted by a test).
- **Reason codes.** The three largest contributions are surfaced with the direction of their effect and
  compared against the expected business prior.
- **Calibrated probabilities.** Class-weighted training inflates probabilities, so a bias-only log-odds
  correction (Platt scaling) is fitted on a dedicated calibration split. Weights, and therefore the
  ranking, AUC and KS, are untouched.
- **Industry score scale.** Points-to-double-the-odds: `score = 600 + (40 / ln 2) * ln(good:bad odds)`,
  so a higher score always means lower risk.
- **Derived risk bands.** The A–E edges are *computed* from probability cut-offs rather than hard-coded,
  so a band label can never contradict the probability reported beside it.
- **Append-only audit trail.** `AuditLog` exposes only `append` and `tail`; no update or delete path
  exists in the code. The decision row and its audit row commit in a single transaction.
- **Observable.** `/healthz` reports liveness and readiness; `/metrics` exposes counters, a latency
  histogram and a score summary.
- **Zero runtime dependencies.** CI parses the AST of `src/` and fails the build if a third-party import
  appears.
- **Reproducible synthetic data.** A fixed-seed generator that solves for its intercept by bisection so
  the mean true probability matches a requested base rate.

## Architecture

```text
  client
    |  POST /v1/score  {"age": 40, ...}
    v
+--------------------------------------------------------------+
| server.py   ThreadingHTTPServer + hand-written router        |
|             - parses request line, headers and body          |
|             - enforces a body-size cap                       |
|             - assigns request_id, sets headers, logs         |
+---------------------------+----------------------------------+
                            v
+--------------------------------------------------------------+
| api.py      transport-neutral service layer                  |
|             - routing, query validation, error envelopes     |
+-------+----------------------------------+-------------------+
        v                                  v
+------------------------+      +------------------------------+
| features.py            |      | storage.py / audit.py        |
| - field contract       |      | - applications (decisions)   |
| - log1p + one-hot      |      | - audit_events (append-only) |
+-----------+------------+      | - both rows, one transaction |
            v                   +------------------------------+
+------------------------+
| model.py               |      score / band / reason codes
| - logistic regression  |               |
| - gradient descent     |               v
| - metrics, JSON model  |      +------------------------------+
+-----------+------------+      | scorecard.py                 |
            v                   | - PD -> score (invertible)   |
   models/model.json            | - score -> band              |
   (plain JSON, diffable)       | - contributions -> reasons   |
                                +------------------------------+
```

Repository layout:

```text
credit-risk-scoring-service/
├── src/riskscore/          # the runtime package (pure standard library)
│   ├── config.py           # Settings dataclass, environment-variable parsing
│   ├── errors.py           # error hierarchy with HTTP status + stable codes
│   ├── features.py         # validation, log/one-hot transforms, feature contract
│   ├── model.py            # logistic scorecard, gradient descent, metrics, persistence
│   ├── scorecard.py        # probability -> score/band/reason-code translation
│   ├── storage.py          # SQLite schema, ApplicationStore, connection helper
│   ├── audit.py            # append-only audit trail
│   ├── metrics.py          # counters, latency histogram, score summary
│   ├── api.py              # transport-neutral router and request handling
│   └── server.py           # http.server transport (ThreadingHTTPServer)
├── scripts/
│   ├── generate_dataset.py # synthetic dataset generator (fixed seed)
│   ├── train_model.py      # train, calibrate, select threshold, report
│   ├── smoke_check.py      # end-to-end check in a temp directory
│   ├── bench.py            # timings for training and scoring
│   └── diagnose.py         # score/band inspection helper
├── tests/                  # unittest suite (244 cases)
├── docs/DESIGN.md
├── MODEL_CARD.md
├── CONTRIBUTING.md
├── Makefile                # GNU make targets
├── run.ps1                 # identical targets for Windows without make
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .github/workflows/ci.yml
├── LICENSE                 # MIT
├── README.md               # Chinese
└── README.en.md            # this file
```

Generated at runtime and git-ignored (fully reproducible): `data/applications.jsonl`,
`models/model.json`, `reports/training_metrics.json`, `reports/training_summary.md`.

## Quickstart

No virtualenv and no `pip install` needed — only Python 3.9+.

```bash
git clone https://github.com/Nausicaa613/credit-risk-scoring-service.git
cd credit-risk-scoring-service
```

With GNU make:

```bash
make data     # generate the synthetic dataset (fixed seed, reproducible)
make train    # train, calibrate, select the threshold, write the reports
make test     # 244 unit and integration tests
make smoke    # end-to-end check in a temp directory
make run      # serve on http://127.0.0.1:8080
make bench    # performance benchmark
```

On Windows without make, `run.ps1` provides identical targets and defaults:

```powershell
.\run.ps1 data
.\run.ps1 train
.\run.ps1 test
.\run.ps1 smoke
.\run.ps1 run
```

Without either wrapper:

```bash
python scripts/generate_dataset.py --rows 6000 --seed 20260101 --output data/applications.jsonl
python scripts/train_model.py --data data/applications.jsonl --model models/model.json
python -m unittest discover -s tests -t . -v
.\run.ps1 run                     # or `make run`; both set PYTHONPATH=src for you
python -m riskscore.server        # only after `make install` (pip install -e .)
```

The scripts and `conftest.py` put `src/` on the import path themselves, and `make run` / `run.ps1 run`
set both the `RISKSCORE_*` variables and `PYTHONPATH=src`, so a fresh clone starts with no install
step. `make install` is only needed if you want `riskscore` importable from anywhere on your machine.

Score an application:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/score \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 40, "annual_income": 300000, "employment_years": 8.0,
    "debt_to_income_ratio": 0.30, "num_delinquencies_24m": 0,
    "credit_history_months": 150, "loan_amount": 200000,
    "loan_term_months": 36, "num_open_accounts": 5,
    "revolving_utilization": 0.30, "purpose": "equipment"
  }'
```

The reply is `201` with a `Location: /v1/applications/<id>` header. Field names below are exact; the
values depend on the model you trained.

```json
{
  "request_id": "req_5f3a91c2b7d4",
  "created_at": "2026-10-04T12:00:00.000Z",
  "result": {
    "application_id": "app_9c1f8e2a4b6d0f31",
    "credit_score": 690,
    "probability_of_default": 0.1472,
    "risk_band": "D",
    "risk_band_label": "elevated risk",
    "decision": "approve",
    "decision_threshold": 0.315,
    "model_version": "0.1.0",
    "expected_default_rate": "12% - 25%",
    "log_odds": -1.755,
    "intercept": -1.318,
    "contributions": [
      {"feature": "revolving_utilization", "contribution": -0.154},
      {"feature": "debt_to_income_ratio", "contribution": -0.109}
    ],
    "reason_codes": [
      {
        "feature": "revolving_utilization",
        "description": "revolving utilisation",
        "contribution": -0.154,
        "direction": "decreases_risk",
        "expected_direction": "risk_up",
        "consistent_with_prior": false
      }
    ]
  }
}
```

Both `contributions` and `reason_codes` are always present. A decline notice that lists only the factors
that raised risk cannot be reviewed, so both directions are reported.

## Results

The dataset is **synthetic**. The numbers below describe performance on that synthetic distribution
only and are not evidence of performance on real lending data.

```bash
make data && make train
```

| Item | Value |
| --- | --- |
| Dataset | 6,000 rows, default rate 0.2187 (1312/6000), 12% label noise |
| Split | train / validation / calibration / test = 3600 / 600 / 600 / 1200 |
| Model selection | dedicated validation set (early stopping on validation AUC) |
| Epochs | 400 (learning rate 0.35, L2 0.001, seed 20260101) |

**Held-out test metrics at threshold 0.315**

| Metric | Value |
| --- | --- |
| AUC | 0.7337 |
| KS | 0.3934 |
| Accuracy | 0.7450 |
| Precision | 0.4380 |
| Recall | 0.5779 |
| F1 | 0.4984 |
| Predicted positive rate | 0.2892 |

Confusion matrix (rows = actual, columns = predicted): TN 742, FP 195, FN 111, TP 152.

**Calibration.** After class-weighted training the mean predicted PD was 0.4639 against an observed
calibration rate of 0.2283. The fitted log-odds offset was **−1.0730**; after correction the mean
predicted PD on test is **0.2541** against an observed test default rate of 0.2192.

**Top features by contribution spread** (measured on the test set)

| Rank | Feature | Weight | Contribution std |
| --- | --- | --- | --- |
| 1 | `revolving_utilization` | +0.3676 | 0.3789 |
| 2 | `debt_to_income_ratio` | +0.3650 | 0.3665 |
| 3 | `num_delinquencies_24m` | +0.1831 | 0.1821 |
| 4 | `log_annual_income` | −0.1621 | 0.1690 |
| 5 | `age` | −0.1611 | 0.1603 |

The directions match business priors: utilisation, debt-to-income and delinquencies raise risk; income
and age lower it.

**Known limitation.** The binned calibration table shows the middle of the distribution is still
over-predicted (bins 2–4 by +0.066 to +0.100). A single global offset can align the mean but cannot
correct the shape distortion introduced by label noise. Because the band edges come from fixed
probability cut-offs, band A is also nearly empty on this dataset.

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/score` | Score one application. `201` with a `Location` header. |
| `GET` | `/v1/applications/<id>` | Fetch a stored decision: original payload plus full response. `200`. |
| `GET` | `/v1/applications?limit=&offset=` | Recent decisions, newest first. `limit` 1–200 (default 20), `offset` default 0. |
| `GET` | `/v1/audit?limit=&application_id=` | Audit trail, newest first. `limit` 1–200 (default 20). |
| `GET` | `/v1/bands` | Risk band definitions with their score and PD ranges. `200`. |
| `GET` | `/healthz` | `200` healthy, `503` degraded. Includes model and database state. |
| `GET` | `/metrics` | In-process metrics. `404` when `RISKSCORE_METRICS_ENABLED=0`. |
| `GET` | `/` or `/v1` | Service index and endpoint list. |

### Request fields

Eleven required fields: `age`, `annual_income`, `employment_years`, `debt_to_income_ratio`,
`num_delinquencies_24m`, `credit_history_months`, `loan_amount`, `loan_term_months`,
`num_open_accounts`, `revolving_utilization`, `purpose`.

`purpose` must be one of `working_capital`, `equipment`, `expansion`, `inventory`, `refinance`,
`personal`, `education`, `other`.

Accepted aliases: `income`/`annualIncome` → `annual_income`, `dti`/`debtToIncomeRatio` →
`debt_to_income_ratio`, `loanAmount` → `loan_amount`, `loanTermMonths`/`term_months` →
`loan_term_months`, `delinquencies` → `num_delinquencies_24m`, `purpose_code` → `purpose`. An optional
client-supplied `application_id` must be 6–64 characters of letters, digits, `_`, `.`, `:` or `-`.

Validation reports **every** problem at once rather than stopping at the first, so a client can fix its
payload in one iteration:

```json
{
  "error": {
    "code": "validation_error",
    "message": "payload failed schema validation",
    "details": {
      "problems": [
        {"field": "age", "problem": "is required"},
        {"field": "purpose", "problem": "is required"}
      ],
      "allowed_fields": ["age", "annual_income", "..."]
    }
  },
  "request_id": "req_5f3a91c2b7d4"
}
```

### Status codes

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `validation_error` | Missing field, wrong type, out of range, or malformed JSON. |
| 404 | `not_found` | Unknown path or application id. |
| 405 | `method_not_allowed` | Path exists but not for this method; an `Allow` header is sent. |
| 411 | `length_required` | Chunked transfer encoding was used. |
| 413 | `payload_too_large` | Body exceeded `RISKSCORE_MAX_BODY_BYTES`. |
| 422 | `unprocessable_entity` | Well-formed JSON that is semantically impossible. |
| 503 | `model_unavailable` | No model loaded; run `make train` first. |

## Model card summary

- **Features:** 12 raw application fields expanding to 18 model inputs. `annual_income` and
  `loan_amount` are `log1p`-compressed; `debt_to_income_ratio` and `revolving_utilization` are clipped
  at 1.5; `purpose` becomes eight one-hot indicators.
- **Training:** class-weighted binary cross-entropy plus an L2 penalty, minimised by batch gradient
  descent with early stopping on validation AUC. Implemented in plain Python.
- **Threshold:** selected on the validation split by maximising F1 (`auto` mode), overridable with
  `--threshold-mode fixed --threshold X`.
- **Reported metrics:** AUC and KS (threshold-free ranking), plus accuracy, precision, recall, F1 and
  the confusion matrix at the chosen threshold.
- **Limitations:** synthetic data; not for real lending decisions; **no fairness or disparate-impact
  evaluation has been performed**; no drift monitoring; a linear model may underfit.

The full card, including intended use, out-of-scope uses and the list of known gaps, is in
[MODEL_CARD.md](MODEL_CARD.md).

## Configuration

All settings come from environment variables; the defaults work out of the box.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RISKSCORE_HOST` | `127.0.0.1` | Bind address. |
| `RISKSCORE_PORT` | `8080` | TCP port. |
| `RISKSCORE_DB_PATH` | `data/riskscore.db` | SQLite file path. |
| `RISKSCORE_MODEL_PATH` | `models/model.json` | Model artifact path. |
| `RISKSCORE_DECISION_THRESHOLD` | `0.35` | Probability of default above which an application is declined. |
| `RISKSCORE_MAX_BODY_BYTES` | `65536` | Request body cap. |
| `RISKSCORE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `RISKSCORE_METRICS_ENABLED` | `1` | Set to `0` to stop exposing `/metrics`. |

## Testing

```bash
make test          # verbose
make test-quiet    # summary only
make check         # compile check plus the end-to-end smoke check
```

`python -m unittest discover -s tests -t . -v`: 244 tests, all passing, in about 16 seconds. The suite
is standard-library `unittest` and covers field validation and aliases, transform correctness, AUC/KS
ranking including ties, training convergence and determinism, exact additivity of contributions, score
monotonicity, band edges and label consistency, SQLite transaction atomicity, the append-only audit
guarantee, routing and every error code, and HTTP/1.1 keep-alive framing **over a real socket**.

## Design decisions and tradeoffs

| Decision | Alternative considered | Why |
| --- | --- | --- |
| Standard library only | FastAPI + scikit-learn | Runs anywhere Python does, with no dependency surface to audit or patch. |
| Stdlib `http.server` | Flask / FastAPI | The request lifecycle stays explicit, which is exactly what a reviewer asks about; the cost is no async and no WebSockets. |
| SQLite | PostgreSQL | Single-file, zero-ops and easy to inspect; the v0.1 write volume does not need a server database. |
| Linear scorecard | Gradient boosting | Explainability is a property of the model rather than an approximation; trades some AUC for auditability. |
| JSON model artifact | pickle / joblib | Diffable and reviewable, with no arbitrary-code-execution risk; the cost is validating the format version on load. |
| Synthetic data with a fixed seed | A public credit dataset | No licensing or privacy constraints, byte-for-byte reproducible, and the true generating process is known, so "the model is bad" is distinguishable from "the data has no signal". |
| One transaction for the decision and audit rows | Two independent writes | One fsync instead of two (measured request time dropped from 59 ms to 33 ms) and no window in which a decision exists without its audit entry. |
| One connection per operation | A connection pool or a single writer thread | Avoids a whole class of threading bugs at roughly 9 ms per operation; this is the documented next optimisation. |

## Performance

Measured on Windows with Python 3.12 (`make bench`):

| Operation | Cost |
| --- | --- |
| Feature engineering | about 13 µs per row (18 features) |
| Training | about 3 ms per epoch (1,200 rows) |
| Scoring, `Scorer.score_vector` | about 0.011 ms per row |
| Model artifact | 3.8 KiB of JSON, about 2.3 ms to load |
| Full request including persistence | about 33 ms, p50 27 ms |

Scoring is microseconds; the request cost is almost entirely SQLite durability. A single transaction
inserting 100 rows takes about 25 ms (0.26 ms per row), which points at connection pooling, batched
commits, or `synchronous=NORMAL` (common with WAL, still safe across an application crash, at the risk
of losing the most recent commits on a power loss) as the next steps.

## Roadmap

- Swap in a gradient boosting backend behind the same interface, keeping `contributions` (via SHAP
  values) so the AUC-versus-interpretability trade can be measured rather than asserted.
- A fairness report: disparate-impact analysis sliced by income decile and loan purpose. **No fairness
  evaluation has been done**, which is a known gap.
- Authentication, multi-tenancy and per-request quotas.
- A storage-driver abstraction with a PostgreSQL implementation, plus a connection pool to remove the
  per-request connection cost.
- Drift monitoring: online comparison of score and feature distributions.
- Replace `/metrics` with the Prometheus text exposition format.

## License

MIT, see [LICENSE](LICENSE).

> **Important:** this is a portfolio and educational project. The dataset is entirely **synthetic**,
> generated locally by `scripts/generate_dataset.py` from a fixed seed; no real applicant data is used
> anywhere in this repository. It is **not** a credit decision system and **must not** be used to make
> lending decisions about real people or businesses. Intended use, limitations and known gaps are
> documented in [MODEL_CARD.md](MODEL_CARD.md).
