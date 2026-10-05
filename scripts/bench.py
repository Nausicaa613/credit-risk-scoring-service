#!/usr/bin/env python3
"""Measure training and scoring performance.

A credit scoring service has an unusual performance profile: the model is
retrained rarely (offline, in batch) while score requests arrive constantly and
must be fast. So the interesting numbers are the *per-request* cost, not the
training throughput.

This script reports:

1. dataset load and feature-building cost,
2. training wall time and milliseconds per epoch,
3. model artifact save/load size and time,
4. in-process scoring latency across a range of batch sizes,
5. a storage breakdown that attributes persistence time to a specific operation,
6. HTTP round-trip latency against a live server, when one is reachable.

No third-party package is needed. Percentiles come from a sort, which is honest
for the thousands of samples measured here.

Usage
-----
    python scripts/bench.py
    python scripts/bench.py --rows 2000 --scoring-rounds 2000
    python scripts/bench.py --base-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from generate_dataset import generate_dataset, write_jsonl  # noqa: E402

from riskscore.features import FEATURE_NAMES, build_features, normalise_payload  # noqa: E402
from riskscore.model import ScorecardModel, TrainingConfig, train_scorecard  # noqa: E402
from riskscore.scorecard import Scorer  # noqa: E402

DEFAULT_MODEL = os.path.join(_ROOT, "models", "model.json")
DEFAULT_DATA = os.path.join(_ROOT, "data", "applications.jsonl")


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------


def timeit(function: Callable[[], Any], *, repeat: int = 1) -> Tuple[float, Any]:
    """Return ``(mean_seconds_per_call, last_result)``."""
    total = 0.0
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = function()
        total += time.perf_counter() - started
    return total / repeat, result


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile; values must be sorted ascending."""
    if not values:
        return 0.0
    index = min(int(round(fraction * (len(values) - 1))), len(values) - 1)
    return values[index]


def report_latency(label: str, samples_ms: List[float]) -> Dict[str, float]:
    """Print and return a latency summary in milliseconds."""
    ordered = sorted(samples_ms)
    summary = {
        "count": float(len(ordered)),
        "min": ordered[0] if ordered else 0.0,
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else 0.0,
    }
    print(
        f"  {label:<28} n={int(summary['count']):<6} "
        f"mean {summary['mean']:8.3f} ms   p50 {summary['p50']:8.3f}   "
        f"p90 {summary['p90']:8.3f}   p99 {summary['p99']:8.3f}   max {summary['max']:8.3f}"
    )
    return summary


def human_bytes(count: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.1f} {unit}" if unit != "B" else f"{count} B"
        count /= 1024.0
    return f"{count:.1f} GiB"


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


def bench_data(rows: int, seed: int, workdir: str) -> Tuple[List[Dict[str, Any]], List[Tuple[float, ...]], List[int], float]:
    print(f"[1/6] generating {rows} synthetic applications")
    started = time.perf_counter()
    dataset = generate_dataset(rows, seed=seed)
    generate_seconds = time.perf_counter() - started
    print(f"  generate                 {generate_seconds:8.3f} s   ({rows / generate_seconds:,.0f} rows/s)")

    path = os.path.join(workdir, "applications.jsonl")
    started = time.perf_counter()
    write_jsonl(dataset, path)
    write_seconds = time.perf_counter() - started
    size = os.path.getsize(path)
    print(f"  write JSONL              {write_seconds:8.3f} s   ({human_bytes(size)})")

    def build() -> List[Tuple[float, ...]]:
        matrix = []
        for row in dataset:
            payload = {k: v for k, v in row.items() if not k.startswith("_") and k != "defaulted"}
            matrix.append(build_features(normalise_payload(payload)).values)
        return matrix

    per_call, matrix = timeit(build, repeat=3)
    assert matrix is not None
    print(
        f"  feature engineering      {per_call:8.3f} s   "
        f"({per_call / rows * 1e6:,.1f} us/row, {len(FEATURE_NAMES)} features)"
    )
    labels = [int(row["defaulted"]) for row in dataset]
    return dataset, matrix, labels, generate_seconds


def bench_training(
    matrix: List[Tuple[float, ...]], labels: List[int], epochs: int
) -> Tuple[ScorecardModel, float]:
    print(f"[2/6] training the scorecard ({epochs} epochs, early stopping on validation AUC)")
    cut = max(1, int(len(matrix) * 0.8))
    started = time.perf_counter()
    model, report = train_scorecard(
        matrix[:cut],
        labels[:cut],
        feature_names=FEATURE_NAMES,
        config=TrainingConfig(epochs=epochs, verbose=False),
        validation=(matrix[cut:], labels[cut:]),
    )
    elapsed = time.perf_counter() - started
    print(f"  train {len(matrix[:cut])} rows       {elapsed:8.3f} s   ({elapsed / epochs * 1000:,.2f} ms/epoch)")
    print(
        f"  holdout AUC {report['test']['auc']:.4f}   KS {report['test']['ks']:.4f}   "
        f"(model selection: {report['dataset']['model_selection']})"
    )
    return model, elapsed


def bench_artifact(model: ScorecardModel, workdir: str) -> None:
    print("[3/6] model artifact")
    path = os.path.join(workdir, "model.json")

    save_seconds, _ = timeit(lambda: model.save(path), repeat=5)
    size = os.path.getsize(path)
    load_seconds, loaded = timeit(lambda: ScorecardModel.load(path), repeat=5)
    assert loaded is not None

    with open(path, "r", encoding="utf-8") as handle:
        parameters = len(json.load(handle)["weights"])
    print(f"  save                     {save_seconds * 1000:8.3f} ms")
    print(f"  load                     {load_seconds * 1000:8.3f} ms")
    print(f"  size                     {human_bytes(size):>10}   {parameters} weights as plain JSON")


def bench_scoring(model: ScorecardModel, matrix: List[Tuple[float, ...]], rounds: int) -> None:
    print(f"[4/6] in-process scoring ({rounds} single-row calls, then batch calls)")
    scorer = Scorer(model)
    sample = matrix[: 2000 if len(matrix) > 2000 else len(matrix)]

    single: List[float] = []
    for index in range(rounds):
        row = sample[index % len(sample)]
        started = time.perf_counter()
        scorer.score_vector(f"app_bench{index:08d}", row)
        single.append((time.perf_counter() - started) * 1000.0)
    report_latency("score_vector (1 row)", single)

    for batch_size in (10, 100, 1000):
        if batch_size > len(sample):
            continue
        batches = max(1, rounds // batch_size)
        per_row: List[float] = []
        for _ in range(batches):
            rows = sample[:batch_size]
            started = time.perf_counter()
            for row in rows:
                scorer.score_vector("app_batch", row)
            per_row.append((time.perf_counter() - started) * 1000.0 / batch_size)
        report_latency(f"score_vector (batch {batch_size})", per_row)

    # The full request path including validation, persistence and audit. Feature
    # rows are post-transform, so this part needs real payloads: it regenerates a
    # small dataset rather than trying to invert the transform.
    from riskscore.api import RequestContext, ScoringService
    from riskscore.config import Settings

    with tempfile.TemporaryDirectory() as db_dir:
        settings = Settings(
            db_path=os.path.join(db_dir, "bench.db"),
            model_path=os.path.join(db_dir, "unused.json"),
            decision_threshold=model.decision_threshold,
        )
        service = ScoringService(settings, model=model)
        requests: List[float] = []
        fresh = generate_dataset(min(len(sample), 500), seed=4242)
        for record in fresh:
            payload = {
                key: value
                for key, value in record.items()
                if not key.startswith("_") and key != "defaulted"
            }
            body = json.dumps(payload).encode("utf-8")
            started = time.perf_counter()
            response = service.handle(
                RequestContext(method="POST", path="/v1/score", body=body, request_id="req_bench")
            )
            requests.append((time.perf_counter() - started) * 1000.0)
            if response.status != 201:
                raise SystemExit(f"benchmark request failed with status {response.status}")
        report_latency("full request (SQLite write)", requests)


def bench_storage(rounds: int = 200) -> None:
    """Break the persistence cost down so a regression is attributable.

    The aggregate request timing is not actionable on its own: it cannot say
    whether the time is spent opening a connection, writing an application row,
    or appending to the audit trail. Each of those is timed separately here, and
    each is timed with real per-operation connections, matching how the service
    actually uses SQLite.
    """
    from riskscore.audit import EVENT_SCORED, AuditLog
    from riskscore.storage import ApplicationStore, connect, init_db, utc_now_iso

    print(f"[6/6] storage breakdown ({rounds} rounds, per-operation connections)")
    with tempfile.TemporaryDirectory(prefix="riskscore-storage-") as directory:
        db_path = os.path.join(directory, "breakdown.db")
        init_db(db_path)
        store = ApplicationStore(db_path)
        audit = AuditLog(db_path)

        open_ms: List[float] = []
        for _ in range(rounds):
            started = time.perf_counter()
            with connect(db_path) as connection:
                connection.execute("SELECT 1").fetchone()
            open_ms.append((time.perf_counter() - started) * 1000.0)
        report_latency("connect + SELECT 1", open_ms)

        read_ms: List[float] = []
        for _ in range(rounds):
            started = time.perf_counter()
            store.count()
            read_ms.append((time.perf_counter() - started) * 1000.0)
        report_latency("count()", read_ms)

        write_ms: List[float] = []
        for index in range(rounds):
            started = time.perf_counter()
            store.save(
                application_id=f"app_bench{index:08d}",
                credit_score=650,
                probability_of_default=0.2,
                risk_band="C",
                decision="approve",
                model_version="bench",
                payload={"age": 40},
                response={"credit_score": 650},
                created_at=utc_now_iso(),
            )
            write_ms.append((time.perf_counter() - started) * 1000.0)
        report_latency("ApplicationStore.save", write_ms)

        audit_ms: List[float] = []
        for index in range(rounds):
            started = time.perf_counter()
            audit.append(EVENT_SCORED, {"n": index}, application_id=f"app_bench{index:08d}")
            audit_ms.append((time.perf_counter() - started) * 1000.0)
        report_latency("AuditLog.append", audit_ms)

        # One transaction covering many writes, to show what batching would buy.
        batched = 100
        started = time.perf_counter()
        with connect(db_path) as connection:
            connection.execute("BEGIN")
            for index in range(batched):
                connection.execute(
                    "INSERT INTO audit_events "
                    "(application_id, created_at, event_type, request_id, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (f"app_bulk{index:08d}", utc_now_iso(), EVENT_SCORED, None, "{}"),
                )
            connection.execute("COMMIT")
        batched_ms = (time.perf_counter() - started) * 1000.0
        print(
            f"  {'single transaction, 100 inserts':<28} total {batched_ms:8.3f} ms   "
            f"({batched_ms / batched:.3f} ms/row)"
        )
        print(
            "  note: one connection per operation is a deliberate simplicity tradeoff; "
            "a pool or a single writer thread is the documented next step"
        )


def bench_http(base_url: str, rounds: int) -> None:
    print(f"[5/6] HTTP round-trip against {base_url}")
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
    body = json.dumps(payload).encode("utf-8")

    try:
        with urllib.request.urlopen(f"{base_url}/healthz", timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
        print(f"  /healthz status          {health.get('status')}   model loaded: {health.get('model', {}).get('loaded')}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  no server at {base_url} ({exc}); start one with 'make run' and re-run")
        return

    samples: List[float] = []
    for _ in range(rounds):
        request = urllib.request.Request(
            f"{base_url}/v1/score",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        samples.append((time.perf_counter() - started) * 1000.0)
    report_latency("POST /v1/score", samples)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=4000, help="synthetic rows for the training benchmark (default 4000)")
    parser.add_argument("--seed", type=int, default=20260101, help="dataset seed")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs (default 200)")
    parser.add_argument("--scoring-rounds", type=int, default=2000, help="single-row scoring calls (default 2000)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="live server for the HTTP benchmark")
    parser.add_argument("--skip-http", action="store_true", help="skip the HTTP benchmark entirely")
    parser.add_argument("--skip-training", action="store_true", help="reuse models/model.json instead of training")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("credit-risk-scoring-service benchmark")
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="riskscore-bench-") as workdir:
        if args.skip_training and os.path.exists(DEFAULT_MODEL):
            print(f"[1/6] loading existing dataset metadata from {DEFAULT_DATA}")
            print("[2/6] loading existing model from models/model.json")
            model = ScorecardModel.load(DEFAULT_MODEL)
            dataset, matrix, labels, _ = bench_data(args.rows, args.seed, workdir)
        else:
            dataset, matrix, labels, _ = bench_data(args.rows, args.seed, workdir)
            model, _ = bench_training(matrix, labels, args.epochs)

        bench_artifact(model, workdir)
        bench_scoring(model, matrix, args.scoring_rounds)
        bench_storage()
        if not args.skip_http:
            bench_http(args.base_url, min(args.scoring_rounds, 500))

    print("=" * 78)
    print("note: numbers vary with CPU, disk and whether the OS page cache is warm.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
