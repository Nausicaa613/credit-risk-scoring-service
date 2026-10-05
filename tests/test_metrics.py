"""Tests for the in-process metrics registry."""

from __future__ import annotations

import json
import threading
import unittest

from riskscore.metrics import LATENCY_BUCKETS_MS, MAX_SCORE_OBSERVATIONS, MetricsRegistry


class CounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MetricsRegistry()

    def test_increment_accumulates(self) -> None:
        self.metrics.increment("custom_total")
        self.metrics.increment("custom_total", 4)
        self.assertEqual(self.metrics.snapshot()["counters"]["custom_total"], 5)

    def test_snapshot_is_empty_but_complete_before_any_traffic(self) -> None:
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["counters"], {})
        self.assertEqual(snapshot["latency"]["count"], 0)
        self.assertEqual(snapshot["latency"]["mean_ms"], 0.0)
        self.assertIsNone(snapshot["credit_score_summary"])

    def test_snapshot_is_json_serialisable(self) -> None:
        self.metrics.observe_request(path="/v1/score", status=201, duration_ms=3.2, bytes_out=120)
        json.dumps(self.metrics.snapshot())

    def test_uptime_is_non_negative(self) -> None:
        self.assertGreaterEqual(self.metrics.uptime_seconds, 0.0)


class RequestObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MetricsRegistry()

    def test_request_counters_and_status_breakdown(self) -> None:
        self.metrics.observe_request(path="/v1/score", status=201, duration_ms=2.0)
        self.metrics.observe_request(path="/v1/score", status=400, duration_ms=1.0)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["counters"]["http_requests_total"], 2)
        self.assertEqual(snapshot["http_status_counts"], {"201": 1, "400": 1})
        self.assertEqual(snapshot["http_path_counts"], {"/v1/score": 2})

    def test_bytes_out_are_accumulated(self) -> None:
        self.metrics.observe_request(path="/", status=200, duration_ms=1.0, bytes_out=50)
        self.metrics.observe_request(path="/", status=200, duration_ms=1.0, bytes_out=70)
        self.assertEqual(self.metrics.snapshot()["counters"]["http_bytes_out_total"], 120)

    def test_latency_summary(self) -> None:
        for duration in (1.0, 3.0, 5.0):
            self.metrics.observe_request(path="/", status=200, duration_ms=duration)
        latency = self.metrics.snapshot()["latency"]
        self.assertEqual(latency["count"], 3)
        self.assertAlmostEqual(latency["mean_ms"], 3.0)
        self.assertAlmostEqual(latency["sum_ms"], 9.0)

    def test_histogram_buckets_are_cumulative(self) -> None:
        for duration in (0.5, 2.0, 7.0, 400.0):
            self.metrics.observe_request(path="/", status=200, duration_ms=duration)
        buckets = self.metrics.snapshot()["latency"]["buckets_cumulative"]
        counts = [bucket["count"] for bucket in buckets]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 4)
        first_bucket = next(b for b in buckets if b["le_ms"] == 1.0)
        self.assertEqual(first_bucket["count"], 1)

    def test_every_configured_bucket_is_reported(self) -> None:
        snapshot = self.metrics.snapshot()
        reported = [bucket["le_ms"] for bucket in snapshot["latency"]["buckets_cumulative"]]
        self.assertEqual(reported, list(LATENCY_BUCKETS_MS))

    def test_very_slow_request_lands_in_no_bucket_before_the_last(self) -> None:
        self.metrics.observe_request(path="/", status=200, duration_ms=99_999.0)
        buckets = self.metrics.snapshot()["latency"]["buckets_cumulative"]
        self.assertEqual(buckets[-1]["count"], 0)


class ScoreObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MetricsRegistry()

    def test_score_counters_and_bands(self) -> None:
        self.metrics.observe_score(
            credit_score=700, probability_of_default=0.05, band="A", decision="approve"
        )
        self.metrics.observe_score(
            credit_score=430, probability_of_default=0.8, band="E", decision="decline"
        )
        counters = self.metrics.snapshot()["counters"]
        self.assertEqual(counters["scores_total"], 2)
        self.assertEqual(counters["approvals_total"], 1)
        self.assertEqual(counters["declines_total"], 1)
        self.assertEqual(counters["band_A_total"], 1)
        self.assertEqual(counters["band_E_total"], 1)
        self.assertEqual(counters["pd_low_bucket_total"], 1)
        self.assertEqual(counters["pd_high_bucket_total"], 1)

    def test_score_summary_percentiles(self) -> None:
        for score in range(400, 500):
            self.metrics.observe_score(
                credit_score=score, probability_of_default=0.2, band="C", decision="approve"
            )
        summary = self.metrics.snapshot()["credit_score_summary"]
        self.assertEqual(summary["observations"], 100)
        self.assertEqual(summary["min"], 400.0)
        self.assertEqual(summary["max"], 499.0)
        self.assertLessEqual(summary["p50"], summary["p90"])
        self.assertLessEqual(summary["p90"], summary["p99"])

    def test_score_buffer_is_bounded(self) -> None:
        for index in range(MAX_SCORE_OBSERVATIONS + 250):
            self.metrics.observe_score(
                credit_score=600, probability_of_default=0.2, band="C", decision="approve"
            )
        summary = self.metrics.snapshot()["credit_score_summary"]
        self.assertEqual(summary["observations"], MAX_SCORE_OBSERVATIONS)
        # The lifetime counter keeps counting even though the buffer is capped.
        self.assertEqual(
            self.metrics.snapshot()["counters"]["scores_total"], MAX_SCORE_OBSERVATIONS + 250
        )


class LifecycleTests(unittest.TestCase):
    def test_reset_clears_everything(self) -> None:
        metrics = MetricsRegistry()
        metrics.observe_request(path="/v1/score", status=201, duration_ms=5.0, bytes_out=10)
        metrics.observe_score(
            credit_score=700, probability_of_default=0.05, band="A", decision="approve"
        )
        metrics.reset()
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"], {})
        self.assertEqual(snapshot["latency"]["count"], 0)
        self.assertIsNone(snapshot["credit_score_summary"])

    def test_concurrent_updates_are_not_lost(self) -> None:
        metrics = MetricsRegistry()
        threads = []
        per_thread = 250

        def worker() -> None:
            for _ in range(per_thread):
                metrics.increment("concurrent_total")
                metrics.observe_request(path="/v1/score", status=201, duration_ms=1.0)

        for _ in range(4):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["concurrent_total"], 4 * per_thread)
        self.assertEqual(snapshot["counters"]["http_requests_total"], 4 * per_thread)
        self.assertEqual(snapshot["latency"]["count"], 4 * per_thread)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
