"""In-process metrics.

Deliberately small: counters, a fixed-bucket latency histogram and an
observation log for scores. The output is JSON so that ``/metrics`` is readable
by a human and parseable by a script, at the cost of not matching the
Prometheus text exposition format. Swapping in a real metrics client is listed
on the roadmap; the interface here (:meth:`MetricsRegistry.snapshot`) is the
only thing that would need to change.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

#: Upper bounds in milliseconds for the latency histogram.
LATENCY_BUCKETS_MS: Tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)

#: How many recent scores to retain for the rolling distribution.
MAX_SCORE_OBSERVATIONS = 1000


@dataclass
class _Bucket:
    upper_ms: float
    count: int = 0


@dataclass
class MetricsRegistry:
    """Thread-safe counters for one process.

    ``http.server`` with ``ThreadingHTTPServer`` means handlers run in parallel,
    so every mutation is guarded by a lock. The critical sections are trivial
    arithmetic, so contention is not a concern.
    """

    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _counters: Counter = field(default_factory=Counter, repr=False)
    _status_counts: Counter = field(default_factory=Counter, repr=False)
    _path_counts: Counter = field(default_factory=Counter, repr=False)
    _buckets: List[_Bucket] = field(
        default_factory=lambda: [_Bucket(upper) for upper in LATENCY_BUCKETS_MS],
        repr=False,
    )
    _latency_sum_ms: float = 0.0
    _latency_count: int = 0
    _scores: Deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_SCORE_OBSERVATIONS), repr=False
    )

    # -- recording ---------------------------------------------------------

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe_request(
        self,
        *,
        path: str,
        status: int,
        duration_ms: float,
        bytes_out: int = 0,
    ) -> None:
        with self._lock:
            self._counters["http_requests_total"] += 1
            self._counters["http_bytes_out_total"] += bytes_out
            self._status_counts[str(status)] += 1
            self._path_counts[path] += 1
            self._latency_sum_ms += duration_ms
            self._latency_count += 1
            for bucket in self._buckets:
                if duration_ms <= bucket.upper_ms:
                    bucket.count += 1

    def observe_score(
        self,
        *,
        credit_score: int,
        probability_of_default: float,
        band: str,
        decision: str,
    ) -> None:
        with self._lock:
            self._counters["scores_total"] += 1
            if decision == "decline":
                self._counters["declines_total"] += 1
            else:
                self._counters["approvals_total"] += 1
            self._scores.append(float(credit_score))
            self._counters[f"band_{band}_total"] += 1
            self._counters[
                "pd_high_bucket_total" if probability_of_default >= 0.35 else "pd_low_bucket_total"
            ] += 1

    # -- reporting ---------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def _score_summary(self) -> Optional[Dict[str, Any]]:
        scores = sorted(self._scores)
        if not scores:
            return None
        count = len(scores)

        def percentile(fraction: float) -> float:
            index = min(int(round(fraction * (count - 1))), count - 1)
            return round(scores[index], 2)

        return {
            "observations": count,
            "min": round(scores[0], 2),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p99": percentile(0.99),
            "max": round(scores[-1], 2),
        }

    def _latency_summary(self) -> Dict[str, Any]:
        with self._lock:
            count = self._latency_count
            total = self._latency_sum_ms
            buckets = [
                {"le_ms": bucket.upper_ms, "count": bucket.count} for bucket in self._buckets
            ]
        return {
            "count": count,
            "sum_ms": round(total, 3),
            "mean_ms": round(total / count, 3) if count else 0.0,
            "buckets_cumulative": buckets,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of everything recorded so far."""
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            statuses = dict(sorted(self._status_counts.items()))
            paths = dict(sorted(self._path_counts.items()))
        return {
            "uptime_seconds": round(self.uptime_seconds, 3),
            "counters": counters,
            "http_status_counts": statuses,
            "http_path_counts": paths,
            "latency": self._latency_summary(),
            "credit_score_summary": self._score_summary(),
        }

    def reset(self) -> None:
        """Clear all state; used by tests."""
        with self._lock:
            self._counters.clear()
            self._status_counts.clear()
            self._path_counts.clear()
            self._latency_sum_ms = 0.0
            self._latency_count = 0
            self._scores.clear()
            for bucket in self._buckets:
                bucket.count = 0


__all__ = ["MetricsRegistry", "LATENCY_BUCKETS_MS", "MAX_SCORE_OBSERVATIONS"]
