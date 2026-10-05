"""Ad-hoc diagnostic used while developing the test suite.

Not part of the deliverable: it exists so score/band behaviour can be inspected
directly without running the whole suite. Run with ``python scripts/diagnose.py``.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riskscore.scorecard import (  # noqa: E402
    BASE_SCORE,
    POINTS_PER_DOUBLING,
    RISK_BANDS,
    band_for_score,
    probability_from_score,
    score_from_probability,
)


def main() -> None:
    print(f"BASE_SCORE={BASE_SCORE} POINTS_PER_DOUBLING={POINTS_PER_DOUBLING}\n")

    print("bands (best to worst):")
    for band in RISK_BANDS:
        lower = "-inf" if math.isinf(band.lower) else f"{band.lower:.4f}"
        upper = "+inf" if math.isinf(band.upper) else f"{band.upper:.4f}"
        print(
            f"  {band.name}  ({lower}, {upper}]  {band.label:<16} "
            f"{band.expected_default_rate}"
        )

    print("\nscore to band mapping:")
    previous = None
    for score in (1000, 900, 842, 841.5, 841.4, 800, 742, 741.5, 741.1, 700, 643, 642.1, 642.0,
                  600, 572, 571.5, 571.1, 571.0, 550, 300):
        name = band_for_score(score).name
        implied = probability_from_score(score)
        flag = "" if previous is None or previous != name else ""
        print(f"  score {score:>7} -> band {name}  implied PD {implied:.4f}{flag}")
        previous = name

    print("\nlower edges and the value 1 point above:")
    for index, band in enumerate(RISK_BANDS):
        if math.isinf(band.lower):
            print(f"  {band.name}: lower is -inf, skipped")
            continue
        above = band.lower + 1.0
        better = RISK_BANDS[index - 1].name if index > 0 else "n/a"
        print(
            f"  {band.name}: lower={band.lower:.4f} -> band {band_for_score(band.lower).name} | "
            f"lower+1={above:.4f} -> band {band_for_score(above).name} (expected {better})"
        )

    print("\nreference points:")
    for probability in (0.01, 0.015, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.60):
        print(f"  PD {probability:<6} -> score {score_from_probability(probability):8.3f}")


if __name__ == "__main__":
    main()
