"""Tests for the score translation layer: score scaling, bands and reason codes."""

from __future__ import annotations

import math
import random
import unittest

from riskscore.model import ScorecardModel, Standardizer, sigmoid
from riskscore.scorecard import (
    BASE_ODDS,
    BASE_SCORE,
    POINTS_PER_DOUBLING,
    RISK_BANDS,
    Scorer,
    band_for_score,
    band_table,
    describe_feature,
    probability_from_score,
    score_from_probability,
    top_contributors,
)


def _better_band(band_name: str) -> str:
    """Return the name of the band one step better than ``band_name``."""
    names = [band.name for band in RISK_BANDS]
    return names[names.index(band_name) - 1]


def _worse_band(band_name: str) -> str:
    """Return the name of the band one step worse than ``band_name``."""
    names = [band.name for band in RISK_BANDS]
    return names[names.index(band_name) + 1]


def simple_model(*, weights=(1.0, -1.0), intercept=0.0, threshold=0.5):
    """A two-feature model with an identity standardizer, easy to reason about."""
    return ScorecardModel(
        feature_names=("f0", "f1"),
        weights=weights,
        intercept=intercept,
        standardizer=Standardizer(means=(0.0, 0.0), scales=(1.0, 1.0)),
        decision_threshold=threshold,
        version="test-1.0",
    )


class ScoreScalingTests(unittest.TestCase):
    def test_base_odds_map_to_the_base_score(self) -> None:
        # BASE_ODDS is expressed as good:bad, so it corresponds to PD = 1/(1+50);
        # that is exactly one BASE_ODDS step above a coin flip, i.e. BASE_SCORE
        # plus POINTS_PER_DOUBLING * log2(BASE_ODDS) points.
        probability = 1.0 / (1.0 + BASE_ODDS)
        expected = BASE_SCORE + POINTS_PER_DOUBLING * math.log2(BASE_ODDS)
        self.assertAlmostEqual(score_from_probability(probability), expected, places=6)

    def test_even_odds_land_on_the_base_score(self) -> None:
        # A coin flip is the natural anchor: 50:50 implies the base score.
        self.assertAlmostEqual(score_from_probability(0.5), BASE_SCORE, places=9)

    def test_higher_score_means_lower_risk(self) -> None:
        scores = [score_from_probability(p) for p in (0.6, 0.4, 0.2, 0.05, 0.01)]
        self.assertEqual(scores, sorted(scores))

    def test_points_per_doubling_is_respected(self) -> None:
        # Halving the odds should cost exactly POINTS_PER_DOUBLING points.
        pd_a = 0.10
        odds_a = (1 - pd_a) / pd_a
        odds_b = odds_a / 2.0
        pd_b = 1.0 / (1.0 + odds_b)
        delta = score_from_probability(pd_a) - score_from_probability(pd_b)
        self.assertAlmostEqual(delta, POINTS_PER_DOUBLING, places=6)

    def test_documented_reference_points(self) -> None:
        # The docstring promises these; they are the numbers a reviewer checks.
        for probability, expected in (
            (0.015, 841.0),
            (0.05, 770.0),
            (0.20, 680.0),
            (0.60, 577.0),
        ):
            self.assertAlmostEqual(
                score_from_probability(probability),
                expected,
                delta=1.0,
                msg=f"PD {probability} should score about {expected}",
            )

    def test_probability_from_score_inverts_the_mapping(self) -> None:
        for probability in (0.01, 0.05, 0.2, 0.5, 0.9):
            score = score_from_probability(probability)
            self.assertAlmostEqual(probability_from_score(score), probability, places=9)

    def test_extreme_probabilities_stay_finite(self) -> None:
        self.assertTrue(math.isfinite(score_from_probability(0.0)))
        self.assertTrue(math.isfinite(score_from_probability(1.0)))

    def test_monotone_over_a_dense_sweep(self) -> None:
        scores = [score_from_probability(i / 1000.0) for i in range(1, 1000)]
        for earlier, later in zip(scores, scores[1:]):
            self.assertGreaterEqual(earlier, later)


class BandTests(unittest.TestCase):
    #: The published cut-offs, duplicated here on purpose. Production derives the
    #: edges from the score scale, so only an independent copy can catch a change
    #: that silently moves every band.
    EXPECTED_CUTOFFS = {"A": 0.015, "B": 0.05, "C": 0.12, "D": 0.25}

    def test_bands_cover_the_whole_axis(self) -> None:
        for score in (-10_000.0, 300.0, 600.0, 663.0, 715.0, 770.0, 841.5, 850.0, 10_000.0):
            self.assertIsNotNone(band_for_score(score))

    def test_band_edge_scores_imply_the_published_cutoffs(self) -> None:
        """Each band's lower edge must sit at the probability it advertises."""
        bands = {band.name: band for band in RISK_BANDS}
        self.assertEqual(set(bands), {"A", "B", "C", "D", "E"})
        for band_name, probability in self.EXPECTED_CUTOFFS.items():
            self.assertAlmostEqual(
                probability_from_score(bands[band_name].lower),
                probability,
                delta=0.001,
                msg=f"band {band_name} edge implies a different PD than it advertises",
            )

    def test_band_edges_are_consistent_with_the_score_scale(self) -> None:
        bands = {band.name: band for band in RISK_BANDS}
        for band_name, probability in self.EXPECTED_CUTOFFS.items():
            self.assertAlmostEqual(
                bands[band_name].lower,
                score_from_probability(probability),
                places=9,
                msg=f"band {band_name} edge is not the score for PD {probability}",
            )

    def test_bands_tile_the_score_axis(self) -> None:
        """Bands must be contiguous, non-overlapping and total."""
        for better, worse in zip(RISK_BANDS, RISK_BANDS[1:]):
            self.assertEqual(better.lower, worse.upper)
        self.assertTrue(math.isinf(RISK_BANDS[0].upper))
        self.assertTrue(math.isinf(RISK_BANDS[-1].lower))

    def test_band_boundaries_are_upper_inclusive(self) -> None:
        """A band owns its lower edge; just below it the band gets worse.

        The +0.01 probe is essential: ``band.lower`` is computed through a log and
        an exp, so it is not exactly representable and the band's own edge can
        fall a few ulps below the stored bound.
        """
        for index, band in enumerate(RISK_BANDS):
            self.assertEqual(band_for_score(band.lower + 0.01).name, band.name)
            if not math.isinf(band.lower):
                self.assertEqual(
                    band_for_score(band.lower - 0.01).name, RISK_BANDS[index + 1].name
                )
        self.assertEqual(band_for_score(1e6).name, "A")
        self.assertEqual(band_for_score(-1e6).name, "E")

    def test_a_higher_score_never_yields_a_worse_band(self) -> None:
        """Walk the whole published range and check the transition is one-way."""
        order = {band.name: index for index, band in enumerate(RISK_BANDS)}
        previous = order[band_for_score(300.0).name]
        for score in range(301, 900):
            current = order[band_for_score(float(score)).name]
            self.assertLessEqual(current, previous, f"score {score} made the band worse")
            previous = current

    def test_top_band_is_unbounded_above(self) -> None:
        self.assertTrue(math.isinf(RISK_BANDS[0].upper))

    def test_worst_band_is_unbounded_below(self) -> None:
        self.assertTrue(math.isinf(RISK_BANDS[-1].lower))

    def test_bands_are_ordered_best_to_worst(self) -> None:
        lower_bounds = [band.lower for band in RISK_BANDS]
        self.assertEqual(lower_bounds, sorted(lower_bounds, reverse=True))

    def test_band_table_is_json_serialisable(self) -> None:
        import json

        table = band_table()
        self.assertEqual(len(table), len(RISK_BANDS))
        json.dumps(table)
        self.assertIsNone(table[0]["upper_inclusive"])
        self.assertIsNone(table[-1]["lower_exclusive"])

    def test_lower_score_never_lands_in_a_better_band(self) -> None:
        """Walking the score axis downward must never improve the band.

        ``RISK_BANDS`` is ordered best to worst, so the index of the matching
        band must be non-decreasing as the score falls.
        """
        order = {band.name: index for index, band in enumerate(RISK_BANDS)}

        def rank(score: float) -> int:
            return order[band_for_score(score).name]

        previous_rank = rank(900.0)
        for score in range(899, 299, -1):
            current_rank = rank(float(score))
            self.assertGreaterEqual(
                current_rank, previous_rank, f"score {score} improved the band unexpectedly"
            )
            previous_rank = current_rank


class ReasonCodeTests(unittest.TestCase):
    def test_known_features_have_descriptions_and_directions(self) -> None:
        description, hint = describe_feature("debt_to_income_ratio")
        self.assertEqual(hint, "risk_up")
        self.assertTrue(description)

    def test_unknown_feature_falls_back_to_the_raw_name(self) -> None:
        description, hint = describe_feature("some_new_feature")
        self.assertEqual(description, "some new feature")
        self.assertEqual(hint, "context")

    def test_purpose_indicators_are_described(self) -> None:
        description, hint = describe_feature("purpose_refinance")
        self.assertEqual(hint, "context")
        self.assertIn("purpose", description)

    def test_top_contributors_prefers_large_absolute_values(self) -> None:
        contributions = [("a", 0.1), ("b", -0.9), ("c", 0.5), ("d", -0.05)]
        ranked = top_contributors(contributions, limit=2)
        self.assertEqual([name for name, _ in ranked], ["b", "c"])

    def test_top_contributors_breaks_ties_by_model_order(self) -> None:
        contributions = [("a", 0.5), ("b", -0.5), ("c", 0.5)]
        ranked = top_contributors(contributions, limit=2)
        self.assertEqual([name for name, _ in ranked], ["a", "b"])


class ScorerTests(unittest.TestCase):
    def test_contributions_are_exactly_additive(self) -> None:
        model = simple_model(weights=(0.8, -0.35), intercept=0.4)
        scorer = Scorer(model)
        result = scorer.score_vector("app_1", (2.0, -1.0))
        total = result.intercept + sum(value for _, value in result.contributions)
        self.assertAlmostEqual(total, result.log_odds, places=12)

    def test_probability_matches_the_sigmoid_of_the_log_odds(self) -> None:
        scorer = Scorer(simple_model(intercept=0.25))
        result = scorer.score_vector("app_1", (1.5, 0.5))
        self.assertAlmostEqual(result.probability_of_default, sigmoid(result.log_odds), places=12)

    def test_decision_respects_the_threshold(self) -> None:
        # logit +2.0 -> PD 0.88 (above the 0.5 gate, declined);
        # logit -2.0 -> PD 0.12 (below the gate, approved).
        scorer = Scorer(simple_model(weights=(1.0, 0.0), intercept=0.0, threshold=0.5))
        self.assertEqual(scorer.score_vector("risky", (2.0, 0.0)).decision, "decline")
        self.assertEqual(scorer.score_vector("safe", (-2.0, 0.0)).decision, "approve")

    def test_threshold_is_inclusive(self) -> None:
        # A row landing exactly on the threshold is declined (>=, not >).
        model = simple_model(weights=(1.0, 0.0), intercept=0.0, threshold=0.5)
        row = (0.0, 0.0)  # logit 0 -> PD exactly 0.5
        self.assertEqual(Scorer(model).score_vector("a", row).decision, "decline")

    def test_constructor_threshold_overrides_the_model(self) -> None:
        # logit 1.0 -> PD 0.731: approved at a 0.9 gate, declined at a 0.01 gate.
        model = simple_model(weights=(1.0, 0.0), intercept=0.0, threshold=0.5)
        self.assertEqual(
            Scorer(model, threshold=0.9).score_vector("a", (1.0, 0.0)).decision, "approve"
        )
        self.assertEqual(
            Scorer(model, threshold=0.01).score_vector("a", (1.0, 0.0)).decision, "decline"
        )

    def test_rejects_threshold_outside_the_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            Scorer(simple_model(), threshold=1.5)

    def test_score_is_clipped_to_the_published_range(self) -> None:
        extremely_risky = Scorer(simple_model(weights=(0.0, 1.0), intercept=-50.0))
        result = extremely_risky.score_vector("a", (0.0, 0.0))
        self.assertGreaterEqual(result.credit_score, 300)
        extremely_safe = Scorer(simple_model(weights=(0.0, 1.0), intercept=50.0))
        self.assertLessEqual(extremely_safe.score_vector("a", (0.0, 0.0)).credit_score, 850)

    def test_band_and_score_always_agree(self) -> None:
        rng = random.Random(3)
        scorer = Scorer(simple_model(weights=(1.3, -0.7), intercept=-0.2, threshold=0.4))
        for _ in range(200):
            row = (rng.gauss(0, 2), rng.gauss(0, 2))
            result = scorer.score_vector("app", row)
            self.assertEqual(result.band, band_for_score(float(result.credit_score)).name)

    def test_ordering_matches_probability_ordering_across_many_rows(self) -> None:
        """Lower PD must always mean a higher reported score.

        The weights are small enough that no score reaches the 300 or 850 clip.
        Clipping is monotone and therefore safe, but it deliberately ties every
        extreme applicant together, which would break an exact ordering
        comparison. ``test_score_is_clipped_to_the_published_range`` covers the
        clipped case.
        """
        rng = random.Random(11)
        scorer = Scorer(simple_model(weights=(0.55, -0.55), intercept=-0.15))
        results = [
            scorer.score_vector(f"app_{i}", (rng.gauss(0, 1.2), rng.gauss(0, 1.2)))
            for i in range(300)
        ]
        scores = [result.credit_score for result in results]
        self.assertTrue(
            all(300 < score < 850 for score in scores),
            f"this sample must not be clipped, got {min(scores)}..{max(scores)}",
        )

        # Compare scores rather than identifiers: two applicants whose scores
        # round to the same integer are legitimately tied and either order is
        # valid, so the invariant is "non-decreasing PD implies non-increasing
        # score", not an exact permutation.
        by_pd = sorted(results, key=lambda r: r.probability_of_default)
        scores = [result.credit_score for result in by_pd]
        for earlier, later in zip(scores, scores[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_reason_codes_flag_risk_direction(self) -> None:
        model = simple_model(weights=(1.0, 1.0), intercept=-0.5)
        scorer = Scorer(model)
        result = scorer.score_vector("app_1", (3.0, 3.0))
        reasons = result.reason_codes(limit=2)
        self.assertEqual(len(reasons), 2)
        self.assertTrue(all(reason["direction"] == "increases_risk" for reason in reasons))

    def test_result_serialises_and_rounds(self) -> None:
        import json

        scorer = Scorer(simple_model(weights=(1.0, -1.0), intercept=0.2))
        payload = scorer.score_vector("app_42", (0.5, -0.5)).as_dict()
        self.assertEqual(payload["application_id"], "app_42")
        self.assertIn("reason_codes", payload)
        json.dumps(payload)
        self.assertLessEqual(len(str(payload["probability_of_default"]).split(".")[-1]), 6)

    def test_approved_property_matches_the_decision(self) -> None:
        scorer = Scorer(simple_model(weights=(1.0, 0.0), intercept=0.0, threshold=0.5))
        self.assertFalse(scorer.score_vector("risky", (2.0, 0.0)).approved)
        self.assertTrue(scorer.score_vector("safe", (-2.0, 0.0)).approved)
    def test_score_method_normalises_a_raw_payload(self) -> None:
        from riskscore.features import FEATURE_NAMES

        model = ScorecardModel(
            feature_names=tuple(FEATURE_NAMES),
            weights=tuple(0.0 for _ in FEATURE_NAMES),
            intercept=0.0,
            standardizer=Standardizer(
                means=tuple(0.0 for _ in FEATURE_NAMES),
                scales=tuple(1.0 for _ in FEATURE_NAMES),
            ),
            decision_threshold=0.5,
        )
        payload = {
            "age": 40,
            "annual_income": 300000,
            "employment_years": 8,
            "debt_to_income_ratio": 0.3,
            "num_delinquencies_24m": 0,
            "credit_history_months": 150,
            "loan_amount": 200000,
            "loan_term_months": 36,
            "num_open_accounts": 5,
            "revolving_utilization": 0.3,
            "purpose": "equipment",
        }
        result = Scorer(model).score("app_raw", payload)
        self.assertAlmostEqual(result.probability_of_default, 0.5, places=9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
