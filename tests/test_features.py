"""Tests for feature validation and transformation."""

from __future__ import annotations

import math
import unittest

from riskscore.errors import ValidationError
from riskscore.features import (
    FEATURE_NAMES,
    PURPOSE_CODES,
    build_features,
    feature_names,
    normalise_payload,
    raw_contributions,
)


def valid_payload(**overrides):
    payload = {
        "age": 35,
        "annual_income": 240000,
        "employment_years": 6.0,
        "debt_to_income_ratio": 0.32,
        "num_delinquencies_24m": 0,
        "credit_history_months": 120,
        "loan_amount": 150000,
        "loan_term_months": 36,
        "num_open_accounts": 5,
        "revolving_utilization": 0.4,
        "purpose": "working_capital",
    }
    payload.update(overrides)
    return payload


class FeatureNamesTests(unittest.TestCase):
    def test_order_is_stable(self) -> None:
        self.assertEqual(feature_names(), list(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), 18)

    def test_money_fields_are_log_compressed(self) -> None:
        self.assertIn("log_annual_income", FEATURE_NAMES)
        self.assertIn("log_loan_amount", FEATURE_NAMES)
        self.assertNotIn("annual_income", FEATURE_NAMES)

    def test_one_indicator_per_purpose(self) -> None:
        indicators = [name for name in FEATURE_NAMES if name.startswith("purpose_")]
        self.assertEqual(len(indicators), len(PURPOSE_CODES))
        self.assertEqual(sorted(indicators), sorted(f"purpose_{c}" for c in PURPOSE_CODES))

    def test_calling_twice_returns_equal_lists(self) -> None:
        self.assertEqual(feature_names(), feature_names())


class NormalisePayloadTests(unittest.TestCase):
    def test_accepts_a_valid_payload(self) -> None:
        normalised = normalise_payload(valid_payload())
        self.assertEqual(normalised["purpose"], "working_capital")
        self.assertAlmostEqual(normalised["annual_income"], 240000.0)

    def test_applies_aliases(self) -> None:
        payload = valid_payload()
        payload["income"] = payload.pop("annual_income")
        payload["dti"] = payload.pop("debt_to_income_ratio")
        normalised = normalise_payload(payload)
        self.assertEqual(normalised["annual_income"], 240000.0)
        self.assertEqual(normalised["debt_to_income_ratio"], 0.32)
        self.assertEqual(
            normalised["_aliased_fields"],
            {"income": "annual_income", "dti": "debt_to_income_ratio"},
        )

    def test_requires_every_field(self) -> None:
        payload = valid_payload()
        del payload["age"]
        del payload["purpose"]
        with self.assertRaises(ValidationError) as context:
            normalise_payload(payload)
        problems = context.exception.details["problems"]
        fields = {problem["field"] for problem in problems}
        self.assertEqual(fields, {"age", "purpose"})
        self.assertEqual(context.exception.status, 400)

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError) as context:
            normalise_payload(valid_payload(favourite_colour="blue"))
        problems = context.exception.details["problems"]
        self.assertTrue(any(p["field"] == "favourite_colour" for p in problems))

    def test_reports_range_violations_together(self) -> None:
        with self.assertRaises(ValidationError) as context:
            normalise_payload(valid_payload(age=200, annual_income=-1, loan_term_months=0))
        problems = context.exception.details["problems"]
        self.assertEqual(len(problems), 3)

    def test_rejects_booleans_as_numbers(self) -> None:
        with self.assertRaises(ValidationError) as context:
            normalise_payload(valid_payload(age=True))
        self.assertIn("boolean", context.exception.message)

    def test_rejects_non_finite_numbers(self) -> None:
        with self.assertRaises(ValidationError):
            normalise_payload(valid_payload(age=float("nan")))
        with self.assertRaises(ValidationError):
            normalise_payload(valid_payload(annual_income=float("inf")))

    def test_accepts_numeric_strings(self) -> None:
        normalised = normalise_payload(valid_payload(age="35"))
        self.assertEqual(normalised["age"], 35.0)

    def test_rejects_junk_strings(self) -> None:
        with self.assertRaises(ValidationError):
            normalise_payload(valid_payload(age="thirty-five"))

    def test_normalises_purpose_spelling(self) -> None:
        normalised = normalise_payload(valid_payload(purpose="  Working-Capital "))
        self.assertEqual(normalised["purpose"], "working_capital")

    def test_rejects_unknown_purpose(self) -> None:
        with self.assertRaises(ValidationError) as context:
            normalise_payload(valid_payload(purpose="gambling"))
        self.assertIn("purpose", context.exception.details["problems"][0]["field"])

    def test_rejects_non_mapping(self) -> None:
        with self.assertRaises(ValidationError):
            normalise_payload(["not", "an", "object"])  # type: ignore[arg-type]


class BuildFeaturesTests(unittest.TestCase):
    def test_length_matches_feature_names(self) -> None:
        vector = build_features(normalise_payload(valid_payload()))
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertEqual(vector.names, FEATURE_NAMES)

    def test_log_compression_is_applied(self) -> None:
        vector = build_features(normalise_payload(valid_payload(annual_income=240000)))
        as_dict = vector.as_dict()
        self.assertAlmostEqual(as_dict["log_annual_income"], math.log1p(240000), places=9)

    def test_only_one_purpose_indicator_is_set(self) -> None:
        for purpose in PURPOSE_CODES:
            vector = build_features(normalise_payload(valid_payload(purpose=purpose)))
            as_dict = vector.as_dict()
            active = [n for n in FEATURE_NAMES if n.startswith("purpose_") and as_dict[n] == 1.0]
            self.assertEqual(active, [f"purpose_{purpose}"])

    def test_zero_values_do_not_break_log_compression(self) -> None:
        vector = build_features(normalise_payload(valid_payload(annual_income=0, loan_amount=0)))
        as_dict = vector.as_dict()
        self.assertEqual(as_dict["log_annual_income"], 0.0)
        self.assertEqual(as_dict["log_loan_amount"], 0.0)

    def test_ratios_are_clipped_above_the_schema_maximum(self) -> None:
        vector = build_features(
            normalise_payload(valid_payload(debt_to_income_ratio=1.5, revolving_utilization=1.5))
        )
        as_dict = vector.as_dict()
        self.assertEqual(as_dict["debt_to_income_ratio"], 1.5)
        self.assertEqual(as_dict["revolving_utilization"], 1.5)

    def test_feature_vector_is_deterministic(self) -> None:
        first = build_features(normalise_payload(valid_payload())).values
        second = build_features(normalise_payload(valid_payload())).values
        self.assertEqual(first, second)


class RawContributionTests(unittest.TestCase):
    def test_returns_weight_times_value(self) -> None:
        names = ["a", "b"]
        pairs = raw_contributions(names, [2.0, 3.0], [0.5, -1.0])
        self.assertEqual(pairs, [("a", 1.0), ("b", -3.0)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
