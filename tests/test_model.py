"""Tests for the scorecard maths: training, metrics, explanation and persistence."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest

from riskscore.model import (
    ScorecardModel,
    Standardizer,
    TrainingConfig,
    evaluate,
    ks_statistic,
    logit,
    roc_auc,
    sigmoid,
    train_scorecard,
    train_test_split,
)


def make_dataset(count: int = 400, seed: int = 7):
    """A linearly separable-enough toy dataset in the model's raw feature space."""
    rng = random.Random(seed)
    feature_names = ("f0", "f1", "f2")
    rows = []
    labels = []
    for _ in range(count):
        row = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
        signal = -1.6 * row[0] + 2.3 * row[1] - 0.4 * row[2]
        probability = sigmoid(signal)
        rows.append(row)
        labels.append(1 if rng.random() < probability else 0)
    return feature_names, rows, labels


class NumericHelperTests(unittest.TestCase):
    def test_sigmoid_bounds(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertLess(sigmoid(-1000.0), 1e-12)
        self.assertGreater(sigmoid(1000.0), 1 - 1e-12)

    def test_sigmoid_is_stable_for_large_negative_inputs(self) -> None:
        self.assertAlmostEqual(sigmoid(-800.0), 0.0, places=12)
        self.assertTrue(math.isfinite(sigmoid(-800.0)))

    def test_logit_inverts_sigmoid(self) -> None:
        for value in (-3.0, -0.5, 0.0, 1.25, 4.0):
            self.assertAlmostEqual(logit(sigmoid(value)), value, places=9)

    def test_logit_clamps_extremes(self) -> None:
        self.assertTrue(math.isfinite(logit(0.0)))
        self.assertTrue(math.isfinite(logit(1.0)))


class StandardizerTests(unittest.TestCase):
    def test_standardises_to_zero_mean_unit_variance(self) -> None:
        rows = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
        standardizer = Standardizer.fit(rows)
        transformed = standardizer.transform(rows)
        for column in range(2):
            values = [row[column] for row in transformed]
            self.assertAlmostEqual(sum(values) / len(values), 0.0, places=9)

    def test_constant_column_gets_unit_scale(self) -> None:
        rows = [(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)]
        standardizer = Standardizer.fit(rows)
        self.assertEqual(standardizer.scales[0], 1.0)
        self.assertEqual(standardizer.transform_row(rows[0])[0], 0.0)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            Standardizer.fit([])

    def test_rejects_ragged_input(self) -> None:
        with self.assertRaises(ValueError):
            Standardizer.fit([(1.0, 2.0), (3.0,)])

    def test_round_trips_through_dict(self) -> None:
        standardizer = Standardizer.fit([(1.0,), (2.0,), (4.0,)])
        restored = Standardizer.from_dict(standardizer.to_dict())
        self.assertEqual(restored, standardizer)

    def test_rejects_zero_scale_on_load(self) -> None:
        with self.assertRaises(ValueError):
            Standardizer.from_dict({"means": [0.0], "scales": [0.0]})

    def test_rejects_mismatched_lengths_on_load(self) -> None:
        with self.assertRaises(ValueError):
            Standardizer.from_dict({"means": [0.0, 1.0], "scales": [1.0]})


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_for_a_seed(self) -> None:
        rows = [(float(i),) for i in range(50)]
        labels = [i % 2 for i in range(50)]
        first = train_test_split(rows, labels, seed=99)
        second = train_test_split(rows, labels, seed=99)
        self.assertEqual(first, second)

    def test_split_size_is_respected(self) -> None:
        rows = [(float(i),) for i in range(100)]
        labels = [i % 2 for i in range(100)]
        train_rows, _, test_rows, _ = train_test_split(rows, labels, test_fraction=0.25)
        self.assertEqual(len(train_rows), 75)
        self.assertEqual(len(test_rows), 25)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            train_test_split([(1.0,)], [0, 1])

    def test_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            train_test_split([(1.0,), (2.0,)], [0, 1], test_fraction=0.0)


class MetricTests(unittest.TestCase):
    def test_auc_is_one_for_perfect_separation(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)

    def test_auc_is_zero_for_inverted_scores(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc(labels, scores), 0.0)

    def test_auc_handles_ties_with_average_ranks(self) -> None:
        # One positive and one negative share a score: AUC must be 0.5, not 1.
        self.assertAlmostEqual(roc_auc([1, 0], [0.5, 0.5]), 0.5)

    def test_auc_is_chance_for_a_single_class(self) -> None:
        self.assertAlmostEqual(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]), 0.5)

    def test_auc_is_symmetric_under_score_rescaling(self) -> None:
        labels = [1, 0, 1, 0, 1]
        scores = [0.9, 0.1, 0.4, 0.6, 0.75]
        rescaled = [10.0 * s for s in scores]
        self.assertAlmostEqual(roc_auc(labels, scores), roc_auc(labels, rescaled))

    def test_ks_is_one_for_perfect_separation(self) -> None:
        self.assertAlmostEqual(ks_statistic([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)

    def test_ks_is_zero_when_distributions_match(self) -> None:
        self.assertAlmostEqual(ks_statistic([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]), 0.0)

    def test_ks_is_zero_for_a_single_class(self) -> None:
        self.assertEqual(ks_statistic([1, 1], [0.3, 0.7]), 0.0)

    def test_evaluate_confusion_matrix(self) -> None:
        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.8, 0.2, 0.1]
        metrics = evaluate(labels, probabilities, threshold=0.5)
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)

    def test_evaluate_threshold_is_inclusive(self) -> None:
        metrics = evaluate([1], [0.35], threshold=0.35)
        self.assertEqual(metrics.true_positive, 1)

    def test_evaluate_handles_zero_division(self) -> None:
        metrics = evaluate([0, 0], [0.1, 0.2], threshold=0.5)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)

    def test_metrics_serialise_to_json(self) -> None:
        metrics = evaluate([1, 0], [0.9, 0.1], threshold=0.5)
        self.assertEqual(json.loads(json.dumps(metrics.as_dict()))["auc"], 1.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc([1, 0], [0.5])
        with self.assertRaises(ValueError):
            evaluate([1, 0], [0.5])


class TrainingTests(unittest.TestCase):
    def test_training_learns_the_signal(self) -> None:
        feature_names, rows, labels = make_dataset()
        model, report = train_scorecard(
            rows,
            labels,
            feature_names=feature_names,
            config=TrainingConfig(epochs=250, learning_rate=0.5, verbose=False),
        )
        self.assertGreater(report["test"]["auc"], 0.7)
        # f1 drives default upward, f0 drives it downward: signs must be learned.
        weights = dict(zip(model.feature_names, model.weights))
        self.assertGreater(weights["f1"], 0)
        self.assertLess(weights["f0"], 0)

    def test_training_is_deterministic(self) -> None:
        feature_names, rows, labels = make_dataset()
        config = TrainingConfig(epochs=60)
        first, _ = train_scorecard(rows, labels, feature_names=feature_names, config=config)
        second, _ = train_scorecard(rows, labels, feature_names=feature_names, config=config)
        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.intercept, second.intercept)

    def test_validation_set_is_used_for_early_stopping(self) -> None:
        feature_names, rows, labels = make_dataset(count=500)
        model, report = train_scorecard(
            rows[:300],
            labels[:300],
            feature_names=feature_names,
            config=TrainingConfig(epochs=80),
            validation=(rows[300:], labels[300:]),
        )
        self.assertEqual(report["dataset"]["model_selection"], "supplied validation set")
        self.assertEqual(report["dataset"]["test_rows"], 200)
        self.assertEqual(len(model.weights), 3)

    def test_rejects_single_class_training_data(self) -> None:
        with self.assertRaises(ValueError):
            train_scorecard([(1.0,), (2.0,)], [1, 1], feature_names=("f0",))

    def test_rejects_empty_training_data(self) -> None:
        with self.assertRaises(ValueError):
            train_scorecard([], [], feature_names=("f0",))

    def test_rejects_feature_width_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            train_scorecard([(1.0,)], [1], feature_names=("f0", "f1"))

    def test_rejects_mismatched_labels(self) -> None:
        with self.assertRaises(ValueError):
            train_scorecard([(1.0,), (2.0,)], [1], feature_names=("f0",))

    def test_rejects_single_class_validation_set(self) -> None:
        feature_names, rows, labels = make_dataset()
        with self.assertRaises(ValueError):
            train_scorecard(
                rows,
                labels,
                feature_names=feature_names,
                validation=(rows[:5], [1, 1, 1, 1, 1]),
            )


class ModelBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        feature_names, rows, labels = make_dataset()
        self.model, self.report = train_scorecard(
            rows,
            labels,
            feature_names=feature_names,
            config=TrainingConfig(epochs=200, learning_rate=0.5),
            trained_at="2026-01-01T00:00:00Z",
        )

    def test_predict_proba_is_in_range(self) -> None:
        probabilities = self.model.predict_proba([(0.0, 0.0, 0.0), (5.0, -5.0, 2.0)])
        self.assertEqual(len(probabilities), 2)
        for probability in probabilities:
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_contributions_sum_to_the_decision_function(self) -> None:
        row = (0.7, -1.2, 0.3)
        contributions = sum(value for _, value in self.model.contributions(row))
        self.assertAlmostEqual(
            self.model.intercept + contributions,
            self.model.decision_function(row),
            places=9,
        )

    def test_predict_uses_the_decision_threshold(self) -> None:
        row = (0.0, 0.0, 0.0)
        probability = self.model.predict_proba_row(row)
        model = ScorecardModel(
            feature_names=self.model.feature_names,
            weights=self.model.weights,
            intercept=self.model.intercept,
            standardizer=self.model.standardizer,
            decision_threshold=probability + 0.01,
        )
        self.assertEqual(model.predict([row]), [0])
        model = ScorecardModel(
            feature_names=model.feature_names,
            weights=model.weights,
            intercept=model.intercept,
            standardizer=model.standardizer,
            decision_threshold=probability - 0.01,
        )
        self.assertEqual(model.predict([row]), [1])

    def test_assert_compatible_rejects_wrong_feature_order(self) -> None:
        with self.assertRaises(ValueError):
            self.model.assert_compatible(("f1", "f0", "f2"))

    def test_assert_compatible_accepts_the_training_order(self) -> None:
        self.model.assert_compatible(self.model.feature_names)


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        feature_names, rows, labels = make_dataset()
        self.model, _ = train_scorecard(
            rows,
            labels,
            feature_names=feature_names,
            config=TrainingConfig(epochs=50),
            trained_at="2026-01-01T00:00:00Z",
        )
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "nested", "model.json")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_save_creates_missing_directories(self) -> None:
        self.model.save(self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_round_trip_preserves_predictions(self) -> None:
        self.model.save(self.path)
        restored = ScorecardModel.load(self.path)
        rows = [(0.5, -0.5, 1.0), (-2.0, 3.0, 0.0)]
        self.assertEqual(self.model.predict_proba(rows), restored.predict_proba(rows))

    def test_round_trip_preserves_metadata(self) -> None:
        self.model.save(self.path)
        restored = ScorecardModel.load(self.path)
        self.assertEqual(restored.trained_at, "2026-01-01T00:00:00Z")
        self.assertEqual(restored.feature_names, self.model.feature_names)
        self.assertEqual(restored.decision_threshold, self.model.decision_threshold)

    def test_artifact_is_valid_json_with_the_expected_keys(self) -> None:
        self.model.save(self.path)
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in (
            "format_version",
            "kind",
            "feature_names",
            "weights",
            "intercept",
            "standardizer",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["kind"], "logistic_scorecard")

    def test_rejects_unknown_format_version(self) -> None:
        payload = self.model.to_dict()
        payload["format_version"] = 999
        with self.assertRaises(ValueError):
            ScorecardModel.from_dict(payload)

    def test_rejects_mismatched_weights(self) -> None:
        payload = self.model.to_dict()
        payload["weights"] = payload["weights"][:-1]
        with self.assertRaises(ValueError):
            ScorecardModel.from_dict(payload)

    def test_rejects_standardizer_width_mismatch(self) -> None:
        payload = self.model.to_dict()
        payload["standardizer"]["means"] = payload["standardizer"]["means"][:-1]
        payload["standardizer"]["scales"] = payload["standardizer"]["scales"][:-1]
        with self.assertRaises(ValueError):
            ScorecardModel.from_dict(payload)

    def test_load_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ScorecardModel.load(os.path.join(self.directory.name, "absent.json"))

    def test_save_does_not_leave_temporary_files(self) -> None:
        self.model.save(self.path)
        leftovers = [name for name in os.listdir(os.path.dirname(self.path)) if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
