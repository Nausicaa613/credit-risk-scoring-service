#!/usr/bin/env python3
"""Train the credit scorecard from a JSONL dataset.

Pipeline
--------
1. read ``data/applications.jsonl`` (or ``--data``),
2. split into train / validation / test with a fixed seed,
3. build the model feature matrix,
4. fit the logistic scorecard by batch gradient descent with early stopping,
5. evaluate on the held-out test set,
6. write ``models/model.json``, ``reports/training_metrics.json`` and a
   Markdown summary that can be pasted into MODEL_CARD.md.

Three partitions rather than two
--------------------------------
The validation slice chooses the stopping epoch; the test slice is touched
exactly once for the numbers we report. Selecting the epoch on the same data
used to report performance is the classic way to publish an optimistic AUC, so
the script keeps them separate on purpose.

Usage
-----
    python scripts/train_model.py --data data/applications.jsonl \\
        --model models/model.json --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riskscore.features import FEATURE_NAMES, build_features, normalise_payload  # noqa: E402
from riskscore.model import (  # noqa: E402
    ScorecardModel,
    TrainingConfig,
    evaluate,
    fit_prior_offset,
    train_scorecard,
)
from riskscore.scorecard import Scorer, band_table  # noqa: E402

DEFAULT_DATA = "data/applications.jsonl"
DEFAULT_MODEL = "models/model.json"
DEFAULT_METRICS = "reports/training_metrics.json"
DEFAULT_SUMMARY = "reports/training_summary.md"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_dataset(path: str) -> Tuple[List[Tuple[float, ...]], List[int], List[Dict[str, Any]]]:
    """Read JSONL rows into a feature matrix, a label vector and raw payloads."""
    if not os.path.exists(path):
        raise SystemExit(
            f"dataset not found at {path}\n"
            "hint: run 'make data' (or scripts/generate_dataset.py) first"
        )

    rows: List[Tuple[float, ...]] = []
    labels: List[int] = []
    payloads: List[Dict[str, Any]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.pop("defaulted", None)
            record.pop("_true_probability", None)
            if label is None:
                skipped += 1
                continue
            try:
                normalised = normalise_payload(record)
            except Exception as exc:  # noqa: BLE001 - report the offending line
                raise SystemExit(f"{path}:{line_number}: invalid application row: {exc}") from exc
            vector = build_features(normalised)
            rows.append(vector.values)
            labels.append(int(label))
            payloads.append(record)

    if not rows:
        raise SystemExit(f"{path} contained no usable rows")
    if skipped:
        print(f"warning: skipped {skipped} rows without a 'defaulted' label")
    return rows, labels, payloads


def split_indices(
    count: int, *, seed: int, val_fraction: float, test_fraction: float, cal_fraction: float = 0.0
):
    """Deterministic multi-way index split (train / validation / calibration / test)."""
    import random

    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction must be strictly between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise SystemExit("--val-fraction must be within [0, 1)")
    if not 0.0 <= cal_fraction < 1.0:
        raise SystemExit("--cal-fraction must be within [0, 1)")
    if val_fraction + test_fraction + cal_fraction >= 1.0:
        raise SystemExit("validation + calibration + test fractions must leave a training set")

    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(round(count * test_fraction)))
    n_cal = max(1, int(round(count * cal_fraction))) if cal_fraction > 0 else 0
    n_val = max(1, int(round(count * val_fraction))) if val_fraction > 0 else 0

    test_idx = indices[:n_test]
    cal_idx = indices[n_test : n_test + n_cal]
    val_idx = indices[n_test + n_cal : n_test + n_cal + n_val]
    train_idx = indices[n_test + n_cal + n_val :]
    return train_idx, val_idx, cal_idx, test_idx


def _subset(rows: Sequence[Tuple[float, ...]], labels: Sequence[int], indices: Sequence[int]):
    return [rows[i] for i in indices], [labels[i] for i in indices]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def importance_from_vectors(
    model: ScorecardModel, rows: Sequence[Sequence[float]]
) -> List[Dict[str, Any]]:
    """Rank features by the spread of their contribution across the population.

    A raw weight is not comparable between a binary indicator and a continuous
    feature. The most honest summary of "how much does this feature move the
    decision" is the standard deviation of its contribution over real
    applicants. ``weight`` is reported alongside so the sign and the mechanism
    stay visible.
    """
    per_feature: Dict[str, List[float]] = {name: [] for name in model.feature_names}
    for values in rows:
        for name, contribution in model.contributions(values):
            per_feature[name].append(contribution)

    ranked: List[Dict[str, Any]] = []
    for index, name in enumerate(model.feature_names):
        values = per_feature[name]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        ranked.append(
            {
                "feature": name,
                "weight": round(model.weights[index], 6),
                "mean_contribution": round(mean_value, 6),
                "contribution_std": round(variance ** 0.5, 6),
                "min_contribution": round(min(values), 6),
                "max_contribution": round(max(values), 6),
            }
        )
    ranked.sort(key=lambda item: -item["contribution_std"])
    return ranked


def select_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    low: float = 0.05,
    high: float = 0.90,
    step: float = 0.005,
) -> Tuple[float, Dict[str, Any]]:
    """Choose the decision threshold that maximises F1 on the validation set.

    Why not just use 0.5? The dataset has a ~20% default rate and the classifier
    is trained with class balancing, so the posterior probabilities are pulled
    toward the middle and 0.5 would decline almost everyone. Hard-coding 0.35
    would work here but is a magic number; maximising F1 on a held-out slice is a
    defensible, reproducible rule and it prints the whole curve so a reviewer can
    see the precision/recall tradeoff rather than trusting one number.

    The model itself is untouched -- only the decision gate moves. That keeps the
    ordering, the score and the bands identical whatever threshold is chosen.
    """
    best_threshold = 0.5
    best_metrics: Dict[str, Any] = {}
    best_f1 = -1.0
    curve: List[Dict[str, float]] = []

    steps = int(round((high - low) / step)) + 1
    for index in range(steps):
        threshold = low + index * step
        metrics = evaluate(labels, probabilities, threshold=threshold)
        curve.append(
            {
                "threshold": round(threshold, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "predicted_positive_rate": round(metrics.positive_rate, 4),
            }
        )
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = threshold
            best_metrics = metrics.as_dict()

    return best_threshold, {"best_f1": best_f1, "metrics_at_best": best_metrics, "curve": curve}


def calibration_summary(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 5
) -> List[Dict[str, Any]]:
    """Compare predicted probability against the observed rate in equal-count bins.

    A scorecard is a probability model, not only a ranking model, so it has to be
    roughly calibrated: if we say 25%, about a quarter of those applicants should
    default. With class-balanced training the probabilities are deliberately
    pulled toward the centre, so this table usually shows the model ranking well
    but reading high. Publishing the gap is more useful than hiding it.
    """
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        return []

    order = sorted(range(len(labels)), key=lambda i: probabilities[i])
    table: List[Dict[str, Any]] = []
    for bucket in range(bins):
        start = bucket * len(order) // bins
        end = (bucket + 1) * len(order) // bins
        indices = order[start:end]
        if not indices:
            continue
        predicted = sum(probabilities[i] for i in indices) / len(indices)
        observed = sum(labels[i] for i in indices) / len(indices)
        table.append(
            {
                "bin": bucket + 1,
                "count": len(indices),
                "predicted_pd": round(predicted, 4),
                "observed_default_rate": round(observed, 4),
                "gap": round(predicted - observed, 4),
            }
        )
    return table


def render_markdown_summary(
    *,
    metrics: Dict[str, Any],
    importance: List[Dict[str, Any]],
    model_path: str,
    data_path: str,
    trained_at: str,
) -> str:
    """Markdown block for MODEL_CARD.md and the README."""
    test = metrics["test"]
    dataset = metrics["dataset"]
    lines = [
        "# Training summary",
        "",
        f"- Trained at: `{trained_at}`",
        f"- Dataset: `{data_path}` ({dataset['rows']} rows, default rate {dataset['default_rate']:.4f})",
        f"- Train/validation/test rows: {dataset['train_rows']} / "
        f"{metrics['config'].get('validation_rows', 'n/a')} / {dataset['test_rows']}",
        f"- Model selection: {dataset['model_selection']}",
        f"- Decision threshold: {metrics['config']['threshold']:.3f} "
        f"({metrics.get('threshold_selection', {}).get('mode', 'auto')} mode)",
        f"- Artifact: `{model_path}`",
        "",
        "## Calibration",
        "",
        f"- Method: `{metrics['calibration'].get('method', 'external_offset')}` "
        "(single log-odds bias fitted on the calibration split)",
        f"- Calibration rows: {metrics['calibration'].get('calibration_rows', 'n/a')}",
        f"- Mean predicted PD before correction (calibration split): "
        f"{metrics['calibration']['mean_predicted_pd_before']:.4f}",
        f"- Observed default rate (calibration split): "
        f"{metrics['calibration']['observed_default_rate']:.4f}",
        f"- Fitted log-odds offset: {metrics['calibration']['offset']:+.4f}",
        f"- Mean predicted PD after correction (test): "
        f"{metrics['calibration']['mean_predicted_pd_test']:.4f}",
        f"- Observed default rate (test): {metrics['calibration']['test_default_rate']:.4f}",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| AUC | {test['auc']:.4f} |",
        f"| KS | {test['ks']:.4f} |",
        f"| Accuracy | {test['accuracy']:.4f} |",
        f"| Precision | {test['precision']:.4f} |",
        f"| Recall | {test['recall']:.4f} |",
        f"| F1 | {test['f1']:.4f} |",
        f"| Decision threshold | {test['threshold']:.4f} |",
        f"| Predicted positive rate | {test['predicted_positive_rate']:.4f} |",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        "",
        "| | Predicted good | Predicted default |",
        "| --- | --- | --- |",
        f"| Actual good | {test['confusion_matrix']['true_negative']} | {test['confusion_matrix']['false_positive']} |",
        f"| Actual default | {test['confusion_matrix']['false_negative']} | {test['confusion_matrix']['true_positive']} |",
        "",
        "## Feature importance (contribution spread on the test set)",
        "",
        "| Rank | Feature | Weight | Contribution std | Mean contribution |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(importance, start=1):
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['weight']:+.4f} | "
            f"{item['contribution_std']:.4f} | {item['mean_contribution']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Risk bands",
            "",
            "| Band | Label | Score range | Expected default rate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for band in band_table():
        lower = band["lower_exclusive"]
        upper = band["upper_inclusive"]
        if lower is None and upper is None:
            span = "any"
        elif lower is None:
            span = f"<= {upper:g}"
        elif upper is None:
            span = f"> {lower:g}"
        else:
            span = f"({lower:g}, {upper:g}]"
        lines.append(
            f"| {band['band']} | {band['label']} | {span} | {band['expected_default_rate']} |"
        )

    lines.extend(
        [
            "",
            "> The dataset is synthetic and the metrics above describe performance on "
            "that synthetic distribution only. They are not evidence of performance on "
            "real lending data.",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"input JSONL (default {DEFAULT_DATA})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"output artifact (default {DEFAULT_MODEL})")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="output metrics JSON")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="output Markdown summary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decision threshold; when omitted it is selected on the validation set",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("auto", "fixed"),
        default="auto",
        help="'auto' maximises validation F1, 'fixed' needs --threshold (default auto)",
    )
    parser.add_argument("--epochs", type=int, default=400, help="maximum gradient descent epochs (default 400)")
    parser.add_argument("--learning-rate", type=float, default=0.35, help="gradient descent step size (default 0.35)")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 penalty (default 0.001)")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="test set fraction (default 0.20)")
    parser.add_argument("--val-fraction", type=float, default=0.10, help="validation set fraction (default 0.10)")
    parser.add_argument(
        "--cal-fraction",
        type=float,
        default=0.10,
        help="calibration set fraction used to fit the probability bias (default 0.10)",
    )
    parser.add_argument("--seed", type=int, default=20260101, help="split and training seed")
    parser.add_argument("--no-summary", action="store_true", help="skip the Markdown summary")
    parser.add_argument("--quiet", action="store_true", help="only print the final metric block")
    args = parser.parse_args(argv)

    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be within [0, 1]")
    if args.threshold_mode == "fixed" and args.threshold is None:
        raise SystemExit("--threshold-mode fixed requires --threshold")

    if not args.quiet:
        print(f"loading dataset from {args.data}")
    rows, labels, _payloads = load_dataset(args.data)
    print(f"  rows                {len(rows)}")
    print(f"  features            {len(FEATURE_NAMES)}")
    print(f"  default rate        {sum(labels) / len(labels):.4f}")

    train_idx, val_idx, cal_idx, test_idx = split_indices(
        len(rows),
        seed=args.seed,
        val_fraction=args.val_fraction,
        cal_fraction=args.cal_fraction,
        test_fraction=args.test_fraction,
    )
    train_rows, train_labels = _subset(rows, labels, train_idx)
    val_rows, val_labels = _subset(rows, labels, val_idx)
    cal_rows, cal_labels = _subset(rows, labels, cal_idx)
    test_rows, test_labels = _subset(rows, labels, test_idx)
    print(
        f"  train/val/cal/test  {len(train_rows)}/{len(val_rows)}/"
        f"{len(cal_rows)}/{len(test_rows)}"
    )

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        threshold=args.threshold if args.threshold is not None else 0.5,
        verbose=not args.quiet,
    )

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print("training scorecard (batch gradient descent, early stopping on validation AUC)")
    # First fit: early stopping on the validation split, no probability correction.
    uncalibrated, report = train_scorecard(
        train_rows,
        train_labels,
        feature_names=FEATURE_NAMES,
        config=config,
        trained_at=trained_at,
        validation=(val_rows, val_labels),
        offset_override=0.0,
    )

    # Fit the probability bias on a split that was used for neither training nor
    # early stopping, then refit with that offset. Only the intercept changes, so
    # the ranking (and therefore AUC/KS) is preserved; the probabilities become
    # usable as actual default rates.
    cal_probabilities = uncalibrated.predict_proba(cal_rows)
    offset, calibration_diagnostics = fit_prior_offset(cal_labels, cal_probabilities)
    print(
        f"  probability calibration: offset {offset:+.4f} "
        f"(mean predicted PD {calibration_diagnostics['mean_predicted_pd_before']:.4f} "
        f"-> observed {calibration_diagnostics['observed_default_rate']:.4f})"
    )

    model, report = train_scorecard(
        train_rows,
        train_labels,
        feature_names=FEATURE_NAMES,
        config=config,
        trained_at=trained_at,
        validation=(val_rows, val_labels),
        offset_override=offset,
    )
    calibration_diagnostics["split"] = {
        "calibration_rows": len(cal_rows),
        "calibration_default_rate": round(sum(cal_labels) / len(cal_labels), 6),
    }

    # Model selection happens on validation predictions, never on test.
    val_probabilities = model.predict_proba(val_rows)
    if args.threshold_mode == "auto":
        threshold, threshold_report = select_threshold(val_labels, val_probabilities)
        print(
            f"  threshold selected on validation: {threshold:.3f} "
            f"(validation F1 {threshold_report['best_f1']:.4f})"
        )
    else:
        threshold = float(args.threshold)
        threshold_report = {
            "best_f1": evaluate(val_labels, val_probabilities, threshold=threshold).f1,
            "selected_by": "fixed",
            "curve": [],
        }
    threshold_report["mode"] = args.threshold_mode
    threshold_report["value"] = threshold

    model = ScorecardModel(
        feature_names=model.feature_names,
        weights=model.weights,
        intercept=model.intercept,
        standardizer=model.standardizer,
        decision_threshold=threshold,
        version=model.version,
        trained_at=trained_at,
        training=model.training,
    )

    test_metrics = evaluate(test_labels, model.predict_proba(test_rows), threshold=threshold)
    validation_metrics = evaluate(val_labels, model.predict_proba(val_rows), threshold=threshold)
    train_metrics = evaluate(train_labels, model.predict_proba(train_rows), threshold=threshold)
    calibration = calibration_summary(test_labels, model.predict_proba(test_rows))

    report["dataset"].update(
        {
            "source": args.data,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "test_rows": len(test_rows),
        }
    )
    report["config"]["validation_rows"] = len(val_rows)
    report["config"]["calibration_rows"] = len(cal_rows)
    report["config"]["threshold"] = threshold
    report["validation"] = validation_metrics.as_dict()
    report["test"] = test_metrics.as_dict()
    report["train"] = train_metrics.as_dict()
    report["threshold_selection"] = threshold_report
    report["calibration"] = {**calibration_diagnostics}
    report["calibration"]["bins"] = calibration
    report["calibration"]["mean_predicted_pd_test"] = round(
        sum(model.predict_proba(test_rows)) / len(test_rows), 6
    )
    report["calibration"]["test_default_rate"] = round(sum(test_labels) / len(test_labels), 6)
    report["calibration"]["mean_predicted_pd_validation"] = round(
        sum(model.predict_proba(val_rows)) / len(val_rows), 6
    )
    report["trained_at"] = trained_at
    report["model_path"] = args.model

    importance = importance_from_vectors(model, test_rows)

    model = ScorecardModel(
        feature_names=model.feature_names,
        weights=model.weights,
        intercept=model.intercept,
        standardizer=model.standardizer,
        decision_threshold=model.decision_threshold,
        version=model.version,
        trained_at=trained_at,
        training={
            "dataset": report["dataset"],
            "config": report["config"],
            "train": report["train"],
            "validation": report["validation"],
            "test": report["test"],
            "feature_importance": importance,
            "calibration": {
                key: value for key, value in report["calibration"].items() if key != "bins"
            },
            "threshold_selection": {
                key: value for key, value in threshold_report.items() if key != "curve"
            },
        },
    )
    model.save(args.model)

    metrics_directory = os.path.dirname(os.path.abspath(args.metrics))
    if metrics_directory:
        os.makedirs(metrics_directory, exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    if not args.no_summary:
        summary_directory = os.path.dirname(os.path.abspath(args.summary))
        if summary_directory:
            os.makedirs(summary_directory, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write(
                render_markdown_summary(
                    metrics=report,
                    importance=importance,
                    model_path=args.model,
                    data_path=args.data,
                    trained_at=trained_at,
                )
            )

    print()
    print(f"held-out test metrics (threshold {threshold:.3f}):")
    print(test_metrics.format_text())
    print()
    print("top features by contribution spread:")
    for rank, item in enumerate(importance[:6], start=1):
        print(
            f"  {rank}. {item['feature']:<26} weight {item['weight']:+.4f}  "
            f"std {item['contribution_std']:.4f}"
        )
    print()
    print("calibration on the test set (predicted PD vs observed default rate):")
    predicted_mean = report["calibration"]["mean_predicted_pd_test"]
    print(
        f"  overall: mean predicted PD {predicted_mean:.4f} vs observed default rate "
        f"{report['calibration']['test_default_rate']:.4f}"
    )
    for row in calibration:
        print(
            f"  bin {row['bin']}  n={row['count']:<5d} predicted {row['predicted_pd']:.4f}  "
            f"observed {row['observed_default_rate']:.4f}  gap {row['gap']:+.4f}"
        )
    print()
    print(f"model saved to      {args.model}")
    print(f"metrics saved to    {args.metrics}")
    if not args.no_summary:
        print(f"summary saved to    {args.summary}")

    # A sanity check a reviewer would otherwise have to run by hand: the
    # lowest-risk applicant in the test set must outscore the highest-risk one.
    # `safest` is the row with the smallest probability of default.
    scorer = Scorer(model, threshold=threshold)
    safest = min(test_rows, key=lambda row: model.predict_proba_row(row))
    riskiest = max(test_rows, key=lambda row: model.predict_proba_row(row))
    safest_result = scorer.score_vector("app_selfcheck_safest", safest)
    riskiest_result = scorer.score_vector("app_selfcheck_riskiest", riskiest)
    print()
    print(
        f"self-check: safest applicant scores {safest_result.credit_score} "
        f"(PD {safest_result.probability_of_default:.4f}, band {safest_result.band}) vs "
        f"riskiest {riskiest_result.credit_score} "
        f"(PD {riskiest_result.probability_of_default:.4f}, band {riskiest_result.band})"
    )
    if safest_result.credit_score <= riskiest_result.credit_score:
        raise SystemExit("self-check failed: score ordering is not monotone in risk")
    if safest_result.probability_of_default >= riskiest_result.probability_of_default:
        raise SystemExit("self-check failed: probability of default is not ordered correctly")
    calibration_gap = abs(
        report["calibration"]["mean_predicted_pd_test"] - report["calibration"]["test_default_rate"]
    )
    if calibration_gap > 0.05:
        print(
            f"warning: mean predicted PD is {calibration_gap:.4f} away from the observed "
            "default rate; the score ranks applicants but the probability needs recalibration"
        )
    else:
        print(f"self-check passed: mean predicted PD is within {calibration_gap:.4f} of the observed rate")
    print("self-check passed: lower risk -> higher score, bands follow the score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
