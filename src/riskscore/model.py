"""The linear logistic scorecard: training, evaluation and persistence.

Why a linear model?
-------------------
A credit scorecard has to survive review. A regulator, an auditor or a credit
committee needs to answer "why did this applicant score 612?" in one sentence.
With a linear model in log-odds space that answer is exact and additive:

``logit(p) = intercept + sum_i (weight_i * feature_i)``

Each term ``weight_i * feature_i`` is that feature's contribution to the
decision, so explainability is a property of the model rather than a separate
approximation bolted on afterwards. Gradient boosting would very likely rank
better on AUC; that is why :class:`ScorecardModel` is deliberately a thin
interface (``predict_proba`` over a feature matrix) and the roadmap notes a
swap-in backend.

Everything here is implemented in pure Python so that the artifact is
reproducible on any machine with a stock interpreter -- no wheels, no compiler,
no numerical library. The datasets this model is expected to train on
(thousands to low millions of rows) fit comfortably in memory.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MODEL_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------


def sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def logit(p: float) -> float:
    """Inverse of :func:`sigmoid`, clamped away from the asymptotes."""
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def dot(weights: Sequence[float], values: Sequence[float]) -> float:
    """Plain dot product; ``zip`` stops at the shorter input."""
    total = 0.0
    for weight, value in zip(weights, values):
        total += weight * value
    return total


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    """Per-column z-score standardisation.

    A linear model trained by gradient descent converges far faster on
    standardised inputs, and it makes the learned weights comparable across
    features (a useful sanity check when reviewing a scorecard). The fitted
    means and scales travel inside the model artifact so serving applies exactly
    the same transform as training.
    """

    means: Tuple[float, ...]
    scales: Tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> "Standardizer":
        if not rows:
            raise ValueError("cannot fit a standardizer on an empty matrix")
        width = len(rows[0])
        for row in rows:
            if len(row) != width:
                raise ValueError("ragged feature matrix: all rows must have equal length")
        means: List[float] = []
        scales: List[float] = []
        for column in range(width):
            values = [row[column] for row in rows]
            mu = mean(values)
            sigma = stdev(values)
            means.append(mu)
            # A constant column carries no signal; scale 1.0 keeps it finite and
            # lets the intercept absorb the level.
            scales.append(sigma if sigma > 1e-12 else 1.0)
        return cls(means=tuple(means), scales=tuple(scales))

    def transform_row(self, row: Sequence[float]) -> Tuple[float, ...]:
        return tuple(
            (value - mu) / sigma for value, mu, sigma in zip(row, self.means, self.scales)
        )

    def transform(self, rows: Iterable[Sequence[float]]) -> List[Tuple[float, ...]]:
        return [self.transform_row(row) for row in rows]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"means": list(self.means), "scales": list(self.scales)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Standardizer":
        means = tuple(float(value) for value in payload["means"])
        scales = tuple(float(value) for value in payload["scales"])
        if len(means) != len(scales):
            raise ValueError("standardizer means/scales length mismatch")
        if any(scale == 0.0 for scale in scales):
            raise ValueError("standardizer scales must be non-zero")
        return cls(means=means, scales=scales)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.means)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def train_test_split(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    test_fraction: float = 0.2,
    seed: int = 20260101,
) -> Tuple[List[Tuple[float, ...]], List[int], List[Tuple[float, ...]], List[int]]:
    """Deterministic split into train/test partitions.

    A fixed seed is used so that ``make train`` reproduces the same model
    artifact byte for byte, which is what makes the metrics in the README
    reproducible by a reviewer.
    """
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have the same length")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    cut = int(round(len(indices) * (1.0 - test_fraction)))
    cut = min(max(cut, 1), len(indices) - 1) if len(indices) > 1 else cut

    train_idx, test_idx = indices[:cut], indices[cut:]
    train_rows = [tuple(rows[i]) for i in train_idx]
    test_rows = [tuple(rows[i]) for i in test_idx]
    train_labels = [int(labels[i]) for i in train_idx]
    test_labels = [int(labels[i]) for i in test_idx]
    return train_rows, train_labels, test_rows, test_labels


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Ties receive average ranks, which is the standard convention and keeps AUC
    well defined for scorecards that emit a limited number of distinct scores.
    Returns 0.5 (chance) when only one class is present.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    rank_sum_positives = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    u = rank_sum_positives - positives * (positives + 1) / 2.0
    return u / (positives * negatives)


def ks_statistic(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Kolmogorov-Smirnov statistic: max separation of the two score CDFs."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0

    positives.sort()
    negatives.sort()
    n_pos, n_neg = len(positives), len(negatives)
    i = j = 0
    best = 0.0
    # Walk the merged score axis and compare the two empirical CDFs.
    while i < n_pos and j < n_neg:
        threshold = min(positives[i], negatives[j])
        while i < n_pos and positives[i] <= threshold:
            i += 1
        while j < n_neg and negatives[j] <= threshold:
            j += 1
        best = max(best, abs(j / n_neg - i / n_pos))
    return best


@dataclass(frozen=True)
class ClassificationMetrics:
    """Threshold-dependent metrics plus the two ranking metrics."""

    auc: float
    ks: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    threshold: float
    support: int

    @property
    def positive_rate(self) -> float:
        predicted = self.true_positive + self.false_positive
        return predicted / self.support if self.support else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "auc": round(self.auc, 6),
            "ks": round(self.ks, 6),
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "confusion_matrix": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
                "false_negative": self.false_negative,
            },
            "threshold": round(self.threshold, 6),
            "predicted_positive_rate": round(self.positive_rate, 6),
            "support": self.support,
        }

    def format_text(self) -> str:
        """Human-readable block for the training script's stdout."""
        lines = [
            f"  AUC        {self.auc:.4f}",
            f"  KS         {self.ks:.4f}",
            f"  Accuracy   {self.accuracy:.4f}",
            f"  Precision  {self.precision:.4f}",
            f"  Recall     {self.recall:.4f}",
            f"  F1         {self.f1:.4f}",
            f"  Threshold  {self.threshold:.4f}",
            "  Confusion matrix (rows = actual, cols = predicted)",
            f"    TN {self.true_negative:6d}   FP {self.false_positive:6d}",
            f"    FN {self.false_negative:6d}   TP {self.true_positive:6d}",
            f"  Support    {self.support}",
        ]
        return "\n".join(lines)


def evaluate(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute ranking and threshold metrics for a set of predictions."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    tp = fp = tn = fn = 0
    for label, probability in zip(labels, probabilities):
        predicted = 1 if probability >= threshold else 0
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    return ClassificationMetrics(
        auc=roc_auc(labels, probabilities),
        ks=ks_statistic(labels, probabilities),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        threshold=threshold,
        support=len(labels),
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_scorecard`."""

    epochs: int = 400
    learning_rate: float = 0.35
    l2: float = 1e-3
    test_fraction: float = 0.2
    seed: int = 20260101
    threshold: float = 0.35
    tol: float = 1e-9
    verbose: bool = False
    #: Undo the class-balancing shift so the output is a usable probability.
    #: See :func:`calibrate_prior_offset`.
    calibrate_prior: bool = True


@dataclass(frozen=True)
class ScorecardModel:
    """A fitted logistic scorecard.

    ``weights`` are expressed in *standardised* feature space; the companion
    :class:`Standardizer` maps raw features into that space at serving time.
    """

    feature_names: Tuple[str, ...]
    weights: Tuple[float, ...]
    intercept: float
    standardizer: Standardizer
    decision_threshold: float = 0.35
    version: str = "0.1.0"
    trained_at: str = ""
    training: Optional[Dict[str, Any]] = None

    # -- inference ---------------------------------------------------------

    def _standardise(self, values: Sequence[float]) -> Tuple[float, ...]:
        return self.standardizer.transform_row(values)

    def decision_function(self, values: Sequence[float]) -> float:
        """Return the log-odds for one raw feature row."""
        return self.intercept + dot(self.weights, self._standardise(values))

    def predict_proba_row(self, values: Sequence[float]) -> float:
        """Estimated probability of default for one raw feature row."""
        return sigmoid(self.decision_function(values))

    def predict_proba(self, rows: Iterable[Sequence[float]]) -> List[float]:
        """Estimated probability of default for a matrix of raw feature rows."""
        return [self.predict_proba_row(tuple(row)) for row in rows]

    def predict(self, rows: Iterable[Sequence[float]]) -> List[int]:
        """Binary decisions at :attr:`decision_threshold`."""
        return [1 if p >= self.decision_threshold else 0 for p in self.predict_proba(rows)]

    def contributions(self, values: Sequence[float]) -> List[Tuple[str, float]]:
        """Per-feature additive contribution to the log-odds.

        ``intercept + sum(contributions) == decision_function(values)`` holds
        exactly, which is what the explainability tests assert.
        """
        standardised = self._standardise(values)
        return [
            (name, weight * value)
            for name, weight, value in zip(self.feature_names, self.weights, standardised)
        ]

    def assert_compatible(self, names: Sequence[str]) -> None:
        """Fail loudly if the caller's feature order differs from training."""
        if tuple(names) != self.feature_names:
            raise ValueError(
                "feature mismatch: model expects "
                f"{list(self.feature_names)} but received {list(names)}"
            )

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "kind": "logistic_scorecard",
            "version": self.version,
            "trained_at": self.trained_at,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "intercept": self.intercept,
            "standardizer": self.standardizer.to_dict(),
            "decision_threshold": self.decision_threshold,
            "training": self.training or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScorecardModel":
        version = payload.get("format_version")
        if version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"unsupported model format_version {version!r}; expected {MODEL_FORMAT_VERSION}"
            )
        feature_names = tuple(str(name) for name in payload["feature_names"])
        weights = tuple(float(weight) for weight in payload["weights"])
        if len(feature_names) != len(weights):
            raise ValueError("feature_names and weights length mismatch")
        standardizer = Standardizer.from_dict(payload["standardizer"])
        if len(standardizer) != len(weights):
            raise ValueError("standardizer width does not match the weight vector")
        return cls(
            feature_names=feature_names,
            weights=weights,
            intercept=float(payload["intercept"]),
            standardizer=standardizer,
            decision_threshold=float(payload.get("decision_threshold", 0.35)),
            version=str(payload.get("version", "0.1.0")),
            trained_at=str(payload.get("trained_at", "")),
            training=payload.get("training") or None,
        )

    def save(self, path: str) -> None:
        """Atomically write the artifact so a crash cannot truncate a live model."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @classmethod
    def load(cls, path: str) -> "ScorecardModel":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def train_scorecard(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    feature_names: Sequence[str],
    config: Optional[TrainingConfig] = None,
    trained_at: str = "",
    validation: Optional[Tuple[Sequence[Sequence[float]], Sequence[int]]] = None,
    offset_override: Optional[float] = None,
) -> Tuple[ScorecardModel, Dict[str, Any]]:
    """Fit a :class:`ScorecardModel` by batch gradient descent.

    The objective is the mean binary cross-entropy plus L2 penalty. Two
    details matter for a credit dataset:

    * **Class balancing.** Default rates are low, so the raw gradient is
      dominated by good loans and the model collapses to a constant. Each class
      is weighted by ``n / (2 * n_class)`` so both contribute equally.
    * **Early stopping on validation AUC.** The weights that rank best on held
      out data are kept, not necessarily the final epoch. AUC is used because
      it is threshold-free, which matches how a scorecard is consumed.

    When ``validation`` is supplied, early stopping uses those rows instead of
    an internal split. That lets a caller hold out a real test set and use a
    disjoint slice for model selection, which is what ``scripts/train_model.py``
    does.

    Returns the model and a metrics dictionary describing the fit.
    """
    config = config or TrainingConfig()
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have the same length")
    if not rows:
        raise ValueError("cannot train on an empty dataset")
    if len(set(labels)) < 2:
        raise ValueError("training data must contain both classes")

    feature_names = tuple(feature_names)
    width = len(feature_names)
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"feature width mismatch: expected {width}, got {len(row)}"
            )

    if validation is None:
        train_rows, train_labels, test_rows, test_labels = train_test_split(
            rows, labels, test_fraction=config.test_fraction, seed=config.seed
        )
        selection_label = "internal holdout"
    else:
        val_rows, val_labels = validation
        if len(val_rows) != len(val_labels):
            raise ValueError("validation rows and labels must have the same length")
        for row in val_rows:
            if len(row) != width:
                raise ValueError(
                    f"validation feature width mismatch: expected {width}, got {len(row)}"
                )
        if len(set(val_labels)) < 2:
            raise ValueError("validation data must contain both classes")
        train_rows, train_labels = [tuple(row) for row in rows], list(labels)
        test_rows, test_labels = [tuple(row) for row in val_rows], list(val_labels)
        selection_label = "supplied validation set"

    standardizer = Standardizer.fit(train_rows)
    train_x = standardizer.transform(train_rows)
    test_x = standardizer.transform(test_rows)

    n_train = len(train_x)
    positives = sum(train_labels)
    negatives = n_train - positives
    if positives == 0 or negatives == 0:  # pragma: no cover - guarded above
        raise ValueError("training split ended up single-class; change the seed")
    weight_positive = n_train / (2.0 * positives)
    weight_negative = n_train / (2.0 * negatives)

    weights = [0.0] * width
    intercept = 0.0

    best_auc = -1.0
    best_state: Tuple[List[float], float] = (list(weights), intercept)
    history: List[Dict[str, float]] = []
    epochs_run = 0

    for epoch in range(1, config.epochs + 1):
        grad_w = [0.0] * width
        grad_b = 0.0
        for row, label in zip(train_x, train_labels):
            predicted = sigmoid(intercept + dot(weights, row))
            sample_weight = weight_positive if label == 1 else weight_negative
            error = (predicted - label) * sample_weight
            for index, value in enumerate(row):
                grad_w[index] += error * value
            grad_b += error

        for index in range(width):
            grad_w[index] = grad_w[index] / n_train + config.l2 * weights[index]
        grad_b /= n_train

        for index in range(width):
            weights[index] -= config.learning_rate * grad_w[index]
        intercept -= config.learning_rate * grad_b

        if epoch % 25 == 0 or epoch == config.epochs:
            train_probabilities = [sigmoid(intercept + dot(weights, row)) for row in train_x]
            test_probabilities = [sigmoid(intercept + dot(weights, row)) for row in test_x]
            train_auc = roc_auc(train_labels, train_probabilities)
            test_auc = roc_auc(test_labels, test_probabilities)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_auc": round(train_auc, 6),
                    "test_auc": round(test_auc, 6),
                    "train_loss": round(
                        _mean_log_loss(train_labels, train_probabilities, weight_positive, weight_negative),
                        6,
                    ),
                }
            )
            epochs_run = epoch
            if config.verbose:
                print(
                    f"    epoch {epoch:4d}  train_auc={train_auc:.4f}  "
                    f"test_auc={test_auc:.4f}"
                )
            if test_auc > best_auc:
                best_auc = test_auc
                best_state = (list(weights), intercept)

    weights, intercept = best_state

    # Undo the class-balancing bias so the output is a usable probability, not
    # just a ranking. An explicit ``offset_override`` (fitted on a dedicated
    # calibration split) takes precedence over the analytic prior correction.
    prior_offset = 0.0
    mean_predicted_before = mean(
        [sigmoid(intercept + dot(weights, row)) for row in train_x]
    )
    if offset_override is not None:
        prior_offset = float(offset_override)
        intercept += prior_offset
        calibration_method = "external_offset"
    elif config.calibrate_prior:
        prior_offset = calibrate_prior_offset(train_labels)
        intercept -= prior_offset
        calibration_method = "analytic_prior"
    else:
        calibration_method = "none"

    model = ScorecardModel(
        feature_names=feature_names,
        weights=tuple(weights),
        intercept=intercept,
        standardizer=standardizer,
        decision_threshold=config.threshold,
        trained_at=trained_at,
    )

    train_probabilities = model.predict_proba(train_x)
    test_probabilities = model.predict_proba(test_x)
    train_metrics = evaluate(train_labels, train_probabilities, threshold=config.threshold)
    test_metrics = evaluate(test_labels, test_probabilities, threshold=config.threshold)

    report: Dict[str, Any] = {
        "dataset": {
            "rows": len(rows),
            "positives": int(sum(labels)),
            "negatives": int(len(labels) - sum(labels)),
            "default_rate": round(sum(labels) / len(labels), 6),
            "train_rows": n_train,
            "test_rows": len(test_x),
            "test_default_rate": round(sum(test_labels) / len(test_labels), 6),
            "model_selection": selection_label,
        },
        "config": {
            "epochs": config.epochs,
            "epochs_run": epochs_run,
            "learning_rate": config.learning_rate,
            "l2": config.l2,
            "seed": config.seed,
            "threshold": config.threshold,
            "test_fraction": config.test_fraction,
            "calibrate_prior": config.calibrate_prior,
        },
        "calibration": {
            "method": calibration_method,
            "prior_offset": round(prior_offset, 6),
            "mean_predicted_pd_before_calibration": round(mean_predicted_before, 6),
            "mean_predicted_pd_train": round(mean(train_probabilities), 6),
            "train_default_rate": round(sum(train_labels) / len(train_labels), 6),
            "test_default_rate": round(sum(test_labels) / len(test_labels), 6),
            "mean_predicted_pd_test": round(mean(test_probabilities), 6),
        },
        "train": train_metrics.as_dict(),
        "test": test_metrics.as_dict(),
        "history": history,
    }

    model = ScorecardModel(
        feature_names=model.feature_names,
        weights=model.weights,
        intercept=model.intercept,
        standardizer=model.standardizer,
        decision_threshold=model.decision_threshold,
        version=model.version,
        trained_at=model.trained_at,
        training={
            "dataset": report["dataset"],
            "config": report["config"],
            "calibration": report["calibration"],
            "train": report["train"],
            "test": report["test"],
        },
    )
    return model, report


def calibrate_prior_offset(labels: Sequence[int]) -> float:
    """Return the log-odds offset that restores the true prior.

    Class-balanced training makes the model behave as though defaults and
    non-defaults were equally frequent, which inflates every predicted
    probability. If the sample prior is ``pi`` and the balanced training prior is
    ``1/2``, the fitted log-odds carry a constant bias of ``log(pi / (1 - pi))``
    and subtracting it returns a probability on the population scale.

    The offset uses the *observed* sample prior rather than a hard-coded number,
    so a portfolio with a 3% default rate and one with a 30% default rate are
    both handled by the same code path.
    """
    positives = sum(1 for label in labels if label == 1)
    total = len(labels)
    if total == 0:
        raise ValueError("cannot calibrate on an empty label vector")
    if positives == 0 or positives == total:
        raise ValueError("cannot calibrate with a single class present")
    prior = positives / total
    return math.log(prior / (1.0 - prior))


def fit_prior_offset(
    labels: Sequence[int], probabilities: Sequence[float]
) -> Tuple[float, Dict[str, float]]:
    """Fit a single-log-odds bias correction (Platt scaling, bias only).

    Class-balanced training inflates predicted probabilities because the model is
    told both classes are equally common. The fix is to add one constant to the
    log-odds so that the *mean predicted probability* matches the *observed*
    frequency in the calibration sample. Only the bias is fitted: the weights,
    and therefore the ranking, the AUC and the KS statistic, are untouched.

    Doing this empirically rather than with the analytic ``log(pi / (1 - pi))``
    correction matters. That closed form assumes the only distortion is the
    class weights; in practice the fitted weights also absorb part of the
    imbalance (a large intercept is pushed into the feature weights), so the
    analytic correction overshoots. Solving for the offset on real predictions
    measures the actual distortion instead of assuming its size.

    Returns ``(offset, diagnostics)``. The offset must be *added* to the
    intercept. When the predicted mean is already above the observed rate the
    offset is negative, which lowers every probability.
    """
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("cannot calibrate on an empty sample")

    observed_rate = sum(labels) / len(labels)
    predicted_mean = mean(list(probabilities))
    target_logit = logit(observed_rate)
    current_logit = logit(predicted_mean)
    offset = target_logit - current_logit

    return offset, {
        "calibration_rows": len(labels),
        "observed_default_rate": observed_rate,
        "mean_predicted_pd_before": predicted_mean,
        "target_log_odds": target_logit,
        "current_log_odds": current_logit,
        "offset": offset,
    }


def _mean_log_loss(
    labels: Sequence[int],
    probabilities: Sequence[float],
    weight_positive: float,
    weight_negative: float,
) -> float:
    """Class-weighted mean cross-entropy, matching the training objective."""
    eps = 1e-12
    total = 0.0
    weight_sum = 0.0
    for label, probability in zip(labels, probabilities):
        probability = min(max(probability, eps), 1.0 - eps)
        sample_weight = weight_positive if label == 1 else weight_negative
        total += -sample_weight * (
            label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
        )
        weight_sum += sample_weight
    return total / weight_sum if weight_sum else 0.0


__all__ = [
    "MODEL_FORMAT_VERSION",
    "Standardizer",
    "ClassificationMetrics",
    "TrainingConfig",
    "ScorecardModel",
    "train_scorecard",
    "train_test_split",
    "roc_auc",
    "ks_statistic",
    "evaluate",
    "calibrate_prior_offset",
    "fit_prior_offset",
    "sigmoid",
    "logit",
]
