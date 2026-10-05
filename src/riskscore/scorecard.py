"""Scorecard translation layer.

The model speaks in log-odds; a credit decision speaks in points, bands and
reason codes. Keeping that translation in its own module means the mapping can
be reviewed, tested and re-calibrated without touching the maths in
:mod:`riskscore.model`.

Two properties are load-bearing and covered by tests:

* **Monotonicity** - a higher probability of default never yields a higher
  score.
* **Additivity** - the reported contributions sum exactly to the log-odds the
  model used, so an explanation can never contradict the decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import features as feature_module
from .model import ScorecardModel, logit, sigmoid

# --------------------------------------------------------------------------
# Score scaling (points-to-double-the-odds)
# --------------------------------------------------------------------------

#: Score assigned to an applicant whose odds of good:bad are 50:1.
BASE_SCORE = 600.0
#: Odds (good:bad) that :data:`BASE_SCORE` corresponds to.
BASE_ODDS = 50.0
#: How many points one doubling of the odds is worth.
POINTS_PER_DOUBLING = 40.0

SCORE_ODDS_FACTOR = POINTS_PER_DOUBLING / math.log(2.0)


def score_from_probability(probability_of_default: float) -> float:
    """Map a probability of default to a credit score.

    ``score = BASE_SCORE + (POINTS_PER_DOUBLING / ln 2) * ln(odds)``, where
    ``odds`` is good:bad. **A higher score means lower risk** -- that is the
    orientation the industry uses and the one every test asserts. Because
    ``ln(odds)`` is strictly decreasing in the probability of default, the
    mapping is monotone.

    Worked reference points (``BASE_SCORE=600``, 40 points per doubling), all
    asserted in ``tests/test_scorecard.py`` so code and documentation cannot
    drift apart:

    =========  =========
    PD         Score
    =========  =========
    1.5%       841
    5%         770
    20%        680
    60%        577
    90%        402
    =========  =========

    Note the compression at the risky end: this is the standard
    points-to-double-the-odds behaviour, where equal score steps mean equal *odds*
    steps rather than equal probability steps. It is why the bands below are cut
    at unequal probability intervals.
    """
    probability = min(max(float(probability_of_default), 1e-12), 1.0 - 1e-12)
    odds = (1.0 - probability) / probability
    return BASE_SCORE + SCORE_ODDS_FACTOR * math.log(odds)


def probability_from_score(score: float) -> float:
    """Inverse of :func:`score_from_probability`; used by calibration tests."""
    odds = math.exp((float(score) - BASE_SCORE) / SCORE_ODDS_FACTOR)
    return 1.0 / (1.0 + odds)


# --------------------------------------------------------------------------
# Risk bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBand:
    """A named risk grade covering ``(lower, upper]`` in score points."""

    name: str
    lower: float
    upper: float
    label: str
    expected_default_rate: str


#: Ordered from best to worst. ``lower`` is exclusive, ``upper`` inclusive,
#: except for the top band which includes its upper bound.
#:
#: The edges are *computed* from the published probability cut-offs rather than
#: typed in by hand, so a band label and the probability of default in the
#: response can never drift apart. If someone edits the score scale, the bands
#: move with it and ``tests/test_scorecard.py`` catches any inconsistency.
_BAND_CUTOFFS: Tuple[Tuple[str, float, str, str], ...] = (
    # name, lower PD bound, human label, expected default rate description
    ("A", 0.015, "very low risk", "< 1.5%"),
    ("B", 0.05, "low risk", "1.5% - 5%"),
    ("C", 0.12, "moderate risk", "5% - 12%"),
    ("D", 0.25, "elevated risk", "12% - 25%"),
    ("E", 0.0, "high risk", "> 25%"),
)


def _build_risk_bands() -> Tuple[RiskBand, ...]:
    """Turn the probability cut-offs above into score intervals.

    Called after :func:`score_from_probability` is defined, because the edges are
    derived from it.
    """
    bands: List[RiskBand] = []
    for index, (name, lower_pd, label, description) in enumerate(_BAND_CUTOFFS):
        lower_score = -math.inf if index == len(_BAND_CUTOFFS) - 1 else score_from_probability(lower_pd)
        upper_score = math.inf if index == 0 else bands[-1].lower
        bands.append(RiskBand(name, lower_score, upper_score, label, description))
    return tuple(bands)


#: Built once, after the score scale is defined, because the edges are derived
#: from :func:`score_from_probability`.
RISK_BANDS: Tuple[RiskBand, ...] = _build_risk_bands()


def band_for_score(score: float) -> RiskBand:
    """Return the band containing ``score``."""
    for band in RISK_BANDS:
        if band.lower < score <= band.upper:
            return band
    return RISK_BANDS[-1]


def band_table() -> List[Dict[str, Any]]:
    """Serialisable description of the banding scheme for the API/docs."""
    table: List[Dict[str, Any]] = []
    for band in RISK_BANDS:
        table.append(
            {
                "band": band.name,
                "label": band.label,
                "lower_exclusive": None if band.lower == -math.inf else band.lower,
                "upper_inclusive": None if band.upper == math.inf else band.upper,
                "expected_default_rate": band.expected_default_rate,
            }
        )
    return table


# --------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------

#: Feature -> (human description, direction hint). ``risk_up`` means a larger
#: value pushes risk up; ``risk_down`` means it pushes risk down.
FEATURE_SEMANTICS: Dict[str, Tuple[str, str]] = {
    "age": ("applicant age", "risk_down"),
    "log_annual_income": ("annual income", "risk_down"),
    "employment_years": ("employment tenure", "risk_down"),
    "debt_to_income_ratio": ("debt-to-income ratio", "risk_up"),
    "num_delinquencies_24m": ("delinquencies in last 24 months", "risk_up"),
    "credit_history_months": ("credit history length", "risk_down"),
    "log_loan_amount": ("requested loan amount", "risk_up"),
    "loan_term_months": ("loan term", "risk_up"),
    "num_open_accounts": ("open accounts", "risk_up"),
    "revolving_utilization": ("revolving utilisation", "risk_up"),
}

#: Purpose indicators share one description.
_PURPOSE_PREFIX = "purpose_"
_PURPOSE_DESCRIPTION = "loan purpose indicator"


def describe_feature(name: str) -> Tuple[str, str]:
    """Return ``(description, direction_hint)`` for a model feature."""
    if name in FEATURE_SEMANTICS:
        return FEATURE_SEMANTICS[name]
    if name.startswith(_PURPOSE_PREFIX):
        return (_PURPOSE_DESCRIPTION, "context")
    return (name.replace("_", " "), "context")


def top_contributors(
    contributions: Sequence[Tuple[str, float]],
    *,
    limit: int = 3,
) -> List[Tuple[str, float]]:
    """Largest absolute contributions first, ties broken by model order."""
    indexed = list(enumerate(contributions))
    indexed.sort(key=lambda pair: (-abs(pair[1][1]), pair[0]))
    return [pair[1] for pair in indexed[:limit]]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreResult:
    """Everything needed to render a scoring decision and explain it."""

    application_id: str
    credit_score: int
    probability_of_default: float
    log_odds: float
    band: str
    band_label: str
    decision: str
    decision_threshold: float
    contributions: Tuple[Tuple[str, float], ...]
    intercept: float
    model_version: str
    expected_default_rate: str

    @property
    def approved(self) -> bool:
        return self.decision == "approve"

    def reason_codes(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Explain the decision using the largest contributions.

        A positive contribution raised the estimated risk; a negative one
        lowered it. Returning both is deliberate: a decline notice that only
        lists negatives is unhelpful to a reviewer.
        """
        reasons: List[Dict[str, Any]] = []
        for name, contribution in top_contributors(self.contributions, limit=limit):
            description, hint = describe_feature(name)
            direction = "increases_risk" if contribution > 0 else "decreases_risk"
            reasons.append(
                {
                    "feature": name,
                    "description": description,
                    "contribution": round(contribution, 6),
                    "direction": direction,
                    "expected_direction": hint,
                    "consistent_with_prior": hint == "context"
                    or (hint == "risk_up" and contribution >= 0)
                    or (hint == "risk_down" and contribution <= 0),
                }
            )
        return reasons

    def as_dict(self, *, reason_limit: int = 3) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "credit_score": self.credit_score,
            "probability_of_default": round(self.probability_of_default, 6),
            "risk_band": self.band,
            "risk_band_label": self.band_label,
            "decision": self.decision,
            "decision_threshold": round(self.decision_threshold, 6),
            "model_version": self.model_version,
            "expected_default_rate": self.expected_default_rate,
            "log_odds": round(self.log_odds, 6),
            "intercept": round(self.intercept, 6),
            "contributions": [
                {"feature": name, "contribution": round(value, 6)}
                for name, value in self.contributions
            ],
            "reason_codes": self.reason_codes(limit=reason_limit),
        }


class Scorer:
    """Turns a validated application into a :class:`ScoreResult`."""

    def __init__(self, model: ScorecardModel, *, threshold: Optional[float] = None) -> None:
        self.model = model
        self.threshold = model.decision_threshold if threshold is None else float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("decision threshold must be within [0, 1]")

    def score(self, application_id: str, payload: Mapping[str, Any]) -> ScoreResult:
        """Score one raw application payload (not yet normalised)."""
        normalised = feature_module.normalise_payload(payload)
        return self.score_normalised(application_id, normalised)

    def score_normalised(
        self, application_id: str, normalised: Mapping[str, Any]
    ) -> ScoreResult:
        """Score a payload that has already passed :func:`normalise_payload`."""
        vector = feature_module.build_features(normalised)
        self.model.assert_compatible(vector.names)
        return self.score_vector(application_id, vector.values)

    def score_vector(
        self, application_id: str, values: Sequence[float]
    ) -> ScoreResult:
        """Score a raw, model-ready feature vector."""
        log_odds = self.model.decision_function(values)
        probability = sigmoid(log_odds)
        raw_score = score_from_probability(probability)
        # Clip once, then derive the band from the clipped value, so the reported
        # score and the reported band can never contradict each other. Both
        # clamps are monotone, so ordering between applicants is unaffected.
        score = min(max(raw_score, 300.0), 850.0)
        credit_score = int(round(score))
        band = band_for_score(score)
        decision = "decline" if probability >= self.threshold else "approve"

        return ScoreResult(
            application_id=application_id,
            credit_score=credit_score,
            probability_of_default=probability,
            log_odds=log_odds,
            band=band.name,
            band_label=band.label,
            decision=decision,
            decision_threshold=self.threshold,
            contributions=tuple(self.model.contributions(values)),
            intercept=self.model.intercept,
            model_version=self.model.version,
            expected_default_rate=band.expected_default_rate,
        )


__all__ = [
    "BASE_SCORE",
    "BASE_ODDS",
    "POINTS_PER_DOUBLING",
    "RiskBand",
    "RISK_BANDS",
    "ScoreResult",
    "Scorer",
    "score_from_probability",
    "probability_from_score",
    "band_for_score",
    "band_table",
    "describe_feature",
    "top_contributors",
    "logit",
]
