"""Feature engineering: raw application payload -> numeric feature vector.

This module owns three responsibilities:

1. **Validation** - reject payloads that are not physically meaningful (negative
   income, a 500 year old applicant, ...). Validation errors are collected so
   the API can report every problem in one response instead of one at a time.
2. **Transform** - turn raw values into the model's input space. Skewed money
   amounts are ``log1p`` compressed and ``purpose`` is one-hot encoded, both so
   that a *linear* model can fit the relationship.
3. **Names** - expose a stable, ordered feature list. The training script and
   the scorer both derive their column order from :func:`feature_names`, which
   is what keeps the offline and online paths consistent.

Feature order is part of the model contract: a ``model.json`` records the exact
feature list it was trained with and scoring refuses to run on a mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .errors import ValidationError

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

#: Valid loan purposes. Order is stable and is part of the encoding contract.
PURPOSE_CODES: Tuple[str, ...] = (
    "working_capital",
    "equipment",
    "expansion",
    "inventory",
    "refinance",
    "personal",
    "education",
    "other",
)

#: Continuous numeric inputs: name -> (minimum, maximum, human description).
NUMERIC_FIELDS: Dict[str, Tuple[float, float, str]] = {
    "age": (18.0, 100.0, "applicant age in years"),
    "annual_income": (0.0, 10_000_000.0, "gross annual income"),
    "employment_years": (0.0, 60.0, "years in current employment"),
    "debt_to_income_ratio": (0.0, 5.0, "existing debt service / income"),
    "num_delinquencies_24m": (0.0, 50.0, "delinquencies in the last 24 months"),
    "credit_history_months": (0.0, 720.0, "months of credit history"),
    "loan_amount": (0.0, 50_000_000.0, "requested principal"),
    "loan_term_months": (1.0, 480.0, "requested term in months"),
    "num_open_accounts": (0.0, 100.0, "open trade lines"),
    "revolving_utilization": (0.0, 5.0, "revolving balance / limit"),
}

#: Required keys. Missing keys are a hard validation failure.
REQUIRED_FIELDS: Tuple[str, ...] = tuple(NUMERIC_FIELDS) + ("purpose",)

#: Alias table so a friendlier payload spelling still works. Kept small and
#: explicit; we do not attempt fuzzy matching.
FIELD_ALIASES: Dict[str, str] = {
    "income": "annual_income",
    "annualIncome": "annual_income",
    "dti": "debt_to_income_ratio",
    "debtToIncomeRatio": "debt_to_income_ratio",
    "loanAmount": "loan_amount",
    "loanTermMonths": "loan_term_months",
    "term_months": "loan_term_months",
    "delinquencies": "num_delinquencies_24m",
    "purpose_code": "purpose",
}

#: Money-like features that get a log1p compression before entering a linear model.
_LOG1P_FIELDS: Tuple[str, ...] = ("annual_income", "loan_amount")

#: Bounded ratios that get clipped rather than log transformed.
_CLIPPED_FIELDS: Tuple[str, ...] = (
    "debt_to_income_ratio",
    "revolving_utilization",
)


def feature_names() -> List[str]:
    """Return the ordered model input names.

    The order is: numeric features (in :data:`NUMERIC_FIELDS` order, with money
    fields renamed to their compressed form) followed by one indicator per
    purpose code.
    """
    names: List[str] = []
    for field in NUMERIC_FIELDS:
        names.append(f"log_{field}" if field in _LOG1P_FIELDS else field)
    names.extend(f"purpose_{code}" for code in PURPOSE_CODES)
    return names


#: Cached because both training and serving call this on every request.
FEATURE_NAMES: Tuple[str, ...] = tuple(feature_names())


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def _coerce_number(raw: Any, *, field: str) -> float:
    """Coerce a JSON scalar to float, rejecting booleans and junk."""
    if isinstance(raw, bool):
        raise ValidationError(
            f"field '{field}' must be a number, got a boolean",
            details={"field": field},
        )
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise ValidationError(
                f"field '{field}' must be a number, got {raw!r}",
                details={"field": field},
            ) from exc
    else:
        raise ValidationError(
            f"field '{field}' must be a number, got {type(raw).__name__}",
            details={"field": field},
        )
    if math.isnan(value) or math.isinf(value):
        raise ValidationError(
            f"field '{field}' must be finite, got {value}",
            details={"field": field},
        )
    return value


def normalise_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply aliases, coerce scalars and enforce the field contract.

    Returns a new dict with canonical keys. Raises :class:`ValidationError`
    listing **all** problems found rather than stopping at the first one, so a
    client can fix its payload in a single iteration.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError("application payload must be a JSON object")

    canonical: Dict[str, Any] = {}
    unknown: List[str] = []
    aliased: Dict[str, str] = {}

    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValidationError("field names must be strings")
        target = FIELD_ALIASES.get(key, key)
        if target in canonical:
            raise ValidationError(
                f"field '{target}' was supplied more than once",
                details={"field": target},
            )
        if target in NUMERIC_FIELDS or target == "purpose":
            canonical[target] = value
            if target != key:
                aliased[key] = target
        else:
            unknown.append(key)

    problems: List[Dict[str, str]] = []
    for field in REQUIRED_FIELDS:
        if field not in canonical:
            problems.append({"field": field, "problem": "is required"})
    for key in unknown:
        problems.append({"field": key, "problem": "is not a recognised field"})
    if problems:
        raise ValidationError(
            "payload failed schema validation",
            details={"problems": problems, "allowed_fields": list(REQUIRED_FIELDS)},
        )

    normalised: Dict[str, Any] = {}
    for field, (low, high, description) in NUMERIC_FIELDS.items():
        value = _coerce_number(canonical[field], field=field)
        if value < low or value > high:
            problems.append(
                {"field": field, "problem": f"must be within [{low:g}, {high:g}] ({description})"}
            )
        normalised[field] = value

    purpose = canonical["purpose"]
    if not isinstance(purpose, str):
        problems.append({"field": "purpose", "problem": "must be a string"})
    else:
        purpose = purpose.strip().lower().replace("-", "_").replace(" ", "_")
        if purpose not in PURPOSE_CODES:
            problems.append(
                {
                    "field": "purpose",
                    "problem": "must be one of " + ", ".join(PURPOSE_CODES),
                }
            )
        normalised["purpose"] = purpose

    if problems:
        raise ValidationError(
            "payload failed value validation",
            details={"problems": problems},
        )

    if aliased:
        normalised["_aliased_fields"] = aliased
    return normalised


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureVector:
    """A model-ready row plus the information needed to explain it."""

    names: Tuple[str, ...]
    values: Tuple[float, ...]

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.values)


def build_features(normalised: Mapping[str, Any]) -> FeatureVector:
    """Transform a :func:`normalise_payload` result into a numeric row."""
    values: List[float] = []
    for field in NUMERIC_FIELDS:
        raw = float(normalised[field])
        if field in _LOG1P_FIELDS:
            values.append(math.log1p(max(raw, 0.0)))
        elif field in _CLIPPED_FIELDS:
            values.append(_clip(raw, 0.0, 1.5))
        else:
            values.append(raw)

    purpose = str(normalised["purpose"])
    values.extend(1.0 if purpose == code else 0.0 for code in PURPOSE_CODES)

    return FeatureVector(names=FEATURE_NAMES, values=tuple(values))


def build_feature_matrix(rows: Iterable[Mapping[str, Any]]) -> List[Tuple[float, ...]]:
    """Convenience wrapper used by the training script and benchmarks."""
    matrix: List[Tuple[float, ...]] = []
    for row in rows:
        matrix.append(build_features(normalise_payload(row)).values)
    return matrix


def raw_contributions(
    names: Sequence[str],
    values: Sequence[float],
    weights: Sequence[float],
) -> List[Tuple[str, float]]:
    """Return ``(feature_name, weight * value)`` pairs in model order."""
    return [(name, weight * value) for name, value, weight in zip(names, values, weights)]


__all__ = [
    "PURPOSE_CODES",
    "NUMERIC_FIELDS",
    "REQUIRED_FIELDS",
    "FIELD_ALIASES",
    "FEATURE_NAMES",
    "FeatureVector",
    "feature_names",
    "normalise_payload",
    "build_features",
    "build_feature_matrix",
    "raw_contributions",
]
