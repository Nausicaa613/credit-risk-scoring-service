#!/usr/bin/env python3
"""Generate a synthetic credit application dataset.

Why synthetic data?
-------------------
Public credit datasets (German Credit, Lending Club extracts, Home Credit) come
with licence restrictions, inconsistent feature definitions and, in the case of
Lending Club, real borrower information. A generator gives us a dataset we can
ship in the repository, regenerate byte for byte, and reason about: we know the
true generating process, so we can tell the difference between "the model is
bad" and "the data has no signal".

Design
------
Each applicant is drawn from a small set of correlated latent factors (income
level, credit discipline, debt burden). Defaults are then sampled from a
logistic function of those factors, so the Bayes-optimal ranking is knowable
and a model that reaches a reasonable AUC is genuinely learning structure
rather than fitting noise. A configurable amount of label noise keeps the task
from being trivially separable.

Output: JSON Lines, one application per line, with the label in ``defaulted``.
The file is deterministic for a given ``--seed``.

Usage
-----
    python scripts/generate_dataset.py --rows 4000 --seed 20260101 \\
        --output data/applications.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riskscore.features import PURPOSE_CODES  # noqa: E402

DEFAULT_ROWS = 4000
DEFAULT_SEED = 20260101
DEFAULT_BASE_RATE = 0.18

#: Latent factor coefficients. Kept in one place so the data generating process
#: is auditable and can be tuned deliberately.
TRUE_COEFFICIENTS: Dict[str, float] = {
    "intercept": -1.45,
    "income_level": -0.95,
    "credit_discipline": -1.15,
    "debt_burden": 1.30,
    "employment_stability": -0.45,
    "purpose_risk": 0.35,
}

#: Relative risk weight per loan purpose, used as a latent driver.
PURPOSE_RISK: Dict[str, float] = {
    "working_capital": 0.10,
    "equipment": -0.05,
    "expansion": 0.05,
    "inventory": 0.15,
    "refinance": 0.35,
    "personal": 0.20,
    "education": -0.15,
    "other": 0.25,
}

#: Multiplier applied to the thin-file indicator. Kept modest so the latent
#: logit does not have a fat tail, which would make the base-rate calibration
#: depend heavily on a handful of extreme applicants.
THIN_FILE_COEFFICIENT = 0.45

#: Latent logit is clamped to this range before the sigmoid. Without a clamp a
#: rare draw can produce a latent probability of 0.999, and the realised default
#: rate then swings with the sample; with it, the generating process stays
#: well behaved and the base rate is reproducible.
LATENT_LOGIT_CLAMP = 3.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _logistic(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _round(value: float, digits: int = 2) -> float:
    return round(value + 0.0, digits)


def _latent_logit(
    *,
    income_level: float,
    credit_discipline: float,
    debt_burden: float,
    employment_stability: float,
    purpose_risk: float,
    is_thin_file: bool,
    intercept_shift: float,
) -> float:
    """The true log-odds of default for one latent profile."""
    coefficients = TRUE_COEFFICIENTS
    raw = (
        coefficients["intercept"]
        + coefficients["income_level"] * income_level
        + coefficients["credit_discipline"] * credit_discipline
        + coefficients["debt_burden"] * debt_burden
        + coefficients["employment_stability"] * employment_stability
        + coefficients["purpose_risk"] * purpose_risk
        + THIN_FILE_COEFFICIENT * (1.0 if is_thin_file else 0.0)
        + intercept_shift
    )
    return _clamp(raw, -LATENT_LOGIT_CLAMP, LATENT_LOGIT_CLAMP)


def _draw_latent(rng: random.Random) -> Dict[str, Any]:
    """Draw the latent factors and the observed features built from them."""
    income_level = _clamp(rng.gauss(0.0, 1.0), -2.5, 2.5)
    credit_discipline = _clamp(rng.gauss(0.0, 1.0), -2.5, 2.5)
    debt_burden = _clamp(rng.gauss(0.0, 1.0), -2.5, 2.5)
    employment_stability = _clamp(0.6 * credit_discipline + rng.gauss(0.0, 0.8), -2.5, 2.5)
    thin_file_latent = -0.7 * credit_discipline + rng.gauss(0.0, 0.5)

    purpose = rng.choices(
        PURPOSE_CODES,
        weights=[18, 12, 14, 10, 16, 12, 6, 12],
        k=1,
    )[0]
    purpose_risk = PURPOSE_RISK[purpose]

    age = _clamp(38 + 9 * employment_stability + rng.gauss(0.0, 8.5), 19, 78)

    income_median = 180_000.0
    annual_income = _clamp(
        income_median * math.exp(0.55 * income_level + rng.gauss(0.0, 0.22)),
        36_000.0,
        4_200_000.0,
    )

    employment_years = _clamp(
        max(0.0, (age - 21.0)) * _logistic(0.9 * employment_stability) * rng.uniform(0.35, 0.95),
        0.0,
        42.0,
    )

    debt_to_income_ratio = _clamp(
        0.30 + 0.16 * debt_burden - 0.07 * income_level + rng.gauss(0.0, 0.07),
        0.0,
        1.6,
    )

    num_delinquencies_24m = max(
        0, int(round(rng.gauss(0.55 - 0.85 * credit_discipline, 0.85)))
    )

    credit_history_months = _clamp(
        (age - 20.0) * 12.0 * _logistic(0.75 * thin_file_latent) * rng.uniform(0.45, 0.95),
        3.0,
        480.0,
    )

    loan_amount = _clamp(
        annual_income * rng.uniform(0.08, 0.65) * math.exp(0.18 * debt_burden),
        20_000.0,
        6_000_000.0,
    )

    loan_term_months = rng.choice([6, 12, 18, 24, 36, 48, 60, 84, 120])
    num_open_accounts = max(0, int(round(rng.gauss(6.5 - 1.2 * credit_discipline, 2.6))))
    revolving_utilization = _clamp(
        0.42 + 0.16 * debt_burden - 0.11 * credit_discipline + rng.gauss(0.0, 0.12),
        0.0,
        1.45,
    )

    return {
        "income_level": income_level,
        "credit_discipline": credit_discipline,
        "debt_burden": debt_burden,
        "employment_stability": employment_stability,
        "purpose_risk": purpose_risk,
        "is_thin_file": credit_history_months < 24.0,
        "purpose": purpose,
        "features": {
            "age": int(round(age)),
            "annual_income": _round(annual_income, 0),
            "employment_years": _round(employment_years, 1),
            "debt_to_income_ratio": _round(debt_to_income_ratio, 3),
            "num_delinquencies_24m": int(num_delinquencies_24m),
            "credit_history_months": int(round(credit_history_months)),
            "loan_amount": _round(loan_amount, 0),
            "loan_term_months": int(loan_term_months),
            "num_open_accounts": int(num_open_accounts),
            "revolving_utilization": _round(revolving_utilization, 3),
        },
    }


def _mean_true_probability(seed: int, *, intercept_shift: float, pilot: int = 800) -> float:
    """Average true default probability under a given intercept shift.

    A fresh generator is used for every evaluation so that bisection does not
    depend on, or disturb, any other random stream.
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(pilot):
        latent = _draw_latent(rng)
        logit_value = _latent_logit(
            income_level=latent["income_level"],
            credit_discipline=latent["credit_discipline"],
            debt_burden=latent["debt_burden"],
            employment_stability=latent["employment_stability"],
            purpose_risk=latent["purpose_risk"],
            is_thin_file=latent["is_thin_file"],
            intercept_shift=intercept_shift,
        )
        total += _logistic(logit_value)
    return total / pilot


def _solve_intercept_shift(seed: int, base_rate: float) -> float:
    """Find the intercept shift whose mean true probability equals ``base_rate``.

    Bisection rather than a closed form: the latent logit is clamped and the
    observed features are clamped too, so the mapping from shift to mean
    probability is monotone but not analytic. 60 iterations pin the shift to far
    tighter than the sampling error of the dataset itself.

    Fixing the *mean probability* (not the mean logit) is the point: the realised
    default rate is ``E[p]``, and ``E[sigmoid(z)] != sigmoid(E[z])``.
    """
    low, high = -25.0, 25.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _mean_true_probability(seed, intercept_shift=mid) < base_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_row(
    rng: random.Random, *, label_noise: float, intercept_shift: float = 0.0
) -> Dict[str, Any]:
    """Draw one applicant and its (possibly noisy) outcome."""
    latent = _draw_latent(rng)
    true_logit = _latent_logit(
        income_level=latent["income_level"],
        credit_discipline=latent["credit_discipline"],
        debt_burden=latent["debt_burden"],
        employment_stability=latent["employment_stability"],
        purpose_risk=latent["purpose_risk"],
        is_thin_file=latent["is_thin_file"],
        intercept_shift=intercept_shift,
    )
    true_probability = _logistic(true_logit)

    # Label noise models the part of real default behaviour no feature set
    # captures (job loss, illness, fraud). Without it AUC would be inflated.
    noisy_probability = _clamp(
        true_probability * (1.0 - label_noise) + label_noise * 0.5, 0.0, 1.0
    )
    defaulted = 1 if rng.random() < noisy_probability else 0

    row: Dict[str, Any] = dict(latent["features"])
    row["purpose"] = latent["purpose"]
    row["defaulted"] = defaulted
    row["_true_probability"] = round(true_probability, 6)
    return row


def generate_dataset(
    rows: int,
    *,
    seed: int = DEFAULT_SEED,
    label_noise: float = 0.12,
    base_rate: float = DEFAULT_BASE_RATE,
) -> List[Dict[str, Any]]:
    """Generate ``rows`` applications with a target default rate near ``base_rate``.

    The latent intercept is shifted so the realised default rate lands close to
    the requested base rate, which keeps the dataset realistic for a lending
    portfolio and makes the class-balancing logic in training observable.
    """
    if rows < 1:
        raise ValueError("rows must be >= 1")
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise must be within [0, 1)")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be strictly between 0 and 1")

    # Step 1: solve for the intercept shift that makes the *mean true
    # probability* equal the requested base rate. Using the mean logit instead
    # would overstate the default rate, because E[sigmoid(z)] > sigmoid(E[z]).
    shift = _solve_intercept_shift(seed, base_rate)

    # Step 2: generate from the requested seed. Same seed and same shift means
    # the dataset is reproducible row for row.
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []
    for _ in range(rows):
        dataset.append(generate_row(rng, label_noise=label_noise, intercept_shift=shift))

    return dataset


def split_features_and_label(row: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Separate the payload from the label and the audit-only diagnostics."""
    payload = {
        key: value
        for key, value in row.items()
        if key not in ("defaulted", "_true_probability")
    }
    return payload, int(row["defaulted"])


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"rows to generate (default {DEFAULT_ROWS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--output", default="data/applications.jsonl", help="output JSONL path")
    parser.add_argument("--label-noise", type=float, default=0.12, help="label noise fraction (default 0.12)")
    parser.add_argument("--base-rate", type=float, default=DEFAULT_BASE_RATE, help="target default rate (default 0.18)")
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    args = parser.parse_args(argv)

    dataset = generate_dataset(
        args.rows,
        seed=args.seed,
        label_noise=args.label_noise,
        base_rate=args.base_rate,
    )
    write_jsonl(dataset, args.output)

    defaults = sum(row["defaulted"] for row in dataset)
    if not args.quiet:
        print(f"wrote {len(dataset)} rows to {args.output}")
        print(f"  seed            {args.seed}")
        print(f"  default rate    {defaults / len(dataset):.4f} ({defaults}/{len(dataset)})")
        print(f"  label noise     {args.label_noise}")
        print(f"  mean true PD    {sum(r['_true_probability'] for r in dataset) / len(dataset):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
