"""Request and response models for the credit risk API.

Field names are the lowercase forms of settings.MODEL_INPUT_COLUMNS. Every range mirrors
settings.FIELD_RANGES, which the data card documents. Unknown fields are rejected so a typo
never silently drops an input. Types are strict: booleans and numeric strings are rejected,
NaN and Infinity are rejected, and an integer is accepted wherever a float is expected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from credit_risk import settings

BATCH_MAX_ITEMS = 100

Decision = Literal["likely_default", "unlikely_default"]
DECISION_LABELS: dict[str, str] = {
    "likely_default": "Likely to default",
    "unlikely_default": "Unlikely to default",
}

STATUS_NOTE = "-2 no consumption, -1 paid in full, 0 revolving credit, 1 to 8 months of delay"
MONTHS = ("September 2005", "August 2005", "July 2005", "June 2005", "May 2005", "April 2005")


def _bounded(column: str, description: str) -> Any:
    """A required field whose bounds come from settings.FIELD_RANGES.

    Float fields also refuse NaN and Infinity, which Python's json module emits by default.
    """
    low, high = settings.FIELD_RANGES[column]
    if isinstance(low, float):
        return Field(..., ge=low, le=high, allow_inf_nan=False, description=description)
    return Field(..., ge=low, le=high, description=description)


class Applicant(BaseModel):
    """One cardholder, described by the 21 raw inputs the model accepts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    limit_bal: float = _bounded(
        "LIMIT_BAL", "Credit limit in NT dollars, including family supplementary credit"
    )
    education: int = _bounded(
        "EDUCATION", "Education: 1 graduate school, 2 university, 3 high school, 4 other"
    )
    age: int = _bounded("AGE", "Age in years")
    pay_0: int = _bounded(
        "PAY_0", f"Repayment status in {MONTHS[0]}, the latest month: {STATUS_NOTE}"
    )
    pay_2: int = _bounded("PAY_2", f"Repayment status in {MONTHS[1]}: {STATUS_NOTE}")
    pay_3: int = _bounded("PAY_3", f"Repayment status in {MONTHS[2]}: {STATUS_NOTE}")
    pay_4: int = _bounded("PAY_4", f"Repayment status in {MONTHS[3]}: {STATUS_NOTE}")
    pay_5: int = _bounded("PAY_5", f"Repayment status in {MONTHS[4]}: {STATUS_NOTE}")
    pay_6: int = _bounded("PAY_6", f"Repayment status in {MONTHS[5]}: {STATUS_NOTE}")
    bill_amt1: float = _bounded(
        "BILL_AMT1", f"Bill statement in {MONTHS[0]}, NT dollars; negative is a credit balance"
    )
    bill_amt2: float = _bounded(
        "BILL_AMT2", f"Bill statement in {MONTHS[1]}, NT dollars; negative is a credit balance"
    )
    bill_amt3: float = _bounded(
        "BILL_AMT3", f"Bill statement in {MONTHS[2]}, NT dollars; negative is a credit balance"
    )
    bill_amt4: float = _bounded(
        "BILL_AMT4", f"Bill statement in {MONTHS[3]}, NT dollars; negative is a credit balance"
    )
    bill_amt5: float = _bounded(
        "BILL_AMT5", f"Bill statement in {MONTHS[4]}, NT dollars; negative is a credit balance"
    )
    bill_amt6: float = _bounded(
        "BILL_AMT6", f"Bill statement in {MONTHS[5]}, NT dollars; negative is a credit balance"
    )
    pay_amt1: float = _bounded("PAY_AMT1", f"Amount paid in {MONTHS[0]}, NT dollars")
    pay_amt2: float = _bounded("PAY_AMT2", f"Amount paid in {MONTHS[1]}, NT dollars")
    pay_amt3: float = _bounded("PAY_AMT3", f"Amount paid in {MONTHS[2]}, NT dollars")
    pay_amt4: float = _bounded("PAY_AMT4", f"Amount paid in {MONTHS[3]}, NT dollars")
    pay_amt5: float = _bounded("PAY_AMT5", f"Amount paid in {MONTHS[4]}, NT dollars")
    pay_amt6: float = _bounded("PAY_AMT6", f"Amount paid in {MONTHS[5]}, NT dollars")

    def to_row(self) -> dict[str, float]:
        """Values keyed by the uppercase model input columns, in model order."""
        data = self.model_dump()
        return {column: data[column.lower()] for column in settings.MODEL_INPUT_COLUMNS}


def applicants_to_frame(items: Sequence[Applicant]) -> pd.DataFrame:
    """One row per applicant with exactly settings.MODEL_INPUT_COLUMNS, in order."""
    return pd.DataFrame(
        [item.to_row() for item in items], columns=list(settings.MODEL_INPUT_COLUMNS)
    )


class Prediction(BaseModel):
    """Score and decision for one applicant."""

    model_config = ConfigDict(protected_namespaces=())

    probability: float = Field(
        description="Probability of default next month, rounded to four decimals"
    )
    threshold: float = Field(
        description="Cost-optimal decision threshold chosen on the validation split"
    )
    decision: Decision = Field(
        description=(
            "likely_default when probability >= threshold, else unlikely_default. The comparison "
            "uses the four-decimal probability in this response, so the two fields never disagree"
        )
    )
    decision_label: str = Field(description="Sentence-case label for the decision")
    model: str = Field(description="Model family behind the score, for example hgb")
    model_version: str = Field(
        description="Package version plus the short git sha the model was trained at"
    )
    disclaimer: str = Field(description="Scope note that travels with every prediction")


class BatchRequest(BaseModel):
    """Up to BATCH_MAX_ITEMS applicants scored in one call."""

    model_config = ConfigDict(extra="forbid")

    items: list[Applicant] = Field(
        min_length=1,
        max_length=BATCH_MAX_ITEMS,
        description=f"Applicants to score, in order; 1 to {BATCH_MAX_ITEMS} per call",
    )

    @field_validator("items", mode="before")
    @classmethod
    def _check_batch_size(cls, value: Any) -> Any:
        if isinstance(value, list) and not 1 <= len(value) <= BATCH_MAX_ITEMS:
            raise ValueError(f"a batch holds 1 to {BATCH_MAX_ITEMS} items, got {len(value)}")
        return value


class BatchResponse(BaseModel):
    """Predictions in the same order as the request items."""

    count: int = Field(description="Number of predictions returned")
    predictions: list[Prediction]


class Health(BaseModel):
    """Liveness and readiness in one small object."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_version: str
