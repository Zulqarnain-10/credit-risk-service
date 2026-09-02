"""Three real held-out applicants (low risk, borderline, high risk) for the demo page.

Usage:
    python -m credit_risk.presets

Rows come from the test split: the one closest to the 10th percentile of predicted
probability, the one closest to the cost-optimal threshold, and the one closest to the
95th percentile. Nothing is invented; the descriptions only restate each row's pattern, and
the borderline one adds where that row sits against the decision threshold.
"""

from __future__ import annotations

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from credit_risk import settings
from credit_risk.features import build_features

log = logging.getLogger(__name__)

API_FIELDS: tuple[str, ...] = tuple(c.lower() for c in settings.MODEL_INPUT_COLUMNS)


def describe(feats: pd.Series) -> str:
    """One plain sentence about the applicant's repayment pattern, without numbers."""
    months = int(feats["delinq_months"])
    recent = bool(feats["delinq_recent"])
    if months == 0:
        late = "has no late payments across the six months on record"
    elif recent and months >= 3:
        late = "was behind on payments in most of the six months on record, including the latest"
    elif recent:
        late = "was late on the most recent statement"
    else:
        late = "had an earlier late payment but is current on the latest statement"

    util = float(feats["util_mean"])
    if util < 0.2:
        use = "uses a small share of the credit limit"
    elif util < 0.7:
        use = "carries a moderate balance against the limit"
    else:
        use = "runs at or above the credit limit"

    ratio = float(feats["pay_ratio_mean"])
    if ratio >= 0.9:
        pay = "pays bills in full"
    elif ratio >= 0.1:
        pay = "pays part of each bill"
    else:
        pay = "pays little of each bill"
    return f"Cardholder who {use}, {pay}, and {late}."


def threshold_note(p: float, threshold: float, digits: int = 4) -> str:
    """Second sentence for the borderline row: where its probability sits against the threshold."""
    if round(float(p), digits) == round(float(threshold), digits):
        return (
            "The applicant sits at the decision threshold; the model's probability rounds "
            "to the threshold."
        )
    return "The applicant sits closest to the decision threshold among the held-out rows."


def _scalar(value) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else round(number, 4)


def pick_rows(p: np.ndarray, threshold: float) -> dict[str, int]:
    """Index of the row closest to each target probability; rows are never reused."""
    targets = {
        "low_risk": float(np.quantile(p, 0.10)),
        "borderline": float(threshold),
        "high_risk": float(np.quantile(p, 0.95)),
    }
    chosen: dict[str, int] = {}
    for key, target in targets.items():
        order = np.argsort(np.abs(p - target), kind="stable")
        for idx in order:
            if int(idx) not in chosen.values():
                chosen[key] = int(idx)
                break
    return chosen


def build_presets(test_df: pd.DataFrame, model, threshold: float, model_version: str) -> dict:
    inputs = list(settings.MODEL_INPUT_COLUMNS)
    p = model.predict_proba(test_df[inputs])[:, 1]
    feats = build_features(test_df)
    labels = {"low_risk": "Low risk", "borderline": "Borderline", "high_risk": "High risk"}
    presets = []
    for key, idx in pick_rows(p, threshold).items():
        row = test_df.iloc[idx]
        description = describe(feats.iloc[idx])
        if key == "borderline":
            description += " " + threshold_note(float(p[idx]), threshold)
        presets.append(
            {
                "id": key,
                "label": labels[key],
                "description": description,
                "source_row_id": int(row[settings.ID_COLUMN]),
                "model_probability": round(float(p[idx]), 6),
                "input": {
                    field: _scalar(row[col]) for field, col in zip(API_FIELDS, inputs, strict=True)
                },
            }
        )
    return {"generated_from": f"test split, model {model_version}", "presets": presets}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with open(settings.METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)
    with open(settings.VERSION_PATH, encoding="utf-8") as fh:
        version = json.load(fh)
    model = joblib.load(settings.MODEL_PATH)
    test_df = pd.read_csv(settings.TEST_CSV_PATH)
    out = build_presets(
        test_df, model, float(metrics["threshold_cost_optimal"]), version["model_version"]
    )
    settings.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.PRESETS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    summary = {x["id"]: (x["source_row_id"], x["model_probability"]) for x in out["presets"]}
    print(f"wrote {settings.PRESETS_PATH} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
