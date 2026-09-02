"""Feature engineering: utilization, payment ratios, delinquency, and trends.

Usage:
    python -m credit_risk.features

build_features is a pure, vectorised function with no fitted state. FeatureBuilder wraps it as
a scikit-learn transformer so the whole model ships as one Pipeline that accepts raw columns.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk import settings

log = logging.getLogger(__name__)

PAY_RATIO_CAP = 5.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with exactly settings.MODEL_FEATURE_COLUMNS, in order.

    Requires the 21 settings.MODEL_INPUT_COLUMNS; extra columns are ignored. Output has no
    NaN or inf. Definitions (k = 1..6):
      util_k         = BILL_AMTk / LIMIT_BAL (0 when LIMIT_BAL <= 0)
      pay_ratio_k    = min(PAY_AMTk / BILL_AMTk, 5.0) when BILL_AMTk > 0 else 1.0
      delinq_*       = statistics of clip(PAY_*, 0) over the six status columns
      bill_trend     = BILL_AMT1 - BILL_AMT6, pay_trend = PAY_AMT1 - PAY_AMT6
      zero_pay_months = count of PAY_AMT* equal to 0
    """
    missing = [c for c in settings.MODEL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing model input columns: {missing}")

    raw = df[list(settings.MODEL_INPUT_COLUMNS)]
    out = raw.copy()

    limit = raw["LIMIT_BAL"].astype(float)
    bills = raw[list(settings.BILL_COLUMNS)].astype(float)
    pays = raw[list(settings.PAY_AMT_COLUMNS)].astype(float)
    status = raw[list(settings.PAY_STATUS_COLUMNS)].astype(float)

    safe_limit = limit.where(limit > 0, other=np.nan)
    utils = bills.div(safe_limit, axis=0).fillna(0.0)
    for k in range(1, 7):
        out[f"util_{k}"] = utils[f"BILL_AMT{k}"].to_numpy()
    out["util_mean"] = utils.mean(axis=1).to_numpy()
    out["util_max"] = utils.max(axis=1).to_numpy()

    positive_bill = bills > 0
    ratios = pays.to_numpy() / bills.where(positive_bill, other=np.nan).to_numpy()
    ratios = np.where(positive_bill.to_numpy(), np.minimum(ratios, PAY_RATIO_CAP), 1.0)
    ratios = np.nan_to_num(ratios, nan=1.0, posinf=PAY_RATIO_CAP, neginf=0.0)
    for k in range(1, 7):
        out[f"pay_ratio_{k}"] = ratios[:, k - 1]
    out["pay_ratio_mean"] = ratios.mean(axis=1)

    delinq = status.clip(lower=0)
    out["delinq_max"] = delinq.max(axis=1).astype(int).to_numpy()
    out["delinq_mean"] = delinq.mean(axis=1).to_numpy()
    out["delinq_months"] = (status > 0).sum(axis=1).astype(int).to_numpy()
    out["delinq_recent"] = (raw["PAY_0"] > 0).astype(int).to_numpy()

    out["bill_trend"] = (raw["BILL_AMT1"] - raw["BILL_AMT6"]).to_numpy()
    out["pay_trend"] = (raw["PAY_AMT1"] - raw["PAY_AMT6"]).to_numpy()
    out["bill_mean"] = bills.mean(axis=1).to_numpy()
    out["pay_amt_mean"] = pays.mean(axis=1).to_numpy()
    out["zero_pay_months"] = (pays == 0).sum(axis=1).astype(int).to_numpy()

    out = out[list(settings.MODEL_FEATURE_COLUMNS)]
    engineered = out[list(settings.ENGINEERED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(engineered).all():
        raise ValueError("non-finite value in engineered features")
    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless transformer: raw model inputs in, MODEL_FEATURE_COLUMNS out."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=list(settings.MODEL_INPUT_COLUMNS))
        return build_features(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(settings.MODEL_FEATURE_COLUMNS, dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_csv(settings.VALIDATED_CSV_PATH)
    engineered = build_features(df)[list(settings.ENGINEERED_COLUMNS)]
    out = pd.concat([df.reset_index(drop=True), engineered.reset_index(drop=True)], axis=1)
    settings.FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(settings.FEATURES_CSV_PATH, index=False, lineterminator="\n")
    print(
        f"wrote {settings.FEATURES_CSV_PATH} rows={out.shape[0]} cols={out.shape[1]} "
        f"engineered={len(settings.ENGINEERED_COLUMNS)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
