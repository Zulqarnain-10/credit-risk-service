"""Contract tests for credit_risk.features: build_features and FeatureBuilder.

Every check is pure: no model, no fitted state, no network. The sample frame comes from the
train split when it exists and from a seeded generator otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk import settings
from credit_risk.features import PAY_RATIO_CAP, FeatureBuilder, build_features

INPUTS = list(settings.MODEL_INPUT_COLUMNS)
PAY_STATUS = list(settings.PAY_STATUS_COLUMNS)
BILLS = list(settings.BILL_COLUMNS)
PAY_AMTS = list(settings.PAY_AMT_COLUMNS)

BASE_ROW: dict[str, float] = {
    "LIMIT_BAL": 50_000.0,
    "EDUCATION": 2,
    "AGE": 35,
    **dict.fromkeys(PAY_STATUS, 0),
    **dict.fromkeys(BILLS, 10_000.0),
    **dict.fromkeys(PAY_AMTS, 1_000.0),
}

EDGE_ROWS: dict[str, dict[str, float]] = {
    "limit_bal_minimum": {
        "LIMIT_BAL": settings.FIELD_RANGES["LIMIT_BAL"][0],
        **dict.fromkeys(BILLS, 500_000.0),
    },
    "limit_bal_maximum": {"LIMIT_BAL": settings.FIELD_RANGES["LIMIT_BAL"][1]},
    "zero_bills": dict.fromkeys(BILLS, 0.0),
    "negative_bills": dict.fromkeys(BILLS, -2_500.0),
    "zero_payments": dict.fromkeys(PAY_AMTS, 0.0),
    "zero_bills_and_payments": {**dict.fromkeys(BILLS, 0.0), **dict.fromkeys(PAY_AMTS, 0.0)},
    "pay_status_all_minus_two": dict.fromkeys(PAY_STATUS, -2),
    "pay_status_all_eight": dict.fromkeys(PAY_STATUS, 8),
}


def make_row(**overrides) -> pd.DataFrame:
    """One valid input row with the given overrides applied."""
    return pd.DataFrame([{**BASE_ROW, **overrides}], columns=INPUTS)


def synthetic_inputs(n: int, seed: int = 0) -> pd.DataFrame:
    """Plausible raw applicants covering the documented value ranges."""
    rng = np.random.default_rng(seed)
    frame: dict[str, np.ndarray] = {
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000.0,
        "EDUCATION": rng.integers(1, 5, n),
        "AGE": rng.integers(21, 70, n),
    }
    for c in PAY_STATUS:
        frame[c] = rng.integers(-2, 9, n)
    for c in BILLS:
        frame[c] = rng.integers(-5_000, 200_001, n).astype(float)
    for c in PAY_AMTS:
        frame[c] = rng.integers(0, 50_001, n).astype(float)
    return pd.DataFrame(frame, columns=INPUTS)


@pytest.fixture(scope="module")
def sample_inputs() -> pd.DataFrame:
    if settings.TRAIN_CSV_PATH.exists():
        return pd.read_csv(settings.TRAIN_CSV_PATH, nrows=200)[INPUTS]
    return synthetic_inputs(200)


def test_output_columns_match_contract(sample_inputs):
    out = build_features(sample_inputs)
    assert list(out.columns) == list(settings.MODEL_FEATURE_COLUMNS)
    assert list(out.columns[: len(INPUTS)]) == INPUTS
    assert len(out) == len(sample_inputs)
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_deterministic_and_input_untouched(sample_inputs):
    before = sample_inputs.copy()
    first = build_features(sample_inputs)
    second = build_features(sample_inputs)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(sample_inputs, before)


@pytest.mark.parametrize("name", sorted(EDGE_ROWS))
def test_edge_rows_are_finite(name):
    out = build_features(make_row(**EDGE_ROWS[name]))
    assert list(out.columns) == list(settings.MODEL_FEATURE_COLUMNS)
    values = out.to_numpy(dtype=float)
    assert not np.isnan(values).any(), f"NaN in edge row {name}"
    assert np.isfinite(values).all(), f"inf in edge row {name}"


def test_zero_and_negative_bills_mean_fully_paid():
    zero = build_features(make_row(**dict.fromkeys(BILLS, 0.0)))
    negative = build_features(make_row(**dict.fromkeys(BILLS, -2_500.0)))
    for k in range(1, 7):
        assert zero[f"pay_ratio_{k}"].iloc[0] == 1.0
        assert negative[f"pay_ratio_{k}"].iloc[0] == 1.0
        assert zero[f"util_{k}"].iloc[0] == 0.0
        assert negative[f"util_{k}"].iloc[0] == -2_500.0 / BASE_ROW["LIMIT_BAL"]
    assert zero["pay_ratio_mean"].iloc[0] == 1.0
    assert negative["pay_ratio_mean"].iloc[0] == 1.0
    assert negative["util_max"].iloc[0] < 0


def test_pay_ratio_cap_and_exact_values():
    assert PAY_RATIO_CAP == 5.0
    row = make_row(
        BILL_AMT1=100.0,
        PAY_AMT1=1_000.0,  # ten times the bill: capped
        BILL_AMT2=0.0,
        PAY_AMT2=500.0,  # no bill: fully paid
        BILL_AMT3=-50.0,
        PAY_AMT3=0.0,  # credit balance: fully paid
        BILL_AMT4=200.0,
        PAY_AMT4=50.0,  # a quarter of the bill
        BILL_AMT5=200.0,
        PAY_AMT5=0.0,  # nothing paid on a real bill
        BILL_AMT6=400.0,
        PAY_AMT6=2_000.0,  # exactly the cap
    )
    out = build_features(row)
    expected = [PAY_RATIO_CAP, 1.0, 1.0, 0.25, 0.0, PAY_RATIO_CAP]
    got = [out[f"pay_ratio_{k}"].iloc[0] for k in range(1, 7)]
    assert got == expected
    assert out["pay_ratio_mean"].iloc[0] == pytest.approx(sum(expected) / 6)


def test_utilization_is_bill_over_limit(sample_inputs):
    out = build_features(sample_inputs)
    limit = sample_inputs["LIMIT_BAL"].to_numpy(dtype=float)
    assert (limit > 0).all()
    per_month = []
    for k in range(1, 7):
        expected = sample_inputs[f"BILL_AMT{k}"].to_numpy(dtype=float) / limit
        np.testing.assert_array_equal(out[f"util_{k}"].to_numpy(), expected)
        per_month.append(expected)
    stacked = np.column_stack(per_month)
    np.testing.assert_allclose(out["util_mean"].to_numpy(), stacked.mean(axis=1), atol=1e-12)
    np.testing.assert_array_equal(out["util_max"].to_numpy(), stacked.max(axis=1))


def test_delinquency_counts_only_positive_statuses():
    mixed = build_features(make_row(**dict(zip(PAY_STATUS, [-2, -1, 0, 1, 3, 8], strict=True))))
    assert mixed["delinq_months"].iloc[0] == 3
    assert mixed["delinq_max"].iloc[0] == 8
    assert mixed["delinq_mean"].iloc[0] == pytest.approx((1 + 3 + 8) / 6)
    assert mixed["delinq_recent"].iloc[0] == 0

    recent = build_features(make_row(PAY_0=2))
    assert recent["delinq_recent"].iloc[0] == 1
    assert recent["delinq_months"].iloc[0] == 1
    assert recent["delinq_max"].iloc[0] == 2

    clean = build_features(make_row(**dict.fromkeys(PAY_STATUS, -2)))
    assert clean["delinq_months"].iloc[0] == 0
    assert clean["delinq_max"].iloc[0] == 0
    assert clean["delinq_mean"].iloc[0] == 0.0
    assert clean["delinq_recent"].iloc[0] == 0

    worst = build_features(make_row(**dict.fromkeys(PAY_STATUS, 8)))
    assert worst["delinq_months"].iloc[0] == 6
    assert worst["delinq_max"].iloc[0] == 8
    assert worst["delinq_mean"].iloc[0] == 8.0
    assert worst["delinq_recent"].iloc[0] == 1


def test_trends_means_and_zero_pay_months():
    bills = dict(zip(BILLS, [6_000.0, 5_000.0, 4_000.0, 3_000.0, 2_000.0, 1_000.0], strict=True))
    pays = dict(zip(PAY_AMTS, [0.0, 300.0, 0.0, 200.0, 0.0, 100.0], strict=True))
    out = build_features(make_row(**bills, **pays))
    assert out["bill_trend"].iloc[0] == 5_000.0
    assert out["pay_trend"].iloc[0] == -100.0
    assert out["bill_mean"].iloc[0] == 3_500.0
    assert out["pay_amt_mean"].iloc[0] == 100.0
    assert out["zero_pay_months"].iloc[0] == 3


def test_batch_matches_row_by_row(sample_inputs):
    head = sample_inputs.head(20)
    batch = build_features(head).reset_index(drop=True)
    rows = pd.concat([build_features(head.iloc[[i]]) for i in range(len(head))])
    pd.testing.assert_frame_equal(batch, rows.reset_index(drop=True))


def test_feature_builder_matches_build_features(sample_inputs):
    builder = FeatureBuilder()
    assert builder.fit(sample_inputs) is builder
    expected = build_features(sample_inputs)
    pd.testing.assert_frame_equal(builder.transform(sample_inputs), expected)
    pd.testing.assert_frame_equal(builder.fit_transform(sample_inputs), expected)
    assert list(builder.get_feature_names_out()) == list(settings.MODEL_FEATURE_COLUMNS)


def test_extra_columns_are_ignored(sample_inputs):
    with_extras = sample_inputs.copy()
    with_extras.insert(0, settings.ID_COLUMN, np.arange(1, len(with_extras) + 1))
    with_extras["SEX"] = 2
    with_extras["MARRIAGE"] = 1
    with_extras[settings.TARGET] = 0
    out = build_features(with_extras)
    assert list(out.columns) == list(settings.MODEL_FEATURE_COLUMNS)
    for col in (settings.ID_COLUMN, settings.TARGET, *settings.PROTECTED_COLUMNS):
        assert col not in out.columns
    pd.testing.assert_frame_equal(out, build_features(sample_inputs))


def test_shuffled_input_columns_give_same_output(sample_inputs):
    shuffled = sample_inputs[INPUTS[::-1]]
    pd.testing.assert_frame_equal(build_features(shuffled), build_features(sample_inputs))


def test_missing_input_column_raises():
    with pytest.raises(ValueError, match="missing model input columns"):
        build_features(make_row().drop(columns=["PAY_AMT6"]))
