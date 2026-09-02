"""Contract tests for credit_risk.validate: the pandera schema and the category collapse.

The valid sample is the first 200 rows of the raw CSV when the fetch stage has run, and a
seeded synthetic frame with the same columns otherwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from credit_risk import settings
from credit_risk.validate import RAW_SCHEMA, VALIDATED_SCHEMA, collapse_categories, validate_frame

RAW_COLUMNS = [settings.ID_COLUMN, *settings.RAW_FEATURE_COLUMNS, settings.TARGET]
SCHEMA_FAILURE = (SchemaError, SchemaErrors)


def synthetic_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A valid raw frame, including the undocumented EDUCATION and MARRIAGE codes."""
    rng = np.random.default_rng(seed)
    frame: dict[str, np.ndarray] = {
        settings.ID_COLUMN: np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 80, n),
    }
    for c in settings.PAY_STATUS_COLUMNS:
        frame[c] = rng.integers(-2, 9, n)
    for c in settings.BILL_COLUMNS:
        frame[c] = rng.integers(-5_000, 300_001, n)
    for c in settings.PAY_AMT_COLUMNS:
        frame[c] = rng.integers(0, 60_001, n)
    frame[settings.TARGET] = rng.integers(0, 2, n)
    return pd.DataFrame(frame, columns=RAW_COLUMNS)


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    if settings.RAW_CSV_PATH.exists():
        return pd.read_csv(settings.RAW_CSV_PATH, nrows=200)
    return synthetic_raw(200)


@pytest.fixture
def raw_copy(raw_sample) -> pd.DataFrame:
    return raw_sample.copy()


def test_valid_sample_passes(raw_sample):
    validated = validate_frame(raw_sample)
    assert list(validated.columns) == RAW_COLUMNS
    assert len(validated) == len(raw_sample)
    assert int(validated.isna().sum().sum()) == 0


def test_synthetic_frame_passes():
    validate_frame(synthetic_raw(50, seed=1))


def test_schema_covers_every_raw_column():
    assert set(RAW_SCHEMA.columns) == set(RAW_COLUMNS)
    assert RAW_SCHEMA.strict is True
    assert RAW_SCHEMA.columns[settings.ID_COLUMN].unique is True
    assert all(not col.nullable for col in RAW_SCHEMA.columns.values())


def _null_age(df: pd.DataFrame) -> pd.DataFrame:
    df["AGE"] = df["AGE"].astype(float)
    df.loc[df.index[0], "AGE"] = np.nan
    return df


def _age_five(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0], "AGE"] = 5
    return df


def _limit_bal_negative(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0], "LIMIT_BAL"] = -1
    return df


def _pay_0_twelve(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0], "PAY_0"] = 12
    return df


def _duplicate_id(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[1], settings.ID_COLUMN] = df.loc[df.index[0], settings.ID_COLUMN]
    return df


def _missing_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["BILL_AMT3"])


def _target_two(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0], settings.TARGET] = 2
    return df


def _extra_column(df: pd.DataFrame) -> pd.DataFrame:
    df["notes"] = "unexpected"
    return df


def _fractional_code(df: pd.DataFrame) -> pd.DataFrame:
    df["EDUCATION"] = df["EDUCATION"].astype(float)
    df.loc[df.index[0], "EDUCATION"] = 1.5
    return df


INVALID_CASES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "null_age": _null_age,
    "age_five": _age_five,
    "limit_bal_negative": _limit_bal_negative,
    "pay_0_twelve": _pay_0_twelve,
    "duplicate_id": _duplicate_id,
    "missing_column": _missing_column,
    "target_two": _target_two,
    "extra_column": _extra_column,
    "fractional_code": _fractional_code,
}


@pytest.mark.parametrize("name", sorted(INVALID_CASES))
def test_invalid_frame_is_rejected(raw_copy, name):
    broken = INVALID_CASES[name](raw_copy)
    with pytest.raises(SCHEMA_FAILURE):
        validate_frame(broken)


def test_collapse_categories_maps_undocumented_codes():
    params = settings.load_params()
    frame = pd.DataFrame(
        {
            "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
            "MARRIAGE": [0, 1, 2, 3, 0, 1, 2],
            "AGE": [30] * 7,
        }
    )
    before = frame.copy()
    out = collapse_categories(frame, params)
    assert out["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3, 3, 1, 2]
    assert out["AGE"].tolist() == [30] * 7
    assert out is not frame
    pd.testing.assert_frame_equal(frame, before)


def test_collapsed_sample_satisfies_validated_schema(raw_sample):
    params = settings.load_params()
    collapsed = collapse_categories(validate_frame(raw_sample), params)
    VALIDATED_SCHEMA.validate(collapsed, lazy=False)
    assert set(collapsed["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(collapsed["MARRIAGE"].unique()) <= {1, 2, 3}
    assert list(collapsed.columns) == RAW_COLUMNS
