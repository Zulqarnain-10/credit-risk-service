"""Schema validation for the raw CSV and collapse of undocumented category codes.

Usage:
    python -m credit_risk.validate

Reads data/raw/credit_default_raw.csv, enforces RAW_SCHEMA (types, ranges, unique ID, no nulls),
collapses EDUCATION {0, 5, 6} -> 4 and MARRIAGE {0} -> 3 per params.yaml, and writes
data/processed/validated.csv with the same columns.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pandera.pandas as pa

from credit_risk import settings
from credit_risk.data import sha256_of

log = logging.getLogger(__name__)


def _numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _integral(series: pd.Series) -> bool:
    values = series.dropna()
    if not _numeric_dtype(values):
        return False
    return bool(np.isfinite(values).all() and (values == np.round(values)).all())


NUMERIC = pa.Check(_numeric_dtype, error="numeric dtype required")
INTEGRAL = pa.Check(_integral, error="whole numbers required")


def _int_column(low: float, high: float, **kwargs) -> pa.Column:
    """Integer-coded column. Any numeric dtype is accepted as long as values are whole."""
    return pa.Column(checks=[INTEGRAL, pa.Check.in_range(low, high)], **kwargs)


def _amount_column(low: float, high: float) -> pa.Column:
    return pa.Column(checks=[NUMERIC, pa.Check.in_range(low, high)])


def _raw_columns() -> dict[str, pa.Column]:
    ranges = settings.FIELD_RANGES
    columns: dict[str, pa.Column] = {
        settings.ID_COLUMN: pa.Column(checks=[INTEGRAL, pa.Check.ge(1)], unique=True),
        "LIMIT_BAL": _amount_column(*ranges["LIMIT_BAL"]),
        "SEX": _int_column(1, 2),
        "EDUCATION": _int_column(0, 6),
        "MARRIAGE": _int_column(0, 3),
        "AGE": _int_column(*ranges["AGE"]),
    }
    for name in settings.PAY_STATUS_COLUMNS:
        columns[name] = _int_column(*ranges[name])
    for name in settings.BILL_COLUMNS:
        columns[name] = _amount_column(*ranges[name])
    for name in settings.PAY_AMT_COLUMNS:
        columns[name] = _amount_column(*ranges[name])
    columns[settings.TARGET] = _int_column(0, 1)
    return columns


RAW_SCHEMA = pa.DataFrameSchema(
    _raw_columns(),
    strict=True,
    ordered=False,
    coerce=False,
    name="credit_default_raw",
    description="Raw UCI id 350 frame: ID, 23 documented attributes, binary target.",
)

# After collapse_categories: only the documented category codes remain.
VALIDATED_SCHEMA = RAW_SCHEMA.update_columns(
    {
        "EDUCATION": {"checks": [INTEGRAL, pa.Check.in_range(1, 4)]},
        "MARRIAGE": {"checks": [INTEGRAL, pa.Check.in_range(1, 3)]},
    }
)


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw frame against RAW_SCHEMA. Raises pandera.errors.SchemaError on failure."""
    try:
        return RAW_SCHEMA.validate(df, lazy=False)
    except pa.errors.SchemaErrors as exc:
        # Column-level failures (for example an unexpected column) arrive as SchemaErrors even
        # in non-lazy mode; surface one error type to callers.
        raise pa.errors.SchemaError(RAW_SCHEMA, df, str(exc)) from exc


def collapse_categories(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Map undocumented EDUCATION and MARRIAGE codes onto the documented 'other' buckets."""
    feat = params["features"]
    out = df.copy()
    edu_codes = list(feat["education_other_codes"])
    mar_codes = list(feat["marriage_other_codes"])
    out.loc[out["EDUCATION"].isin(edu_codes), "EDUCATION"] = int(feat["education_other_value"])
    out.loc[out["MARRIAGE"].isin(mar_codes), "MARRIAGE"] = int(feat["marriage_other_value"])
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = settings.load_params()
    if settings.RAW_ZIP_PATH.exists():
        observed = sha256_of(settings.RAW_ZIP_PATH)
        expected = params["data"]["zip_sha256"]
        if observed != expected:
            raise RuntimeError(f"zip sha256 mismatch: expected {expected}, got {observed}")
        log.info("zip sha256 verified: %s", observed)
    else:
        log.warning("archive not present at %s; skipping zip hash check", settings.RAW_ZIP_PATH)

    df = pd.read_csv(settings.RAW_CSV_PATH)
    validated = validate_frame(df)
    log.info("schema ok: %d rows, %d columns", *validated.shape)
    before = {
        "EDUCATION": validated["EDUCATION"].value_counts().sort_index().to_dict(),
        "MARRIAGE": validated["MARRIAGE"].value_counts().sort_index().to_dict(),
    }
    collapsed = collapse_categories(validated, params)
    VALIDATED_SCHEMA.validate(collapsed, lazy=False)
    n_edu = int(validated["EDUCATION"].isin(params["features"]["education_other_codes"]).sum())
    n_mar = int(validated["MARRIAGE"].isin(params["features"]["marriage_other_codes"]).sum())
    log.info("collapsed EDUCATION codes on %d rows, MARRIAGE codes on %d rows", n_edu, n_mar)
    log.info("category counts before collapse: %s", before)

    settings.VALIDATED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(settings.VALIDATED_CSV_PATH, index=False, lineterminator="\n")
    print(
        f"wrote {settings.VALIDATED_CSV_PATH} rows={collapsed.shape[0]} "
        f"cols={collapsed.shape[1]} education_collapsed={n_edu} marriage_collapsed={n_mar}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
