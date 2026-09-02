"""Batch scoring from the command line.

Usage:
    python -m credit_risk.predict --input rows.csv --output scored.csv

The input CSV holds the 21 raw model input columns; header case does not matter and extra
columns such as ID pass through untouched. The output is the input plus `probability` and
`decision`, from the same model and threshold the API uses.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk import settings
from credit_risk.api.model_loader import ModelBundle, load_bundle

log = logging.getLogger(__name__)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map a case-insensitive header onto the uppercase model input names."""
    wanted = {column.lower(): column for column in settings.MODEL_INPUT_COLUMNS}
    mapping = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key in wanted:
            mapping[column] = wanted[key]
    renamed = df.rename(columns=mapping)
    duplicates = sorted(set(renamed.columns[renamed.columns.duplicated()].tolist()))
    if duplicates:
        raise ValueError(f"input has duplicate model columns: {duplicates}")
    missing = [column for column in settings.MODEL_INPUT_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"input is missing model columns: {missing}")
    return renamed


def score_frame(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """The input frame, headers untouched, plus probability and decision columns."""
    inputs = normalise_columns(df)[list(settings.MODEL_INPUT_COLUMNS)]
    nulls = inputs.columns[inputs.isna().any()].tolist()
    if nulls:
        raise ValueError(f"null values in model columns: {nulls}")
    inputs = inputs.apply(pd.to_numeric)
    probabilities = bundle.predict_proba(inputs)
    out = df.copy()
    out["probability"] = np.round(probabilities, 6)
    out["decision"] = np.where(
        probabilities >= bundle.threshold, "likely_default", "unlikely_default"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m credit_risk.predict",
        description="Score a CSV of applicants with the shipped model.",
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV with the raw input columns")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the scored CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold; default is threshold_cost_optimal from reports/metrics.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_csv(args.input)
    if df.empty:
        print(f"error: {args.input} has no rows", file=sys.stderr)
        return 1
    bundle = load_bundle()
    if args.threshold is not None:
        bundle.threshold = float(args.threshold)
    try:
        scored = score_frame(df, bundle)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False, lineterminator="\n")
    flagged = int((scored["decision"] == "likely_default").sum())
    print(
        f"scored {len(scored)} rows, mean probability {scored['probability'].mean():.4f}, "
        f"{flagged} likely_default at threshold {bundle.threshold:g}, wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
