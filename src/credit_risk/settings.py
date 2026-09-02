"""Paths, column contracts, and runtime settings shared by every module.

Nothing here reads a secret. The service needs none at runtime.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Repo root: CREDIT_RISK_HOME if set, else the first ancestor holding params.yaml, else cwd."""
    env = os.environ.get("CREDIT_RISK_HOME")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "params.yaml").exists():
            return parent
    return Path.cwd().resolve()


REPO_ROOT: Path = _find_repo_root()
PARAMS_PATH = REPO_ROOT / "params.yaml"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = REPO_ROOT / "configs"

RAW_ZIP_PATH = RAW_DIR / "uci_350.zip"
RAW_XLS_PATH = RAW_DIR / "default of credit card clients.xls"
RAW_CSV_PATH = RAW_DIR / "credit_default_raw.csv"
VALIDATED_CSV_PATH = PROCESSED_DIR / "validated.csv"
FEATURES_CSV_PATH = PROCESSED_DIR / "features.csv"
SPLIT_DIR = PROCESSED_DIR / "splits"
TRAIN_CSV_PATH = SPLIT_DIR / "train.csv"
VAL_CSV_PATH = SPLIT_DIR / "val.csv"
TEST_CSV_PATH = SPLIT_DIR / "test.csv"
SPLIT_MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"

MODEL_PATH = MODELS_DIR / "model.joblib"
VERSION_PATH = MODELS_DIR / "version.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
IMPORTANCE_PATH = REPORTS_DIR / "importance.json"
FAIRNESS_PATH = REPORTS_DIR / "fairness.json"
LOADTEST_PATH = REPORTS_DIR / "loadtest.json"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_BASELINE_PATH = REPORTS_DIR / "drift_baseline.json"
PRESETS_PATH = CONFIGS_DIR / "presets.json"

# ---------------------------------------------------------------------------
# Column contracts (UCI dataset id 350, Yeh and Lien 2009)
# ---------------------------------------------------------------------------

TARGET = "default_next_month"
RAW_TARGET_NAME = "default payment next month"
ID_COLUMN = "ID"

PROTECTED_COLUMNS: tuple[str, ...] = ("SEX", "MARRIAGE")
PAY_STATUS_COLUMNS: tuple[str, ...] = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
BILL_COLUMNS: tuple[str, ...] = tuple(f"BILL_AMT{i}" for i in range(1, 7))
PAY_AMT_COLUMNS: tuple[str, ...] = tuple(f"PAY_AMT{i}" for i in range(1, 7))

# The 23 attributes exactly as the dataset documents them, in dataset order.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_COLUMNS,
    *PAY_AMT_COLUMNS,
)

# Raw attributes the model is allowed to see (protected attributes removed).
MODEL_INPUT_COLUMNS: tuple[str, ...] = tuple(
    c for c in RAW_FEATURE_COLUMNS if c not in PROTECTED_COLUMNS
)

# Engineered features produced by credit_risk.features.build_features, in output order.
ENGINEERED_COLUMNS: tuple[str, ...] = (
    *(f"util_{i}" for i in range(1, 7)),
    "util_mean",
    "util_max",
    *(f"pay_ratio_{i}" for i in range(1, 7)),
    "pay_ratio_mean",
    "delinq_max",
    "delinq_mean",
    "delinq_months",
    "delinq_recent",
    "bill_trend",
    "pay_trend",
    "bill_mean",
    "pay_amt_mean",
    "zero_pay_months",
)

MODEL_FEATURE_COLUMNS: tuple[str, ...] = (*MODEL_INPUT_COLUMNS, *ENGINEERED_COLUMNS)

# Value ranges that the data card documents and the API schema mirrors.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (1.0, 2_000_000.0),
    "EDUCATION": (1, 4),
    "AGE": (18, 100),
    **{c: (-2, 9) for c in PAY_STATUS_COLUMNS},
    **{c: (-2_000_000.0, 2_000_000.0) for c in BILL_COLUMNS},
    **{c: (0.0, 2_000_000.0) for c in PAY_AMT_COLUMNS},
}

DISCLAIMER = (
    "Demonstration system trained on the 2005 Taiwan credit-card dataset (UCI id 350). "
    "Not a U.S. underwriting model. Do not use it for real credit decisions."
)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def load_params(path: Path | None = None) -> dict:
    """Load params.yaml as a plain dict."""
    with open(path or PARAMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Runtime settings for the API
# ---------------------------------------------------------------------------


DEFAULT_PORT = 8000
LOG_LEVEL_NAMES: tuple[str, ...] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def api_port() -> int:
    """PORT from the environment, default 8000.

    A blank or non-numeric value falls back to the default with a warning. A number outside
    1..65535 raises ValueError with a one-line message.
    """
    raw = os.environ.get("PORT", "").strip()
    if raw == "":
        if "PORT" in os.environ:
            log.warning("PORT is empty, using %d", DEFAULT_PORT)
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        log.warning("PORT=%r is not an integer, using %d", raw, DEFAULT_PORT)
        return DEFAULT_PORT
    if not 1 <= port <= 65535:
        raise ValueError(f"PORT must be an integer between 1 and 65535, got {port}")
    return port


def log_level() -> str:
    """LOG_LEVEL from the environment as a logging level name, default INFO.

    Case does not matter. A blank value is INFO; an unknown name is INFO with a warning.
    """
    raw = os.environ.get("LOG_LEVEL", "").strip().upper()
    if raw == "":
        return "INFO"
    if raw in LOG_LEVEL_NAMES:
        return raw
    log.warning("LOG_LEVEL=%r is not a logging level, using INFO", raw)
    return "INFO"


def prediction_log_path() -> Path:
    """Where the API appends one JSON line per prediction. Ephemeral on free hosting."""
    env = os.environ.get("PREDICTION_LOG_PATH")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "credit_risk_predictions.jsonl"
