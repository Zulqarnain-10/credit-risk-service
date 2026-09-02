"""Contract tests for the shipped model and its receipts.

Loads models/model.joblib, models/version.json, and reports/metrics.json. The scoring tests
re-measure the headline numbers on the held-out test split and skip when that DVC-managed
file is absent (for example in CI before dvc repro).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from credit_risk import settings
from credit_risk.features import FeatureBuilder

INPUTS = list(settings.MODEL_INPUT_COLUMNS)
METRIC_TOLERANCE = 0.002

VERSION_KEYS = (
    "model",
    "package_version",
    "model_version",
    "git_sha",
    "git_sha_short",
    "git_dirty",
    "mlflow_experiment",
    "mlflow_run_id",
    "baseline_mlflow_run_id",
    "registered",
    "registered_model",
    "data_sha256",
    "trained_at",
    "n_train",
    "n_val",
    "n_features",
    "feature_columns",
    "hgb_n_iter",
    "libraries",
)
LIBRARY_KEYS = ("python", "scikit-learn", "numpy", "pandas", "mlflow", "joblib")
METRICS_KEYS = (
    "model",
    "trained_at",
    "git_sha",
    "mlflow_run_id",
    "data_sha256",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "positive_rate_test",
    "roc_auc",
    "pr_auc",
    "brier",
    "ks",
    "validation",
    "threshold_cost_optimal",
    "threshold_precision_target",
    "precision_target_met",
    "at_threshold",
    "at_precision_target",
    "threshold_selection",
    "calibration",
    "baseline_logreg",
    "lift_over_baseline",
)
THRESHOLD_BLOCK_KEYS = (
    "threshold",
    "precision",
    "recall",
    "f1",
    "selection_rate",
    "tn",
    "fp",
    "fn",
    "tp",
    "expected_cost_per_1000",
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        pytest.skip(f"{path} is absent; {hint}")


def synthetic_inputs(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame: dict[str, np.ndarray] = {
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000.0,
        "EDUCATION": rng.integers(1, 5, n),
        "AGE": rng.integers(21, 70, n),
    }
    for c in settings.PAY_STATUS_COLUMNS:
        frame[c] = rng.integers(-2, 9, n)
    for c in settings.BILL_COLUMNS:
        frame[c] = rng.integers(-5_000, 200_001, n).astype(float)
    for c in settings.PAY_AMT_COLUMNS:
        frame[c] = rng.integers(0, 50_001, n).astype(float)
    return pd.DataFrame(frame, columns=INPUTS)


def _confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return tn, fp, fn, tp


@pytest.fixture(scope="module")
def model() -> Pipeline:
    _require(settings.MODEL_PATH, "run the train stage first")
    return joblib.load(settings.MODEL_PATH)


@pytest.fixture(scope="module")
def version() -> dict:
    _require(settings.VERSION_PATH, "run the train stage first")
    return _load_json(settings.VERSION_PATH)


@pytest.fixture(scope="module")
def metrics() -> dict:
    _require(settings.METRICS_PATH, "run the evaluate stage first")
    return _load_json(settings.METRICS_PATH)


@pytest.fixture(scope="module")
def held_out() -> pd.DataFrame:
    _require(settings.TEST_CSV_PATH, "the split is DVC-managed, run dvc repro to score it")
    return pd.read_csv(settings.TEST_CSV_PATH)


@pytest.fixture(scope="module")
def scored(model, held_out) -> tuple[np.ndarray, np.ndarray]:
    y = held_out[settings.TARGET].to_numpy().astype(int)
    p = model.predict_proba(held_out[INPUTS])[:, 1]
    return y, p


@pytest.fixture(scope="module")
def applicants() -> pd.DataFrame:
    """A few raw applicants carrying only the 21 model input columns."""
    if settings.TEST_CSV_PATH.exists():
        return pd.read_csv(settings.TEST_CSV_PATH, nrows=25)[INPUTS].reset_index(drop=True)
    if settings.PRESETS_PATH.exists():
        presets = _load_json(settings.PRESETS_PATH)["presets"]
        rows = [{k.upper(): v for k, v in preset["input"].items()} for preset in presets]
        return pd.DataFrame(rows, columns=INPUTS)
    return synthetic_inputs(25)


def test_model_is_a_feature_pipeline(model):
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "features"
    assert isinstance(model.steps[0][1], FeatureBuilder)
    assert model.steps[-1][0] == "clf"
    assert hasattr(model.steps[-1][1], "predict_proba")


def test_feature_names_exclude_protected_attributes(model, version):
    names = list(model.named_steps["features"].get_feature_names_out())
    assert names == list(settings.MODEL_FEATURE_COLUMNS)
    seen = set(names) | set(version["feature_columns"])
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_names_in_"):
        assert list(clf.feature_names_in_) == names
        seen |= set(clf.feature_names_in_)
    assert clf.n_features_in_ == len(settings.MODEL_FEATURE_COLUMNS)
    for protected in settings.PROTECTED_COLUMNS:
        assert protected not in seen
        assert not any(protected in name.upper() for name in seen)


def test_held_out_metrics_match_metrics_json(metrics, held_out, scored):
    y, p = scored
    assert len(held_out) == metrics["n_test"]
    assert float(y.mean()) == pytest.approx(metrics["positive_rate_test"], abs=1e-4)
    fpr, tpr, _ = roc_curve(y, p)
    measured = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ks": float(np.max(np.abs(tpr - fpr))),
    }
    for key, value in measured.items():
        reported = metrics[key]
        assert abs(value - reported) <= METRIC_TOLERANCE, (
            f"{key}: measured {value:.4f}, reports/metrics.json says {reported}"
        )


def test_confusion_at_cost_threshold_reproduces(metrics, scored):
    y, p = scored
    block = metrics["at_threshold"]
    threshold = metrics["threshold_cost_optimal"]
    assert block["threshold"] == threshold
    tn, fp, fn, tp = _confusion(y, p, threshold)
    assert (tn, fp, fn, tp) == (block["tn"], block["fp"], block["fn"], block["tp"])
    assert tn + fp + fn + tp == metrics["n_test"]
    assert block["selection_rate"] == pytest.approx((tp + fp) / len(y), abs=1e-4)
    assert block["precision"] == pytest.approx(tp / (tp + fp), abs=1e-4)
    assert block["recall"] == pytest.approx(tp / (tp + fn), abs=1e-4)


def test_single_row_equals_batch_prediction(model, applicants):
    batch = model.predict_proba(applicants)[:, 1]
    assert batch.shape == (len(applicants),)
    for i in range(len(applicants)):
        one = applicants.iloc[[i]].reset_index(drop=True)
        assert list(one.columns) == INPUTS
        assert len(one) == 1
        single = model.predict_proba(one)[:, 1]
        assert single.shape == (1,)
        assert single[0] == pytest.approx(batch[i], abs=1e-9)


def test_probabilities_within_unit_interval(model, applicants):
    proba = model.predict_proba(applicants)
    assert proba.shape == (len(applicants), 2)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_column_order_does_not_matter(model, applicants):
    rng = np.random.default_rng(0)
    order = [INPUTS[i] for i in rng.permutation(len(INPUTS))]
    assert order != INPUTS
    expected = model.predict_proba(applicants)[:, 1]
    np.testing.assert_array_equal(model.predict_proba(applicants[order])[:, 1], expected)


def test_extra_columns_never_change_the_prediction(model, applicants):
    expected = model.predict_proba(applicants)[:, 1]
    for sex, marriage in ((1, 1), (2, 3)):
        with_extras = applicants.copy()
        with_extras.insert(0, settings.ID_COLUMN, np.arange(1, len(applicants) + 1))
        with_extras["SEX"] = sex
        with_extras["MARRIAGE"] = marriage
        with_extras[settings.TARGET] = 1
        np.testing.assert_array_equal(model.predict_proba(with_extras)[:, 1], expected)


def test_version_json_has_contract_keys(version):
    params = settings.load_params()
    missing = [key for key in VERSION_KEYS if key not in version]
    assert not missing, f"models/version.json lacks {missing}"
    assert version["model"] in {"hgb", "logreg"}
    assert version["n_features"] == len(settings.MODEL_FEATURE_COLUMNS)
    assert version["feature_columns"] == list(settings.MODEL_FEATURE_COLUMNS)
    expected_version = "{}+{}".format(version["package_version"], version["git_sha_short"])
    assert version["model_version"] == expected_version
    assert isinstance(version["git_dirty"], bool)
    assert isinstance(version["registered"], bool)
    assert version["registered_model"] == params["train"]["mlflow"]["registered_model"]
    assert version["mlflow_experiment"] == params["train"]["mlflow"]["experiment"]
    assert version["data_sha256"] == params["data"]["zip_sha256"]
    assert version["n_train"] > 0
    assert version["n_val"] > 0
    assert version["trained_at"].endswith("Z")
    assert version["hgb_n_iter"] is None or version["hgb_n_iter"] >= 1
    for key in LIBRARY_KEYS:
        assert key in version["libraries"], f"libraries block lacks {key}"


def test_metrics_json_has_contract_keys(metrics):
    missing = [key for key in METRICS_KEYS if key not in metrics]
    assert not missing, f"reports/metrics.json lacks {missing}"
    for block in ("at_threshold", "at_precision_target"):
        absent = [key for key in THRESHOLD_BLOCK_KEYS if key not in metrics[block]]
        assert not absent, f"{block} lacks {absent}"
    assert {"roc_auc", "pr_auc", "brier"} <= set(metrics["validation"])
    assert {"roc_auc", "pr_auc", "brier", "ks"} <= set(metrics["baseline_logreg"])
    assert {"roc_auc", "pr_auc"} <= set(metrics["lift_over_baseline"])
    assert metrics["threshold_selection"]["selected_on"] == "validation"
    assert 0.0 < metrics["threshold_cost_optimal"] < 1.0
    assert 0.0 < metrics["threshold_precision_target"] < 1.0
    assert isinstance(metrics["precision_target_met"], bool)
    assert metrics["at_threshold"]["threshold"] == metrics["threshold_cost_optimal"]
    assert metrics["at_precision_target"]["threshold"] == metrics["threshold_precision_target"]
    for key in ("roc_auc", "pr_auc", "brier", "ks", "positive_rate_test"):
        assert 0.0 <= metrics[key] <= 1.0
    calibration = metrics["calibration"]
    assert len(calibration["mean_predicted"]) == len(calibration["fraction_positive"])
    assert len(calibration["mean_predicted"]) == len(calibration["counts"])
    assert sum(calibration["counts"]) == metrics["n_test"]


def test_metrics_json_agrees_with_version_json(metrics, version):
    for key in ("model", "mlflow_run_id", "git_sha", "data_sha256", "trained_at"):
        assert metrics[key] == version[key], key
    assert metrics["n_features"] == version["n_features"] == len(settings.MODEL_FEATURE_COLUMNS)
    assert metrics["n_train"] == version["n_train"]
    assert metrics["n_val"] == version["n_val"]
