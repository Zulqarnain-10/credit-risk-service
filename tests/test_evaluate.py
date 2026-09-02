"""Unit tests for credit_risk.evaluate: thresholds, metrics, calibration, importance, stage.

Every frame is synthetic and small, and the models are tiny logistic pipelines fitted inside
the tests. No MLflow, no network, no real reports are touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import evaluate, settings
from credit_risk.features import FeatureBuilder

INPUTS = list(settings.MODEL_INPUT_COLUMNS)

# A six-row hand case: sorted by score the labels read 0, 1, 0, 1, 0, 1 from the top.
Y = np.array([0, 0, 1, 1, 1, 0])
P = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
COSTS = {"cost_false_negative": 5.0, "cost_false_positive": 1.0}

# A validation set whose cost-optimal threshold is 0.5 and whose precision first reaches 0.6 at 0.3.
VAL_Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
VAL_P = np.array([0.55, 0.65, 0.75, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45])

METRICS_KEYS = (
    "model",
    "model_version",
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
BLOCK_KEYS = (
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


def make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """Plausible applicants whose default label depends on repayment status."""
    rng = np.random.default_rng(seed)
    status_values = np.arange(-2, 9)
    weights = np.array([0.12, 0.18, 0.45, 0.12, 0.08, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])
    weights = weights / weights.sum()
    frame: dict[str, np.ndarray] = {
        settings.ID_COLUMN: np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000.0,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    }
    for c in settings.PAY_STATUS_COLUMNS:
        frame[c] = rng.choice(status_values, size=n, p=weights)
    for c in settings.BILL_COLUMNS:
        frame[c] = rng.integers(-5_000, 200_001, n).astype(float)
    for c in settings.PAY_AMT_COLUMNS:
        frame[c] = rng.integers(0, 50_001, n).astype(float)
    delinquency = np.clip(frame["PAY_0"], 0, None) + 0.5 * np.clip(frame["PAY_2"], 0, None)
    logit = -1.8 + 0.7 * delinquency
    frame[settings.TARGET] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(frame)


def fit_pipeline(df: pd.DataFrame, seed: int = 0) -> Pipeline:
    model = Pipeline(
        [
            ("features", FeatureBuilder()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(df[INPUTS], df[settings.TARGET])
    return model


def threshold_params(target_precision: float = 0.6, step: float = 0.1) -> dict:
    return {
        "threshold": {
            "cost_false_negative": 5.0,
            "cost_false_positive": 1.0,
            "target_precision": target_precision,
            "grid_step": step,
        }
    }


def stage_params() -> dict:
    return {
        **threshold_params(target_precision=0.6, step=0.05),
        "train": {"seed": 0},
        "evaluate": {"calibration_bins": 5, "importance_repeats": 2, "importance_top_n": 5},
    }


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {"train": make_frame(400, 1), "val": make_frame(200, 2), "test": make_frame(200, 3)}


@pytest.fixture(scope="module")
def model(frames: dict[str, pd.DataFrame]) -> Pipeline:
    return fit_pipeline(frames["train"])


# ---------------------------------------------------------------------------
# Confusion counts, KS, compute_metrics
# ---------------------------------------------------------------------------


def test_confusion_at_counts_sum_to_n():
    tn, fp, fn, tp = evaluate.confusion_at(Y, P, 0.5)
    assert (tn, fp, fn, tp) == (2, 1, 1, 2)
    assert tn + fp + fn + tp == len(Y)


def test_confusion_at_threshold_is_inclusive():
    _, fp, fn, tp = evaluate.confusion_at([1, 0], [0.5, 0.5], 0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_ks_statistic_hand_cases():
    assert evaluate.ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert evaluate.ks_statistic(Y, P) == pytest.approx(1 / 3)


def test_compute_metrics_hand_case():
    out = evaluate.compute_metrics(Y, P, 0.5, COSTS)
    assert (out["tn"], out["fp"], out["fn"], out["tp"]) == (2, 1, 1, 2)
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["selection_rate"] == 0.5
    assert out["roc_auc"] == pytest.approx(6 / 9, abs=1e-4)
    assert out["brier"] == pytest.approx(0.22, abs=1e-4)
    assert out["ks"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["threshold"] == 0.5
    # one false negative at cost 5 plus one false positive at cost 1, over six applications
    assert out["expected_cost_per_1000"] == pytest.approx(1000.0)


def test_compute_metrics_without_costs_and_without_positives():
    out = evaluate.compute_metrics(Y, P, 1.01)
    assert "expected_cost_per_1000" not in out
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0
    assert out["selection_rate"] == 0.0
    assert (out["tn"], out["fp"], out["fn"], out["tp"]) == (3, 0, 3, 0)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_threshold_grid_spans_zero_to_one():
    assert evaluate.threshold_grid(0.25).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    fine = evaluate.threshold_grid(0.005)
    assert len(fine) == 201
    assert fine[0] == 0.0
    assert fine[-1] == 1.0
    assert np.all(np.diff(fine) > 0)


def test_choose_thresholds_finds_the_known_optimum():
    chosen = evaluate.choose_thresholds(VAL_Y, VAL_P, threshold_params())
    assert chosen["threshold_cost_optimal"] == 0.5
    assert chosen["validation_cost_at_optimal"] == 0.0
    assert chosen["threshold_precision_target"] == 0.3
    assert chosen["precision_target_met"] is True
    assert chosen["validation_precision_at_target"] == pytest.approx(4 / 6, abs=1e-4)
    assert chosen["cost_false_negative"] == 5.0
    assert chosen["cost_false_positive"] == 1.0
    assert chosen["target_precision"] == 0.6
    assert chosen["grid_step"] == 0.1


def test_choose_thresholds_falls_back_to_max_precision():
    chosen = evaluate.choose_thresholds(VAL_Y, VAL_P, threshold_params(target_precision=1.5))
    assert chosen["precision_target_met"] is False
    assert chosen["threshold_precision_target"] == 0.5
    assert chosen["validation_precision_at_target"] == 1.0


# ---------------------------------------------------------------------------
# Calibration and importance
# ---------------------------------------------------------------------------


def test_calibration_table_shape_and_counts():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    table = evaluate.calibration_table(y, p, 10)
    assert table["bins"] == 10
    assert table["strategy"] == "quantile"
    n_bins = len(table["counts"])
    assert 1 <= n_bins <= 10
    assert len(table["mean_predicted"]) == n_bins
    assert len(table["fraction_positive"]) == n_bins
    assert sum(table["counts"]) == 200
    assert all(c > 0 for c in table["counts"])
    assert table["mean_predicted"] == sorted(table["mean_predicted"])
    assert 0.0 <= table["ece"] <= 1.0


def test_calibration_table_merges_tied_bins():
    p = np.array([0.2] * 50 + [0.8] * 50)
    y = np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10)
    table = evaluate.calibration_table(y, p, 10)
    assert len(table["counts"]) < 10
    assert sum(table["counts"]) == 100
    assert table["mean_predicted"] == [0.2, 0.8]
    assert table["fraction_positive"] == [0.2, 0.8]
    assert table["ece"] == 0.0


def test_feature_importance_ranks_every_feature(frames, model):
    val = frames["val"]
    out = evaluate.feature_importance(model, val[INPUTS], val[settings.TARGET], stage_params())
    assert out["method"] == "permutation_importance"
    assert out["scoring"] == "roc_auc"
    assert out["evaluated_on"] == "validation"
    assert out["n_repeats"] == 2
    assert out["seed"] == 0
    assert len(out["features"]) == 5
    assert set(out["all_features_ranked"]) == set(settings.MODEL_FEATURE_COLUMNS)
    assert len(out["all_features_ranked"]) == len(settings.MODEL_FEATURE_COLUMNS)
    means = [f["importance_mean"] for f in out["features"]]
    assert means == sorted(means, reverse=True)
    assert [f["feature"] for f in out["features"]] == out["all_features_ranked"][:5]
    assert set(out["features"][0]) == {"feature", "importance_mean", "importance_std"}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def test_main_writes_metrics_importance_and_figures(tmp_path: Path, monkeypatch, frames, model):
    reports = tmp_path / "reports"
    models_dir = tmp_path / "models"
    splits = tmp_path / "splits"
    for folder in (reports, models_dir, splits):
        folder.mkdir()
    joblib.dump(model, models_dir / "model.joblib")
    joblib.dump(fit_pipeline(frames["train"], seed=1), models_dir / "baseline.joblib")
    frames["val"].to_csv(splits / "val.csv", index=False)
    frames["test"].to_csv(splits / "test.csv", index=False)
    version = {
        "model": "logreg",
        "model_version": "0.0.0+test",
        "trained_at": "2026-01-01T00:00:00Z",
        "git_sha": "0" * 40,
        "mlflow_run_id": None,
        "data_sha256": "f" * 64,
        "n_train": len(frames["train"]),
        "n_val": len(frames["val"]),
        "n_features": len(settings.MODEL_FEATURE_COLUMNS),
    }
    (models_dir / "version.json").write_text(json.dumps(version), encoding="utf-8")

    monkeypatch.setattr(settings, "load_params", lambda path=None: stage_params())
    monkeypatch.setattr(settings, "VERSION_PATH", models_dir / "version.json")
    monkeypatch.setattr(settings, "MODEL_PATH", models_dir / "model.joblib")
    monkeypatch.setattr(evaluate, "BASELINE_PATH", models_dir / "baseline.joblib")
    monkeypatch.setattr(settings, "VAL_CSV_PATH", splits / "val.csv")
    monkeypatch.setattr(settings, "TEST_CSV_PATH", splits / "test.csv")
    monkeypatch.setattr(settings, "REPORTS_DIR", reports)
    monkeypatch.setattr(settings, "FIGURES_DIR", reports / "figures")
    monkeypatch.setattr(settings, "METRICS_PATH", reports / "metrics.json")
    monkeypatch.setattr(settings, "IMPORTANCE_PATH", reports / "importance.json")

    assert evaluate.main([]) == 0

    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert set(METRICS_KEYS) <= set(metrics)
    assert metrics["model"] == "logreg"
    assert metrics["model_version"] == "0.0.0+test"
    assert metrics["n_train"] == 400
    assert metrics["n_val"] == 200
    assert metrics["n_test"] == 200
    assert metrics["n_features"] == len(settings.MODEL_FEATURE_COLUMNS)
    assert 0.0 < metrics["positive_rate_test"] < 1.0
    for key in ("roc_auc", "pr_auc", "brier", "ks"):
        assert 0.0 <= metrics[key] <= 1.0
        assert 0.0 <= metrics["baseline_logreg"][key] <= 1.0
    assert set(metrics["validation"]) == {"roc_auc", "pr_auc", "brier"}
    for block in ("at_threshold", "at_precision_target"):
        assert tuple(metrics[block]) == BLOCK_KEYS
        counts = metrics[block]
        assert counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == 200
    assert metrics["at_threshold"]["threshold"] == metrics["threshold_cost_optimal"]
    assert metrics["at_precision_target"]["threshold"] == metrics["threshold_precision_target"]
    selection = metrics["threshold_selection"]
    assert selection["selected_on"] == "validation"
    assert selection["cost_false_negative"] == 5.0
    assert selection["cost_false_positive"] == 1.0
    assert selection["target_precision"] == 0.6
    assert selection["grid_step"] == 0.05
    assert metrics["calibration"]["bins"] == 5
    assert sum(metrics["calibration"]["counts"]) == 200
    lift = metrics["lift_over_baseline"]
    assert lift["roc_auc"] == pytest.approx(
        metrics["roc_auc"] - metrics["baseline_logreg"]["roc_auc"], abs=1e-4
    )
    assert lift["pr_auc"] == pytest.approx(
        metrics["pr_auc"] - metrics["baseline_logreg"]["pr_auc"], abs=1e-4
    )

    importance = json.loads((reports / "importance.json").read_text(encoding="utf-8"))
    assert len(importance["features"]) == 5
    assert len(importance["all_features_ranked"]) == len(settings.MODEL_FEATURE_COLUMNS)

    for name in ("roc", "pr", "calibration", "importance"):
        figure = reports / "figures" / f"{name}.png"
        assert figure.is_file(), name
        assert figure.stat().st_size > 0, name
