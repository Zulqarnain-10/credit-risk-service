"""Unit tests for credit_risk.fairness: per-group rates, gaps, the parity ratio, and the stage.

The toy frames carry hand-set labels and predictions so every rate is known in advance.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import fairness, settings
from credit_risk.features import FeatureBuilder

INPUTS = list(settings.MODEL_INPUT_COLUMNS)
BANDS = [21, 30, 40, 50, 100]

# Four men then four women.
TOY = pd.DataFrame({"SEX": [1, 1, 1, 1, 2, 2, 2, 2]})
TOY_Y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
TOY_PRED = np.array([1, 0, 1, 0, 1, 0, 0, 0])


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


def fit_pipeline(df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("features", FeatureBuilder()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    model.fit(df[INPUTS], df[settings.TARGET])
    return model


# ---------------------------------------------------------------------------
# group_report and summarise
# ---------------------------------------------------------------------------


def test_group_report_rates_by_sex():
    rows = fairness.group_report(TOY, TOY_Y, TOY_PRED, "SEX", labels=fairness.SEX_LABELS)
    assert [r["group"] for r in rows] == ["male", "female"]
    assert [r["value"] for r in rows] == [1, 2]
    male, female = rows
    assert male == {
        "group": "male",
        "value": 1,
        "n": 4,
        "positive_rate": 0.5,
        "selection_rate": 0.5,
        "tpr": 0.5,
        "fpr": 0.5,
        "precision": 0.5,
    }
    assert female == {
        "group": "female",
        "value": 2,
        "n": 4,
        "positive_rate": 0.25,
        "selection_rate": 0.25,
        "tpr": 1.0,
        "fpr": 0.0,
        "precision": 1.0,
    }


def test_summarise_gaps_and_parity_ratio():
    rows = fairness.group_report(TOY, TOY_Y, TOY_PRED, "SEX", labels=fairness.SEX_LABELS)
    summary = fairness.summarise(rows)
    assert summary["groups"] is rows
    assert summary["max_gap"] == {"selection_rate": 0.25, "tpr": 0.5, "fpr": 0.5}
    assert summary["demographic_parity_ratio"] == 0.5


def test_group_report_without_labels_uses_the_raw_codes():
    rows = fairness.group_report(TOY, TOY_Y, TOY_PRED, "SEX")
    assert [r["group"] for r in rows] == ["1", "2"]
    assert [r["value"] for r in rows] == [1, 2]


def test_group_report_rates_are_none_without_a_denominator():
    df = pd.DataFrame({"G": [1, 1, 2, 2]})
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    rows = fairness.group_report(df, y, pred, "G")
    first, second = rows
    assert first["tpr"] is None
    assert first["precision"] is None
    assert first["fpr"] == 0.0
    assert second["fpr"] is None
    assert second["tpr"] == 1.0
    assert second["precision"] == 1.0
    summary = fairness.summarise(rows)
    assert summary["max_gap"] == {"selection_rate": 1.0, "tpr": 0.0, "fpr": 0.0}
    assert summary["demographic_parity_ratio"] == 0.0


def test_summarise_ratio_is_none_when_nobody_is_selected():
    rows = fairness.group_report(TOY, TOY_Y, np.zeros(len(TOY), dtype=int), "SEX")
    summary = fairness.summarise(rows)
    assert summary["demographic_parity_ratio"] is None
    assert summary["max_gap"] == {"selection_rate": 0.0, "tpr": 0.0, "fpr": 0.0}


def test_summarise_of_no_groups_is_empty():
    summary = fairness.summarise([])
    assert summary["max_gap"] == {"selection_rate": None, "tpr": None, "fpr": None}
    assert summary["demographic_parity_ratio"] is None


def test_group_report_age_bands_skip_rows_outside_every_band():
    df = pd.DataFrame({"AGE": [25, 35, 35, 45, 20, 100]})
    y = np.array([1, 1, 0, 0, 1, 1])
    pred = np.array([1, 1, 1, 0, 1, 1])
    rows = fairness.group_report(df, y, pred, "AGE", bands=BANDS)
    assert [r["group"] for r in rows] == ["21-29", "30-39", "40-49"]
    assert [r["value"] for r in rows] == ["21-29", "30-39", "40-49"]
    assert [r["n"] for r in rows] == [1, 2, 1]
    thirties = rows[1]
    assert thirties["positive_rate"] == 0.5
    assert thirties["selection_rate"] == 1.0
    assert thirties["tpr"] == 1.0
    assert thirties["fpr"] == 1.0
    assert thirties["precision"] == 0.5


def test_group_report_band_edges_are_inclusive_low_exclusive_high():
    df = pd.DataFrame({"AGE": [21, 29, 30, 99]})
    ones = np.ones(4, dtype=int)
    rows = fairness.group_report(df, ones, ones, "AGE", bands=BANDS)
    assert {r["group"]: r["n"] for r in rows} == {"21-29": 2, "30-39": 1, "50-99": 1}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def test_main_writes_the_fairness_report(tmp_path: Path, monkeypatch):
    train = make_frame(300, 1)
    test = make_frame(120, 2)
    model = fit_pipeline(train)
    joblib.dump(model, tmp_path / "model.joblib")
    test.to_csv(tmp_path / "test.csv", index=False)
    metrics = {"threshold_cost_optimal": 0.25, "model_version": "0.0.0+test"}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    monkeypatch.setattr(
        settings, "load_params", lambda path=None: {"fairness": {"age_bands": BANDS}}
    )
    monkeypatch.setattr(settings, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "TEST_CSV_PATH", tmp_path / "test.csv")
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "FAIRNESS_PATH", tmp_path / "reports" / "fairness.json")

    assert fairness.main([]) == 0

    report = json.loads((tmp_path / "reports" / "fairness.json").read_text(encoding="utf-8"))
    assert report["evaluated_on"] == "test"
    assert report["n"] == 120
    assert report["threshold"] == 0.25
    assert report["model_version"] == "0.0.0+test"
    assert isinstance(report["note"], str)
    assert report["note"]
    assert [g["group"] for g in report["sex"]["groups"]] == ["male", "female"]
    assert sum(g["n"] for g in report["sex"]["groups"]) == 120
    assert set(report["sex"]["max_gap"]) == set(fairness.GAP_METRICS)
    assert report["age_band"]["bands"] == BANDS
    assert sum(g["n"] for g in report["age_band"]["groups"]) == 120
    assert all(0.0 <= g["selection_rate"] <= 1.0 for g in report["age_band"]["groups"])

    p = model.predict_proba(test[INPUTS])[:, 1]
    male = test["SEX"].to_numpy() == 1
    expected = round(float((p[male] >= 0.25).mean()), 4)
    assert report["sex"]["groups"][0]["selection_rate"] == expected
    assert report["sex"]["groups"][0]["n"] == int(male.sum())
