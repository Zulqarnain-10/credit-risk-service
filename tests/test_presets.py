"""Unit tests for credit_risk.presets: row selection, descriptions, and the presets file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import presets, settings
from credit_risk.features import FeatureBuilder

INPUTS = list(settings.MODEL_INPUT_COLUMNS)
DIGIT = re.compile(r"\d")

# (delinq_months, delinq_recent, util_mean, pay_ratio_mean) for four distinct repayment patterns.
PATTERNS = [(0, 0, 0.1, 1.0), (3, 1, 0.5, 0.5), (1, 1, 0.9, 0.05), (1, 0, 0.5, 0.5)]


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


def feats(months: int, recent: int, util: float, ratio: float) -> pd.Series:
    return pd.Series(
        {
            "delinq_months": months,
            "delinq_recent": recent,
            "util_mean": util,
            "pay_ratio_mean": ratio,
        }
    )


@pytest.fixture(scope="module")
def fitted() -> tuple[pd.DataFrame, Pipeline]:
    return make_frame(80, 2), fit_pipeline(make_frame(300, 1))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_api_fields_are_the_lowercase_model_inputs():
    assert tuple(c.lower() for c in INPUTS) == presets.API_FIELDS
    assert len(presets.API_FIELDS) == 21
    assert "sex" not in presets.API_FIELDS
    assert "marriage" not in presets.API_FIELDS


def test_pick_rows_nearest_to_each_target():
    p = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99, 0.15])
    chosen = presets.pick_rows(p, threshold=0.31)
    assert list(chosen) == ["low_risk", "borderline", "high_risk"]
    assert chosen == {"low_risk": 1, "borderline": 3, "high_risk": 8}
    assert chosen["low_risk"] == int(np.argmin(np.abs(p - np.quantile(p, 0.10))))
    assert chosen["high_risk"] == int(np.argmin(np.abs(p - np.quantile(p, 0.95))))


def test_pick_rows_never_reuses_a_row():
    chosen = presets.pick_rows(np.array([0.5, 0.5, 0.5]), threshold=0.5)
    assert sorted(chosen.values()) == [0, 1, 2]


def test_scalar_keeps_whole_numbers_as_int():
    assert presets._scalar(5.0) == 5
    assert isinstance(presets._scalar(5.0), int)
    assert isinstance(presets._scalar(np.int64(7)), int)
    assert presets._scalar(np.float64(2.5)) == 2.5
    assert presets._scalar(1.23456) == 1.2346


@pytest.mark.parametrize(("months", "recent", "util", "ratio"), PATTERNS)
def test_describe_is_one_plain_sentence_without_numbers(months, recent, util, ratio):
    text = presets.describe(feats(months, recent, util, ratio))
    assert text[0].isupper()
    assert text.endswith(".")
    assert text.count(".") == 1
    assert not DIGIT.search(text)


def test_describe_distinguishes_the_patterns():
    variants = {presets.describe(feats(*args)) for args in PATTERNS}
    assert len(variants) == len(PATTERNS)


def test_threshold_note_marks_the_rounded_match():
    at_threshold = presets.threshold_note(0.15504, 0.155)
    nearby = presets.threshold_note(0.1612, 0.155)
    assert at_threshold != nearby
    for text in (at_threshold, nearby):
        assert "threshold" in text
        assert text.endswith(".")
        assert not DIGIT.search(text)


# ---------------------------------------------------------------------------
# build_presets and the stage
# ---------------------------------------------------------------------------


def test_build_presets_uses_real_rows(fitted):
    test, model = fitted
    p = model.predict_proba(test[INPUTS])[:, 1]
    threshold = float(np.median(p))
    out = presets.build_presets(test, model, threshold, "0.0.0+test")
    assert out["generated_from"] == "test split, model 0.0.0+test"
    rows = out["presets"]
    assert [r["id"] for r in rows] == ["low_risk", "borderline", "high_risk"]
    assert [r["label"] for r in rows] == ["Low risk", "Borderline", "High risk"]
    probabilities = [r["model_probability"] for r in rows]
    assert probabilities == sorted(probabilities)
    assert len({r["source_row_id"] for r in rows}) == 3

    ids = test[settings.ID_COLUMN].to_numpy()
    for row in rows:
        idx = int(np.flatnonzero(ids == row["source_row_id"])[0])
        source = test.iloc[idx]
        assert tuple(row["input"]) == presets.API_FIELDS
        for field, col in zip(presets.API_FIELDS, INPUTS, strict=True):
            assert row["input"][field] == pytest.approx(float(source[col]))
        assert row["model_probability"] == round(float(p[idx]), 6)
        assert not DIGIT.search(row["description"])
        assert row["description"].endswith(".")
    assert (
        abs(rows[1]["model_probability"] - threshold) <= float(np.min(np.abs(p - threshold))) + 1e-6
    )
    assert "threshold" in rows[1]["description"]
    assert "threshold" not in rows[0]["description"]


def test_main_writes_the_presets_file(tmp_path: Path, monkeypatch, fitted, capsys):
    test, model = fitted
    joblib.dump(model, tmp_path / "model.joblib")
    test.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"threshold_cost_optimal": 0.3}), encoding="utf-8"
    )
    (tmp_path / "version.json").write_text(
        json.dumps({"model_version": "0.0.0+test"}), encoding="utf-8"
    )
    monkeypatch.setattr(settings, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(settings, "VERSION_PATH", tmp_path / "version.json")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "TEST_CSV_PATH", tmp_path / "test.csv")
    monkeypatch.setattr(settings, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(settings, "PRESETS_PATH", tmp_path / "configs" / "presets.json")

    assert presets.main([]) == 0

    raw = (tmp_path / "configs" / "presets.json").read_bytes()
    assert b"\r\n" not in raw
    out = json.loads(raw.decode("utf-8"))
    assert out["generated_from"] == "test split, model 0.0.0+test"
    assert [r["id"] for r in out["presets"]] == ["low_risk", "borderline", "high_risk"]
    assert all(set(r["input"]) == set(presets.API_FIELDS) for r in out["presets"])
    assert "wrote" in capsys.readouterr().out
