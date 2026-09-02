"""Smoke test: both models fit on a small stratified sample well inside the time budget.

The fit test carries the slow marker because it trains real estimators, but it stays in the
default run: on params.smoke_train.sample_rows rows it finishes in about a second. The sample
comes from the train split, else the features CSV, else a seeded synthetic frame.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from credit_risk import settings, train

SEED = 0
VAL_FRACTION = 0.2
TIME_BUDGET_SECONDS = 60.0
MIN_WINNER_ROC_AUC = 0.6
INPUTS = list(settings.MODEL_INPUT_COLUMNS)


def synthetic_frame(n: int, seed: int = SEED) -> pd.DataFrame:
    """Plausible applicants whose default label depends on repayment status."""
    rng = np.random.default_rng(seed)
    status_values = np.arange(-2, 9)
    status_weights = np.array([0.12, 0.18, 0.45, 0.12, 0.08, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])
    status_weights = status_weights / status_weights.sum()
    frame: dict[str, np.ndarray] = {
        settings.ID_COLUMN: np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000.0,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    }
    for c in settings.PAY_STATUS_COLUMNS:
        frame[c] = rng.choice(status_values, size=n, p=status_weights)
    for c in settings.BILL_COLUMNS:
        frame[c] = rng.integers(-5_000, 200_001, n).astype(float)
    for c in settings.PAY_AMT_COLUMNS:
        frame[c] = rng.integers(0, 50_001, n).astype(float)
    delinquency = np.clip(frame["PAY_0"], 0, None) + 0.5 * np.clip(frame["PAY_2"], 0, None)
    logit = -1.8 + 0.7 * delinquency
    frame[settings.TARGET] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(frame)


def source_frame() -> pd.DataFrame:
    for path in (settings.TRAIN_CSV_PATH, settings.FEATURES_CSV_PATH):
        if path.exists():
            return pd.read_csv(path)
    return synthetic_frame(6_000)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df[settings.TARGET], random_state=seed)
    return sample.reset_index(drop=True)


def test_pick_winner_prefers_higher_auc_and_breaks_ties_toward_primary():
    assert train.pick_winner({"logreg": {"roc_auc": 0.80}, "hgb": {"roc_auc": 0.75}}) == "logreg"
    assert train.pick_winner({"logreg": {"roc_auc": 0.70}, "hgb": {"roc_auc": 0.75}}) == "hgb"
    assert train.pick_winner({"logreg": {"roc_auc": 0.75}, "hgb": {"roc_auc": 0.75}}) == "hgb"
    tied = {"logreg": {"roc_auc": 0.75}, "hgb": {"roc_auc": 0.75}}
    assert train.pick_winner(tied, primary="logreg") == "logreg"


@pytest.mark.slow
def test_smoke_fit_both_models_under_budget():
    params = settings.load_params()
    n_rows = int(params["smoke_train"]["sample_rows"])
    sample = stratified_sample(source_frame(), n_rows, SEED)
    assert len(sample) == n_rows
    assert set(sample[settings.TARGET].unique()) == {0, 1}
    train_df, val_df = train_test_split(
        sample, test_size=VAL_FRACTION, stratify=sample[settings.TARGET], random_state=SEED
    )

    started = time.perf_counter()
    result = train.fit_models(train_df, val_df, params, mlflow_enabled=False)
    elapsed = time.perf_counter() - started

    assert elapsed < TIME_BUDGET_SECONDS, f"fit_models took {elapsed:.1f}s on {n_rows} rows"
    assert set(result["models"]) == set(train.MODEL_NAMES)
    assert set(result["val_metrics"]) == set(train.MODEL_NAMES)
    assert set(result["fit_seconds"]) == set(train.MODEL_NAMES)
    for name in train.MODEL_NAMES:
        model = result["models"][name]
        assert isinstance(model, Pipeline)
        check_is_fitted(model.named_steps["clf"])
        proba = model.predict_proba(val_df[INPUTS])[:, 1]
        assert proba.shape == (len(val_df),)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert set(result["val_metrics"][name]) == {"roc_auc", "pr_auc", "brier"}
        assert result["fit_seconds"][name] >= 0.0

    winner = result["winner"]
    assert winner in train.MODEL_NAMES
    best_auc = max(m["roc_auc"] for m in result["val_metrics"].values())
    assert result["val_metrics"][winner]["roc_auc"] == best_auc
    assert best_auc > MIN_WINNER_ROC_AUC
    assert train.hgb_n_iter(result["models"]["hgb"]) >= 1
    assert all(run_id is None for run_id in result["run_ids"].values())
    assert all(uri is None for uri in result["model_uris"].values())
