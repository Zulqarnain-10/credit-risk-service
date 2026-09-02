"""Unit tests for credit_risk.monitoring: PSI helpers, the baseline, drift summary, psi command.

The drift subcommand runs against a stand-in for evidently, so no report library is imported.
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

from credit_risk import monitoring, settings
from credit_risk.features import FeatureBuilder

INPUTS = list(settings.MODEL_INPUT_COLUMNS)
DRIFT = {
    "seed": 7,
    "limit_bal_scale": 0.85,
    "age_shift_years": 4,
    "bill_amt_scale": 1.25,
    "pay_status_shift_share": 0.5,
}
PERTURBATION = {**DRIFT, "pay_status_rows_shifted": 30}


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


def fake_payload(columns: list[str], drifted: set[str], *, with_tests: bool = True) -> dict:
    """A snapshot in the shape evidently's Report.json() produces, one ValueDrift per column."""
    metrics, tests = [], []
    for i, col in enumerate(columns):
        metric_id = f"m{i}"
        categorical = col in monitoring.CATEGORICAL_DRIFT_COLUMNS
        metrics.append(
            {
                "id": metric_id,
                "config": {
                    "type": "evidently:metric:ValueDrift",
                    "column": col,
                    "method": "Jensen-Shannon distance"
                    if categorical
                    else "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.3 if col in drifted else 0.02,
            }
        )
        if with_tests:
            status = "FAIL" if col in drifted else "SUCCESS"
            tests.append({"metric_config": {"metric_id": metric_id}, "status": status})
    share = len(drifted) / len(columns)
    metrics.append(
        {
            "id": "dataset",
            "config": {"type": "evidently:metric:DriftedColumnsCount"},
            "value": {"count": len(drifted), "share": share},
        }
    )
    if with_tests:
        status = "FAIL" if share >= monitoring.DRIFT_SHARE_THRESHOLD else "SUCCESS"
        tests.append({"metric_config": {"metric_id": "dataset"}, "status": status})
    return {"metrics": metrics, "tests": tests}


def full_summary() -> dict:
    features = [
        {"feature": "AGE", "stat_test": "w", "score": 0.3, "threshold": 0.1, "drifted": True},
        {
            "feature": "LIMIT_BAL",
            "stat_test": "w",
            "score": 0.01,
            "threshold": 0.1,
            "drifted": False,
        },
    ]
    return monitoring.build_drift_summary(
        features=features,
        dataset={},
        psi_block={"probability": 0.04, "AGE": 0.3},
        n_reference=100,
        n_current=50,
        perturbation=dict(PERTURBATION),
        model_version="0.0.0+test",
        method="fake",
        method_detail="fake detail",
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    return make_frame(240, 1)


@pytest.fixture(scope="module")
def model(train_frame: pd.DataFrame) -> Pipeline:
    return fit_pipeline(train_frame)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_quantile_edges_from_reference_values():
    assert monitoring.quantile_edges(np.arange(101), 4) == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert monitoring.quantile_edges([3.0, 3.0, 3.0]) == [3.0, 3.0]
    assert monitoring.quantile_edges([1.0, np.nan, np.inf, 2.0], 1) == [1.0, 2.0]
    with pytest.raises(ValueError, match="no finite values"):
        monitoring.quantile_edges([np.nan])


def test_histogram_proportions_sum_to_one_and_keep_outliers():
    edges = [0.0, 1.0, 2.0, 3.0]
    props = monitoring.histogram_proportions([0.5, 1.5, 1.5, 2.5], edges)
    assert props == [0.25, 0.5, 0.25]
    assert sum(props) == pytest.approx(1.0)
    assert monitoring.histogram_proportions([-10.0, 99.0], edges) == [0.5, 0.0, 0.5]
    assert monitoring.histogram_proportions([np.nan, 1.5], edges) == [0.0, 1.0, 0.0]
    assert monitoring.histogram_proportions([], edges) == [0.0, 0.0, 0.0]
    assert monitoring.histogram_proportions([1.0], [5.0, 5.0]) == [1.0]


def test_histogram_proportions_on_quantile_edges_are_even():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    edges = monitoring.quantile_edges(values, 10)
    props = monitoring.histogram_proportions(values, edges)
    assert len(props) == 10
    assert sum(props) == pytest.approx(1.0)
    assert all(p == pytest.approx(0.1, abs=0.01) for p in props)


def test_psi_identical_is_zero_and_shift_is_positive():
    expected = [0.1, 0.2, 0.3, 0.4]
    assert monitoring.psi(expected, expected) == 0.0
    shifted = [0.4, 0.3, 0.2, 0.1]
    value = monitoring.psi(expected, shifted)
    assert value > 0.0
    assert value == pytest.approx(monitoring.psi(shifted, expected))
    hand = sum((a - e) * np.log(a / e) for e, a in zip(expected, shifted, strict=True))
    assert value == pytest.approx(hand)


def test_psi_epsilon_keeps_empty_bins_finite():
    expected = [0.5, 0.5, 0.0]
    actual = [0.4, 0.4, 0.2]
    value = monitoring.psi(expected, actual)
    assert np.isfinite(value)
    assert value > 0.0
    clipped = [0.5, 0.5, 1e-4]
    hand = sum((a - e) * np.log(a / e) for e, a in zip(clipped, actual, strict=True))
    assert value == pytest.approx(hand)
    assert monitoring.psi(expected, actual, eps=1e-2) < value


def test_psi_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        monitoring.psi([0.5, 0.5], [1.0])


@pytest.mark.parametrize(
    ("value", "band"),
    [
        (0.0, "stable"),
        (0.0999, "stable"),
        (0.1, "moderate"),
        (0.25, "moderate"),
        (0.2501, "significant"),
    ],
)
def test_psi_band(value, band):
    assert monitoring.psi_band(value) == band


def test_psi_against_baseline_scores_only_present_columns():
    baseline = {
        "columns": {
            "a": {"edges": [0.0, 1.0, 2.0], "proportions": [0.5, 0.5]},
            "absent": {"edges": [0.0, 1.0], "proportions": [1.0]},
        }
    }
    same = pd.DataFrame({"a": [0.5, 1.5, 0.5, 1.5]})
    assert monitoring.psi_against_baseline(same, baseline) == {"a": 0.0}
    shifted = pd.DataFrame({"a": [1.5, 1.5, 1.5, 0.5]})
    assert monitoring.psi_against_baseline(shifted, baseline)["a"] > 0.0


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_reference_frame_with_and_without_a_model(train_frame, model):
    plain = monitoring.reference_frame(train_frame)
    assert list(plain.columns) == list(settings.MODEL_FEATURE_COLUMNS)
    scored = monitoring.reference_frame(train_frame, model)
    expected = [*settings.MODEL_FEATURE_COLUMNS, monitoring.PROBABILITY_COLUMN]
    assert list(scored.columns) == expected
    assert scored[monitoring.PROBABILITY_COLUMN].between(0.0, 1.0).all()


def test_build_baseline_on_a_small_frame(train_frame, model):
    baseline = monitoring.build_baseline(train_frame, model, "0.0.0+test", n_bins=5)
    assert baseline["reference"] == "train split"
    assert baseline["n"] == len(train_frame)
    assert baseline["model_version"] == "0.0.0+test"
    assert baseline["n_bins"] == 5
    assert baseline["created_at"].endswith("Z")
    assert tuple(baseline["columns"]) == monitoring.BASELINE_COLUMNS
    for name, ref in baseline["columns"].items():
        assert len(ref["proportions"]) == len(ref["edges"]) - 1, name
        assert sum(ref["proportions"]) == pytest.approx(1.0), name
        assert ref["edges"] == sorted(ref["edges"]), name
    assert len(baseline["columns"][monitoring.PROBABILITY_COLUMN]["edges"]) == 6
    scored = monitoring.reference_frame(train_frame, model)
    assert set(monitoring.psi_against_baseline(scored, baseline).values()) == {0.0}


# ---------------------------------------------------------------------------
# Drift report helpers
# ---------------------------------------------------------------------------


def test_perturb_batch_applies_the_documented_changes(train_frame):
    original = train_frame.copy()
    batch, n_shifted = monitoring.perturb_batch(train_frame, DRIFT)
    pd.testing.assert_frame_equal(train_frame, original)
    np.testing.assert_allclose(batch["LIMIT_BAL"], original["LIMIT_BAL"] * 0.85)
    assert (batch["AGE"] == original["AGE"] + 4).all()
    for col in settings.BILL_COLUMNS:
        np.testing.assert_allclose(batch[col], original[col] * 1.25)
    assert 0 < n_shifted < len(batch)
    assert (batch["PAY_0"] != original["PAY_0"]).sum() <= n_shifted
    assert batch["PAY_0"].max() <= monitoring.PAY_STATUS_MAX
    assert (batch["PAY_0"] - original["PAY_0"]).isin([0, 1]).all()

    same, none = monitoring.perturb_batch(train_frame, {**DRIFT, "pay_status_shift_share": 0.0})
    assert none == 0
    assert (same["PAY_0"] == original["PAY_0"]).all()
    every, all_rows = monitoring.perturb_batch(
        train_frame, {**DRIFT, "pay_status_shift_share": 1.0}
    )
    assert all_rows == len(batch)
    capped = np.minimum(original["PAY_0"] + 1, monitoring.PAY_STATUS_MAX)
    assert (every["PAY_0"] == capped).all()


def test_scored_frame_holds_probability_and_score(train_frame, model):
    frame = monitoring.scored_frame(train_frame, model)
    assert (frame[monitoring.SCORE_COLUMN] == frame[monitoring.PROBABILITY_COLUMN]).all()
    assert set(monitoring.DRIFT_COLUMNS) <= set(frame.columns)


def test_drifted_from_score_follows_the_test_kind():
    assert monitoring._drifted_from_score("K-S p_value", 0.01, 0.05) is True
    assert monitoring._drifted_from_score("K-S p_value", 0.2, 0.05) is False
    assert monitoring._drifted_from_score("Wasserstein distance (normed)", 0.1, 0.1) is True
    assert monitoring._drifted_from_score("Jensen-Shannon distance", 0.05, 0.1) is False
    assert monitoring._drifted_from_score(None, 0.5, 0.1) is True


def test_parse_evidently_snapshot_orders_rows_and_reads_tests():
    columns = ["LIMIT_BAL", "EDUCATION", "AGE"]
    payload = fake_payload(columns, {"AGE"})
    features, dataset = monitoring.parse_evidently_snapshot(payload, columns)
    assert [f["feature"] for f in features] == columns
    assert [f["drifted"] for f in features] == [False, False, True]
    assert features[1]["stat_test"] == "Jensen-Shannon distance"
    assert features[2]["score"] == 0.3
    assert features[2]["threshold"] == 0.1
    assert dataset["id"] == "dataset"
    assert dataset["count"] == 1
    assert dataset["share"] == pytest.approx(1 / 3)
    assert dataset["dataset_drift"] is False


def test_parse_evidently_snapshot_falls_back_to_the_score_rule():
    columns = ["LIMIT_BAL", "AGE"]
    payload = fake_payload(columns, {"AGE"}, with_tests=False)
    features, dataset = monitoring.parse_evidently_snapshot(payload, columns)
    assert [f["drifted"] for f in features] == [False, True]
    assert dataset["dataset_drift"] is None


def test_parse_evidently_snapshot_requires_every_column():
    with pytest.raises(RuntimeError, match="no drift result"):
        monitoring.parse_evidently_snapshot(fake_payload(["AGE"], set()), ["AGE", "LIMIT_BAL"])


def test_build_drift_summary_counts_and_bands():
    summary = full_summary()
    assert summary["n_features"] == 2
    assert summary["n_drifted"] == 1
    assert summary["drift_share"] == 0.5
    assert summary["drift_share_threshold"] == monitoring.DRIFT_SHARE_THRESHOLD
    assert summary["dataset_drift"] is True
    assert summary["psi"] == {"probability": 0.04, "AGE": 0.3}
    assert summary["psi_bands"] == {"probability": "stable", "AGE": "significant"}
    assert summary["reference"] == {"name": "train split", "n": 100}
    assert summary["current"]["name"] == monitoring.CURRENT_BATCH_NAME
    assert summary["current"]["n"] == 50
    assert summary["current"]["perturbation"] == PERTURBATION
    assert summary["model_version"] == "0.0.0+test"
    assert summary["method"] == "fake"
    assert summary["method_detail"] == "fake detail"
    assert summary["psi_reference"] == "reports/drift_baseline.json"
    assert summary["generated_at"].endswith("Z")


def test_build_drift_summary_honours_the_dataset_verdict_and_empty_input():
    explicit = monitoring.build_drift_summary(
        features=full_summary()["features"],
        dataset={"dataset_drift": False},
        psi_block={},
        n_reference=1,
        n_current=1,
        perturbation={},
        model_version="v",
        method="m",
        method_detail="d",
    )
    assert explicit["dataset_drift"] is False
    empty = monitoring.build_drift_summary(
        features=[],
        dataset={},
        psi_block={},
        n_reference=0,
        n_current=0,
        perturbation={},
        model_version="v",
        method="m",
        method_detail="d",
    )
    assert empty["n_features"] == 0
    assert empty["drift_share"] == 0.0
    assert empty["dataset_drift"] is False


def test_render_drift_header_states_the_batch_is_simulated():
    html = monitoring.render_drift_header(full_summary())
    assert "The current batch is simulated" in html
    assert "0.0.0+test" in html
    assert monitoring.CURRENT_BATCH_NAME in html
    assert "<td>AGE</td>" in html
    assert "significant" in html
    assert "0.0400" in html
    assert "seed 7" in html
    assert "30 rows" in html
    assert "fake detail" in html


def test_render_drift_header_without_drifted_columns_or_score():
    summary = full_summary()
    for row in summary["features"]:
        row["drifted"] = False
    summary["n_drifted"] = 0
    summary["psi"] = {"AGE": 0.3}
    summary["psi_bands"] = {"AGE": "significant"}
    html = monitoring.render_drift_header(summary)
    assert "No column crossed its drift threshold." in html
    assert "[todo]" in html


def test_render_drift_html_inserts_title_and_header():
    summary = full_summary()
    page = "<html><head><meta charset='utf-8'></head><body><p>report</p></body></html>"
    out = monitoring.render_drift_html(page, summary)
    assert out.startswith(f"<html><head><title>{monitoring.CURRENT_BATCH_NAME}</title>")
    assert out.index("crs-drift") < out.index("<p>report</p>")
    bare = monitoring.render_drift_html("<div>no body</div>", summary)
    assert bare.endswith("<div>no body</div>")
    assert "crs-drift" in bare


# ---------------------------------------------------------------------------
# Prediction-log helpers
# ---------------------------------------------------------------------------


def test_read_prediction_log_skips_bad_lines(tmp_path: Path):
    good = {
        "ts": "t",
        "model_version": "v1",
        "probability": 0.25,
        "decision": "x",
        "inputs": {"limit_bal": 50000, "age": 30},
    }
    log = tmp_path / "predictions.jsonl"
    write_lines(
        log,
        [
            json.dumps(good),
            "",
            "not json",
            json.dumps({"probability": 0.5}),
            json.dumps({"inputs": {"age": 1}}),
            json.dumps([1, 2]),
            json.dumps({**good, "probability": 0.75, "model_version": None}),
        ],
    )
    frame, skipped = monitoring.read_prediction_log(log)
    assert skipped == 4
    assert len(frame) == 2
    assert list(frame.columns) == ["LIMIT_BAL", "AGE", "probability", "model_version"]
    assert frame["probability"].tolist() == [0.25, 0.75]
    assert frame["model_version"].iloc[0] == "v1"
    assert frame["model_version"].isna().tolist() == [False, True]


def test_log_features_recomputes_engineered_columns_when_inputs_are_complete(train_frame):
    logged = train_frame[INPUTS].head(5).copy()
    logged[monitoring.PROBABILITY_COLUMN] = [0.1, 0.2, 0.3, 0.4, 0.5]
    logged["model_version"] = "v1"
    out = monitoring.log_features(logged)
    assert set(settings.ENGINEERED_COLUMNS) <= set(out.columns)
    assert out[monitoring.PROBABILITY_COLUMN].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert (out["model_version"] == "v1").all()
    partial = logged.drop(columns=["AGE"])
    assert monitoring.log_features(partial) is partial


def test_build_psi_report_and_table(tmp_path: Path):
    baseline = {
        "reference": "train split",
        "n": 100,
        "model_version": "v1",
        "columns": {
            "probability": {"edges": [0.0, 0.5, 1.0], "proportions": [0.5, 0.5]},
            "AGE": {"edges": [20.0, 40.0, 60.0], "proportions": [0.5, 0.5]},
            "util_1": {"edges": [0.0, 1.0], "proportions": [1.0]},
        },
    }
    frame = pd.DataFrame(
        {
            "probability": [0.1, 0.9, 0.2, 0.8],
            "AGE": [25, 25, 25, 25],
            "model_version": ["v1", "v1", "v2", None],
        }
    )
    report = monitoring.build_psi_report(
        frame, baseline, tmp_path / "log.jsonl", tmp_path / "baseline.json", skipped=3
    )
    assert report["n_rows"] == 4
    assert report["n_skipped_lines"] == 3
    assert report["reference"] == {"name": "train split", "n": 100, "model_version": "v1"}
    assert report["model_versions"] == {"v1": 2, "v2": 1, "unknown": 1}
    assert [c["column"] for c in report["columns"]] == ["probability", "AGE"]
    assert report["missing_columns"] == ["util_1"]
    assert report["columns"][0] == {"column": "probability", "psi": 0.0, "band": "stable"}
    assert report["columns"][1]["band"] == "significant"
    assert report["n_stable"] == 1
    assert report["n_moderate"] == 0
    assert report["n_significant"] == 1
    assert report["bands"] == monitoring.PSI_BANDS
    assert report["generated_at"].endswith("Z")

    table = monitoring.format_psi_table(report)
    assert "PSI against train split (n=100, model v1)" in table
    assert "log rows=4" in table
    assert "skipped lines=3" in table
    assert "not in log: util_1" in table
    assert "significant" in table
    assert "bands:" in table


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def test_build_parser_registers_the_three_subcommands():
    parser = monitoring.build_parser()
    args = parser.parse_args(["baseline"])
    assert args.command == "baseline"
    assert args.bins == monitoring.DEFAULT_BINS
    assert args.func is monitoring.cmd_baseline
    args = parser.parse_args(["psi", "--log", "x.jsonl"])
    assert args.func is monitoring.cmd_psi
    assert args.json is None
    args = parser.parse_args(["drift"])
    assert args.func is monitoring.cmd_drift
    assert args.report == settings.DRIFT_REPORT_PATH
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_baseline_writes_the_reference_file(tmp_path: Path, monkeypatch, train_frame, model):
    joblib.dump(model, tmp_path / "model.joblib")
    train_frame.to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "version.json").write_text(
        json.dumps({"model_version": "0.0.0+test"}), encoding="utf-8"
    )
    out = tmp_path / "reports" / "drift_baseline.json"
    monkeypatch.setattr(settings, "VERSION_PATH", tmp_path / "version.json")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "TRAIN_CSV_PATH", tmp_path / "train.csv")
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "DRIFT_BASELINE_PATH", out)

    assert monitoring.main(["baseline", "--bins", "4"]) == 0

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    baseline = json.loads(raw.decode("utf-8"))
    assert baseline["n_bins"] == 4
    assert baseline["n"] == len(train_frame)
    assert baseline["model_version"] == "0.0.0+test"
    assert tuple(baseline["columns"]) == monitoring.BASELINE_COLUMNS


def test_main_psi_reports_against_the_baseline(tmp_path: Path, train_frame, model, capsys):
    baseline = monitoring.build_baseline(train_frame, model, "0.0.0+test", n_bins=4)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    p = model.predict_proba(train_frame[INPUTS])[:, 1]
    records = []
    for i in range(20):
        row = train_frame.iloc[i]
        record = {
            "ts": "t",
            "model_version": "0.0.0+test" if i else "other",
            "probability": float(p[i]),
            "decision": "x",
            "inputs": {c.lower(): float(row[c]) for c in INPUTS},
        }
        records.append(json.dumps(record))
    log_path = tmp_path / "predictions.jsonl"
    write_lines(log_path, [*records, "garbage"])
    out_json = tmp_path / "psi.json"

    argv = [
        "psi",
        "--log",
        str(log_path),
        "--baseline",
        str(baseline_path),
        "--json",
        str(out_json),
    ]
    assert monitoring.main(argv) == 0

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["n_rows"] == 20
    assert report["n_skipped_lines"] == 1
    assert [c["column"] for c in report["columns"]] == list(monitoring.BASELINE_COLUMNS)
    assert report["missing_columns"] == []
    assert report["model_versions"] == {"0.0.0+test": 19, "other": 1}
    captured = capsys.readouterr().out
    assert "PSI against train split" in captured
    assert "wrote" in captured


def test_main_psi_fails_without_usable_rows(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"columns": {}}), encoding="utf-8")
    log_path = tmp_path / "empty.jsonl"
    write_lines(log_path, ["nope"])
    assert monitoring.main(["psi", "--log", str(log_path), "--baseline", str(baseline_path)]) == 1


def drift_setup(tmp_path: Path, monkeypatch, train_frame, model) -> list[str]:
    """Write the inputs the drift command reads and return its argv."""
    test_frame = make_frame(60, 2)
    joblib.dump(model, tmp_path / "model.joblib")
    train_frame.to_csv(tmp_path / "train.csv", index=False)
    test_frame.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "version.json").write_text(
        json.dumps({"model_version": "0.0.0+test"}), encoding="utf-8"
    )
    baseline = monitoring.build_baseline(train_frame, model, "0.0.0+test", n_bins=4)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(settings, "load_params", lambda path=None: {"drift": DRIFT})
    monkeypatch.setattr(settings, "VERSION_PATH", tmp_path / "version.json")
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "TRAIN_CSV_PATH", tmp_path / "train.csv")
    monkeypatch.setattr(settings, "TEST_CSV_PATH", tmp_path / "test.csv")
    return [
        "drift",
        "--baseline",
        str(baseline_path),
        "--report",
        str(tmp_path / "out" / "drift_report.html"),
        "--summary",
        str(tmp_path / "out" / "drift_summary.json"),
    ]


def test_main_drift_with_a_stand_in_for_evidently(tmp_path: Path, monkeypatch, train_frame, model):
    argv = drift_setup(tmp_path, monkeypatch, train_frame, model)
    seen: dict = {}

    def fake_run(reference, current, columns):
        seen["n_reference"] = len(reference)
        seen["n_current"] = len(current)
        seen["columns"] = list(columns)
        html = "<html><head></head><body><p>evidently</p></body></html>"
        return fake_payload(list(columns), {"AGE", "LIMIT_BAL"}), html, "0.0-test"

    monkeypatch.setattr(monitoring, "run_evidently_drift", fake_run)

    assert monitoring.main(argv) == 0

    assert seen["columns"] == list(monitoring.DRIFT_COLUMNS)
    assert seen["n_reference"] == len(train_frame)
    assert seen["n_current"] == 60
    summary = json.loads((tmp_path / "out" / "drift_summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "evidently"
    assert "0.0-test" in summary["method_detail"]
    assert summary["model_version"] == "0.0.0+test"
    assert summary["n_features"] == len(monitoring.DRIFT_COLUMNS)
    assert summary["n_drifted"] == 2
    assert summary["dataset_drift"] is False
    assert summary["reference"]["n"] == len(train_frame)
    assert summary["current"]["n"] == 60
    perturbation = summary["current"]["perturbation"]
    assert perturbation["seed"] == 7
    assert perturbation["age_shift_years"] == 4
    assert 0 <= perturbation["pay_status_rows_shifted"] <= 60
    assert set(summary["psi"]) == set(monitoring.BASELINE_COLUMNS)
    assert summary["psi"]["AGE"] > 0.0
    assert set(summary["psi_bands"]) == set(summary["psi"])
    html = (tmp_path / "out" / "drift_report.html").read_text(encoding="utf-8")
    assert f"<title>{monitoring.CURRENT_BATCH_NAME}</title>" in html
    assert "<p>evidently</p>" in html
    assert "The current batch is simulated" in html


def test_main_drift_exits_when_evidently_is_missing(
    tmp_path: Path, monkeypatch, train_frame, model
):
    argv = drift_setup(tmp_path, monkeypatch, train_frame, model)

    def missing(*args, **kwargs):
        raise ImportError("no evidently here")

    monkeypatch.setattr(monitoring, "run_evidently_drift", missing)
    with pytest.raises(SystemExit) as excinfo:
        monitoring.main(argv)
    assert excinfo.value.code == 1
    assert not (tmp_path / "out").exists()
