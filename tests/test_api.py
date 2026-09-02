"""API tests against the real, shipped model artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk import __version__, settings
from credit_risk.api import __main__ as api_main
from credit_risk.api.app import (
    LOGGING_CONFIG_PATH,
    MAX_BODY_BYTES,
    STATIC_DIR,
    TITLE,
    configure_logging,
    create_app,
)
from credit_risk.api.schemas import BATCH_MAX_ITEMS, Applicant, applicants_to_frame

JSON_HEADERS = {"content-type": "application/json"}

VERSION_METRIC_KEYS = {
    "roc_auc",
    "pr_auc",
    "brier",
    "ks",
    "threshold_cost_optimal",
    "n_test",
    "trained_at",
}


def test_health(client, version_info):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": version_info["model_version"],
    }


def test_version_keys(client, version_info, metrics):
    body = client.get("/version").json()
    for key, value in version_info.items():
        assert body[key] == value
    assert body["disclaimer"] == settings.DISCLAIMER
    assert set(body["metrics"]) == VERSION_METRIC_KEYS
    for key in VERSION_METRIC_KEYS:
        assert body["metrics"][key] == metrics[key]
    if settings.LOADTEST_PATH.is_file():
        assert {"p95_ms", "host"} <= set(body["loadtest"])
    else:
        assert "loadtest" not in body


def test_version_loadtest_block_when_report_exists(client, tmp_path, monkeypatch):
    report = tmp_path / "loadtest.json"
    report.write_text(
        json.dumps({"p95_ms": 12.5, "host": "test host", "rps": 80.0, "ignored": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "LOADTEST_PATH", report)
    body = client.get("/version").json()
    assert body["loadtest"] == {"p95_ms": 12.5, "host": "test host", "rps": 80.0}


def test_presets(client, presets):
    body = client.get("/presets").json()
    assert body["presets"] == presets
    assert [preset["id"] for preset in body["presets"]] == [
        "low_risk",
        "borderline",
        "high_risk",
    ]


def test_importance(client, importance):
    body = client.get("/importance").json()
    assert body == importance
    assert body["features"][0]["feature"] == importance["all_features_ranked"][0]


@pytest.mark.parametrize("index", [0, 1, 2], ids=["low_risk", "borderline", "high_risk"])
def test_predict_matches_preset(client, presets, metrics, version_info, index):
    preset = presets[index]
    response = client.post("/predict", json=preset["input"])
    assert response.status_code == 200
    body = response.json()

    # Full precision through the same frame conversion the endpoint uses.
    bundle = client.app.state.bundle
    raw = float(bundle.predict_proba(applicants_to_frame([Applicant(**preset["input"])]))[0])
    assert abs(raw - preset["model_probability"]) < 1e-6

    assert abs(body["probability"] - round(preset["model_probability"], 4)) < 1e-6
    assert body["threshold"] == metrics["threshold_cost_optimal"]
    # The decision is taken on the four-decimal probability the response carries.
    expected = "likely_default" if round(raw, 4) >= body["threshold"] else "unlikely_default"
    assert body["decision"] == expected
    assert body["decision"] == (
        "likely_default" if body["probability"] >= body["threshold"] else "unlikely_default"
    )
    assert (
        body["decision_label"]
        == {
            "likely_default": "Likely to default",
            "unlikely_default": "Unlikely to default",
        }[expected]
    )
    assert body["model"] == version_info["model"]
    assert body["model_version"] == version_info["model_version"]
    assert body["disclaimer"] == settings.DISCLAIMER


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0.0, "likely_default"),
        (-1e-9, "likely_default"),
        (-0.00004, "likely_default"),
        (-0.00006, "unlikely_default"),
        (-0.0001, "unlikely_default"),
    ],
    ids=["at_threshold", "just_below", "rounds_up", "rounds_down", "one_step_below"],
)
def test_decision_uses_the_returned_probability(
    client, preset_payload, monkeypatch, offset, expected
):
    """A probability that displays as the threshold reads likely_default, never the reverse."""
    bundle = client.app.state.bundle
    raw = bundle.threshold + offset
    monkeypatch.setattr(bundle, "predict_proba", lambda frame: np.full(len(frame), raw))
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == round(raw, 4)
    assert body["decision"] == expected
    assert body["decision"] == (
        "likely_default" if body["probability"] >= body["threshold"] else "unlikely_default"
    )


def test_predict_missing_field_422(client, preset_payload):
    preset_payload.pop("age")
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "age" and item["type"] == "missing" for item in detail)


@pytest.mark.parametrize(
    ("field", "token", "error_type"),
    [
        ("limit_bal", "NaN", "finite_number"),
        ("bill_amt1", "Infinity", "finite_number"),
        ("pay_amt3", "-Infinity", "finite_number"),
        ("pay_0", "NaN", "int_type"),
    ],
)
def test_predict_non_finite_token_422(client, preset_payload, field, token, error_type):
    """Python's json module emits NaN and Infinity; they must be a 422, not a 500."""
    preset_payload[field] = "__TOKEN__"
    raw = json.dumps(preset_payload).replace('"__TOKEN__"', token)
    response = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == field and item["type"] == error_type for item in detail)


def test_predict_batch_non_finite_token_422(client, presets):
    items = [dict(preset["input"]) for preset in presets]
    items[1]["limit_bal"] = "__TOKEN__"
    raw = json.dumps({"items": items}).replace('"__TOKEN__"', "NaN")
    response = client.post("/predict/batch", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"] == ["body", "items", 1, "limit_bal"] for item in detail)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("pay_0", True, "int_type"),
        ("age", "30", "int_type"),
        ("limit_bal", "50000", "float_type"),
        ("limit_bal", False, "float_type"),
    ],
)
def test_predict_rejects_coerced_types_422(client, preset_payload, field, value, error_type):
    """Strict types: a boolean or a numeric string never scores as a number."""
    preset_payload[field] = value
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == field and item["type"] == error_type for item in detail)


def test_predict_accepts_integers_and_floats_for_amounts(client, preset_payload):
    assert all(isinstance(value, int) for value in preset_payload.values())
    assert client.post("/predict", json=preset_payload).status_code == 200
    preset_payload["limit_bal"] = float(preset_payload["limit_bal"]) + 0.5
    assert client.post("/predict", json=preset_payload).status_code == 200


def test_predict_body_over_cap_413(client, preset_payload):
    """A declared Content-Length above the cap is refused before the body is parsed."""
    preset_payload["note"] = "x" * MAX_BODY_BYTES
    raw = json.dumps(preset_payload)
    assert len(raw) > MAX_BODY_BYTES
    response = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert response.status_code == 413
    assert str(MAX_BODY_BYTES) in response.json()["detail"]


def test_predict_chunked_body_over_cap_413(client):
    """A chunked body with no Content-Length is refused once it streams past the cap."""

    async def chunks():
        for _ in range(4):
            yield b"[" + b" " * (MAX_BODY_BYTES // 2)

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            return await http.post("/predict", content=chunks(), headers=JSON_HEADERS)

    response = asyncio.run(run())
    assert "content-length" not in response.request.headers
    assert response.request.headers["transfer-encoding"] == "chunked"
    assert response.status_code == 413
    assert str(MAX_BODY_BYTES) in response.json()["detail"]


def test_predict_body_under_cap_is_parsed(client, preset_payload):
    preset_payload["note"] = "x" * 1024
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_predict_out_of_range_422(client, preset_payload):
    preset_payload["age"] = settings.FIELD_RANGES["AGE"][0] - 1
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "age" and "greater_than_equal" in item["type"] for item in detail)


def test_predict_unknown_field_422(client, preset_payload):
    preset_payload["sex"] = 2
    response = client.post("/predict", json=preset_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "sex" and item["type"] == "extra_forbidden" for item in detail)


def test_predict_batch_of_three(client, presets):
    response = client.post("/predict/batch", json={"items": [p["input"] for p in presets]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    for prediction, preset in zip(body["predictions"], presets, strict=True):
        assert abs(prediction["probability"] - round(preset["model_probability"], 4)) < 1e-6
        assert prediction["disclaimer"] == settings.DISCLAIMER


def test_predict_batch_over_cap_422(client, preset_payload):
    items = [preset_payload] * (BATCH_MAX_ITEMS + 1)
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"][-1] == "items"
    assert str(BATCH_MAX_ITEMS) in detail[0]["msg"]
    assert str(BATCH_MAX_ITEMS + 1) in detail[0]["msg"]


def test_predict_batch_empty_422(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422


def test_metrics_exposes_prediction_counter(client, preset_payload):
    assert client.post("/predict", json=preset_payload).status_code == 200
    if STATIC_DIR.is_dir():
        assert client.get("/static/style.css").status_code == 200
    assert client.get("/metrics").status_code == 200  # a scrape that must not count itself
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "credit_risk_predictions_total" in text
    assert "http_request_duration" in text
    counts = re.findall(r'credit_risk_predictions_total\{decision="[a-z_]+"\} (\d+\.\d+)', text)
    assert len(counts) == 2
    assert sum(float(value) for value in counts) >= 1
    handlers = set(re.findall(r'handler="([^"]+)"', text))
    assert "/predict" in handlers
    assert "/static" not in handlers
    assert "/metrics" not in handlers


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    index = STATIC_DIR / "index.html"
    if index.is_file():
        assert response.text == index.read_text(encoding="utf-8")
        assert client.get("/static/style.css").status_code == 200
    else:
        assert "Credit risk service" in response.text


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == TITLE
    assert spec["info"]["version"] == __version__
    assert settings.DISCLAIMER in spec["info"]["description"]
    assert {"/health", "/version", "/presets", "/importance", "/predict", "/predict/batch"} <= set(
        spec["paths"]
    )
    applicant = spec["components"]["schemas"]["Applicant"]
    assert applicant["additionalProperties"] is False
    assert set(applicant["required"]) == {c.lower() for c in settings.MODEL_INPUT_COLUMNS}
    assert applicant["properties"]["age"]["minimum"] == settings.FIELD_RANGES["AGE"][0]


def test_prediction_log_line_written(tmp_path, monkeypatch, preset_payload):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    fresh_app = create_app()
    with TestClient(fresh_app) as fresh_client:
        assert fresh_client.app.state.prediction_log_path == log_path
        response = fresh_client.post("/predict", json=preset_payload)
        assert response.status_code == 200
    lines = log_path.read_bytes().decode("utf-8").split("\n")
    assert lines[-1] == ""
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert set(record) == {"ts", "model_version", "probability", "decision", "inputs"}
    assert record["ts"].endswith("Z")
    assert record["model_version"] == response.json()["model_version"]
    assert record["decision"] == response.json()["decision"]
    assert abs(record["probability"] - response.json()["probability"]) < 1e-4
    assert record["inputs"] == preset_payload


def test_prediction_log_failure_never_fails_request(tmp_path, monkeypatch, preset_payload):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path))  # a directory cannot be appended to
    with TestClient(create_app()) as fresh_client:
        response = fresh_client.post("/predict", json=preset_payload)
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, 8000, False),
        ("", 8000, True),
        ("   ", 8000, True),
        ("abc", 8000, True),
        ("8000.0", 8000, True),
        ("8127", 8127, False),
        (" 7860 ", 7860, False),
    ],
    ids=["unset", "empty", "blank", "letters", "decimal", "custom", "padded"],
)
def test_api_port_falls_back_to_default(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", raw)
    with caplog.at_level(logging.WARNING, logger="credit_risk.settings"):
        assert settings.api_port() == expected
    warnings = [record for record in caplog.records if record.name == "credit_risk.settings"]
    assert bool(warnings) is warns
    if warns:
        assert "8000" in warnings[0].getMessage()


@pytest.mark.parametrize("raw", ["0", "65536", "70000", "-1"])
def test_api_port_rejects_impossible_values(monkeypatch, raw):
    monkeypatch.setenv("PORT", raw)
    with pytest.raises(ValueError, match="between 1 and 65535"):
        settings.api_port()


def test_main_starts_on_default_port_when_port_is_blank(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    monkeypatch.setattr(api_main.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))
    monkeypatch.setenv("PORT", "")
    assert api_main.main() == 0
    assert calls["port"] == 8000
    assert calls["workers"] == 1


def test_main_exits_with_one_line_on_impossible_port(monkeypatch, capsys):
    calls: dict = {}
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    monkeypatch.setattr(api_main.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))
    monkeypatch.setenv("PORT", "70000")
    assert api_main.main() == 2
    assert calls == {}
    err = capsys.readouterr().err
    assert err.strip() == "PORT must be an integer between 1 and 65535, got 70000"


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, "INFO", False),
        ("", "INFO", False),
        ("debug", "DEBUG", False),
        (" Warning ", "WARNING", False),
        ("ERROR", "ERROR", False),
        ("verbose", "INFO", True),
    ],
    ids=["unset", "empty", "lower", "padded", "upper", "unknown"],
)
def test_log_level_helper(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", raw)
    with caplog.at_level(logging.WARNING, logger="credit_risk.settings"):
        assert settings.log_level() == expected
    warnings = [record for record in caplog.records if record.name == "credit_risk.settings"]
    assert bool(warnings) is warns


def test_configure_logging_honours_log_level(monkeypatch, tmp_path):
    """The root level follows LOG_LEVEL with the YAML config and with the basicConfig fallback."""
    if not LOGGING_CONFIG_PATH.is_file():
        pytest.skip("configs/logging.yaml is not in this checkout")
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        configure_logging()
        assert root.level == logging.DEBUG
        assert logging.getLogger("uvicorn.access").level == logging.WARNING

        root.handlers[:] = []  # basicConfig configures only a root without handlers
        monkeypatch.setenv("LOG_LEVEL", "warning")
        configure_logging(tmp_path / "missing.yaml")
        assert root.level == logging.WARNING
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_predict_cli_scores_csv(tmp_path, presets, capsys):
    from credit_risk import predict

    rows = pd.DataFrame([preset["input"] for preset in presets])
    rows.insert(0, "id", [preset["source_row_id"] for preset in presets])
    source = tmp_path / "rows.csv"
    rows.to_csv(source, index=False)
    output = tmp_path / "scored.csv"

    assert predict.main(["--input", str(source), "--output", str(output)]) == 0
    scored = pd.read_csv(output)
    assert list(scored.columns) == [*rows.columns, "probability", "decision"]
    for value, preset in zip(scored["probability"], presets, strict=True):
        assert abs(value - preset["model_probability"]) < 1e-6
    assert set(scored["decision"]) <= {"likely_default", "unlikely_default"}
    summary = capsys.readouterr().out.strip().splitlines()[-1]
    assert summary.startswith("scored 3 rows, mean probability ")


def test_predict_cli_rejects_missing_columns(tmp_path, presets, capsys):
    from credit_risk import predict

    rows = pd.DataFrame([preset["input"] for preset in presets]).drop(columns=["age"])
    source = tmp_path / "rows.csv"
    rows.to_csv(source, index=False)
    assert predict.main(["--input", str(source), "--output", str(tmp_path / "out.csv")]) == 2
    assert "AGE" in capsys.readouterr().err
