"""Shared fixtures: a TestClient with the lifespan run, and the preset applicants.

PREDICTION_LOG_PATH is set before the app module is imported so no test appends to the
machine's temp directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk import settings


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def prediction_log_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("prediction_log") / "predictions.jsonl"
    os.environ["PREDICTION_LOG_PATH"] = str(path)
    return path


@pytest.fixture(scope="session")
def app(prediction_log_path: Path):
    from credit_risk.api.app import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app) -> Iterator[TestClient]:
    """A client whose lifespan has run, so the model is loaded once per session."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def version_info() -> dict:
    return _read_json(settings.VERSION_PATH)


@pytest.fixture(scope="session")
def metrics() -> dict:
    return _read_json(settings.METRICS_PATH)


@pytest.fixture(scope="session")
def importance() -> dict:
    return _read_json(settings.IMPORTANCE_PATH)


@pytest.fixture(scope="session")
def presets() -> list[dict]:
    return _read_json(settings.PRESETS_PATH)["presets"]


@pytest.fixture
def preset_payload(presets: list[dict]) -> dict:
    """The first preset's input, as the API expects it. A fresh copy per test."""
    return dict(presets[0]["input"])
