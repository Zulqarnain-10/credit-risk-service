"""Load the model and its receipts once, and score raw input frames.

The loader reads under settings.MODELS_DIR, settings.REPORTS_DIR, and settings.CONFIGS_DIR.
The model, version.json, and metrics.json are required. presets.json, importance.json, and
loadtest.json are optional so a slim serving image still starts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk import settings

log = logging.getLogger(__name__)

VERSION_METRIC_KEYS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "brier",
    "ks",
    "threshold_cost_optimal",
    "n_test",
    "trained_at",
)
LOADTEST_KEYS: tuple[str, ...] = (
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "rps",
    "error_rate",
    "requests",
    "concurrency",
    "host",
    "timestamp",
)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_json_optional(path: Path) -> dict | None:
    """The file's contents, or None with a warning when it is absent."""
    if not path.is_file():
        log.warning("optional artifact missing: %s", path)
        return None
    return read_json(path)


@dataclass
class ModelBundle:
    """The fitted pipeline plus everything the endpoints answer from."""

    pipeline: object
    version: dict
    metrics: dict
    threshold: float
    presets: dict | None = None
    importance: dict | None = None

    @property
    def model_name(self) -> str:
        return str(self.version.get("model", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.version.get("model_version", "unknown"))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Default probabilities for a frame that holds settings.MODEL_INPUT_COLUMNS."""
        inputs = frame[list(settings.MODEL_INPUT_COLUMNS)]
        return np.asarray(self.pipeline.predict_proba(inputs)[:, 1], dtype=float)


def load_bundle(
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    configs_dir: Path | None = None,
) -> ModelBundle:
    """Read the artifacts from disk. Call once per process."""
    models_dir = models_dir or settings.MODELS_DIR
    reports_dir = reports_dir or settings.REPORTS_DIR
    configs_dir = configs_dir or settings.CONFIGS_DIR

    model_path = models_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    pipeline = joblib.load(model_path)
    version = read_json(models_dir / "version.json")
    metrics = read_json(reports_dir / "metrics.json")
    threshold = float(metrics["threshold_cost_optimal"])
    bundle = ModelBundle(
        pipeline=pipeline,
        version=version,
        metrics=metrics,
        threshold=threshold,
        presets=read_json_optional(configs_dir / "presets.json"),
        importance=read_json_optional(reports_dir / "importance.json"),
    )
    log.info(
        "model artifacts loaded",
        extra={
            "model": bundle.model_name,
            "model_version": bundle.model_version,
            "threshold": threshold,
            "models_dir": str(models_dir),
        },
    )
    return bundle


def loadtest_block(path: Path | None = None) -> dict | None:
    """Latency receipt for /version. Read per call so a new load test shows without a restart."""
    path = path or settings.LOADTEST_PATH
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    return {key: data[key] for key in LOADTEST_KEYS if key in data}


def version_payload(bundle: ModelBundle) -> dict:
    """models/version.json plus headline metrics, the load test when present, and the disclaimer."""
    payload = dict(bundle.version)
    payload["metrics"] = {key: bundle.metrics.get(key) for key in VERSION_METRIC_KEYS}
    loadtest = loadtest_block()
    if loadtest is not None:
        payload["loadtest"] = loadtest
    payload["disclaimer"] = settings.DISCLAIMER
    return payload
