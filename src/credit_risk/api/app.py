"""FastAPI application: demo page, prediction endpoints, receipts, and Prometheus metrics.

The model and its reports load once in the lifespan handler. Every prediction is appended
to a JSONL log; a failure to write that log is a warning, never an error for the caller.
"""

from __future__ import annotations

import json
import logging
import logging.config
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from credit_risk import __version__, settings
from credit_risk.api import model_loader
from credit_risk.api.schemas import (
    DECISION_LABELS,
    Applicant,
    BatchRequest,
    BatchResponse,
    Health,
    Prediction,
    applicants_to_frame,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
LOGGING_CONFIG_PATH = settings.CONFIGS_DIR / "logging.yaml"

MAX_BODY_BYTES = 262_144

TITLE = "Credit risk service"
DESCRIPTION = (
    "Scores the probability that a credit-card holder defaults next month and returns a "
    "decision at the cost-optimal threshold chosen on the validation split. Request bodies "
    f"above {MAX_BODY_BYTES // 1024} KB are refused with status 413. " + settings.DISCLAIMER
)

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Credit risk service</title></head>
<body>
<main>
<h1>Credit risk service</h1>
<p>The demo page is not bundled with this build. The API is up: see <a href="/docs">/docs</a>,
<a href="/version">/version</a>, and <a href="/health">/health</a>.</p>
<p>{disclaimer}</p>
</main>
</body>
</html>
"""

PREDICTIONS_TOTAL = Counter(
    "credit_risk_predictions_total",
    "Predictions served, by decision at the cost-optimal threshold.",
    ["decision"],
)
for _decision in DECISION_LABELS:
    PREDICTIONS_TOTAL.labels(decision=_decision)


def configure_logging(config_path: Path | None = None) -> None:
    """Structured JSON logs from configs/logging.yaml when present, else basicConfig.

    The root level comes from LOG_LEVEL (default INFO) in both cases.
    """
    config_path = config_path or LOGGING_CONFIG_PATH
    level = settings.log_level()
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                logging.config.dictConfig(yaml.safe_load(fh))
            logging.getLogger().setLevel(level)
            return
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
            logging.basicConfig(level=level)
            log.warning("could not apply %s, using basicConfig: %s", config_path, exc)
            return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class BodySizeLimit:
    """ASGI middleware that refuses request bodies above max_bytes with status 413.

    A declared Content-Length above the cap is refused before the app runs. A chunked body
    is counted as it streams and refused once it passes the cap, before the JSON parser
    sees the whole payload.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.detail = f"request body exceeds {max_bytes} bytes"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": self.detail}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing, so this becomes a 413.
                    raise HTTPException(status_code=413, detail=self.detail)
            return message

        await self.app(scope, limited_receive, send)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats, which JSON cannot carry, with their text form."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def validation_error_422(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 body, made renderable when the request carried NaN or Infinity.

    Python's json module emits and accepts those tokens; the default handler echoes the
    offending input and the JSON encoder then refuses it, turning a 422 into a 500.
    """
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


def read_index_html() -> str:
    """The demo page when it ships with the package, else a small placeholder."""
    if INDEX_PATH.is_file():
        return INDEX_PATH.read_text(encoding="utf-8")
    return PLACEHOLDER_HTML.format(disclaimer=settings.DISCLAIMER)


def append_prediction_log(path: Path, records: list[dict]) -> None:
    """Append one JSON line per record. Never raises: a failed write is a warning."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as exc:
        log.warning("prediction log write failed at %s: %s", path, exc)


def get_bundle(request: Request) -> model_loader.ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return bundle


def score(
    bundle: model_loader.ModelBundle, items: list[Applicant], log_path: Path
) -> list[Prediction]:
    """Score applicants in one pipeline call, count them, and log them.

    The decision compares the four-decimal probability the response carries against the
    threshold, so a displayed probability equal to the threshold reads likely_default.
    """
    probabilities = bundle.predict_proba(applicants_to_frame(items))
    ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    predictions: list[Prediction] = []
    records: list[dict] = []
    for item, probability in zip(items, probabilities, strict=True):
        p = float(probability)
        shown = round(p, 4)
        decision = "likely_default" if shown >= bundle.threshold else "unlikely_default"
        PREDICTIONS_TOTAL.labels(decision=decision).inc()
        predictions.append(
            Prediction(
                probability=shown,
                threshold=bundle.threshold,
                decision=decision,
                decision_label=DECISION_LABELS[decision],
                model=bundle.model_name,
                model_version=bundle.model_version,
                disclaimer=settings.DISCLAIMER,
            )
        )
        records.append(
            {
                "ts": ts,
                "model_version": bundle.model_version,
                "probability": round(p, 6),
                "decision": decision,
                "inputs": item.model_dump(),
            }
        )
    append_prediction_log(log_path, records)
    return predictions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load artifacts once; resolve the prediction log path once."""
    if not logging.getLogger().handlers:
        configure_logging()
    app.state.bundle = model_loader.load_bundle()
    app.state.prediction_log_path = settings.prediction_log_path()
    log.info(
        "api ready",
        extra={
            "model_version": app.state.bundle.model_version,
            "prediction_log": str(app.state.prediction_log_path),
        },
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=TITLE,
        version=__version__,
        description=DESCRIPTION,
        license_info={"name": "MIT"},
        contact={"name": "Syed Zulqarnain Hassan", "url": "https://zulqarnainhassan.com"},
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_error_422)
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(read_index_html())

    @app.get("/health", response_model=Health, tags=["service"])
    def health(request: Request) -> Health:
        bundle = get_bundle(request)
        return Health(status="ok", model_loaded=True, model_version=bundle.model_version)

    @app.get("/version", tags=["service"])
    def version(request: Request) -> dict[str, Any]:
        """models/version.json plus headline test metrics and the disclaimer."""
        return model_loader.version_payload(get_bundle(request))

    @app.get("/presets", tags=["receipts"])
    def presets(request: Request) -> dict[str, Any]:
        """Three real held-out applicants from configs/presets.json."""
        bundle = get_bundle(request)
        if bundle.presets is None:
            raise HTTPException(status_code=404, detail="presets.json is not in this build")
        return bundle.presets

    @app.get("/importance", tags=["receipts"])
    def importance(request: Request) -> dict[str, Any]:
        """Permutation importance from reports/importance.json."""
        bundle = get_bundle(request)
        if bundle.importance is None:
            raise HTTPException(status_code=404, detail="importance.json is not in this build")
        return bundle.importance

    @app.post("/predict", response_model=Prediction, tags=["predict"])
    def predict(applicant: Applicant, request: Request) -> Prediction:
        """Probability of default next month and the decision at the cost-optimal threshold."""
        bundle = get_bundle(request)
        return score(bundle, [applicant], request.app.state.prediction_log_path)[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Score 1 to 100 applicants in one call; predictions keep the request order."""
        bundle = get_bundle(request)
        predictions = score(bundle, batch.items, request.app.state.prediction_log_path)
        return BatchResponse(count=len(predictions), predictions=predictions)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # The handler label for the mounted StaticFiles app is the mount path "/static", so the
    # pattern anchors on the prefix rather than requiring a trailing slash.
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["^/metrics$", "^/static"],
    ).instrument(app).expose(app, tags=["service"])
    return app


app = create_app()
