"""Load test for the prediction endpoint.

Usage:
    python -m credit_risk.loadtest --url http://127.0.0.1:8000 --requests 300 --concurrency 10 \
        --host "local uvicorn, Windows 11, Python 3.12"

Sends the first demo preset to POST /predict with httpx.AsyncClient under an asyncio semaphore.
One warm-up request is sent first and excluded from every number. Percentiles are read from the
sorted latencies (nearest rank). Errors are non-200 responses plus exceptions. GET /version is
read once before the run so the receipt names the model it measured; when that call fails the
receipt simply carries no model_version.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from credit_risk import settings

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "/predict"
DEFAULT_REQUESTS = 300
DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT_S = 30.0


def default_host() -> str:
    return f"{platform.system()} {platform.release()}, Python {platform.python_version()}"


def load_payload(presets_path: Path = settings.PRESETS_PATH) -> tuple[str, dict]:
    """First preset's id and API input from configs/presets.json."""
    with open(presets_path, encoding="utf-8") as fh:
        presets = json.load(fh)["presets"]
    if not presets:
        raise ValueError(f"no presets in {presets_path}")
    first = presets[0]
    return str(first["id"]), dict(first["input"])


VERSION_KEYS = ("model_version", "git_sha_short")


def fetch_version(base: str, timeout_s: float) -> dict[str, str]:
    """model_version and git_sha_short from GET <base>/version; empty when unavailable."""
    try:
        response = httpx.get(base + "/version", timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("%s/version unavailable (%s); receipt carries no model version", base, exc)
        return {}
    if not isinstance(body, dict):
        log.warning("%s/version returned a non-object body; receipt carries no model version", base)
        return {}
    identity = {key: body[key] for key in VERSION_KEYS if isinstance(body.get(key), str)}
    return {key: value for key, value in identity.items() if value}


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    if not sorted_values:
        raise ValueError("no values")
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


async def _one_request(
    client: httpx.AsyncClient, url: str, payload: dict, semaphore: asyncio.Semaphore
) -> tuple[float, int | None, str | None]:
    """Latency in ms, status code (None on exception), and the error text if any."""
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            return (time.perf_counter() - start) * 1000.0, None, type(exc).__name__
        latency = (time.perf_counter() - start) * 1000.0
        error = None if response.status_code == 200 else f"http_{response.status_code}"
        return latency, response.status_code, error


async def run_load(
    url: str, payload: dict, n_requests: int, concurrency: int, timeout_s: float
) -> dict:
    """Warm up once, then fire n_requests with at most `concurrency` in flight."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        warmup_latency, warmup_status, warmup_error = await _one_request(
            client, url, payload, semaphore
        )
        if warmup_status is None:
            raise ConnectionError(f"warm-up request to {url} failed: {warmup_error}")
        if warmup_error:
            log.warning("warm-up returned %s; continuing", warmup_error)
        started = time.perf_counter()
        results = await asyncio.gather(
            *(_one_request(client, url, payload, semaphore) for _ in range(n_requests))
        )
        duration_s = time.perf_counter() - started
    return {
        "warmup": {"latency_ms": warmup_latency, "status": warmup_status},
        "results": results,
        "duration_s": duration_s,
    }


def summarize(run: dict, n_requests: int) -> dict:
    results = run["results"]
    latencies = sorted(lat for lat, status, _ in results if status is not None)
    errors = sum(1 for _, _, error in results if error)
    status_counts: dict[str, int] = {}
    for _, status, error in results:
        key = str(status) if status is not None else str(error)
        status_counts[key] = status_counts.get(key, 0) + 1
    duration_s = run["duration_s"]
    stats = {
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "mean_ms": None,
        "min_ms": None,
        "max_ms": None,
    }
    if latencies:
        stats = {
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "p99_ms": round(percentile(latencies, 99), 2),
            "mean_ms": round(sum(latencies) / len(latencies), 2),
            "min_ms": round(latencies[0], 2),
            "max_ms": round(latencies[-1], 2),
        }
    return {
        **stats,
        "rps": round(n_requests / duration_s, 2) if duration_s > 0 else None,
        "error_rate": round(errors / n_requests, 4) if n_requests else 0.0,
        "errors": int(errors),
        "duration_s": round(duration_s, 3),
        "status_counts": status_counts,
        "warmup_excluded": 1,
        "latency_basis": "responses of any status; exceptions count as errors without latency",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m credit_risk.loadtest", description="Measure prediction latency."
    )
    parser.add_argument("--url", required=True, help="service base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="measured requests")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="in flight")
    parser.add_argument("--host", default=None, help="where the service ran, for the receipt")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="POST path to hit")
    parser.add_argument(
        "--output", type=Path, default=settings.LOADTEST_PATH, help="where the JSON receipt goes"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    if args.requests < 1 or args.concurrency < 1:
        log.error("--requests and --concurrency must be at least 1")
        return 2
    preset_id, payload = load_payload()
    base = args.url.rstrip("/")
    endpoint = "/" + args.endpoint.lstrip("/")
    target = base + endpoint
    identity = fetch_version(base, args.timeout)
    try:
        run = asyncio.run(run_load(target, payload, args.requests, args.concurrency, args.timeout))
    except ConnectionError as exc:
        log.error("%s", exc)
        return 1
    summary = summarize(run, args.requests)
    receipt = {
        "url": base,
        "host": args.host or default_host(),
        "endpoint": endpoint,
        "requests": int(args.requests),
        "concurrency": int(args.concurrency),
        "payload_preset": preset_id,
        **identity,
        **summary,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "client": {"python": platform.python_version(), "httpx": httpx.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(receipt, fh, indent=2)
        fh.write("\n")
    print(
        f"loadtest {endpoint} model={receipt.get('model_version', 'unknown')} "
        f"requests={receipt['requests']} concurrency={receipt['concurrency']} "
        f"p50={receipt['p50_ms']}ms p95={receipt['p95_ms']}ms p99={receipt['p99_ms']}ms "
        f"mean={receipt['mean_ms']}ms rps={receipt['rps']} errors={receipt['errors']} "
        f"error_rate={receipt['error_rate']} -> {args.output}"
    )
    return 0 if receipt["error_rate"] < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
