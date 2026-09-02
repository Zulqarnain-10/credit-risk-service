# Serving image for credit-risk-service.
# Stage 1 installs the pinned serving dependencies and the package into a virtualenv.
# Stage 2 copies that virtualenv plus the model, its receipts, and the runtime configs, nothing else.
# Runs as the non-root user "app" (uid 1000, which Hugging Face Spaces expects) and listens on $PORT.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Serving dependencies only: matplotlib, evidently, dvc, and mlflow stay out of this image.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# The package itself. Its dependencies are already pinned above, so skip resolution here.
# Then drop what the service never imports: bytecode caches, bundled test suites, and pip.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps . \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv/lib -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/pip \
              /opt/venv/lib/python3.12/site-packages/pip-*.dist-info \
              /opt/venv/bin/pip /opt/venv/bin/pip3 /opt/venv/bin/pip3.12

# Runtime files the service reads through CREDIT_RISK_HOME. loadtest.json is optional.
COPY params.yaml /stage/params.yaml
COPY configs/presets.json configs/logging.yaml /stage/configs/
COPY models/model.joblib models/version.json /stage/models/
COPY reports/ /tmp/reports/
RUN mkdir -p /stage/reports \
    && cp /tmp/reports/metrics.json /tmp/reports/importance.json /stage/reports/ \
    && if [ -f /tmp/reports/loadtest.json ]; then cp /tmp/reports/loadtest.json /stage/reports/; fi


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CREDIT_RISK_HOME=/app \
    HOME=/home/app \
    PORT=8000

RUN useradd --create-home --uid 1000 --user-group app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /stage/ /app/

USER app

EXPOSE 8000

# Slim images ship no curl, so the probe uses the standard library. PORT may be overridden (7860 on Spaces).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "credit_risk.api"]
