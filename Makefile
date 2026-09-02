# Convenience targets. Every recipe is a single command that also runs as-is in PowerShell.
# Override the interpreter with: make PYTHON=.venv/Scripts/python.exe <target>

PYTHON ?= python
IMAGE ?= ghcr.io/zulqarnain-10/credit-risk-service:latest
API_URL ?= http://127.0.0.1:8000
LOADTEST_HOST ?= local uvicorn, Windows 11, Python 3.12, single process

.PHONY: setup data repro train test lint format serve loadtest drift docker-build docker-run sync-readme clean

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt -e .

data:
	$(PYTHON) -m credit_risk.data fetch

repro:
	$(PYTHON) -m dvc repro

train:
	$(PYTHON) -m credit_risk.train

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

serve:
	$(PYTHON) -m credit_risk.api

loadtest:
	$(PYTHON) -m credit_risk.loadtest --url $(API_URL) --requests 300 --concurrency 10 --host "$(LOADTEST_HOST)"

drift:
	$(PYTHON) -m credit_risk.monitoring drift

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

sync-readme:
	$(PYTHON) -m credit_risk.sync_readme

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.space_build', 'htmlcov', 'build', 'dist'] + [str(q) for d in ('src', 'tests', 'scripts') for q in pathlib.Path(d).rglob('__pycache__')]]; [pathlib.Path(f).unlink(missing_ok=True) for f in ('.coverage', 'coverage.xml')]"
