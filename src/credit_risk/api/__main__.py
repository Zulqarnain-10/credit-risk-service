"""Run the API with uvicorn: python -m credit_risk.api

Port comes from PORT (default 8000; Hugging Face Spaces sets 7860). A blank or non-numeric
PORT falls back to 8000 with a warning; a port outside 1..65535 exits with a one-line message.
One worker, host 0.0.0.0.
"""

from __future__ import annotations

import sys

import uvicorn

from credit_risk import settings
from credit_risk.api.app import configure_logging


def main() -> int:
    configure_logging()
    try:
        port = settings.api_port()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    uvicorn.run(
        "credit_risk.api.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
