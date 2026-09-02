"""Stage and push the Hugging Face Space that serves credit-risk-service.

Copies exactly what the Dockerfile needs into .space_build/, writes a Space README with the
front matter Hugging Face expects (that README never enters the git repository), and uploads
the folder to spaces/<hf_user>/<space> with the commit message "Deploy <sha>". The token is
read from the HF_TOKEN environment variable only. --dry-run stages and lists the files
without uploading.

    python scripts/deploy_space.py --hf-user <user> --space credit-risk-service --sha <sha>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = REPO_ROOT / ".space_build"
GITHUB_URL = "https://github.com/Zulqarnain-10/credit-risk-service"
SPACE_PORT = 7860

REQUIRED_FILES = (
    "Dockerfile",
    "pyproject.toml",
    "requirements.txt",
    "params.yaml",
    "models/model.joblib",
    "models/version.json",
    "reports/metrics.json",
    "reports/importance.json",
)
OPTIONAL_FILES = ("reports/loadtest.json",)
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".DS_Store")

BUILDING_STAGES = frozenset({"BUILDING", "RUNNING_BUILDING", "APP_STARTING"})
ERROR_STAGES = frozenset(
    {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "STOPPED", "PAUSED", "DELETING"}
)

README_FRONT_MATTER = """---
title: Credit risk service
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Credit-default risk model as a tested, monitored API
---
"""

README_BODY = """
# Credit risk service

Live endpoint for the credit-default risk model built in
[Zulqarnain-10/credit-risk-service]({github}). The pipeline (DVC and MLflow), the FastAPI
service, the tests, the CI/CD workflows, the model card, and every reported number live in
that repository. This Space only runs the container image that the repository Dockerfile
builds{deployed_from}. The demo page is at the root, the OpenAPI schema at /docs, and
/health, /version, and /metrics report service state.

Demonstration system trained on the 2005 Taiwan credit-card dataset (UCI id 350). Not a U.S.
underwriting model. Do not use it for real credit decisions. Free Spaces sleep after a period
of inactivity and take a few seconds to wake. The prediction log written by the service is
ephemeral here.
"""


def current_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def copy_file(relative: str, required: bool) -> None:
    source = REPO_ROOT / relative
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"required file is missing: {relative}")
        return
    target = STAGE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def patch_port(dockerfile: Path) -> None:
    """Default the container port to 7860, which Docker Spaces expect."""
    text = dockerfile.read_text(encoding="utf-8")
    if text.count("PORT=8000") != 1:
        print("warning: could not patch the default PORT in the staged Dockerfile", file=sys.stderr)
        return
    dockerfile.write_text(
        text.replace("PORT=8000", f"PORT={SPACE_PORT}"), encoding="utf-8", newline="\n"
    )


def write_readme(sha: str) -> None:
    if sha == "unknown":
        deployed_from = ""
    else:
        deployed_from = f", deployed from commit [{sha[:7]}]({GITHUB_URL}/commit/{sha})"
    body = README_BODY.format(github=GITHUB_URL, deployed_from=deployed_from)
    (STAGE_DIR / "README.md").write_text(README_FRONT_MATTER + body, encoding="utf-8", newline="\n")


def stage(sha: str) -> list[Path]:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir()
    for relative in REQUIRED_FILES:
        copy_file(relative, required=True)
    for relative in OPTIONAL_FILES:
        copy_file(relative, required=False)
    shutil.copytree(
        REPO_ROOT / "src" / "credit_risk", STAGE_DIR / "src" / "credit_risk", ignore=COPY_IGNORE
    )
    shutil.copytree(REPO_ROOT / "configs", STAGE_DIR / "configs", ignore=COPY_IGNORE)
    patch_port(STAGE_DIR / "Dockerfile")
    write_readme(sha)
    return sorted(path for path in STAGE_DIR.rglob("*") if path.is_file())


def print_files(files: list[Path]) -> None:
    total = 0
    print(f"Staged {len(files)} files in {STAGE_DIR.relative_to(REPO_ROOT).as_posix()}/")
    for path in files:
        size = path.stat().st_size
        total += size
        print(f"  {size:>10,d}  {path.relative_to(STAGE_DIR).as_posix()}")
    print(f"  {total:>10,d}  total bytes")


def wait_for_build(api, repo_id: str, timeout: int) -> int:
    """Poll the Space runtime until the new build is running. Returns 1 on a build error."""
    start = time.monotonic()
    seen_building = False
    last = None
    while time.monotonic() - start < timeout:
        try:
            runtime = api.get_space_runtime(repo_id)
            stage_name = str(getattr(runtime.stage, "value", runtime.stage))
        except Exception as exc:  # the wait is best effort; the smoke test is the real check
            stage_name = f"unknown ({type(exc).__name__})"
        if stage_name != last:
            print(f"Space stage: {stage_name} ({time.monotonic() - start:.0f} s)")
            last = stage_name
        if stage_name in ERROR_STAGES:
            print(f"error: Space is in stage {stage_name}", file=sys.stderr)
            return 1
        if stage_name in BUILDING_STAGES:
            seen_building = True
        elif stage_name == "RUNNING" and (seen_building or time.monotonic() - start > 120):
            print("Space is running.")
            return 0
        time.sleep(15)
    print(
        f"warning: Space did not reach RUNNING within {timeout} s; the smoke test keeps polling",
        file=sys.stderr,
    )
    return 0


def upload(hf_user: str, space: str, sha: str, token: str, wait_seconds: int) -> int:
    # Imported here so --dry-run works without huggingface_hub installed.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{hf_user}/{space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    try:
        api.add_space_variable(repo_id, "PORT", str(SPACE_PORT))
    except Exception as exc:  # the staged Dockerfile already defaults to 7860
        print(f"warning: could not set the PORT variable on the Space: {exc}", file=sys.stderr)
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy {sha}",
        delete_patterns=["*"],
    )
    print(f"Pushed {repo_id}: https://huggingface.co/spaces/{repo_id}")
    if wait_seconds > 0:
        return wait_for_build(api, repo_id, wait_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hf-user", required=True, help="Hugging Face user or organisation")
    parser.add_argument("--space", default="credit-risk-service", help="Space name")
    parser.add_argument("--sha", help="commit being deployed (default: git rev-parse HEAD)")
    parser.add_argument("--dry-run", action="store_true", help="stage and list, do not upload")
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="seconds to wait for the Space build after pushing (default 0: do not wait)",
    )
    args = parser.parse_args(argv)

    sha = args.sha or current_sha()
    files = stage(sha)
    print_files(files)
    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN is not set", file=sys.stderr)
        return 2
    return upload(args.hf_user, args.space, sha, token, args.wait)


if __name__ == "__main__":
    sys.exit(main())
