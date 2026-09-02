"""Fetch the UCI id 350 archive, verify its hashes, and write the raw CSV.

Usage:
    python -m credit_risk.data fetch
    python -m credit_risk.data info

The archive is downloaded from params.data.url, checked against params.data.zip_sha256,
and the extracted workbook is checked against params.data.xls_sha256. A hash mismatch is a
hard failure. If the download itself fails after retries, the ucimlrepo package is used as a
fallback and the manifest records that source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_risk import settings

log = logging.getLogger(__name__)

FETCH_MANIFEST_PATH = settings.RAW_DIR / "fetch_manifest.json"
EXPECTED_COLUMNS: tuple[str, ...] = (
    settings.ID_COLUMN,
    *settings.RAW_FEATURE_COLUMNS,
    settings.TARGET,
)
USER_AGENT = "credit-risk-service/0.1 (+https://github.com/Zulqarnain-10/credit-risk-service)"


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, retries: int = 3, backoff: float = 2.0) -> None:
    """Download url to dest, retrying with exponential backoff on any network error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            log.info("downloaded %s (%d bytes)", url, dest.stat().st_size)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            log.warning("download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise RuntimeError(f"could not download {url} after {retries} attempts") from last_error


def verify_sha256(path: Path, expected: str, label: str) -> str:
    """Return the file's sha256 or raise if it differs from the expected value."""
    actual = sha256_of(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} sha256 mismatch for {path.name}: expected {expected}, got {actual}. "
            "The upstream file changed or the download is corrupt; refusing to continue."
        )
    log.info("%s sha256 verified: %s", label, actual)
    return actual


def _extract_xls(zip_path: Path, xls_name: str, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if xls_name not in members:
            raise RuntimeError(f"{xls_name!r} not found in archive; members: {members}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(xls_name) as src, open(dest, "wb") as out:
            out.write(src.read())
    return dest


def _check_columns(df: pd.DataFrame) -> pd.DataFrame:
    got = list(df.columns)
    if got != list(EXPECTED_COLUMNS):
        raise RuntimeError(f"unexpected raw columns: {got}")
    return df


def fetch_from_archive(params: dict) -> tuple[pd.DataFrame, dict]:
    """Download the UCI zip, verify both hashes, and read the workbook."""
    data = params["data"]
    download(data["url"], settings.RAW_ZIP_PATH)
    zip_sha = verify_sha256(settings.RAW_ZIP_PATH, data["zip_sha256"], "zip")
    xls_path = _extract_xls(settings.RAW_ZIP_PATH, data["xls_name"], settings.RAW_XLS_PATH)
    xls_sha = verify_sha256(xls_path, data["xls_sha256"], "xls")
    df = pd.read_excel(xls_path, header=1, engine="xlrd")
    df = df.rename(columns={settings.RAW_TARGET_NAME: settings.TARGET})
    if df.columns[0] != settings.ID_COLUMN:
        df = df.rename(columns={df.columns[0]: settings.ID_COLUMN})
    df = _check_columns(df)
    manifest = {
        "data_source": "uci_archive",
        "url": data["url"],
        "zip_sha256": zip_sha,
        "xls_sha256": xls_sha,
    }
    return df, manifest


def fetch_from_ucimlrepo(params: dict) -> tuple[pd.DataFrame, dict]:
    """Fallback: pull the dataset through ucimlrepo and map X1..X23 / Y to the UCI names."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=int(params["data"]["uci_id"]))
    features = dataset.data.features
    targets = dataset.data.targets
    if features.shape[1] != len(settings.RAW_FEATURE_COLUMNS):
        raise RuntimeError(f"ucimlrepo returned {features.shape[1]} feature columns, expected 23")
    features = features.copy()
    features.columns = list(settings.RAW_FEATURE_COLUMNS)
    target = targets.iloc[:, 0].rename(settings.TARGET)
    ids = dataset.data.ids
    if ids is not None and ids.shape[1] >= 1:
        id_col = ids.iloc[:, 0].rename(settings.ID_COLUMN)
    else:
        id_col = pd.Series(range(1, len(features) + 1), name=settings.ID_COLUMN)
    df = pd.concat([id_col.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df[settings.TARGET] = target.reset_index(drop=True)
    df = _check_columns(df)
    manifest = {
        "data_source": "ucimlrepo",
        "url": params["data"]["url"],
        "zip_sha256": None,
        "xls_sha256": None,
    }
    return df, manifest


def fetch(params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fetch the raw frame, preferring the verified archive and falling back to ucimlrepo."""
    params = params or settings.load_params()
    try:
        df, manifest = fetch_from_archive(params)
    except RuntimeError as exc:
        if "sha256 mismatch" in str(exc) or "unexpected raw columns" in str(exc):
            raise
        log.warning("archive download failed (%s); falling back to ucimlrepo", exc)
        df, manifest = fetch_from_ucimlrepo(params)
    return df, manifest


def write_raw_csv(df: pd.DataFrame, path: Path = settings.RAW_CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def cmd_fetch(_: argparse.Namespace) -> int:
    params = settings.load_params()
    df, manifest = fetch(params)
    path = write_raw_csv(df)
    positive_rate = float(df[settings.TARGET].mean())
    manifest.update(
        {
            "csv_path": str(path.relative_to(settings.REPO_ROOT)),
            "csv_sha256": sha256_of(path),
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "positive_rate": round(positive_rate, 4),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
    )
    with open(FETCH_MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(
        f"wrote {path} rows={df.shape[0]} cols={df.shape[1]} "
        f"positive_rate={positive_rate:.4f} source={manifest['data_source']}"
    )
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    params = settings.load_params()["data"]
    if not settings.RAW_CSV_PATH.exists():
        print(f"raw CSV not found at {settings.RAW_CSV_PATH}")
        print("run 'python -m credit_risk.data fetch' first")
        return 1
    df = pd.read_csv(settings.RAW_CSV_PATH)
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"positive rate ({settings.TARGET}): {df[settings.TARGET].mean():.4f}")
    print(f"expected zip sha256: {params['zip_sha256']}")
    print(f"expected xls sha256: {params['xls_sha256']}")
    for label, path in (("zip", settings.RAW_ZIP_PATH), ("xls", settings.RAW_XLS_PATH)):
        if path.exists():
            print(f"observed {label} sha256: {sha256_of(path)}")
        else:
            print(f"observed {label} sha256: file not present ({path.name})")
    print(f"csv sha256: {sha256_of(settings.RAW_CSV_PATH)}")
    if FETCH_MANIFEST_PATH.exists():
        with open(FETCH_MANIFEST_PATH, encoding="utf-8") as fh:
            print(f"data source: {json.load(fh).get('data_source')}")
    print(f"nulls: {int(df.isna().sum().sum())}")
    duplicates = int(df.drop(columns=[settings.ID_COLUMN]).duplicated().sum())
    print(f"exact duplicate rows (ignoring ID): {duplicates}")
    print("column ranges (min, max):")
    for column in df.columns:
        print(f"  {column}: {int(df[column].min())}, {int(df[column].max())}")
    print("category counts:")
    for column in ("SEX", "EDUCATION", "MARRIAGE", settings.TARGET):
        counts = df[column].value_counts().sort_index()
        print(f"  {column}: {counts.to_dict()}")
    print("rows with a negative bill:")
    for column in settings.BILL_COLUMNS:
        print(f"  {column}: {int((df[column] < 0).sum())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m credit_risk.data", description="Fetch and inspect the raw dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download, verify, and write data/raw/credit_default_raw.csv")
    sub.add_parser("info", help="print shape, positive rate, and sha256 values")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"fetch": cmd_fetch, "info": cmd_info}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
