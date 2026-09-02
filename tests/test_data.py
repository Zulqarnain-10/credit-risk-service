"""Unit tests for credit_risk.data: hashes, archive extraction, download retries, fallbacks.

No network: urllib and ucimlrepo are replaced with stand-ins, the archive is a temporary zip,
and the workbook reader returns a synthetic frame.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk import data, settings

RAW_COLUMNS = list(data.EXPECTED_COLUMNS)
XLS_NAME = "default of credit card clients.xls"
XLS_BYTES = b"not a workbook, just bytes with a known hash"
URL = "https://example.invalid/uci.zip"


def raw_frame(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """A valid raw frame in dataset column order."""
    rng = np.random.default_rng(seed)
    frame: dict[str, np.ndarray] = {
        settings.ID_COLUMN: np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 101, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 80, n),
    }
    for c in settings.PAY_STATUS_COLUMNS:
        frame[c] = rng.integers(-2, 9, n)
    for c in settings.BILL_COLUMNS:
        frame[c] = rng.integers(-5_000, 300_001, n)
    for c in settings.PAY_AMT_COLUMNS:
        frame[c] = rng.integers(0, 60_001, n)
    frame[settings.TARGET] = rng.integers(0, 2, n)
    return pd.DataFrame(frame, columns=RAW_COLUMNS)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def zip_bytes_for(member: str = XLS_NAME, payload: bytes = XLS_BYTES) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def archive_params(zip_payload: bytes, xls_payload: bytes = XLS_BYTES) -> dict:
    return {
        "data": {
            "uci_id": 350,
            "url": URL,
            "zip_sha256": sha(zip_payload),
            "xls_sha256": sha(xls_payload),
            "xls_name": XLS_NAME,
            "target": settings.TARGET,
        }
    }


def fake_download(zip_payload: bytes):
    def _download(url, dest, retries=3, backoff=2.0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_payload)

    return _download


def excel_frame(first_column: str = settings.ID_COLUMN) -> pd.DataFrame:
    """What pd.read_excel returns: the dataset's own target name and a possibly unnamed ID."""
    renames = {settings.TARGET: settings.RAW_TARGET_NAME, settings.ID_COLUMN: first_column}
    return raw_frame().rename(columns=renames)


def install_fake_ucimlrepo(
    monkeypatch, n: int = 12, *, with_ids: bool = True, n_features: int = 23
):
    """Register a stand-in ucimlrepo module and return the frame it encodes plus the calls."""
    frame = raw_frame(n)
    features = frame[list(settings.RAW_FEATURE_COLUMNS)[:n_features]].copy()
    features.columns = [f"X{i}" for i in range(1, n_features + 1)]
    targets = frame[[settings.TARGET]].rename(columns={settings.TARGET: "Y"})
    ids = frame[[settings.ID_COLUMN]] if with_ids else None
    dataset = types.SimpleNamespace(
        data=types.SimpleNamespace(features=features, targets=targets, ids=ids)
    )
    calls: list[dict] = []

    def fetch_ucirepo(**kwargs):
        calls.append(kwargs)
        return dataset

    module = types.ModuleType("ucimlrepo")
    module.fetch_ucirepo = fetch_ucirepo
    monkeypatch.setitem(sys.modules, "ucimlrepo", module)
    return frame, calls


class FakeResponse:
    """A urlopen result that hands the body out in small chunks."""

    def __init__(self, payload: bytes, chunk: int = 7):
        self._buffer = io.BytesIO(payload)
        self._chunk = chunk

    def read(self, size: int = -1) -> bytes:
        limit = min(size, self._chunk) if size and size > 0 else self._chunk
        return self._buffer.read(limit)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point every raw-data path at a temporary folder."""
    raw = tmp_path / "raw"
    monkeypatch.setattr(settings, "RAW_DIR", raw)
    monkeypatch.setattr(settings, "RAW_ZIP_PATH", raw / "uci_350.zip")
    monkeypatch.setattr(settings, "RAW_XLS_PATH", raw / XLS_NAME)
    monkeypatch.setattr(settings, "RAW_CSV_PATH", raw / "credit_default_raw.csv")
    monkeypatch.setattr(data, "FETCH_MANIFEST_PATH", raw / "fetch_manifest.json")
    return raw


# ---------------------------------------------------------------------------
# Hashes, extraction, columns
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib_across_chunks(tmp_path: Path):
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 5_000
    path.write_bytes(payload)
    assert len(payload) > (1 << 20)
    assert data.sha256_of(path) == sha(payload)


def test_verify_sha256_accepts_any_case_and_rejects_a_wrong_hash(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    expected = sha(b"abc")
    assert data.verify_sha256(path, expected.upper(), "zip") == expected
    with pytest.raises(RuntimeError, match="zip sha256 mismatch"):
        data.verify_sha256(path, "0" * 64, "zip")


def test_extract_xls_writes_the_member_and_rejects_a_missing_one(tmp_path: Path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes_for())
    out = data._extract_xls(archive, XLS_NAME, tmp_path / "raw" / "book.xls")
    assert out == tmp_path / "raw" / "book.xls"
    assert out.read_bytes() == XLS_BYTES
    with pytest.raises(RuntimeError, match="not found in archive"):
        data._extract_xls(archive, "other.xls", tmp_path / "raw" / "other.xls")


def test_check_columns_requires_the_exact_order():
    frame = raw_frame()
    assert data._check_columns(frame) is frame
    with pytest.raises(RuntimeError, match="unexpected raw columns"):
        data._check_columns(frame.drop(columns=["AGE"]))
    with pytest.raises(RuntimeError, match="unexpected raw columns"):
        data._check_columns(frame[RAW_COLUMNS[::-1]])


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def test_download_streams_the_body_to_dest(tmp_path: Path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return FakeResponse(b"x" * 1000)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "nested" / "archive.zip"
    data.download(URL, dest)
    assert dest.read_bytes() == b"x" * 1000
    assert calls == [(URL, data.USER_AGENT, 60)]


def test_download_retries_with_backoff_then_raises(tmp_path: Path, monkeypatch):
    attempts = []
    sleeps = []

    def failing(request, timeout=None):
        attempts.append(request.full_url)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="could not download") as excinfo:
        data.download(URL, tmp_path / "x.zip", retries=3, backoff=2.0)
    assert len(attempts) == 3
    assert sleeps == [2.0, 4.0]
    assert isinstance(excinfo.value.__cause__, urllib.error.URLError)
    assert not (tmp_path / "x.zip").exists()


# ---------------------------------------------------------------------------
# Archive fetch and the ucimlrepo fallback
# ---------------------------------------------------------------------------


def test_fetch_from_archive_verifies_both_hashes_and_reads_the_workbook(raw_dir, monkeypatch):
    zip_payload = zip_bytes_for()
    monkeypatch.setattr(data, "download", fake_download(zip_payload))
    seen = {}

    def fake_read_excel(path, header=None, engine=None):
        seen.update({"path": Path(path), "header": header, "engine": engine})
        return excel_frame("Unnamed: 0")

    monkeypatch.setattr(data.pd, "read_excel", fake_read_excel)

    frame, manifest = data.fetch_from_archive(archive_params(zip_payload))

    assert list(frame.columns) == RAW_COLUMNS
    assert seen == {"path": raw_dir / XLS_NAME, "header": 1, "engine": "xlrd"}
    assert (raw_dir / "uci_350.zip").read_bytes() == zip_payload
    assert (raw_dir / XLS_NAME).read_bytes() == XLS_BYTES
    assert manifest == {
        "data_source": "uci_archive",
        "url": URL,
        "zip_sha256": sha(zip_payload),
        "xls_sha256": sha(XLS_BYTES),
    }


def test_fetch_refuses_a_zip_with_the_wrong_hash(raw_dir, monkeypatch):
    zip_payload = zip_bytes_for()
    monkeypatch.setattr(data, "download", fake_download(zip_payload))
    params = archive_params(zip_payload)
    params["data"]["zip_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="zip sha256 mismatch"):
        data.fetch_from_archive(params)

    def no_fallback(_params):
        pytest.fail("a hash mismatch must never fall back to ucimlrepo")

    monkeypatch.setattr(data, "fetch_from_ucimlrepo", no_fallback)
    with pytest.raises(RuntimeError, match="zip sha256 mismatch"):
        data.fetch(params)


def test_fetch_refuses_a_workbook_with_the_wrong_hash(raw_dir, monkeypatch):
    zip_payload = zip_bytes_for()
    monkeypatch.setattr(data, "download", fake_download(zip_payload))
    with pytest.raises(RuntimeError, match="xls sha256 mismatch"):
        data.fetch_from_archive(archive_params(zip_payload, xls_payload=b"different"))


def test_fetch_from_ucimlrepo_maps_columns(monkeypatch):
    expected, calls = install_fake_ucimlrepo(monkeypatch)
    frame, manifest = data.fetch_from_ucimlrepo(archive_params(b""))
    assert calls == [{"id": 350}]
    assert list(frame.columns) == RAW_COLUMNS
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    assert manifest == {
        "data_source": "ucimlrepo",
        "url": URL,
        "zip_sha256": None,
        "xls_sha256": None,
    }


def test_fetch_from_ucimlrepo_numbers_rows_without_ids(monkeypatch):
    install_fake_ucimlrepo(monkeypatch, with_ids=False)
    frame, _ = data.fetch_from_ucimlrepo(archive_params(b""))
    assert frame[settings.ID_COLUMN].tolist() == list(range(1, 13))


def test_fetch_from_ucimlrepo_rejects_a_wrong_feature_count(monkeypatch):
    install_fake_ucimlrepo(monkeypatch, n_features=22)
    with pytest.raises(RuntimeError, match="expected 23"):
        data.fetch_from_ucimlrepo(archive_params(b""))


def test_fetch_falls_back_to_ucimlrepo_when_the_download_fails(raw_dir, monkeypatch):
    def failing_download(url, dest, retries=3, backoff=2.0):
        raise RuntimeError(f"could not download {url} after {retries} attempts")

    monkeypatch.setattr(data, "download", failing_download)
    expected, calls = install_fake_ucimlrepo(monkeypatch)
    frame, manifest = data.fetch(archive_params(b""))
    assert calls == [{"id": 350}]
    assert manifest["data_source"] == "ucimlrepo"
    assert len(frame) == len(expected)


# ---------------------------------------------------------------------------
# CSV, manifest, and the command line
# ---------------------------------------------------------------------------


def test_write_raw_csv_uses_lf(tmp_path: Path):
    path = data.write_raw_csv(raw_frame(), tmp_path / "out" / "raw.csv")
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.split(b"\n")[0].decode("utf-8") == ",".join(RAW_COLUMNS)
    assert len(pd.read_csv(path)) == 12


def test_main_fetch_writes_the_csv_and_manifest(raw_dir, monkeypatch, capsys):
    frame = raw_frame(20)
    manifest_in = {"data_source": "uci_archive", "url": URL, "zip_sha256": "z", "xls_sha256": "x"}
    monkeypatch.setattr(data, "fetch", lambda params: (frame, dict(manifest_in)))
    monkeypatch.setattr(settings, "REPO_ROOT", raw_dir.parent)
    # write_raw_csv binds its default path at import time, so redirect it to the temporary folder.
    csv_path = raw_dir / "credit_default_raw.csv"
    write_raw_csv = data.write_raw_csv
    monkeypatch.setattr(data, "write_raw_csv", lambda df, path=csv_path: write_raw_csv(df, path))

    assert data.main(["fetch"]) == 0

    assert csv_path.is_file()
    manifest = json.loads((raw_dir / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "uci_archive"
    assert manifest["n_rows"] == 20
    assert manifest["n_columns"] == len(RAW_COLUMNS)
    assert manifest["csv_sha256"] == data.sha256_of(csv_path)
    assert manifest["positive_rate"] == round(float(frame[settings.TARGET].mean()), 4)
    assert manifest["fetched_at"].endswith("Z")
    assert Path(manifest["csv_path"]) == Path("raw/credit_default_raw.csv")
    assert "source=uci_archive" in capsys.readouterr().out


def test_main_info_without_the_raw_csv(raw_dir, capsys):
    assert data.main(["info"]) == 1
    assert "raw CSV not found" in capsys.readouterr().out


def test_main_info_prints_the_summary(raw_dir, capsys):
    frame = raw_frame(20)
    frame["BILL_AMT1"] = 100
    frame.loc[0, "BILL_AMT1"] = -100
    # Row 1 becomes an exact twin of row 0 apart from its ID, so it also carries the negative bill.
    frame.iloc[1] = frame.iloc[0]
    frame.loc[1, settings.ID_COLUMN] = 2
    data.write_raw_csv(frame, raw_dir / "credit_default_raw.csv")
    (raw_dir / "fetch_manifest.json").write_text(
        json.dumps({"data_source": "uci_archive"}), encoding="utf-8"
    )

    assert data.main(["info"]) == 0

    out = capsys.readouterr().out
    assert "shape: 20 rows x 25 columns" in out
    assert "data source: uci_archive" in out
    assert "file not present" in out
    assert "exact duplicate rows (ignoring ID): 1" in out
    assert "  BILL_AMT1: 2\n" in out
    assert "nulls: 0" in out


def test_build_parser_requires_a_subcommand():
    parser = data.build_parser()
    assert parser.parse_args(["fetch"]).command == "fetch"
    assert parser.parse_args(["info"]).command == "info"
    with pytest.raises(SystemExit):
        parser.parse_args([])
