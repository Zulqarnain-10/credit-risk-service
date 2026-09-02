"""Unit tests for credit_risk.split: proportions, stratification, manifest hashes, determinism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk import settings, split

N = 1000
POSITIVES = 300
ZIP_SHA = "a" * 64


def make_frame(n: int = N, positives: int = POSITIVES, seed: int = 0) -> pd.DataFrame:
    """A frame with exactly the requested number of positive labels."""
    rng = np.random.default_rng(seed)
    label = np.zeros(n, dtype=int)
    label[:positives] = 1
    rng.shuffle(label)
    return pd.DataFrame(
        {
            settings.ID_COLUMN: np.arange(1, n + 1),
            "LIMIT_BAL": rng.integers(1, 101, n) * 10_000.0,
            "AGE": rng.integers(21, 70, n),
            settings.TARGET: label,
        }
    )


def make_params(seed: int = 42, train: float = 0.6, val: float = 0.2, test: float = 0.2) -> dict:
    return {
        "split": {"seed": seed, "train": train, "val": val, "test": test},
        "data": {"zip_sha256": ZIP_SHA},
    }


@pytest.fixture(autouse=True)
def no_fetch_manifest(tmp_path: Path, monkeypatch):
    """Point the fetch manifest at a file that does not exist so data_source is 'unknown'."""
    monkeypatch.setattr(split, "FETCH_MANIFEST_PATH", tmp_path / "absent_manifest.json")


def test_ids_sha256_is_order_independent():
    expected = hashlib.sha256(b"1,2,3").hexdigest()
    assert split.ids_sha256([3, 1, 2]) == expected
    assert split.ids_sha256(pd.Series([1, 2, 3])) == expected
    assert split.ids_sha256(np.array([2.0, 3.0, 1.0])) == expected
    assert split.ids_sha256([1, 2, 4]) != expected


def test_make_split_proportions_and_stratification():
    df = make_frame()
    train, val, test, manifest = split.make_split(df, make_params())
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    for frame in (train, val, test):
        assert frame[settings.TARGET].mean() == pytest.approx(0.3, abs=1e-3)
        assert list(frame.columns) == list(df.columns)
        assert list(frame.index) == list(range(len(frame)))
    ids = [set(frame[settings.ID_COLUMN]) for frame in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert ids[0] | ids[1] | ids[2] == set(df[settings.ID_COLUMN])
    assert manifest["n"] == {"train": 600, "val": 200, "test": 200}
    assert manifest["n_total"] == N
    for name in ("train", "val", "test"):
        assert manifest["positive_rate"][name] == pytest.approx(0.3, abs=1e-3)
    assert manifest["fractions"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert manifest["seed"] == 42
    assert manifest["stratify"] == settings.TARGET
    assert manifest["data_sha256"] == ZIP_SHA
    assert manifest["data_source"] == "unknown"


def test_make_split_manifest_hashes_match_the_frames():
    df = make_frame()
    train, val, test, manifest = split.make_split(df, make_params())
    for name, frame in (("train", train), ("val", val), ("test", test)):
        digest = manifest["id_sha256"][name]
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
        assert digest == split.ids_sha256(frame[settings.ID_COLUMN])
    assert len(set(manifest["id_sha256"].values())) == 3


def test_make_split_is_deterministic_and_seed_sensitive():
    df = make_frame()
    first = split.make_split(df, make_params(seed=42))
    second = split.make_split(df.copy(), make_params(seed=42))
    for a, b in zip(first[:3], second[:3], strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert first[3] == second[3]
    other = split.make_split(df, make_params(seed=7))
    assert other[3]["id_sha256"]["train"] != first[3]["id_sha256"]["train"]
    assert other[3]["n"] == first[3]["n"]


def test_make_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        split.make_split(make_frame(), make_params(train=0.5, val=0.2, test=0.2))


def test_data_source_reads_the_fetch_manifest(tmp_path: Path, monkeypatch):
    assert split._data_source() == "unknown"
    manifest = tmp_path / "fetch_manifest.json"
    manifest.write_text(json.dumps({"data_source": "ucimlrepo"}), encoding="utf-8")
    monkeypatch.setattr(split, "FETCH_MANIFEST_PATH", manifest)
    assert split._data_source() == "ucimlrepo"
    manifest.write_text("{}", encoding="utf-8")
    assert split._data_source() == "unknown"


def test_main_writes_three_csvs_and_the_manifest(tmp_path: Path, monkeypatch, capsys):
    df = make_frame(500, 150)
    features_csv = tmp_path / "features.csv"
    df.to_csv(features_csv, index=False)
    split_dir = tmp_path / "splits"
    monkeypatch.setattr(settings, "load_params", lambda path=None: make_params())
    monkeypatch.setattr(settings, "FEATURES_CSV_PATH", features_csv)
    monkeypatch.setattr(settings, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(settings, "SPLIT_MANIFEST_PATH", split_dir / "split_manifest.json")

    assert split.main([]) == 0

    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n"] == {"train": 300, "val": 100, "test": 100}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(split_dir / f"{name}.csv")
        assert len(frame) == manifest["n"][name]
        assert split.ids_sha256(frame[settings.ID_COLUMN]) == manifest["id_sha256"][name]
    assert b"\r\n" not in (split_dir / "train.csv").read_bytes()
    assert "wrote splits to" in capsys.readouterr().out
