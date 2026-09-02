"""Unit tests for credit_risk.sync_readme: receipts, formatting, block replacement, --check.

Only the public README renderer is covered. A temporary repository root holds hand-built
receipt files whose numbers are obviously synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_risk import sync_readme as sr

METRICS = {
    "n_train": 1800,
    "n_val": 600,
    "n_test": 600,
    "positive_rate_test": 0.2222,
    "roc_auc": 0.8123,
    "pr_auc": 0.6123,
    "brier": 0.1234,
    "ks": 0.4567,
    "threshold_cost_optimal": 0.155,
    "threshold_precision_target": 0.505,
    "precision_target_met": True,
    "at_threshold": {
        "threshold": 0.155,
        "precision": 0.3611,
        "recall": 0.7911,
        "selection_rate": 0.4811,
    },
    "at_precision_target": {
        "threshold": 0.505,
        "precision": 0.6011,
        "recall": 0.2511,
        "selection_rate": 0.0911,
    },
    "threshold_selection": {
        "selected_on": "validation",
        "cost_false_negative": 5.0,
        "cost_false_positive": 1.0,
        "target_precision": 0.6,
    },
    "baseline_logreg": {"roc_auc": 0.7123, "pr_auc": 0.5123, "brier": 0.1534, "ks": 0.3567},
}
LOADTEST = {"p95_ms": 123.45, "requests": 300, "concurrency": 10, "host": "test host"}
FAIRNESS = {
    "sex": {"demographic_parity_ratio": 0.9123},
    "age_band": {"demographic_parity_ratio": 0.7123},
}
VERSION = {
    "model": "hgb",
    "trained_at": "2026-01-01T00:00:00Z",
    "git_sha_short": "abc1234",
    "data_sha256": "0123456789abcdef" * 4,
}
DOCS = {"metrics": METRICS, "loadtest": LOADTEST, "fairness": FAIRNESS, "version": VERSION}
README = "# Title\n\nIntro.\n\n<!-- metrics:start -->\nold block\n<!-- metrics:end -->\n\nOutro.\n"
RENDERED_VALUES = (
    "0.8123",
    "0.7123",
    "0.6123",
    "0.5123",
    "0.1234",
    "0.1534",
    "0.4567",
    "0.3567",
    "0.155",
    "0.505",
    "0.3611",
    "0.7911",
    "0.4811",
    "0.6011",
    "0.2511",
    "0.0911",
    "123.45",
    "300",
    "test host",
    "0.9123",
    "2026-01-01T00:00:00Z",
    "abc1234",
    "0123456789ab",
    "1,800",
    "600",
    "5:1",
    "0.60",
    "HGB",
)


def write_receipts(root: Path, *names: str) -> None:
    for name in names or tuple(DOCS):
        path = root / sr.SOURCES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DOCS[name]), encoding="utf-8")


def readme_text(root: Path) -> str:
    return (root / "README.md").read_bytes().decode("utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_bytes(README.encode("utf-8"))
    return tmp_path


@pytest.fixture
def receipts() -> sr.Receipts:
    return sr.Receipts(metrics=METRICS, loadtest=LOADTEST, fairness=FAIRNESS, version=VERSION)


# ---------------------------------------------------------------------------
# Receipts and formatting
# ---------------------------------------------------------------------------


def test_receipts_get_walks_nested_keys(receipts):
    assert receipts.get("metrics", "roc_auc") == 0.8123
    assert receipts.get("metrics", "baseline_logreg", "ks") == 0.3567
    assert receipts.get("metrics", "nope") is None
    assert receipts.get("metrics", "roc_auc", "deeper") is None
    assert receipts.missing() == []
    empty = sr.Receipts(None, None, None, None)
    assert empty.get("version", "model") is None
    assert empty.missing() == list(sr.SOURCES.values())


def test_load_receipts_reads_what_exists(tmp_path: Path):
    write_receipts(tmp_path, "metrics", "version")
    (tmp_path / "reports" / "loadtest.json").write_text("[1, 2]", encoding="utf-8")
    loaded = sr.load_receipts(tmp_path)
    assert loaded.metrics == METRICS
    assert loaded.version == VERSION
    assert loaded.loadtest is None
    assert loaded.fairness is None
    assert loaded.missing() == ["reports/loadtest.json", "reports/fairness.json"]


def test_formatters_render_numbers_and_todo():
    assert sr.f4(0.79094) == "0.7909"
    assert sr.f4(1) == "1.0000"
    assert sr.f3(0.1549) == "0.155"
    assert sr.f2(123.456) == "123.46"
    assert sr.fint(18000) == "18,000"
    assert sr.fint(6000.0) == "6,000"
    assert sr.ftext("hgb") == "hgb"
    assert sr.fshort("0123456789abcdef0123") == "0123456789ab"
    assert sr.fshort("abc", 2) == "ab"
    for fn in (sr.f4, sr.f3, sr.f2, sr.fint, sr.ftext, sr.fshort):
        assert fn(None) == sr.TODO
    for fn in (sr.f4, sr.f3, sr.f2, sr.fint):
        assert fn(True) == sr.TODO
        assert fn("12") == sr.TODO
    assert sr.ftext("") == sr.TODO
    assert sr.ftext(3) == sr.TODO
    assert sr.is_number(2.5)
    assert sr.is_number(3)
    assert not sr.is_number(True)
    assert not sr.is_number("1")


def test_derived_fields(receipts):
    assert sr.cost_ratio(receipts) == "5:1"
    assert sr.model_label(receipts) == "HGB"
    logreg = sr.Receipts(metrics=None, loadtest=None, fairness=None, version={"model": "logreg"})
    assert sr.model_label(logreg) == "logreg"
    empty = sr.Receipts(None, None, None, None)
    assert sr.cost_ratio(empty) == sr.TODO
    assert sr.model_label(empty) == sr.TODO


# ---------------------------------------------------------------------------
# Rendering and block replacement
# ---------------------------------------------------------------------------


def test_render_readme_carries_every_receipt(receipts):
    block = sr.render_readme(receipts)
    assert block.startswith(sr.GENERATED_NOTE)
    for value in RENDERED_VALUES:
        assert value in block, value
    assert sr.TODO not in block
    assert sr.START not in block
    assert sr.END not in block


def test_render_readme_without_receipts_renders_todo():
    block = sr.render_readme(sr.Receipts(None, None, None, None))
    assert block.startswith(sr.GENERATED_NOTE)
    assert block.count(sr.TODO) >= 10
    assert "0.8123" not in block


def test_replace_block_keeps_markers_on_their_own_lines():
    out = sr.replace_block(README, "new block")
    assert out == README.replace("old block", "new block")
    assert out.count(sr.START) == 1
    assert out.count(sr.END) == 1
    assert sr.replace_block(out, "new block") == out
    with pytest.raises(ValueError, match="exactly one"):
        sr.replace_block("no markers", "x")
    with pytest.raises(ValueError, match="exactly one"):
        sr.replace_block(README + sr.START, "x")
    with pytest.raises(ValueError, match="before"):
        sr.replace_block(f"{sr.END}\n{sr.START}\n", "x")


# ---------------------------------------------------------------------------
# sync and main
# ---------------------------------------------------------------------------


def test_sync_updates_then_is_idempotent(root: Path):
    write_receipts(root)
    assert sr.sync(root, check=True) == 1
    assert readme_text(root) == README

    assert sr.sync(root, check=False) == 0
    first = (root / "README.md").read_bytes()
    assert b"\r\n" not in first
    updated = first.decode("utf-8")
    assert updated.startswith("# Title\n\nIntro.\n\n<!-- metrics:start -->\n" + sr.GENERATED_NOTE)
    assert updated.endswith("<!-- metrics:end -->\n\nOutro.\n")
    assert "old block" not in updated
    for value in RENDERED_VALUES:
        assert value in updated, value
    assert sr.TODO not in updated

    assert sr.sync(root, check=True) == 0
    assert sr.sync(root, check=False) == 0
    assert (root / "README.md").read_bytes() == first


def test_sync_renders_todo_when_receipts_are_missing(root: Path, capsys):
    assert sr.sync(root, check=False) == 0
    updated = readme_text(root)
    assert sr.TODO in updated
    assert "old block" not in updated
    assert sr.GENERATED_NOTE in updated
    out = capsys.readouterr().out
    for path in sr.SOURCES.values():
        assert path in out


def test_sync_fills_only_what_is_present(root: Path):
    write_receipts(root, "metrics")
    assert sr.sync(root, check=False) == 0
    updated = readme_text(root)
    assert "0.8123" in updated
    assert "test host" not in updated
    assert sr.TODO in updated


def test_sync_fails_on_a_readme_without_markers(root: Path, capsys):
    (root / "README.md").write_bytes(b"# No markers\n")
    write_receipts(root)
    assert sr.sync(root, check=False) == 2
    assert "error" in capsys.readouterr().err
    assert readme_text(root) == "# No markers\n"


def test_sync_skips_when_the_readme_is_absent(tmp_path: Path):
    write_receipts(tmp_path)
    assert sr.sync(tmp_path, check=True) == 0
    assert not (tmp_path / "README.md").exists()


def test_main_check_reports_a_stale_block(root: Path, capsys):
    write_receipts(root)
    assert sr.main(["--root", str(root), "--check"]) == 1
    assert "would change" in capsys.readouterr().err
    assert readme_text(root) == README
    assert sr.main(["--root", str(root)]) == 0
    assert sr.main(["--root", str(root), "--check"]) == 0
    assert "0.8123" in readme_text(root)
