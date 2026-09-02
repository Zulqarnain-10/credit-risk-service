"""Compare a reproduced reports/metrics.json with the committed one.

Every numeric leaf must agree within a tolerance, 0.005 by default. For values whose
magnitude exceeds 1 (confusion-matrix counts, calibration counts, cost per 1000) the same
tolerance is applied relative to the larger magnitude, so 0.005 means 0.5 percent there.
Identity keys (trained_at, git_sha, mlflow_run_id, model_version) are ignored. n_train, n_val,
n_test, data_sha256, and model must be identical, as must every string and boolean.
Exit code 0 when everything agrees, 1 on any mismatch, 2 when an input cannot be read.

Typical use in CI:
    git show HEAD:reports/metrics.json > committed.json
    python scripts/compare_metrics.py --committed committed.json --fresh reports/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRESH = REPO_ROOT / "reports" / "metrics.json"
DEFAULT_TOLERANCE = 0.005
IGNORED_KEYS = frozenset({"trained_at", "git_sha", "mlflow_run_id", "model_version"})
EXACT_KEYS = frozenset({"n_train", "n_val", "n_test", "data_sha256", "model"})
MISSING = object()


@dataclass
class Row:
    path: str
    committed: Any
    fresh: Any
    status: str
    ok: bool

    @property
    def delta(self) -> float | None:
        if is_number(self.committed) and is_number(self.fresh):
            return self.fresh - self.committed
        return None


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts and lists into {"a.b[0]": leaf}, keeping document order."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(node, list):
        out = {}
        for index, value in enumerate(node):
            out.update(flatten(value, f"{prefix}[{index}]"))
        return out
    return {prefix: node}


def leaf_key(path: str) -> str:
    """The last dotted segment of a path without its list index."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def within(a: float, b: float, tolerance: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def compare(committed: dict, fresh: dict, tolerance: float) -> list[Row]:
    flat_committed = flatten(committed)
    flat_fresh = flatten(fresh)
    paths = list(flat_committed) + [p for p in flat_fresh if p not in flat_committed]
    rows: list[Row] = []
    for path in paths:
        a = flat_committed.get(path, MISSING)
        b = flat_fresh.get(path, MISSING)
        key = leaf_key(path)
        if key in IGNORED_KEYS:
            rows.append(Row(path, a, b, "ignored", True))
        elif a is MISSING:
            rows.append(Row(path, a, b, "missing in committed", False))
        elif b is MISSING:
            rows.append(Row(path, a, b, "missing in fresh", False))
        elif key in EXACT_KEYS or not (is_number(a) and is_number(b)):
            ok = a == b
            rows.append(Row(path, a, b, "equal" if ok else "differs", ok))
        else:
            ok = within(a, b, tolerance)
            rows.append(Row(path, a, b, "within tolerance" if ok else "outside tolerance", ok))
    return rows


def fmt(value: Any) -> str:
    if value is MISSING:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def fmt_delta(row: Row) -> str:
    delta = row.delta
    if delta is None:
        return ""
    if isinstance(row.committed, int) and isinstance(row.fresh, int):
        return f"{delta:+d}"
    return f"{delta:+.4f}"


def summary(rows: list[Row]) -> str:
    compared = [r for r in rows if r.status != "ignored"]
    failed = [r for r in compared if not r.ok]
    if failed:
        return f"{len(failed)} of {len(compared)} compared leaves differ or fall outside tolerance."
    return f"All {len(compared)} compared leaves agree."


def render_text(rows: list[Row], tolerance: float) -> str:
    table = [("key", "committed", "fresh", "delta", "status")]
    table += [(r.path, fmt(r.committed), fmt(r.fresh), fmt_delta(r), r.status) for r in rows]
    widths = [max(len(line[i]) for line in table) for i in range(5)]
    lines = [f"Tolerance {tolerance} on numeric leaves, scaled by magnitude above 1."]
    for line in table:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip())
    lines.append(summary(rows))
    return "\n".join(lines)


def render_markdown(rows: list[Row], tolerance: float, committed: Path, fresh: Path) -> str:
    lines = [
        "## Metrics comparison",
        "",
        f"Committed: `{committed.as_posix()}`. Fresh: `{fresh.as_posix()}`. "
        f"Tolerance {tolerance} on numeric leaves, scaled by magnitude above 1.",
        "",
        "| Key | Committed | Fresh | Delta | Status |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.path}` | {fmt(r.committed)} | {fmt(r.fresh)} | {fmt_delta(r)} | {r.status} |"
        )
    lines += ["", summary(rows)]
    return "\n".join(lines)


def load(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--committed",
        required=True,
        type=Path,
        help="metrics.json as committed, for example from git show HEAD:reports/metrics.json",
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        default=DEFAULT_FRESH,
        help="freshly reproduced metrics.json (default: reports/metrics.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"allowed absolute difference on numeric leaves (default {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="print a Markdown table instead of plain text"
    )
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)

    try:
        committed = load(args.committed)
        fresh = load(args.fresh)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = compare(committed, fresh, args.tolerance)
    if args.markdown:
        report = render_markdown(rows, args.tolerance, args.committed, args.fresh)
    else:
        report = render_text(rows, args.tolerance)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
