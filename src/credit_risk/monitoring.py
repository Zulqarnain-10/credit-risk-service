"""Drift monitoring: reference distributions, PSI, and drift reports.

Usage:
    python -m credit_risk.monitoring baseline
    python -m credit_risk.monitoring drift
    python -m credit_risk.monitoring psi --log predictions.jsonl [--baseline ...] [--json out.json]

Subcommands register themselves in SUBCOMMANDS, one function per subcommand that adds its
parser and sets a handler via set_defaults(func=...). The baseline subcommand and the shared
helpers psi() and histogram_proportions() live here; drift and psi subcommands extend the list.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk import settings
from credit_risk.features import build_features

log = logging.getLogger(__name__)

PROBABILITY_COLUMN = "probability"
BASELINE_COLUMNS: tuple[str, ...] = (
    PROBABILITY_COLUMN,
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1",
    "util_1",
    "util_mean",
    "delinq_max",
    "delinq_months",
    "bill_mean",
    "pay_amt_mean",
)
DEFAULT_BINS = 10

# Drift report: the model score is compared under the name "score" next to every model feature.
SCORE_COLUMN = "score"
DRIFT_COLUMNS: tuple[str, ...] = (*settings.MODEL_FEATURE_COLUMNS, SCORE_COLUMN)
CATEGORICAL_DRIFT_COLUMNS: tuple[str, ...] = (
    "EDUCATION",
    *settings.PAY_STATUS_COLUMNS,
    "delinq_recent",
)
REFERENCE_NAME = "train split"
CURRENT_BATCH_NAME = "Simulated production batch"
DRIFT_SHARE_THRESHOLD = 0.5
PAY_STATUS_MAX = 8

PSI_STABLE_MAX = 0.1
PSI_MODERATE_MAX = 0.25
PSI_BANDS = {
    "stable": f"< {PSI_STABLE_MAX}",
    "moderate": f"{PSI_STABLE_MAX} to {PSI_MODERATE_MAX}",
    "significant": f"> {PSI_MODERATE_MAX}",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def quantile_edges(values, n_bins: int = DEFAULT_BINS) -> list[float]:
    """Bin edges at reference quantiles. Ties collapse duplicate edges, so fewer bins can result."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("no finite values to bin")
    edges = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([edges[0], edges[0]])
    return [float(e) for e in edges]


def histogram_proportions(values, edges) -> list[float]:
    """Share of values per bin. Outer bins are open-ended so out-of-range values still count."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    edges = np.asarray(edges, dtype=float)
    n_bins = max(len(edges) - 1, 1)
    if arr.size == 0:
        return [0.0] * n_bins
    bin_ids = np.searchsorted(edges[1:-1], arr, side="right")
    counts = np.bincount(bin_ids, minlength=n_bins)
    return [float(c) for c in counts / arr.size]


def psi(expected_props, actual_props, eps: float = 1e-4) -> float:
    """Population stability index between two proportion vectors over the same bins."""
    expected = np.asarray(expected_props, dtype=float)
    actual = np.asarray(actual_props, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError("expected and actual proportions must have the same length")
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_band(value: float) -> str:
    """Standard bands: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant."""
    if value < PSI_STABLE_MAX:
        return "stable"
    if value <= PSI_MODERATE_MAX:
        return "moderate"
    return "significant"


def psi_against_baseline(frame: pd.DataFrame, baseline: dict) -> dict[str, float]:
    """PSI for every baseline column present in the frame, binned on the saved reference edges."""
    out: dict[str, float] = {}
    for name, ref in baseline["columns"].items():
        if name not in frame.columns:
            continue
        actual = histogram_proportions(frame[name], ref["edges"])
        out[name] = round(psi(ref["proportions"], actual), 4)
    return out


def reference_frame(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Raw inputs plus engineered features and, when a model is given, its score."""
    frame = build_features(df)
    if model is not None:
        inputs = df[list(settings.MODEL_INPUT_COLUMNS)]
        frame[PROBABILITY_COLUMN] = model.predict_proba(inputs)[:, 1]
    return frame


def build_baseline(
    train_df: pd.DataFrame, model, model_version: str, n_bins: int = DEFAULT_BINS
) -> dict:
    frame = reference_frame(train_df, model)
    columns = {}
    for name in BASELINE_COLUMNS:
        edges = quantile_edges(frame[name], n_bins)
        columns[name] = {
            "edges": edges,
            "proportions": histogram_proportions(frame[name], edges),
        }
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "reference": "train split",
        "n": len(train_df),
        "model_version": model_version,
        "n_bins": int(n_bins),
        "binning": "quantile edges from the reference; outer bins are open-ended",
        "columns": columns,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Drift report helpers
# ---------------------------------------------------------------------------


def perturb_batch(df: pd.DataFrame, drift_params: dict) -> tuple[pd.DataFrame, int]:
    """Apply the documented perturbation from params.drift to a copy of the frame.

    Scales LIMIT_BAL and BILL_AMT1..6, shifts AGE, and moves PAY_0 up one step (capped at 8)
    for a seeded random share of rows. Returns the batch and the number of rows whose PAY_0 moved.
    """
    rng = np.random.default_rng(int(drift_params["seed"]))
    out = df.copy()
    out["LIMIT_BAL"] = out["LIMIT_BAL"].astype(float) * float(drift_params["limit_bal_scale"])
    out["AGE"] = out["AGE"] + int(drift_params["age_shift_years"])
    for col in settings.BILL_COLUMNS:
        out[col] = out[col].astype(float) * float(drift_params["bill_amt_scale"])
    shifted = rng.random(len(out)) < float(drift_params["pay_status_shift_share"])
    out.loc[shifted, "PAY_0"] = np.minimum(out.loc[shifted, "PAY_0"] + 1, PAY_STATUS_MAX)
    return out, int(shifted.sum())


def scored_frame(df: pd.DataFrame, model) -> pd.DataFrame:
    """Model features plus the model score, held as both 'probability' and 'score'."""
    frame = reference_frame(df, model)
    frame[SCORE_COLUMN] = frame[PROBABILITY_COLUMN]
    return frame


def run_evidently_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> tuple[dict, str, str]:
    """Run evidently's DataDriftPreset. Returns the snapshot payload, its HTML, and the version."""
    import evidently
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    categorical = [c for c in columns if c in CATEGORICAL_DRIFT_COLUMNS]
    numerical = [c for c in columns if c not in CATEGORICAL_DRIFT_COLUMNS]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    report = Report(
        [DataDriftPreset(columns=columns, drift_share=DRIFT_SHARE_THRESHOLD)], include_tests=True
    )
    snapshot = report.run(
        current_data=Dataset.from_pandas(current[columns], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
        name=CURRENT_BATCH_NAME,
    )
    payload = json.loads(snapshot.json())
    return payload, snapshot.get_html_str(as_iframe=False), evidently.__version__


def _drifted_from_score(stat_test: str, score: float, threshold: float) -> bool:
    """Evidently's rule: p-value tests drift below the threshold, distance tests at or above it."""
    if "p_value" in (stat_test or "").lower():
        return score < threshold
    return score >= threshold


def parse_evidently_snapshot(payload: dict, columns: list[str]) -> tuple[list[dict], dict]:
    """Per-column drift rows in the given order, plus the dataset-level drifted-columns block."""
    by_id: dict[str, dict] = {}
    dataset: dict = {}
    for metric in payload.get("metrics", []):
        config = metric.get("config", {})
        kind = str(config.get("type", ""))
        if kind.endswith("ValueDrift"):
            by_id[metric["id"]] = {
                "feature": config["column"],
                "stat_test": config.get("method"),
                "score": float(metric["value"]),
                "threshold": config.get("threshold"),
                "drifted": None,
            }
        elif kind.endswith("DriftedColumnsCount"):
            dataset = {
                "id": metric["id"],
                "count": int(metric["value"]["count"]),
                "share": float(metric["value"]["share"]),
                "dataset_drift": None,
            }
    for test in payload.get("tests", []):
        metric_id = test.get("metric_config", {}).get("metric_id")
        failed = str(test.get("status", "")).upper().endswith("FAIL")
        if metric_id in by_id:
            by_id[metric_id]["drifted"] = failed
        elif metric_id and metric_id == dataset.get("id"):
            dataset["dataset_drift"] = failed
    by_feature = {row["feature"]: row for row in by_id.values()}
    features = []
    for col in columns:
        row = by_feature.get(col)
        if row is None:
            raise RuntimeError(f"evidently returned no drift result for column {col}")
        if row["drifted"] is None:
            row["drifted"] = _drifted_from_score(row["stat_test"], row["score"], row["threshold"])
        row["score"] = round(row["score"], 4)
        features.append(row)
    return features, dataset


def build_drift_summary(
    features: list[dict],
    dataset: dict,
    psi_block: dict[str, float],
    n_reference: int,
    n_current: int,
    perturbation: dict,
    model_version: str,
    method: str,
    method_detail: str,
) -> dict:
    n_drifted = sum(1 for row in features if row["drifted"])
    drift_share = n_drifted / len(features) if features else 0.0
    dataset_drift = dataset.get("dataset_drift")
    if dataset_drift is None:
        dataset_drift = drift_share >= DRIFT_SHARE_THRESHOLD
    return {
        "generated_at": _utc_now(),
        "method": method,
        "method_detail": method_detail,
        "model_version": model_version,
        "reference": {"name": REFERENCE_NAME, "n": int(n_reference)},
        "current": {
            "name": CURRENT_BATCH_NAME,
            "n": int(n_current),
            "source": "test split with the perturbation in params.drift; no production traffic",
            "perturbation": perturbation,
        },
        "n_features": len(features),
        "n_drifted": int(n_drifted),
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "dataset_drift": bool(dataset_drift),
        "features": features,
        "psi": psi_block,
        "psi_bands": {name: psi_band(value) for name, value in psi_block.items()},
        "psi_reference": settings.DRIFT_BASELINE_PATH.relative_to(settings.REPO_ROOT).as_posix(),
    }


_HEADER_CSS = """
.crs-drift{font-family:"IBM Plex Sans","Segoe UI",Arial,sans-serif;background:#0A1628;
color:#E9EEF6;padding:32px 40px;line-height:1.6}
.crs-drift .kicker{font-family:"IBM Plex Mono",Consolas,monospace;font-size:.75rem;
letter-spacing:.08em;text-transform:uppercase;color:#A9B8CF}
.crs-drift h1{font-family:"Bricolage Grotesque","IBM Plex Sans","Segoe UI",Arial,sans-serif;
font-size:2rem;font-weight:800;margin:.25rem 0 .75rem;color:#FFFFFF}
.crs-drift p{max-width:70rem;margin:.5rem 0}
.crs-drift .notice{border-left:3px solid #FF6B35;background:#0F1F38;padding:.75rem 1rem;
max-width:70rem;border-radius:6px}
.crs-drift .stats{display:flex;flex-wrap:wrap;gap:2rem;margin:1.25rem 0}
.crs-drift .stat{font-family:"IBM Plex Mono",Consolas,monospace;font-size:.8125rem;color:#A9B8CF}
.crs-drift .stat span{display:block;font-size:1.5rem;color:#FF6B35}
.crs-drift table{border-collapse:collapse;font-family:"IBM Plex Mono",Consolas,monospace;
font-size:.8125rem;margin:.5rem 0 1rem}
.crs-drift th,.crs-drift td{padding:.3rem .9rem .3rem 0;border-bottom:1px solid #22375A;
text-align:left}
.crs-drift td.num{text-align:right}
.crs-drift .significant{color:#FF6B35}
.crs-drift .moderate{color:#2D8CFF}
.crs-drift .stable{color:#A9B8CF}
.crs-drift .meta{font-family:"IBM Plex Mono",Consolas,monospace;font-size:.75rem;color:#A9B8CF}
"""


def _perturbation_sentence(perturbation: dict) -> str:
    return (
        f"LIMIT_BAL scaled by {perturbation['limit_bal_scale']}, AGE shifted by "
        f"{perturbation['age_shift_years']} years, BILL_AMT1 to BILL_AMT6 scaled by "
        f"{perturbation['bill_amt_scale']}, and PAY_0 moved up one step (capped at "
        f"{PAY_STATUS_MAX}) for a random {perturbation['pay_status_shift_share']} share of rows "
        f"({perturbation['pay_status_rows_shifted']} rows, seed {perturbation['seed']})."
    )


def render_drift_header(summary: dict) -> str:
    """HTML block that states the current batch is simulated and lists the headline numbers."""
    esc = html_lib.escape
    reference = summary["reference"]
    current = summary["current"]
    psi_rows = "".join(
        f"<tr><td>{esc(name)}</td><td class='num'>{value:.4f}</td>"
        f"<td class='{esc(summary['psi_bands'][name])}'>{esc(summary['psi_bands'][name])}</td></tr>"
        for name, value in summary["psi"].items()
    )
    drifted_rows = "".join(
        f"<tr><td>{esc(row['feature'])}</td><td>{esc(str(row['stat_test']))}</td>"
        f"<td class='num'>{row['score']:.4f}</td><td class='num'>{row['threshold']}</td></tr>"
        for row in summary["features"]
        if row["drifted"]
    )
    drifted_table = (
        "<table><thead><tr><th>Drifted column</th><th>Test</th><th>Score</th><th>Threshold</th>"
        f"</tr></thead><tbody>{drifted_rows}</tbody></table>"
        if drifted_rows
        else "<p>No column crossed its drift threshold.</p>"
    )
    score_psi = summary["psi"].get(PROBABILITY_COLUMN)
    score_psi_text = f"{score_psi:.4f}" if score_psi is not None else "[todo]"
    dataset_drift_text = "yes" if summary["dataset_drift"] else "no"
    return f"""
<style>{_HEADER_CSS}</style>
<section class="crs-drift">
  <div class="kicker">Data drift report, model {esc(summary["model_version"])}</div>
  <h1>{esc(CURRENT_BATCH_NAME)}</h1>
  <div class="notice">
    <p>The current batch is simulated. It is the held-out test split (n = {current["n"]}) with a
    deliberate perturbation applied in code: {esc(_perturbation_sentence(current["perturbation"]))}
    The reference is the {esc(reference["name"])} (n = {reference["n"]}). No production traffic
    was used.</p>
  </div>
  <div class="stats">
    <div class="stat">Columns compared<span>{summary["n_features"]}</span></div>
    <div class="stat">Columns drifted<span>{summary["n_drifted"]}</span></div>
    <div class="stat">Drift share<span>{summary["drift_share"]:.4f}</span></div>
    <div class="stat">Dataset drift<span>{dataset_drift_text}</span></div>
    <div class="stat">PSI of the score<span>{score_psi_text}</span></div>
  </div>
  {drifted_table}
  <table>
    <thead><tr><th>PSI column</th><th>PSI</th><th>Band</th></tr></thead>
    <tbody>{psi_rows}</tbody>
  </table>
  <p class="meta">PSI bands: {esc(PSI_BANDS["stable"])} stable, {esc(PSI_BANDS["moderate"])}
  moderate, {esc(PSI_BANDS["significant"])} significant. Reference histograms from
  {esc(summary["psi_reference"])}. Dataset drift is declared when the drift share reaches
  {summary["drift_share_threshold"]}. Method: {esc(summary["method_detail"])}.
  Generated at {esc(summary["generated_at"])}. The evidently report follows.</p>
</section>
"""


def render_drift_html(base_html: str, summary: dict) -> str:
    """Prepend the simulated-batch header to evidently's page and give it a title."""
    title = f"<title>{html_lib.escape(CURRENT_BATCH_NAME)}</title>"
    header = render_drift_header(summary)
    page = base_html.replace("<head>", f"<head>{title}", 1)
    if "<body>" in page:
        return page.replace("<body>", f"<body>{header}", 1)
    return header + page


# ---------------------------------------------------------------------------
# Prediction-log helpers (psi subcommand)
# ---------------------------------------------------------------------------


def read_prediction_log(path: Path) -> tuple[pd.DataFrame, int]:
    """Parse the API's JSONL log into a frame of uppercase inputs plus probability.

    Lines that are not JSON objects with a probability and an inputs dict are skipped and counted.
    """
    rows: list[dict] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            inputs = record.get("inputs") if isinstance(record, dict) else None
            if not isinstance(inputs, dict) or "probability" not in record:
                skipped += 1
                continue
            row = {str(k).upper(): v for k, v in inputs.items()}
            row[PROBABILITY_COLUMN] = float(record["probability"])
            row["model_version"] = record.get("model_version")
            rows.append(row)
    return pd.DataFrame(rows), skipped


def log_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute engineered features when every model input is logged; else keep raw columns."""
    missing = [c for c in settings.MODEL_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        log.warning("log inputs lack %d model columns; PSI limited to logged columns", len(missing))
        return frame
    features = build_features(frame)
    features[PROBABILITY_COLUMN] = frame[PROBABILITY_COLUMN].to_numpy()
    if "model_version" in frame.columns:
        features["model_version"] = frame["model_version"].to_numpy()
    return features


def build_psi_report(
    frame: pd.DataFrame, baseline: dict, log_path: Path, baseline_path: Path, skipped: int
) -> dict:
    scores = psi_against_baseline(frame, baseline)
    columns = [
        {"column": name, "psi": value, "band": psi_band(value)} for name, value in scores.items()
    ]
    missing = [name for name in baseline["columns"] if name not in scores]
    versions = (
        frame["model_version"].fillna("unknown").astype(str).value_counts().to_dict()
        if "model_version" in frame.columns
        else {}
    )
    return {
        "generated_at": _utc_now(),
        "log": str(log_path),
        "baseline": str(baseline_path),
        "reference": {
            "name": baseline.get("reference", REFERENCE_NAME),
            "n": int(baseline.get("n", 0)),
            "model_version": baseline.get("model_version"),
        },
        "n_rows": len(frame),
        "n_skipped_lines": int(skipped),
        "model_versions": {k: int(v) for k, v in versions.items()},
        "bands": PSI_BANDS,
        "columns": columns,
        "missing_columns": missing,
        "n_stable": sum(1 for c in columns if c["band"] == "stable"),
        "n_moderate": sum(1 for c in columns if c["band"] == "moderate"),
        "n_significant": sum(1 for c in columns if c["band"] == "significant"),
    }


def format_psi_table(report: dict) -> str:
    reference = report["reference"]
    lines = [
        f"PSI against {reference['name']} (n={reference['n']}, model "
        f"{reference['model_version']}); log rows={report['n_rows']}, "
        f"skipped lines={report['n_skipped_lines']}",
        f"{'column':<16}{'psi':>8}  band",
    ]
    lines.extend(
        f"{row['column']:<16}{row['psi']:>8.4f}  {row['band']}" for row in report["columns"]
    )
    if report["missing_columns"]:
        lines.append(f"not in log: {', '.join(report['missing_columns'])}")
    lines.append(
        f"bands: {PSI_BANDS['stable']} stable, {PSI_BANDS['moderate']} moderate, "
        f"{PSI_BANDS['significant']} significant"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_baseline(args: argparse.Namespace) -> int:
    with open(settings.VERSION_PATH, encoding="utf-8") as fh:
        version = json.load(fh)
    model = joblib.load(settings.MODEL_PATH)
    train_df = pd.read_csv(settings.TRAIN_CSV_PATH)
    baseline = build_baseline(train_df, model, version["model_version"], n_bins=args.bins)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.DRIFT_BASELINE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(baseline, fh, indent=2)
        fh.write("\n")
    print(
        f"wrote {settings.DRIFT_BASELINE_PATH} n={baseline['n']} "
        f"columns={len(baseline['columns'])} bins={args.bins}"
    )
    return 0


def add_baseline_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "baseline", help="reference histograms of scores and key features from the train split"
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="quantile bins")
    parser.set_defaults(func=cmd_baseline)


def cmd_drift(args: argparse.Namespace) -> int:
    drift_params = settings.load_params()["drift"]
    version = _read_json(settings.VERSION_PATH)
    baseline = _read_json(args.baseline)
    model = joblib.load(settings.MODEL_PATH)
    train_df = pd.read_csv(settings.TRAIN_CSV_PATH)
    test_df = pd.read_csv(settings.TEST_CSV_PATH)

    current_raw, n_shifted = perturb_batch(test_df, drift_params)
    reference = scored_frame(train_df, model)
    current = scored_frame(current_raw, model)
    columns = list(DRIFT_COLUMNS)

    try:
        payload, base_html, evidently_version = run_evidently_drift(reference, current, columns)
    except ImportError as exc:
        log.error("evidently is required for the drift report: pip install -e .[monitoring]")
        raise SystemExit(1) from exc

    features, dataset = parse_evidently_snapshot(payload, columns)
    perturbation = {
        "seed": int(drift_params["seed"]),
        "limit_bal_scale": float(drift_params["limit_bal_scale"]),
        "age_shift_years": int(drift_params["age_shift_years"]),
        "bill_amt_scale": float(drift_params["bill_amt_scale"]),
        "pay_status_shift_share": float(drift_params["pay_status_shift_share"]),
        "pay_status_rows_shifted": n_shifted,
    }
    summary = build_drift_summary(
        features=features,
        dataset=dataset,
        psi_block=psi_against_baseline(current, baseline),
        n_reference=len(reference),
        n_current=len(current),
        perturbation=perturbation,
        model_version=version["model_version"],
        method="evidently",
        method_detail=(
            f"evidently {evidently_version} DataDriftPreset, per-column test chosen by evidently "
            "(numerical: Wasserstein distance normed by the reference, categorical: "
            "Jensen-Shannon distance, both at threshold 0.1 for samples above 1000 rows); "
            "PSI from the saved reference histograms"
        ),
    )
    _write_json(args.summary, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_drift_html(base_html, summary))
    print(
        f"wrote {args.report} and {args.summary} method={summary['method']} "
        f"n_features={summary['n_features']} n_drifted={summary['n_drifted']} "
        f"drift_share={summary['drift_share']} dataset_drift={summary['dataset_drift']} "
        f"psi_score={summary['psi'].get(PROBABILITY_COLUMN)}"
    )
    return 0


def add_drift_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "drift",
        help="evidently drift report: train split versus the simulated production batch",
    )
    parser.add_argument(
        "--baseline", type=Path, default=settings.DRIFT_BASELINE_PATH, help="reference histograms"
    )
    parser.add_argument(
        "--report", type=Path, default=settings.DRIFT_REPORT_PATH, help="HTML output path"
    )
    parser.add_argument(
        "--summary", type=Path, default=settings.DRIFT_SUMMARY_PATH, help="JSON output path"
    )
    parser.set_defaults(func=cmd_drift)


def cmd_psi(args: argparse.Namespace) -> int:
    baseline = _read_json(args.baseline)
    frame, skipped = read_prediction_log(args.log)
    if frame.empty:
        log.error("no usable prediction rows in %s (skipped %d lines)", args.log, skipped)
        return 1
    report = build_psi_report(log_features(frame), baseline, args.log, args.baseline, skipped)
    foreign = [v for v in report["model_versions"] if v != baseline.get("model_version")]
    if foreign:
        log.warning("log holds model versions not in the baseline: %s", ", ".join(foreign))
    print(format_psi_table(report))
    if args.json is not None:
        _write_json(args.json, report)
        print(f"wrote {args.json}")
    return 0


def add_psi_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "psi", help="PSI of logged predictions and inputs against the reference histograms"
    )
    parser.add_argument("--log", type=Path, required=True, help="API prediction log (JSONL)")
    parser.add_argument(
        "--baseline", type=Path, default=settings.DRIFT_BASELINE_PATH, help="reference histograms"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    parser.set_defaults(func=cmd_psi)


SUBCOMMANDS: list[Callable[[argparse._SubParsersAction], None]] = [
    add_baseline_parser,
    add_drift_parser,
    add_psi_parser,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m credit_risk.monitoring", description="Drift monitoring commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in SUBCOMMANDS:
        register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
