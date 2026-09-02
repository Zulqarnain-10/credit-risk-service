"""Choose thresholds on validation, report held-out test metrics, importance, and figures.

Usage:
    python -m credit_risk.evaluate

Thresholds are selected on the validation split only. Every headline number in
reports/metrics.json is then measured once on the untouched test split.
"""

from __future__ import annotations

import json
import logging
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from credit_risk import settings
from credit_risk.features import build_features
from credit_risk.train import BASELINE_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

NAVY, BLUE, CORAL = "#0A1628", "#2D8CFF", "#FF6B35"
INK_2, LINE = "#3D4F6B", "#D8DFEA"


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_at(y, p, threshold: float) -> tuple[int, int, int, int]:
    """(tn, fp, fn, tp) when predicting positive for p >= threshold."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, dtype=float) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return tn, fp, fn, tp


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(np.abs(tpr - fpr)))


def compute_metrics(y, p, threshold: float, costs: dict | None = None) -> dict:
    """Ranking metrics plus the confusion block at one threshold, all rounded to 4 dp.

    costs, when given, holds cost_false_negative and cost_false_positive and adds
    expected_cost_per_1000 applications.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tn, fp, fn, tp = confusion_at(y, p, threshold)
    n = len(y)
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "roc_auc": _r(roc_auc_score(y, p)),
        "pr_auc": _r(average_precision_score(y, p)),
        "brier": _r(brier_score_loss(y, p)),
        "ks": _r(ks_statistic(y, p)),
        "threshold": _r(threshold),
        "precision": _r(precision),
        "recall": _r(recall),
        "f1": _r(f1),
        "selection_rate": _r(predicted_positive / n) if n else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if costs is not None:
        total = float(costs["cost_false_negative"]) * fn + float(costs["cost_false_positive"]) * fp
        out["expected_cost_per_1000"] = _r(total / n * 1000) if n else 0.0
    return out


def threshold_grid(step: float) -> np.ndarray:
    n_steps = round(1.0 / step)
    return np.round(np.linspace(0.0, 1.0, n_steps + 1), 6)


def choose_thresholds(y_val, p_val, params: dict) -> dict:
    """Cost-optimal and precision-target thresholds, both chosen on validation."""
    cfg = params["threshold"]
    c_fn, c_fp = float(cfg["cost_false_negative"]), float(cfg["cost_false_positive"])
    target = float(cfg["target_precision"])
    grid = threshold_grid(float(cfg["grid_step"]))
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)

    costs = np.empty(len(grid))
    precisions = np.empty(len(grid))
    for i, t in enumerate(grid):
        _, fp, fn, tp = confusion_at(y, p, t)
        costs[i] = c_fn * fn + c_fp * fp
        precisions[i] = tp / (tp + fp) if (tp + fp) else 0.0

    i_cost = int(np.argmin(costs))
    meets = np.flatnonzero(precisions >= target)
    if len(meets):
        i_prec, met = int(meets[0]), True
    else:
        i_prec, met = int(np.argmax(precisions)), False
    return {
        "threshold_cost_optimal": _r(grid[i_cost]),
        "threshold_precision_target": _r(grid[i_prec]),
        "precision_target_met": met,
        "validation_cost_at_optimal": _r(costs[i_cost]),
        "validation_precision_at_target": _r(precisions[i_prec]),
        "cost_false_negative": c_fn,
        "cost_false_positive": c_fp,
        "target_precision": target,
        "grid_step": float(cfg["grid_step"]),
    }


def calibration_table(y, p, n_bins: int) -> dict:
    """Reliability table with quantile bins; ties can merge bins, so counts are reported."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], p)
    mean_predicted, fraction_positive, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        mean_predicted.append(_r(p[mask].mean()))
        fraction_positive.append(_r(y[mask].mean()))
        counts.append(int(mask.sum()))
    weights = np.asarray(counts) / len(p)
    ece = float(
        np.sum(weights * np.abs(np.asarray(fraction_positive) - np.asarray(mean_predicted)))
    )
    return {
        "bins": int(n_bins),
        "strategy": "quantile",
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "counts": counts,
        "ece": _r(ece),
    }


def feature_importance(model: Pipeline, X_val: pd.DataFrame, y_val, params: dict) -> dict:
    """Permutation importance on the steps after FeatureBuilder, so names are feature names."""
    cfg = params["evaluate"]
    head = Pipeline(model.steps[1:])
    features = build_features(X_val)
    result = permutation_importance(
        head,
        features,
        np.asarray(y_val).astype(int),
        scoring="roc_auc",
        n_repeats=int(cfg["importance_repeats"]),
        random_state=int(params["train"]["seed"]),
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    names = list(features.columns)
    ranked = [
        {
            "feature": names[i],
            "importance_mean": _r(result.importances_mean[i]),
            "importance_std": _r(result.importances_std[i]),
        }
        for i in order
    ]
    return {
        "method": "permutation_importance",
        "scoring": "roc_auc",
        "evaluated_on": "validation",
        "n_repeats": int(cfg["importance_repeats"]),
        "seed": int(params["train"]["seed"]),
        "features": ranked[: int(cfg["importance_top_n"])],
        "all_features_ranked": [names[i] for i in order],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=NAVY, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(color=LINE, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)


def _save(fig, name: str) -> None:
    settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.FIGURES_DIR / f"{name}.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_roc(y, p_model, p_base, model_name: str, auc_model: float, auc_base: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y, p_model)
    fpr_b, tpr_b, _ = roc_curve(y, p_base)
    ax.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"{model_name} (AUC {auc_model:.4f})")
    ax.plot(fpr_b, tpr_b, color=CORAL, lw=1.8, label=f"logistic baseline (AUC {auc_base:.4f})")
    ax.plot([0, 1], [0, 1], color=NAVY, lw=1, ls="--", label="chance")
    _style(ax, "ROC curve, held-out test split", "False positive rate", "True positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "roc")


def plot_pr(y, p_model, p_base, model_name: str, ap_model: float, ap_base: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y, p_model)
    precision_b, recall_b, _ = precision_recall_curve(y, p_base)
    ax.plot(recall, precision, color=BLUE, lw=2.2, label=f"{model_name} (AP {ap_model:.4f})")
    ax.plot(
        recall_b, precision_b, color=CORAL, lw=1.8, label=f"logistic baseline (AP {ap_base:.4f})"
    )
    base_rate = float(np.mean(y))
    ax.axhline(base_rate, color=NAVY, lw=1, ls="--", label=f"positive rate ({base_rate:.4f})")
    _style(ax, "Precision-recall curve, held-out test split", "Recall", "Precision")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "pr")


def plot_calibration(table: dict, p, model_name: str) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.plot([0, 1], [0, 1], color=NAVY, lw=1, ls="--", label="perfect calibration")
    ax.plot(
        table["mean_predicted"],
        table["fraction_positive"],
        color=BLUE,
        lw=2,
        marker="o",
        label=f"{model_name} (ECE {table['ece']:.4f})",
    )
    _style(ax, "Calibration, held-out test split (quantile bins)", "", "Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax2.hist(np.asarray(p, dtype=float), bins=25, range=(0, 1), color=CORAL)
    _style(ax2, "", "Predicted probability", "Count")
    fig.tight_layout()
    _save(fig, "calibration")


def plot_importance(importance: dict) -> None:
    rows = importance["features"][::-1]
    names = [r["feature"] for r in rows]
    means = [r["importance_mean"] for r in rows]
    stds = [r["importance_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.5))
    ax.barh(names, means, xerr=stds, color=BLUE, ecolor=NAVY, capsize=2, height=0.62)
    _style(
        ax,
        f"Permutation importance (ROC-AUC drop, validation, {importance['n_repeats']} repeats)",
        "Mean decrease in ROC-AUC",
        "",
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "importance")


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _block(metrics: dict) -> dict:
    keys = (
        "threshold",
        "precision",
        "recall",
        "f1",
        "selection_rate",
        "tn",
        "fp",
        "fn",
        "tp",
        "expected_cost_per_1000",
    )
    return {k: metrics[k] for k in keys}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = settings.load_params()
    inputs = list(settings.MODEL_INPUT_COLUMNS)
    with open(settings.VERSION_PATH, encoding="utf-8") as fh:
        version = json.load(fh)
    model: Pipeline = joblib.load(settings.MODEL_PATH)
    baseline: Pipeline = joblib.load(BASELINE_PATH)
    val_df = pd.read_csv(settings.VAL_CSV_PATH)
    test_df = pd.read_csv(settings.TEST_CSV_PATH)
    X_val, y_val = val_df[inputs], val_df[settings.TARGET].to_numpy().astype(int)
    X_test, y_test = test_df[inputs], test_df[settings.TARGET].to_numpy().astype(int)

    p_val = model.predict_proba(X_val)[:, 1]
    thresholds = choose_thresholds(y_val, p_val, params)
    log.info("thresholds chosen on validation: %s", thresholds)

    p_test = model.predict_proba(X_test)[:, 1]
    p_base = baseline.predict_proba(X_test)[:, 1]
    costs = params["threshold"]
    at_cost = compute_metrics(y_test, p_test, thresholds["threshold_cost_optimal"], costs)
    at_prec = compute_metrics(y_test, p_test, thresholds["threshold_precision_target"], costs)
    base = compute_metrics(y_test, p_base, thresholds["threshold_cost_optimal"], costs)
    calibration = calibration_table(y_test, p_test, int(params["evaluate"]["calibration_bins"]))
    val_metrics = compute_metrics(y_val, p_val, thresholds["threshold_cost_optimal"])

    metrics = {
        "model": version["model"],
        "model_version": version["model_version"],
        "trained_at": version["trained_at"],
        "git_sha": version["git_sha"],
        "mlflow_run_id": version["mlflow_run_id"],
        "data_sha256": version["data_sha256"],
        "n_train": int(version["n_train"]),
        "n_val": int(version["n_val"]),
        "n_test": len(test_df),
        "n_features": int(version["n_features"]),
        "positive_rate_test": _r(y_test.mean()),
        "roc_auc": at_cost["roc_auc"],
        "pr_auc": at_cost["pr_auc"],
        "brier": at_cost["brier"],
        "ks": at_cost["ks"],
        "validation": {k: val_metrics[k] for k in ("roc_auc", "pr_auc", "brier")},
        "threshold_cost_optimal": thresholds["threshold_cost_optimal"],
        "threshold_precision_target": thresholds["threshold_precision_target"],
        "precision_target_met": thresholds["precision_target_met"],
        "at_threshold": _block(at_cost),
        "at_precision_target": _block(at_prec),
        "threshold_selection": {
            "selected_on": "validation",
            "cost_false_negative": thresholds["cost_false_negative"],
            "cost_false_positive": thresholds["cost_false_positive"],
            "target_precision": thresholds["target_precision"],
            "grid_step": thresholds["grid_step"],
            "validation_precision_at_target": thresholds["validation_precision_at_target"],
        },
        "calibration": calibration,
        "baseline_logreg": {k: base[k] for k in ("roc_auc", "pr_auc", "brier", "ks")},
        "lift_over_baseline": {
            "roc_auc": _r(at_cost["roc_auc"] - base["roc_auc"]),
            "pr_auc": _r(at_cost["pr_auc"] - base["pr_auc"]),
        },
    }

    importance = feature_importance(model, X_val, y_val, params)

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.METRICS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    with open(settings.IMPORTANCE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(importance, fh, indent=2)
        fh.write("\n")

    name = metrics["model"]
    plot_roc(y_test, p_test, p_base, name, metrics["roc_auc"], base["roc_auc"])
    plot_pr(y_test, p_test, p_base, name, metrics["pr_auc"], base["pr_auc"])
    plot_calibration(calibration, p_test, name)
    plot_importance(importance)

    print(
        f"test roc_auc={metrics['roc_auc']} pr_auc={metrics['pr_auc']} brier={metrics['brier']} "
        f"ks={metrics['ks']} threshold={metrics['threshold_cost_optimal']} "
        f"baseline_roc_auc={base['roc_auc']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
