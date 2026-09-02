"""Selection rate, TPR, FPR, and precision by sex and by age band on the held-out test split.

Usage:
    python -m credit_risk.fairness

SEX and MARRIAGE are never model inputs. They stay in the split files only so this report can
be computed from held-out labels and predictions at the cost-optimal threshold. AGE is a model
input; it is reported by band so the effect of retaining it is visible.
"""

from __future__ import annotations

import json
import logging
import sys
from itertools import pairwise

import joblib
import numpy as np
import pandas as pd

from credit_risk import settings

log = logging.getLogger(__name__)

SEX_LABELS = {1: "male", 2: "female"}
GAP_METRICS = ("selection_rate", "tpr", "fpr")


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)


def _rate(numerator: int, denominator: int) -> float | None:
    return _r(numerator / denominator) if denominator else None


def _band_labels(bands: list[int]) -> list[str]:
    return [f"{lo}-{hi - 1}" for lo, hi in pairwise(bands)]


def group_report(
    df: pd.DataFrame,
    y_true,
    y_pred,
    column: str,
    bands: list[int] | None = None,
    labels: dict | None = None,
) -> list[dict]:
    """Per-group outcome rates for one attribute.

    bands, when given, cut the column into [lo, hi) intervals. labels maps raw codes to names.
    Rows outside every band are skipped. Rates that have no denominator are None.
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    if bands is not None:
        keys = pd.cut(df[column], bins=bands, right=False, labels=_band_labels(bands))
    else:
        keys = df[column]
    rows: list[dict] = []
    for key in list(keys.cat.categories) if bands is not None else sorted(keys.dropna().unique()):
        mask = (keys == key).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        yg, pg = y[mask], pred[mask]
        tp = int(((pg == 1) & (yg == 1)).sum())
        fp = int(((pg == 1) & (yg == 0)).sum())
        fn = int(((pg == 0) & (yg == 1)).sum())
        tn = int(((pg == 0) & (yg == 0)).sum())
        name = str(key)
        if labels is not None:
            name = str(labels.get(key, key))
        row = {
            "group": name,
            "value": str(key) if bands is not None else int(key),
            "n": n,
            "positive_rate": _r(yg.mean()),
            "selection_rate": _r(pg.mean()),
            "tpr": _rate(tp, tp + fn),
            "fpr": _rate(fp, fp + tn),
            "precision": _rate(tp, tp + fp),
        }
        rows.append(row)
    return rows


def summarise(groups: list[dict]) -> dict:
    """max_gap for each rate and the demographic parity ratio (min / max selection rate)."""
    max_gap = {}
    for metric in GAP_METRICS:
        values = [g[metric] for g in groups if g[metric] is not None]
        max_gap[metric] = _r(max(values) - min(values)) if values else None
    rates = [g["selection_rate"] for g in groups if g["selection_rate"] is not None]
    ratio = _r(min(rates) / max(rates)) if rates and max(rates) > 0 else None
    return {"groups": groups, "max_gap": max_gap, "demographic_parity_ratio": ratio}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = settings.load_params()
    with open(settings.METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)
    threshold = float(metrics["threshold_cost_optimal"])
    model = joblib.load(settings.MODEL_PATH)
    test_df = pd.read_csv(settings.TEST_CSV_PATH)
    y_true = test_df[settings.TARGET].to_numpy().astype(int)
    p = model.predict_proba(test_df[list(settings.MODEL_INPUT_COLUMNS)])[:, 1]
    y_pred = (p >= threshold).astype(int)
    bands = [int(b) for b in params["fairness"]["age_bands"]]

    report = {
        "evaluated_on": "test",
        "n": len(test_df),
        "threshold": _r(threshold),
        "model_version": metrics.get("model_version"),
        "note": (
            "SEX and MARRIAGE are never model inputs; AGE is a model input and is reported "
            "by band to show its effect"
        ),
        "sex": summarise(group_report(test_df, y_true, y_pred, "SEX", labels=SEX_LABELS)),
        "age_band": {
            "bands": bands,
            **summarise(group_report(test_df, y_true, y_pred, "AGE", bands=bands)),
        },
    }
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.FAIRNESS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(
        f"wrote {settings.FAIRNESS_PATH} threshold={threshold} "
        f"sex_dpr={report['sex']['demographic_parity_ratio']} "
        f"age_dpr={report['age_band']['demographic_parity_ratio']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
