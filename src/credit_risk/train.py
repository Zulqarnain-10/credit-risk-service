"""Train the logistic-regression baseline and the HGB model, log both to MLflow, keep the winner.

Usage:
    python -m credit_risk.train

Both models are sklearn Pipelines that start with FeatureBuilder, so they accept the raw model
input columns. The winner is the model with the higher validation ROC-AUC (ties go to the
primary model from params.yaml). It is saved as models/model.joblib; the baseline is always
saved as models/baseline_logreg.joblib. models/version.json records provenance.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import __version__, settings
from credit_risk.features import FeatureBuilder

log = logging.getLogger(__name__)

MODEL_NAMES: tuple[str, ...] = ("logreg", "hgb")
BASELINE_PATH = settings.MODELS_DIR / "baseline_logreg.joblib"
NAVY, BLUE, CORAL = "#0A1628", "#2D8CFF", "#FF6B35"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


# Files that dvc repro rewrites; a diff there does not make the source tree dirty.
PIPELINE_OUTPUTS = ("dvc.lock", "models/", "reports/", "configs/presets.json", "data/")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=settings.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_info() -> dict:
    """Current commit sha, its short form, and whether tracked files are modified."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return {"git_sha": "unknown", "git_sha_short": "unknown", "git_dirty": False}
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    # Pipeline outputs change while dvc repro runs; only modified source counts as dirty.
    changed = [path for path in _status_paths(status) if not path.startswith(PIPELINE_OUTPUTS)]
    return {"git_sha": sha, "git_sha_short": sha[:7], "git_dirty": bool(changed)}


def _status_paths(porcelain: str) -> list[str]:
    """Paths from git status --porcelain, tolerant of stripped leading columns and renames."""
    paths = []
    for raw in porcelain.splitlines():
        line = raw.strip()
        if not line:
            continue
        path = line.split(maxsplit=1)[1] if " " in line else line
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.append(path.strip().strip('"').replace("\\", "/"))
    return paths


def library_versions() -> dict:
    def _version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mlflow": _version("mlflow"),
        "joblib": joblib.__version__,
    }


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, else the local file store under the repo root.

    MLflow 3 keeps the filesystem backend behind an opt-in flag; it is set here so the
    default file:./mlruns store works without extra setup.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI") or (settings.REPO_ROOT / "mlruns").as_uri()
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def make_logreg(params: dict) -> Pipeline:
    cfg = params["train"]["logreg"]
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg["C"]),
                    max_iter=int(cfg["max_iter"]),
                    random_state=int(params["train"]["seed"]),
                ),
            ),
        ]
    )


def make_hgb(params: dict) -> Pipeline:
    cfg = params["train"]["hgb"]
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=float(cfg["learning_rate"]),
                    max_iter=int(cfg["max_iter"]),
                    max_leaf_nodes=int(cfg["max_leaf_nodes"]),
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    l2_regularization=float(cfg["l2_regularization"]),
                    early_stopping=bool(cfg["early_stopping"]),
                    validation_fraction=float(cfg["validation_fraction"]),
                    n_iter_no_change=int(cfg["n_iter_no_change"]),
                    random_state=int(params["train"]["seed"]),
                ),
            ),
        ]
    )


MODEL_FACTORIES = {"logreg": make_logreg, "hgb": make_hgb}


def validation_metrics(y_true, p) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
    }


def pick_winner(val_metrics: dict, primary: str = "hgb") -> str:
    """Highest validation ROC-AUC wins; ties go to the primary model."""
    best = max(val_metrics.values(), key=lambda m: m["roc_auc"])["roc_auc"]
    tied = [name for name, m in val_metrics.items() if m["roc_auc"] == best]
    return primary if primary in tied else tied[0]


def hgb_n_iter(model: Pipeline) -> int | None:
    """Boosting iterations actually run (early stopping can cut max_iter short)."""
    clf = model.named_steps.get("clf")
    if isinstance(clf, HistGradientBoostingClassifier) and hasattr(clf, "n_iter_"):
        return int(clf.n_iter_)
    return None


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _flat_params(name: str, params: dict) -> dict:
    cfg = params["train"][name]
    out = {f"{name}.{k}": v for k, v in cfg.items()}
    out["seed"] = params["train"]["seed"]
    return out


def _serialization_kwargs() -> dict:
    """MLflow 3 serialises sklearn models with skops and must trust FeatureBuilder by name.

    Older MLflow versions have no such parameter; cloudpickle handles the custom step there.
    """
    import inspect

    import mlflow.sklearn

    trusted = ["credit_risk.features.FeatureBuilder"]
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        return {"skops_trusted_types": trusted}
    return {"serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE}


def _validation_figure(y_val, p_val, name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_val, p_val)
    precision, recall, _ = precision_recall_curve(y_val, p_val)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(fpr, tpr, color=BLUE, lw=2, label=name)
    ax1.plot([0, 1], [0, 1], color=NAVY, lw=1, ls="--", label="chance")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (validation)")
    ax1.legend(frameon=False)
    ax2.plot(recall, precision, color=CORAL, lw=2, label=name)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (validation)")
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _log_run(
    name: str,
    model: Pipeline,
    metrics: dict,
    params: dict,
    X_val: pd.DataFrame,
    y_val,
    p_val,
    tags: dict,
) -> tuple[str, str | None]:
    """Log one training run to MLflow. Returns (run_id, logged model uri or None)."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model": name, **tags})
        mlflow.log_params(_flat_params(name, params))
        mlflow.log_params({"n_features": len(settings.MODEL_FEATURE_COLUMNS)})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        n_iter = hgb_n_iter(model)
        if n_iter is not None:
            mlflow.log_metric("hgb_n_iter", n_iter)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            joblib.dump(model, tmp_path / f"{name}.joblib", compress=3)
            pred = (np.asarray(p_val) >= 0.5).astype(int)
            y_arr = np.asarray(y_val).astype(int)
            confusion = {
                "threshold": 0.5,
                "tn": int(((pred == 0) & (y_arr == 0)).sum()),
                "fp": int(((pred == 1) & (y_arr == 0)).sum()),
                "fn": int(((pred == 0) & (y_arr == 1)).sum()),
                "tp": int(((pred == 1) & (y_arr == 1)).sum()),
            }
            with open(tmp_path / "val_metrics.json", "w", encoding="utf-8") as fh:
                json.dump({**metrics, "confusion_at_0.5": confusion}, fh, indent=2)
            _validation_figure(y_val, p_val, name, tmp_path / "val_curves.png")
            mlflow.log_artifacts(tmp)

        model_uri: str | None = None
        try:
            signature = infer_signature(X_val.head(100), np.asarray(p_val[:100]))
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                pip_requirements=[
                    f"scikit-learn=={sklearn.__version__}",
                    f"pandas=={pd.__version__}",
                    f"numpy=={np.__version__}",
                ],
                **_serialization_kwargs(),
            )
            model_uri = info.model_uri
        except Exception as exc:
            log.warning("mlflow model logging failed for %s: %s", name, exc)
        return run.info.run_id, model_uri


def register_winner(model_uri: str | None, run_id: str | None, params: dict) -> bool:
    """Register the winner in the MLflow model registry. Returns False if the store refuses."""
    if not model_uri and not run_id:
        return False
    import mlflow

    name = params["train"]["mlflow"]["registered_model"]
    uri = model_uri or f"runs:/{run_id}/model"
    try:
        version = mlflow.register_model(uri, name)
    except Exception as exc:
        log.warning("model registration failed (%s); recording registered=false", exc)
        return False
    log.info("registered %s version %s", name, version.version)
    return True


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict,
    *,
    mlflow_enabled: bool = True,
    git_sha: str | None = None,
) -> dict:
    """Fit both models on train, score on validation, pick the winner.

    Returns {"winner", "models", "val_metrics", "run_ids", "model_uris", "fit_seconds"}.
    """
    inputs = list(settings.MODEL_INPUT_COLUMNS)
    X_train, y_train = train_df[inputs], train_df[settings.TARGET].to_numpy().astype(int)
    X_val, y_val = val_df[inputs], val_df[settings.TARGET].to_numpy().astype(int)

    models: dict[str, Pipeline] = {}
    val_metrics: dict[str, dict] = {}
    probabilities: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    for name in MODEL_NAMES:
        model = MODEL_FACTORIES[name](params)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds[name] = round(time.perf_counter() - started, 2)
        p_val = model.predict_proba(X_val)[:, 1]
        models[name] = model
        probabilities[name] = p_val
        val_metrics[name] = validation_metrics(y_val, p_val)
        log.info("%s fitted in %.1fs: %s", name, fit_seconds[name], val_metrics[name])

    winner = pick_winner(val_metrics, params["train"].get("primary", "hgb"))
    run_ids: dict[str, str | None] = dict.fromkeys(MODEL_NAMES)
    model_uris: dict[str, str | None] = dict.fromkeys(MODEL_NAMES)

    if mlflow_enabled:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(params["train"]["mlflow"]["experiment"])
        tags = {
            "git_sha": git_sha or git_info()["git_sha"],
            "package_version": __version__,
            "data_sha256": params["data"]["zip_sha256"],
        }
        for name in MODEL_NAMES:
            run_ids[name], model_uris[name] = _log_run(
                name,
                models[name],
                val_metrics[name],
                params,
                X_val,
                y_val,
                probabilities[name],
                {**tags, "winner": str(name == winner).lower()},
            )

    return {
        "winner": winner,
        "models": models,
        "val_metrics": val_metrics,
        "run_ids": run_ids,
        "model_uris": model_uris,
        "fit_seconds": fit_seconds,
    }


def build_version(result: dict, params: dict, n_train: int, n_val: int, registered: bool) -> dict:
    winner = result["winner"]
    git = git_info()
    return {
        "model": winner,
        "package_version": __version__,
        "model_version": f"{__version__}+{git['git_sha_short']}",
        **git,
        "mlflow_experiment": params["train"]["mlflow"]["experiment"],
        "mlflow_run_id": result["run_ids"][winner],
        "baseline_mlflow_run_id": result["run_ids"]["logreg"],
        "registered": registered,
        "registered_model": params["train"]["mlflow"]["registered_model"],
        "data_sha256": params["data"]["zip_sha256"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_features": len(settings.MODEL_FEATURE_COLUMNS),
        "feature_columns": list(settings.MODEL_FEATURE_COLUMNS),
        "hgb_n_iter": hgb_n_iter(result["models"]["hgb"]),
        "validation_metrics": result["val_metrics"],
        "fit_seconds": result["fit_seconds"],
        "libraries": library_versions(),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    params = settings.load_params()
    train_df = pd.read_csv(settings.TRAIN_CSV_PATH)
    val_df = pd.read_csv(settings.VAL_CSV_PATH)

    result = fit_models(train_df, val_df, params, mlflow_enabled=True)
    winner = result["winner"]
    registered = register_winner(result["model_uris"][winner], result["run_ids"][winner], params)

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["models"][winner], settings.MODEL_PATH, compress=3)
    joblib.dump(result["models"]["logreg"], BASELINE_PATH, compress=3)
    version = build_version(result, params, len(train_df), len(val_df), registered)
    with open(settings.VERSION_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(version, fh, indent=2)
        fh.write("\n")

    print(
        f"winner={winner} val={result['val_metrics']} registered={registered} "
        f"run_id={result['run_ids'][winner]} total={time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
