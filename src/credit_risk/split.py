"""Stratified train, validation, test split with a saved manifest.

Usage:
    python -m credit_risk.split

Two train_test_split calls (hold out 40 percent, then halve it) so the split is reproducible
from params.split.seed alone. The manifest records sizes, positive rates, and a sha256 of the
sorted ID list per split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk import settings
from credit_risk.data import FETCH_MANIFEST_PATH

log = logging.getLogger(__name__)


def ids_sha256(ids) -> str:
    """sha256 of the comma-joined, sorted ID list."""
    joined = ",".join(str(int(i)) for i in sorted(int(i) for i in ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _data_source() -> str:
    if FETCH_MANIFEST_PATH.exists():
        with open(FETCH_MANIFEST_PATH, encoding="utf-8") as fh:
            return str(json.load(fh).get("data_source", "unknown"))
    return "unknown"


def make_split(
    df: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (train, val, test, manifest) stratified by the target."""
    cfg = params["split"]
    seed = int(cfg["seed"])
    f_train, f_val, f_test = float(cfg["train"]), float(cfg["val"]), float(cfg["test"])
    if abs(f_train + f_val + f_test - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    target = settings.TARGET

    train_df, rest = train_test_split(
        df, test_size=f_val + f_test, stratify=df[target], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest, test_size=f_test / (f_val + f_test), stratify=rest[target], random_state=seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    manifest = {
        "seed": seed,
        "fractions": {"train": f_train, "val": f_val, "test": f_test},
        "n_total": len(df),
        "n": {k: len(v) for k, v in splits.items()},
        "positive_rate": {k: round(float(v[target].mean()), 4) for k, v in splits.items()},
        "id_sha256": {k: ids_sha256(v[settings.ID_COLUMN]) for k, v in splits.items()},
        "data_sha256": params["data"]["zip_sha256"],
        "data_source": _data_source(),
        "stratify": target,
    }
    return train_df, val_df, test_df, manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = settings.load_params()
    df = pd.read_csv(settings.FEATURES_CSV_PATH)
    train_df, val_df, test_df, manifest = make_split(df, params)

    settings.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        frame.to_csv(settings.SPLIT_DIR / f"{name}.csv", index=False, lineterminator="\n")
    with open(settings.SPLIT_MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(
        f"wrote splits to {settings.SPLIT_DIR}: n={manifest['n']} "
        f"positive_rate={manifest['positive_rate']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
