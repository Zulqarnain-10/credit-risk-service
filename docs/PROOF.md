# Proof: where every public number comes from

Every number a reader meets in the README, on the demo page, in the model card, or in the data
card is produced by the pipeline and written to a file. This page maps each one to that file, the
JSON key, and the command that wrote it. Values are copied from the run recorded in
`models/version.json` (`trained_at` 2026-09-02T17:05:32Z, `git_sha_short` 412322f). After a
retrain the files win; refresh this page from them.

Commands run from the repository root inside the activated virtual environment. Stage names
refer to `dvc.yaml`; `dvc repro` runs them all in order.

## README, synced block

Rendered by `python -m credit_risk.sync_readme`. `--check` exits 1 when the block and the files
disagree.

| Number | Value | File | Key | Command |
|---|---|---|---|---|
| Test rows | 6,000 | `reports/metrics.json` | `n_test` | `python -m credit_risk.evaluate` (stage `evaluate`) |
| Test positive rate | 0.2212 | `reports/metrics.json` | `positive_rate_test` | stage `evaluate` |
| ROC-AUC, HGB | 0.7909 | `reports/metrics.json` | `roc_auc` | stage `evaluate` |
| ROC-AUC, baseline | 0.7672 | `reports/metrics.json` | `baseline_logreg.roc_auc` | stage `evaluate` |
| PR-AUC, HGB | 0.5744 | `reports/metrics.json` | `pr_auc` | stage `evaluate` |
| PR-AUC, baseline | 0.5219 | `reports/metrics.json` | `baseline_logreg.pr_auc` | stage `evaluate` |
| Brier, HGB | 0.1316 | `reports/metrics.json` | `brier` | stage `evaluate` |
| Brier, baseline | 0.1386 | `reports/metrics.json` | `baseline_logreg.brier` | stage `evaluate` |
| KS, HGB | 0.4495 | `reports/metrics.json` | `ks` | stage `evaluate` |
| KS, baseline | 0.4239 | `reports/metrics.json` | `baseline_logreg.ks` | stage `evaluate` |
| Cost matrix | 5:1 | `reports/metrics.json` | `threshold_selection.cost_false_negative`, `.cost_false_positive` (declared in `params.yaml` `threshold`) | stage `evaluate` |
| Cost-optimal threshold | 0.155 | `reports/metrics.json` | `threshold_cost_optimal` | stage `evaluate` |
| Precision, recall, selection rate at it | 0.3667, 0.7905, 0.4768 | `reports/metrics.json` | `at_threshold.precision`, `.recall`, `.selection_rate` | stage `evaluate` |
| Precision target | 0.60 | `reports/metrics.json` | `threshold_selection.target_precision` (declared in `params.yaml`) | stage `evaluate` |
| Precision-target threshold | 0.365 | `reports/metrics.json` | `threshold_precision_target` | stage `evaluate` |
| Precision, recall, selection rate at it | 0.6144, 0.4574, 0.1647 | `reports/metrics.json` | `at_precision_target.precision`, `.recall`, `.selection_rate` | stage `evaluate` |
| p95 latency | the file's `p95_ms`, in ms; it changes with every load-test run, so the synced block copies it and this page does not | `reports/loadtest.json` | `p95_ms` | `python -m credit_risk.loadtest --url http://127.0.0.1:8000 --requests 300 --concurrency 10 --host "local uvicorn, Windows 11, Python 3.12, single process"` |
| Requests, concurrency, host | 300, 10, local uvicorn, Windows 11, Python 3.12, single process | `reports/loadtest.json` | `requests`, `concurrency`, `host` | same command |
| Demographic parity ratio by sex | 0.9234 | `reports/fairness.json` | `sex.demographic_parity_ratio` | `python -m credit_risk.fairness` (stage `fairness`) |
| Demographic parity ratio by age band | 0.7887 | `reports/fairness.json` | `age_band.demographic_parity_ratio` | stage `fairness` |
| Trained at | 2026-09-02T17:05:32Z | `models/version.json` | `trained_at` | `python -m credit_risk.train` (stage `train`) |
| Git sha (short) | 412322f | `models/version.json` | `git_sha_short` | stage `train` |
| Data sha256 (first 12) | 56c885f84457 | `models/version.json` | `data_sha256` (declared in `params.yaml` `data.zip_sha256`, verified by stage `fetch`) | stage `train` |
| Train, validation, test rows | 18,000 / 6,000 / 6,000 | `reports/metrics.json` | `n_train`, `n_val`, `n_test` | stage `evaluate` (also `models/version.json` `n_train`, `n_val`) |

## README, prose

| Number | Value | File | Key | Command |
|---|---|---|---|---|
| Dataset rows | 30,000 | `data/processed/splits/split_manifest.json` (rebuilt by `dvc repro`, not committed; copied into `data/DATA_CARD.md`) | `n_total`; equals `n_train + n_val + n_test` in `reports/metrics.json` | `python -m credit_risk.split` (stage `split`) |
| Attributes | 23 | `src/credit_risk/settings.py` | `RAW_FEATURE_COLUMNS` (length) | code, checked by `tests/test_schema.py` |
| Positive rate | 0.2212 | `reports/metrics.json` | `positive_rate_test` (all three splits share it, `split_manifest.json` `positive_rate`) | stage `evaluate` |
| DVC stages | 9 | `dvc.yaml` | stage count | `python -m dvc dag` |
| Raw inputs, features | 21, 45 | `models/version.json` | `feature_columns` (45 names; the first 21 are the raw inputs), `n_features` | stage `train` |
| First preset returns | 0.0526, `unlikely_default` | `configs/presets.json` | `presets[0].model_probability` (0.052585, rounded to 4 dp by the API), compared with `threshold_cost_optimal` | `python -m credit_risk.presets` (stage `presets`) |
| First preset payload | 21 field values | `configs/presets.json` | `presets[0].input` | stage `presets` |
| Batch cap | 100 | `src/credit_risk/api/schemas.py` | `BATCH_MAX_ITEMS` | code, checked by `tests/test_api.py` |
| Drift perturbation | 0.85, 4 years, 1.25, 15 percent | `reports/drift_summary.json` | `current.perturbation.limit_bal_scale`, `.age_shift_years`, `.bill_amt_scale`, `.pay_status_shift_share` (declared in `params.yaml` `drift`) | `python -m credit_risk.monitoring drift` |
| Drifted-column count, drift share, dataset-drift verdict, PSI per column | read from the file; the README points at it instead of quoting them | `reports/drift_summary.json` | `n_drifted`, `n_features`, `drift_share`, `drift_share_threshold`, `dataset_drift`, `psi.*` | `python -m credit_risk.monitoring drift` |
| CI tolerance | 0.005 | `.github/workflows/ci.yml` | `--tolerance 0.005` on `scripts/compare_metrics.py` | CI job `reproduce` |
| Image size warning | 400 MB | `.github/workflows/ci.yml` | the `Image size` step | CI job `docker` |
| Selection rate at the cost threshold | 0.4768 | `reports/metrics.json` | `at_threshold.selection_rate` | stage `evaluate` |
| Ports | 8000 local, 7860 on the Space | `Dockerfile` (`PORT=8000`), `scripts/deploy_space.py` (`SPACE_PORT = 7860`) | | build and deploy |

## Demo page tiles (`src/credit_risk/api/static/index.html`)

The page never hard-codes a number; each tile reads an endpoint, and each endpoint reads a file.

| Tile | Endpoint | File | Key |
|---|---|---|---|
| Model version in the eyebrow | `GET /version` | `models/version.json` | `model_version` |
| Disclaimer | `GET /version` | `src/credit_risk/settings.py` | `DISCLAIMER` |
| Test split n | `GET /version` | `reports/metrics.json` | `metrics.n_test` |
| ROC-AUC | `GET /version` | `reports/metrics.json` | `metrics.roc_auc` |
| PR-AUC | `GET /version` | `reports/metrics.json` | `metrics.pr_auc` |
| Brier | `GET /version` | `reports/metrics.json` | `metrics.brier` |
| Trained date | `GET /version` | `reports/metrics.json` | `metrics.trained_at` |
| p95 latency and host | `GET /version` | `reports/loadtest.json` | `loadtest.p95_ms`, `loadtest.host` (chip reads `[todo]` when the file is absent from the build) |
| Decision threshold in the result card | `GET /version` | `reports/metrics.json` | `metrics.threshold_cost_optimal` |
| Preset buttons and form values | `GET /presets` | `configs/presets.json` | `presets[*].input`, `presets[*].label` |
| Probability, decision, threshold, version after submit | `POST /predict` | `models/model.joblib`, `reports/metrics.json` | response fields `probability`, `decision_label`, `threshold`, `model_version` |
| Top global drivers (eight bars) | `GET /importance` | `reports/importance.json` | `features[0:8].feature`, `.importance_mean`; method line from `n_repeats`, `evaluated_on` |
| Curl payload in "Run it yourself" | `GET /presets` | `configs/presets.json` | `presets[0].input` |

## Model card (`models/MODEL_CARD.md`)

| Section | File | Keys | Command |
|---|---|---|---|
| Provenance table | `models/version.json` | `model`, `model_version`, `package_version`, `git_sha`, `git_dirty`, `trained_at`, `mlflow_experiment`, `mlflow_run_id`, `baseline_mlflow_run_id`, `registered`, `registered_model`, `data_sha256`, `n_train`, `n_val`, `n_features`, `hgb_n_iter`, `libraries` | stage `train` |
| Test rows in the provenance table | `reports/metrics.json` | `n_test` | stage `evaluate` |
| Data paragraph | `data/processed/splits/split_manifest.json` | `n_total`, `seed`, `fractions`, `positive_rate` | stage `split` |
| Hyperparameters | `params.yaml` | `train.logreg`, `train.hgb`, `train.seed` | declared |
| Validation table | `models/version.json` | `validation_metrics.hgb`, `validation_metrics.logreg` | stage `train` |
| Metrics table | `reports/metrics.json` | `roc_auc`, `pr_auc`, `brier`, `ks`, `baseline_logreg.*`, `lift_over_baseline.*`, `positive_rate_test` | stage `evaluate` |
| Threshold policy table | `reports/metrics.json` | `threshold_selection.*`, `threshold_cost_optimal`, `threshold_precision_target`, `precision_target_met`, `at_threshold.*`, `at_precision_target.*` | stage `evaluate` |
| "flags 47.68 percent, catches 79.05 percent" | `reports/metrics.json` | `at_threshold.selection_rate` (0.4768), `at_threshold.recall` (0.7905) as percentages | stage `evaluate` |
| Fairness tables and gaps | `reports/fairness.json` | `threshold`, `sex.groups[*]`, `sex.max_gap`, `sex.demographic_parity_ratio`, `age_band.*` | stage `fairness` |
| Calibration table and ECE | `reports/metrics.json` | `calibration.mean_predicted`, `.fraction_positive`, `.counts`, `.ece`, `.bins` | stage `evaluate` |
| Top drivers table | `reports/importance.json` | `features[*].feature`, `.importance_mean`, `.importance_std`, `n_repeats`, `seed` | stage `evaluate` |

## Data card (`data/DATA_CARD.md`)

| Section | Source | Command |
|---|---|---|
| Hashes | `params.yaml` `data.zip_sha256`, `data.xls_sha256`; observed CSV hash | `python -m credit_risk.data info` |
| Shape, nulls, label counts, column ranges, category counts, negative bills, duplicate rows | printed by the command | `python -m credit_risk.data info` |
| Validation ranges | `src/credit_risk/settings.py` `FIELD_RANGES`, `src/credit_risk/validate.py` `RAW_SCHEMA` | code, checked by `tests/test_schema.py` |
| Split table | `data/processed/splits/split_manifest.json` | stage `split` |

## How CI reproduces these numbers

The `reproduce` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every
push to main and every pull request:

1. Check out the commit and install `requirements-pipeline.txt` plus the package.
2. Save the committed `reports/metrics.json` aside with `git show HEAD:reports/metrics.json`.
3. Run `python -m dvc repro -f`, which downloads the UCI archive, verifies its sha256, and rebuilds
   every stage from scratch on the runner, MLflow included (`MLFLOW_TRACKING_URI=file:./mlruns`).
4. Run `python scripts/compare_metrics.py --committed <saved> --fresh reports/metrics.json
   --tolerance 0.005`. Every numeric leaf must agree within 0.005 (scaled by magnitude for values
   above 1, so confusion-matrix counts and cost per 1,000 get the same relative slack); `n_train`,
   `n_val`, `n_test`, `data_sha256`, and `model` must be identical; `trained_at`, `git_sha`,
   `mlflow_run_id`, and `model_version` are ignored because they change on every run. Any other
   difference fails the job.
5. Upload the reproduced `reports/`, `models/version.json`, and `configs/presets.json` as the
   workflow artifact `reproduced-reports`, so the fresh numbers can be read next to the committed
   ones.

Four more jobs run alongside it: `lint` (ruff), `test`, `gitleaks`, and `docker` (build the
image, start it, score the first preset through `scripts/smoke_live.py`). The `test` job first
runs `python -m credit_risk.sync_readme --check`, then builds the data splits on the runner with
`python -m dvc repro split` (the split files are not committed), then runs pytest with coverage;
with the splits present, the contract tests in `tests/test_model_contract.py` re-score the shipped
model on the test split and require each headline metric to be within 0.002 of
`reports/metrics.json` instead of skipping.

Badge: [![CI](https://github.com/Zulqarnain-10/credit-risk-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Zulqarnain-10/credit-risk-service/actions/workflows/ci.yml)

CI run: [Actions run 33664125882](https://github.com/Zulqarnain-10/credit-risk-service/actions/runs/33664125882)

MLflow run: [todo: MLflow run link if DagsHub is added]. Locally, the winner's run id is
`models/version.json` `mlflow_run_id` (864b4665479043508627044a72119946) in experiment
`credit-risk` under `file:./mlruns`; the baseline is `baseline_mlflow_run_id`
(9d122ced460c4bd7a9c53b04e299b813).

The README block itself is checked by `python -m credit_risk.sync_readme --check`, which exits 1
when the rendered block differs from the file. Run it locally before committing; CI's `test` job
runs it before pytest, and the Retrain workflow runs `python -m credit_risk.sync_readme` after
`dvc repro` so its pull request carries the refreshed block.
