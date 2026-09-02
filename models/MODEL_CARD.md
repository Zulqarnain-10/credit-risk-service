# Model card: `credit-risk-service`

This is a demonstration system trained on the 2005 Taiwan credit-card dataset (UCI id 350). It
is not a U.S. underwriting model and must not be used for real credit decisions. The same
sentence is returned by the API in every prediction response.

Every number below is copied from `reports/metrics.json`, `reports/fairness.json`,
`reports/importance.json`, or `models/version.json` produced by `dvc repro`. Nothing is typed
from memory. Metrics are rounded to four decimal places as stored in those files.

## Provenance (`models/version.json`)

| Field | Value |
|---|---|
| Shipped model | `hgb` (scikit-learn `HistGradientBoostingClassifier` inside a `Pipeline`) |
| Model version | `0.1.0+412322f` |
| Package version | 0.1.0 |
| Git sha at training time | `412322fa6f89c726eb5a14dcbd669c817d1bc220` (`git_dirty`: false) |
| Trained at | 2026-09-02T17:05:32Z |
| MLflow experiment | `credit-risk` (local file store, `mlruns/`) |
| MLflow run id (winner) | `864b4665479043508627044a72119946` |
| MLflow run id (baseline) | `9d122ced460c4bd7a9c53b04e299b813` |
| Registered model | `credit-risk`, `registered`: true |
| Data sha256 (UCI zip) | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| Rows used | 18,000 train, 6,000 validation, 6,000 test |
| Features seen by the model | 45 (21 raw inputs plus 24 engineered) |
| Boosting iterations run | 103 of a maximum of 400 (early stopping) |
| Libraries | Python 3.12.2, scikit-learn 1.9.0, numpy 2.5.2, pandas 2.3.3, mlflow 3.15.2, joblib 1.6.0 |

## Intended use

- Purpose: show a complete, reproducible path from raw public data to a monitored prediction
  endpoint. The prediction is the probability that a cardholder misses next month's payment,
  plus a decision at a documented threshold.
- Users: people reviewing the repository, the demo page, or the API.
- Out of scope: any real lending, pricing, collections, or account decision, in any country. The
  data is a single 2005 snapshot from one Taiwanese issuer; the cost matrix is illustrative; no
  legal or compliance review has been done.

## Data

UCI Machine Learning Repository dataset 350, Yeh and Lien (2009), CC BY 4.0. 30,000 rows,
positive rate 0.2212. Stratified 60 / 20 / 20 split with seed 42; each split has positive rate
0.2212. Full schema, quirks, hashes, and split manifest: `data/DATA_CARD.md`.

## Features

The pipeline's first step is `FeatureBuilder`, a stateless transformer that takes the 21 raw
model inputs and appends 24 engineered columns (`src/credit_risk/features.py`). The model sees
all 45.

Raw inputs (21): `LIMIT_BAL`, `EDUCATION`, `AGE`, `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`,
`PAY_6`, `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`.

Engineered (24), with k = 1..6 over the six monthly columns:

| Feature | Formula |
|---|---|
| `util_k` (6) | `BILL_AMTk / LIMIT_BAL`, 0 when `LIMIT_BAL` is not positive |
| `util_mean` | mean of `util_1` to `util_6` |
| `util_max` | max of `util_1` to `util_6` |
| `pay_ratio_k` (6) | `min(PAY_AMTk / BILL_AMTk, 5.0)` when `BILL_AMTk > 0`, else 1.0 (no bill or a credit balance counts as fully paid) |
| `pay_ratio_mean` | mean of `pay_ratio_1` to `pay_ratio_6` |
| `delinq_max` | `max(clip(PAY_*, 0))` |
| `delinq_mean` | `mean(clip(PAY_*, 0))` |
| `delinq_months` | count of `PAY_*` columns with a value above 0 |
| `delinq_recent` | 1 if `PAY_0 > 0`, else 0 |
| `bill_trend` | `BILL_AMT1 - BILL_AMT6` |
| `pay_trend` | `PAY_AMT1 - PAY_AMT6` |
| `bill_mean` | mean of `BILL_AMT1` to `BILL_AMT6` |
| `pay_amt_mean` | mean of `PAY_AMT1` to `PAY_AMT6` |
| `zero_pay_months` | count of `PAY_AMT*` columns equal to 0 |

All features are deterministic, vectorized, and finite; nothing is fitted in this step.

## Excluded and retained attributes

Excluded: `SEX` and `MARRIAGE`. They are protected characteristics, they are not inputs to the
pipeline at any step, and the shipped `model.joblib` contains no reference to them. They stay in
the split files only so the fairness report can be computed from held-out labels.

Retained: `AGE` and `EDUCATION`. In this dataset they carry legitimate repayment-capacity signal
(income stage and stability), and the fairness report below measures what keeping them does to
outcomes by age band. This is a deliberate, documented choice for a demonstration system. A real
lender would need a legal review of both attributes under the rules that apply to it (in the
United States that includes the Equal Credit Opportunity Act) before using either one, and
should expect to drop or constrain them.

## Model choice and hyperparameters (`params.yaml`)

Two models are trained on the train split and scored on the validation split. Seed 42 everywhere.
HGB's early stopping holds out a further 15 percent of the train split as its own stopping set
(`validation_fraction` 0.15 in `params.yaml`, seeded), so its trees are fit on 15,300 of the
18,000 train rows (18,000 x 0.85) and the other 2,700 only decide when to stop. The 6,000-row
validation split is used for model selection, thresholds, and importance, never for fitting.

| Model | Pipeline | Hyperparameters |
|---|---|---|
| `logreg` (baseline) | `FeatureBuilder` -> `StandardScaler` -> `LogisticRegression` | `C` 1.0, `max_iter` 2000 |
| `hgb` (primary) | `FeatureBuilder` -> `HistGradientBoostingClassifier` | `learning_rate` 0.05, `max_iter` 400, `max_leaf_nodes` 31, `min_samples_leaf` 40, `l2_regularization` 1.0, `early_stopping` true, `validation_fraction` 0.15, `n_iter_no_change` 30 |

Winner rule: the model with the higher validation ROC-AUC ships as `models/model.joblib`; a tie
goes to `hgb`. The baseline always ships as `models/baseline_logreg.joblib`.

Validation results that decided the winner (`models/version.json`, `validation_metrics`):

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| `hgb` | 0.7756 | 0.5336 | 0.1380 |
| `logreg` | 0.7490 | 0.4903 | 0.1434 |

## Metrics

Measured on the held-out test split, n = 6000 (`reports/metrics.json`). The test split was not
used for any modeling or threshold decision.

| Metric | `hgb` (shipped) | `logreg` baseline | Lift |
|---|---|---|---|
| ROC-AUC | 0.7909 | 0.7672 | 0.0237 |
| PR-AUC (average precision) | 0.5744 | 0.5219 | 0.0525 |
| Brier score | 0.1316 | 0.1386 | |
| KS statistic | 0.4495 | 0.4239 | |

Test positive rate 0.2212. Figures: `reports/figures/roc.png`, `pr.png`, `calibration.png`,
`importance.png`.

## Threshold policy

Both thresholds are chosen on the validation split and then applied, unchanged, to the test
split. The cost matrix in `params.yaml` is illustrative, not a business calibration: a missed
default (false negative) is assumed to cost 5.0 and a wrongly declined good customer (false
positive) 1.0. The grid step is 0.005.

| Policy | Threshold | Precision | Recall | F1 | Selection rate | TN | FP | FN | TP | Expected cost per 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cost-optimal (the API default) | 0.155 | 0.3667 | 0.7905 | 0.5010 | 0.4768 | 2861 | 1812 | 278 | 1049 | 533.6667 |
| Precision target 0.60 | 0.365 | 0.6144 | 0.4574 | 0.5244 | 0.1647 | 4292 | 381 | 720 | 607 | 663.5000 |

The precision target was met on validation (validation precision 0.6029 at 0.365,
`precision_target_met`: true). Under the 5:1 cost matrix the optimal threshold is low, so the
service flags 47.68 percent of test applicants and catches 79.05 percent of the defaults. That is
what the illustrative costs imply; a different matrix gives a different operating point.

## Fairness (`reports/fairness.json`)

Computed on the test split at the cost-optimal threshold 0.155, from held-out labels and the
protected attributes that are never model inputs.

By sex (`SEX`, 1 = male, 2 = female per the dataset documentation):

| Group | n | Label positive rate | Selection rate | TPR | FPR | Precision |
|---|---|---|---|---|---|---|
| male | 2372 | 0.2411 | 0.5000 | 0.7972 | 0.4056 | 0.3845 |
| female | 3628 | 0.2081 | 0.4617 | 0.7854 | 0.3766 | 0.3540 |

Largest gap: selection rate 0.0383, TPR 0.0118, FPR 0.0290. Demographic parity ratio (min / max
selection rate) 0.9234.

By age band (`AGE`, lower bound inclusive, upper bound exclusive, bands from `params.yaml`):

| Group | n | Label positive rate | Selection rate | TPR | FPR | Precision |
|---|---|---|---|---|---|---|
| 21-29 | 1926 | 0.2186 | 0.5000 | 0.8100 | 0.4133 | 0.3541 |
| 30-39 | 2206 | 0.1990 | 0.4275 | 0.7722 | 0.3418 | 0.3595 |
| 40-49 | 1296 | 0.2454 | 0.4977 | 0.7893 | 0.4029 | 0.3891 |
| 50-99 | 572 | 0.2605 | 0.5420 | 0.7919 | 0.4539 | 0.3806 |

Largest gap: selection rate 0.1145, TPR 0.0378, FPR 0.1121. Demographic parity ratio 0.7887.

Reading: the model selects men slightly more often than women, in line with the higher label rate
among men in this data; true positive rates are close across both attributes. The age gap is
larger: the 30-39 band is selected least and has the lowest false positive rate, the 50-99 band
the most. Part of that tracks the label rates, part is the retained `AGE` feature. This report
measures the gap; it does not claim the gap is acceptable.

## Calibration

Reliability on the test split with 10 quantile bins of 600 rows each (`reports/metrics.json`,
`calibration`). Expected calibration error 0.0109.

| Mean predicted | 0.0408 | 0.0612 | 0.0815 | 0.1066 | 0.1333 | 0.1638 | 0.2051 | 0.2744 | 0.4099 | 0.7090 |
|---|---|---|---|---|---|---|---|---|---|---|
| Observed rate | 0.0517 | 0.0600 | 0.0750 | 0.1250 | 0.1250 | 0.1517 | 0.1917 | 0.2867 | 0.4217 | 0.7233 |

The raw HGB scores track observed default rates closely across the range, so no post-hoc
calibration layer is applied. The probability the API returns is the model's own output.

## Top global drivers (`reports/importance.json`)

Permutation importance, ROC-AUC drop on the validation split, 10 repeats, seed 42. Mean and
standard deviation of the drop:

| Rank | Feature | Mean drop | Std |
|---|---|---|---|
| 1 | `delinq_max` | 0.0419 | 0.0023 |
| 2 | `PAY_0` | 0.0229 | 0.0015 |
| 3 | `PAY_AMT2` | 0.0046 | 0.0006 |
| 4 | `bill_mean` | 0.0045 | 0.0014 |
| 5 | `delinq_mean` | 0.0038 | 0.0008 |
| 6 | `pay_amt_mean` | 0.0029 | 0.0007 |
| 7 | `bill_trend` | 0.0024 | 0.0009 |
| 8 | `LIMIT_BAL` | 0.0023 | 0.0015 |
| 9 | `util_2` | 0.0021 | 0.0012 |
| 10 | `util_3` | 0.0020 | 0.0007 |
| 11 | `PAY_AMT1` | 0.0019 | 0.0007 |
| 12 | `util_max` | 0.0017 | 0.0003 |
| 13 | `EDUCATION` | 0.0016 | 0.0007 |
| 14 | `zero_pay_months` | 0.0015 | 0.0005 |
| 15 | `pay_ratio_3` | 0.0012 | 0.0005 |

Recent and worst delinquency status dominate; amounts and utilization add smaller, consistent
signal. `AGE` is not in the top 15.

## Limitations

- One issuer, one country, one snapshot: Taiwan, April to September 2005. Nothing here transfers
  to another market or decade without re-estimation.
- No time-based validation. The split is a random stratified one, so the metrics do not measure
  how the model degrades as behavior drifts. The drift tooling in this repo simulates that
  instead of measuring it on real future data.
- No macroeconomic, bureau, or income features. Six months of card history is all the model sees.
- Undocumented category codes (`EDUCATION` 0, 5, 6 and `MARRIAGE` 0) and the meaning of
  `PAY_*` codes -2 and 0 are interpreted, not specified by the publisher (`data/DATA_CARD.md`).
- The cost matrix and the precision target are illustrative settings from `params.yaml`.
- `AGE` and `EDUCATION` are model inputs. See the section above for why, and for the review a
  real deployment would require.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0, UCI Machine Learning Repository dataset 350, Yeh and
Lien (2009).
