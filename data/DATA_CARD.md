# Data card: Default of Credit Card Clients (UCI id 350)

Every number on this page is printed by `python -m credit_risk.data info` (raw file facts),
written to `data/processed/splits/split_manifest.json` (split facts), or, where the text says so,
counted directly from the split CSVs. Nothing is typed from memory.

## Source and citation

- Repository: UCI Machine Learning Repository, dataset id 350, "Default of Credit Card Clients".
  Page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Archive: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
- Citation: Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. Expert Systems with
  Applications, 36(2), 2473-2480.
- Content: credit-card holders in Taiwan, billing and repayment records from April to
  September 2005, with the label "default payment next month" (October 2005).
- License: CC BY 4.0, verified on the UCI dataset page on 2026-09-02. Attribution is given above.

## Integrity

| File | sha256 |
|---|---|
| `uci_350.zip` (params.yaml `data.zip_sha256`) | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `default of credit card clients.xls` (params.yaml `data.xls_sha256`) | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| `data/raw/credit_default_raw.csv` (derived, observed) | `84ba892a2a55a0d711259f30084753f22b14b0c56303baf25ced4e7600230248` |

The `fetch` stage downloads the archive, checks the zip hash, extracts the workbook, checks its
hash, and only then reads it (`pandas.read_excel`, `header=1`, xlrd engine). A hash mismatch stops
the pipeline. The `validate` stage checks the zip hash again before enforcing the schema.

## Shape

- 30,000 rows, 25 columns: `ID`, 23 attributes, one binary target.
- No missing values in any column.
- Target `default_next_month`: 23,364 rows labeled 0, 6,636 rows labeled 1. Positive rate 0.2212.

## Schema

Types are as read from the workbook (every column is a whole number). "Observed" values come from
`python -m credit_risk.data info` on the raw CSV above. Amounts are in New Taiwan dollars.

| Column | Type | Documented meaning | Observed |
|---|---|---|---|
| `ID` | int | Row identifier | 1 to 30,000, unique |
| `LIMIT_BAL` | int | Credit limit, individual plus family or supplementary credit | 10,000 to 1,000,000 |
| `SEX` | int | 1 = male, 2 = female | 1: 11,888 rows; 2: 18,112 rows |
| `EDUCATION` | int | 1 = graduate school, 2 = university, 3 = high school, 4 = others | 0 to 6; 0: 14, 1: 10,585, 2: 14,030, 3: 4,917, 4: 123, 5: 280, 6: 51 |
| `MARRIAGE` | int | 1 = married, 2 = single, 3 = others | 0 to 3; 0: 54, 1: 13,659, 2: 15,964, 3: 323 |
| `AGE` | int | Age in years | 21 to 79 |
| `PAY_0` | int | Repayment status, September 2005 | -2 to 8 |
| `PAY_2` | int | Repayment status, August 2005 | -2 to 8 |
| `PAY_3` | int | Repayment status, July 2005 | -2 to 8 |
| `PAY_4` | int | Repayment status, June 2005 | -2 to 8 |
| `PAY_5` | int | Repayment status, May 2005 | -2 to 8 |
| `PAY_6` | int | Repayment status, April 2005 | -2 to 8 |
| `BILL_AMT1` | int | Bill statement amount, September 2005 | -165,580 to 964,511 |
| `BILL_AMT2` | int | Bill statement amount, August 2005 | -69,777 to 983,931 |
| `BILL_AMT3` | int | Bill statement amount, July 2005 | -157,264 to 1,664,089 |
| `BILL_AMT4` | int | Bill statement amount, June 2005 | -170,000 to 891,586 |
| `BILL_AMT5` | int | Bill statement amount, May 2005 | -81,334 to 927,171 |
| `BILL_AMT6` | int | Bill statement amount, April 2005 | -339,603 to 961,664 |
| `PAY_AMT1` | int | Amount paid, September 2005 | 0 to 873,552 |
| `PAY_AMT2` | int | Amount paid, August 2005 | 0 to 1,684,259 |
| `PAY_AMT3` | int | Amount paid, July 2005 | 0 to 896,040 |
| `PAY_AMT4` | int | Amount paid, June 2005 | 0 to 621,000 |
| `PAY_AMT5` | int | Amount paid, May 2005 | 0 to 426,529 |
| `PAY_AMT6` | int | Amount paid, April 2005 | 0 to 528,666 |
| `default_next_month` | int | Target: 1 = defaulted the next month, 0 = did not | 0: 23,364 rows; 1: 6,636 rows |

### Validation rules (`src/credit_risk/validate.py`)

The pandera schema `RAW_SCHEMA` rejects a frame that has any extra or missing column, any null,
any non-integral value, a duplicate `ID`, or a value outside these ranges:

| Column | Allowed range |
|---|---|
| `ID` | at least 1, unique |
| `LIMIT_BAL` | 1 to 2,000,000 |
| `SEX` | 1 to 2 |
| `EDUCATION` | 0 to 6 before collapse, 1 to 4 after |
| `MARRIAGE` | 0 to 3 before collapse, 1 to 3 after |
| `AGE` | 18 to 100 |
| `PAY_0`, `PAY_2` to `PAY_6` | -2 to 9 |
| `BILL_AMT1` to `BILL_AMT6` | -2,000,000 to 2,000,000 |
| `PAY_AMT1` to `PAY_AMT6` | 0 to 2,000,000 |
| `default_next_month` | 0 to 1 |

The same ranges are mirrored by the API request schema (`settings.FIELD_RANGES`).

## Known quirks

- Undocumented `EDUCATION` codes. The UCI page documents 1 to 4. Codes 0, 5, and 6 appear on
  14, 280, and 51 rows. The `validate` stage maps all three to 4 ("others"), so 345 rows change.
  The mapping is declared in `params.yaml` (`features.education_other_codes`,
  `features.education_other_value`).
- Undocumented `MARRIAGE` code. The UCI page documents 1 to 3. Code 0 appears on 54 rows and is
  mapped to 3 ("others") by the same stage (`features.marriage_other_codes`,
  `features.marriage_other_value`).
- `PAY_0` naming. The dataset names the September status column `PAY_0` and then continues with
  `PAY_2` to `PAY_6`. There is no `PAY_1`. The repo keeps the original names everywhere so the
  columns can be traced back to the source.
- `PAY_*` code meanings. The UCI page documents -1 = paid duly and 1 to 9 = months of payment
  delay. The observed values run from -2 to 8, so codes -2 and 0 are not documented. The common
  reading, and the one this repo uses in its feature definitions, is -2 = no consumption that
  month, -1 = paid in full, 0 = revolving credit (minimum paid, balance carried), and 1 to 8 =
  months of delay. The model treats the column as an ordered integer; the delinquency features
  clip negative codes to 0.
- Negative bill amounts. A negative statement means a credit balance (the customer overpaid or
  received a refund). Rows with a negative bill per column: `BILL_AMT1` 590, `BILL_AMT2` 669,
  `BILL_AMT3` 655, `BILL_AMT4` 675, `BILL_AMT5` 655, `BILL_AMT6` 688. The payment-ratio
  features treat a bill of zero or less as fully paid (ratio 1.0).
- Duplicate rows. Ignoring `ID`, 35 rows are exact duplicates of an earlier row. They are kept:
  they are plausible distinct customers with identical records, and dropping them would change the
  published row count without a documented reason. The split is by row, not by attribute tuple,
  so a pair can land on both sides of it: counted directly from the split CSVs, 15 of the 35 pairs
  straddle train and a held-out split, with 9 test rows and 6 validation rows having an exact twin
  (all 23 attributes and the label) in train.
- Protected attributes. `SEX` and `MARRIAGE` are validated and carried through the split files
  but are never model inputs. They exist downstream only for `reports/fairness.json`.

## Split

Stratified on the target, seed 42, two `train_test_split` calls (hold out 40 percent, then halve
it). From `data/processed/splits/split_manifest.json`:

| Split | Rows | Positive rate | sha256 of the sorted, comma-joined `ID` list |
|---|---|---|---|
| train | 18,000 | 0.2212 | `35ba2fa78ca2508b81aed935897a4e759dc97e92b16c14c7342ef2944f355169` |
| val | 6,000 | 0.2212 | `9ae1ab0240855ddaad3737693b9cf35a8bd227d2f403a05666c25d2df0bb1f28` |
| test | 6,000 | 0.2212 | `6383a8a8c1e0aed9faf0e2c923cef0cc0b4510ad841988cdd4ecdabd440fd0eb` |

Fractions 0.6 / 0.2 / 0.2 of 30,000 rows. `data_sha256` in the manifest is the zip hash above and
`data_source` records whether the rows came from the verified archive (`uci_archive`) or the
`ucimlrepo` fallback. This run used `uci_archive`.

## Handling and provenance

- Raw and processed data are never committed. `data/raw/` and `data/processed/` are gitignored
  and managed by DVC; `dvc repro` re-fetches the archive deterministically and rebuilds every
  derived file. The hashes above make a silent upstream change impossible to miss.
- If the UCI download fails three times, `fetch` falls back to `ucimlrepo.fetch_ucirepo(id=350)`
  and maps its `X1` to `X23` and `Y` columns onto the names above; the manifest records that source.
- The dataset contains no direct identifiers. `ID` is a row number assigned by the publisher.
- Downstream files: `data/processed/validated.csv` (same 25 columns, categories collapsed),
  `data/processed/features.csv` (plus 24 engineered columns), `data/processed/splits/*.csv`.
