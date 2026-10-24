# Medical Insurance Charge Prediction Using Machine Learning

**Student:** ____________________  **Roll number:** ____________________

**Department / College:** ____________________  **Guide:** ____________________

## Abstract

This project develops a supervised regression system for estimating numerical medical insurance charges from a six-feature profile. It preserves source data, audits cleaning, separates model selection from final evaluation, compares four approaches, and provides a local browser interface for individual and batch predictions. Gradient boosting obtained the lowest training cross-validation mean absolute error. A held-out test set is used to report MAE, RMSE, and R².

## Problem statement

Build a reproducible application that learns the relationship between age, sex, body mass index, dependents, smoking status, US region, and the charge recorded in an educational benchmark. The output is a continuous number. The application does not quote an actual policy premium or decide insurance eligibility.

## Objectives

- Preserve and audit the dataset.
- Compare a constant baseline, regularized linear model, random forest, and gradient boosting.
- Prevent preprocessing and model-selection leakage into the test set.
- Evaluate predictions with regression metrics.
- Provide form inputs, profile comparison, batch prediction, and downloadable results.

## Dataset

The public `insurance.csv` benchmark contains 1,338 rows and seven columns. Six columns are inputs and `charges` is the target. One exact duplicate is removed, leaving 1,337 rows. No missing cells are present in the downloaded data. Source attribution and a SHA-256 fingerprint are included in `data/SOURCE.md`.

Inputs are bounded to the benchmark's observed ranges: age 18–64, BMI 15.96–53.13, and dependents 0–5. The recorded categories for sex are female/male, smoking status no/yes, and region northeast/northwest/southeast/southwest. These categories describe this limited dataset, not all possible people or locations.

## Data preparation

The original CSV is preserved. Categorical values are stripped and lowercased; numeric fields are parsed; missing and nonfinite rows are excluded and logged. Exact duplicate records are removed before splitting. Targets must be nonnegative. Large charges are retained because high medical costs can be meaningful; no arbitrary outlier deletion is performed.

The split manifest preserves zero-based original CSV record indices. Numeric imputation/scaling and categorical imputation/one-hot encoding are placed inside a scikit-learn pipeline. They are learned from training partitions only. On this complete dataset the imputation steps are safeguards rather than an applied repair.

## System architecture

```text
Source CSV → audit and cleaning → 80/20 split
  → five-fold comparison within training data
  → select lowest mean CV MAE
  → fit selected model on all training rows
  → evaluate on held-out test rows
  → save preprocessing and model together
  → local form / CSV → schema validation → prediction → download
```

The application uses Python's built-in HTTP server bound to 127.0.0.1. The browser runs HTML, CSS, and JavaScript. There are no external web assets or API keys. The server uses the same prediction module as the CLI.

## Algorithms

**Mean baseline:** always returns the mean training charge. It establishes a simple reference.

**Ridge regression:** a linear model with L2 regularization. It balances squared prediction error against coefficient size.

**Random forest:** an ensemble of 200 decision trees; each leaf has at least three training rows.

**Gradient boosting:** 250 sequential depth-two trees, learning rate 0.04, and Huber loss. Each stage improves the current prediction; robust loss reduces the influence of some unusually large errors. Hyperparameters are fixed before test evaluation.

## Evaluation design

A reproducible 80/20 split produces 1,069 training rows and 268 test rows. Stratification preserves the smoking-status mix. Five-fold shuffled cross-validation is performed only on training rows. Lowest mean CV MAE determines the winning algorithm. Preprocessing is fitted anew inside each fold. The selected pipeline is refitted on all training rows and evaluated once on the held-out test set.

| Algorithm | Mean training CV MAE (USD) | Mean CV RMSE (USD) | Mean CV R² |
|---|---:|---:|---:|
| Mean baseline | 9,155.46 | 12,098.18 | -0.0154 |
| Ridge regression | 4,398.39 | 6,297.01 | 0.7225 |
| Random forest | 2,778.84 | 4,839.69 | 0.8328 |
| Gradient boosting | 2,097.37 | 4,842.59 | 0.8344 |

Gradient boosting is selected by MAE even though random forest has a marginally lower CV RMSE. This follows the declared selection rule rather than switching metrics after seeing results.

| Selected-model test metric | Value |
|---|---:|
| MAE | $1,352.65 |
| RMSE | $3,297.50 |
| R² | 0.9246 |

MAE is the average absolute error. RMSE puts greater weight on large errors. R² compares squared error with a mean-prediction reference. R² must not be presented as a classification accuracy percentage. Include the actual-versus-predicted and residual figures from `figures/`.

The test MAE is lower than training cross-validation MAE. That difference illustrates sampling variation in a small dataset; it is not evidence of guaranteed accuracy. The larger RMSE relative to MAE indicates some substantial errors.

## Implementation and verification

`schema.py` defines input rules. `train.py` produces data artifacts, CV comparison, metrics, and charts. `predict.py` loads the saved pipeline. `run.py` serves the local interface and prediction endpoints. `web/` supplies the interface.

Verification checks the original-data hash, cleaning result, train/test separation, training-only scaling statistics, reproduction of saved test predictions and metrics, CV-based selection, and rejection of invalid profiles. Browser demonstration covers profile prediction, scenario comparison, evaluation views, and five-row CSV prediction.

## Limitations and future work

The benchmark is small, historical, and limited to a few recorded categories and US regions. Original collection provenance is not fully established by the repository. There is no independent external validation, fairness evaluation, calibrated prediction interval, temporal validation, current pricing information, or demonstrated transfer to Indian policies. The same row features do not establish causal effects.

Future academic extensions could use a richer appropriately licensed local dataset, repeated or nested cross-validation, validated uncertainty estimates, feature-attribution analysis, and explicit subgroup error analysis. Any actual insurance use would require a separate problem definition, representative data, and validation process.

## Conclusion

The project demonstrates a complete regression workflow from source CSV to a usable prediction interface. Model comparison, traceable data cleaning, and transparent measured errors are central to the submission.

## References

1. Dataset repository: https://github.com/stedy/Machine-Learning-with-R-datasets
2. Dataset listing: https://www.kaggle.com/datasets/mirichoi0218/insurance
3. Scikit-learn: https://scikit-learn.org/stable/
4. Full source attribution: `data/SOURCE.md`.
