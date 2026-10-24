# Explain this project from scratch

## What is the project?

It predicts a numerical **medical insurance charge** from six profile features. It is supervised learning because examples have known answers. It is **regression** because the answer is an amount of money, not a label such as yes/no.

The dataset target is `charges`. It is not an actual current policy premium. This benchmark uses US regions and USD; changing a currency symbol to rupees would not turn it into an Indian insurance model.

## Why choose it for evaluation?

The inputs are understandable, the output is easy to demonstrate, and the dataset is small enough for a normal laptop. You can show data cleaning, categorical encoding, a baseline, three model candidates, held-out evaluation, and a working application. Your contribution is building and evaluating this complete application, not inventing a new algorithm or collecting the dataset.

## What are the six inputs?

| Input | Meaning | Accepted values |
|---|---|---|
| age | Age in years | Whole number, 18–64 |
| sex | Dataset's recorded sex category | female / male |
| bmi | Body mass index value | 15.96–53.13 |
| children | Covered children / dependents | Whole number, 0–5 |
| smoker | Recorded smoking status | yes / no |
| region | Recorded US area | northeast / northwest / southeast / southwest |

The seventh training column, charges, is the answer. Do not include it in the prediction input, because the model must estimate it.

## What does clean data mean?

The downloaded file contains 1,338 rows, one exact duplicate, and no missing values. The code normalizes categorical strings, checks numeric values, removes the duplicate, and records what happened. The clean data has 1,337 rows. High costs are not removed just because they are large. Raw data is preserved so the cleaning can be reviewed.

## How does the model learn?

1. Split the clean data into 1,069 training rows and 268 test rows.
2. Keep the test set aside.
3. Convert categories into numeric indicator columns with one-hot encoding. For example, region becomes several 0/1 columns instead of arbitrary numbers that imply an order.
4. Scale numeric inputs using statistics learned from training rows. Scaling is useful for ridge regression and is retained in the shared preprocessing pipeline.
5. Compare the mean baseline, ridge regression, random forest, and gradient boosting using five-fold cross-validation inside the training set.
6. Select the model with lowest mean CV MAE.
7. Fit that model on all 1,069 training rows.
8. Evaluate it on the 268 held-out test rows.
9. Save the preprocessing steps and fitted model together in a joblib file.

The fixed random seed makes the split repeatable. Pipelines prevent validation/test information from influencing preprocessing. This is how we avoid data leakage.

## What is gradient boosting?

Think of the prediction as a starting guess plus a series of small corrections learned by shallow decision trees. New trees improve the combined prediction step by step. This project uses 250 trees of maximum depth two with learning rate 0.04 and Huber loss.

It is selected because it has the lowest average cross-validation MAE in this experiment. A different dataset or evaluation split could produce a different winner.

## What do the results mean?

- **Training CV MAE: $2,097.37.** The average validation-fold absolute error used in model selection.
- **Test MAE: $1,352.65.** The average absolute error on this one held-out test set.
- **Test RMSE: $3,297.50.** A metric that penalizes large mistakes more strongly.
- **Test R²: 0.9246.** The selected model reduces squared test error relative to a mean-value reference. It is not “92.46% classification accuracy.”

For example, if a prediction is $8,000 and the actual recorded charge is $10,000, the absolute error for that row is $2,000. MAE averages such errors across all test rows.

MAE does not guarantee that every prediction is within $1,352.65. It also does not create a prediction interval. Test error can change considerably with the split, as the difference between CV and test results shows.

## How does the website get a prediction?

The browser collects the six values and sends a JSON request to `/api/predict` on your local machine. Python validates the fields, loads the fitted pipeline, applies the learned transformations, and calls `predict()`. The server returns the charge estimate. JavaScript displays it and can create a downloadable CSV.

Batch mode sends CSV text to `/api/batch`. The same validation and model are used. No names or account information are needed, and the server does not store submitted profiles. A browser comparison is held only in page memory.

## Technologies

| Technology | Job |
|---|---|
| Python | Training, validation, prediction, local server |
| pandas | CSV reading and table operations |
| NumPy | Numeric operations and finite-value checks |
| scikit-learn | Preprocessing, regressors, CV, and metrics |
| Matplotlib | Evaluation charts |
| Joblib | Saving/loading the fitted pipeline |
| HTML / CSS | Interface structure and visual styling |
| JavaScript | Form submission, tabs, comparison, CSV download |
| Python HTTP server | Local browser-to-model connection |
| Jupyter notebook (optional) | Guided exploration for evaluation |

## Read the code in this order

1. `data/clean.csv` — identify the six inputs and target.
2. `src/schema.py` — see input types, categories, and boundaries.
3. `src/train.py` — follow cleaning, split, CV, selection, final evaluation, and persistence.
4. `reports/model_comparison.csv` and `reports/metrics.json` — check actual results.
5. `src/predict.py` — see inference using the saved pipeline.
6. `run.py` — see local routes and request validation.
7. `web/index.html`, `style.css`, `app.js` — see the browser interface.
8. `src/verify.py` — see reproducibility and validation checks.

## What can I honestly claim?

“I built an end-to-end regression application with data auditing, fair model comparison, reproducible test results, and a local interface.”

Do not claim it is a new algorithm, a clinically validated application, a live insurance quotation system, or a reliable predictor of current Indian premiums.
