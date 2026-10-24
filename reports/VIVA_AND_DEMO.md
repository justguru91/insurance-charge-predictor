# Viva and evaluation demo

## A 60-second introduction

“My mini project predicts a numerical medical insurance charge from age, sex, BMI, dependents, smoking status, and US region. I use a public benchmark with 1,338 rows. I preserve the original data and remove one exact duplicate. I compare a mean baseline, ridge regression, random forest, and gradient boosting through five-fold cross-validation on training data only. Gradient boosting wins by MAE. I evaluate it on 268 held-out rows and provide a browser interface for form inputs, profile comparison, and CSV batch prediction. This is an academic regression demo, not a live insurance premium quote.”

## A five-minute demo

1. Open http://127.0.0.1:8502 and describe the six input fields. Explain that the output is USD medical charges.
2. Choose **Family profile**, then **Predict charges**. Expected result with the included model: **$8,934.37**. Add it to comparison.
3. Choose **Smoker profile**, predict, and add it to comparison. Expected result: **$40,856.24**. Only smoking status differs between these two presets. State that the model learns an association; this is not a causal experiment.
4. Open **Model evaluation**. Explain selection by CV MAE, then the held-out MAE, RMSE, and R². Show the scatter and residual plots.
5. Open **Data & workflow**. Point out 1,338 original rows, 1,337 clean rows, one duplicate, and no missing values.
6. Open **Batch prediction**, select **Use included example**, run the predictions, and download the output.
7. Finish by naming one limitation: historical US benchmark data are not current Indian policy prices.

## Common questions

**Why regression?** The target is a continuous numeric amount. Classification predicts a discrete category.

**Why a baseline?** It shows whether the learned models improve over always predicting the mean charge.

**Why ridge instead of only ordinary linear regression?** Ridge is a regularized linear baseline. Its penalty discourages unnecessarily large coefficients.

**What is one-hot encoding?** It creates binary indicator columns for categories without introducing an artificial numeric order.

**Why five-fold cross-validation?** Each training record is used for validation once across five rounds. It compares models without spending the held-out test set on selection.

**What is leakage?** Evaluation information influences training or model selection. Learning scalers/encoders inside each fold prevents preprocessing leakage.

**Which algorithm wins?** Gradient boosting has the lowest mean CV MAE, about $2,097.37, in this experiment.

**Why not choose by the test score?** The test set should estimate final behavior, not guide algorithm selection.

**Why is test MAE smaller than CV MAE?** Sampling variation: this held-out subset is easier for the model. This is why one split cannot guarantee future performance.

**Is 0.9246 R² an accuracy percentage?** No. It is a regression score based on squared error relative to a mean-value reference.

**Why is RMSE much larger than MAE?** A few large errors affect RMSE more strongly.

**Why retain outliers?** High charges can be valid observations, so deleting them merely because they are high could distort the target.

**Can it predict in rupees?** The model was not trained on Indian prices. Currency conversion does not solve that domain mismatch.

**Does smoking cause the amount shown?** A model comparison is associative and cannot establish causality.

**What makes this a complete project?** Audited data, consistent preprocessing, baseline/model comparison, separate evaluation, saved-model inference, interface, validation, and documentation.

## Submission checklist

- Fill your name, roll number, college, department, and guide in the report.
- Include the dataset source and actual measured results.
- Show screenshots of a prediction, model evaluation, and data audit.
- Be able to explain each of the six features and the difference between MAE/RMSE/R².
- Carry a copy of the whole folder and verify the app runs before evaluation.
