"""Check data traceability, holdout isolation, saved metrics and input validation."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from predict import ROOT, predict, load_bundle
from schema import FEATURES, NUMERIC


def main():
    raw = pd.read_csv(ROOT / 'data/raw.csv')
    clean = pd.read_csv(ROOT / 'data/clean.csv')
    audit = json.loads((ROOT / 'reports/data_audit.json').read_text())
    assert hashlib.sha256((ROOT / 'data/raw.csv').read_bytes()).hexdigest() == audit['source_sha256']
    assert len(raw) == audit['raw_rows'] and len(clean) == audit['clean_rows']
    assert not clean.duplicated().any() and not clean.isna().any().any()
    expected = raw.drop_duplicates()
    pd.testing.assert_frame_equal(clean.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    print('PASS: original-data hash, exact cleaning result, no duplicate or missing cells')
    manifest = pd.read_csv(ROOT / 'data/split_manifest.csv')
    train_ids = manifest.loc[manifest.split == 'train', 'source_row_id']
    test_ids = manifest.loc[manifest.split == 'test', 'source_row_id']
    assert manifest.source_row_id.is_unique
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(expected.index)
    bundle = load_bundle()
    fitted_numeric = bundle['pipeline'].named_steps['prepare'].named_transformers_['numeric']
    np.testing.assert_allclose(fitted_numeric.named_steps['scale'].mean_, raw.loc[train_ids, NUMERIC].mean().to_numpy())
    print('PASS: train/test separation and scaler statistics fitted only on training records')
    saved = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    assert set(saved.source_row_id) == set(test_ids)
    np.testing.assert_allclose(saved.actual_charges, raw.loc[saved.source_row_id, 'charges'])
    regenerated = bundle['pipeline'].predict(raw.loc[saved.source_row_id, FEATURES])
    np.testing.assert_allclose(saved.predicted_charges, regenerated)
    metrics = bundle['metrics']
    for key, calculated in [
        ('test_mae_usd', mean_absolute_error(saved.actual_charges, regenerated)),
        ('test_rmse_usd', root_mean_squared_error(saved.actual_charges, regenerated)),
        ('test_r2', r2_score(saved.actual_charges, regenerated)),
    ]:
        assert np.isclose(metrics[key], calculated)
    comparison = pd.read_csv(ROOT / 'reports/model_comparison.csv')
    assert comparison.loc[comparison.cv_mae_usd.idxmin(), 'model'] == metrics['selected_model']
    print('PASS: saved model predictions reproduce reported test metrics and CV selection')
    example = pd.read_csv(ROOT / 'data/example_profiles.csv')
    answer = predict(example)
    assert len(answer) == 5 and np.isfinite(answer.predicted_charges_usd).all()
    for value in [example.drop(columns='age'), example.assign(charges=1), example.assign(age=65), example.assign(age=30.5), example.assign(bmi=np.nan), example.assign(bmi=np.inf), example.assign(children=-1), example.assign(smoker='unknown'), example.assign(sex=None), example.assign(region='india')]:
        try:
            predict(value)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Invalid profile accepted')
    print('PASS: batch inference and rejection of missing, extra, nonfinite, out-of-range and unknown fields')
    print('ALL CHECKS PASSED')

if __name__ == '__main__':
    main()
