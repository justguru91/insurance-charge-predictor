"""Train, compare and evaluate regression models using local benchmark data."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from schema import FEATURES, NUMERIC, CATEGORICAL, validate_features

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    for folder in ['data', 'models', 'reports/figures']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / 'data/raw.csv'
    if not raw_path.exists():
        raise FileNotFoundError('Place the attributed insurance.csv dataset in data/raw.csv. See data/SOURCE.md.')
    raw = pd.read_csv(raw_path)
    if set(raw.columns) != set(FEATURES + ['charges']):
        raise ValueError('Unexpected dataset columns.')
    normalized = raw.copy()
    for col in CATEGORICAL:
        normalized[col] = normalized[col].astype('string').str.strip().str.lower()
    for col in NUMERIC + ['charges']:
        normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    missing_rows = normalized.isna().any(axis=1)
    complete = normalized.loc[~missing_rows].copy()
    duplicated = complete.duplicated(keep='first')
    clean = complete.loc[~duplicated].copy()
    clean[FEATURES] = validate_features(clean[FEATURES])
    if not (clean.charges >= 0).all():
        raise ValueError('Negative target charges found.')
    clean.to_csv(ROOT / 'data/clean.csv', index=False)
    rejected = normalized.loc[missing_rows].copy()
    rejected['reason'] = 'missing or nonfinite field'
    duplicate_rows = complete.loc[duplicated].copy()
    duplicate_rows['reason'] = 'exact duplicate'
    pd.concat([rejected, duplicate_rows]).rename_axis('source_row_id').to_csv(ROOT / 'data/removed_rows.csv')
    X, y = clean[FEATURES], clean.charges
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=X.smoker)
    train_ids = set(X_train.index)
    pd.DataFrame({'source_row_id': clean.index, 'split': ['train' if i in train_ids else 'test' for i in clean.index]}).to_csv(ROOT / 'data/split_manifest.csv', index=False)
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='error', sparse_output=False))]), CATEGORICAL)
    ])
    regressors = {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'Ridge regression': Ridge(alpha=1.0),
        'Random forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1),
        'Gradient boosting': GradientBoostingRegressor(n_estimators=250, learning_rate=.04, max_depth=2, loss='huber', random_state=SEED),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    candidates, rows = {}, []
    for name, regressor in regressors.items():
        pipeline = Pipeline([('prepare', clone(preprocess)), ('model', regressor)])
        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
        rows.append({'model': name, 'cv_mae_usd': float(-scores['test_mae'].mean()), 'cv_mae_std': float(scores['test_mae'].std()), 'cv_rmse_usd': float(-scores['test_rmse'].mean()), 'cv_r2': float(scores['test_r2'].mean())})
        candidates[name] = pipeline
    comparison = pd.DataFrame(rows).sort_values('cv_mae_usd').reset_index(drop=True)
    comparison.to_csv(ROOT / 'reports/model_comparison.csv', index=False)
    name = str(comparison.loc[0, 'model'])
    model = candidates[name].fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {'selected_model': name, 'selection_rule': 'Lowest mean 5-fold training CV MAE', 'test_mae_usd': mean_absolute_error(y_test, predictions), 'test_rmse_usd': root_mean_squared_error(y_test, predictions), 'test_r2': r2_score(y_test, predictions), 'train_rows': len(X_train), 'test_rows': len(X_test), 'seed': SEED, 'currency': 'USD'}
    save_json(ROOT / 'reports/metrics.json', metrics)
    pd.DataFrame({'source_row_id': X_test.index, 'actual_charges': y_test, 'predicted_charges': predictions, 'absolute_error': abs(y_test - predictions)}).to_csv(ROOT / 'reports/test_predictions.csv', index=False)
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib', 'scipy']}
    save_json(ROOT / 'reports/tested_versions.json', {'python': platform.python_version(), **versions})
    audit = {'raw_rows': len(raw), 'clean_rows': len(clean), 'duplicates_removed': int(duplicated.sum()), 'incomplete_rows_removed': int(missing_rows.sum()), 'raw_missing_cells': int(raw.isna().sum().sum()), 'clean_missing_cells': int(clean.isna().sum().sum()), 'features': FEATURES, 'target': 'charges', 'source_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(), 'numeric_ranges': {c: {'min': float(clean[c].min()), 'max': float(clean[c].max())} for c in NUMERIC}, 'outlier_policy': 'Retain finite nonnegative charges. High costs may be meaningful; no arbitrary trimming.'}
    save_json(ROOT / 'reports/data_audit.json', audit)
    joblib.dump({'pipeline': model, 'features': FEATURES, 'metrics': metrics, 'versions': versions}, ROOT / 'models/charge_model.joblib')
    example = X_test.iloc[:5]
    example.to_csv(ROOT / 'data/example_profiles.csv', index=False)
    summary = {'raw_rows': len(raw), 'clean_rows': len(clean), 'duplicates_removed': int(duplicated.sum()), 'missing_cells': int(clean.isna().sum().sum()), 'features': FEATURES, 'comparison': comparison.to_dict(orient='records'), 'metrics': metrics, 'target_min_usd': float(y.min()), 'target_max_usd': float(y.max()), 'preview': clean.head(8).to_dict(orient='records')}
    save_json(ROOT / 'reports/dashboard.json', summary)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#334155', 'text.color': '#0f172a', 'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_test, predictions, s=18, alpha=.6, c='#2563eb', edgecolors='none')
    limits = [min(y_test.min(), predictions.min())*.9, max(y_test.max(), predictions.max())*1.05]
    ax.plot(limits, limits, '--', c='#f59e0b', lw=1.5, label='Perfect prediction')
    ax.set(xlabel='Actual charges (USD)', ylabel='Predicted charges (USD)', title='Held-out test: actual vs predicted')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(ROOT / 'reports/figures/actual_vs_predicted.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(predictions, y_test-predictions, s=18, alpha=.6, c='#0d9488', edgecolors='none')
    ax.axhline(0, c='#64748b', ls='--'); ax.set(xlabel='Predicted charges (USD)', ylabel='Actual − predicted (USD)', title='Held-out prediction residuals')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/figures/residuals.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sorted_comp = comparison.sort_values('cv_mae_usd', ascending=False)
    ax.barh(sorted_comp.model, sorted_comp.cv_mae_usd, color=['#cbd5e1', '#94a3b8', '#60a5fa', '#2563eb'])
    ax.set(xlabel='Mean cross-validation MAE (USD) — lower is better', title='Model comparison using training folds')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/figures/model_comparison.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(y, bins=30, color='#2563eb', alpha=.85)
    ax.set(xlabel='Charges (USD)', ylabel='Records', title='Benchmark charge distribution')
    fig.tight_layout(); fig.savefig(ROOT / 'reports/figures/charge_distribution.png', dpi=170); plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
