"""Shared feature schema and validation for training and inference."""
import numpy as np
import pandas as pd

FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUMERIC = ['age', 'bmi', 'children']
CATEGORICAL = ['sex', 'smoker', 'region']
CATEGORIES = {'sex': ['female', 'male'], 'smoker': ['no', 'yes'], 'region': ['northeast', 'northwest', 'southeast', 'southwest']}
# Restrict the demo to the observed support of this benchmark.
RANGES = {'age': (18, 64), 'bmi': (15.96, 53.13), 'children': (0, 5)}


def validate_features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Provide at least one profile.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing, extra = set(FEATURES) - set(frame.columns), set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(f'Expected exactly {FEATURES}. Missing: {sorted(missing)}; extra: {sorted(extra)}.')
    out = frame[FEATURES].copy()
    for column in NUMERIC:
        if out[column].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f'{column} must be a number, not a boolean.')
        out[column] = pd.to_numeric(out[column], errors='raise')
        low, high = RANGES[column]
        if not np.isfinite(out[column]).all() or not out[column].between(low, high).all():
            raise ValueError(f'{column} must be a finite number between {low} and {high}, the dataset range.')
    for column in ['age', 'children']:
        if not (out[column] == np.floor(out[column])).all():
            raise ValueError(f'{column} must be a whole number.')
        out[column] = out[column].astype(int)
    for column, choices in CATEGORIES.items():
        out[column] = out[column].astype('string').str.strip().str.lower()
        if out[column].isna().any() or not out[column].isin(choices).all():
            raise ValueError(f'{column} must be one of {choices}.')
    return out
