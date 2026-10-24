"""Shared prediction module for the form, batch API and command line."""
from functools import lru_cache
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
try:
    from .schema import FEATURES, validate_features
except ImportError:
    from schema import FEATURES, validate_features
ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def load_bundle():
    path = ROOT / 'models/charge_model.joblib'
    if not path.exists():
        raise FileNotFoundError('Run python src/train.py first.')
    return joblib.load(path)


def predict(frame):
    data = validate_features(frame)
    bundle = load_bundle()
    values = bundle['pipeline'].predict(data)
    if not np.isfinite(values).all():
        raise ValueError('The model returned a nonfinite prediction.')
    # The returned value is the unmodified regression output.
    result = data.copy()
    result['predicted_charges_usd'] = np.round(values, 2)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    print(predict(pd.read_csv(args.csv)).to_csv(index=False))
