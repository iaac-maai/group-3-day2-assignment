"""Recompute the published regression evaluation from trusted, hash-pinned files."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

FEATURES = ['Area_num', 'BHK_num', 'Bathrooms_num', 'Balconies_num',
            'Furnished_clean', 'Facing', 'Location', 'Property Type']
HASHES = {
    'model': 'b4193b09cc1316008936387fd28f65b07960447feab5d1eaa279ed8745791262',
    'data': '354d28e07432137cdc97a2197a556269137e74135af01186420e2e2917a19ee9',
}

def metrics(y, predictions):
    return {'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': root_mean_squared_error(y, predictions)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=Path('model.joblib'))
    parser.add_argument('--output', type=Path, default=Path('evaluation-rerun.json'))
    args = parser.parse_args()
    for key in HASHES:
        if hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() != HASHES[key]:
            raise ValueError(f'{key} checksum mismatch')
    data = pd.read_csv(args.data).dropna(subset=['Price_num'])
    Xtr, Xte, ytr, yte = train_test_split(
        data[FEATURES], data.Price_num, test_size=.2, random_state=42)
    model = joblib.load(args.model)  # Trusted pickle only; checksum is not a safety proof.
    report = {'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
              'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'joblib': joblib.__version__}, 'artifact_test': metrics(yte, model.predict(Xte)),
              'rows': len(data), 'train': len(Xtr), 'test': len(Xte), 'cv': {}}
    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    for name, estimator in [('selected_random_forest', clone(model)),
                            ('baseline_mean', DummyRegressor(strategy='mean')),
                            ('baseline_median', DummyRegressor(strategy='median'))]:
        fitted = clone(estimator).fit(Xtr, ytr)
        scores = cross_validate(estimator, Xtr, ytr,
            cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scoring, n_jobs=1)
        result = {'test': metrics(yte, fitted.predict(Xte)), 'folds': {}, 'mean': {}, 'std': {}}
        for metric in scoring:
            values = scores['test_' + metric] * (1 if metric == 'r2' else -1)
            result['folds'][metric] = values.tolist()
            result['mean'][metric] = float(values.mean())
            result['std'][metric] = float(values.std(ddof=0))
        report['cv'][name] = result
        if name == 'selected_random_forest':
            report['refit_matches_export'] = bool(np.allclose(
                fitted.predict(Xte), model.predict(Xte), rtol=0, atol=1e-8))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['artifact_test'], indent=2))
    print(f'Evaluation saved to {args.output}')

if __name__ == '__main__':
    main()
