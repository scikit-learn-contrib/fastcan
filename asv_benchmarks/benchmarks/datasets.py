from pathlib import Path

import numpy as np
from joblib import Memory
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# memory location for caching datasets
M = Memory(location=str(Path(__file__).resolve().parent / "cache"))


@M.cache
def _digits_dataset(n_samples=None, dtype=np.float32):
    X, y = load_digits(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    X = X[:n_samples]
    y = y[:n_samples]

    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _synth_regression_dataset(n_samples=10000, n_features=200, dtype=np.float32):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 10,
        noise=50,
        random_state=0,
    )
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)

    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _co2_dataset(dtype=np.float32):
    X, y = fetch_openml(data_id=41187, return_X_y=True, as_frame=False)
    X = X[:, [1, 3]]
    X = X.astype(dtype, copy=False)
    n_samples = len(y)
    n_test = int(n_samples * 0.1)

    mask_train = np.arange(n_samples) < (n_samples - n_test)
    X, X_val = X[mask_train], X[~mask_train]
    y_train, y_val = y[mask_train], y[~mask_train]
    return X, X_val, y_train, y_val
