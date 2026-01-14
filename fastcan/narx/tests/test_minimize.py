"""Test NARX with minimize solver"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from fastcan.narx import make_narx
from fastcan.utils import mask_missing_values


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize(
    "method", ["default", None, "L-BFGS-B", "dogleg", "trust-ncg", "CG", "Nelder-Mead"]
)
def test_minimize(nan, multi_output, method):
    """Test NARX with minimize solver on synthetic data."""

    def make_data(multi_output, nan, rng):
        if multi_output:
            n_samples = 1000
            max_delay = 3
            e0 = rng.normal(0, 0.1, n_samples)
            e1 = rng.normal(0, 0.02, n_samples)
            u0 = rng.uniform(0, 1, n_samples + max_delay)
            u1 = rng.normal(0, 0.1, n_samples + max_delay)
            y0 = np.zeros(n_samples + max_delay)
            y1 = np.zeros(n_samples + max_delay)
            for i in range(max_delay, n_samples + max_delay):
                y0[i] = (
                    0.5 * y0[i - 1]
                    + 0.8 * y1[i - 1]
                    + 0.3 * u0[i] ** 2
                    + 2 * u0[i - 1] * u0[i - 3]
                    + 1.5 * u0[i - 2] * u1[i - 3]
                    + 1
                )
                y1[i] = (
                    0.6 * y1[i - 1]
                    - 0.2 * y0[i - 1] * y1[i - 2]
                    + 0.3 * u1[i] ** 2
                    + 1.5 * u1[i - 2] * u0[i - 3]
                    + 0.5
                )
            y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
            X = np.c_[u0[max_delay:], u1[max_delay:]]
            n_outputs = 2
        else:
            rng = np.random.default_rng(12345)
            n_samples = 1000
            max_delay = 3
            e = rng.normal(0, 0.1, n_samples)
            u0 = rng.uniform(0, 1, n_samples + max_delay)
            u1 = rng.normal(0, 0.1, n_samples)
            y = np.zeros(n_samples + max_delay)
            for i in range(max_delay, n_samples + max_delay):
                y[i] = (
                    0.5 * y[i - 1]
                    + 0.3 * u0[i] ** 2
                    + 2 * u0[i - 1] * u0[i - 3]
                    + 1.5 * u0[i - 2] * u1[i - max_delay]
                    + 1
                )
            y = y[max_delay:] + e
            X = np.c_[u0[max_delay:], u1]
            n_outputs = 1

        if nan:
            X_nan_ids = rng.choice(n_samples, 20, replace=False)
            y_nan_ids = rng.choice(n_samples, 10, replace=False)
            X[X_nan_ids] = np.nan
            y[y_nan_ids] = np.nan

        X = np.asfortranarray(X)
        y = np.asfortranarray(y)
        return X, y, n_outputs

    rng = np.random.default_rng(12345)
    X, y, _ = make_data(multi_output, nan, rng)

    fit_kwargs = {"solver": "minimize"}
    if method != "default":
        fit_kwargs["method"] = method

    narx_score = make_narx(
        X,
        y,
        n_terms_to_select=[5, 4] if multi_output else 4,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(X, y, coef_init="one_step_ahead", **fit_kwargs)

    assert r2_score(*mask_missing_values(y, narx_score.predict(X, y_init=y))) > 0.97
