"Test ssc"

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn import config_context
from sklearn.linear_model import LinearRegression

from fastcan.utils import mask_missing_values, ols, ssc


@pytest.mark.parametrize("array_type", ["numpy", "pytorch"])
def test_sum_errs(array_type, monkeypatch):
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)

    if array_type == "pytorch":
        monkeypatch.setenv("SCIPY_ARRAY_API", "1")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        torch.set_default_device(device)
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)

        with config_context(array_api_dispatch=True):
            indices, scores = ols(X_torch, y_torch, 5)

        indices = indices.cpu().numpy()
        scores_sum = float(scores.sum().cpu())
    else:
        indices, scores = ols(X, y, 5)
        scores_sum = scores.sum()

    y_hat = (
        LinearRegression(fit_intercept=False)
        .fit(X[:, indices], y)
        .predict(X[:, indices])
    )
    e = y - y_hat
    # Sum of Error Reduction Ratio
    serrs = 1 - np.dot(e, e) / np.dot(y, y)
    assert_almost_equal(actual=scores_sum, desired=serrs, decimal=6)


def test_pearson_r():
    """Test Pearson's correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random(100)
    y = rng.random(100)
    r2 = ssc(X.reshape(-1, 1), y.reshape(-1, 1))
    gtruth_r2 = np.corrcoef(X, y)[0, 1] ** 2
    assert_almost_equal(actual=r2, desired=gtruth_r2)


def test_multi_r():
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)
    r2 = ssc(X, y.reshape(-1, 1))
    gtruth_r2 = LinearRegression().fit(X, y).score(X, y)
    assert_almost_equal(actual=r2, desired=gtruth_r2)

    X = rng.random(100)
    y = rng.random((100, 10))
    r2 = ssc(X.reshape(-1, 1), y)
    gtruth_r2 = LinearRegression().fit(y, X).score(y, X)
    assert_almost_equal(actual=r2, desired=gtruth_r2)


def test_mask_missing():
    """Test mask missing values."""
    assert mask_missing_values() is None

    # Check that invalid arguments yield ValueError
    with pytest.raises(ValueError):
        mask_missing_values([0], [0, 1])

    rng = np.random.default_rng(12345)
    a = rng.random((100, 10))
    b = rng.random((100, 2))
    c = rng.random(100)
    a[10, 0] = np.nan
    a_masked = mask_missing_values(a)
    mask_valid = mask_missing_values(a, return_mask=True)
    assert a_masked.shape == (99, 10)
    assert_array_equal(a_masked, a[mask_valid])
    b[20, 1] = np.nan
    c[30] = np.nan
    a_masked, b_masked, c_mask = mask_missing_values(a, b, c)
    mask_valid = mask_missing_values(a, b, c, return_mask=True)
    assert a_masked.shape == (97, 10)
    assert b_masked.shape == (97, 2)
    assert c_mask.shape == (97,)
    assert_array_equal(a_masked, a[mask_valid])
    assert_array_equal(b_masked, b[mask_valid])
    assert_array_equal(c_mask, c[mask_valid])
    assert np.isfinite(a_masked).all()
    assert np.isfinite(b_masked).all()
    assert np.isfinite(c_mask).all()
