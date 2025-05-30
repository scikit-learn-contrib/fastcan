"Test ssc"

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.linear_model import LinearRegression

from fastcan.utils import mask_missing_values, ols, ssc


def test_sum_errs():
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)

    indices, scores = ols(X, y, 5)

    y_hat = (
        LinearRegression(fit_intercept=False)
        .fit(X[:, indices], y)
        .predict(X[:, indices])
    )
    e = y - y_hat
    # Sum of Error Reduction Ratio
    serrs = 1 - np.dot(e, e) / np.dot(y, y)
    assert_almost_equal(actual=scores.sum(), desired=serrs)


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
    assert_array_equal(actual=a_masked, desired=a[mask_valid])
    b[20, 1] = np.nan
    c[30] = np.nan
    a_masked, b_masked, c_mask = mask_missing_values(a, b, c)
    mask_valid = mask_missing_values(a, b, c, return_mask=True)
    assert a_masked.shape == (97, 10)
    assert b_masked.shape == (97, 2)
    assert c_mask.shape == (97,)
    assert_array_equal(actual=a_masked, desired=a[mask_valid])
    assert_array_equal(actual=b_masked, desired=b[mask_valid])
    assert_array_equal(actual=c_mask, desired=c[mask_valid])
    assert np.isfinite(a_masked).all()
    assert np.isfinite(b_masked).all()
    assert np.isfinite(c_mask).all()
