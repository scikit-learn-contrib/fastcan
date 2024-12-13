"Test ssc"

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression

from fastcan.utils import ols, ssc


def test_sum_errs():
    """Test multiple correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 10))
    y = rng.random(100)

    indices, scores = ols(X, y, 5)

    y_hat = LinearRegression(fit_intercept=False)\
                .fit(X[:, indices], y)\
                .predict(X[:, indices])
    e = y-y_hat
    # Sum of Error Reduction Ratio
    serrs = 1 - np.dot(e, e)/np.dot(y, y)
    assert_almost_equal(actual=scores.sum(), desired=serrs)

def test_pearson_r():
    """Test Pearson's correlation."""
    rng = np.random.default_rng(12345)
    X = rng.random(100)
    y = rng.random(100)
    r2 = ssc(X.reshape(-1, 1), y.reshape(-1, 1))
    gtruth_r2 = np.corrcoef(X, y)[0, 1]**2
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
