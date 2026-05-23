"""Test LazyFastCan."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.cross_decomposition import CCA
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.estimator_checks import check_estimator

from fastcan import LazyFastCan
from fastcan.narx import (
    gen_poly_features,
    gen_time_shift_features,
    make_poly_ids,
    make_time_shift_features,
    make_time_shift_ids,
)


def test_lazyfastcan_is_sklearn_estimator():
    check_estimator(LazyFastCan())


def test_lazy_poly_selection():
    """
    Test if LazyFastCan correctly selects informative polynomial features,
    and if the ssc score is aligned with the ground truth.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 5
    n_informative = 3
    X = rng.normal(size=(n_samples, n_features))
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X[:, :n_informative])
    n_polys = X_poly.shape[1]
    w = rng.normal(size=n_polys)
    e = rng.normal(scale=0.01, size=n_samples)
    y = X_poly @ w + e

    reg = LinearRegression().fit(X_poly, y)
    gtruth_ssc = reg.score(X_poly, y)

    poly_ids = make_poly_ids(n_features, degree=2)
    feat_gen = partial(gen_poly_features, ids=poly_ids)

    poly_filter = LazyFastCan(
        n_features_to_select=n_polys,
        feature_generator=feat_gen,
    )
    poly_filter.fit(X, y)
    ssc = poly_filter.scores_.sum()
    assert abs(ssc - gtruth_ssc) < 1e-5


def test_lazy_time_shift_selection():
    """
    Test if LazyFastCan correctly selects informative time-shifted features,
    and if the ssc score is aligned with the ground truth.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    max_delay = 10
    n_features = 5
    n_informative = 3
    X = rng.normal(size=(n_samples + max_delay, n_features))
    time_shift_ids = make_time_shift_ids(n_informative, max_delay=max_delay)
    X_shifted = make_time_shift_features(
        X[:, :n_informative], ids=time_shift_ids, mode="edge"
    )
    n_time_shifts = X_shifted.shape[1]
    w = rng.normal(size=n_time_shifts)
    e = rng.normal(scale=0.01, size=n_samples + max_delay)
    y = X_shifted @ w + e

    reg = LinearRegression().fit(X_shifted[max_delay:], y[max_delay:])
    gtruth_ssc = reg.score(X_shifted[max_delay:], y[max_delay:])

    feat_gen = partial(gen_time_shift_features, ids=time_shift_ids)

    time_shift_filter = LazyFastCan(
        n_features_to_select=n_time_shifts,
        feature_generator=feat_gen,
        n_init=max_delay,
    )
    time_shift_filter.fit(X, y)
    ssc = time_shift_filter.scores_.sum()
    assert abs(ssc - gtruth_ssc) < 1e-5


def test_lazy_ssc_consistent_with_cca():
    """
    Test if the ssc score from LazyFastCan is consistent with
    the ssc score from CCA.
    """
    n_samples = 200
    n_features = 20
    n_targets = 10
    n_informative = 10

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    cca = CCA(n_components=n_targets)
    cca.fit(X, y)
    X_c, Y_c = cca.transform(X, y)
    corrcoef = np.corrcoef(X_c, Y_c, rowvar=False).diagonal(offset=n_targets)
    gtruth_ssc = sum(corrcoef**2)

    correlation_filter = LazyFastCan(
        n_features_to_select=n_features,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    assert_almost_equal(actual=ssc, desired=gtruth_ssc)


def test_lazy_const_feats():
    """
    Test if LazyFastCan can skip constant features without errors.
    """
    n_samples = 100
    n_features = 10
    n_informative = 5

    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    X[:, n_informative:] = 1.0  # Make some features constant
    w = rng.normal(size=n_informative)
    e = rng.normal(scale=0.01, size=n_samples)
    y = X[:, :n_informative] @ w + e

    const_filter = LazyFastCan(
        n_features_to_select=n_informative,
    )
    const_filter.fit(X, y)
    selected_ids = const_filter.get_support(indices=True)
    assert selected_ids.tolist() == list(range(n_informative))


def test_lazy_redundant_feats():
    """
    Test if LazyFastCan can skip redundant features without errors.
    """
    n_samples = 100
    n_features = 10
    n_informative = 5

    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    X[:, n_informative:] = X[:, :n_informative]  # Make some features redundant
    w = rng.normal(size=n_informative)
    e = rng.normal(scale=0.01, size=n_samples)
    y = X[:, :n_informative] @ w + e

    redundant_filter = LazyFastCan(
        n_features_to_select=n_informative,
    )
    redundant_filter.fit(X, y)
    selected_ids = redundant_filter.get_support(indices=True)
    assert selected_ids.tolist() == list(range(n_informative))


def test_lazy_errors():
    """
    Test if LazyFastCan raises expected errors.
    """

    def _gen_float_idx(X, skip_indices):
        n_features = X.shape[1]
        for j in range(n_features):
            if j in skip_indices:
                continue
            yield float(j), X[:, j]

    def _gen_neg_idx(X, skip_indices):
        n_features = X.shape[1]
        for j in range(n_features):
            if j in skip_indices:
                continue
            yield -1, X[:, j]

    def _gen_nd_feats(X, skip_indices):
        n_features = X.shape[1]
        for j in range(n_features):
            if j in skip_indices:
                continue
            yield j, X[:, [j]]

    def _gen_short_feats(X, skip_indices):
        n_features = X.shape[1]
        for j in range(n_features):
            if j in skip_indices:
                continue
            yield j, X[10:, j]

    def _gen_nan_feats(X, skip_indices):
        n_features = X.shape[1]
        for j in range(n_features):
            if j in skip_indices:
                continue
            yield j, np.r_[X[1:, j], np.nan]

    n_samples = 100
    n_features = 10
    n_informative = 5

    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    X[:, n_informative:] = X[:, :n_informative]
    w = rng.normal(size=n_informative)
    y = X[:, :n_informative] @ w

    filter_float_idx = LazyFastCan(
        feature_generator=_gen_float_idx,
    )
    with pytest.raises(TypeError, match="is not an integer"):
        filter_float_idx.fit(X, y)

    filter_neg_idx = LazyFastCan(
        feature_generator=_gen_neg_idx,
    )
    with pytest.raises(ValueError, match="is negative"):
        filter_neg_idx.fit(X, y)

    filter_nd_feats = LazyFastCan(
        feature_generator=_gen_nd_feats,
    )
    with pytest.raises(ValueError, match="is not one-dimensional"):
        filter_nd_feats.fit(X, y)

    filter_short_feats = LazyFastCan(
        feature_generator=_gen_short_feats,
    )
    with pytest.raises(ValueError, match="expected length is"):
        filter_short_feats.fit(X, y)

    filter_nan_feats = LazyFastCan(
        feature_generator=_gen_nan_feats,
    )
    with pytest.raises(ValueError, match="contains non-finite values"):
        filter_nan_feats.fit(X, y)

    filter_noidx = LazyFastCan(n_features_to_select=n_informative + 1)
    with pytest.raises(RuntimeError, match="No more features can be selected"):
        filter_noidx.fit(X, y)

    X[:, n_informative:] = 1.0
    filter_noscore = LazyFastCan(n_features_to_select=n_informative + 1)
    with pytest.raises(RuntimeError, match="No improvement can be achieved"):
        filter_noscore.fit(X, y)
