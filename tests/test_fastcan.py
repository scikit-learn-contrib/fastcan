"""Test FastCan"""

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.cross_decomposition import CCA
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator

from fastcan import FastCan


def test_fastcan_is_sklearn_estimator():
    check_estimator(FastCan())


@pytest.mark.parametrize("beam_width", [1, 3])
def test_select_kbest_classif(beam_width):
    # Test whether the relative univariate feature selection
    # gets the correct items in a simple classification problem
    # with the k best heuristic
    n_samples = 200
    n_features = 20
    n_classes = 8
    n_informative = 5

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    reg = LinearRegression().fit(X[:, :n_informative], y)
    gtruth_ssc = reg.score(X[:, :n_informative], y)

    correlation_filter = FastCan(
        n_features_to_select=n_informative,
        beam_width=beam_width,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    # Test whether the ssc from the fastcan is consistent
    # with the mcc from the linear regression
    assert abs(ssc - gtruth_ssc) < 1e-5

    support = correlation_filter.get_support()
    gtruth = np.zeros(n_features)
    gtruth[:n_informative] = 1
    assert_array_equal(support, gtruth)


@pytest.mark.parametrize("beam_width", [1, 2])
def test_indices_include_exclude(beam_width):
    # Test whether fastcan can select informative features based
    # on some pre-include features and pre-exclude features
    n_samples = 20
    n_features = 20
    n_targets = 8
    n_informative = 5
    indices_params = [0, 3]

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    include_filter = FastCan(
        n_features_to_select=n_informative,
        indices_include=indices_params,
        beam_width=beam_width,
    )
    exclude_filter = FastCan(
        n_features_to_select=n_informative,
        indices_exclude=indices_params,
        beam_width=beam_width,
    )
    include_filter.fit(X, y)
    exclude_filter.fit(X, y)

    include_support = include_filter.get_support()
    exclude_support = exclude_filter.get_support()
    gtruth = np.zeros(n_features)
    gtruth[:n_informative] = 1
    assert_array_equal(include_support, gtruth)
    gtruth[indices_params] = 0
    assert_array_equal(exclude_support[:n_informative], gtruth[:n_informative])


def test_ssc_consistent_with_cca():
    # Test whether the ssc got from the fastcan is consistent
    # with the ssc got from CCA
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

    correlation_filter = FastCan(
        n_features_to_select=n_features,
    )
    correlation_filter.fit(X, y)
    ssc = correlation_filter.scores_.sum()
    assert_almost_equal(actual=ssc, desired=gtruth_ssc)


@pytest.mark.parametrize("beam_width", [1, 2])
def test_h_eta_consistency(beam_width):
    # Test whether the ssc got from h-correlation is
    # consistent with the ssc got from eta-cosine
    n_samples = 200
    n_features = 20
    n_targets = 10
    n_informative = 10
    n_to_select = 5

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )

    h_correlation = FastCan(
        n_features_to_select=n_to_select, eta=False, beam_width=beam_width
    )
    eta_cosine = FastCan(
        n_features_to_select=n_to_select, eta=True, beam_width=beam_width
    )
    h_correlation.fit(X, y)
    eta_cosine.fit(X, y)
    assert_array_almost_equal(h_correlation.scores_.sum(), eta_cosine.scores_.sum())
    assert set(h_correlation.indices_) == set(eta_cosine.indices_)


def test_raise_errors():
    # Test whether fastcan raise errors properly
    n_samples = 20
    n_features = 20
    n_classes = 10
    n_informative = 15
    n_redundant = n_features - n_informative

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_redundant=n_redundant,
        flip_y=0,
        shuffle=False,
        random_state=0,
    )

    selector_n_select = FastCan(
        n_features_to_select=n_features + 1,
    )
    selector_n_inclusions = FastCan(
        n_features_to_select=n_features, indices_include=range(n_features + 1)
    )
    selector_eta_for_small_size_samples = FastCan(
        n_features_to_select=n_features, eta=True
    )

    selector_indices_include_bounds = FastCan(
        n_features_to_select=n_features, indices_include=[-1]
    )

    selector_indices_include_ndim = FastCan(
        n_features_to_select=n_features, indices_include=[[0]]
    )

    selector_include_exclude_intersect = FastCan(
        n_features_to_select=n_features,
        indices_include=[0, 1],
        indices_exclude=[1, 2],
    )
    selector_n_candidates = FastCan(
        n_features_to_select=n_features,
        indices_exclude=[1, 2],
    )
    selector_too_many_inclusions = FastCan(
        n_features_to_select=2,
        indices_include=[1, 2, 3],
    )

    with pytest.raises(ValueError, match=r"n_features_to_select .*"):
        selector_n_select.fit(X, y)

    with pytest.raises(ValueError, match=r"The number of indices .*"):
        selector_n_inclusions.fit(X, y)

    with pytest.raises(ValueError, match=r"Out of bounds. .*"):
        selector_indices_include_bounds.fit(X, y)

    with pytest.raises(ValueError, match=r"Found indices_params with dim .*"):
        selector_indices_include_ndim.fit(X, y)

    with pytest.raises(ValueError, match=r"`eta` cannot be True, .*"):
        selector_eta_for_small_size_samples.fit(X, y)

    with pytest.raises(ValueError, match=r"`indices_include` and `indices_exclu.*"):
        selector_include_exclude_intersect.fit(X, y)

    with pytest.raises(ValueError, match=r"n_features - n_exclusions should.*"):
        selector_n_candidates.fit(X, y)

    with pytest.raises(ValueError, match=r"n_features_to_select should.*"):
        selector_too_many_inclusions.fit(X, y)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_cython_errors():
    # Test whether fastcan raise cython errors properly
    rng = np.random.default_rng(0)
    n_samples = 20
    n_informative = 3
    x_sub = rng.random((n_samples, n_informative))
    y = rng.random((n_samples))

    selector_no_cand = FastCan(
        n_features_to_select=n_informative + 1,
    )

    with pytest.raises(RuntimeError, match=r"No candidate feature can .*"):
        # No candidate
        selector_no_cand.fit(np.c_[x_sub, x_sub[:, 0] + x_sub[:, 1]], y)
