"""Test beam search"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures

from fastcan import FastCan, refine


def test_beam_reg():
    # Test whether beam search works correctly with a toy dataset.
    X, y = load_diabetes(return_X_y=True)
    X = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
    greedy = FastCan(n_features_to_select=10).fit(X, y)
    beam = FastCan(n_features_to_select=10, beam_width=10).fit(X, y)

    assert set(beam.indices_) != set(greedy.indices_)
    assert beam.scores_.sum() > greedy.scores_.sum()

    greedy_ids, greedy_scores = refine(greedy)
    beam_ids, beam_scores = refine(beam)
    assert set(beam_ids) != set(greedy_ids)
    assert beam_scores.sum() > greedy_scores.sum()


def test_beam_error():
    # Test whether beam search raise error when beam_width
    # or n_features_to_select is too large.
    n_samples = 50
    n_features = 20
    rng = np.random.default_rng(0)
    X_origin = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)

    X = X_origin.copy()
    # Should pass without error
    FastCan(n_features_to_select=1, beam_width=n_features).fit(X, y)
    # Should raise an error
    with pytest.raises(ValueError, match=r"beam_width should <= .*"):
        FastCan(n_features_to_select=1, indices_include=[0], beam_width=n_features).fit(
            X, y
        )

    X = X_origin.copy()
    X[:, [0, 1, 2]] = 0  # Zero feature
    with pytest.raises(ValueError, match=r"Beam Search: Not enough valid candidates.*"):
        FastCan(n_features_to_select=17, beam_width=3).fit(X, y)

    X = X_origin.copy()
    X[:, 0] = X[:, 1] = X[:, 2] = X[:, 3]  # Duplicate feature
    with pytest.raises(ValueError, match=r"Beam Search: Not enough valid candidates.*"):
        FastCan(n_features_to_select=18, beam_width=3).fit(X, y)

    X = X_origin.copy()
    X[:, range(8)] = 0  # Zero feature
    with pytest.raises(ValueError, match=r"Beam Search: Not enough valid candidates.*"):
        FastCan(n_features_to_select=10, beam_width=11).fit(X, y)


def test_n_inclusions():
    # Test when n_inclusions == n_features_to_select
    n_samples = 50
    n_features = 20
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)
    selector = FastCan(
        n_features_to_select=5,
        indices_include=[0, 1, 2, 3, 4],
        beam_width=3,
    ).fit(X, y)
    assert selector.indices_.tolist() == [0, 1, 2, 3, 4]
