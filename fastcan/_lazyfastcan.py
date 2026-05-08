"""
Lazy FastCan selector.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral

import numpy as np
from scipy.linalg import orth
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ._fastcan import _check_X_y


def _classical_gram_schmidt(x, W):
    """Classical Gram-Schmidt orthogonalization.
    Parameters
    ----------
    x: (n_samples,) Features for orthogonalization.
    W: (n_samples, n_selected) Selected orthonormal features,
        where n_selected can be zero.

    Returns
    -------
    q: (n_samples,) Orthonormal features.
    """
    q = x - W @ (W.T @ x)
    norm = np.linalg.norm(q)
    if norm == 0:
        norm = 1.0
    return q / norm


def _default_feature_generator(X, skip_indices):
    """Default feature generator that yields each column of X as a feature."""
    n_features = X.shape[1]
    for j in range(n_features):
        if j in skip_indices:
            continue
        yield j, X[:, j]


def _check_generated_features(idx, feature, n_samples):
    """Check the validity of generated features."""
    if not isinstance(idx, Integral):
        raise TypeError(f"Generated feature index {idx} is not an integer.")
    if idx < 0:
        raise ValueError(f"Generated feature index {idx} is negative.")
    if feature.ndim != 1:
        raise ValueError(f"Generated feature with index {idx} is not one-dimensional.")
    if feature.shape[0] != n_samples:
        raise ValueError(
            f"Generated feature with index {idx} has length {feature.shape[0]}, "
            f"but expected length is {n_samples}."
        )
    if not np.isfinite(feature).all():
        raise ValueError(
            f"Generated feature with index {idx} contains non-finite values."
        )


class LazyFastCan(SelectorMixin, BaseEstimator):
    """Lazy version of FastCan selector.

    .. versionadded:: 0.5.1

    Parameters
    ----------
    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    feature_generator : callable or None, default=None
        A generator that takes the input data `X` and an array of `skip_indices`
        and yields `(index, feature)` tuples,
        where `index` should be from 0 to n_features-1.
        The `skip_indices` indicates which features should be skipped.
        If the generator requires additional parameters,
        you can use `functools.partial` to wrap the generator with those parameters.

    Examples
    --------
    >>> from functools import partial
    >>> from fastcan import LazyFastCan
    >>> from fastcan.narx import gen_time_shift_features, make_time_shift_ids
    >>> fg = partial(gen_time_shift_features, ids=make_time_shift_ids(2, 2))
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [[0, 0], [0, 1], [1, 1], [0, 2]]
    >>> LazyFastCan(2, feature_generator=fg).fit(X, y).get_support()
    array([False,  True,  True, False])
    """

    _parameter_constraints: dict = {
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "feature_generator": [callable, None],
    }

    def __init__(
        self,
        n_features_to_select=1,
        feature_generator=None,
    ):
        self.n_features_to_select = n_features_to_select
        self.feature_generator = feature_generator

    def fit(self, X, y):
        self._validate_params()
        X, y = _check_X_y(self, X, y, order="C")
        feature_generator = self.feature_generator or _default_feature_generator

        n_samples = X.shape[0]

        y_transformed = orth(y - y.mean(0))
        W = np.zeros((n_samples, self.n_features_to_select))
        indices = np.zeros(self.n_features_to_select, dtype=int)
        scores = np.zeros(self.n_features_to_select, dtype=float)

        max_feat_idx = -1

        for i in range(self.n_features_to_select):
            best_score = -np.inf
            best_index = -1
            best_feat = None
            for j, feat in feature_generator(X, skip_indices=indices[:i]):
                _check_generated_features(j, feat, n_samples)
                max_feat_idx = max(max_feat_idx, j)
                feat_centered = feat - feat.mean()
                feat_orth = _classical_gram_schmidt(feat_centered, W[:, :i])
                r = feat_orth.T @ y_transformed
                score = r @ r
                if score > best_score:
                    best_score = score
                    best_index = j
                    best_feat = feat_orth
            indices[i] = best_index
            scores[i] = best_score
            W[:, i] = best_feat
        self.indices_ = indices
        self.scores_ = scores
        self.n_features_ = max_feat_idx + 1
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        support = np.zeros(self.n_features_, dtype=bool)
        support[self.indices_] = True
        return support
