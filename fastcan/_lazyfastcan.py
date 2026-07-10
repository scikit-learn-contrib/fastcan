"""
Lazy FastCan selector.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils._array_api import (
    get_namespace_and_device,
    move_to,
    supported_float_dtypes,
)
from sklearn.utils._param_validation import Interval

from ._fastcan import _check_indices_params, _check_X_y


def _classical_gram_schmidt(x, W, xp):
    """Classical Gram-Schmidt orthogonalization.
    Parameters
    ----------
    x: (n_samples, batch_size) Features for orthogonalization.
    W: (n_samples, n_selected) Selected orthonormal features,
        where n_selected can be zero.

    Returns
    -------
    q: (n_samples, batch_size) Orthonormal features.
    """
    q = x - W @ (W.T @ x)
    norm = xp.linalg.vector_norm(q, axis=0)
    norm_safe = xp.where(norm == 0, xp.ones_like(norm), norm)
    return q / norm_safe


def _default_feature_generator(X, skip_indices, batch_size=16):
    """Default feature generator that yields batches of columns from X."""
    xp, _, device_ = get_namespace_and_device(X)
    n_features = X.shape[1]
    skip_indices = move_to(skip_indices, xp=np, device="cpu")
    skip_indices = _check_indices_params(skip_indices, n_features)

    valid_mask = np.ones(n_features, dtype=bool)
    valid_mask[skip_indices] = False

    # 2. Extract all valid indices at once
    valid_indices = np.flatnonzero(valid_mask)
    valid_indices = move_to(valid_indices, xp=xp, device=device_)

    n_valid = valid_indices.shape[0]
    for i in range(0, n_valid, batch_size):
        stop_idx = min(i + batch_size, n_valid)
        batch_idx = valid_indices[i:stop_idx]
        yield batch_idx, xp.take(X, batch_idx, axis=1)


def _check_generated_features(ids, feats, n_samples, sample_mask, xp):
    """Check the validity of generated features."""
    if (not hasattr(ids, "ndim")) or (ids.ndim != 1):
        raise ValueError(
            f"Generated feature index {ids} is not a one-dimensional array."
        )
    if (not hasattr(feats, "ndim")) or (feats.ndim != 2):
        raise ValueError(
            f"Generated feature with index {ids} is not two-dimensional array."
        )
    if not xp.isdtype(ids.dtype, xp.int64):
        raise TypeError(f"Generated feature index {ids} is not an array of integers.")
    if xp.any(ids < 0):
        raise ValueError(f"Generated feature index {ids} contains negative values.")
    if feats.shape[0] != n_samples:
        raise ValueError(
            f"Generated feature with index {ids} has length {feats.shape[0]}, "
            f"but expected length is {n_samples}."
        )
    feature_masked = feats[sample_mask]
    if not xp.all(xp.isfinite(feature_masked)):
        raise ValueError(
            f"Generated feature with index {ids} contains non-finite values."
        )
    return feature_masked


class LazyFastCan(BaseEstimator):
    """Lazy version of FastCan selector.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    feature_generator : callable or None, default=None
        A generator that takes the input data `X` and an array of `skip_indices`
        and yields `(index, feature)` tuples,
        where `index` should be an 1D array of integers from 0 to n_features-1.
        The feature should be a 2D array of shape (n_samples, batch_size).
        The `skip_indices` indicates which features should be skipped.
        If the generator requires additional parameters,
        you can use `functools.partial` to wrap the generator with those parameters.

    sample_mask : ndarray of shape (n_samples,), default=None
        Bool mask for valid samples.

    tol : float, default=1e-6
        Tolerance for linear dependence check.
        The classical Gram-Schmidt fails when abs(x.T @ W) > `tol`.

    Attributes
    ----------
    indices_ : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features. The order of the indices
        is corresponding to the feature selection process.

    scores_ : ndarray of shape (n_features_to_select,), dtype=float
        The h-correlation scores of selected features. The order of
        the scores is corresponding to the feature selection process.


    Examples
    --------
    >>> from functools import partial
    >>> from fastcan import LazyFastCan
    >>> from fastcan.narx import gen_time_shift_features, make_time_shift_ids
    >>> fg = partial(gen_time_shift_features, ids=make_time_shift_ids(2, 2))
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [[0, 0], [0, 1], [1, 1], [0, 2]]
    >>> LazyFastCan(2, feature_generator=fg).fit(X, y).indices_
    array([2, 1])
    """

    _parameter_constraints: dict = {
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "feature_generator": [callable, None],
        "sample_mask": ["array-like", None],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_features_to_select=1,
        feature_generator=None,
        sample_mask=None,
        tol=1e-6,
    ):
        self.n_features_to_select = n_features_to_select
        self.feature_generator = feature_generator
        self.sample_mask = sample_mask
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        xp, is_array_api, device_ = get_namespace_and_device(X)
        if is_array_api:
            y = move_to(y, xp=xp, device=device_)
        X, y = _check_X_y(self, X, y, dtype=supported_float_dtypes(xp, device_), xp=xp)
        feature_generator = self.feature_generator or _default_feature_generator
        n_samples = X.shape[0]
        if self.sample_mask is None:
            sample_mask = xp.ones(n_samples, dtype=xp.bool, device=device_)
        else:
            sample_mask = check_array(self.sample_mask, dtype=xp.bool, ensure_2d=False)
            if sample_mask.shape[0] != n_samples:
                raise ValueError(
                    f"The length of sample_mask {sample_mask.shape[0]} does not match "
                    f"the number of samples {n_samples}."
                )
        y_masked = y[sample_mask]
        y_transformed, _ = xp.linalg.qr(
            y_masked - xp.mean(y_masked, axis=0), mode="reduced"
        )
        n_samples_masked = y_masked.shape[0]
        W = xp.zeros(
            (n_samples_masked, self.n_features_to_select),
            dtype=X.dtype,
            device=device_,
        )
        indices = xp.zeros(self.n_features_to_select, dtype=xp.int64)
        scores = xp.zeros(self.n_features_to_select, dtype=X.dtype, device=device_)

        max_feat_idx = xp.asarray(-1, device=device_)

        for i in range(self.n_features_to_select):
            best_score = -xp.inf
            best_index = None
            best_feat = None
            for id_batch, feat_batch in feature_generator(X, skip_indices=indices[:i]):
                feat_batch = _check_generated_features(
                    id_batch, feat_batch, n_samples, sample_mask, xp
                )
                max_feat_idx = xp.maximum(max_feat_idx, xp.max(id_batch))
                feat_centered = feat_batch - xp.mean(feat_batch, axis=0)
                feat_orth = _classical_gram_schmidt(feat_centered, W[:, :i], xp)
                # Linear dependence check
                if i > 0:
                    g = feat_orth.T @ W[:, :i]
                    g_max = xp.max(xp.abs(g), axis=1)
                    valid_mask = g_max <= self.tol
                    if not xp.any(valid_mask):
                        continue
                    valid_indices = xp.nonzero(valid_mask)[0]
                    feat_valid = xp.take(feat_orth, valid_indices, axis=1)
                else:
                    valid_indices = xp.arange(feat_orth.shape[1], device=device_)
                    feat_valid = feat_orth

                r = feat_valid.T @ y_transformed
                score_batch = xp.sum(r * r, axis=1)

                idx = xp.argmax(score_batch)
                score = float(score_batch[idx])

                if score > best_score:
                    best_score = score
                    valid_idx_in_batch = int(valid_indices[int(idx)])
                    best_index = id_batch[valid_idx_in_batch]
                    best_feat = feat_orth[:, valid_idx_in_batch]
            if best_index is None:
                raise RuntimeError(
                    "No more features can be selected when adding "
                    f"the {i + 1}th feature."
                )
            if best_score == 0:
                raise RuntimeError(
                    f"No improvement can be achieved when adding the {i + 1}th feature."
                )
            indices[i] = best_index
            scores[i] = best_score
            W[:, i] = best_feat
        self.indices_ = move_to(indices, xp=xp, device=device_)
        self.scores_ = scores
        self.n_features_ = max_feat_idx + 1
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        return tags
