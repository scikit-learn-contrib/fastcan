"""Utils functions."""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.utils import _safe_indexing, check_consistent_length, check_X_y
from sklearn.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def ssc(X, y):
    """Sum of the squared canonical correlation coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples, n_outputs)
        Target matrix.

    Returns
    -------
    ssc : float
        Sum of the squared canonical correlation coefficients.

    Examples
    --------
    >>> from fastcan.utils import ssc
    >>> X = [[1], [-1], [0]]
    >>> y = [[0], [1], [-1]]
    >>> ssc(X, y)
    np.float64(0.25)
    """
    X, y = check_X_y(
        X, y, dtype=float, ensure_2d=True, multi_output=True, ensure_min_samples=2
    )
    n_components = min(X.shape[1], y.shape[1])
    cca = CCA(n_components=n_components)
    X_c, y_c = cca.fit_transform(X, y)
    corrcoef = np.diagonal(np.corrcoef(X_c, y_c, rowvar=False), offset=n_components)
    return sum(corrcoef**2)


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
        "t": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def ols(X, y, t=1):
    """Orthogonal least-squares.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    t : int, default=1
        The parameter is the absolute number of features to select.

    Returns
    -------
    indices : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features. The order of the indices
        is corresponding to the feature selection process.

    scores : ndarray of shape (n_features_to_select,), dtype=float
        The scores of selected features. The order of
        the scores is corresponding to the feature selection process.

    Examples
    --------
    >>> from fastcan.utils import ols
    >>> X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    >>> y = [1, 0, 1, 0]
    >>> indices, scores = ols(X, y, 2)
    >>> indices
    array([0, 2])
    >>> scores
    array([0.5, 0.5])
    """
    X, y = check_X_y(X, y, dtype=float, ensure_2d=True)
    n_features = X.shape[1]
    w = X / np.linalg.norm(X, axis=0)
    v = y / np.linalg.norm(y)
    mask = np.zeros(n_features, dtype=bool)
    r2 = np.zeros(n_features, dtype=float)
    indices = np.zeros(t, dtype=int)
    scores = np.zeros(t, dtype=float)

    for i in range(t):
        for j in range(n_features):
            if not mask[j]:
                r = w[:, j] @ v
                r2[j] = r**2
        d = np.argmax(r2)
        indices[i] = d
        scores[i] = r2[d]
        if i == t - 1:
            return indices, scores
        mask[d] = True
        r2[d] = 0
        for j in range(n_features):
            if not mask[j]:
                w[:, j] = w[:, j] - w[:, d] * (w[:, d] @ w[:, j])
                w[:, j] /= np.linalg.norm(w[:, j], axis=0)


@validate_params(
    {
        "return_mask": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def mask_missing_values(*arrays, return_mask=False):
    """Remove missing values for all arrays.

    Parameters
    ----------
    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Arrays with consistent first dimension.

    return_mask : bool, default=False
        If True, return a mask of valid values.
        If False, return the arrays with missing values removed.

    Returns
    -------
    mask_valid : ndarray of shape (n_samples,)
        Mask of valid values.

    masked_arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Arrays with missing values removed.
        The order of the arrays is the same as the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan.utils import mask_missing_values
    >>> a = [[1, 2], [3, np.nan], [5, 6]]
    >>> b = [1, 2, 3]
    >>> mask_missing_values(a, b)
    [[[1, 2], [5, 6]], [1, 3]]
    >>> mask_missing_values(a, b, return_mask=True)
    array([ True, False,  True])
    """
    if len(arrays) == 0:
        return None
    check_consistent_length(*arrays)
    mask_valid = np.all(np.isfinite(np.c_[arrays]), axis=1)
    if return_mask:
        return mask_valid
    return [_safe_indexing(x, mask_valid) for x in arrays]
