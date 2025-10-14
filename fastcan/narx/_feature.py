"""
Time-series features for NARX model.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import math
import warnings
from itertools import combinations_with_replacement
from numbers import Integral

import numpy as np
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, validate_params


@validate_params(
    {"X": ["array-like"], "ids": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def make_time_shift_features(X, ids):
    """Make time shift features.

    Parameters
    ----------
    X : array-likeof shape (n_samples, n_features)
        The data to transform, column by column.

    ids : array-like of shape (n_outputs, 2)
        The unique id numbers of output features, which are
        (feature_idx, delay).

    Returns
    -------
    out : ndarray of shape (n_samples, n_outputs)
        The matrix of features, where `n_outputs` is the number of time shift
        features generated from the combination of inputs.

    Examples
    --------
    >>> from fastcan.narx import make_time_shift_features
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> ids = [[0, 0], [0, 1], [1, 1]]
    >>> make_time_shift_features(X, ids)
    array([[ 1., nan, nan],
           [ 3.,  1.,  2.],
           [ 5.,  3.,  4.],
           [ 7.,  5.,  6.]])
    """
    X = check_array(X, ensure_2d=True, dtype=float, ensure_all_finite="allow-nan")
    ids = check_array(ids, ensure_2d=True, dtype=int)
    n_samples = X.shape[0]
    n_outputs = ids.shape[0]
    out = np.zeros([n_samples, n_outputs])
    for i, id_temp in enumerate(ids):
        out[:, i] = np.r_[
            np.full(id_temp[1], np.nan),
            X[: -id_temp[1] or None, id_temp[0]],
        ]

    return out


@validate_params(
    {
        "n_features": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_delay": [
            Interval(Integral, 0, None, closed="left"),
        ],
        "include_zero_delay": ["boolean", "array-like"],
    },
    prefer_skip_nested_validation=True,
)
def make_time_shift_ids(
    n_features=1,
    max_delay=1,
    include_zero_delay=False,
):
    """Generate ids for time shift features.
    (variable_index, delay_number)

    Parameters
    ----------
    n_features: int, default=1
        The number of input features.

    max_delay : int, default=1
        The maximum delay of time shift features.

    include_zero_delay : {bool, array-like} of shape (n_features,) default=False
        Whether to include the original (zero-delay) features.

    Returns
    -------
    ids : array-like of shape (`n_output_features_`, 2)
        The unique id numbers of output features.

    Examples
    --------
    >>> from fastcan.narx import make_time_shift_ids
    >>> make_time_shift_ids(2, max_delay=3, include_zero_delay=[True, False])
    array([[0, 0],
           [0, 1],
           [0, 2],
           [0, 3],
           [1, 1],
           [1, 2],
           [1, 3]])
    """
    if isinstance(include_zero_delay, bool):
        return np.stack(
            np.meshgrid(
                range(n_features),
                range(not include_zero_delay, max_delay + 1),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 2)

    include_zero_delay = check_array(include_zero_delay, ensure_2d=False, dtype=bool)
    if include_zero_delay.shape[0] != n_features:
        raise ValueError(
            f"The length of `include_zero_delay`={include_zero_delay} "
            f"should be equal to `n_features`={n_features}."
        )

    ids = np.stack(
        np.meshgrid(
            range(n_features),
            range(max_delay + 1),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 2)
    exclude_zero_delay_idx = np.where(~include_zero_delay)[0]
    mask = np.isin(ids[:, 0], exclude_zero_delay_idx) & (ids[:, 1] == 0)
    return ids[~mask]


@validate_params(
    {"X": ["array-like"], "ids": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def make_poly_features(X, ids):
    """Make polynomial features.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to transform, column by column.

    ids : array-like of shape (n_outputs, degree)
        The unique id numbers of output features, which are formed
        of non-negative int values.

    Returns
    -------
    out.T : ndarray of shape (n_samples, n_outputs)
        The matrix of features, where n_outputs is the number of polynomial
        features generated from the combination of inputs.

    Examples
    --------
    >>> from fastcan.narx import make_poly_features
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> ids = [[0, 0], [0, 1], [1, 1], [0, 2]]
    >>> make_poly_features(X, ids)
    array([[ 1.,  1.,  1.,  2.],
           [ 1.,  3.,  9.,  4.],
           [ 1.,  5., 25.,  6.],
           [ 1.,  7., 49.,  8.]])
    """
    X = check_array(X, ensure_2d=True, dtype=float, ensure_all_finite="allow-nan")
    ids = check_array(ids, ensure_2d=True, dtype=int)
    n_samples = X.shape[0]
    n_outputs, degree = ids.shape

    # Generate polynomial features
    out = np.ones([n_outputs, n_samples])
    unique_features = np.unique(ids)
    unique_features = unique_features[unique_features != 0]
    for i in range(degree):
        for j in unique_features:
            mask = ids[:, i] == j
            out[mask, :] *= X[:, j - 1]

    return out.T


@validate_params(
    {
        "n_features": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "degree": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_poly": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def make_poly_ids(
    n_features=1,
    degree=1,
    max_poly=None,
    random_state=None,
):
    """Generate ids for polynomial features.
    (variable_index, variable_index, ...)
    variable_index starts from 1, and 0 represents constant.

    Parameters
    ----------
    n_features: int, default=1
        The number of input features.

    degree : int, default=1
        The maximum degree of polynomial features.

    max_poly : int, default=None
        Maximum number of ids of polynomial features to generate.
        Randomly selected by reservoir sampling.
        If None, all possible ids are returned.

    random_state : int or RandomState instance, default=None
        Used when `max_poly` is not None to subsample ids of polynomial features.
        See :term:`Glossary <random_state>` for details.

    Returns
    -------
    ids : array-like of shape (n_outputs, degree)
        The unique id numbers of output features.

    Examples
    --------
    >>> from fastcan.narx import make_poly_ids
    >>> make_poly_ids(2, degree=3)
    array([[0, 0, 1],
           [0, 0, 2],
           [0, 1, 1],
           [0, 1, 2],
           [0, 2, 2],
           [1, 1, 1],
           [1, 1, 2],
           [1, 2, 2],
           [2, 2, 2]])
    """
    n_total = math.comb(n_features + degree, degree) - 1
    if n_total > np.iinfo(np.intp).max:
        msg = (
            "The current configuration would "
            f"result in {n_total} features which is too large to be "
            f"indexed by {np.intp().dtype.name}."
        )
        raise ValueError(msg)
    if n_total > 10_000_000:
        warnings.warn(
            "Total number of polynomial features is larger than 10,000,000! "
            f"The current configuration would result in {n_total} features. "
            "This may take a while.",
            UserWarning,
        )
    if max_poly is not None and max_poly < n_total:
        # reservoir sampling
        rng = np.random.default_rng(random_state)
        reservoir = []
        for i, comb in enumerate(
            combinations_with_replacement(range(n_features + 1), degree)
        ):
            if i < max_poly:
                reservoir.append(comb)
            else:
                j = rng.integers(0, i + 1)
                if j < max_poly:
                    reservoir[j] = comb
        ids = np.array(reservoir)
    else:
        ids = np.array(
            list(combinations_with_replacement(range(n_features + 1), degree))
        )

    const_id = np.where((ids == 0).all(axis=1))
    return np.delete(ids, const_id, 0)  # remove the constant feature


def _validate_time_shift_poly_ids(
    time_shift_ids, poly_ids, n_samples=None, n_features=None, n_outputs=None
):
    if n_samples is None:
        n_samples = np.inf
    if n_features is None:
        n_features = np.inf
    if n_outputs is None:
        n_outputs = np.inf

    # Validate time_shift_ids
    time_shift_ids_ = check_array(
        time_shift_ids,
        ensure_2d=True,
        dtype=int,
    )
    if time_shift_ids_.shape[1] != 2:
        raise ValueError(
            "time_shift_ids should have shape (n_variables, 2), "
            f"but got {time_shift_ids_.shape}."
        )
    if (time_shift_ids_[:, 0].min() < 0) or (
        time_shift_ids_[:, 0].max() >= n_features + n_outputs
    ):
        raise ValueError(
            "The element x of the first column of time_shift_ids should "
            f"satisfy 0 <= x < {n_features + n_outputs}."
        )
    if (time_shift_ids_[:, 1].min() < 0) or (time_shift_ids_[:, 1].max() >= n_samples):
        raise ValueError(
            "The element x of the second column of time_shift_ids should "
            f"satisfy 0 <= x < {n_samples}."
        )
    # Validate poly_ids
    poly_ids_ = check_array(
        poly_ids,
        ensure_2d=True,
        dtype=int,
    )
    if (poly_ids_.min() < 0) or (poly_ids_.max() > time_shift_ids_.shape[0]):
        raise ValueError(
            "The element x of poly_ids should "
            f"satisfy 0 <= x <= {time_shift_ids_.shape[0]}."
        )
    return time_shift_ids_, poly_ids_


def _validate_feat_delay_ids(
    feat_ids, delay_ids, n_samples=None, n_features=None, n_outputs=None
):
    """Validate feat_ids and delay_ids."""
    if n_samples is None:
        n_samples = np.inf
    if n_features is None:
        n_features = np.inf
    if n_outputs is None:
        n_outputs = np.inf

    # Validate feat_ids
    feat_ids_ = check_array(
        feat_ids,
        ensure_2d=True,
        dtype=np.int32,
        order="C",
    )
    if (feat_ids_.min() < -1) or (feat_ids_.max() > n_features + n_outputs - 1):
        raise ValueError(
            "The element x of feat_ids should "
            f"satisfy -1 <= x <= {n_features + n_outputs - 1}."
        )
    # Check if any row of feat_ids only contains -1
    if np.all(feat_ids_ == -1, axis=1).any():
        raise ValueError("`feat_ids` should not contain rows that only have -1.")
    # Validate delay_ids
    delay_ids_ = check_array(
        delay_ids,
        ensure_2d=True,
        dtype=np.int32,
        order="C",
    )
    if delay_ids_.shape != feat_ids_.shape:
        raise ValueError(
            "The shape of delay_ids should be equal to "
            f"the shape of feat_ids {feat_ids_.shape}, "
            f"but got {delay_ids_.shape}."
        )
    if ((delay_ids_ == -1) != (feat_ids_ == -1)).any():
        raise ValueError(
            "The element x of delay_ids should be -1 "
            "if and only if the element x of feat_ids is -1."
        )
    if (delay_ids_.min() < -1) or (delay_ids_.max() >= n_samples):
        raise ValueError(
            f"The element x of delay_ids should satisfy -1 <= x < {n_samples}."
        )
    return feat_ids_, delay_ids_


@validate_params(
    {
        "feat_ids": ["array-like"],
        "delay_ids": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def fd2tp(feat_ids, delay_ids):
    """
    Convert feat_ids and delay_ids to time_shift_ids and poly_ids.
    The polynomial terms, e.g., x0(k-1)^2, x0(k-2)x1(k-3), can be
    represented by two ways:

    #. feat_ids and delay_ids, e.g., [[0, 0], [0, 1]] and [[1, 1], [2, 3]]

    #. time_shift_ids and poly_ids, e.g., [[0, 1], [0, 2], [1, 3]] and [[1, 1], [2, 3]]

    For feat_ids, [0, 0] and [0, 1] represent x0*x0 and x0*x1, while
    for delay_ids, [1, 1] and [2, 3] represent the delays of features in feat_ids.

    For time_shift_ids, [0, 1], [0, 2], and [1, 3] represents x0(k-1), x0(k-2),
    and x1(k-3), respectively. For poly_ids, [1, 1] and [2, 3] represent the first
    variable multiplying the first variable given by time_shift_ids, i.e.,
    x0(k-1)*x0(k-1), and the second variable multiplying the third variable, i.e.,
    x0(k-1)*x1(k-3).

    Parameters
    ----------
    feat_ids : array-like of shape (n_terms, degree), default=None
        The unique id numbers of features to form polynomial terms.
        The id -1 stands for the constant 1.
        The id 0 to n are the index of features.

    delay_ids : array-like of shape (n_terms, degree), default=None
        The delays of each feature in polynomial terms.
        The id -1 stands for empty.
        The id 0 stands for 0 delay.
        The positive integer id k stands for k-th delay.

    Returns
    -------
    time_shift_ids : array-like of shape (n_variables, 2), default=None
        The unique id numbers of time shift variables, which are
        (feature_idx, delay).

    poly_ids : array-like of shape (n_polys, degree), default=None
        The unique id numbers of polynomial terms, excluding the intercept.
        The id 0 stands for the constant 1.
        The id 1 to n are the index+1 of time_shift_ids.

    Examples
    --------
    >>> from fastcan.narx import fd2tp
    >>> # Encode x0(k-1), x0(k-2)x1(k-3)
    >>> feat_ids = [[-1, 0], [0, 1]]
    >>> delay_ids = [[-1, 1], [2, 3]]
    >>> time_shift_ids, poly_ids = fd2tp(feat_ids, delay_ids)
    >>> print(time_shift_ids)
    [[0 1]
     [0 2]
     [1 3]]
    >>> print(poly_ids)
    [[0 1]
     [2 3]]
    """
    _feat_ids, _delay_ids = _validate_feat_delay_ids(feat_ids, delay_ids)
    featd = np.c_[_feat_ids.flatten(), _delay_ids.flatten()]
    # Ensure featd has at least one [-1, -1]
    time_shift_ids = np.unique(np.r_[[[-1, -1]], featd], axis=0)
    poly_ids = np.array(
        [np.where((time_shift_ids == row).all(axis=1))[0][0] for row in featd]
    ).reshape(_feat_ids.shape)
    time_shift_ids = time_shift_ids[time_shift_ids[:, 0] != -1]
    return time_shift_ids, poly_ids


@validate_params(
    {
        "time_shift_ids": ["array-like"],
        "poly_ids": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def tp2fd(time_shift_ids, poly_ids):
    """
    Convert time_shift_ids and poly_ids to feat_ids and delay_ids.
    The polynomial terms, e.g., x0(k-1)^2, x0(k-2)x1(k-3), can be
    represented by two ways:

    #. feat_ids and delay_ids, e.g., [[0, 0], [0, 1]] and [[1, 1], [2, 3]]

    #. time_shift_ids and poly_ids, e.g., [[0, 1], [0, 2], [1, 3]] and [[1, 1], [2, 3]]

    For feat_ids, [0, 0] and [0, 1] represent x0*x0 and x0*x1, while
    for delay_ids, [1, 1] and [2, 3] represent the delays of features in feat_ids.

    For time_shift_ids, [0, 1], [0, 2], and [1, 3] represents x0(k-1), x0(k-2),
    and x1(k-3), respectively. For poly_ids, [1, 1] and [2, 3] represent the first
    variable multiplying the first variable given by time_shift_ids, i.e.,
    x0(k-1)*x0(k-1), and the second variable multiplying the third variable, i.e.,
    x0(k-1)*x1(k-3).

    Parameters
    ----------
    time_shift_ids : array-like of shape (n_variables, 2)
        The unique id numbers of time shift variables, which are
        (feature_idx, delay).

    poly_ids : array-like of shape (n_polys, degree)
        The unique id numbers of polynomial terms, excluding the intercept.
        The id 0 stands for the constant 1.
        The id 1 to n are the index+1 of time_shift_ids.

    Returns
    -------
    feat_ids : array-like of shape (n_terms, degree), default=None
        The unique id numbers of features to form polynomial terms.
        The id -1 stands for the constant 1.
        The id 0 to n are the index of features.

    delay_ids : array-like of shape (n_terms, degree), default=None
        The delays of each feature in polynomial terms.
        The id -1 stands for empty.
        The id 0 stands for 0 delay.
        The positive integer id k stands for k-th delay.

    Examples
    --------
    >>> from fastcan.narx import tp2fd
    >>> # Encode x0(k-1), x0(k-2)x1(k-3)
    >>> time_shift_ids = [[0, 1], [0, 2], [1, 3]]
    >>> poly_ids = [[0, 1], [2, 3]]
    >>> feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    >>> print(feat_ids)
    [[-1  0]
     [ 0  1]]
    >>> print(delay_ids)
    [[-1  1]
     [ 2  3]]
    """
    _time_shift_ids, _poly_ids = _validate_time_shift_poly_ids(
        time_shift_ids,
        poly_ids,
    )
    feat_ids = np.full_like(_poly_ids, -1, dtype=int)
    delay_ids = np.full_like(_poly_ids, -1, dtype=int)
    for i, poly_id in enumerate(_poly_ids):
        for j, variable_id in enumerate(poly_id):
            if variable_id != 0:
                feat_ids[i, j] = _time_shift_ids[variable_id - 1, 0]
                delay_ids[i, j] = _time_shift_ids[variable_id - 1, 1]
    feat_ids = feat_ids
    delay_ids = delay_ids
    return feat_ids, delay_ids
