"""
Mini-batch selection.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral, Real

import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_X_y

from ._cancorr_fast import _greedy_search  # type: ignore[attr-defined]
from ._fastcan import _prepare_search


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "batch_size": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=True,
)
def minibatch(X, y, n_features_to_select=1, batch_size=1, tol=0.01, verbose=1):
    """Feature selection using :class:`fastcan.FastCan` with mini batches.

    It is suitable for selecting a very large number of features
    even larger than the number of samples.

    The function splits `n_features_to_select` into `n_outputs` parts and selects
    features for each part separately, ignoring the redundancy among outputs.
    In each part, the function selects features batch-by-batch. The batch size is less
    than or equal to `batch_size`.
    Like correlation filters, which select features one-by-one without considering
    the redundancy between two features, the function ignores the redundancy between
    two mini-batches.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples, n_outputs)
        Target matrix.

    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    batch_size : int, default=1
        The upper bound of the number of features in a mini-batch.
        It is recommended that batch_size be less than n_samples.

    tol : float, default=0.01
        Tolerance for linear dependence check.

    verbose : int, default=1
        The verbosity level.

    Returns
    -------
    indices : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features.

    Examples
    --------
    >>> from fastcan import minibatch
    >>> X = [[1, 1, 0], [0.01, 0, 0], [-1, 0, 1], [0, 0, 0]]
    >>> y = [1, 0, -1, 0]
    >>> indices = minibatch(X, y, 3, batch_size=2, verbose=0)
    >>> print(f"Indices: {indices}")
    Indices: [0 1 2]
    """
    X, y = check_X_y(X, y, ensure_2d=True, multi_output=True, order="F")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features = X.shape[1]
    n_outputs = y.shape[1]

    if n_features_to_select > n_features:
        raise ValueError(
            f"n_features_to_select {n_features_to_select} "
            f"must be <= n_features {n_features}."
        )

    n_threads = _openmp_effective_n_threads()

    n_to_select_split = np.diff(
        np.linspace(
            0, n_features_to_select, num=n_outputs + 1, endpoint=True, dtype=int
        )
    )
    X_transformed_ = X - X.mean(0)
    y_transformed_ = y - y.mean(0)
    indices_include = np.zeros(0, dtype=int)  # just an empty array
    indices_select = np.zeros(0, dtype=int)

    for i in range(n_outputs):
        y_i = y_transformed_[:, [i]]
        n_selected_i = 0
        while n_to_select_split[i] > n_selected_i:
            batch_size_temp = min(batch_size, n_to_select_split[i] - n_selected_i)
            indices, scores, mask = _prepare_search(
                n_features,
                batch_size_temp,
                indices_include,
                indices_select,
            )
            try:
                _greedy_search(
                    X=np.copy(X_transformed_, order="F"),
                    V=y_i,
                    t=batch_size_temp,
                    tol=tol,
                    num_threads=n_threads,
                    verbose=0,
                    mask=mask,
                    indices=indices,
                    scores=scores,
                )
            except RuntimeError:
                # If the batch size is too large, _greedy_search cannot find enough
                # samples to form a non-singular matrix. Then, reduce the batch size.
                indices = indices[indices != -1]
                batch_size_temp = indices.size
            indices_select = np.r_[indices_select, indices]
            n_selected_i += batch_size_temp
            if verbose == 1:
                print(
                    f"Progress: {indices_select.size}/{n_features_to_select}", end="\r"
                )
    if verbose == 1:
        print()
    return indices_select
