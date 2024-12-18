"""
Feature selection with mini-batch.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_X_y

from ._cancorr_fast import _forward_search  # type: ignore
from ._fastcan import FastCan, _prepare_search


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
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=False,
)
def minibatch(X, y, n_features_to_select=1, batch_size=1, verbose=1):
    """Feature selection using :class:`fastcan.FastCan` with mini batches.

    It is suitable for selecting a very large number of features
    even larger than the number of samples.

    Similar to the correlation filter which selects each feature without considering
    the redundancy, the function selects features in mini-batch and the
    redundancy between the two mini-batches will be ignored.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples, n_outputs)
        Target matrix.

    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    batch_size : int, default=1
        The number of features in a mini-batch.
        It is recommended that batch_size be less than n_samples.

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
    X, y = check_X_y(X, y, ensure_2d=True, multi_output=True)
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
    indices_select = np.zeros(0, dtype=int)
    for i in range(n_outputs):
        y_i = y[:, i]
        batch_split_i = np.diff(
            np.r_[
                np.arange(n_to_select_split[i], step=batch_size, dtype=int),
                n_to_select_split[i],
            ]
        )
        for j, batch_size_j in enumerate(batch_split_i):
            if j == 0:
                selector_j = FastCan(
                    batch_size_j, indices_exclude=indices_select, verbose=0
                ).fit(X, y_i)
                X_transformed_ = deepcopy(selector_j.X_transformed_)
                indices = selector_j.indices_
            else:
                indices, scores, mask = _prepare_search(
                    n_features,
                    batch_size_j,
                    selector_j.indices_include_,
                    np.r_[selector_j.indices_exclude_, indices_select],
                )
                _forward_search(
                    X=X_transformed_,
                    V=selector_j.y_transformed_,
                    t=batch_size_j,
                    tol=selector_j.tol,
                    num_threads=n_threads,
                    verbose=0,
                    mask=mask,
                    indices=indices,
                    scores=scores,
                )
            indices_select = np.r_[indices_select, indices]
            if verbose == 1:
                print(
                    f"Progress: {indices_select.size}/{n_features_to_select}", end="\r"
                )
    if verbose == 1:
        print()
    return indices_select
