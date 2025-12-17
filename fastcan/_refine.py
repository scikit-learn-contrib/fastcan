"""
Refine fastcan selection results.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral

import numpy as np
from sklearn.utils._openmp_helpers import (  # ty: ignore[unresolved-import]
    _openmp_effective_n_threads,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted

from ._cancorr_fast import _greedy_search
from ._fastcan import FastCan, _prepare_search


@validate_params(
    {
        "selector": [FastCan],
        "drop": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"all"}),
            "array-like",
        ],
        "max_iter": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=True,
)
def refine(selector, drop=1, max_iter=None, verbose=1):
    """Two-stage refining for the results of :class:`fastcan.FastCan`.

    In the refining process, the selected features will be dropped, and
    the vacancy positions will be refilled from the candidate features.

    The processing of a vacant position is refilled after searching all
    candidate features is called an `iteration`.

    The processing of a vacant position is refilled by a different features
    from the dropped one, which increase the SSC of the selected features
    is called a `valid iteration`.

    Parameters
    ----------
    selector : FastCan
        FastCan selector.

    drop : int or array-like of shape (n_drops,) or "all", default=1
        The number of the selected features dropped for the consequent
        reselection.

    max_iter : int, default=None
        The maximum number of valid iterations in the refining process.

    verbose : int, default=1
        The verbosity level.

    Returns
    -------
    indices : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features.

    scores : ndarray of shape (n_features_to_select,), dtype=float
        The h-correlation/eta-cosine of selected features.

    References
    ----------
    * Zhang L., Li K., Bai E. W. and Irwin G. W. (2015).
        Two-stage orthogonal least squares methods for  neural network construction.
        IEEE Transactions on Neural Networks and Learning Systems, 26(8), 1608-1621.

    Examples
    --------
    >>> from fastcan import FastCan, refine
    >>> X = [[1, 1, 0], [0.01, 0, 0], [-1, 0, 1], [0, 0, 0]]
    >>> y = [1, 0, -1, 0]
    >>> selector = FastCan(2, verbose=0).fit(X, y)
    >>> print(f"Indices: {selector.indices_}", f", SSC: {selector.scores_.sum():.5f}")
    Indices: [0 1] , SSC: 0.99998
    >>> indices, scores = refine(selector, drop=1, verbose=0)
    >>> print(f"Indices: {indices}", f", SSC: {scores.sum():.5f}")
    Indices: [1 2] , SSC: 1.00000
    """
    check_is_fitted(selector)
    X_transformed_ = np.copy(selector.X_transformed_, order="F")
    n_features = selector.n_features_in_
    n_features_to_select = selector.n_features_to_select
    indices_include = selector.indices_include_
    indices_exclude = selector.indices_exclude_

    n_inclusions = indices_include.size
    n_selections = n_features_to_select - n_inclusions
    n_threads = _openmp_effective_n_threads()

    if drop == "all":
        drop = np.arange(1, n_selections)
    else:
        drop = np.atleast_1d(drop).astype(int)

    if (drop.max() >= n_selections) or (drop.min() < 1):
        raise ValueError(
            "`drop` should be between `1<=drop<n_features_to_select-n_inclusions`, "
            f"but got drop={drop} and n_selections={n_selections}."
        )

    if max_iter is None:
        max_iter = np.inf

    n_iters = 0
    n_valid_iters = 0
    best_scores = selector.scores_
    best_indices = selector.indices_
    best_ssc = selector.scores_.sum()
    indices_temp = best_indices
    for drop_n in drop:
        i = 0
        while i < n_features_to_select:
            rolled_indices = np.r_[
                indices_include, np.roll(indices_temp[n_inclusions:], -1)
            ]
            indices, scores, mask = _prepare_search(
                n_features,
                n_features_to_select,
                rolled_indices[:-drop_n],
                indices_exclude,
            )
            _greedy_search(
                X=X_transformed_,
                V=selector.y_transformed_,
                t=selector.n_features_to_select,
                tol=selector.tol,
                num_threads=n_threads,
                verbose=0,
                mask=mask,
                indices=indices,
                scores=scores,
            )

            if (scores.sum() > best_ssc) and (set(indices) != set(best_indices)):
                i = 0
                n_valid_iters += 1
                best_scores = scores
                best_indices = indices
                best_ssc = scores.sum()
            else:
                i += 1

            indices_temp = indices
            n_iters += 1
            if verbose == 1:
                print(
                    f"No. of iterations: {n_iters}, "
                    f"No. of valid iterations {n_valid_iters}, "
                    f"SSC: {best_scores.sum():.5f}",
                    end="\r",
                )

            if n_iters >= max_iter:
                if verbose == 1:
                    print()
                return best_indices, best_scores

    if verbose == 1:
        print()
    return best_indices, best_scores
