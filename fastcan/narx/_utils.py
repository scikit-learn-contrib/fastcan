"""NARX-related utilities."""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import (
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted

from .._fastcan import FastCan
from .._refine import refine
from ..utils import mask_missing_values
from ._base import NARX, _prepare_poly_terms, _validate_session_sizes
from ._feature import (
    make_poly_ids,
    make_time_shift_ids,
    tp2fd,
)


@validate_params(
    {
        "narx": [NARX],
        "term_space": [Interval(Integral, 1, None, closed="left")],
        "coef_space": [Interval(Integral, 1, None, closed="left")],
        "float_precision": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def print_narx(
    narx,
    term_space=20,
    coef_space=10,
    float_precision=3,
):
    """Print a NARX model as a Table which contains y idx, Term, and Coef.
    y idx is used to indicate which output (assuming it is a multi-output case)
    the term belongs to.

    Parameters
    ----------
    narx : NARX model
        The NARX model to be printed.

    term_space: int, default=20
        The space for the column of Term.

    coef_space: int, default=10
        The space for the column of Coef.

    float_precision: int, default=3
        The number of places after the decimal for Coef.

    Returns
    -------
    table : str
        The table of output index, terms, and coefficients of the NARX model.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from fastcan.narx import print_narx, NARX
    >>> X, y = load_diabetes(return_X_y=True)
    >>> print_narx(NARX().fit(X, y), term_space=10, coef_space=5, float_precision=0)
    | yid |   Term   |Coef |
    |-----|----------|-----|
    |  0  |Intercept | 152 |
    |  0  |  X[k,0]  | -10 |
    |  0  |  X[k,1]  |-240 |
    |  0  |  X[k,2]  | 520 |
    |  0  |  X[k,3]  | 324 |
    |  0  |  X[k,4]  |-792 |
    |  0  |  X[k,5]  | 477 |
    |  0  |  X[k,6]  | 101 |
    |  0  |  X[k,7]  | 177 |
    |  0  |  X[k,8]  | 751 |
    |  0  |  X[k,9]  | 68  |
    """
    check_is_fitted(narx)

    def _get_term_str(term_feat_ids, term_delay_ids):
        term_str = ""
        for _, (feat_id, delay_id) in enumerate(zip(term_feat_ids, term_delay_ids)):
            if -1 < feat_id < narx.n_features_in_:
                if delay_id == 0:
                    term_str += f"*X[k,{feat_id}]"
                else:
                    term_str += f"*X[k-{delay_id},{feat_id}]"
            elif feat_id >= narx.n_features_in_:
                term_str += f"*y_hat[k-{delay_id},{feat_id - narx.n_features_in_}]"
        return term_str[1:]

    yid_space = 5
    print(
        f"|{'yid':^{yid_space}}"
        + f"|{'Term':^{term_space}}"
        + f"|{'Coef':^{coef_space}}|"
    )
    print(f"|{'-' * yid_space}|{'-' * term_space}|{'-' * coef_space}|")
    for i in range(narx.n_outputs_):
        print(
            f"|{i:^{yid_space}}|"
            + f"{'Intercept':^{term_space}}|"
            + f"{narx.intercept_[i]:^{coef_space}.{float_precision}f}|"
        )
    for i, term_id in enumerate(zip(narx.feat_ids_, narx.delay_ids_)):
        print(
            f"|{narx.output_ids_[i]:^{yid_space}}|"
            + f"{_get_term_str(*term_id):^{term_space}}|"
            + f"{narx.coef_[i]:^{coef_space}.{float_precision}f}|"
        )


@validate_params(
    {
        "X": ["array-like", None],
        "y": ["array-like"],
        "n_terms_to_select": [
            Interval(Integral, 1, None, closed="left"),
            "array-like",
        ],
        "max_delay": [
            Interval(Integral, 0, None, closed="left"),
        ],
        "poly_degree": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "fit_intercept": ["boolean"],
        "session_sizes": [None, "array-like"],
        "max_candidates": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "include_zero_delay": [None, "array-like"],
        "static_indices": [None, "array-like"],
        "refine_verbose": ["verbose"],
        "refine_drop": [
            None,
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"all"}),
        ],
        "refine_max_iter": [
            None,
            Interval(Integral, 1, None, closed="left"),
        ],
    },
    prefer_skip_nested_validation=True,
)
def make_narx(
    X,
    y,
    n_terms_to_select,
    max_delay=1,
    poly_degree=1,
    *,
    fit_intercept=True,
    session_sizes=None,
    max_candidates=None,
    random_state=None,
    include_zero_delay=None,
    static_indices=None,
    refine_verbose=1,
    refine_drop=None,
    refine_max_iter=None,
    **params,
):
    """Find `time_shift_ids`, `poly_ids`, `output_ids` for a NARX model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target vector or matrix.

    n_terms_to_select : int or array-like of shape (n_outputs,)
        The parameter is the absolute number of polynomial terms to select for
        each output. If `n_terms_to_select` is an integer, it is the
        same for all outputs.

    max_delay : int, default=1
        The maximum delay of time shift features.

    poly_degree : int, default=1
        The maximum degree of polynomial features.

    fit_intercept : bool, default=True
        Whether to fit the intercept. If set to False, intercept will be zeros.

    session_sizes : array-like of shape (n_sessions,), default=None
        The sizes of measurement sessions for time-series.
        The sum of session_sizes should be equal to n_samples.
        If None, the whole data is treated as one session.

        .. versionadded:: 0.5.0

    max_candidates : int, default=None
        Maximum number of candidate polynomial terms retained before selection.
        Randomly selected by reservoir sampling.
        If None, all candidates are considered.

    random_state : int or RandomState instance, default=None
        Used when `max_candidates` is not None to subsample candidate terms.
        See :term:`Glossary <random_state>` for details.

    include_zero_delay : {None, array-like} of shape (n_features,) default=None
        Whether to include the original (zero-delay) features.

    static_indices : {None, array-like} of shape (n_static_features,) default=None
        The indices of static features without time delay.

        .. note::
            If the corresponding include_zero_delay of the static features is False, the
            static feature will be excluded from candidate features.

    refine_verbose : int, default=1
        The verbosity level of refine.

    refine_drop : int or "all", default=None
            The number of the selected features dropped for the consequent
            reselection. If `drop` is None, no refining will be performed.

    refine_max_iter : int, default=None
        The maximum number of valid iterations in the refining process.

    **params : dict
            Keyword arguments passed to
            `fastcan.FastCan`.

    Returns
    -------
    narx : NARX
        NARX instance.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from fastcan.narx import make_narx, print_narx
    >>> rng = np.random.default_rng(12345)
    >>> n_samples = 1000
    >>> max_delay = 3
    >>> e = rng.normal(0, 0.1, n_samples)
    >>> u0 = rng.uniform(0, 1, n_samples+max_delay) # input
    >>> u1 = rng.normal(0, 0.1, n_samples) # static input (i.e. no time shift)
    >>> y = np.zeros(n_samples+max_delay) # output
    >>> for i in range(max_delay, n_samples+max_delay):
    ...     y[i] = (0.5*y[i-1] + 0.3*u0[i]**2 + 2*u0[i-1]*u0[i-3] +
    ...             1.5*u0[i-2]*u1[i-max_delay] + 1)
    >>> y = y[max_delay:]+e
    >>> X = np.c_[u0[max_delay:], u1]
    >>> narx = make_narx(X=X,
    ...     y=y,
    ...     n_terms_to_select=4,
    ...     max_delay=3,
    ...     poly_degree=2,
    ...     static_indices=[1],
    ...     eta=True,
    ...     verbose=0,
    ...     refine_verbose=0,
    ...     refine_drop=1)
    >>> print(f"{mean_squared_error(y, narx.fit(X, y).predict(X)):.4f}")
    0.0289
    >>> print_narx(narx)
    | yid |        Term        |   Coef   |
    |-----|--------------------|----------|
    |  0  |     Intercept      |  1.050   |
    |  0  |    y_hat[k-1,0]    |  0.484   |
    |  0  |   X[k,0]*X[k,0]    |  0.306   |
    |  0  | X[k-1,0]*X[k-3,0]  |  2.000   |
    |  0  |  X[k-2,0]*X[k,1]   |  1.528   |
    """
    y = check_array(y, dtype=float, ensure_2d=False, ensure_all_finite="allow-nan")
    if X is None:
        X = np.empty((len(y), 0), dtype=float, order="C")  # Create 0 feature input
    else:
        X = check_array(
            X,
            dtype=float,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
            ensure_min_features=0,
        )
        check_consistent_length(X, y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_samples, n_outputs = y.shape
    if isinstance(n_terms_to_select, Integral):
        n_terms_to_select = np.full(n_outputs, n_terms_to_select, dtype=int)
    else:
        n_terms_to_select = column_or_1d(n_terms_to_select, dtype=int, warn=True)
        if len(n_terms_to_select) != n_outputs:
            raise ValueError(
                "The length of `n_terms_to_select` should be equal to "
                f"the number of outputs, {n_outputs}, but got "
                f"{len(n_terms_to_select)}."
            )
    session_sizes_cumsum = _validate_session_sizes(session_sizes, n_samples)

    xy_hstack = np.c_[X, y]
    n_features = X.shape[1]

    if include_zero_delay is None:
        _include_zero_delay = [True] * n_features + [False] * n_outputs
    else:
        _include_zero_delay = include_zero_delay + [False] * n_outputs

    time_shift_ids_all = make_time_shift_ids(
        n_features=xy_hstack.shape[1],
        max_delay=max_delay,
        include_zero_delay=_include_zero_delay,
    )

    if static_indices is None:
        static_indices = []

    time_shift_ids_all = np.delete(
        time_shift_ids_all,
        (
            np.isin(time_shift_ids_all[:, 0], static_indices)
            & (time_shift_ids_all[:, 1] > 0)
        ),
        0,
    )
    poly_ids_all = make_poly_ids(
        time_shift_ids_all.shape[0],
        poly_degree,
        max_poly=max_candidates,
        random_state=random_state,
    )

    poly_terms = _prepare_poly_terms(
        xy_hstack,
        time_shift_ids_all,
        poly_ids_all,
        session_sizes_cumsum,
        max_delay,
    )
    # Remove missing values
    poly_terms_masked, y_masked = mask_missing_values(poly_terms, y)

    selected_poly_ids = []
    for i in range(n_outputs):
        csf = FastCan(
            n_terms_to_select[i],
            **params,
        ).fit(poly_terms_masked, y_masked[:, i])
        if refine_drop is not None:
            indices, _ = refine(
                csf, drop=refine_drop, max_iter=refine_max_iter, verbose=refine_verbose
            )
            support = np.zeros(shape=poly_ids_all.shape[0], dtype=bool)
            support[indices] = True
        else:
            support = csf.get_support()
        selected_poly_ids.append(poly_ids_all[support])

    selected_poly_ids = np.vstack(selected_poly_ids)

    time_shift_ids = time_shift_ids_all[
        np.unique(selected_poly_ids[selected_poly_ids.nonzero()]) - 1, :
    ]
    poly_ids = (
        rankdata(
            np.r_[[[0] * poly_degree], selected_poly_ids],
            method="dense",
        ).reshape(-1, poly_degree)[1:]
        - 1
    )

    output_ids = [i for i in range(n_outputs) for _ in range(n_terms_to_select[i])]
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    return NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=fit_intercept,
    )
