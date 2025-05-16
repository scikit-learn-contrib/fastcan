"""
The module related to nonlinear autoregressive exogenous (NARX) model for system
identification.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import math
import warnings
from itertools import combinations_with_replacement
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ._fastcan import FastCan
from ._narx_fast import (  # type: ignore[attr-defined]
    _predict_step,
    _update_cfd,
    _update_terms,
)
from ._refine import refine
from .utils import mask_missing_values


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
    array([[1., 1., 2.],
           [3., 1., 2.],
           [5., 3., 4.],
           [7., 5., 6.]])
    """
    X = check_array(X, ensure_2d=True, dtype=float, ensure_all_finite="allow-nan")
    ids = check_array(ids, ensure_2d=True, dtype=int)
    n_samples = X.shape[0]
    n_outputs = ids.shape[0]
    out = np.zeros([n_samples, n_outputs])
    for i, id_temp in enumerate(ids):
        out[:, i] = np.r_[
            np.full(id_temp[1], X[0, id_temp[0]]),
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
    },
    prefer_skip_nested_validation=True,
)
def make_poly_ids(
    n_features=1,
    degree=1,
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
    n_outputs = math.comb(n_features + degree, degree) - 1
    if n_outputs > np.iinfo(np.intp).max:
        msg = (
            "The output that would result from the current configuration would"
            f" have {n_outputs} features which is too large to be"
            f" indexed by {np.intp().dtype.name}."
        )
        raise ValueError(msg)

    ids = np.array(
        list(
            combinations_with_replacement(
                range(n_features + 1),
                degree,
            )
        )
    )

    const_id = np.where((ids == 0).all(axis=1))
    return np.delete(ids, const_id, 0)  # remove the constant feature


def _valiate_time_shift_poly_ids(
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
    _time_shift_ids, _poly_ids = _valiate_time_shift_poly_ids(
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


class NARX(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """The Nonlinear Autoregressive eXogenous (NARX) model class.
    For example, a (polynomial) NARX model is like
    y(k) = y(k-1)*u(k-1) + u(k-1)^2 + u(k-2) + 1.5
    where y(k) is the system output at the k-th time step,
    u(k) is the system input at the k-th time step,
    u and y is called features,
    u(k-1) is called a (time shift) variable,
    u(k-1)^2 is called a (polynomial) term, and
    1.5 is called an intercept.

    Parameters
    ----------
    feat_ids : array-like of shape (n_terms, degree), default=None
        The unique id numbers of features to form polynomial terms.
        The id -1 stands for the constant 1.
        The id 0 to `n_features_in_`-1 are the input features.
        The id `n_features_in_` to `n_features_in_ + n_outputs_`-1 are
        the output features.
        E.g., for a 2-input 1-output system, the feat_ids [[-1, 0], [-1, 1], [0, 2]]
        may represent the polynomial terms 1*u(k-1, 0), 1*u(k, 1),
        and u(k-1, 0)*y(k-2, 0).

    delay_ids : array-like of shape (n_terms, degree), default=None
        The delays of each feature in polynomial terms.
        The id -1 stands for empty.
        The id 0 stands for 0 delay.
        The positive integer id k stands for k-th delay.
        E.g., for the polynomial terms 1*u(k-1, 0), 1*u(k, 1),
        and u(k-1, 0)*y(k-2, 0), the delay_ids [[-1, 1], [-1, 0], [1, 2]].

    output_ids : array-like of shape (n_polys,), default=None
        The id numbers indicate which output the polynomial term belongs to.
        It is useful in multi-output case.

    fit_intercept : bool, default=True
        Whether to fit the intercept. If set to False, intercept will be zeros.

    Attributes
    ----------
    coef_ : array of shape (`n_features_in_`,)
        Estimated coefficients for the linear regression problem.

    intercept_ : array of shape (`n_outputs_`,)
        Independent term in the linear model.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_outputs_ : int
        Number of outputs seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    max_delay_ : int
        The maximum time delay of the time shift variables.

    feat_ids_ : array-like of shape (n_terms, degree)
        The unique id numbers of features to form polynomial terms.
        The id -1 stands for the constant 1.

    delay_ids_ : array-like of shape (n_terms, degree)
        The delays of each feature in polynomial terms.
        The id -1 stands for empty.

    References
    ----------
    * Billings, Stephen A. (2013).
        Nonlinear System Identification: Narmax Methods in the Time, Frequency,
        and Spatio-Temporal Domains.

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan.narx import NARX, print_narx
    >>> rng = np.random.default_rng(12345)
    >>> n_samples = 1000
    >>> max_delay = 3
    >>> e = rng.normal(0, 0.1, n_samples)
    >>> u = rng.uniform(0, 1, n_samples+max_delay) # input
    >>> y = np.zeros(n_samples+max_delay) # output
    >>> for i in range(max_delay, n_samples+max_delay):
    ...     y[i] = 0.5*y[i-1] + 0.7*u[i-2] + 1.5*u[i-1]*u[i-3] + 1
    >>> y = y[max_delay:]+e
    >>> X = u[max_delay:].reshape(-1, 1)
    >>> feat_ids = [[-1, 1], # 1*y
    ...             [-1, 0], # 1*u
    ...             [0, 0]]  # u^2
    >>> delay_ids = [[-1, 1], # 1*y(k-1)
    ...              [-1, 2], # 1*u(k-2)
    ...              [1, 3]]  # u(k-1)*u(k-3)
    >>> narx = NARX(feat_ids=feat_ids,
    ...             delay_ids=delay_ids).fit(X, y, coef_init="one_step_ahead")
    >>> print_narx(narx)
    | yid |        Term        |   Coef   |
    =======================================
    |  0  |     Intercept      |  1.008   |
    |  0  |    y_hat[k-1,0]    |  0.498   |
    |  0  |      X[k-2,0]      |  0.701   |
    |  0  | X[k-1,0]*X[k-3,0]  |  1.496   |
    """

    _parameter_constraints: dict = {
        "feat_ids": [None, "array-like"],
        "delay_ids": [None, "array-like"],
        "output_ids": [None, "array-like"],
        "fit_intercept": ["boolean"],
    }

    def __init__(
        self,
        *,  # keyword call only
        feat_ids=None,
        delay_ids=None,
        output_ids=None,
        fit_intercept=True,
    ):
        self.feat_ids = feat_ids
        self.delay_ids = delay_ids
        self.output_ids = output_ids
        self.fit_intercept = fit_intercept

    @validate_params(
        {
            "coef_init": [None, StrOptions({"one_step_ahead"}), "array-like"],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X, y, sample_weight=None, coef_init=None, **params):
        """
        Fit narx model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, `n_features_in_`)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, `n_outputs_`)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        coef_init : array-like of shape (n_terms,), default=None
            The initial values of coefficients and intercept for optimization.
            When `coef_init` is None, the model will be a One-Step-Ahead NARX.
            When `coef_init` is `one_step_ahead`, the model will be a Multi-Step-Ahead
            NARX whose coefficients and intercept are initialized by the a
            One-Step-Ahead NARX.
            When `coef_init` is an array, the model will be a Multi-Step-Ahead
            NARX whose coefficients and intercept are initialized by the array.

            .. note::
                When coef_init is `one_step_ahead`, the model will be trained as a
                Multi-Step-Ahead NARX, rather than a One-Step-Ahead NARX.

        **params : dict
            Keyword arguments passed to
            `scipy.optimize.least_squares`.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        check_X_params = dict(dtype=float, order="C", ensure_all_finite="allow-nan")
        check_y_params = dict(
            ensure_2d=False, dtype=float, order="C", ensure_all_finite="allow-nan"
        )
        X, y = validate_data(
            self, X, y, validate_separately=(check_X_params, check_y_params)
        )
        check_consistent_length(X, y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, ensure_non_negative=True
        )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs_ = y.shape[1]
        n_samples, n_features = X.shape

        if self.feat_ids is None:
            feat_ids_ = make_poly_ids(n_features, 1) - 1
        else:
            feat_ids_ = self.feat_ids

        if self.delay_ids is None:
            delay_ids_ = np.copy(feat_ids_)
            delay_ids_[(feat_ids_ > -1) & (feat_ids_ < n_features)] = 0
            delay_ids_[(feat_ids_ >= n_features)] = 1
        else:
            delay_ids_ = self.delay_ids

        # Validate feat_ids and delay_ids
        self.feat_ids_, self.delay_ids_ = _validate_feat_delay_ids(
            feat_ids_,
            delay_ids_,
            n_samples=n_samples,
            n_features=n_features,
            n_outputs=self.n_outputs_,
        )

        n_terms = self.feat_ids_.shape[0]
        # Validate output_ids
        if self.output_ids is None:
            self.output_ids_ = np.zeros(n_terms, dtype=np.int32)
        else:
            self.output_ids_ = column_or_1d(
                self.output_ids,
                dtype=np.int32,
                warn=True,
            )
            if len(self.output_ids_) != n_terms:
                raise ValueError(
                    "The length of output_ids should be equal to "
                    f"the number of polynomial terms, {n_terms}, "
                    f"but got {len(self.output_ids_)}."
                )
            if (self.output_ids_.min() < 0) or (
                self.output_ids_.max() >= self.n_outputs_
            ):
                raise ValueError(
                    "The element x of output_ids should "
                    f"satisfy 0 <= x < {self.n_outputs_}."
                )
        # Check if self.output_ids_ contains all values from 0 to n_outputs-1
        if set(self.output_ids_) != set(range(self.n_outputs_)):
            warnings.warn(
                f"output_ids got {self.output_ids_}, which does not "
                f"contain all values from 0 to {self.n_outputs_ - 1}."
                "The prediction for the missing outputs will be a constant"
                "(i.e., intercept).",
                UserWarning,
            )

        self.max_delay_ = self.delay_ids_.max()

        if isinstance(coef_init, (type(None), str)):
            # fit a one-step-ahead NARX model
            time_shift_ids, poly_ids = fd2tp(self.feat_ids_, self.delay_ids_)
            xy_hstack = np.c_[X, y]
            osa_narx = LinearRegression(fit_intercept=self.fit_intercept)
            time_shift_vars = make_time_shift_features(xy_hstack, time_shift_ids)
            poly_terms = make_poly_features(time_shift_vars, poly_ids)
            # Remove missing values
            poly_terms_masked, y_masked, sample_weight_masked = mask_missing_values(
                poly_terms, y, sample_weight
            )
            coef = np.zeros(n_terms, dtype=float)
            intercept = np.zeros(self.n_outputs_, dtype=float)
            for i in range(self.n_outputs_):
                output_i_mask = self.output_ids_ == i
                if np.sum(output_i_mask) == 0:
                    if self.fit_intercept:
                        intercept[i] = np.mean(y_masked[:, i])
                    else:
                        intercept[i] = 0.0
                    continue
                osa_narx.fit(
                    poly_terms_masked[:, output_i_mask],
                    y_masked[:, i],
                    sample_weight_masked,
                )
                coef[output_i_mask] = osa_narx.coef_
                intercept[i] = osa_narx.intercept_
            if coef_init is None:
                self.coef_ = coef
                self.intercept_ = intercept
                return self

            if self.fit_intercept:
                coef_init = np.r_[coef, intercept]
            else:
                coef_init = coef
        else:
            coef_init = check_array(
                coef_init,
                ensure_2d=False,
                dtype=float,
            )
            if self.fit_intercept:
                n_coef_intercept = n_terms + self.n_outputs_
            else:
                n_coef_intercept = n_terms
            if coef_init.shape[0] != n_coef_intercept:
                raise ValueError(
                    "`coef_init` should have the shape of "
                    f"({(n_coef_intercept,)}), but got {coef_init.shape}."
                )

        grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_cfd_ids(
            self.feat_ids_, self.delay_ids_, self.output_ids_, X.shape[1]
        )
        sample_weight_sqrt = np.sqrt(sample_weight).reshape(-1, 1)
        res = least_squares(
            NARX._loss,
            x0=coef_init,
            jac=NARX._grad,
            args=(
                _predict_step,
                X,
                y,
                self.feat_ids_,
                self.delay_ids_,
                self.output_ids_,
                self.fit_intercept,
                sample_weight_sqrt,
                grad_yyd_ids,
                grad_coef_ids,
                grad_feat_ids,
                grad_delay_ids,
            ),
            **params,
        )
        if self.fit_intercept:
            self.coef_ = res.x[: -self.n_outputs_]
            self.intercept_ = res.x[-self.n_outputs_ :]
        else:
            self.coef_ = res.x
            self.intercept_ = np.zeros(self.n_outputs_, dtype=float)
        return self

    @staticmethod
    def _predict(
        expression,
        X,
        y_ref,
        coef,
        intercept,
        feat_ids,
        delay_ids,
        output_ids,
    ):
        n_samples = X.shape[0]
        n_ref, n_outputs = y_ref.shape
        max_delay = np.max(delay_ids)
        y_hat = np.zeros((n_samples, n_outputs), dtype=float)
        at_init = True
        init_k = 0
        for k in range(n_samples):
            if not np.all(np.isfinite(X[k])):
                at_init = True
                init_k = k + 1
                y_hat[k] = np.nan
                continue
            if k - init_k == max_delay:
                at_init = False

            if at_init:
                y_hat[k] = y_ref[k % n_ref]
            else:
                y_hat[k] = intercept
                expression(
                    X,
                    y_hat,
                    y_hat[k],
                    coef,
                    feat_ids,
                    delay_ids,
                    output_ids,
                    k,
                )
            if np.any(y_hat[k] > 1e20):
                y_hat[k:] = 1e20
                return y_hat
        return y_hat

    @staticmethod
    def _get_cfd_ids(feat_ids, delay_ids, output_ids, n_features_in):
        """
        Get ids of CFD (Coef, Feature, and Delay) matrix to update dyn(k)/dx.
        Maps coefficients to their corresponding features and delays.
        """

        # Initialize cfd_ids as a list of lists n_outputs *  n_outputs * max_delay
        # axis-0 (i): [dy0(k)/dx, dy1(k)/dx, ..., dyn(k)/dx]
        # axis-1 (j): [dy0(k-d)/dx, dy1(k-d)/dx, ..., dyn(k-d)/dx]
        # axis-2 (d): [dyj(k-1)/dx, dyj(k-2)/dx, ..., dyj(k-max_delay)/dx]
        grad_yyd_ids = []
        grad_coef_ids = []
        grad_feat_ids = []
        grad_delay_ids = []

        for coef_id, (term_feat_ids, term_delay_ids) in enumerate(
            zip(feat_ids, delay_ids)
        ):
            row_y_id = output_ids[coef_id]  # y(k, id)
            for var_id, (feat_id, delay_id) in enumerate(
                zip(term_feat_ids, term_delay_ids)
            ):
                if feat_id >= n_features_in and delay_id > 0:
                    col_y_id = feat_id - n_features_in  # y(k-1, id)
                    grad_yyd_ids.append([row_y_id, col_y_id, delay_id - 1])
                    grad_coef_ids.append(coef_id)
                    grad_feat_ids.append(np.delete(term_feat_ids, var_id))
                    grad_delay_ids.append(np.delete(term_delay_ids, var_id))

        return (
            np.array(grad_yyd_ids, dtype=np.int32),
            np.array(grad_coef_ids, dtype=np.int32),
            np.array(grad_feat_ids, dtype=np.int32),
            np.array(grad_delay_ids, dtype=np.int32),
        )

    @staticmethod
    def _update_dydx(
        X,
        y_hat,
        coef,
        feat_ids,
        delay_ids,
        output_ids,
        fit_intercept,
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    ):
        """
        Computation of the Jacobian matrix dydx.

        Returns
        -------
        dydx : ndarray of shape (n_samples, n_outputs, n_x)
            Jacobian matrix of the outputs with respect to coefficients and intercepts.
        """
        n_samples, n_y = y_hat.shape
        max_delay = np.max(delay_ids)
        n_coefs = feat_ids.shape[0]
        if fit_intercept:
            n_x = n_coefs + n_y  # Total number of coefficients and intercepts
            y_ids = np.r_[output_ids, np.arange(n_y)]
        else:
            n_x = n_coefs
            y_ids = output_ids

        x_ids = np.arange(n_x)

        dydx = np.zeros((n_samples, n_y, n_x), dtype=float)
        at_init = True
        init_k = 0
        for k in range(n_samples):
            if not np.all(np.isfinite(X[k])):
                at_init = True
                init_k = k + 1
                continue
            if k - init_k == max_delay:
                at_init = False

            if at_init:
                continue
            # Compute terms for time step k
            terms = np.ones(n_x, dtype=float)
            _update_terms(
                X,
                y_hat,
                terms,
                feat_ids,
                delay_ids,
                k,
            )

            # Update constant terms of Jacobian
            dydx[k, y_ids, x_ids] = terms

            # Update dynamic terms of Jacobian
            if max_delay > 0 and grad_yyd_ids.size > 0:
                cfd = np.zeros((n_y, n_y, max_delay), dtype=float)
                _update_cfd(
                    X,
                    y_hat,
                    cfd,
                    coef,
                    grad_yyd_ids,
                    grad_coef_ids,
                    grad_feat_ids,
                    grad_delay_ids,
                    k,
                )
                for d in range(max_delay):
                    dydx[k] += cfd[:, :, d] @ dydx[k - d - 1]

            # Handle divergence
            if np.any(dydx[k] > 1e20):
                dydx[k:] = 1e20
                break

        return dydx

    @staticmethod
    def _loss(
        coef_intercept,
        expression,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        fit_intercept,
        sample_weight_sqrt,
        *args,
    ):
        # Sum of squared errors
        n_outputs = y.shape[1]
        if fit_intercept:
            coef = coef_intercept[:-n_outputs]
            intercept = coef_intercept[-n_outputs:]
        else:
            coef = coef_intercept
            intercept = np.zeros(n_outputs, dtype=float)

        y_hat = NARX._predict(
            expression,
            X,
            y,
            coef,
            intercept,
            feat_ids,
            delay_ids,
            output_ids,
        )

        y_masked, y_hat_masked, sample_weight_sqrt_masked = mask_missing_values(
            y, y_hat, sample_weight_sqrt
        )

        return (sample_weight_sqrt_masked * (y_hat_masked - y_masked)).sum(axis=1)

    @staticmethod
    def _grad(
        coef_intercept,
        expression,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        fit_intercept,
        sample_weight_sqrt,
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    ):
        # Sum of squared errors
        n_outputs = y.shape[1]
        if fit_intercept:
            coef = coef_intercept[:-n_outputs]
            intercept = coef_intercept[-n_outputs:]
        else:
            coef = coef_intercept
            intercept = np.zeros(n_outputs, dtype=float)

        y_hat = NARX._predict(
            expression,
            X,
            y,
            coef,
            intercept,
            feat_ids,
            delay_ids,
            output_ids,
        )
        dydx = NARX._update_dydx(
            X,
            y_hat,
            coef,
            feat_ids,
            delay_ids,
            output_ids,
            fit_intercept,
            grad_yyd_ids,
            grad_coef_ids,
            grad_feat_ids,
            grad_delay_ids,
        )

        mask_valid = mask_missing_values(y, y_hat, sample_weight_sqrt, return_mask=True)

        sample_weight_sqrt_masked = sample_weight_sqrt[mask_valid]
        dydx_masked = dydx[mask_valid]

        return dydx_masked.sum(axis=1) * sample_weight_sqrt_masked

    @validate_params(
        {
            "y_init": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X, y_init=None):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, `n_features_in_`)
            Samples.

        y_init : array-like of shape (n_init, `n_outputs_`), default=None
            The initial values for the prediction of y.
            It should at least have one sample.

        Returns
        -------
        y_hat : array-like of shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            dtype=float,
            order="C",
            reset=False,
            ensure_all_finite="allow-nan",
        )
        if y_init is None:
            y_init = np.zeros((self.max_delay_, self.n_outputs_))
        else:
            y_init = check_array(
                y_init,
                ensure_2d=False,
                dtype=float,
                ensure_min_samples=0,
                ensure_all_finite="allow-nan",
            )
            if y_init.ndim == 1:
                y_init = y_init.reshape(-1, 1)
            if y_init.shape[1] != self.n_outputs_:
                raise ValueError(
                    f"`y_init` should have {self.n_outputs_} outputs "
                    f"but got {y_init.shape[1]}."
                )
        y_hat = NARX._predict(
            _predict_step,
            X,
            y_init,
            self.coef_,
            self.intercept_,
            self.feat_ids_,
            self.delay_ids_,
            self.output_ids_,
        )
        if self.n_outputs_ == 1:
            y_hat = y_hat.flatten()
        return y_hat

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


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
    ========================
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
    print("=" * (yid_space + term_space + coef_space + 4))
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
        "X": ["array-like"],
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
            The number of the selected features dropped for the consequencing
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
    =======================================
    |  0  |     Intercept      |  1.054   |
    |  0  |    y_hat[k-1,0]    |  0.483   |
    |  0  |   X[k,0]*X[k,0]    |  0.307   |
    |  0  | X[k-1,0]*X[k-3,0]  |  1.999   |
    |  0  |  X[k-2,0]*X[k,1]   |  1.527   |
    """
    X = check_array(X, dtype=float, ensure_2d=True, ensure_all_finite="allow-nan")
    y = check_array(y, dtype=float, ensure_2d=False, ensure_all_finite="allow-nan")
    check_consistent_length(X, y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_outputs = y.shape[1]
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

    time_shift_ids_all = np.delete(
        time_shift_ids_all,
        (
            np.isin(time_shift_ids_all[:, 0], static_indices)
            & (time_shift_ids_all[:, 1] > 0)
        ),
        0,
    )
    time_shift_vars = make_time_shift_features(xy_hstack, time_shift_ids_all)

    poly_ids_all = make_poly_ids(
        time_shift_ids_all.shape[0],
        poly_degree,
    )
    poly_terms = make_poly_features(time_shift_vars, poly_ids_all)

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
