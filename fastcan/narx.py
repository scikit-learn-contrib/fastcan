"""
The module related to nonlinear autoregressive exogenous (NARX) model for system
identification.
"""

import math
from itertools import combinations_with_replacement
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ._fastcan import FastCan
from ._refine import refine


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
    return np.delete(ids, const_id, 0)  # remove the constant featrue


def _mask_missing_value(*arr):
    """Remove missing value for all arrays."""
    mask_nomissing = np.all(np.isfinite(np.c_[arr]), axis=1)
    return tuple([x[mask_nomissing] for x in arr])


class NARX(RegressorMixin, BaseEstimator):
    """The Nonlinear Autoregressive eXogenous (NARX) model class.
    For example, a (polynomial) NARX model is like
    y(t) = y(t-1)*u(t-1) + u(t-1)^2 + u(t-2) + 1.5
    where y(t) is the system output at time t,
    u(t) is the system input at time t,
    u(t-1) is called a (time shift) variable, and
    u(t-1)^2 is called a (polynomial) term.

    Parameters
    ----------
    time_shift_ids : array-like of shape (n_variables, 2), default=None
        The unique id numbers of time shift variables, which are
        (feature_idx, delay). The ids are used to generate time
        shift variables, such as u(k-1), and y(k-2).

    poly_ids : array-like of shape (n_polys, degree), default=None
        The unique id numbers of polynomial terms, excluding the intercept.
        Here n_terms = n_polys + 1 (intercept).
        The ids are used to generate polynomial terms, such as u(k-1)^2,
        and u(k-1)*y(k-2).

    Attributes
    ----------
    coef_ : array of shape (`n_features_in_`,)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    expression_ : str
        The lambda exprssion of the model in the string form.

    max_delay_ : int
        The maximum time delay of the time shift variables.

    time_shift_ids_ : array-like of shape (n_variables, 2)
        The unique id numbers of time shift variables, which are
        (feature_idx, delay).

    poly_ids_ : array-like of shape (n_polys, degree)
        The unique id numbers of polynomial terms, excluding the intercept.

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
    >>> time_shift_ids = [[0, 1], # u(k-1)
    ...                   [0, 2], # u(k-2)
    ...                   [0, 3], # u(k-3)
    ...                   [1, 1]] # y(k-1)
    >>> poly_ids = [[0, 2], # 1*u(k-1)
    ...             [0, 4], # 1*y(k-1)
    ...             [1, 3]] # u(k-1)*u(k-3)
    >>> narx = NARX(time_shift_ids=time_shift_ids,
    ...             poly_ids=poly_ids).fit(X, y, coef_init="one_step_ahead")
    >>> print_narx(narx)
    |        Term        |   Coef   |
    =================================
    |     Intercept      |  1.008   |
    |      X[k-2,0]      |  0.701   |
    |     y_hat[k-1]     |  0.498   |
    | X[k-1,0]*X[k-3,0]  |  1.496   |
    """

    _parameter_constraints: dict = {
        "time_shift_ids": [None, "array-like"],
        "poly_ids": [None, "array-like"],
    }

    def __init__(
        self,
        *,  # keyword call only
        time_shift_ids=None,
        poly_ids=None,
    ):
        self.time_shift_ids = time_shift_ids
        self.poly_ids = poly_ids

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

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample, which are used for a One-Step-Ahead
            NARX.

        coef_init : array-like of shape (n_terms,), default=None
            The initial values of coefficients and intercept for optimization.
            When `coef_init` is None, the model will be a One-Step-Ahead NARX.
            When `coef_init` is `one_step_ahead`, the model will be a Multi-Step-Ahead
            NARX whose coefficients and intercept are initialized by the a
            One-Step-Ahead NARX.
            When `coef_init` is an array, the model will be a Multi-Step-Ahead
            NARX whose coefficients and intercept are initialized by the array.

            .. note::
                When coef_init is None, missing values (i.e., np.nan) are allowed.

        **params : dict
            Keyword arguments passed to
            `scipy.optimize.least_squares`.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        X = validate_data(
            self,
            X,
            dtype=float,
            ensure_all_finite="allow-nan",
        )
        y = column_or_1d(y, dtype=float, warn=True)
        check_consistent_length(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)

        if self.time_shift_ids is None:
            self.time_shift_ids_ = make_time_shift_ids(
                n_features=X.shape[1],
                max_delay=0,
                include_zero_delay=True,
            )
        else:
            self.time_shift_ids_ = check_array(
                self.time_shift_ids,
                ensure_2d=True,
                dtype=Integral,
            )
            if (self.time_shift_ids_[:, 0].min() < 0) or (
                self.time_shift_ids_[:, 0].max() >= X.shape[1] + 1
            ):
                raise ValueError(
                    "The element x of the first column of time_shift_ids should "
                    f"satisfy 0 <= x < {X.shape[1]+1}."
                )
            if (self.time_shift_ids_[:, 1].min() < 0) or (
                self.time_shift_ids_[:, 1].max() >= X.shape[0]
            ):
                raise ValueError(
                    "The element x of the second column of time_shift_ids should "
                    f"satisfy 0 <= x < {X.shape[0]}."
                )

        if self.poly_ids is None:
            self.poly_ids_ = make_poly_ids(X.shape[1], 1)
        else:
            self.poly_ids_ = check_array(
                self.poly_ids,
                ensure_2d=True,
                dtype=Integral,
            )
            if (self.poly_ids_.min() < 0) or (
                self.poly_ids_.max() > self.time_shift_ids_.shape[0]
            ):
                raise ValueError(
                    "The element x of poly_ids should "
                    f"satisfy 0 <= x <= {self.time_shift_ids_.shape[0]}."
                )

        self.max_delay_ = self.time_shift_ids_[:, 1].max()
        n_terms = self.poly_ids_.shape[0] + 1

        if isinstance(coef_init, (type(None), str)):
            # fit a one-step-ahead NARX model
            xy_hstack = np.c_[X, y]
            osa_narx = LinearRegression()
            time_shift_vars = make_time_shift_features(xy_hstack, self.time_shift_ids_)
            poly_terms = make_poly_features(time_shift_vars, self.poly_ids_)
            # Remove missing values
            poly_terms_masked, y_masked, sample_weight_masked = _mask_missing_value(
                poly_terms, y, sample_weight
            )

            osa_narx.fit(poly_terms_masked, y_masked, sample_weight_masked)
            if coef_init is None:
                self.coef_ = osa_narx.coef_
                self.intercept_ = osa_narx.intercept_
                return self

            coef_init = np.r_[osa_narx.coef_, osa_narx.intercept_]
        else:
            coef_init = check_array(
                coef_init,
                ensure_2d=False,
                dtype=np.float64,
            )
            if coef_init.shape[0] != n_terms:
                raise ValueError(
                    "`coef_init` should have the shape of "
                    f"(`n_terms`,), i.e., ({n_terms,}), "
                    f"but got {coef_init.shape}."
                )

        lsq = least_squares(
            NARX._residual,
            x0=coef_init,
            args=(
                self._expression,
                X,
                y,
                self.max_delay_,
            ),
            **params,
        )
        self.coef_ = lsq.x[:-1]
        self.intercept_ = lsq.x[-1]
        return self

    def _get_variable(self, time_shift_id, X, y_hat, k):
        if time_shift_id[0] < self.n_features_in_:
            variable = X[k - time_shift_id[1], time_shift_id[0]]
        else:
            variable = y_hat[k - time_shift_id[1]]
        return variable

    def _get_term(self, term_id, X, y_hat, k):
        term = 1
        for _, variable_id in enumerate(term_id):
            if variable_id != 0:
                time_shift_id = self.time_shift_ids_[variable_id - 1]
                term *= self._get_variable(time_shift_id, X, y_hat, k)
        return term

    def _expression(self, X, y_hat, coef, intercept, k):
        y_pred = intercept
        for i, term_id in enumerate(self.poly_ids_):
            y_pred += coef[i] * self._get_term(term_id, X, y_hat, k)
        return y_pred

    @staticmethod
    def _predict(expression, X, y_ref, coef, intercept, max_delay):
        n_samples = X.shape[0]
        n_ref = len(y_ref)
        y_hat = np.zeros(n_samples)
        at_init = True
        init_k = 0
        for k in range(n_samples):
            if ~np.all(np.isfinite(X[k])):
                at_init = True
                init_k = k + 1
                y_hat[k] = np.nan
                continue
            if k - init_k == max_delay:
                at_init = False

            if at_init:
                y_hat[k] = y_ref[k % n_ref]
            else:
                y_hat[k] = expression(X, y_hat, coef, intercept, k)
            if np.any(y_hat[k] > 1e20):
                y_hat[k:] = 1e20
                return y_hat
        return y_hat

    @staticmethod
    def _residual(
        coef_intercept,
        expression,
        X,
        y,
        max_delay,
    ):
        coef = coef_intercept[:-1]
        intercept = coef_intercept[-1]

        y_hat = NARX._predict(expression, X, y, coef, intercept, max_delay)

        y_masked, y_hat_masked = _mask_missing_value(y, y_hat)

        return y_masked - y_hat_masked

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

        y_init : array-like of shape (`n_init`,), default=None
            The initial values for the prediction of y.
            It should at least have one sample.

        Returns
        -------
        y_hat : array-like of shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        if y_init is None:
            y_init = np.zeros(self.max_delay_)
        else:
            y_init = column_or_1d(y_init, dtype=float)
            if y_init.shape[0] < 1:
                raise ValueError(
                    "`y_init` should at least have one sample "
                    f"but got {y_init.shape}."
                )

        return NARX._predict(
            self._expression,
            X,
            y_init,
            self.coef_,
            self.intercept_,
            self.max_delay_,
        )

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
    """Print a NARX model as a Table which contains Term and Coef.

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
        The table of terms and coefficients of the NARX model.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from fastcan.narx import print_narx, NARX
    >>> X, y = load_diabetes(return_X_y=True)
    >>> print_narx(NARX().fit(X, y), term_space=10, coef_space=5, float_precision=0)
    |   Term   |Coef |
    ==================
    |Intercept | 152 |
    | X[k-0,0] | -10 |
    | X[k-0,1] |-240 |
    | X[k-0,2] | 520 |
    | X[k-0,3] | 324 |
    | X[k-0,4] |-792 |
    | X[k-0,5] | 477 |
    | X[k-0,6] | 101 |
    | X[k-0,7] | 177 |
    | X[k-0,8] | 751 |
    | X[k-0,9] | 68  |
    """
    check_is_fitted(narx)

    def _get_variable_str(time_shift_id):
        if time_shift_id[0] < narx.n_features_in_:
            variable_str = f"X[k-{time_shift_id[1]},{time_shift_id[0]}]"
        else:
            variable_str = f"y_hat[k-{time_shift_id[1]}]"
        return variable_str

    def _get_term_str(term_id):
        term_str = ""
        for _, variable_id in enumerate(term_id):
            if variable_id != 0:
                time_shift_id = narx.time_shift_ids_[variable_id - 1]
                term_str += "*" + _get_variable_str(time_shift_id)
        return term_str[1:]

    print(f"|{'Term':^{term_space}}" + f"|{'Coef':^{coef_space}}|")
    print("=" * (term_space + coef_space + 3))
    print(
        f"|{'Intercept':^{term_space}}|"
        + f"{narx.intercept_:^{coef_space}.{float_precision}f}|"
    )
    for i, term_id in enumerate(narx.poly_ids_):
        print(
            f"|{_get_term_str(term_id):^{term_space}}|"
            + f"{narx.coef_[i]:^{coef_space}.{float_precision}f}|"
        )


@validate_params(
    {
        "X": ["array-like"],
        "y": ["array-like"],
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_delay": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "poly_degree": [
            Interval(Integral, 1, None, closed="left"),
        ],
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
    n_features_to_select,
    max_delay=1,
    poly_degree=1,
    *,
    include_zero_delay=None,
    static_indices=None,
    refine_verbose=1,
    refine_drop=None,
    refine_max_iter=None,
    **params,
):
    """Find `time_shift_ids` and `poly_ids` for a NARX model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    n_features_to_select : int
        The parameter is the absolute number of features to select.

    max_delay : int, default=1
        The maximum delay of time shift features.

    poly_degree : int, default=1
        The maximum degree of polynomial features.

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
    ...     n_features_to_select=4,
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
    |        Term        |   Coef   |
    =================================
    |     Intercept      |  1.054   |
    |     y_hat[k-1]     |  0.483   |
    | X[k-0,0]*X[k-0,0]  |  0.307   |
    | X[k-1,0]*X[k-3,0]  |  1.999   |
    | X[k-2,0]*X[k-0,1]  |  1.527   |
    """
    X = check_array(X, dtype=float, ensure_2d=True, ensure_all_finite="allow-nan")
    y = column_or_1d(y, dtype=float)
    check_consistent_length(X, y)

    xy_hstack = np.c_[X, y]
    n_features = X.shape[1]

    if include_zero_delay is None:
        _include_zero_delay = [True] * n_features + [False]
    else:
        _include_zero_delay = include_zero_delay + [False]

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
    poly_terms_masked, y_masked = _mask_missing_value(poly_terms, y)

    csf = FastCan(
        n_features_to_select,
        **params,
    ).fit(poly_terms_masked, y_masked)
    if refine_drop is not None:
        indices, _ = refine(
            csf, drop=refine_drop, max_iter=refine_max_iter, verbose=refine_verbose
        )
        support = np.zeros(shape=csf.n_features_in_, dtype=bool)
        support[indices] = True
    else:
        support = csf.get_support()
    selected_poly_ids = poly_ids_all[support]
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

    return NARX(time_shift_ids=time_shift_ids, poly_ids=poly_ids)
