"""
NARX model.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT
import warnings
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ..utils import mask_missing_values
from ._feature import (
    _validate_feat_delay_ids,
    fd2tp,
    make_poly_features,
    make_poly_ids,
    make_time_shift_features,
)
from ._narx_fast import _predict, _update_der


class _MemoizeOpt:
    """Cache of residual, jac, and hess during optimization.
    x: parameters
    func: function to compute residual, jac, hess
    flag:
        0: compute loss and grad
        1: compute loss, grad, and hess
    """

    def __init__(self, func, flag):
        self.func = func
        self.flag = flag
        self._residual = None
        self._jac = None
        self._hess = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        if (
            not np.all(x == self.x)
            or self._residual is None
            or self._jac is None
            or (self._hess is None and self.flag == 1)
        ):
            self.x = np.asarray(x).copy()
            self._residual, self._jac, self._hess = self.func(x, *args, self.flag)

    def residual(self, x, *args):
        """R = sqrt(sw) * (y - y_hat)"""
        self._compute_if_needed(x, *args)
        return self._residual

    def jac(self, x, *args):
        """J = sqrt(sw) * dydx"""
        self._compute_if_needed(x, *args)
        return self._jac

    def hess(self, x, *args):
        """Hessian of least squares loss.
        H = J^T @ J + sum(R * sqrt(sw) * d2ydx2)
        H: (n_samples * n_outputs, n_x, n_x)"""
        self._compute_if_needed(x, *args)
        return self._hess

    def loss(self, x, *args):
        """Least squares loss: 0.5 * sum(R^2)"""
        self._compute_if_needed(x, *args)
        return 0.5 * np.sum(np.square(self._residual))

    def grad(self, x, *args):
        """Gradient of least squares loss.
        G = sw * R * dydx =  J^T @ R"""
        self._compute_if_needed(x, *args)
        return np.transpose(self._jac).T @ self._residual


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

    output_ids : array-like of shape (n_terms,), default=None
        The id numbers indicate which output the polynomial term belongs to.
        It is useful in multi-output case.

    fit_intercept : bool, default=True
        Whether to fit the intercept. If set to False, intercept will be zeros.

    Attributes
    ----------
    coef_ : array of shape (n_terms,)
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
    |-----|--------------------|----------|
    |  0  |     Intercept      |  1.008   |
    |  0  |    y_hat[k-1,0]    |  0.498   |
    |  0  |      X[k-2,0]      |  0.701   |
    |  0  | X[k-1,0]*X[k-3,0]  |  1.496   |
    """

    # Notify type checker ty
    n_features_in_: int

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
            "X": [None, "array-like"],
            "coef_init": [None, StrOptions({"one_step_ahead"}), "array-like"],
            "sample_weight": [None, "array-like"],
            "session_sizes": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(
        self, X, y, sample_weight=None, coef_init=None, session_sizes=None, **params
    ):
        """
        Fit narx model.

        Parameters
        ----------
        X : array-like of shape (n_samples, `n_features_in_`) or None
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

        session_sizes : array-like of shape (n_sessions,), default=None
            The sizes of measurement sessions for time-series.
            The sum of session_sizes should be equal to n_samples.
            If None, the whole data is treated as one session.

            .. versionadded:: 0.5

        **params : dict
            Keyword arguments passed to
            `scipy.optimize.least_squares`.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        none_inputs = False
        if X is None:  # Auto-regressive model
            X = np.empty((1, 0), dtype=float, order="C")  # Skip validation
            none_inputs = True
        check_X_params = dict(
            dtype=float, order="C", ensure_all_finite="allow-nan", ensure_min_features=0
        )
        check_y_params = dict(
            ensure_2d=False, dtype=float, order="C", ensure_all_finite="allow-nan"
        )
        X, y = validate_data(
            self, X, y, validate_separately=(check_X_params, check_y_params)
        )
        if none_inputs:
            X = np.empty((len(y), 0), dtype=float, order="C")  # Create 0 feature input
        else:
            check_consistent_length(X, y)
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._y_ndim = y.ndim
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs_ = y.shape[1]
        n_samples, n_features = X.shape

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, ensure_non_negative=True
        )

        session_sizes_cumsum = _validate_session_sizes(session_sizes, n_samples)

        if self.feat_ids is None:
            if n_features == 0:
                feat_ids_ = make_poly_ids(self.n_outputs_, 1) - 1
            else:
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
            poly_terms = _prepare_poly_terms(
                xy_hstack,
                time_shift_ids,
                poly_ids,
                session_sizes_cumsum,
                self.max_delay_,
            )
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

        flag = 0
        if flag == 0:
            need_hess = False
        else:
            need_hess = True

        grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_jc_ids(
            self.feat_ids_, self.delay_ids_, self.output_ids_, X.shape[1]
        )
        (hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_feat_ids, hess_delay_ids) = (
            NARX._get_hc_ids(
                grad_yyd_ids,
                grad_coef_ids,
                grad_feat_ids,
                grad_delay_ids,
                need_hess=need_hess,
            )
        )
        combined_term_ids, unique_feat_ids, unique_delay_ids = NARX._get_term_ids(
            np.vstack([self.feat_ids_, grad_feat_ids, hess_feat_ids]),
            np.vstack([self.delay_ids_, grad_delay_ids, hess_delay_ids]),
        )
        const_term_ids = combined_term_ids[:n_terms]
        grad_term_ids = combined_term_ids[n_terms:]
        n_grad = grad_feat_ids.shape[0]
        grad_term_ids = combined_term_ids[n_terms : n_terms + n_grad]
        hess_term_ids = combined_term_ids[n_terms + n_grad :]
        sample_weight_sqrt = np.sqrt(sample_weight).reshape(-1, 1)
        if self.fit_intercept:
            y_ids = np.r_[self.output_ids_, np.arange(self.n_outputs_, dtype=np.int32)]
        else:
            y_ids = self.output_ids_
        memoize_opt = _MemoizeOpt(NARX._func, flag=flag)
        res = least_squares(
            memoize_opt.residual,
            x0=coef_init,
            jac=memoize_opt.jac,
            args=(
                X,
                y,
                self.feat_ids_,
                self.delay_ids_,
                self.output_ids_,
                self.fit_intercept,
                sample_weight_sqrt,
                session_sizes_cumsum,
                self.max_delay_,
                y_ids,
                grad_yyd_ids,
                grad_coef_ids,
                unique_feat_ids,
                unique_delay_ids,
                const_term_ids,
                grad_term_ids,
                hess_yyd_ids,
                hess_coef_ids,
                hess_term_ids,
                hess_yd_ids,
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
    def _get_term_ids(feat_ids, delay_ids):
        """
        Get unique ids of terms.
        Returns:
            term_ids: array-like of shape (n_terms,)
                The indices of unique terms for each original term.
            unique_feat_ids: array-like of shape (n_unique, degree)
                The unique feat_ids of terms.
            unique_delay_ids: array-like of shape (n_unique, degree)
                The unique delay_ids of terms.
        """
        combined_ids = np.hstack([feat_ids, delay_ids])
        unique_combined_ids, term_ids = np.unique(
            combined_ids, axis=0, return_inverse=True
        )
        n_degree = feat_ids.shape[1]
        unique_feat_ids = np.ascontiguousarray(unique_combined_ids[:, :n_degree])
        unique_delay_ids = np.ascontiguousarray(unique_combined_ids[:, n_degree:])
        term_ids = np.ascontiguousarray(term_ids, dtype=np.int32)
        return term_ids, unique_feat_ids, unique_delay_ids

    @staticmethod
    def _get_hc_ids(
        grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids, need_hess
    ):
        """
        Get ids of HC (Hessian Coefficient) matrix to update d2yi(k)/dx2.
        HC matrix has shape in (n_x, max_delay (in), n_outputs (out), n_outputs (in)).
        d2ydx2 has shape in (n_samples, n_x, n_outputs (out), n_x).
        JC matrix has shape in (max_delay (in), n_outputs (out), n_outputs (in)).
        dydx has shape in (n_samples, n_outputs (out), n_x).
        The updating rule is given by:
        d2ydx2[k, i] += JC[d] @ d2ydx2[k-d-1, i]
        d2ydx2[k, i] += HC[i, d] @ dydx[k-d-1] for d in range(0, max_delay)
        d2ydx2[k, i] += Constant terms
        """
        n_degree = grad_feat_ids.shape[1]
        hess_yyd_ids = np.zeros((0, 3), dtype=np.int32)
        hess_yd_ids = np.zeros((0, 2), dtype=np.int32)
        hess_coef_ids = np.zeros(0, dtype=np.int32)
        hess_feat_ids = np.zeros((0, n_degree), dtype=np.int32)
        hess_delay_ids = np.zeros((0, n_degree), dtype=np.int32)

        if not need_hess:
            return (
                hess_yyd_ids,
                hess_yd_ids,
                hess_coef_ids,
                hess_feat_ids,
                hess_delay_ids,
            )

        for yyd_id, coef_id, feat_ids, delay_ids in zip(
            grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids
        ):
            # In jac, x * term will generate a term with coef 1.
            # In hess, it will be
            # d term / dx = 1 * d y_in * yi * yj
            # hess_yyd_ids = np.vstack([hess_yyd_ids, yyd_id])
            # hess_yd_ids = np.vstack([hess_yd_ids, [-1, -1]])  # empty
            # hess_coef_ids = np.append(hess_coef_ids, [-1])  # constant 1
            # hess_feat_ids = np.vstack([hess_feat_ids, feat_ids])
            # hess_delay_ids = np.vstack([hess_delay_ids, delay_ids])
            for var_id, (feat_id, delay_id) in enumerate(zip(feat_ids, delay_ids)):
                #  d JC / dx = coef * d y_in * d yi * yj
                # hess_yyd_ids: y_out and y_in
                # hess_coef_ids: coef
                # hess_feat_ids: yj ..
                # hess_delay_ids: yj ..
                hess_yyd_ids = np.vstack([hess_yyd_ids, yyd_id])
                # feat_ids and delay_ids contain spaceholder -1 to keep poly_degree size
                # so d term / dx = 1 * d y_in * yi * yj will automatically included in
                # the following operations.
                hess_yd_ids = np.vstack([hess_yd_ids, [feat_id, delay_id]])
                # If yd_ids is empty, [-1, -1], then coef should be constant 1.
                # This will be handled in _update_hc
                hess_coef_ids = np.append(hess_coef_ids, coef_id)
                hess_feat_ids = np.vstack([hess_feat_ids, feat_ids])
                hess_delay_ids = np.vstack([hess_delay_ids, delay_ids])
                hess_feat_ids[-1][var_id] = -1
                hess_delay_ids[-1][var_id] = -1

        return (
            hess_yyd_ids.astype(np.int32),
            hess_yd_ids.astype(np.int32),
            hess_coef_ids.astype(np.int32),
            hess_feat_ids.astype(np.int32),
            hess_delay_ids.astype(np.int32),
        )

    @staticmethod
    def _get_jc_ids(feat_ids, delay_ids, output_ids, n_features_in):
        """
        Get ids of JC (Jacobian Coefficient) matrix to update dyi(k)/dx.
        JC matrix has shape in (max_delay (in), n_outputs (out), n_outputs (in)).
        dydx has shape in (n_samples, n_outputs (out), n_x).
        The updating rule is given by:
        dydx[k] = terms
        dydx[k] += JC[d] @ dydx[k-d-1] for d in range(0, max_delay)

        Denote dyi(k)/dx as the derivative of the ith output with respect to x.
        dyi(k)/dx should be a linear combination of dyj(k-1)/dx, dyj(k-2)/dx, ...,
        dyj(k-max_delay)/dx, with some coefficients, which are the polynomials of
        y and u. We call dyi/dx and dyj/dx as output y and input y, respectively.

        Therefore, to update dyi(k)/dx, we need to know which output to update,
        that is index i; and which element contributes to it, that is index
        j and delay d; and the polynomial formula of the corresponding coefficient,
        so we need grad_delay_ids, grad_coef_ids, and grad_feat_ids. The i, j, d will
        be saved in grad_yyd_ids. It should be noted that for different parameters,
        i.e., x, these ids are the same.

        The JC matrix stores the coefficients, which computed by grad_*_ids. The
        locations of the coefficients are specified by grad_yyd_ids.
        axis-0 (d) delay of input y: dyj(k-1)/dx, dyj(k-2)/dx, ..., dyj(k-max_delay)/dx
        axis-1 (i) output y: dy0(k)/dx, dy1(k)/dx, ..., dyn(k)/dx
        axis-2 (j) input y: dy0(k-d)/dx, dy1(k-d)/dx, ..., dyn(k-d)/dx
        It should be noted that delay 1 in axis-0 is at location 0, so we do
        `delay_id - 1` in grad_yyd_ids.
        """

        n_degree = feat_ids.shape[1]
        grad_yyd_ids = np.zeros((0, 3), dtype=np.int32)
        grad_coef_ids = np.zeros(0, dtype=int)
        grad_feat_ids = np.zeros((0, n_degree), dtype=np.int32)
        grad_delay_ids = np.zeros((0, n_degree), dtype=np.int32)

        for coef_id, (term_feat_ids, term_delay_ids) in enumerate(
            zip(feat_ids, delay_ids)
        ):
            out_y_id = output_ids[coef_id]  # y(k, id), output
            for var_id, (feat_id, delay_id) in enumerate(
                zip(term_feat_ids, term_delay_ids)
            ):
                if feat_id >= n_features_in and delay_id > 0:
                    in_y_id = feat_id - n_features_in  # y(k-d, id), input
                    grad_yyd_ids = np.vstack(
                        [grad_yyd_ids, [out_y_id, in_y_id, delay_id - 1]]
                    )
                    grad_coef_ids = np.append(grad_coef_ids, coef_id)
                    grad_feat_ids = np.vstack([grad_feat_ids, term_feat_ids])
                    grad_delay_ids = np.vstack([grad_delay_ids, term_delay_ids])
                    grad_feat_ids[-1][var_id] = -1
                    grad_delay_ids[-1][var_id] = -1

        return (
            grad_yyd_ids.astype(np.int32),
            grad_coef_ids.astype(np.int32),
            grad_feat_ids.astype(np.int32),
            grad_delay_ids.astype(np.int32),
        )

    @staticmethod
    def _func(
        coef_intercept,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        fit_intercept,
        sample_weight_sqrt,
        session_sizes_cumsum,
        max_delay,
        y_ids,
        grad_yyd_ids,
        grad_coef_ids,
        unique_feat_ids,
        unique_delay_ids,
        const_term_ids,
        grad_term_ids,
        hess_yyd_ids,
        hess_coef_ids,
        hess_term_ids,
        hess_yd_ids,
        flag,
    ):
        """
        Compute residual, Jacobian, and optionally Hessian.

        Parameters
        ----------
        flag : int
            0: compute residual and Jacobian
            1: compute residual, Jacobian, and Hessian

        Returns
        -------
        residual : array of shape (n_samples,)
        jac : array of shape (n_samples, n_x)
        hess : None or array
            None when flag=0, hessian array when flag=1
        """
        n_samples, n_outputs = y.shape
        n_x = coef_intercept.shape[0]
        if fit_intercept:
            coef = coef_intercept[:-n_outputs]
            intercept = coef_intercept[-n_outputs:]
        else:
            coef = coef_intercept
            intercept = np.zeros(n_outputs, dtype=float)

        # Compute prediction
        y_hat = np.zeros((n_samples, n_outputs), dtype=float)
        _predict(
            X,
            y,
            coef,
            intercept,
            feat_ids,
            delay_ids,
            output_ids,
            session_sizes_cumsum,
            max_delay,
            y_hat,
        )

        # Compute Jacobian
        dydx = np.zeros((n_samples, n_outputs, n_x), dtype=float)
        jc = np.zeros((max_delay, n_outputs, n_outputs), dtype=float)

        term_libs = np.ones((n_samples, unique_feat_ids.shape[0]), dtype=float)
        hc = np.zeros((n_x, max_delay, n_outputs, n_outputs), dtype=float)
        d2ydx2 = np.zeros((n_samples, n_x, n_outputs, n_x), dtype=float)

        _update_der(
            X,
            y_hat,
            max_delay,
            session_sizes_cumsum,
            flag,
            y_ids.astype(np.int32),
            coef,
            unique_feat_ids,
            unique_delay_ids,
            const_term_ids,
            grad_yyd_ids,
            grad_coef_ids,
            grad_term_ids,
            hess_yyd_ids,
            hess_coef_ids,
            hess_term_ids,
            hess_yd_ids,
            term_libs,
            jc,
            hc,
            dydx,
            d2ydx2,
        )

        # Mask missing values
        mask_valid = mask_missing_values(y, y_hat, sample_weight_sqrt, return_mask=True)
        y_masked = y[mask_valid]
        y_hat_masked = y_hat[mask_valid]
        sample_weight_sqrt_masked = sample_weight_sqrt[mask_valid]
        dydx_masked = dydx[mask_valid]

        # Compute residuals
        residual = (sample_weight_sqrt_masked * (y_hat_masked - y_masked)).flatten()

        # Compute Jacobian
        # jac (n_samples, n_outputs (out), n_x) to (n_samples * n_outputs, n_x)
        jac = np.reshape(
            dydx_masked * sample_weight_sqrt_masked[..., np.newaxis], (-1, n_x)
        )

        # Compute Hessian if needed
        if flag == 1:
            # d2ydx2 has shape in (n_samples, n_x, n_outputs (out), n_x)
            d2ydx2_masked = (
                d2ydx2[mask_valid]
                * sample_weight_sqrt_masked[..., np.newaxis, np.newaxis]
            )
            # reshape to (n_samples, n_outputs (out), n_x, n_x)
            d2ydx2_masked = d2ydx2_masked.swapaxes(1, 2)
            # reshape to (n_samples * n_outputs, n_x, n_x)
            d2ydx2_masked = np.reshape(d2ydx2_masked, (-1, n_x, n_x))
            hess = jac.T @ jac + np.tensordot(residual, d2ydx2_masked, axes=1)
            hess = 0.5 * (hess + hess.T)  # force symmetric
        else:
            hess = None

        # residual: (n_samples * n_outputs,)
        # jac: (n_samples * n_outputs, n_x)
        # hess: (n_x, n_x)
        return residual, jac, hess

    @validate_params(
        {
            "X": ["array-like", Interval(Integral, 1, None, closed="left")],
            "y_init": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X, y_init=None):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, `n_features_in_`) or int
            When X is an array, it is input data.
            When (nonlinear) AR model is adopted, where `n_features_in_` is 0,
            X can be an integer, which indicates the total steps to predict.

        y_init : array-like of shape (n_init,) or (n_init, `n_outputs_`), default=None
            The initial values for the prediction of y.
            It should at least have one sample.

        Returns
        -------
        y_hat : array-like of shape (n_samples,) or (n_samples, `n_outputs_`)
            Returns predicted values. The number of dimensions is the same as that
            of y in :term:`fit`.
        """
        check_is_fitted(self)

        if isinstance(X, Integral):
            if self.n_features_in_ == 0:
                X = np.empty((X, 0), dtype=float, order="C")
            else:
                raise ValueError(
                    "X should be an array-like of shape (n_samples, n_features_in_) "
                    f"but got an integer {X}, when `n_features_in_` is not 0."
                )
        else:
            X = validate_data(
                self,
                X,
                dtype=float,
                order="C",
                reset=False,
                ensure_all_finite="allow-nan",
                ensure_min_features=0,
            )
        if y_init is None:
            y_init = np.zeros((self.max_delay_, self.n_outputs_))
        else:
            y_init = check_array(
                y_init,
                ensure_2d=False,
                dtype=float,
                order="C",
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
        n_samples = X.shape[0]
        y_hat = np.zeros((n_samples, self.n_outputs_), dtype=float)
        session_sizes_cumsum = np.array([n_samples], dtype=np.int32)
        _predict(
            X,
            y_init,
            self.coef_,
            self.intercept_,
            self.feat_ids_,
            self.delay_ids_,
            self.output_ids_,
            session_sizes_cumsum,
            self.max_delay_,
            y_hat,
        )
        if self._y_ndim == 1:
            y_hat = y_hat.flatten()
        return y_hat

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _validate_session_sizes(session_sizes, n_samples):
    if session_sizes is None:
        return np.array([n_samples], dtype=np.int32)
    session_sizes = column_or_1d(
        session_sizes,
        dtype=np.int32,
        warn=True,
    )
    if (session_sizes <= 0).any():
        raise ValueError(
            "All elements of session_sizes should be positive, "
            f"but got {session_sizes}."
        )
    if session_sizes.sum() != n_samples:
        raise ValueError(
            "The sum of session_sizes should be equal to n_samples, "
            f"but got {session_sizes.sum()} != {n_samples}."
        )
    return np.cumsum(session_sizes, dtype=np.int32)


def _prepare_poly_terms(
    xy_hstack, time_shift_ids, poly_ids, session_sizes_cumsum, max_delay
):
    time_shift_vars = make_time_shift_features(xy_hstack, time_shift_ids)
    for start in session_sizes_cumsum[:-1]:
        time_shift_vars[start : start + max_delay] = np.nan
    poly_terms = make_poly_features(time_shift_vars, poly_ids)
    return poly_terms
