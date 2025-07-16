"""
NARX model.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._param_validation import StrOptions, validate_params
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
from ._narx_fast import (  # type: ignore[attr-defined]
    _predict_step,
    _update_cfd,
    _update_terms,
)


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
        X : {array-like, sparse matrix} of shape (n_samples, `n_features_in_`) or None
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
        check_X_params = dict(
            dtype=float, order="C", ensure_all_finite="allow-nan", ensure_min_features=0
        )
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
            ensure_min_features=0,
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
