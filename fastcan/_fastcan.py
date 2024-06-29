"""
Feature selection
"""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import check_array
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ._cancorr_fast import _forward_search  # type: ignore


class FastCan(SelectorMixin, BaseEstimator):
    """Forward feature selector according to the sum of squared
    canonical correlation coefficients (SSC).

    .. note::

        The numpy data types used for Cython can be found in
        |numpy_dtype|_ and |sklearn_cython_dtype|_.

    Parameters
    ----------
    n_features_to_select : int
        The parameter is the absolute number of features to select.

    inclusive_indices : array-like of shape (n_inclusions,), default=None
        The indices of the prerequisite features.

    eta : bool, default=None
        Whether to use eta-cosine method.

    tol : float, default=0.01
        Tolerance for linear dependence check.

        When abs(w.T*x) > `tol`, the modified Gram-Schmidt is failed as
        the feature `x` is linear dependent to the selected features,
        and `mask` for that feature will True.

    verbose : int, default=1
        The verbosity level.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    scores_: ndarray of shape (n_features_to_select,), dtype=float
        The h-correlation/eta-cosine of selected features.

    References
    ----------
    * Zhang, S., & Lang, Z. Q. (2022).
        Orthogonal least squares based fast feature selection for
        linear classification. Pattern Recognition, 123, 108419.

    * Zhang, S., Wang, T., Sun L., Worden, K., & Cross, E. J. (2024).
        Canonical-correlation-based fast feature selection for
        structural health monitoring.

    Examples
    --------
    >>> from fastcan import FastCan
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...     [-2.79, -0.02, -0.85 ],
    ...     [-1.34, -0.48, -2.55 ],
    ...     [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = FastCan(n_features_to_select=2, verbose=0).fit(X, y)
    >>> selector.get_support()
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "inclusive_indices": [None, "array-like"],
        "eta": ["boolean", None],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_features_to_select,
        inclusive_indices=None,
        eta=None,
        tol=0.01,
        verbose=1,
    ):
        self.n_features_to_select = n_features_to_select
        self.inclusive_indices = inclusive_indices
        self.eta = eta
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Preprare data for h-correlation or eta-cosine methods and select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples, n_outputs)
            Target matrix.


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()
        # X y
        check_X_params = {"order": "F"}
        check_y_params = {"ensure_2d": False, "order": "F"}
        X, y = self._validate_data(
            X=X,
            y=y,
            multi_output=True,
            validate_separately=(check_X_params, check_y_params),
        )
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = y.reshape(-1, 1)

        # inclusive_indices
        if self.inclusive_indices is None:
            self.inclusive_indices = np.zeros(0, dtype=int)
        else:
            self.inclusive_indices = check_array(
                self.inclusive_indices,
                ensure_2d=False,
                dtype=int,
                ensure_min_samples=0,
            )

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        if self.n_features_to_select >= n_features:
            raise ValueError(
                f"n_features_to_select {self.n_features_to_select} "
                f"must be < n_features {n_features}."
            )

        if self.inclusive_indices.shape[0] >= n_features:
            raise ValueError(
                f"n_inclusions {self.inclusive_indices.shape[0]} must "
                f"be < n_features {n_features}."
            )

        # Method determination
        if self.eta is None:
            if n_samples > n_features + n_outputs:
                self.eta = True
            else:
                self.eta = False
        if n_samples < n_features + n_outputs and self.eta:
            raise ValueError(
                "`eta` cannot be True, when n_samples < n_features+n_outputs."
            )

        if self.eta:
            xy_hstack = np.hstack((X, y))
            xy_centered = xy_hstack - xy_hstack.mean(0)
            singular_values, unitary_arrays = np.linalg.svd(
                xy_centered, full_matrices=False
            )[1:]
            qxy_transformed = singular_values.reshape(-1, 1) * unitary_arrays
            qxy_transformed = np.asfortranarray(qxy_transformed)
            X_transformed = qxy_transformed[:, :n_features]
            y_transformed = qxy_transformed[:, n_features:]
        else:
            X_transformed = X - X.mean(0)
            y_transformed = y - y.mean(0)

        mask, indices, scores = self._prepare_data(
            self.inclusive_indices,
        )
        n_threads = _openmp_effective_n_threads()
        _forward_search(
            X=X_transformed,
            V=y_transformed,
            t=self.n_features_to_select,
            tol=self.tol,
            num_threads=n_threads,
            verbose=self.verbose,
            mask=mask,
            indices=indices,
            scores=scores,
        )
        if -1 in indices:
            raise RuntimeError("The selection is interrupted by error!!!")
        if self.verbose == 1:
            print()
        support = np.zeros(shape=self.n_features_in_, dtype=bool)
        support[indices] = True
        self.support_ = support
        self.scores_ = scores
        return self

    def _prepare_data(self, inclusive_indices):
        """Prepare data for _forward_search()
        When h-correlation method is used, n_samples_ = n_samples.
        When eta-cosine method is used, n_samples_ = n_features+n_outputs.

        Parameters
        ----------
        inclusive_indices : array-like of shape (n_inclusions,), dtype=int
        The indices of the prerequisite features.

        Returns
        -------
        mask : ndarray of shape (n_features,), dtype=np.ubyte, order='F'
            Mask for valid candidate features.
            The data type is unsigned char.

        indices: ndarray of shape (n_features_to_select,), dtype=np.intc, order='F'
            The indices vector of selected features, initiated with -1.
            The data type is signed int.

        scores: ndarray of shape (n_features_to_select,), dtype=float, order='F'
            The h-correlation/eta-cosine of selected features.
        """
        mask = np.ones(self.n_features_in_, dtype=np.ubyte, order="F")
        # initiated with -1
        indices = np.full(self.n_features_to_select, -1, dtype=np.intc, order="F")
        for i in range(inclusive_indices.shape[0]):
            indices[i] = inclusive_indices[i]
        scores = np.zeros(self.n_features_to_select, dtype=float, order="F")
        return mask, indices, scores

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_