"""
Fast computation of prediction and gradient for narx.
"""
# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython cimport final
import numpy as np
from sklearn.utils._cython_blas cimport RowMajor, NoTrans
from sklearn.utils._cython_blas cimport _gemm


@final
cdef inline void _update_terms(
    const double[:, ::1] X,       # IN
    const double[:, ::1] y_hat,   # IN
    double* terms,                # OUT
    const int[:, ::1] feat_ids,     # IN
    const int[:, ::1] delay_ids,    # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate all terms for the given features and delays at timestep k.
    """
    cdef:
        Py_ssize_t i
        Py_ssize_t n_coefs = feat_ids.shape[0]

    for i in range(n_coefs):
        terms[i] = _evaluate_term(
            X, y_hat, feat_ids[i], delay_ids[i], k
        )


@final
cdef inline void _predict_step(
    const double[:, ::1] X,       # IN
    const double[:, ::1] y_hat,   # IN
    double[::1] y_pred,           # OUT
    const double[::1] coef,       # IN
    const int[:, ::1] feat_ids,     # IN
    const int[:, ::1] delay_ids,    # IN
    const int[::1] output_ids,      # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate the expression for all outputs at timestep k.
    """
    cdef:
        Py_ssize_t n_coefs = feat_ids.shape[0]
        Py_ssize_t i, output_i

    # Add all terms
    for i in range(n_coefs):
        output_i = output_ids[i]
        y_pred[output_i] += coef[i] * _evaluate_term(
            X, y_hat, feat_ids[i], delay_ids[i], k
        )


@final
cdef inline double _evaluate_term(
    const double[:, ::1] X,       # IN
    const double[:, ::1] y_hat,   # IN
    const int[::1] feat_ids,    # IN
    const int[::1] delay_ids,   # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate a term based on feature and delay IDs.
    """
    cdef:
        Py_ssize_t n_feats = X.shape[1]
        Py_ssize_t n_coefs = feat_ids.shape[0]
        double term = 1.0
        Py_ssize_t i, feat_id

    for i in range(n_coefs):
        feat_id = feat_ids[i]
        if feat_id != -1:
            if feat_id < n_feats:
                term *= X[k - delay_ids[i], feat_id]
            else:
                term *= y_hat[k - delay_ids[i], feat_id - n_feats]

    return term


@final
cdef inline void _update_dcf(
    const double[:, ::1] X,           # IN
    const double[:, ::1] y_hat,       # IN
    double[:, :, ::1] dcf,            # OUT
    const double[::1] coef,           # IN
    const int[:, ::1] grad_yyd_ids,     # IN
    const int[:, ::1] grad_delay_ids,   # IN
    const int[::1] grad_coef_ids,       # IN
    const int[:, ::1] grad_feat_ids,    # IN
    const int k,                        # IN
) noexcept nogil:
    """
    Updates DCF matrix based on the current state.
    """
    cdef:
        Py_ssize_t n_grad_terms = grad_yyd_ids.shape[0]
        Py_ssize_t i, row_y_id, col_y_id, delay_id_1

    memset(&dcf[0, 0, 0], 0, dcf.shape[0] * dcf.shape[1] * dcf.shape[2] * sizeof(double))

    for i in range(n_grad_terms):
        row_y_id = grad_yyd_ids[i, 0]
        col_y_id = grad_yyd_ids[i, 1]
        delay_id_1 = grad_yyd_ids[i, 2]

        dcf[delay_id_1, row_y_id, col_y_id] += coef[grad_coef_ids[i]] * \
            _evaluate_term(
                X, y_hat, grad_feat_ids[i], grad_delay_ids[i], k
            )


@final
cpdef void _predict(
    const double[:, ::1] X,                 # IN
    const double[:, ::1] y_ref,             # IN
    const double[::1] coef,                 # IN
    const double[::1] intercept,            # IN
    const int[:, ::1] feat_ids,             # IN
    const int[:, ::1] delay_ids,            # IN
    const int[::1] output_ids,              # IN
    const int[::1] session_sizes_cumsum,    # IN
    double[:, ::1] y_hat,                   # OUT
) noexcept nogil:
    """
    Vectorized (Cython) variant of Python NARX._predict.
    Returns y_hat array (n_samples, n_outputs).
    """
    cdef:
        Py_ssize_t max_delay
    with gil:
        max_delay = np.max(delay_ids)
    cdef:
        Py_ssize_t n_samples = X.shape[0]
        Py_ssize_t n_ref = y_ref.shape[0]
        Py_ssize_t k, s = 0
        Py_ssize_t init_k = 0
        bint at_init = True

    for k in range(n_samples):
        if k == session_sizes_cumsum[s]:
            s += 1
            at_init = True
            init_k = k
        with gil:
            if not np.all(np.isfinite(X[k])):
                at_init = True
                init_k = k + 1
                y_hat[k] = np.nan
                continue
        if k - init_k == max_delay:
            at_init = False

        if at_init:
            y_hat[k] = y_ref[k % n_ref]
            with gil:
                if not np.all(np.isfinite(y_hat[k])):
                    at_init = True
                    init_k = k + 1
        else:
            y_hat[k] = intercept
            _predict_step(
                X,
                y_hat,
                y_hat[k],
                coef,
                feat_ids,
                delay_ids,
                output_ids,
                k,
            )
        # Handle divergence
        with gil:
            if np.max(np.abs(y_hat[k])) > 1e20:
                break


@final
cpdef void _update_dydx(
    const double[:, ::1] X,
    const double[:, ::1] y_hat,
    const double[::1] coef,
    const int[:, ::1] feat_ids,
    const int[:, ::1] delay_ids,
    const int[::1] y_ids,
    const int[:, ::1] grad_yyd_ids,
    const int[:, ::1] grad_delay_ids,
    const int[::1] grad_coef_ids,
    const int[:, ::1] grad_feat_ids,
    const int[::1] session_sizes_cumsum,
    double[:, :, ::1] dydx,                 # OUT
    double[:, :, ::1] dcf,                  # OUT
) noexcept nogil:
    """
    Computation of the Jacobian matrix dydx.

    Returns
    -------
    dydx : ndarray of shape (n_samples, n_outputs, n_x)
        Jacobian matrix of the outputs with respect to coefficients and intercepts.
    """
    cdef Py_ssize_t max_delay
    cdef bint not_empty
    with gil:
        max_delay = np.max(delay_ids)
        not_empty = max_delay > 0 and grad_yyd_ids.size > 0
    cdef:
        Py_ssize_t n_samples = y_hat.shape[0]
        Py_ssize_t n_coefs = feat_ids.shape[0]
        Py_ssize_t k, i, d, s = 0
        Py_ssize_t M = dcf.shape[1]      # n_outputs
        Py_ssize_t N = dydx.shape[2]     # n_x
        Py_ssize_t init_k = 0
        bint at_init = True
        bint is_finite
        double* terms_intercepts = <double*> malloc(sizeof(double) * N)

    # Set intercepts
    for i in range(n_coefs, N):
        terms_intercepts[i] = 1.0

    for k in range(n_samples):
        if k == session_sizes_cumsum[s]:
            s += 1
            at_init = True
            init_k = k
            continue
        with gil:
            is_finite = np.all(np.isfinite(X[k])) and np.all(np.isfinite(y_hat[k]))
        if not is_finite:
            at_init = True
            init_k = k + 1
            continue
        if k - init_k == max_delay:
            at_init = False

        if at_init:
            continue
        # Compute terms for time step k (no effect on intercepts)
        _update_terms(
            X,
            y_hat,
            terms_intercepts,
            feat_ids,
            delay_ids,
            k,
        )

        # Update constant terms of Jacobian
        for i in range(N):
            dydx[k, y_ids[i], i] = terms_intercepts[i]

        # Update dynamic terms of Jacobian
        if not_empty:
            _update_dcf(
                X,
                y_hat,
                dcf,
                coef,
                grad_yyd_ids,
                grad_delay_ids,
                grad_coef_ids,
                grad_feat_ids,
                k,
            )
            for d in range(max_delay):
                # dydx[k] += dcf[d] @ dydx[k-d-1]
                # dcf[d] (M,M), dydx[k] (M,N)
                _gemm(
                    RowMajor, NoTrans, NoTrans,
                    M, N, M,
                    1.0, &dcf[d, 0, 0], M, &dydx[k-d-1, 0, 0], N,
                    1.0, &dydx[k, 0, 0], N,
                )

        # Handle divergence
        with gil:
            if np.max(np.abs(dydx[k])) > 1e20:
                break
    free(terms_intercepts)
