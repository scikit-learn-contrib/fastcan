"""
Fast computation of prediction and gradient for narx.
"""
# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from libc.string cimport memset, memmove
from cython cimport final
import numpy as np
from sklearn.utils._cython_blas cimport RowMajor, NoTrans, Trans
from sklearn.utils._cython_blas cimport _gemm, _gemv


@final
cdef inline void _update_terms(
    const double[:, ::1] X,       # IN
    const double[:, ::1] y_hat,   # IN
    double[::1] terms,                # OUT
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
cdef inline void _update_jc(
    const double[::1] coefs,            # IN
    const int[:, ::1] yyd_ids,          # IN
    const int[::1] coef_ids,            # IN
    const int[::1] term_ids,            # IN
    const double[::1] term_lib,         # IN
    double[:, :, ::1] jc,               # OUT
) noexcept nogil:
    """
    Updates JC matrix based on the current state.
    axis-0 (d) delay of input y: dyj(k-1)/dx, dyj(k-2)/dx, ..., dyj(k-max_delay)/dx
    axis-1 (i) output y: dy0(k)/dx, dy1(k)/dx, ..., dyn(k)/dx
    axis-2 (j) input y: dy0(k-d)/dx, dy1(k-d)/dx, ..., dyn(k-d)/dx
    It should be noted that delay 1 in axis-0 is at location 0, so we do
    `delay_id - 1` in jac_yyd_ids.
    """
    cdef:
        Py_ssize_t n_terms = yyd_ids.shape[0]
        Py_ssize_t i, out_y_id, in_y_id, delay_id_1
        double coef, term

    memset(&jc[0, 0, 0], 0, jc.shape[0] * jc.shape[1] * jc.shape[2] * sizeof(double))

    for i in range(n_terms):
        out_y_id = yyd_ids[i, 0]
        in_y_id = yyd_ids[i, 1]
        delay_id_1 = yyd_ids[i, 2] - 1
        coef = coefs[coef_ids[i]]
        term = term_lib[term_ids[i]]

        jc[delay_id_1, out_y_id, in_y_id] += coef * term


@final
cdef inline void _update_hc(
    const double[::1] coefs,            # IN
    const int[:, ::1] yyd_ids,          # IN
    const int[::1] coef_ids,            # IN
    const int[::1] term_ids,            # IN
    const double[::1] term_lib,            # IN
    const int[:, ::1] yd_ids,           # IN
    const double[:, :, ::1] dydx,             # IN
    const Py_ssize_t k,            # IN
    const Py_ssize_t k2,             # IN
    double[:, :, :, ::1] hc,               # OUT
    double[:, :, :, ::1] d2ydx2,            # OUT initialized with 0.0
) noexcept nogil:
    """
    Updates HC matrix based on the current state.
    HC matrix has shape in (n_x, max_delay (in), n_outputs (out), n_outputs (in)).
    The second axis corresponds to delay, where delay 1 is at position 0.
    d2ydx2 : ndarray of shape (n_samples, n_x, n_outputs (out), n_x)
    """
    cdef:
        Py_ssize_t n_terms = yyd_ids.shape[0]
        Py_ssize_t n_x = hc.shape[0]
        Py_ssize_t i, j, x, out_y_id, in_y_id, delay_id
        double coef, dydx_k, term, val_xj

    memset(
        &hc[0, 0, 0, 0],
        0,
        hc.shape[0] * hc.shape[1] * hc.shape[2] * hc.shape[3] * sizeof(double)
    )

    for i in range(n_terms):
        out_y_id = yyd_ids[i, 0]
        in_y_id = yyd_ids[i, 1]
        delay_id = yyd_ids[i, 2]
        term = term_lib[term_ids[i]]
        coef = coefs[coef_ids[i]]
        for j in range(n_x):
            if yd_ids[i, 0] == -1:
                # Constant
                # d term / dx = 1 * d y_in * yi * yj
                x = coef_ids[i]
                dydx_k = dydx[k - delay_id, in_y_id, j]
                val_xj = dydx_k * term
                d2ydx2[k2, x, out_y_id, j] += val_xj
                d2ydx2[k2, j, out_y_id, x] += val_xj
            else:
                # Dynamic updating by HC[i, d-1] @ dydx[k-d]
                # d JC / dx = coef * d y_in * d yi * yj
                # dydx has shape in (n_samples, n_outputs (out), n_x)
                dydx_k = dydx[k - yd_ids[i, 1], yd_ids[i, 0], j]
                hc[j, delay_id-1, out_y_id, in_y_id] += coef * dydx_k * term


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
    const int max_delay,                    # IN
    double[:, ::1] y_hat,                   # OUT
) noexcept nogil:
    """
    Vectorized (Cython) variant of Python NARX._predict.
    Returns y_hat array (n_samples, n_outputs).
    """
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
cdef inline void _update_d2ydx2(
    const double[::1] coefs,            # IN
    const int[:, ::1] hess_yyd_ids,          # IN
    const int[::1] hess_coef_ids,            # IN
    const int[::1] hess_term_ids,
    const double[::1] term_lib,            # IN
    const int[:, ::1] hess_yd_ids,           # IN
    const double[:, :, ::1] dydx,             # IN
    const double[:, :, ::1] jc,
    const Py_ssize_t M,
    const Py_ssize_t N,
    const int max_delay,
    const Py_ssize_t k,
    const Py_ssize_t k2,
    double[:, :, :, ::1] hc,
    double[:, :, :, ::1] d2ydx2,            # OUT initialized with 0.0
) noexcept nogil:
    cdef:
        Py_ssize_t i, d
    # Update dynamic terms of Hessian
    _update_hc(
        coefs,
        hess_yyd_ids,
        hess_coef_ids,
        hess_term_ids,
        term_lib,
        hess_yd_ids,
        dydx,
        k,
        k2,
        hc,
        d2ydx2,
    )
    for i in range(N):
        for d in range(max_delay):
            # d2ydx2[k, i] += jc[d] @ d2ydx2[k-d-1, i]
            # jc[d] (M,M), d2ydx2[k-d-1, i] (M,N)
            _gemm(
                RowMajor, NoTrans, NoTrans,
                M, N, M,
                1.0, &jc[d, 0, 0], M, &d2ydx2[k2-d-1, i, 0, 0], N,
                1.0, &d2ydx2[k2, i, 0, 0], N,
            )
            # d2ydx2[k, i] += hc[i, d] @ dydx[k-d-1]
            # hc[i, d] (M,M), dydx[k-d-1] (M,N)
            _gemm(
                RowMajor, NoTrans, NoTrans,
                M, N, M,
                1.0, &hc[i, d, 0, 0], M, &dydx[k-d-1, 0, 0], N,
                1.0, &d2ydx2[k2, i, 0, 0], N,
            )


@final
cpdef void _update_der(
    const int mode,
    const double[:, ::1] X,
    const double[:, ::1] y_hat,
    const int max_delay,
    const int[::1] session_sizes_cumsum,
    const int[::1] y_ids,
    const double[::1] coefs,
    const int[:, ::1] unique_feat_ids,      # IN
    const int[:, ::1] unique_delay_ids,     # IN
    const int[::1] const_term_ids,
    const int[:, ::1] jac_yyd_ids,
    const int[::1] jac_coef_ids,
    const int[::1] jac_term_ids,
    const int[:, ::1] hess_yyd_ids,
    const int[::1] hess_coef_ids,
    const int[::1] hess_term_ids,
    const int[:, ::1] hess_yd_ids,
    const double[::1] p,                          # IN arbitrary vector
    double[:, ::1] term_libs,               # initialized with 1.0
    double[:, :, ::1] jc,
    double[:, :, :, ::1] hc,
    double[:, :, ::1] dydx,                 # OUT initialized with 0.0
    double[:, :, :, ::1] d2ydx2,            # OUT initialized with 0.0
    double[:, :, ::1] d2ydx2p,              # OUT initialized with 0.0
) noexcept nogil:
    """
    Computation of dydx and d2ydx2 matrix.
    mode:
        0 - only dydx
        1 - both dydx and d2ydx2

    Returns
    -------
    dydx : ndarray of shape (n_samples, n_outputs, n_x)
        First derivative matrix of the outputs with respect to coefficients and intercepts.
    d2ydx2 : ndarray of shape (n_samples, n_x, n_outputs (out), n_x)
        Second derivative matrix of the outputs with respect to coefficients and intercepts
    d2ydx2p : ndarray of shape (n_samples, n_outputs, n_x)
        When mode == 2, d2ydx2 is of shape (max_delay + 1, n_x, n_outputs (out), n_x)
        Second derivative matrix times an arbitrary vector p.
    """
    cdef:
        Py_ssize_t n_samples = y_hat.shape[0]
        Py_ssize_t k, i, d, s = 0
        Py_ssize_t M = jc.shape[1]      # n_outputs
        Py_ssize_t N = dydx.shape[2]     # n_x
        Py_ssize_t n_const = const_term_ids.shape[0]
        Py_ssize_t n_jac = jac_term_ids.shape[0]
        Py_ssize_t n_hess = hess_term_ids.shape[0]
        Py_ssize_t init_k = 0
        bint at_init = True
        bint is_finite
        bint jac_not_empty, hess_not_empty

    with gil:
        jac_not_empty = max_delay > 0 and n_jac > 0
        hess_not_empty = max_delay > 0 and n_hess > 0

    for k in range(n_samples):
        # Check if at init
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
        # Compute terms for time step k
        _update_terms(
            X,
            y_hat,
            term_libs[k],
            unique_feat_ids,
            unique_delay_ids,
            k,
        )
        with gil:
            if np.max(np.abs(term_libs[k])) > 1e20:
                break

        # Update constant terms of Jacobian
        for i in range(n_const):
            dydx[k, y_ids[i], i] = term_libs[k, const_term_ids[i]]

        # Update intercepts if any
        for i in range(n_const, N):
            dydx[k, y_ids[i], i] = 1.0

        # Update dynamic terms of Jacobian/Hessian
        if jac_not_empty:
            _update_jc(
                coefs,
                jac_yyd_ids,
                jac_coef_ids,
                jac_term_ids,
                term_libs[k],
                jc,
            )
            for d in range(max_delay):
                # dydx[k] += jc[d] @ dydx[k-d-1]
                # jc[d] (M,M), dydx[k] (M,N)
                _gemm(
                    RowMajor, NoTrans, NoTrans,
                    M, N, M,
                    1.0, &jc[d, 0, 0], M, &dydx[k-d-1, 0, 0], N,
                    1.0, &dydx[k, 0, 0], N,
                )
            if mode == 1 and hess_not_empty:
                # Update dynamic terms of Hessian
                _update_d2ydx2(
                    coefs,
                    hess_yyd_ids,
                    hess_coef_ids,
                    hess_term_ids,
                    term_libs[k],
                    hess_yd_ids,
                    dydx,
                    jc,
                    M,
                    N,
                    max_delay,
                    k,
                    k,
                    hc,
                    d2ydx2,
                )

                # Handle divergence
                with gil:
                    if (
                        not np.all(np.isfinite(d2ydx2[k])) or
                        np.max(np.abs(d2ydx2[k])) > 1e20
                    ):
                        break

            if mode == 2 and hess_not_empty:
                # Update d2ydx2p
                # d2ydx2 values move forward by 1 step d2ydx2[k] -> d2ydx2[k-1]
                memmove(
                    &d2ydx2[0, 0, 0, 0],
                    &d2ydx2[1, 0, 0, 0],
                    max_delay * N * M * N * sizeof(double)
                )
                memset(
                    &d2ydx2[max_delay, 0, 0, 0],
                    0,
                    N * M * N * sizeof(double)
                )

                # Update dynamic terms of Hessian
                _update_d2ydx2(
                    coefs,
                    hess_yyd_ids,
                    hess_coef_ids,
                    hess_term_ids,
                    term_libs[k],
                    hess_yd_ids,
                    dydx,
                    jc,
                    M,
                    N,
                    max_delay,
                    k,
                    max_delay,
                    hc,
                    d2ydx2,
                )

                # Compute d2ydx2p[k] = d2ydx2[k].T @ p
                # d2ydx2[k] shape (N, M, N), p shape (N)
                # target d2ydx2p[k] shape (M, N)
                # Treat d2ydx2[k] as (N, M*N) matrix
                _gemv(
                    RowMajor, Trans,
                    N, M * N,
                    1.0, &d2ydx2[max_delay, 0, 0, 0], M * N,
                    &p[0], 1,
                    0.0, &d2ydx2p[k, 0, 0], 1
                )

                # Handle divergence
                with gil:
                    if (
                        not np.all(np.isfinite(d2ydx2p[k])) or
                        not np.all(np.isfinite(d2ydx2[max_delay])) or
                        (np.max(np.abs(d2ydx2p[k])) > 1e20) or
                        (np.max(np.abs(d2ydx2[max_delay])) > 1e20)
                    ):
                        break

        # Handle divergence
        with gil:
            if not np.all(np.isfinite(dydx[k])) or np.max(np.abs(dydx[k])) > 1e20:
                break
