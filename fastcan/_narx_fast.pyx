"""
Fast gradient computation for narx
"""
# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from cython cimport floating, final


@final
cpdef void _update_terms(
    const floating[:, ::1] X,       # IN
    const floating[:, ::1] y_hat,   # IN
    floating[::1] terms,            # OUT
    const int[:, ::1] feat_ids,     # IN
    const int[:, ::1] delay_ids,    # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate all terms for the given features and delays at timestep k.
    """
    cdef:
        int i
        int n_coefs = feat_ids.shape[0]

    for i in range(n_coefs):
        terms[i] = _evaluate_term(
            X, y_hat, feat_ids[i], delay_ids[i], k
        )


@final
cpdef void _predict_step(
    const floating[:, ::1] X,       # IN
    const floating[:, ::1] y_hat,   # IN
    floating[::1] y_pred,           # OUT
    const floating[::1] coef,       # IN
    const int[:, ::1] feat_ids,     # IN
    const int[:, ::1] delay_ids,    # IN
    const int[::1] output_ids,      # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate the expression for all outputs at timestep k.
    """
    cdef:
        int n_terms = feat_ids.shape[0]
        int i, output_i

    # Add all terms
    for i in range(n_terms):
        output_i = output_ids[i]
        y_pred[output_i] += coef[i] * _evaluate_term(
            X, y_hat, feat_ids[i], delay_ids[i], k
        )


@final
cdef floating _evaluate_term(
    const floating[:, ::1] X,       # IN
    const floating[:, ::1] y_hat,   # IN
    const int[::1] feat_ids,        # IN
    const int[::1] delay_ids,       # IN
    const int k,                    # IN
) noexcept nogil:
    """
    Evaluate a term based on feature and delay IDs.
    """
    cdef:
        int n_feats = X.shape[1]
        int n_vars = feat_ids.shape[0]
        floating term = 1.0
        int i, feat_id

    for i in range(n_vars):
        feat_id = feat_ids[i]
        if feat_id != -1:
            if feat_id < n_feats:
                term *= X[k - delay_ids[i], feat_id]
            else:
                term *= y_hat[k - delay_ids[i], feat_id - n_feats]

    return term


@final
cpdef void _update_cfd(
    const floating[:, ::1] X,           # IN
    const floating[:, ::1] y_hat,       # IN
    floating[:, :, ::1] cfd,            # OUT
    const floating[::1] coef,           # IN
    const int[:, ::1] grad_yyd_ids,     # IN
    const int[::1] grad_coef_ids,       # IN
    const int[:, ::1] grad_feat_ids,    # IN
    const int[:, ::1] grad_delay_ids,   # IN
    const int k,                        # IN
) noexcept nogil:
    """
    Updates CFD matrix based on the current state.
    """
    cdef:
        int n_grad_terms = grad_yyd_ids.shape[0]
        int i, row_y_id, col_y_id, delay_id_1

    for i in range(n_grad_terms):
        row_y_id = grad_yyd_ids[i, 0]
        col_y_id = grad_yyd_ids[i, 1]
        delay_id_1 = grad_yyd_ids[i, 2]

        cfd[row_y_id, col_y_id, delay_id_1] += coef[grad_coef_ids[i]] * \
            _evaluate_term(
                X, y_hat, grad_feat_ids[i], grad_delay_ids[i], k
            )
