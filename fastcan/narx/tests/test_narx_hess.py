"""Test Hessian matrix of NARX"""

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime

from fastcan.narx import NARX


def _hessian_wrapper(
    coef_intercept,
    X,
    y,
    feat_ids,
    delay_ids,
    output_ids,
    fit_intercept,
    sample_weight_sqrt,
    session_sizes_cumsum,
    grad_yyd_ids,
    grad_coef_ids,
    grad_feat_ids,
    grad_delay_ids,
    flag,
):
    (hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_feat_ids, hess_delay_ids) = (
        NARX._get_hc_ids(
            grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids, need_hess=True
        )
    )

    combined_term_ids, unique_feat_ids, unique_delay_ids = NARX._get_term_ids(
        np.vstack([feat_ids, grad_feat_ids, hess_feat_ids]),
        np.vstack([delay_ids, grad_delay_ids, hess_delay_ids]),
    )
    n_terms = feat_ids.shape[0]
    n_grad = grad_feat_ids.shape[0]
    const_term_ids = combined_term_ids[:n_terms]
    grad_term_ids = combined_term_ids[n_terms : n_terms + n_grad]
    hess_term_ids = combined_term_ids[n_terms + n_grad :]

    max_delay = int(delay_ids.max())
    n_outputs = y.shape[1]
    if fit_intercept:
        y_ids = np.asarray(
            np.r_[output_ids, np.arange(n_outputs, dtype=np.int32)], dtype=np.int32
        )
    else:
        y_ids = np.asarray(output_ids, dtype=np.int32)

    res, jac, hess = NARX._func(
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
        flag=1,
    )
    return res, jac, hess


def _approx_numeric_hessian(params, wrapper_func, epsilon=1e-6):
    def grad_component(param_vec, idx):
        res_i, jac_i, _ = wrapper_func(param_vec)
        return (jac_i.T @ res_i)[idx]

    rows = [
        approx_fprime(params, grad_component, epsilon, i) for i in range(params.size)
    ]
    return np.vstack(rows)


def test_simple_num():
    """Simple model
    test model: y(k) = 0.4*y(k-1) + u(k-1) + 1
    initial y[-1] = 0
    u[0] = u[1] = u[2] = 1.5
    """
    # Ground truth
    X = np.array([1.5, 1.5, 1.5]).reshape(-1, 1)
    y = np.array([1, 2.9, 3.66]).reshape(-1, 1)

    feat_ids = np.array([1, 0], dtype=np.int32).reshape(-1, 1)
    delay_ids = np.array([1, 1], dtype=np.int32).reshape(-1, 1)
    output_ids = np.array([0, 0], dtype=np.int32)
    coef = np.array([0.4, 1])
    intercept = np.array([1], dtype=float)
    sample_weight = np.array([1, 1, 1], dtype=float).reshape(-1, 1)
    sample_weight_sqrt = np.sqrt(sample_weight)
    session_sizes = np.array([len(y)], dtype=np.int32)

    grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_jc_ids(
        feat_ids, delay_ids, output_ids, 1
    )

    _, _, hess = _hessian_wrapper(
        np.r_[coef, intercept],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        sample_weight_sqrt,
        session_sizes,
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
        flag=1,
    )
    params = np.r_[coef, intercept]
    wrapper_func = partial(
        _hessian_wrapper,
        X=X,
        y=y,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=True,
        sample_weight_sqrt=sample_weight_sqrt,
        session_sizes_cumsum=session_sizes,
        grad_yyd_ids=grad_yyd_ids,
        grad_coef_ids=grad_coef_ids,
        grad_feat_ids=grad_feat_ids,
        grad_delay_ids=grad_delay_ids,
        flag=0,
    )
    hess_num = _approx_numeric_hessian(params, wrapper_func)
    assert_allclose(hess, hess_num, rtol=0.1, atol=0.01)


def test_complex():
    """Complex model"""
    # Simulated model
    rng = np.random.default_rng(12345)
    n_samples = 200
    max_delay = 3
    e0 = rng.normal(0, 0.01, n_samples)
    e1 = rng.normal(0, 0.01, n_samples)
    u0 = rng.uniform(0, 0.1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples + max_delay)
    y0 = np.zeros(n_samples + max_delay)
    y1 = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y0[i] = (
            0.5 * y0[i - 1]
            + 0.8 * y1[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - 3]
            + 1
        )
        y1[i] = (
            0.6 * y1[i - 1]
            - 0.2 * y0[i - 1] * y1[i - 2]
            + 0.3 * u1[i] ** 2
            + 1.5 * u1[i - 2] * u0[i - 3]
            + 0.5
        )
    y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
    X = np.c_[u0[max_delay:], u1[max_delay:]]
    sample_weight_sqrt = np.sqrt(np.ones((y.shape[0], 1)))
    session_sizes = np.array([len(y)], dtype=np.int32)

    feat_ids = np.array(
        [
            [-1, 2],
            [-1, 3],
            [0, 0],
            [0, 0],
            [0, 1],
            [-1, 3],
            [2, 3],
            [1, 1],
            [1, 0],
        ],
        dtype=np.int32,
    )

    delay_ids = np.array(
        [
            [-1, 1],
            [-1, 1],
            [0, 0],
            [1, 3],
            [2, 3],
            [-1, 1],
            [1, 2],
            [0, 0],
            [2, 3],
        ],
        dtype=np.int32,
    )

    output_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    coef = np.array(
        [
            0.5,
            0.8,
            0.3,
            2,
            1.5,
            0.6,
            -0.2,
            0.3,
            1.5,
        ]
    )

    intercept = np.array([1, 0.5])

    grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_jc_ids(
        feat_ids, delay_ids, output_ids, X.shape[1]
    )

    _, _, hess = _hessian_wrapper(
        np.r_[coef, intercept],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        sample_weight_sqrt,
        session_sizes,
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
        flag=1,
    )
    params = np.r_[coef, intercept]
    wrapper_func = partial(
        _hessian_wrapper,
        X=X,
        y=y,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=True,
        sample_weight_sqrt=sample_weight_sqrt,
        session_sizes_cumsum=session_sizes,
        grad_yyd_ids=grad_yyd_ids,
        grad_coef_ids=grad_coef_ids,
        grad_feat_ids=grad_feat_ids,
        grad_delay_ids=grad_delay_ids,
        flag=0,
    )
    hess_num = _approx_numeric_hessian(params, wrapper_func)
    assert_allclose(hess, hess_num, rtol=0.1, atol=0.01)
