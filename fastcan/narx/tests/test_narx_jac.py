"""Test Jacobian matrix of NARX"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from scipy.integrate import odeint
from sklearn.metrics import r2_score

from fastcan.narx import NARX, make_narx
from fastcan.narx._narx_fast import _predict


def _derivative_wrapper(
    coef_intercept,
    X,
    y,
    feat_ids,
    delay_ids,
    output_ids,
    fit_intercept,
    sample_weight_sqrt,
    session_sizes_cumsum,
    jac_yyd_ids,
    jac_coef_ids,
    jac_feat_ids,
    jac_delay_ids,
):
    # Construct unique terms
    (hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_feat_ids, hess_delay_ids) = (
        NARX._get_hc_ids(
            jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids, X.shape[1], mode=0
        )
    )

    n_terms = feat_ids.shape[0]
    combined_term_ids, unique_feat_ids, unique_delay_ids = NARX._get_term_ids(
        np.vstack([feat_ids, jac_feat_ids, hess_feat_ids]),
        np.vstack([delay_ids, jac_delay_ids, hess_delay_ids]),
    )
    const_term_ids = combined_term_ids[:n_terms]
    n_jac = jac_feat_ids.shape[0]
    jac_term_ids = combined_term_ids[n_terms : n_terms + n_jac]
    hess_term_ids = combined_term_ids[n_terms + n_jac :]

    max_delay = int(delay_ids.max())
    n_outputs = y.shape[1]
    if fit_intercept:
        y_ids = np.asarray(
            np.r_[output_ids, np.arange(n_outputs, dtype=np.int32)], dtype=np.int32
        )
    else:
        y_ids = np.asarray(output_ids, dtype=np.int32)

    _, jac, _ = NARX._func(
        coef_intercept,
        0,
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
        unique_feat_ids,
        unique_delay_ids,
        const_term_ids,
        jac_yyd_ids,
        jac_coef_ids,
        jac_term_ids,
        hess_yyd_ids,
        hess_coef_ids,
        hess_term_ids,
        hess_yd_ids,
    )
    return jac


def test_simple():
    """Simple model
    test model: y(k) = 0.4*y(k-1) + u(k-1) + 1
    initial dy/dx = 0
    u(0) = 0, u(1) = 1.5, u(2) = 1.5, u(3) = 1.5
    y(0) = 0, y(1) = 1,   y(2) = 2.9, y(3) = 3.66
    """
    # Ground truth
    X = np.array([0, 1.5, 1.5, 1.5]).reshape(-1, 1)
    y = np.array([0, 1, 2.9, 3.66]).reshape(-1, 1)

    feat_ids = np.array([1, 0], dtype=np.int32).reshape(-1, 1)
    delay_ids = np.array([1, 1], dtype=np.int32).reshape(-1, 1)
    output_ids = np.array([0, 0], dtype=np.int32)
    coef = np.array([0.4, 1])
    intercept = np.array([1], dtype=float)
    sample_weight = np.array([1, 1, 1, 1], dtype=float).reshape(-1, 1)

    max_delay = int(delay_ids.max())

    y_hat = np.zeros_like(y, dtype=float)
    _predict(
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        y_hat=y_hat,
        session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
        max_delay=max_delay,
    )

    assert_array_equal(y_hat, y)

    delta_w = 0.00001
    coef_1 = np.array([0.4 + delta_w, 1])

    y_hat_1 = np.zeros_like(y, dtype=float)
    _predict(
        X=X,
        y_ref=y,
        coef=coef_1,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        y_hat=y_hat_1,
        session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
        max_delay=max_delay,
    )

    jac_truth = np.c_[
        np.array(
            [
                0,
                y_hat_1[0, 0],
                coef_1[0] * y_hat_1[0, 0] + y_hat_1[1, 0],
                coef_1[0] ** 2 * y_hat_1[0, 0]
                + coef_1[0] * y_hat_1[1, 0]
                + y_hat_1[2, 0],
            ]
        ).reshape(-1, 1),
        np.array(
            [
                0,
                X[0, 0],
                coef_1[0] * X[0, 0] + X[1, 0],
                coef_1[0] ** 2 * X[0, 0] + coef_1[0] * X[1, 0] + X[2, 0],
            ]
        ).reshape(-1, 1),
        np.array([0, 1, coef_1[0] + 1, coef_1[0] ** 2 + coef_1[0] + 1]).reshape(-1, 1),
    ]

    jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids = NARX._get_jc_ids(
        feat_ids, delay_ids, output_ids, 1
    )
    jac = _derivative_wrapper(
        np.r_[coef_1, intercept],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(sample_weight),
        np.array([len(y)], dtype=np.int32),
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
    )

    assert_almost_equal(jac, jac_truth, decimal=15)
    jac_0 = _derivative_wrapper(
        np.r_[coef_1, 0],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(sample_weight),
        np.array([len(y)], dtype=np.int32),
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
    )
    jac = _derivative_wrapper(
        coef_1,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        False,
        np.sqrt(sample_weight),
        np.array([len(y)], dtype=np.int32),
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
    )
    assert_almost_equal(jac, jac_0[:, :-1])


def test_complex():
    """Complex model"""
    # Simulated model
    rng = np.random.default_rng(12345)
    n_samples = 2000
    max_delay = 3
    e0 = rng.normal(0, 0.1, n_samples)
    e1 = rng.normal(0, 0.02, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
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

    # NARX Jacobian
    jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids = NARX._get_jc_ids(
        feat_ids, delay_ids, output_ids, X.shape[1]
    )
    jac = _derivative_wrapper(
        np.r_[coef, intercept],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(np.ones((y.shape[0], 1))),
        np.array([len(y)], dtype=np.int32),
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
    )

    # Numerical gradient
    y_hat_0 = np.zeros_like(y, dtype=float)
    _predict(
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        y_hat=y_hat_0,
        session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
        max_delay=max_delay,
    )
    e_0 = y_hat_0 - y

    delta_w = 0.00001
    for i in range(len(coef) + len(intercept)):
        if i < len(coef):
            coef_1 = np.copy(coef)
            coef_1[i] += delta_w
            intercept_1 = np.copy(intercept)
        else:
            coef_1 = np.copy(coef)
            intercept_1 = np.copy(intercept)
            intercept_1[i - len(coef)] += delta_w

        y_hat_1 = np.zeros_like(y, dtype=float)
        _predict(
            X=X,
            y_ref=y,
            coef=coef_1,
            intercept=intercept_1,
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            output_ids=output_ids,
            y_hat=y_hat_1,
            session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
            max_delay=max_delay,
        )

        e_1 = y_hat_1 - y
        jac_num = (e_1 - e_0) / delta_w

        assert_allclose(jac[:, i], jac_num.flatten(), rtol=0.1, atol=0.01)

    jac = _derivative_wrapper(
        coef,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        False,
        np.sqrt(np.ones((y.shape[0], 1))),
        np.array([len(y)], dtype=np.int32),
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
    )
    y_hat_0 = np.zeros_like(y, dtype=float)
    _predict(
        X=X,
        y_ref=y,
        coef=coef,
        intercept=np.array([0, 0], dtype=float),
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        y_hat=y_hat_0,
        session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
        max_delay=max_delay,
    )
    e_0 = y_hat_0 - y

    for i in range(len(coef)):
        coef_1 = np.copy(coef)
        coef_1[i] += delta_w

        y_hat_1 = np.zeros_like(y, dtype=float)
        _predict(
            X=X,
            y_ref=y,
            coef=coef_1,
            intercept=np.array([0, 0], dtype=float),
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            output_ids=output_ids,
            y_hat=y_hat_1,
            session_sizes_cumsum=np.array([len(y)], dtype=np.int32),
            max_delay=max_delay,
        )

        e_1 = y_hat_1 - y
        jac_num = (e_1 - e_0) / delta_w

        assert_allclose(jac[:, i], jac_num.flatten(), rtol=0.1, atol=0.01)


def test_score_nan():
    """Test fitting scores when data contain nan."""

    def duffing_equation(y, t):
        """Non-autonomous system"""
        y1, y2 = y
        u = 2.5 * np.cos(2 * np.pi * t)
        dydt = [y2, -0.1 * y2 + y1 - 0.25 * y1**3 + u]
        return dydt

    dur = 10
    n_samples = 1000

    dur = 10
    n_samples = 1000
    rng = np.random.default_rng(12345)
    e_train = rng.normal(0, 0.0002, n_samples)
    e_test = rng.normal(0, 0.0002, n_samples)

    t = np.linspace(0, dur, n_samples)

    sol = odeint(duffing_equation, [0.6, 0.8], t)
    u_train = 2.5 * np.cos(2 * np.pi * t).reshape(-1, 1)
    y_train = sol[:, 0] + e_train

    sol = odeint(duffing_equation, [0.6, -0.8], t)
    u_test = 2.5 * np.cos(2 * np.pi * t).reshape(-1, 1)
    y_test = sol[:, 0] + e_test

    max_delay = 3

    narx_model = make_narx(
        X=u_train,
        y=y_train,
        n_terms_to_select=5,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    )

    narx_model.fit(u_train, y_train, coef_init="one_step_ahead")
    y_train_msa_pred = narx_model.predict(u_train, y_init=y_train[:max_delay])
    y_test_msa_pred = narx_model.predict(u_test, y_init=y_test[:max_delay])

    assert r2_score(y_train, y_train_msa_pred) > 0.99
    assert r2_score(y_test, y_test_msa_pred) > -11

    u_all = np.r_[u_train, [[np.nan]], u_test]
    y_all = np.r_[y_train, [np.nan], y_test]
    narx_model = make_narx(
        X=u_all,
        y=y_all,
        n_terms_to_select=5,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    )
    narx_model.fit(u_all, y_all, coef_init="one_step_ahead")
    y_train_msa_pred = narx_model.predict(u_train, y_init=y_train[:max_delay])
    y_test_msa_pred = narx_model.predict(u_test, y_init=y_test[:max_delay])

    assert r2_score(y_train, y_train_msa_pred) > 0.98
    assert r2_score(y_test, y_test_msa_pred) > 0.99
