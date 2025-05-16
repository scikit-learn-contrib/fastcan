"""Test Jacobian matrix of NARX"""

import numpy as np
from fastcan._narx_fast import _predict_step  # type: ignore[attr-defined]
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from scipy.integrate import odeint
from sklearn.metrics import r2_score

from fastcan.narx import NARX, make_narx


def test_simple():
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

    y_hat = NARX._predict(
        _predict_step,
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )

    assert_array_equal(y_hat, y)

    delta_w = 0.00001
    coef_1 = np.array([0.4 + delta_w, 1])

    y_hat_1 = NARX._predict(
        _predict_step,
        X=X,
        y_ref=y,
        coef=coef_1,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )

    grad_truth = np.array(
        [
            np.sum(
                np.array([0, y_hat_1[0, 0], y_hat_1[1, 0] + coef_1[0]]).reshape(-1, 1)
            ),
            np.sum(
                np.array([0, X[0, 0], X[0, 0] * coef_1[0] + X[0, 0]]).reshape(-1, 1)
            ),
            np.sum(np.array([0, 1, coef_1[0] + 1]).reshape(-1, 1)),
        ]
    )

    grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_cfd_ids(
        feat_ids, delay_ids, output_ids, 1
    )
    grad = NARX._grad(
        np.r_[coef_1, intercept],
        _predict_step,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(sample_weight),
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    )

    assert_almost_equal(grad.sum(axis=0), grad_truth, decimal=4)
    grad_0 = NARX._grad(
        np.r_[coef_1, 0],
        _predict_step,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(sample_weight),
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    )
    grad = NARX._grad(
        coef_1,
        _predict_step,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        False,
        np.sqrt(sample_weight),
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    )
    assert_almost_equal(grad.sum(axis=0), grad_0.sum(axis=0)[:-1])


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
    grad_yyd_ids, grad_coef_ids, grad_feat_ids, grad_delay_ids = NARX._get_cfd_ids(
        feat_ids, delay_ids, output_ids, X.shape[1]
    )
    grad = NARX._grad(
        np.r_[coef, intercept],
        _predict_step,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        np.sqrt(np.ones((y.shape[0], 1))),
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    )

    # Numerical gradient
    y_hat_0 = NARX._predict(
        _predict_step,
        X=X,
        y_ref=y,
        coef=coef,
        intercept=intercept,
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
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

        y_hat_1 = NARX._predict(
            _predict_step,
            X=X,
            y_ref=y,
            coef=coef_1,
            intercept=intercept_1,
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            output_ids=output_ids,
        )

        e_1 = y_hat_1 - y
        grad_num = (e_1 - e_0).sum(axis=1) / delta_w

        assert_allclose(grad.sum(axis=0)[i], grad_num.sum(), rtol=1e-1)

    grad = NARX._grad(
        coef,
        _predict_step,
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        False,
        np.sqrt(np.ones((y.shape[0], 1))),
        grad_yyd_ids,
        grad_coef_ids,
        grad_feat_ids,
        grad_delay_ids,
    )
    y_hat_0 = NARX._predict(
        _predict_step,
        X=X,
        y_ref=y,
        coef=coef,
        intercept=[0, 0],
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
    )
    e_0 = y_hat_0 - y

    for i in range(len(coef)):
        coef_1 = np.copy(coef)
        coef_1[i] += delta_w

        y_hat_1 = NARX._predict(
            _predict_step,
            X=X,
            y_ref=y,
            coef=coef_1,
            intercept=[0, 0],
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            output_ids=output_ids,
        )

        e_1 = y_hat_1 - y
        grad_num = (e_1 - e_0).sum(axis=1) / delta_w

        assert_allclose(grad.sum(axis=0)[i], grad_num.sum(), rtol=1e-1)


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

    y0 = None
    if y0 is None:
        n_init = 10
        x0 = np.linspace(0, 2, n_init)
        y0_y = np.cos(np.pi * x0)
        y0_x = np.sin(np.pi * x0)
        y0 = np.c_[y0_x, y0_y]
    else:
        n_init = len(y0)

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
