"""Test NARX"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator

from fastcan.narx import (
    NARX,
    _fd2pt,
    _mask_missing_value,
    _pt2fd,
    make_narx,
    make_poly_ids,
    make_time_shift_ids,
    print_narx,
)


def test_narx_is_sklearn_estimator():
    with pytest.warns(UserWarning, match="output_ids got"):
        check_estimator(NARX())


def test_poly_ids():
    with pytest.raises(ValueError, match=r"The output that would result from the .*"):
        make_poly_ids(10, 1000)


def test_time_ids():
    with pytest.raises(ValueError, match=r"The length of `include_zero_delay`.*"):
        make_time_shift_ids(3, 2, [False, True, False, True])


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("nan", [False, True])
def test_narx(nan, multi_output):
    """Test NARX"""
    if multi_output:
        rng = np.random.default_rng(12345)
        n_samples = 1000
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
                - 0.2 * y0[i - 1]*y1[i - 2]
                + 0.3 * u1[i] ** 2
                + 1.5 * u1[i - 2] * u0[i - 3]
                + 0.5
            )
        y = np.c_[y0[max_delay:]+e0, y1[max_delay:]+e1]
        X = np.c_[u0[max_delay:], u1[max_delay:]]
        n_outputs = 2
    else:
        rng = np.random.default_rng(12345)
        n_samples = 1000
        max_delay = 3
        e = rng.normal(0, 0.1, n_samples)
        u0 = rng.uniform(0, 1, n_samples + max_delay)
        u1 = rng.normal(0, 0.1, n_samples)
        y = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y[i] = (
                0.5 * y[i - 1]
                + 0.3 * u0[i] ** 2
                + 2 * u0[i - 1] * u0[i - 3]
                + 1.5 * u0[i - 2] * u1[i - max_delay]
                + 1
            )
        y = y[max_delay:] + e
        X = np.c_[u0[max_delay:], u1]
        n_outputs = 1

    if nan:
        X_nan_ids = rng.choice(n_samples, 20, replace=False)
        y_nan_ids = rng.choice(n_samples, 10, replace=False)
        X[X_nan_ids] = np.nan
        y[y_nan_ids] = np.nan

    if multi_output:
        narx_score = make_narx(
            X,
            y,
            n_terms_to_select=[5, 4],
            max_delay=3,
            poly_degree=2,
            verbose=0,
        ).fit(X, y)
    else:
        narx_score = make_narx(
            X,
            y,
            n_terms_to_select=4,
            max_delay=3,
            poly_degree=2,
            verbose=0,
        ).fit(X, y)

    assert r2_score(*_mask_missing_value(y, narx_score.predict(X, y_init=y))) > 0.5

    params = {
        "n_terms_to_select": rng.integers(low=2, high=4),
        "max_delay": rng.integers(low=0, high=10),
        "poly_degree": rng.integers(low=2, high=5),
    }

    narx_default = make_narx(X=X, y=y, **params)

    if multi_output:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"]*2
    else:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"]

    params["include_zero_delay"] = [False, True]
    narx_0_delay = make_narx(X=X, y=y, **params)
    _, time_shift_ids = _fd2pt(narx_0_delay.feat_ids, narx_0_delay.delay_ids)
    time_ids_u0 = time_shift_ids[time_shift_ids[:, 0] == 0]
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    time_ids_y = time_shift_ids[time_shift_ids[:, 0] == 2]
    assert ~np.isin(0, time_ids_u0[:, 1]) or (time_ids_u0.size == 0)
    assert np.isin(0, time_ids_u1[:, 1]) or (time_ids_u1.size == 0)
    assert ~np.isin(0, time_ids_y[:, 1]) or (time_ids_y.size == 0)

    params["static_indices"] = [1]
    narx_static = make_narx(X=X, y=y, **params)
    _, time_shift_ids = _fd2pt(narx_static.feat_ids, narx_static.delay_ids)
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    if time_ids_u1.size != 0:
        assert time_ids_u1[0, 1] == 0

    params["refine_drop"] = 1
    params["refine_max_iter"] = 10
    narx_drop = make_narx(X=X, y=y, **params)
    narx_drop_coef = narx_drop.fit(X, y).coef_

    time_shift_ids = make_time_shift_ids(
        X.shape[1] + n_outputs, 5, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    if multi_output:
        n_terms = poly_ids.shape[0]
        output_ids = [0] * n_terms
        output_ids[-1] = 1
    else:
        output_ids = None
    feat_ids, delay_ids = _pt2fd(poly_ids, time_shift_ids)
    narx_osa = NARX(
        feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
    ).fit(X, y)
    assert narx_osa.coef_.size == poly_ids.shape[0]
    narx_osa_msa = narx_drop.fit(X, y, coef_init="one_step_ahead")
    narx_osa_msa_coef = narx_osa_msa.coef_
    narx_array_init_msa = narx_osa_msa.fit(
        X, y, coef_init=np.zeros(narx_osa_msa_coef.size + n_outputs)
    )
    assert np.any(narx_array_init_msa.coef_ != narx_drop_coef)
    assert np.any(narx_osa_msa_coef != narx_array_init_msa.coef_)

    if multi_output:
        y_init = np.ones((narx_array_init_msa.max_delay_, n_outputs))
    else:
        y_init = [1] * narx_array_init_msa.max_delay_
    y_hat = narx_array_init_msa.predict(X, y_init=y_init)
    assert_array_equal(y_hat[:narx_array_init_msa.max_delay_], y_init)

    print_narx(narx_array_init_msa)

    with pytest.raises(ValueError, match=r"`coef_init` should have the shape of .*"):
        narx_array_init_msa.fit(X, y, coef_init=np.zeros(narx_osa_msa_coef.size))

    time_shift_ids = make_time_shift_ids(
        X.shape[1] + n_outputs + 1, 3, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = _pt2fd(poly_ids, time_shift_ids)
    if multi_output:
        n_terms = poly_ids.shape[0]
        output_ids = [0] * n_terms
        output_ids[-1] = 1
    else:
        output_ids = None
    with pytest.raises(ValueError, match=r"The element x of feat_ids should satisfy.*"):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
        ).fit(X, y)

    time_shift_ids = np.array(
        [
            [0, 0],
            [0, -1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = _pt2fd(poly_ids, time_shift_ids)
    n_terms = poly_ids.shape[0]
    output_ids = [0] * n_terms
    output_ids[-1] = 1
    with pytest.raises(ValueError, match=r"The element x of delay_ids should be -1.*"):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
        ).fit(X, y)

    time_shift_ids = make_time_shift_ids(
        X.shape[1] + n_outputs, 3, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = _pt2fd(poly_ids, time_shift_ids)
    delay_ids_shape_err = np.delete(delay_ids, 0, axis=0)
    n_terms = poly_ids.shape[0]
    output_ids = [0] * n_terms
    output_ids[-1] = 1
    with pytest.raises(
        ValueError, match=r"The shape of delay_ids should be equal to .*"
    ):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids_shape_err, output_ids=output_ids
        ).fit(X, y)
    delay_ids_max_err = np.copy(delay_ids)
    delay_ids_max_err[0, 1] = X.shape[0]
    with pytest.raises(
        ValueError, match=r"The element x of delay_ids should satisfy -1.*"
    ):
        narx_osa = NARX(
            feat_ids=feat_ids, delay_ids=delay_ids_max_err, output_ids=output_ids
        ).fit(X, y)


def test_mulit_output_warn_error():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[1, 1], [2, 2]])
    feat_ids, delay_ids = _pt2fd(poly_ids, time_shift_ids)

    with pytest.warns(UserWarning, match="output_ids got"):
        narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
        narx.fit(X, y)

    with pytest.raises(ValueError, match="The length of output_ids should"):
        narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0],
    )
        narx.fit(X, y)

    with pytest.raises(
        ValueError, match=r"The element x of output_ids should satisfy 0 <=.*"
    ):
        narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0, 2],
    )
        narx.fit(X, y)

    with pytest.raises(ValueError, match="The length of `n_terms_to_select` should"):
        make_narx(X=X, y=y, n_terms_to_select=[2], max_delay=3, poly_degree=2)

    with pytest.raises(ValueError, match="`y_init` should have "):
        narx = make_narx(X=X, y=y, n_terms_to_select=[2, 2], max_delay=3, poly_degree=2)
        narx.fit(X, y)
        narx.predict(X, y_init=[1, 1, 1])



def test_sample_weight():
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 3
    e = rng.normal(0, 0.1, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = (
            0.5 * y[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - max_delay]
            + 1
        )
    y = y[max_delay:] + e
    X = np.c_[u0[max_delay:], u1]

    sample_weight = np.ones(n_samples)
    sample_weight[:10] = 0  # Set the first 10 samples to have zero weight

    narx = make_narx(X=X, y=y, n_terms_to_select=3, max_delay=3, poly_degree=2)
    narx.fit(X, y, sample_weight=sample_weight)
    coef_w = narx.coef_
    narx.fit(X, y)
    coef_ = narx.coef_

    assert np.any(coef_w != coef_)

def test_divergence():
    # Test divergence of NARX model
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 3
    e = rng.normal(0, 0.1, n_samples)
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = (
            0.5 * y[i - 1]
            + 0.3 * u0[i] ** 2
            + 2 * u0[i - 1] * u0[i - 3]
            + 1.5 * u0[i - 2] * u1[i - max_delay]
            + 1
        )
    y = y[max_delay:] + e
    X = np.c_[u0[max_delay:], u1]
    narx = make_narx(X, y, 3, 3, 2)
    narx.fit(X, y, coef_init=[-10, 0, 0, 0])
    y_hat = narx.predict(X, y)
    assert np.all(y_hat<=1e20)
