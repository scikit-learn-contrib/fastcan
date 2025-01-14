"""Test NARX"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

from fastcan.narx import NARX, make_narx, make_poly_ids, make_time_shift_ids, print_narx


def test_narx_is_sklearn_estimator():
    check_estimator(NARX())


def test_poly_ids():
    with pytest.raises(ValueError, match=r"The output that would result from the .*"):
        make_poly_ids(10, 1000)


def test_time_ids():
    with pytest.raises(ValueError, match=r"The length of `include_zero_delay`.*"):
        make_time_shift_ids(3, 2, [False, True, False, True])


@pytest.mark.parametrize("nan", [False, True])
def test_narx(nan):
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

    if nan:
        X_nan_ids = rng.choice(n_samples, 20, replace=False)
        y_nan_ids = rng.choice(n_samples, 10, replace=False)
        X[X_nan_ids] = np.nan
        y[y_nan_ids] = np.nan

    params = {
        "n_features_to_select": rng.integers(low=2, high=4),
        "max_delay": rng.integers(low=0, high=10),
        "poly_degree": rng.integers(low=2, high=5),
    }

    narx_default = make_narx(X=X, y=y, **params)

    assert narx_default.poly_ids.shape[0] == params["n_features_to_select"]

    params["include_zero_delay"] = [False, True]
    narx_0_delay = make_narx(X=X, y=y, **params)
    time_shift_ids = narx_0_delay.time_shift_ids
    time_ids_u0 = time_shift_ids[time_shift_ids[:, 0] == 0]
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    time_ids_y = time_shift_ids[time_shift_ids[:, 0] == 2]
    assert ~np.isin(0, time_ids_u0[:, 1]) or (time_ids_u0.size == 0)
    assert np.isin(0, time_ids_u1[:, 1]) or (time_ids_u1.size == 0)
    assert ~np.isin(0, time_ids_y[:, 1]) or (time_ids_y.size == 0)

    params["static_indices"] = [1]
    narx_static = make_narx(X=X, y=y, **params)
    time_shift_ids = narx_static.time_shift_ids
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    if time_ids_u1.size != 0:
        assert time_ids_u1[0, 1] == 0

    params["refine_drop"] = 1
    params["refine_max_iter"] = 10
    narx_drop = make_narx(X=X, y=y, **params)
    assert np.any(narx_drop.poly_ids != narx_static.poly_ids)
    narx_drop_coef = narx_drop.fit(X, y).coef_

    time_shift_ids = make_time_shift_ids(X.shape[1] + 1, 5, include_zero_delay=False)
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    narx_osa = NARX(time_shift_ids=time_shift_ids, poly_ids=poly_ids).fit(X, y)
    assert narx_osa.coef_.size == poly_ids.shape[0]
    narx_osa_msa = narx_drop.fit(X, y, coef_init="one_step_ahead")
    narx_osa_msa_coef = narx_osa_msa.coef_
    assert np.any(narx_osa_msa_coef != narx_drop_coef)
    narx_array_init_msa = narx_osa_msa.fit(
        X, y, coef_init=np.zeros(narx_osa_msa_coef.size + 1)
    )
    assert np.any(narx_array_init_msa.coef_ != narx_osa_msa_coef)

    y_init = [1] * narx_array_init_msa.max_delay_
    y_hat = narx_array_init_msa.predict(X, y_init=y_init)
    assert_array_equal(y_hat[:3], y_init)

    print_narx(narx_array_init_msa)

    with pytest.raises(ValueError, match=r"`y_init` should at least have one .*"):
        narx_array_init_msa.predict(X, y_init=[])

    with pytest.raises(ValueError, match=r"`coef_init` should have the shape of .*"):
        narx_array_init_msa.fit(X, y, coef_init=np.zeros(narx_osa_msa_coef.size))

    time_shift_ids = make_time_shift_ids(X.shape[1] + 2, 3, include_zero_delay=False)
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    with pytest.raises(ValueError, match=r"The element x of the first column of tim.*"):
        narx_osa = NARX(time_shift_ids=time_shift_ids, poly_ids=poly_ids).fit(X, y)

    time_shift_ids = np.array(
        [
            [0, 0],
            [0, -1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    with pytest.raises(ValueError, match=r"The element x of the second column of ti.*"):
        narx_osa = NARX(time_shift_ids=time_shift_ids, poly_ids=poly_ids).fit(X, y)

    time_shift_ids = make_time_shift_ids(X.shape[1] + 1, 3, include_zero_delay=False)
    poly_ids = make_poly_ids(time_shift_ids.shape[0] + 1, 2)
    with pytest.raises(ValueError, match=r"The element x of poly_ids should .*"):
        narx_osa = NARX(time_shift_ids=time_shift_ids, poly_ids=poly_ids).fit(X, y)
