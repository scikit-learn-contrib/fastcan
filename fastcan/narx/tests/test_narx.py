"""Test NARX"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator

from fastcan.narx import (
    NARX,
    fd2tp,
    make_narx,
    make_poly_features,
    make_poly_ids,
    make_time_shift_features,
    make_time_shift_ids,
    print_narx,
    tp2fd,
)
from fastcan.utils import mask_missing_values


def test_narx_is_sklearn_estimator():
    # Skip 0 feature check for NARX, as AR models have no features
    expected_failures = {
        "check_estimators_empty_data_messages": ("NARX can handle 0 feature."),
    }
    with pytest.warns(UserWarning, match="output_ids got"):
        check_estimator(NARX(), expected_failed_checks=expected_failures)


def test_poly_ids(monkeypatch):
    with pytest.raises(ValueError, match=r"The current configuration would .*"):
        make_poly_ids(10, 1000)

    # Mock combinations_with_replacement to avoid heavy computation
    monkeypatch.setattr(
        "fastcan.narx._feature.combinations_with_replacement",
        lambda *args, **kwargs: iter([[0, 0]]),
    )
    with pytest.warns(UserWarning, match=r"Total number of polynomial features .*"):
        make_poly_ids(18, 10)


def test_time_ids():
    with pytest.raises(ValueError, match=r"The length of `include_zero_delay`.*"):
        make_time_shift_ids(3, 2, [False, True, False, True])


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("nan", [False, True])
def test_narx(nan, multi_output):
    """Test NARX"""

    def make_data(multi_output, nan, rng):
        if multi_output:
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
                    - 0.2 * y0[i - 1] * y1[i - 2]
                    + 0.3 * u1[i] ** 2
                    + 1.5 * u1[i - 2] * u0[i - 3]
                    + 0.5
                )
            y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
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

        X = np.asfortranarray(X)
        y = np.asfortranarray(y)
        return X, y, n_outputs

    rng = np.random.default_rng(12345)
    X, y, n_outputs = make_data(multi_output, nan, rng)

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

    assert r2_score(*mask_missing_values(y, narx_score.predict(X, y_init=y))) > 0.5

    params = {
        "n_terms_to_select": rng.integers(low=2, high=4),
        "max_delay": rng.integers(low=0, high=10),
        "poly_degree": rng.integers(low=2, high=5),
    }

    narx_default = make_narx(X=X, y=y, **params)

    if multi_output:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"] * 2
    else:
        assert narx_default.feat_ids.shape[0] == params["n_terms_to_select"]

    params["include_zero_delay"] = [False, True]
    narx_0_delay = make_narx(X=X, y=y, **params)
    time_shift_ids, _ = fd2tp(narx_0_delay.feat_ids, narx_0_delay.delay_ids)
    time_ids_u0 = time_shift_ids[time_shift_ids[:, 0] == 0]
    time_ids_u1 = time_shift_ids[time_shift_ids[:, 0] == 1]
    time_ids_y = time_shift_ids[time_shift_ids[:, 0] == 2]
    assert ~np.isin(0, time_ids_u0[:, 1]) or (time_ids_u0.size == 0)
    assert np.isin(0, time_ids_u1[:, 1]) or (time_ids_u1.size == 0)
    assert ~np.isin(0, time_ids_y[:, 1]) or (time_ids_y.size == 0)

    params["static_indices"] = [1]
    narx_static = make_narx(X=X, y=y, **params)
    time_shift_ids, _ = fd2tp(narx_static.feat_ids, narx_static.delay_ids)
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
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    narx_osa = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X, y
    )
    assert narx_osa.coef_.size == poly_ids.shape[0]
    narx_osa_msa = narx_drop.fit(X, y, coef_init="one_step_ahead")
    narx_osa_msa_coef = narx_osa_msa.coef_
    narx_array_init_msa = narx_osa_msa.fit(
        X, y, coef_init=np.zeros(narx_osa_msa_coef.size + n_outputs)
    )
    assert np.any(narx_array_init_msa.coef_ != narx_drop_coef)
    assert np.any(narx_osa_msa_coef != narx_array_init_msa.coef_)

    if multi_output:
        y_init = np.ones((narx_array_init_msa.max_delay_, n_outputs), order="F")
    else:
        y_init = [1] * narx_array_init_msa.max_delay_
    y_hat = narx_array_init_msa.predict(X, y_init=y_init)
    assert_array_equal(y_hat[: narx_array_init_msa.max_delay_], y_init)

    with pytest.raises(ValueError, match=r"`coef_init` should have the shape of .*"):
        narx_array_init_msa.fit(X, y, coef_init=np.zeros(narx_osa_msa_coef.size))

    time_shift_ids = make_time_shift_ids(
        X.shape[1] + n_outputs + 1, 3, include_zero_delay=False
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
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
            [0, 1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
    delay_ids[0, 0] = -2
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
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)
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


def test_mulit_output_warn():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    for i in range(2):
        if i == 0:
            # X only, grad does not have dynamic part
            time_shift_ids = np.array([[0, 1], [1, 1]])
            poly_ids = np.array([[1, 1], [2, 2]])
        else:
            time_shift_ids = np.array([[0, 0], [1, 1], [2, 1]])
            poly_ids = np.array([[1, 1], [2, 2], [0, 3]])
        feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

        with pytest.warns(UserWarning, match="output_ids got"):
            narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids)
            narx.fit(X, y)
        y_pred = narx.predict(X)
        assert_almost_equal(
            np.std(y_pred[narx.max_delay_ :, 1] - np.mean(y[:, 1])), 0.0
        )

        X_nan = np.copy(X)
        y_nan = np.copy(y)
        X_nan[4, 0] = np.nan
        y_nan[4, 1] = np.nan
        for coef_init in [None, "one_step_ahead"]:
            with pytest.warns(UserWarning, match="output_ids got"):
                y_pred = narx.fit(X_nan, y_nan, coef_init=coef_init).predict(X_nan)
            y_nan_masked, y_pred_masked = mask_missing_values(y_nan, y_pred)
            assert_almost_equal(
                np.std(
                    y_pred_masked[y_pred_masked[:, 0] != 0, 1]
                    - np.mean(y_nan_masked[:, 1])
                ),
                0.0,
            )


def test_fit_intercept():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[1, 1], [2, 2]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        fit_intercept=False,
    )
    narx.fit(X, y)
    assert_almost_equal(narx.intercept_, 0.0)
    narx.fit(X, y, coef_init="one_step_ahead")
    assert_almost_equal(narx.intercept_, 0.0)

    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[1, 1], [2, 2]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    narx = make_narx(X, y, 1, 2, 2, fit_intercept=False)
    narx.fit(X, y)
    assert_array_equal(narx.intercept_, [0.0, 0.0])
    narx.fit(X, y, coef_init="one_step_ahead")
    assert_array_equal(narx.intercept_, [0.0, 0.0])

    with pytest.warns(UserWarning, match="output_ids got"):
        narx = NARX(
            feat_ids=feat_ids,
            delay_ids=delay_ids,
            fit_intercept=False,
        )
        narx.fit(X, y)
        assert_array_equal(narx.intercept_, [0.0, 0.0])
        narx.fit(X, y, coef_init=[0, 0])
        assert_array_equal(narx.intercept_, [0.0, 0.0])


def test_mulit_output_error():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[1, 1], [2, 2]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

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

    with pytest.raises(ValueError, match=r"`feat_ids` should not contain rows that.*"):
        narx = NARX(
            feat_ids=np.array([[0, 1], [-1, -1]]),
            delay_ids=np.array([[0, 1], [-1, -1]]),
            output_ids=[0, 1],
        )
        narx.fit(X, y)


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

    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
    sw = rng.integers(0, 5, size=12)
    X_repeated = np.repeat(X, sw, axis=0)
    y_repeated = np.repeat(y, sw)
    narx_osa = NARX().fit(X_repeated, y_repeated)
    narx_no_sw = NARX().fit(X_repeated, y_repeated, coef_init=[0] * 3)
    assert_allclose(
        np.r_[narx_osa.coef_, narx_osa.intercept_],
        np.r_[narx_no_sw.coef_, narx_no_sw.intercept_],
    )
    narx_sw = NARX().fit(X, y, sample_weight=sw, coef_init=[0] * 3)
    assert_allclose(
        np.r_[narx_no_sw.coef_, narx_no_sw.intercept_],
        np.r_[narx_sw.coef_, narx_sw.intercept_],
    )


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
    div_idx = np.where(np.abs(y_hat) > 1e20)[0][0] + 1
    assert np.all(y_hat[div_idx:] == 0)


def test_tp2fd():
    time_shift_ids = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
        ]
    )
    poly_ids = make_poly_ids(time_shift_ids.shape[0], 2)
    with pytest.raises(ValueError, match=r"time_shift_ids should have shape.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [-1, 1],
            [1, 1],
            [1, 2],
        ]
    )
    with pytest.raises(ValueError, match=r"The element x of the first column of tim.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [0, -1],
            [1, 1],
            [1, 2],
        ]
    )
    with pytest.raises(ValueError, match=r"The element x of the second column of ti.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)
    time_shift_ids = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
        ]
    )
    poly_ids[-1][-1] = 5
    with pytest.raises(ValueError, match=r"The element x of poly_ids should.*"):
        _, _ = tp2fd(time_shift_ids, poly_ids)


def test_print_narx(capsys):
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    feat_ids = np.array([[0, 1], [1, 2]])
    delay_ids = np.array([[1, 0], [2, 2]])

    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=[0, 1],
    )
    narx.fit(X, y)
    print_narx(narx)
    captured = capsys.readouterr()
    # Check if the header is present in the output
    assert "| yid |        Term        |   Coef   |" in captured.out
    # Check if the separator line is present
    assert "|-----|--------------------|----------|" in captured.out
    # Check if the intercept line for yid 0 is present
    assert "|  0  |     Intercept      |" in captured.out
    # Check if the intercept line for yid 1 is present
    assert "|  1  |     Intercept      |" in captured.out
    # Check if the term line for yid 0 is present
    assert "|  0  |  X[k-1,0]*X[k,1]   |" in captured.out
    # Check if the term line for yid 1 is present
    assert "|  1  |X[k-2,1]*y_hat[k-2,0]|" in captured.out


def test_make_narx_refine_print(capsys):
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2)
    _ = make_narx(
        X,
        y,
        n_terms_to_select=2,
        max_delay=2,
        poly_degree=2,
        refine_drop=1,
    )
    captured = capsys.readouterr()
    assert "No. of iterations: " in captured.out


def test_make_narx_max_candidates():
    """Test max_candidates and random_state in make_narx."""
    rng = np.random.default_rng(12345)
    X = rng.random((100, 2))
    y = rng.random((100, 1))
    max_delay = 3
    poly_degree = 10
    n_terms_to_select = 5
    max_candidates = 20

    # With the same random_state, the results should be identical
    narx1 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=123,
        verbose=0,
    )
    narx2 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=123,
        verbose=0,
    )
    assert_array_equal(narx1.feat_ids, narx2.feat_ids)
    assert_array_equal(narx1.delay_ids, narx2.delay_ids)

    # With different random_state, the results should be different
    narx3 = make_narx(
        X,
        y,
        n_terms_to_select=n_terms_to_select,
        max_delay=max_delay,
        poly_degree=poly_degree,
        max_candidates=max_candidates,
        random_state=456,
        verbose=0,
    )
    assert not np.array_equal(narx1.feat_ids, narx3.feat_ids)

    # Check if number of selected terms is correct
    assert narx1.feat_ids.shape[0] == n_terms_to_select


@pytest.mark.parametrize("max_delay", [1, 3, 7, 10])
def test_nan_split(max_delay):
    n_sessions = 10
    n_samples_per_session = 100
    X = np.random.rand(n_samples_per_session, 2)
    y = np.random.rand(n_samples_per_session, 2)
    for _ in range(n_sessions - 1):
        X = np.r_[
            X,
            [[np.nan, np.nan]] * max_delay,
            np.random.rand(n_samples_per_session, 2),
        ]
        y = np.r_[
            y,
            [[np.nan, np.nan]] * max_delay,
            np.random.rand(n_samples_per_session, 2),
        ]
    narx = make_narx(
        X,
        y,
        n_terms_to_select=10,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    ).fit(
        X,
        y,
    )

    xy_hstack = np.c_[X, y]
    time_shift_ids, poly_ids = fd2tp(narx.feat_ids_, narx.delay_ids_)
    time_shift_vars = make_time_shift_features(xy_hstack, time_shift_ids)
    poly_terms = make_poly_features(time_shift_vars, poly_ids)
    poly_terms_masked, y_masked = mask_missing_values(poly_terms, y)
    assert poly_terms_masked.shape[0] == y_masked.shape[0]
    assert poly_terms_masked.shape[0] == n_sessions * (
        n_samples_per_session - narx.max_delay_
    )


def test_default_narx_handles_zero_features():
    """Check that default NARX handles X with 0 features without error."""
    X = np.empty((10, 0))
    y = np.random.rand(10, 1)
    NARX().fit(X, y)


def test_auto_reg():
    """Test auto-regression with NARX"""
    rng = np.random.default_rng(12345)
    n_samples = 100
    max_delay = 2
    e0 = rng.normal(0, 0.01, n_samples)
    e1 = rng.normal(0, 0.01, n_samples)
    y0 = np.ones(n_samples + max_delay)
    y1 = np.ones(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y0[i] = 0.5 * y0[i - 1] + 0.8 * y1[i - 1] + 1
        y1[i] = 0.6 * y1[i - 1] - 0.2 * y0[i - 1] * y1[i - 2] + 0.5
    y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
    X = np.empty((n_samples, 0))  # No features, only auto-regression

    model = make_narx(
        X,
        y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=2,
        verbose=0,
    )
    model.fit(X, y)
    y_pred = model.predict(X, y_init=y[: model.max_delay_])
    assert r2_score(y, y_pred) > 0.5

    model = make_narx(
        None,
        y,
        n_terms_to_select=2,
        max_delay=max_delay,
        poly_degree=2,
        verbose=0,
    )
    model.fit(None, y)
    y_pred = model.predict(len(y), y_init=y[: model.max_delay_])
    assert r2_score(y, y_pred) > 0.5


def test_auto_reg_error():
    X = np.empty((10, 1))
    y = np.random.rand(10, 1)
    model = NARX().fit(X, y)
    with pytest.raises(ValueError, match=r"X should be an array-like of shape.*"):
        model.predict(len(y), y_init=y[: model.max_delay_])


def test_predict_ndim():
    """Test the ndim of predict output"""
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    time_shift_ids = np.array([[0, 1], [1, 1]])
    poly_ids = np.array([[1, 1], [2, 2]])
    feat_ids, delay_ids = tp2fd(time_shift_ids, poly_ids)

    model = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=None,
    )
    model.fit(X, y)
    y_hat = model.predict(X, y_init=y[: model.max_delay_])
    assert y_hat.ndim == 1

    model.fit(X, y.reshape(-1, 1))
    y_hat = model.predict(X, y_init=y[: model.max_delay_])
    assert y_hat.ndim == 2
