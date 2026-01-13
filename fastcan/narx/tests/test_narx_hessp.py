from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fastcan.narx import NARX
from fastcan.narx._base import _OptMemoize


@pytest.mark.parametrize("seed", [10, 42, 123, 999, 2024])
def test_random(seed):
    """Test that hessp(p) returns Hess @ p with random configurations."""
    rng = np.random.default_rng(seed)

    # Random configuration generation
    n_features = rng.integers(1, 11)  # (1, 10)
    n_outputs = rng.integers(1, 6)  # (1, 5)
    max_delay = rng.integers(1, 11)  # (1, 10)
    n_degrees = rng.integers(1, 6)  # (1, 5)
    n_terms = n_outputs + rng.integers(1, 11)  # (1, 10)

    n_in_out_1 = n_features + n_outputs - 1

    # feat_ids: values in [-1, n_in_out_1]
    feat_ids = rng.integers(
        -1, n_in_out_1 + 1, size=(n_terms, n_degrees), dtype=np.int32
    )

    # delay_ids: values in [1, max_delay]
    delay_ids = rng.integers(
        1, max_delay + 1, size=(n_terms, n_degrees), dtype=np.int32
    )
    delay_ids[feat_ids == -1] = -1

    # output_ids: values in [0, n_outputs-1].
    # Ensure at least one term per output so we don't have empty hessians
    output_ids = np.concatenate(
        [np.arange(n_outputs), rng.integers(0, n_outputs, size=n_terms - n_outputs)]
    )
    rng.shuffle(output_ids)
    output_ids = output_ids.astype(np.int32)

    # Synthetic data
    n_samples = 10 + max_delay
    X = rng.standard_normal((n_samples, n_features))
    y = rng.standard_normal((n_samples, n_outputs))

    # Define NARX model
    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=True,
    )

    # Use patch to capture the arguments passed to _MemoizeOpt/least_squares
    with patch("fastcan.narx._base.least_squares") as mock_ls:
        # Initial coeffs: should match number of terms + n_outputs (intercepts)
        n_params = n_terms + n_outputs
        coef_init = rng.standard_normal(n_params) * 0.01

        narx.fit(X, y, coef_init=coef_init)

        # Capture arguments passed to least_squares
        # call arguments: (fun, x0, ..., args=args_tuple, ...)
        call_kwargs = mock_ls.call_args.kwargs
        args_tuple = call_kwargs["args"]
        sample_weight_sqrt = args_tuple[6]

    # 1. Compute Hessian matrix using mode=2
    memoize = _OptMemoize(
        NARX._opt_residual,
        NARX._opt_jac,
        NARX._opt_hess,
        NARX._opt_hessp,
        sample_weight_sqrt,
    )
    H = memoize.hess(coef_init, *args_tuple)

    # 2. Compute Hessp product using mode=3 with random vector p
    p = rng.standard_normal(len(coef_init)) * 0.01
    Hp = memoize.hessp(coef_init, p, *args_tuple)

    # 3. Verify Hp == H @ p
    Hp_expected = H @ p

    assert_allclose(
        Hp,
        Hp_expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg="hessp result does not match H @ p",
    )


def test_complex():
    """Test complex model for hessp vs hess @ p."""
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
            + 2 * u0[i - 1] * y1[i - 1]
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
            [0, 3],
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
            [1, 1],
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

    # Define NARX model
    narx = NARX(
        feat_ids=feat_ids,
        delay_ids=delay_ids,
        output_ids=output_ids,
        fit_intercept=True,
    )

    # Use patch to capture the arguments passed to _MemoizeOpt/least_squares
    with patch("fastcan.narx._base.least_squares") as mock_ls:
        coef_init = np.r_[coef, intercept]

        narx.fit(X, y, coef_init=coef_init)

        # Capture arguments passed to least_squares
        call_kwargs = mock_ls.call_args.kwargs
        args_tuple = call_kwargs["args"]
        sample_weight_sqrt = args_tuple[6]

    # 1. Compute Hessian matrix using mode=2
    memoize = _OptMemoize(
        NARX._opt_residual,
        NARX._opt_jac,
        NARX._opt_hess,
        NARX._opt_hessp,
        sample_weight_sqrt,
    )
    H = memoize.hess(coef_init, *args_tuple)

    # 2. Compute Hessp product using mode=3 with random vector p
    p = rng.standard_normal(len(coef_init))
    Hp = memoize.hessp(coef_init, p, *args_tuple)

    # 3. Verify Hp == H @ p
    Hp_expected = H @ p

    assert_allclose(
        Hp,
        Hp_expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg="hessp result does not match H @ p",
    )
