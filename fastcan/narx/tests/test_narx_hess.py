"""Test Hessian matrix of NARX"""

from collections import Counter
from functools import partial

import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime

from fastcan.narx import NARX
from fastcan.narx._narx_fast import _predict, _update_der


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
    jac_yyd_ids,
    jac_coef_ids,
    jac_feat_ids,
    jac_delay_ids,
    return_hess=True,
):
    mode = 1
    (hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_feat_ids, hess_delay_ids) = (
        NARX._get_hc_ids(
            jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids, X.shape[1], mode=1
        )
    )

    combined_term_ids, unique_feat_ids, unique_delay_ids = NARX._get_term_ids(
        np.vstack([feat_ids, jac_feat_ids, hess_feat_ids]),
        np.vstack([delay_ids, jac_delay_ids, hess_delay_ids]),
    )
    n_terms = feat_ids.shape[0]
    n_jac = jac_feat_ids.shape[0]
    const_term_ids = combined_term_ids[:n_terms]
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

    if return_hess:
        res, jac, hess, _ = NARX._func(
            coef_intercept,
            mode,
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
        return res, jac, hess
    else:
        # Compute prediction
        n_samples, n_outputs = y.shape
        n_x = coef_intercept.shape[0]
        if fit_intercept:
            coef = coef_intercept[:-n_outputs]
            intercept = coef_intercept[-n_outputs:]
        else:
            coef = coef_intercept
            intercept = np.zeros(n_outputs, dtype=float)

        y_hat = np.zeros((n_samples, n_outputs), dtype=float)
        _predict(
            X,
            y,
            coef,
            intercept,
            feat_ids,
            delay_ids,
            output_ids,
            session_sizes_cumsum,
            max_delay,
            y_hat,
        )

        # Compute Jacobian
        dydx = np.zeros((n_samples, n_outputs, n_x), dtype=float)
        jc = np.zeros((max_delay, n_outputs, n_outputs), dtype=float)

        term_libs = np.ones((n_samples, unique_feat_ids.shape[0]), dtype=float)
        hc = np.zeros((n_x, max_delay, n_outputs, n_outputs), dtype=float)
        d2ydx2 = np.zeros((n_samples, n_x, n_outputs, n_x), dtype=float)

        p = np.zeros(1, dtype=float)
        d2ydx2p = np.zeros((1, 1, 1), dtype=float)

        _update_der(
            mode,
            X,
            y_hat,
            max_delay,
            session_sizes_cumsum,
            y_ids,
            coef,
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
            p,
            term_libs,
            jc,
            hc,
            dydx,
            d2ydx2,
            d2ydx2p,
        )
        return y_hat, d2ydx2


def _approx_numeric_hessian(params, wrapper_func, epsilon=1e-6):
    def jac_component(param_vec, idx):
        res_i, jac_i, _ = wrapper_func(param_vec)
        return (jac_i.T @ res_i)[idx]

    rows = [
        approx_fprime(params, jac_component, epsilon, i) for i in range(params.size)
    ]
    return np.vstack(rows)


def _d2ydx2_simple(y, u, a, b):
    n_samples = len(y)
    ya = np.zeros(n_samples)
    yb = np.zeros(n_samples)
    yc = np.zeros(n_samples)
    yaa = np.zeros(n_samples)
    yab = np.zeros(n_samples)
    yac = np.zeros(n_samples)
    ybb = np.zeros(n_samples)
    ybc = np.zeros(n_samples)
    ycc = np.zeros(n_samples)
    for k in range(1, n_samples):
        yaa[k] = ya[k - 1] + ya[k - 1] + a * yaa[k - 1]
        yab[k] = yb[k - 1] + a * yab[k - 1]
        yac[k] = yc[k - 1] + a * yac[k - 1]
        ybb[k] = a * ybb[k - 1]
        ybc[k] = a * ybc[k - 1]
        ycc[k] = a * ycc[k - 1]
        ya[k] = y[k - 1] + a * ya[k - 1]
        yb[k] = u[k - 1] + a * yb[k - 1]
        yc[k] = 1 + a * yc[k - 1]
    d2ydx2 = np.array(
        [
            [yaa, yab, yac],
            [yab, ybb, ybc],
            [yac, ybc, ycc],
        ]
    )
    return d2ydx2


def test_simple():
    """Simple model
    test model: y(k) = 0.4*y(k-1) + u(k-1) + 1
    initial dy/dx = 0, d2y/dx2 = 0
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
    sample_weight_sqrt = np.sqrt(sample_weight)
    session_sizes = np.array([len(y)], dtype=np.int32)

    jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids = NARX._get_jc_ids(
        feat_ids, delay_ids, output_ids, 1
    )

    delta_w = 0.00001
    coef_1 = np.array([0.4 + delta_w, 1])

    y_hat, d2ydx2 = _hessian_wrapper(
        np.r_[coef_1, intercept],
        X,
        y,
        feat_ids,
        delay_ids,
        output_ids,
        True,
        sample_weight_sqrt,
        session_sizes,
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
        return_hess=False,
    )
    d2ydx2 = d2ydx2.squeeze(axis=2)

    d2ydx2_truth = _d2ydx2_simple(y_hat.flatten(), X.flatten(), coef_1[0], coef_1[1])
    d2ydx2_truth = d2ydx2_truth.transpose(2, 0, 1)

    assert np.all(d2ydx2 == d2ydx2_truth)


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
    sample_weight_sqrt = np.sqrt(np.ones((y.shape[0], 1)))
    session_sizes = np.array([len(y)], dtype=np.int32)

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

    jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids = NARX._get_jc_ids(
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
        jac_yyd_ids,
        jac_coef_ids,
        jac_feat_ids,
        jac_delay_ids,
        return_hess=True,
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
        jac_yyd_ids=jac_yyd_ids,
        jac_coef_ids=jac_coef_ids,
        jac_feat_ids=jac_feat_ids,
        jac_delay_ids=jac_delay_ids,
    )
    hess_num = _approx_numeric_hessian(params, wrapper_func)
    assert_allclose(hess, hess_num, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", [10, 42, 123, 999, 2024])
def test_symbolic_hess(seed):
    """Use sympy to verify Hessian computation"""

    def parse_coef(sym):
        """
        x6 -> ("coef", 6)
        u1_1 -> ("u", 1, 1)
        """
        name = sym.name
        if name.startswith("x"):
            return ("coef", int(name[1:]))

        if name.startswith("u"):
            base, delay = name.split("_")
            return ("u", int(base[1:]), int(delay))

        raise ValueError("Not a coefficient")

    def parse_derivative(d):
        """
        Derivative(y*, ...) ->
        (out, delay, i, j)
        """
        if not isinstance(d, sp.Derivative):
            raise TypeError("Expected a Derivative")

        # --- extract output id and delay ---
        f = d.expr
        fname = f.func.__name__  # e.g. "y1_2"
        out, delay = map(int, fname[1:].split("_"))

        # --- extract differentiation variables ---
        vars_ = d.variables
        n = d.derivative_count

        if n == 1:
            # Derivative(y*, x_i)
            i = int(vars_[0].name[1:])
            return (out, delay, i, -1)

        if n == 2:
            if len(vars_) == 1:
                # Derivative(y*, (x_i, 2))
                i = int(vars_[0].name[1:])
                return (out, delay, i, i)
            elif len(vars_) == 2:
                # Derivative(y*, x_i, x_j)
                i = int(vars_[0].name[1:])
                j = int(vars_[1].name[1:])
                return (out, delay, i, j)

        raise ValueError(f"Unsupported derivative structure: {d}")

    def factor_to_tuple(f):
        if isinstance(f, sp.Symbol):
            return parse_coef(f)

        if isinstance(f, sp.Derivative):
            return parse_derivative(f)

        if isinstance(f, sp.Function):
            fname = f.func.__name__
            out, delay = map(int, fname[1:].split("_"))
            return (out, delay, -1, -1)

        if f.is_Number:
            return ("const", float(f))

        if isinstance(f, sp.Pow):
            base, exp = f.as_base_exp()
            if exp.is_Integer and exp > 0:
                item = factor_to_tuple(base)
                if isinstance(item, list):
                    return item * int(exp)
                else:
                    return [item] * int(exp)

        raise TypeError(f"Unsupported factor: {f}")

    def get_sympy_hess(feat_ids, delay_ids, output_ids):
        n_terms = feat_ids.shape[0]
        n_outputs = len(np.unique(output_ids))
        n_features = feat_ids.max() - n_outputs + 1
        n_x = n_terms + n_outputs
        max_delay = delay_ids.max()
        n_degrees = feat_ids.shape[1]

        xx = sp.symbols(f"x0:{n_x}")
        ff = [[None] * (max_delay + 1) for _ in range(n_features + n_outputs)]

        for i in range(n_features + n_outputs):
            for j in range(max_delay + 1):
                if i < n_features:
                    ff[i][j] = sp.symbols(f"u{i}_{j}")
                else:
                    y_id = i - n_features
                    ff[i][j] = sp.Function(f"y{y_id}_{j}")(*xx)  # ty: ignore[call-non-callable]

        yy = [None] * n_outputs
        for i in range(n_outputs):
            yy[i] = xx[-n_outputs + i]
        for i in range(n_terms):
            feat_id = feat_ids[i]
            delay_id = delay_ids[i]
            out_id = output_ids[i]
            term = xx[i]
            for j in range(n_degrees):
                f_id = feat_id[j]
                d_id = delay_id[j]
                if f_id == -1:
                    continue
                else:
                    term *= ff[f_id][d_id]
            yy[out_id] += term

        theta = sp.Matrix(xx)
        H_ys = [sp.hessian(y, theta) for y in yy]

        c_odij = [[] for _ in range(n_outputs)]
        for h_id, H in enumerate(H_ys):
            for j, dx2 in enumerate(H):
                terms = sp.Add.make_args(dx2)
                for i, term in enumerate(terms):
                    factors = sp.Mul.make_args(term)
                    items = []
                    for f in factors:
                        item = factor_to_tuple(f)
                        if isinstance(item, list):
                            items.extend(item)
                        else:
                            items.append(item)

                    const_item = next((x for x in items if x[0] == "const"), None)
                    if const_item:
                        val = int(const_item[1])
                        items.remove(const_item)
                        if val == 0:
                            continue
                        for _ in range(val - 1):
                            c_odij[h_id].append(list(items))

                    c_odij[h_id].append(items)
        return c_odij

    def get_narx_hess(feat_ids, delay_ids, output_ids):
        n_terms = feat_ids.shape[0]
        n_outputs = len(np.unique(output_ids))
        n_features = feat_ids.max() - n_outputs + 1
        n_x = n_terms + n_outputs

        mode = 1

        jac_yyd_ids, jac_coef_ids, jac_feat_ids, jac_delay_ids = NARX._get_jc_ids(
            feat_ids, delay_ids, output_ids, n_features
        )
        (hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_feat_ids, hess_delay_ids) = (
            NARX._get_hc_ids(
                jac_yyd_ids,
                jac_coef_ids,
                jac_feat_ids,
                jac_delay_ids,
                n_features,
                mode,
            )
        )
        combined_term_ids, unique_feat_ids, unique_delay_ids = NARX._get_term_ids(
            np.vstack([feat_ids, jac_feat_ids, hess_feat_ids]),
            np.vstack([delay_ids, jac_delay_ids, hess_delay_ids]),
        )

        jac_term_ids = combined_term_ids[n_terms:]
        n_jac = jac_feat_ids.shape[0]
        jac_term_ids = combined_term_ids[n_terms : n_terms + n_jac]
        hess_term_ids = combined_term_ids[n_terms + n_jac :]

        hess_vars = [[] for _ in range(n_outputs)]
        for yyd_id, coef_id, term_id in zip(jac_yyd_ids, jac_coef_ids, jac_term_ids):
            h_id, out, delay = yyd_id
            feat_ids = unique_feat_ids[term_id]
            delay_ids = unique_delay_ids[term_id]

            term_tuple = []
            for feat_id, delay_id in zip(feat_ids, delay_ids):
                if feat_id == -1:
                    continue
                if feat_id < n_features:
                    term_tuple += [("u", int(feat_id), int(delay_id))]
                else:
                    term_tuple += [(int(feat_id - n_features), int(delay_id), -1, -1)]
            coef_tuple = ("coef", int(coef_id))

            for i in range(n_x):
                for j in range(n_x):
                    der_tuple = (int(out), int(delay), i, j)
                    hess_vars[h_id] += [[coef_tuple] + term_tuple + [der_tuple]]

        for yyd_id, yd_id, coef_id, term_id in zip(
            hess_yyd_ids, hess_yd_ids, hess_coef_ids, hess_term_ids
        ):
            h_id, out_0, delay_0 = yyd_id
            out_1, delay_1 = yd_id
            feat_ids = unique_feat_ids[term_id]
            delay_ids = unique_delay_ids[term_id]

            term_tuple = []
            for feat_id, delay_id in zip(feat_ids, delay_ids):
                if feat_id == -1:
                    continue
                if feat_id < n_features:
                    term_tuple += [("u", int(feat_id), int(delay_id))]
                else:
                    term_tuple += [(int(feat_id - n_features), int(delay_id), -1, -1)]

            for i in range(n_x):
                if out_1 == -1:
                    # Constant term
                    der_tuple = [(int(out_0), int(delay_0), i, -1)]
                    hess_vars[h_id] += [term_tuple + der_tuple]
                    hess_vars[h_id] += [term_tuple + der_tuple]
                else:
                    coef_tuple = ("coef", int(coef_id))
                    for j in range(n_x):
                        der_tuple = [
                            (int(out_0), int(delay_0), j, -1),
                            (int(out_1), int(delay_1), i, -1),
                        ]
                        hess_vars[h_id] += [[coef_tuple] + term_tuple + der_tuple]
        return hess_vars

    def normalize_tuple(t):
        # (out, delay, i, j) -> normalize i, j order if it's a cross term
        if len(t) == 4 and t[3] != -1:
            i, j = t[2], t[3]
            return (t[0], t[1], min(i, j), max(i, j))
        return t

    def normalize_term(term):
        norm_tuples = [normalize_tuple(t) for t in term]
        norm_tuples.sort(key=lambda x: str(x))
        return tuple(norm_tuples)

    rng = np.random.default_rng(seed)
    n_features = rng.integers(1, 11)  # (1, 10)
    n_outputs = rng.integers(1, 6)  # (1, 5)
    max_delay = rng.integers(1, 11)  # (1, 10)
    n_degress = rng.integers(1, 6)  # (1, 5)
    n_terms = n_outputs + rng.integers(1, 11)  # (1, 10)

    n_in_out_1 = n_features + n_outputs - 1

    # feat_ids: values in [-1, n_in_out_1]
    feat_ids = rng.integers(
        -1, n_in_out_1 + 1, size=(n_terms, n_degress), dtype=np.int32
    )

    # delay_ids: values in [1, max_delay]
    delay_ids = rng.integers(
        1, max_delay + 1, size=(n_terms, n_degress), dtype=np.int32
    )
    delay_ids[feat_ids == -1] = -1

    # output_ids: values in [0, n_outputs-1].
    # Ensure at least one term per output so we don't have empty hessians
    output_ids = np.concatenate(
        [np.arange(n_outputs), rng.integers(0, n_outputs, size=n_terms - n_outputs)]
    )
    rng.shuffle(output_ids)
    output_ids = output_ids.astype(np.int32)

    sym_hess = get_sympy_hess(feat_ids, delay_ids, output_ids)
    narx_hess = get_narx_hess(feat_ids, delay_ids, output_ids)

    for out_i in range(len(sym_hess)):
        # Transform
        sym_norm = [normalize_term(t) for t in sym_hess[out_i]]
        narx_norm = [normalize_term(t) for t in narx_hess[out_i]]

        # Compare using sets to find unique structural differences
        s_sym = set(sym_norm)
        s_narx = set(narx_norm)

        intersection = s_sym.intersection(s_narx)
        diff_sym_h = s_sym - s_narx  # in sym but not narx
        diff_h_sym = s_narx - s_sym  # in narx but not sym
        c_sym = Counter(sym_norm)
        c_narx = Counter(narx_norm)

        assert len(sym_hess[out_i]) == len(narx_hess[out_i])
        assert len(s_sym) == len(s_narx)
        assert len(intersection) == len(s_sym)
        assert len(diff_sym_h) == 0
        assert len(diff_h_sym) == 0
        # Use Counter to check if elements are identical including duplicates
        assert c_sym == c_narx
