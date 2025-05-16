"""
=======================
Mulit-output NARX model
=======================

.. currentmodule:: fastcan

In this example, we illustrate how to build a multi-output polynomial
NARX model for time series prediction.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

# %%
# Prepare data
# ------------
#
# First, a simulated time series dataset is generated from the following nonlinear
# system.
#
# .. math::
#     y_0(k) &=\\
#     &0.5y_0(k-1) + 0.8y_1(k-1) + 0.3u_0(k)^2 + 2u_0(k-1)u_0(k-3) +
#     1.5u_0(k-2)u_1(k-3) + 1\\
#     y_1(k) &=\\
#     &0.6y_1(k-1) - 0.2y_0(k-1)y_1(k-2) + 0.3u_1(k)^2 + 1.5u_1(k-2)u_0(k-3)
#     + 0.5
#
#
# where :math:`k` is the time index,
# :math:`u_0` and :math:`u_1` are input signals,
# and :math:`y_0` and :math:`y_1` are output signals.

import numpy as np

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
        - 0.2 * y0[i - 1] * y1[i - 2]
        + 0.3 * u1[i] ** 2
        + 1.5 * u1[i - 2] * u0[i - 3]
        + 0.5
    )
y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
X = np.c_[u0[max_delay:], u1[max_delay:]]


# %%
# Identify the mulit-output NARX model
# ------------------------------------
# We provide :meth:`narx.make_narx` to automatically find the model
# structure. `n_terms_to_select` can be a list to indicate the number
# of terms (excluding intercept) for each output.

from fastcan.narx import make_narx, print_narx

narx_model = make_narx(
    X=X,
    y=y,
    n_terms_to_select=[5, 4],
    max_delay=3,
    poly_degree=2,
    verbose=0,
).fit(X, y)

print_narx(narx_model, term_space=30)


# %%
# Plot NARX prediction performance
# --------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y_pred = narx_model.predict(
    X[:100],
    y_init=y[: narx_model.max_delay_],  # Set the initial values of the prediction to
    # the true values
)

plt.plot(y[:100], label="True")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Time index k")
plt.legend(["y0 True", "y1 True", "y0 Pred", "y1 Pred"], loc="right")
plt.title(f"NARX prediction R-squared: {r2_score(y[:100], y_pred):.5f}")
plt.show()
