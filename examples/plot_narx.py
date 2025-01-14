"""
===============================================
Nonlinear AutoRegressive eXogenous (NARX) model
===============================================

.. currentmodule:: fastcan

In this example, we illustrate how to build a polynomial
NARX model for time series prediction.
"""

# Authors: Sikai Zhang
# SPDX-License-Identifier: MIT

# %%
# Prepare data
# ------------
#
# First, a simulated time series dataset is generated from the following nonlinear
# system.
#
# .. math::
#     y(k) = 0.5y(k-1) + 0.3u_0(k)^2 + 2u_0(k-1)u_0(k-3) + 1.5u_0(k-2)u_1(k-3) + 1
#
# where :math:`k` is the time index,
# :math:`u_0` and :math:`u_1` are input signals,
# and :math:`y` is the output signal.


import numpy as np

rng = np.random.default_rng(12345)
n_samples = 1000
max_delay = 3
e = rng.normal(0, 0.1, n_samples)
u0 = rng.uniform(0, 1, n_samples+max_delay)
u1 = rng.normal(0, 0.1, n_samples+max_delay)
y = np.zeros(n_samples+max_delay)
for i in range(max_delay, n_samples+max_delay):
    y[i] = 0.5*y[i-1]+0.3*u0[i]**2+2*u0[i-1]*u0[i-3]+1.5*u0[i-2]*u1[i-3]+1
y = y[max_delay:]+e
X = np.c_[u0[max_delay:], u1[max_delay:]]

# %%
# Build term libriary
# -------------------
# To build a reduced polynomial NARX model, it is normally have two steps:
#
# #. Search the structure of the model, i.e., the terms in the model, e.g.,
#    :math:`u_0(k-1)u_0(k-3)`, :math:`u_0(k-2)u_1(k-3)`, etc.
#
# #. Learn the coefficients of the terms.
#
# To search the structure of the model, the candidate term libriary should be
# constructed by the following two steps.
#
# #. Time-shifted variables: the raw input-output data, i.e., :math:`u_0(k)`,
#    :math:`u_1(k)`, and :math:`y(k)`, are converted into :math:`u_0(k-1)`,
#    :math:`u_1(k-2)`, etc.
#
# #. Nonlinear terms: the time-shifted variables are onverted to nonlinear terms
#    via polynomial basis functions, e.g., :math:`u_0(k-1)^2`,
#    :math:`u_0(k-1)u_0(k-3)`, etc.
#
#   .. rubric:: References
#
#   * `"Nonlinear system identification: NARMAX methods in the time, frequency,
#     and spatio-temporal domains" <https://doi.org/10.1002/9781118535561>`_
#     Billings, S. A. John Wiley & Sons, (2013).
#
#
# Make time-shifted variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

from fastcan.narx import make_time_shift_features, make_time_shift_ids

time_shift_ids = make_time_shift_ids(
    n_features=3, # Number of inputs (2) and output (1) signals
    max_delay=3, # Maximum time delays
    include_zero_delay = [True, True, False], # Whether to include zero delay
    # for each signal. The output signal should not have zero delay.
)

time_shift_vars = make_time_shift_features(np.c_[X, y], time_shift_ids)

# %%
# Make nonlinear terms
# ^^^^^^^^^^^^^^^^^^^^

from fastcan.narx import make_poly_features, make_poly_ids

poly_ids = make_poly_ids(
    n_features=time_shift_vars.shape[1], # Number of time-shifted variables
    degree=2, # Maximum polynomial degree
)

poly_terms = make_poly_features(time_shift_vars, poly_ids)

# %%
# Term selection
# --------------
# After the term library is constructed, the terms can be selected by :class:`FastCan`,
# whose :math:`X` is the nonlinear terms and :math:`y` is the output signal.

from fastcan import FastCan

selector = FastCan(
    n_features_to_select=4, # 4 terms should be selected
).fit(poly_terms, y)

support = selector.get_support()
selected_poly_ids = poly_ids[support]


# %%
# Build NARX model
# ----------------
# As the reduced polynomial NARX is a linear function of the nonlinear tems,
# the coefficient of each term can be easily estimated by oridnary least squares.
# In the printed NARX model, it is found that :class:`FastCan` selects the correct
# terms and the coefficients are close to the true values.

from fastcan.narx import NARX, print_narx

narx_model = NARX(
    time_shift_ids=time_shift_ids,
    poly_ids = selected_poly_ids,
)

narx_model.fit(X, y)

print_narx(narx_model)
# %%
# Automaticated NARX modelling workflow
# -------------------------------------
# We provide :meth:`narx.make_narx` to automaticate the workflow above.

from fastcan.narx import make_narx

auto_narx_model = make_narx(
    X=X,
    y=y,
    n_features_to_select=4,
    max_delay=3,
    poly_degree=2,
    verbose=0,
).fit(X, y)

print_narx(auto_narx_model)


# %%
# Plot NARX prediction performance
# --------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y_pred = narx_model.predict(
    X[:100],
    y_init=y[:narx_model.max_delay_] # Set the initial values of the prediction to
    # the true values
)

plt.plot(y[:100], label="True")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Time index k")
plt.legend()
plt.title(f"NARX prediction R-squared: {r2_score(y[:100], y_pred):.5f}")
plt.show()
