"""
=========================
Fisher's criterion in LDA
=========================

.. currentmodule:: fastcan

In this examples, we will demonstrate the canonical correaltion coefficient
between the features ``X`` and the one-hot encoded target ``y`` has equivalent
relationship with Fisher's criterion in LDA (Linear Discriminant Analysis).
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

# %%
# Prepare data
# ------------
# We use ``iris`` dataset and transform this multiclass data to multilabel data by
# one-hot encoding. Here, drop="first" is necessary; otherwise, the transformed target
# is not full column rank.

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

X, y = datasets.load_iris(return_X_y=True)
# drop="first" is necessary; otherwise, the transformed target is not full column rank
y_enc = OneHotEncoder(
    drop="first",
    sparse_output=False,
).fit_transform(y.reshape(-1, 1))

# %%
# Compute Fisher's criterion
# --------------------------
# The intermediate product of ``LinearDiscriminantAnalysis`` in ``sklearn`` is
# Fisher's criterion, when ``solver="eigen"``. However, it does not provide an interface
# to export it, so we reproduce it manually.

import numpy as np
from scipy import linalg
from sklearn.covariance import empirical_covariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(solver="eigen").fit(X, y)
Sw = clf.covariance_  # within scatter
St = empirical_covariance(X)  # total scatter
Sb = St - Sw  # between scatter
fishers_criterion, _ = linalg.eigh(Sb, Sw)

fishers_criterion = np.sort(fishers_criterion)[::-1]
n_nonzero = min(X.shape[1], clf.classes_.shape[0] - 1)
# remove the eigenvalues which are close to zero
fishers_criterion = fishers_criterion[:n_nonzero]
# get canonical correlation coefficients from convert Fisher's criteria
r2 = fishers_criterion / (1 + fishers_criterion)

# %%
# Compute SSC
# -----------
# Compute the sum of squared canonical correlation coefficients (SSC). It can be found
# that the result obtained by :class:`FastCan`/CCA (Canonical Correlation Analysis) is
# the same as LDA.

from fastcan import FastCan

ssc = FastCan(4, verbose=0).fit(X, y_enc).scores_.sum()

print(f"SSC from LDA: {r2.sum():5f}")
print(f"SSC from CCA: {ssc:5f}")
