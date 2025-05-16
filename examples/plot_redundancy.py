"""
=================================
Performance on redundant features
=================================

.. currentmodule:: fastcan

In this examples, we will compare the performance of feature selectors on the
datasets, which contain redundant features.
Here four types of features should be distinguished:

* Useless features: the features do not contribute to the target
* Dependent informative features: the features contribute to the target and form
  the redundant features
* Redundant features: the features are constructed by linear transformation of
  dependent informative features
* Independent informative features: the features contribute to the target but
  does not contribute to the redundant features.

.. note::
    If we do not distinguish dependent and independent informative features and use
    informative features to form both the target and the redundant features. The task
    will be much easier.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

# %%
# Define dataset generator
# ------------------------

import numpy as np


def make_redundant(
    n_samples,
    n_features,
    dep_info_ids,
    indep_info_ids,
    redundant_ids,
    random_seed,
):
    """Make a dataset with linearly redundant features.

    Parameters
    ----------
    n_samples : int
        The number of samples.

    n_features : int
        The number of features.

    dep_info_ids : list[int]
        The indices of dependent informative features.

    indep_info_ids : list[int]
        The indices of independent informative features.

    redundant_ids : list[int]
        The indices of redundant features.

    random_seed : int
        Random seed.

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.
    """
    rng = np.random.default_rng(random_seed)
    info_ids = dep_info_ids + indep_info_ids
    n_dep_info = len(dep_info_ids)
    n_info = len(info_ids)
    n_redundant = len(redundant_ids)
    informative_coef = rng.random(n_info)
    redundant_coef = rng.random((n_dep_info, n_redundant))

    X = rng.random((n_samples, n_features))
    y = np.dot(X[:, info_ids], informative_coef)

    X[:, redundant_ids] = X[:, dep_info_ids] @ redundant_coef
    return X, y


# %%
# Define score function
# ---------------------
# This function is used to compute the number of correct features missed by selectors.
#
# * For independent informative features, selectors should select all of them.
# * For dependent informative features, selectors only need to select any
#   ``n_dep_info``-combination of the set ``dep_info_ids`` + ``redundant_ids``. That
#   means if the indices of dependent informative features are :math:`[0, 1]` and the
#   indices of the redundant features are :math:`[5]`, then the correctly selected
#   indices can be any of :math:`[0, 1]`, :math:`[0, 5]`, and :math:`[1, 5]`.


def get_n_missed(dep_info_ids, indep_info_ids, redundant_ids, selected_ids):
    """Get the number of features miss selected."""
    n_redundant = len(redundant_ids)
    n_missed_indep = len(np.setdiff1d(indep_info_ids, selected_ids))
    n_missed_dep = (
        len(np.setdiff1d(dep_info_ids + redundant_ids, selected_ids)) - n_redundant
    )
    n_missed_dep = max(n_missed_dep, 0)
    return n_missed_indep + n_missed_dep


# %%
# Prepare selectors
# -----------------
# We compare :class:`FastCan` with eight selectors of :mod:`sklearn`, which
# include the Select From a Model (SFM) algorithm, the Recursive Feature Elimination
# (RFE) algorithm, the Sequential Feature Selection (SFS) algorithm, and Select K Best
# (SKB) algorithm.
# The list of the selectors are given below:
#
# * fastcan: :class:`FastCan` selector
# * skb_reg: is the SKB algorithm ranking features with ANOVA (analysis of variance)
#   F-statistic and p-values
# * skb_mir: is the SKB algorithm ranking features mutual information for regression
# * sfm_lsvr: the SFM algorithm with a linear support vector regressor
# * sfm_rfr: the SFM algorithm with a random forest regressor
# * rfe_lsvr: is the RFE algorithm with a linear support vector regressor
# * rfe_rfr: is the RFE algorithm with a random forest regressor
# * sfs_lsvr: is the forward SFS algorithm with a linear support vector regressor
# * sfs_rfr: is the forward SFS algorithm with a random forest regressor


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
)
from sklearn.svm import LinearSVR

from fastcan import FastCan

lsvr = LinearSVR(C=1, dual="auto", max_iter=100000, random_state=0)
rfr = RandomForestRegressor(n_estimators=10, random_state=0)


N_SAMPLES = 1000
N_FEATURES = 10
DEP_INFO_IDS = [2, 4, 7, 9]
INDEP_INFO_IDS = [0, 1, 6]
REDUNDANT_IDS = [5, 8]
N_SELECTED = len(DEP_INFO_IDS + INDEP_INFO_IDS)
N_REPEATED = 10

selector_dict = {
    # Smaller `tol` makes fastcan more sensitive to redundancy
    "fastcan": FastCan(N_SELECTED, tol=1e-7, verbose=0),
    "skb_reg": SelectKBest(f_regression, k=N_SELECTED),
    "skb_mir": SelectKBest(mutual_info_regression, k=N_SELECTED),
    "sfm_lsvr": SelectFromModel(lsvr, max_features=N_SELECTED, threshold=-np.inf),
    "sfm_rfr": SelectFromModel(rfr, max_features=N_SELECTED, threshold=-np.inf),
    "rfe_lsvr": RFE(lsvr, n_features_to_select=N_SELECTED, step=1),
    "rfe_rfr": RFE(rfr, n_features_to_select=N_SELECTED, step=1),
    "sfs_lsvr": SequentialFeatureSelector(lsvr, n_features_to_select=N_SELECTED, cv=2),
    "sfs_rfr": SequentialFeatureSelector(rfr, n_features_to_select=N_SELECTED, cv=2),
}

# %%
# Run test
# --------

import time

N_SELECTORS = len(selector_dict)
n_missed = np.zeros((N_REPEATED, N_SELECTORS), dtype=int)
elapsed_time = np.zeros((N_REPEATED, N_SELECTORS), dtype=float)

for i in range(N_REPEATED):
    data, target = make_redundant(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        dep_info_ids=DEP_INFO_IDS,
        indep_info_ids=INDEP_INFO_IDS,
        redundant_ids=REDUNDANT_IDS,
        random_seed=i,
    )
    for j, selector in enumerate(selector_dict.values()):
        start_time = time.time()
        result_ids = selector.fit(data, target).get_support(indices=True)
        elapsed_time[i, j] = time.time() - start_time
        n_missed[i, j] = get_n_missed(
            dep_info_ids=DEP_INFO_IDS,
            indep_info_ids=INDEP_INFO_IDS,
            redundant_ids=REDUNDANT_IDS,
            selected_ids=result_ids,
        )

# %%
# Plot results
# ------------
# :class:`FastCan` correctly selects all informative features with zero missed
# features.

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
ax1.set_ylabel("No. of missed features")
ax2.set_ylabel("Elapsed time (s)")
rects = ax1.bar(selector_dict.keys(), n_missed.sum(0), width=0.5)
ax1.bar_label(rects, n_missed.sum(0), padding=3)
ax2.semilogy(selector_dict.keys(), elapsed_time.mean(0), marker="o", color="tab:orange")
plt.xlabel("Selector")
plt.title("Performance of selectors on datasets with linearly redundant features")
plt.show()
