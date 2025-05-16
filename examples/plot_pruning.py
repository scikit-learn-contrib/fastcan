"""
============
Data pruning
============

.. currentmodule:: fastcan

This example shows how to prune dataset with :func:`minibatch`
based on :class:`FastCan`.
The method is compared to random data pruning.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

# %%
# Load data and prepare baseline
# ------------------------------
# We use ``iris`` dataset and logistic regression model to demonstrate data pruning.
# The baseline model is a logistic regression model trained on the entire dataset.
# Here, 110 samples are used as the training data, which is intentionally made
# imbalanced, to test data pruning methods.
# The coefficients of the model trained on the pruned dataset will be compared to
# the baseline model with R-squared score.
# The higher R-squared score, the better the pruning.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
baseline_lr = LogisticRegression(max_iter=1000).fit(iris["data"], iris["target"])
X_train = iris["data"][10:120]
y_train = iris["target"][10:120]

# %%
# Random data pruning
# -------------------
# There are 110 samples in the training dataset.
# The random pruning method selected samples assuming a uniform distribution
# over all data.

import numpy as np


def _random_pruning(X, y, n_samples_to_select: int, random_state: int):
    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(y.size, n_samples_to_select, replace=False)
    pruned_lr = LogisticRegression(max_iter=1000).fit(X[ids_random], y[ids_random])
    return pruned_lr.coef_, pruned_lr.intercept_


# %%
# FastCan data pruning
# --------------------
# To use :class:`FastCan` to prune the data, there are two steps:
#
# #. Learn the atoms by Dictionary Learning (here we use ``KMeans``)
# #. Select the samples by :func:`minibatch` according to the multiple correlation
#    between each atom and the batch of samples.

from sklearn.cluster import KMeans

from fastcan import minibatch


def _fastcan_pruning(
    X,
    y,
    n_samples_to_select: int,
    random_state: int,
    n_atoms: int,
    batch_size: int,
):
    kmeans = KMeans(
        n_clusters=n_atoms,
        random_state=random_state,
    ).fit(X)
    atoms = kmeans.cluster_centers_
    ids_fastcan = minibatch(
        X.T, atoms.T, n_samples_to_select, batch_size=batch_size, verbose=0
    )
    pruned_lr = LogisticRegression(max_iter=1000).fit(X[ids_fastcan], y[ids_fastcan])
    return pruned_lr.coef_, pruned_lr.intercept_


# %%
# Visualize selected samples
# --------------------------------------------------
# Use principal component analysis (PCA) to visualize the distribution of the samples,
# and to compare the difference between the selection of ``Random`` pruning and
# ``FastCan`` pruning.
# For clearer viewing of the selection, only 10 samples are selected from the training
# data by the pruning methods.
# The results show that ``FastCan`` selects 3 setosa, 4 versicolor,
# and 3 virginica, while ``Random`` select 6, 2, and 2, respectively.
# The imbalanced selection of ``Random`` is caused by the imbalanced training data,
# while ``FastCan``, benefited from the dictionary learning (k-means), can overcome
# the imbalance issue.

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(X, y, target_names, n_samples_to_select, random_state):
    pca = PCA(2).fit(X)
    pcs_all = pca.transform(X)

    kmeans = KMeans(
        n_clusters=10,
        random_state=random_state,
    ).fit(X)
    atoms = kmeans.cluster_centers_
    pcs_atoms = pca.transform(atoms)

    ids_fastcan = minibatch(X.T, atoms.T, n_samples_to_select, batch_size=1, verbose=0)
    pcs_fastcan = pca.transform(X[ids_fastcan])

    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(X.shape[0], n_samples_to_select, replace=False)
    pcs_random = pca.transform(X[ids_random])

    plt.scatter(pcs_fastcan[:, 0], pcs_fastcan[:, 1], s=50, marker="o", label="FastCan")
    plt.scatter(pcs_random[:, 0], pcs_random[:, 1], s=50, marker="*", label="Random")
    plt.scatter(pcs_atoms[:, 0], pcs_atoms[:, 1], s=100, marker="+", label="Atoms")
    cmap = plt.get_cmap("Dark2")
    for i, label in enumerate(target_names):
        mask = y == i
        plt.scatter(
            pcs_all[mask, 0], pcs_all[mask, 1], s=5, label=label, color=cmap(i + 2)
        )
    plt.xlabel("The First Principle Component")
    plt.ylabel("The Second Principle Component")
    plt.legend(ncol=2)


plot_pca(X_train, y_train, iris.target_names, 10, 123)

# %%
# Compare pruning methods
# -----------------------
# 80 samples are selected from 110 training data with ``Random`` pruning and
# ``FastCan`` pruning. The results show that ``FastCan`` pruning gives a higher
# median value of R-squared and a lower standard deviation.

from sklearn.metrics import r2_score


def plot_box(X, y, baseline, n_samples_to_select: int, n_random: int):
    r2_fastcan = np.zeros(n_random)
    r2_random = np.zeros(n_random)
    for i in range(n_random):
        coef, intercept = _fastcan_pruning(
            X, y, n_samples_to_select, i, n_atoms=40, batch_size=2
        )
        r2_fastcan[i] = r2_score(
            np.c_[coef, intercept], np.c_[baseline.coef_, baseline.intercept_]
        )

        coef, intercept = _random_pruning(X, y, n_samples_to_select, i)
        r2_random[i] = r2_score(
            np.c_[coef, intercept], np.c_[baseline.coef_, baseline.intercept_]
        )

    plt.boxplot(np.c_[r2_fastcan, r2_random])
    plt.ylabel("R2")
    plt.xticks(ticks=[1, 2], labels=["FastCan", "Random"])
    plt.show()


plot_box(X_train, y_train, baseline_lr, n_samples_to_select=80, n_random=100)
