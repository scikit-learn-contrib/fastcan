"""Test feature selection with mini-batch"""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import OneHotEncoder

from fastcan import minibatch


def test_data_pruning():
    # Test large batch size issue for data pruning
    n_atoms = 10
    random_state = 0
    n_to_select = 100
    batch_size = 5

    X, _ = load_iris(return_X_y=True)
    kmeans = KMeans(
        n_clusters=n_atoms,
        random_state=random_state,
    ).fit(X)
    atoms = kmeans.cluster_centers_
    indices = minibatch(X.T, atoms.T, n_to_select, batch_size=batch_size, verbose=0)
    assert np.unique(indices).size == n_to_select
    assert indices.size == n_to_select


def test_select_minibatch_cls():
    # Test whether refine work correctly with random samples.
    n_samples = 200
    n_features = 30
    n_informative = 20
    n_classes = 8
    n_repeated = 5
    n_to_select = 18

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    indices = minibatch(X, y, n_to_select, batch_size=3)
    assert np.unique(indices).size == n_to_select
    assert indices.size == n_to_select

    Y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    indices = minibatch(X, Y, n_to_select, batch_size=2)
    assert np.unique(indices).size == n_to_select
    assert indices.size == n_to_select


def test_minibatch_error():
    # Test refine raise error.
    n_samples = 200
    n_features = 20
    n_informative = 10
    n_classes = 8
    n_repeated = 5

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    with pytest.raises(ValueError, match=r"n_features_to_select .*"):
        _ = minibatch(X, y, n_features + 1, batch_size=3)
