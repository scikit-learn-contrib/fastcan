"""Test refine"""

import pytest
from sklearn.datasets import make_classification

from fastcan import FastCan, refine


def test_select_refine_cls():
    # Test whether refine work correctly with random samples.
    n_samples = 200
    n_features = 20
    n_informative = 10
    n_classes = 8
    n_repeated = 5
    n_to_select = 10

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

    selector = FastCan(n_to_select).fit(X, y)
    _, scores_1 = refine(selector, drop=1)
    _, scores_23 = refine(selector, drop=[2, 3], verbose=0)
    _, scores_all = refine(selector, drop="all", max_iter=20, verbose=1)

    selector = FastCan(n_to_select, indices_include=[1, 5]).fit(X, y)
    indices_inc, _ = refine(selector, drop=1)

    assert selector.scores_.sum() <= scores_1.sum()
    assert selector.scores_.sum() <= scores_23.sum()
    assert selector.scores_.sum() <= scores_all.sum()
    assert (indices_inc[0] == 1) and (indices_inc[1] == 5)


def test_refine_error():
    # Test refine raise error.
    n_samples = 200
    n_features = 20
    n_informative = 10
    n_classes = 8
    n_repeated = 5
    n_to_select = 10

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

    selector = FastCan(n_to_select, indices_include=[0])
    selector.fit(X, y)

    with pytest.raises(ValueError, match=r"`drop` should be between .*"):
        refine(selector, drop=n_to_select, verbose=0)
