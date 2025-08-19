.. currentmodule:: fastcan

.. _unsupervised:

==============================
Unsupervised feature selection
==============================

We can use :class:`FastCan` to do unsupervised feature selection.
The unsupervised application of :class:`FastCan` tries to select features, which
maximize the sum of the squared canonical correlation (SSC) with the principal
components (PCs) acquired from PCA (principal component analysis) of the feature
matrix :math:`X`. See the example below.

    >>> from sklearn.decomposition import PCA
    >>> from sklearn import datasets
    >>> from fastcan import FastCan
    >>> iris = datasets.load_iris()
    >>> X = iris["data"]
    >>> pca = PCA(n_components=2)
    >>> X_pcs = pca.fit_transform(X)
    >>> selector = FastCan(n_features_to_select=2, verbose=0).fit(X, X_pcs[:, :2])
    >>> selector.indices_
    array([2, 1], dtype=int32)

.. note::
    There is no guarantee that this unsupervised :class:`FastCan` will select
    the optimal subset of the features, which has the highest SSC with PCs.
    Because :class:`FastCan` selects features in a greedy manner, which may lead to
    suboptimal results.

.. rubric:: References

* `"Manifold learning-based unsupervised feature selection for structural health
  monitoring" <https://mrforum.com/wp-content/uploads/open_access/9781644903513/9.pdf>`_
  Wang, T., & Sun, L.
  Materials Research Proceedings, 50, 82-89, (2025).

* `"Automatic Selection of Optimal Structures for Population-Based
  Structural Health Monitoring" <https://doi.org/10.1007/978-3-031-34946-1_10>`_
  Wang, T., Worden, K., Wagg, D.J., Cross, E.J., Maguire, A.E., & Lin, W.
  In: Conference Proceedings of the Society for Experimental Mechanics Series.
  Springer, Cham. (2023).