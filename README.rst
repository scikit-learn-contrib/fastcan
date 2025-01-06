FastCan: A Fast Canonical-Correlation-Based Feature Selection Algorithm
=======================================================================
|conda| |Codecov| |CI| |Doc| |PythonVersion| |PyPi| |Black| |ruff| |pixi|

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/fastcan.svg
   :target: https://anaconda.org/conda-forge/fastcan

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/fastcan/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/fastcan

.. |CI| image:: https://github.com/scikit-learn-contrib/fastcan/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/scikit-learn-contrib/fastcan/actions

.. |Doc| image:: https://readthedocs.org/projects/fastcan/badge/?version=latest
   :target: https://fastcan.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/fastcan.svg
   :target: https://pypi.org/project/fastcan/

.. |PyPi| image:: https://img.shields.io/pypi/v/fastcan
   :target: https://pypi.org/project/fastcan

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff

.. |pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
   :target: https://pixi.sh

FastCan is a feature selection method, which has following advantages:

#. Extremely **fast**.

#. Support unsupervised feature selection.

#. Support multioutput feature selection.

#. Skip redundant features.

#. Evaluate relative usefulness of features.

Check `Home Page <https://fastcan.readthedocs.io/en/latest/?badge=latest>`_ for more information.

Installation
------------

Install **FastCan** via PyPi:

* Run ``pip install fastcan``

Or via conda-forge:

* Run ``conda install -c conda-forge fastcan``

Getting Started
---------------
>>> from fastcan import FastCan
>>> X = [[ 0.87, -1.34,  0.31 ],
...     [-2.79, -0.02, -0.85 ],
...     [-1.34, -0.48, -2.55 ],
...     [ 1.92,  1.48,  0.65 ]]
>>> y = [[0, 0], [1, 1], [0, 0], [1, 0]] # Multioutput feature selection
>>> selector = FastCan(n_features_to_select=2, verbose=0).fit(X, y)
>>> selector.get_support()
array([ True,  True, False])
>>> selector.get_support(indices=True) # Sorted indices
array([0, 1])
>>> selector.indices_ # Indices in selection order
array([1, 0], dtype=int32)
>>> selector.scores_ # Scores for selected features in selection order
array([0.91162413, 0.71089547])
>>> # Here Feature 2 must be included
>>> selector = FastCan(n_features_to_select=2, indices_include=[2], verbose=0).fit(X, y)
>>> # We can find the feature which is useful when working with Feature 2
>>> selector.indices_
array([2, 0], dtype=int32)
>>> selector.scores_
array([0.34617598, 0.95815008])


NARX Time Series Modelling
--------------------------
FastCan can be used for system identification.
In particular, we provide a submodule `fastcan.narx` to build Nonlinear AutoRegressive eXogenous (NARX) models.
For more information, check our `Home Page <https://fastcan.readthedocs.io/en/latest/?badge=latest>`_.


Citation
--------

FastCan is a Python implementation of the following papers.

If you use the `h-correlation` method in your work please cite the following reference:

.. code:: bibtex

   @article{ZHANG2022108419,
      title = {Orthogonal least squares based fast feature selection for linear classification},
      journal = {Pattern Recognition},
      volume = {123},
      pages = {108419},
      year = {2022},
      issn = {0031-3203},
      doi = {https://doi.org/10.1016/j.patcog.2021.108419},
      url = {https://www.sciencedirect.com/science/article/pii/S0031320321005951},
      author = {Sikai Zhang and Zi-Qiang Lang},
      keywords = {Feature selection, Orthogonal least squares, Canonical correlation analysis, Linear discriminant analysis, Multi-label, Multivariate time series, Feature interaction},
      }

If you use the `eta-cosine` method in your work please cite the following reference:

.. code:: bibtex

   @article{ZHANG2025111895,
      title = {Canonical-correlation-based fast feature selection for structural health monitoring},
      journal = {Mechanical Systems and Signal Processing},
      volume = {223},
      pages = {111895},
      year = {2025},
      issn = {0888-3270},
      doi = {https://doi.org/10.1016/j.ymssp.2024.111895},
      url = {https://www.sciencedirect.com/science/article/pii/S0888327024007933},
      author = {Sikai Zhang and Tingna Wang and Keith Worden and Limin Sun and Elizabeth J. Cross},
      keywords = {Multivariate feature selection, Filter method, Canonical correlation analysis, Feature interaction, Feature redundancy, Structural health monitoring},
      }
