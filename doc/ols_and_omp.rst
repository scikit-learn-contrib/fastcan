.. currentmodule:: fastcan

.. _ols_omp:

===========================
Comparison with OLS and OMP
===========================

:class:`FastCan` has a close relationship with Orthogonal Least Squares (OLS) [1]_
and Orthogonal Matching Pursuit (OMP) [2]_.
The detailed difference between OLS and OMP can be found in [3]_.
Here, let's briefly compare the three methods.


Assume we have a feature matrix :math:`X_s \in \mathbb{R}^{N\times t}`, which contains
:math:`t` selected features, and a target vector :math:`y \in \mathbb{R}^{N\times 1}`.
Then the residual :math:`r \in \mathbb{R}^{N\times 1}` of the least-squares can be
found by

.. math::
    r = y - X_s \beta \;\; \text{where} \;\; \beta =  (X_s^\top X_s)^{-1}X_s^\top y

When evaluating a new candidate feature :math:`x_i \in \mathbb{R}^{N\times 1}`

* for OMP, the feature which maximizes :math:`r^\top x_i` will be selected,
* for OLS, the feature which maximizes :math:`r^\top w_i` will be selected, where
  :math:`w_i \in \mathbb{R}^{N\times 1}` is the projection of :math:`x_i` on the
  orthogonal subspace so that it is orthogonal to :math:`X_s`, i.e.,
  :math:`X_s^\top w_i = \mathbf{0} \in \mathbb{R}^{t\times 1}`,
* for :class:`FastCan` (h-correlation algorithm), it is almost same as OLS, but the
  difference is that in :class:`FastCan`, :math:`X_s`, :math:`y`, and :math:`x_i`
  are centered (i.e., zero mean in each column) before the selection.

The small difference makes the feature ranking criterion of :class:`FastCan` is
equivalent to the sum of squared canonical correlation coefficients, which gives
it the following advantages over OLS and OMP:

* Affine invariance: if features are polluted by affine transformation, i.e., scaled
  and/or added some constants, the selection result given by :class:`FastCan` will be
  unchanged. See :ref:`sphx_glr_auto_examples_plot_affinity.py`.
* Multioutput: as :class:`FastCan` use canonical correlation for feature ranking, it is
  naturally support feature selection on dataset with multioutput.


.. rubric:: References

.. [1] `"Orthogonal least squares methods and their application to non-linear
    system identification" <https://doi.org/10.1080/00207178908953472>`_ Chen, S.,
    Billings, S. A., & Luo, W. International Journal of control, 50(5),
    1873-1896 (1989).

.. [2] `"Matching pursuits with time-frequency dictionaries"
    <https://doi.org/10.1109/78.258082>`_ Mallat, S. G., & Zhang, Z.
    IEEE Transactions on signal processing, 41(12), 3397-3415 (1993).

.. [3] `"On the difference between Orthogonal Matching Pursuit and Orthogonal Least
    Squares" <https://eprints.soton.ac.uk/142469/1/BDOMPvsOLS07.pdf>`_ Blumensath, T.,
    & Davies, M. E. Technical report, University of Edinburgh, (2007).