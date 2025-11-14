.. currentmodule:: fastcan

.. _multioutput:

==============================
Multioutput feature selection
==============================

We can use :class:`FastCan` to handle multioutput feature selection, which means
target ``y`` can be a matrix. For regression, :class:`FastCan` can be used for
MIMO (Multi-Input Multi-Output) data. For classification, it can be used for
multilabel data. Actually, for multiclass classification, which has one output with
multiple categories, multioutput feature selection can also be useful. The multiclass
classification can be converted to multilabel classification by one-hot encoding
target ``y``. The canonical correlation coefficient between the features ``X`` and the
one-hot encoded target ``y`` has equivalent relationship with Fisher's criterion in
LDA (Linear Discriminant Analysis) [1]_. Applying :class:`FastCan` to the converted
multioutput data may result in better accuracy in the following classification task
than applying it directly to the original single-label data. See Figure 5 in [2]_.

Relationship on multiclass data
===============================
Assume the feature matrix is :math:`X \in \mathbb{R}^{N\times n}`, the multiclass
target vector is :math:`y \in \mathbb{R}^{N\times 1}`, and the one-hot encoded target
matrix is :math:`Y \in \mathbb{R}^{N\times m}`. Then, the Fisher's criterion for
:math:`X` and :math:`y` is denoted as :math:`J` and the canonical correlation
coefficient between :math:`X` and :math:`Y` is denoted as :math:`R`. The relationship
between :math:`J` and :math:`R` is given by

.. math::
    J = \frac{R^2}{1-R^2}

or

.. math::
    R^2 = \frac{J}{1+J}

It should be noted that the number of the Fisher's criterion and the canonical
correlation coefficient is not only one. The number of the non-zero canonical
correlation coefficients is no more than :math:`\min (n, m)`, and each canonical correlation
coefficient is one-to-one correspondence to each Fisher's criterion.

.. rubric:: References

.. [1] `"Orthogonal least squares based fast feature selection for
  linear classification" <https://doi.org/10.1016/j.patcog.2021.108419>`_
  Zhang, S., & Lang, Z. Q. Pattern Recognition, 123, 108419 (2022).

.. [2] `"Canonical-correlation-based fast feature selection for structural
  health monitoring" <https://doi.org/10.1016/j.ymssp.2024.111895>`_
  Zhang, S., Wang, T., Worden, K., Sun L., & Cross, E. J.
  Mechanical Systems and Signal Processing, 223, 111895 (2025).

.. rubric:: Examples

* See :ref:`sphx_glr_auto_examples_plot_fisher.py` for an example of
  the equivalent relationship between CCA and LDA on multiclass data.
