.. _redundancy:

==================
Feature redundancy
==================

.. currentmodule:: fastcan

:class:`FastCan` can effectively skip the linearly redundant features.
Here a feature :math:`x_r\in \mathbb{R}^{N\times 1}` is linearly
redundant to a set of features :math:`X\in \mathbb{R}^{N\times n}` means that
:math:`x_r` can be obtained from an affine transformation of :math:`X`, given by

.. math::
    x_r = Xa + b

where :math:`a\in \mathbb{R}^{n\times 1}` and :math:`b\in \mathbb{R}^{N\times 1}`.
In other words, the feature can be acquired by a linear transformation of :math:`X`,
i.e. :math:`Xa`, and a translation, i.e. :math:`+b`.

This capability of :class:`FastCan` is benefited from the
`Modified Gram-Schmidt <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`_,
which gives large rounding-errors when linearly redundant features appears.

.. rubric:: References

* `"Canonical-correlation-based fast feature selection for structural
  health monitoring" <https://doi.org/10.1016/j.ymssp.2024.111895>`_
  Zhang, S., Wang, T., Worden, K., Sun L., & Cross, E. J.
  Mechanical Systems and Signal Processing, 223, 111895 (2025).

.. rubric:: Examples

* See :ref:`sphx_glr_auto_examples_plot_redundancy.py` for an example of
  feature selection on datasets with redundant features.
