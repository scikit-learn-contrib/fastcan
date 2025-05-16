.. currentmodule:: fastcan

.. _pruning:

===================================================
Dictionary learning based unsupervised data pruning
===================================================

Different from feature selection, which reduces the size of dataset in column-wise,
data pruning reduces the size of dataset in row-wise.
To use :class:`FastCan` for unsupervised data pruning, the target :math:`Y` matrix is
obtained first with `dictionary learning <https://scikit-learn.org/stable/modules/decomposition.html#dictionary-learning>`_.
Dictionary learning will learn a ``dictionary`` which is composed of atoms.
The atoms should be very representative, so that each sample of dataset can be represented (with errors)
by sparse linear combinations of the atoms.
We use these atoms as the target :math:`Y` and select samples based on their correlation with :math:`Y`.

One challenge to use :class:`FastCan` for data pruning is that the number to select is much larger than feature selection.
Normally, this number is higher than the number of features, which will make the pruned data matrix singular.
In other words, :class:`FastCan` will easily think the pruned data is redundant and no additional sample
should be selected, as any additional samples can be represented by linear combinations of the selected samples.
Therefore, the number to select has to be set to small.

To solve this problem, we use :func:`minibatch` to loose the redundancy check of :class:`FastCan`.
The original :class:`FastCan` checks the redundancy within :math:`X_s \in \mathbb{R}^{n\times t}`, 
which contains :math:`t` selected samples and n features,
and the redundancy within :math:`Y \in \mathbb{R}^{n\times m}`, which contains :math:`m` atoms :math:`y_i`.
:func:`minibatch` ranks samples with multiple correlation coefficients between :math:`X_b \in \mathbb{R}^{n\times b}` and :math:`y_i`,
where :math:`b` is batch size and :math:`b <= t`, instead of canonical correlation coefficients between :math:`X_s` and :math:`Y`,
which is used in :class:`FastCan`.
Therefore, :func:`minibatch` looses the redundancy check in two ways.

#. it uses :math:`y_i` instead of :math:`Y`, so no redundancy check is performed within :math:`Y`
#. it uses :math:`X_b` instead of :math:`X_s`, so :func:`minibatch` only checks the redundancy within a batch :math:`X_b`, but does not
   check the redundancy between batches.


.. rubric:: References

* `"Dictionary-learning-based data pruning for system identification"
  <https://doi.org/10.48550/arXiv.2502.11484>`_
  Wang, T., Zhang, S., & Sun L.
  arXiv (2025).


.. rubric:: Examples

* See :ref:`sphx_glr_auto_examples_plot_pruning.py` for an example of dictionary learning based data pruning.
