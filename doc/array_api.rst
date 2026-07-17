.. _array_api:

========================
Array API support status
========================

.. currentmodule:: fastcan

The adoption of the array API standard allows fastcan to be accelerated by GPUs.
More detailed information about the introduction and the usage of the array API standard can be found in `Array API support <https://scikit-learn.org/stable/modules/array_api.html>`_.
Array API support status for estimators and tools in fastcan:

- ✅ :func:`utils.ols`
- ✅ :class:`LazyFastCan`
- ✅ :func:`narx.gen_time_shift_features`
- ✅ :func:`narx.gen_poly_features`
- ✅ :func:`narx.make_narx` (with `lazy=True`)
- ✅ :func:`narx.make_time_shift_features`
- ✅ :func:`narx.make_poly_features`
- ✅ :func:`utils.mask_missing_values`
- ⏳ :class:`FastCan`
- ⏳ :class:`narx.NARX`
- ⏳ :func:`minibatch`