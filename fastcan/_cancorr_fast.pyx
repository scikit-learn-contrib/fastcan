"""
Fast feature selection with sum squared canonical correlation coefficients
"""
# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython cimport floating, final
from cython.parallel import prange
from scipy.linalg.cython_blas cimport isamax, idamax
from sklearn.utils._cython_blas cimport ColMajor, NoTrans
from sklearn.utils._cython_blas cimport _dot, _scal, _nrm2, _gemm, _axpy
from sklearn.utils._typedefs cimport uint8_t


@final
cdef int _bsum(
    const uint8_t* x,
    int n,
) noexcept nogil:
    """Computes the sum of the vector of bool elements.
    """
    cdef:
        int total = 0
        int i
    for i in range(n):
        total += x[i]
    return total


@final
cdef int _iamax(
    int n,
    const floating* x,
    int incx,
) noexcept nogil:
    """
    Given a vector x, the function return the position of
    the vector element x(i) that has the largest absolute value for
    real flavors.
    """
    if floating is float:
        return isamax(&n, <float *> x, &incx) - 1
    else:
        return idamax(&n, <double *> x, &incx) - 1


@final
cdef bint _normv(
    const floating* x,          # IN/OUT
    int n_samples,              # IN
) noexcept nogil:
    """
    Vector normalization by Euclidean norm.
    x (IN) : (n_samples,) Vector.
    x (OUT) : (n_samples,) Normalized vector.
    n_samples (IN) : number of samples.
    Return: Mask the constant vector.
    """
    cdef:
        floating x_norm

    x_norm = _nrm2(n_samples, x, 1)
    if x_norm == 0.0:
        return True
    x_norm = 1.0/x_norm
    _scal(n_samples, x_norm, x, 1)
    return False


@final
cdef floating _sscvm(
    const floating* w,          # IN
    const floating* V,          # IN
    int n_samples,              # IN
    int n_targets,              # IN
) noexcept nogil:
    """
    Sum of squared correlation coefficients.
    w : (n_samples,) Centred orthogonalized feature vector.
    V : (n_samples, n_targets) Centred orthogonalized target matrix.
    n_samples (IN) : number of samples.
    n_targets (IN) : column number of V
    r2 : (nw,) Sum of squared correlation coefficients, where r2i means the
        coefficient of determination between wi and V.
    """
    cdef:
        # R : (nw * nv) R**2 contains the pairwise h-correlation or eta-cosine, where
        #     rij means the h-correlation or eta-cosine between wi and vj.
        floating* r = <floating*> malloc(sizeof(floating) * n_targets)
        floating r2

    # r = w*V (w is treated as (1, n_samples))
    _gemm(ColMajor, NoTrans, NoTrans, 1, n_targets, n_samples, 1.0,
          w, 1, V, n_samples, 0.0, r, 1)
    # r2 = r*r.T

    r2 = _dot(n_targets, r, 1, r, 1)

    free(r)
    return r2


@final
cdef void _mgsvv(
    const floating* w,          # IN
    const floating* x,          # IN/OUT
    int n_samples,              # IN
) noexcept nogil:
    """
    Modified Gram-Schmidt process. x = x - w*w.T*x
    w : (n_samples,) Centred orthonormal selected feature vector.
    x (IN) : (n_samples,) Centred remaining feature vector.
    x (OUT) : (n_samples,) Centred remaining feature vector, which is orthogonal to w.
    n_samples (IN) : number of samples.
    """
    cdef:
        floating r

    # r = w.T*x
    r = _dot(n_samples, w, 1, x, 1)
    # x = x - w*r
    _axpy(n_samples, -r, w, 1, x, 1)


@final
cpdef int _forward_search(
    floating[::1, :] X,               # IN/OUT
    floating[::1, :] V,               # IN
    int t,                            # IN
    floating tol,                     # IN
    int num_threads,                  # IN
    int verbose,                      # IN
    uint8_t[::1] mask,                # IN/TEMP
    int[::1] indices,                 # OUT
    floating[::1] scores,             # OUT
) except -1 nogil:
    """
    Greedy search with SSC.
    X (IN) : (n_samples, n_features) Feature matrix.
    V (IN) : (n_samples, n_outputs) Orthonormal target matrix.
    W (OUT) : (n_samples, n_features) Centered normalized feature matrix, which
              is orthonormal to selected features and M.
    t : Non-negative integer. The number of features to be selected.
    tol : Tolerance for linear dependence check.
    mask (n_features, ) Mask for candidate features.
    indices: (t,) The indices vector of selected features, initiated with -1.
    scores: (t,) The h-correlation/eta-cosine of selected features.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_targets = V.shape[1]
        # OpenMP (in Windows) requires signed integral for prange
        int n_features = X.shape[1]
        floating* r2 = <floating*> malloc(sizeof(floating) * n_features)
        floating g, ssc = 0.0
        int i, j
        int index = -1

    memset(&r2[0], 0, n_features * sizeof(floating))

    for i in range(t):
        if i == 0:
            # Preprocessing
            for j in range(n_features):
                if not mask[j]:
                    mask[j] = _normv(&X[0, j], n_samples)
        else:
            mask[index] = True
            r2[index] = 0
            # Make X orthogonal to X[:, indices[i-1]]
            for j in prange(n_features, nogil=True, schedule="static",
                            chunksize=1, num_threads=num_threads):
                if not mask[j]:
                    _mgsvv(&X[0, index], &X[0, j], n_samples)
                    _normv(&X[0, j], n_samples)
                    # Linear dependence check
                    g = _dot(n_samples, &X[0, index], 1, &X[0, j], 1)
                    if abs(g) > tol:
                        mask[j] = True
                        r2[j] = 0

            if _bsum(&mask[0], n_features) == n_features:
                raise RuntimeError(
                    "No candidate feature can be found to form a non-singular "
                    f"matrix with the {i} selected features."
                )
        if indices[i] != -1:
            index = indices[i]
            scores[i] = _sscvm(&X[0, index], &V[0, 0], n_samples, n_targets)
        else:
            # Score for X
            for j in range(n_features):
                if not mask[j]:
                    r2[j] = _sscvm(&X[0, j], &V[0, 0], n_samples, n_targets)

            # Find max scores and update indices, X, mask, and scores
            index = _iamax(n_features, &r2[0], 1)
            if r2[index] == 0:
                raise RuntimeError(
                    f"No improvement can be found when selecting the {i}th feature."
                )
            indices[i] = index
            scores[i] = r2[index]

        ssc += scores[i]
        if verbose == 1:
            with gil:
                print(f"Progress: {i+1}/{t}, SSC: {ssc:.5f}", end="\r")

    if verbose == 1:
        with gil:
            print()
    free(r2)
    return 0
