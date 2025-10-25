"""
Beam search.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.linalg import orth


def _beam_search(
    X, V, n_features_to_select, beam_width, indices_include, mask_exclude, tol, verbose
):
    """
    Perform beam search to find the best subset of features.

    Parameters:
    X : np.ndarray
        The transformed input feature matrix.
    V : np.ndarray
        The transformed target variable.
    n_features_to_select : int
        The total number of features to select.
    beam_width : int
        The number of top candidates to keep at each step.
    indices_include : list
        The indices of features that must be included in the selection.
    mask_exclude : np.ndarray, dtype=bool
        A boolean mask indicating which features to exclude.
    tol : float
        Tolerance for numerical stability in Gram-Schmidt process.
    verbose : bool
        If True, print progress information.

    Returns:
    indices : np.ndarray, dtype=np.int32
        The indices of the selected features.
    """

    n_features = X.shape[1]
    n_inclusions = len(indices_include)

    X, _ = _safe_normalize(X)

    for i in range(n_features_to_select - n_inclusions):
        if i == 0:
            X_support, X_selected = _prepare_candidates(
                X, mask_exclude, indices_include
            )
            beams_selected_ids = [indices_include for _ in range(beam_width)]
            W_selected = orth(X_selected)
            selected_score = np.sum((W_selected.T @ V) ** 2)
            beams_scores = _gram_schmidt(
                X, X_support, X_selected, selected_score, V, tol
            )
            beams_selected_ids, top_k_scores = _select_top_k(
                beams_scores[None, :],
                beams_selected_ids,
                beam_width,
            )
            continue
        beams_scores = np.zeros((beam_width, n_features))
        for beam_idx in range(beam_width):
            X_support, X_selected = _prepare_candidates(
                X, mask_exclude, beams_selected_ids[beam_idx]
            )
            beams_scores[beam_idx] = _gram_schmidt(
                X, X_support, X_selected, top_k_scores[beam_idx], V, tol
            )
        beams_selected_ids, top_k_scores = _select_top_k(
            beams_scores,
            beams_selected_ids,
            beam_width,
        )
        if verbose:
            print(
                f"Beam Search: {i + 1 + n_inclusions}/{n_features_to_select}, "
                f"Best Beam: {np.argmax(top_k_scores)}, "
                f"Beam SSC: {top_k_scores.max():.5f}",
                end="\r",
            )
    if verbose:
        print()
    best_beam = np.argmax(top_k_scores)
    indices = beams_selected_ids[best_beam]
    return np.array(indices, dtype=np.int32, order="F")


def _prepare_candidates(X, mask_exclude, indices_selected):
    X_support = np.copy(~mask_exclude)
    X_support[indices_selected] = False
    X_selected = X[:, indices_selected]
    return X_support, X_selected


def _select_top_k(
    beams_scores,
    ids_selected,
    beam_width,
):
    # For explore wider: make each feature in each selection iteration can
    # only be selected once.
    # For explore deeper: allow different beams to select the same feature
    # at the different selection iteration.
    n_features = beams_scores.shape[1]
    beams_max = np.argmax(beams_scores, axis=0)
    scores_max = beams_scores[beams_max, np.arange(n_features)]
    n_valid = np.sum(beams_scores.any(axis=0))
    n_selected = len(ids_selected[0])
    if n_valid < beam_width:
        raise ValueError(
            "Beam Search: Not enough valid candidates to select "
            f"beam width number of features, when selecting feature {n_selected + 1}. "
            "Please decrease beam_width or n_features_to_select."
        )

    top_k_ids = np.argpartition(scores_max, -beam_width)[-beam_width:]
    new_ids_selected = [[] for _ in range(beam_width)]
    for k, beam_k in enumerate(beams_max[top_k_ids]):
        new_ids_selected[k] = ids_selected[beam_k] + [top_k_ids[k]]
    top_k_scores = scores_max[top_k_ids]
    return new_ids_selected, top_k_scores


def _gram_schmidt(X, X_support, X_selected, selected_score, V, tol, modified=True):
    X = np.copy(X)
    if modified:
        # Change to Modified Gram-Schmidt
        W_selected = orth(X_selected)
    scores = np.zeros(X.shape[1])
    for i, support in enumerate(X_support):
        if not support:
            continue
        xi = X[:, i : i + 1]
        for j in range(W_selected.shape[1]):
            proj = W_selected[:, j : j + 1].T @ xi
            xi -= proj * W_selected[:, j : j + 1]
        wi, X_support[i] = _safe_normalize(xi)
        if not X_support[i]:
            continue
        if np.any(np.abs(wi.T @ W_selected) > tol):
            X_support[i] = False
            continue
        scores[i] = np.sum((wi.T @ V) ** 2)
    scores += selected_score
    scores[~X_support] = 0
    return scores


def _safe_normalize(X):
    norm = np.linalg.norm(X, axis=0)
    non_zero_support = norm != 0
    norm[~non_zero_support] = 1
    return X / norm, non_zero_support
