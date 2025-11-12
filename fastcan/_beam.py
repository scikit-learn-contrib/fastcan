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
    Beam search with SSC.

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
            mask, X_selected = _prepare_candidates(X, mask_exclude, indices_include)
            if X_selected.shape[1] == 0:
                beams_scores = np.sum((X.T @ V) ** 2, axis=1)
                beams_scores[mask] = 0
            else:
                W_selected = orth(X_selected)
                selected_score = np.sum((W_selected.T @ V) ** 2)
                beams_scores = _mgs_ssc(X, V, W_selected, mask, selected_score, tol)
            beams_selected_ids = [indices_include for _ in range(beam_width)]
            beams_selected_ids, top_k_scores = _select_top_k(
                beams_scores[None, :],
                beams_selected_ids,
                beam_width,
            )
            continue
        beams_scores = np.zeros((beam_width, n_features))
        for beam_idx in range(beam_width):
            mask, X_selected = _prepare_candidates(
                X, mask_exclude, beams_selected_ids[beam_idx]
            )
            W_selected = orth(X_selected)
            beams_scores[beam_idx] = _mgs_ssc(
                X, V, W_selected, mask, top_k_scores[beam_idx], tol
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
    mask = np.copy(mask_exclude)
    mask[indices_selected] = True
    X_selected = X[:, indices_selected]
    return mask, X_selected


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


def _mgs_ssc(X, V, W_selected, mask, selected_score, tol):
    X = np.copy(X)
    proj = W_selected.T @ X
    X -= W_selected @ proj
    W, non_zero_mask = _safe_normalize(X)
    mask |= non_zero_mask
    linear_independent_mask = np.any(np.abs(W.T @ W_selected) > tol, axis=1)
    mask |= linear_independent_mask
    scores = np.sum((W.T @ V) ** 2, axis=1)
    scores += selected_score
    scores[mask] = 0
    return scores


def _safe_normalize(X):
    norm = np.linalg.norm(X, axis=0)
    non_zero_mask = norm == 0
    norm[non_zero_mask] = 1
    return X / norm, non_zero_mask
