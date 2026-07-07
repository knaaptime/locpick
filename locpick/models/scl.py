"""Spatially Correlated Logit (SCL) model for location choice estimation.

This module implements the SCL model proposed by Bhat and Guo (2004), which
captures spatial correlation between contiguous alternatives using a paired
Generalised Nested Logit (PGNL) structure.  The model is a member of the
GEV class and yields closed-form choice probabilities.

The SCL model uses a single dissimilarity parameter :math:`\\rho` that
governs the correlation between spatially adjacent alternatives.  When
:math:`\\rho = 1`, the model collapses to the Multinomial Logit (MNL).

References
----------
Bhat, C.R. and Guo, J.Y. (2004). A Mixed Spatially Correlated Logit Model:
    Formulation and Application to Residential Choice Modeling.
    *Transportation Research Part B*, 38(2), 147–168.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .._kernels.constants import NEG_INF
from ._spatial import (
    EdgeStructure as EdgeStructure,
)
from ._spatial import (
    _resolve_spatial_graph as _resolve_spatial_graph,
)
from ._spatial import (
    constrain_rho as constrain_rho,
)
from ._spatial import (
    naturalize_rho as naturalize_rho,
)

# ---------------------------------------------------------------------------
# SCL probability kernel (NumPy fallback)
# ---------------------------------------------------------------------------


def _scl_log_probs_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute SCL log-probabilities (NumPy backend).

    Implements the choice probability from Bhat & Guo (2004, Eq. 9):

    .. math::

        P_i = \\sum_{j \\neq i} P_{i|ij} \\times P_{ij}

    where the conditional probability within paired nest :math:`(i,j)` is:

    .. math::

        P_{i|ij} = \\frac{(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho}}
        {(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
         (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}}

    and the nest probability is:

    .. math::

        P_{ij} = \\frac{\\left[(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
        (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}\\right]^{\\rho}}
        {\\sum_{k<l}\\left[(\\alpha_{k,kl}\\, e^{V_k})^{1/\\rho} +
        (\\alpha_{l,kl}\\, e^{V_l})^{1/\\rho}\\right]^{\\rho}}

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    rho : float
        Dissimilarity parameter in (0, 1].  When ``rho = 1``, the model
        reduces to MNL.
    design_matrix : np.ndarray, shape (n_obs * n_alts, k)
        Design matrix of explanatory variables.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix :math:`\\alpha_{i,j}`.
    edge_list : list of (int, int)
        Paired-nest edges defining the spatial correlation structure.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Binary availability matrix.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    """

    # Step 1: systematic utility
    V = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Step 2: sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        V = V + np.log(np.maximum(sr, 1e-30))

    # Step 3: availability masking
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    V = np.where(avail > 0, V, NEG_INF)

    # Step 4: compute SCL probabilities
    inv_rho = 1.0 / rho
    exp_V = np.exp(np.clip(V, -500, 500))  # (n_obs, n_alts)

    # Pre-compute (α_{i,ij} * exp(V_i))^{1/ρ} for all (i, j) pairs
    # allocation: (n_alts, n_alts), exp_V: (n_obs, n_alts)
    # For each alt i: alloc_exp_V[obs, i, j] = allocation[i, j] * exp_V[obs, i]
    # Then raise to 1/ρ
    alloc_exp_V = allocation[None, :, :] * exp_V[:, :, None]  # (n_obs, n_alts, n_alts)
    alloc_exp_V = np.clip(alloc_exp_V, 1e-30, 1e30)
    alloc_exp_V_inv_rho = np.power(alloc_exp_V, inv_rho)  # (n_obs, n_alts, n_alts)

    n_edges = len(edge_list)
    if n_edges == 0:
        # No spatial correlation → MNL
        log_sum_exp_V = logsumexp(V, axis=1)
        log_probs = V - log_sum_exp_V[:, None]
        return log_probs

    # Identify connected vs isolated alternatives
    connected = set()
    for i, j in edge_list:
        connected.add(i)
        connected.add(j)
    isolated = sorted(set(range(n_alts)) - connected)

    # Compute nest values for each edge
    # nest_vals: (n_obs, n_edges)
    nest_vals = np.zeros((n_obs, n_edges), dtype=np.float64)

    for idx, (i, j) in enumerate(edge_list):
        term_i = alloc_exp_V_inv_rho[:, i, j]  # (α_{i,ij} * exp(V_i))^{1/ρ}
        term_j = alloc_exp_V_inv_rho[:, j, i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
        nest_vals[:, idx] = np.power(term_i + term_j, rho)

    # Denominator: sum of nest values + exp(V_i) for isolated alternatives
    # In the SCL framework, isolated alts are "degenerate nests" with
    # a single member, so their nest value is just exp(V_i).
    if isolated:
        iso_exp_V = exp_V[:, isolated]  # (n_obs, n_isolated)
        denom_components = np.column_stack([nest_vals, iso_exp_V])
    else:
        denom_components = nest_vals

    # Use log of sum (not logsumexp) since nest_vals are already on natural scale
    log_denom = np.log(np.maximum(denom_components.sum(axis=1), 1e-300))  # (n_obs,)

    # Compute log-probabilities for each alternative
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    # Build a mapping from alt index to its edges
    alt_to_edges: dict[int, list[tuple[int, bool]]] = {}
    for idx, (i, j) in enumerate(edge_list):
        alt_to_edges.setdefault(i, []).append((idx, True))  # i is the first node
        alt_to_edges.setdefault(j, []).append((idx, False))  # j is the second node

    # Connected alternatives: P_i = Σ_{j ∈ N(i)} P_{i|ij} * P_{ij}
    for alt_i in connected:
        edge_contributions = []
        for edge_idx, is_first in alt_to_edges[alt_i]:
            i, j = edge_list[edge_idx]
            if is_first:
                # alt_i is node i in edge (i, j)
                # P_{i|ij} = (α_{i,ij} * exp(V_i))^{1/ρ} / [(α_{i,ij} * exp(V_i))^{1/ρ} + (α_{j,ij} * exp(V_j))^{1/ρ}]
                my_term = alloc_exp_V_inv_rho[:, alt_i, j]  # (α_{i,ij} * exp(V_i))^{1/ρ}
                other_term = alloc_exp_V_inv_rho[:, j, alt_i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
            else:
                # alt_i is node j in edge (i, j)
                # P_{j|ij} = (α_{j,ij} * exp(V_j))^{1/ρ} / [(α_{i,ij} * exp(V_i))^{1/ρ} + (α_{j,ij} * exp(V_j))^{1/ρ}]
                my_term = alloc_exp_V_inv_rho[:, alt_i, i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
                other_term = alloc_exp_V_inv_rho[:, i, alt_i]  # (α_{i,ij} * exp(V_i))^{1/ρ}

            # log P_{i|ij} = log(my_term) - log(my_term + other_term)
            log_cond = np.log(np.maximum(my_term, 1e-300)) - np.log(
                np.maximum(my_term + other_term, 1e-300)
            )
            # log P_{ij} = log(nest_val) - log_denom
            log_nest = np.log(np.maximum(nest_vals[:, edge_idx], 1e-300)) - log_denom

            edge_contributions.append(log_cond + log_nest)

        # log P_i = logsumexp over all edge contributions
        if len(edge_contributions) == 1:
            log_probs[:, alt_i] = edge_contributions[0]
        else:
            log_probs[:, alt_i] = logsumexp(np.column_stack(edge_contributions), axis=1)

    # Isolated alternatives: P_i = exp(V_i) / denom_total
    # These are "degenerate nests" with a single member
    for alt_i in isolated:
        log_probs[:, alt_i] = V[:, alt_i] - log_denom

    # Zero out unavailable alternatives
    log_probs = np.where(avail > 0, log_probs, NEG_INF)

    return log_probs


def _scl_ll_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute SCL log-likelihood (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    rho : float
        Dissimilarity parameter in (0, 1].
    design_matrix, allocation, edge_list, n_obs, n_alts, available, inclusion_probs
        See :func:`_scl_log_probs_numpy`.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    weights : np.ndarray or None, shape (n_obs,)
        Observation-level weights.

    Returns
    -------
    float
        Log-likelihood value.
    """
    log_probs = _scl_log_probs_numpy(
        beta,
        rho,
        design_matrix,
        allocation,
        edge_list,
        n_obs,
        n_alts,
        available=available,
        inclusion_probs=inclusion_probs,
    )

    # Chosen log-probabilities
    chosen_2d = np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts)
    chosen_log_probs = (log_probs * chosen_2d).sum(axis=1)

    # Avoid log(0)
    chosen_log_probs = np.maximum(chosen_log_probs, -1e30)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        chosen_log_probs = chosen_log_probs * w

    return float(chosen_log_probs.sum())
